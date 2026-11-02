import os, time, json, sys, yaml, logging, asyncio, subprocess, re
import inspect
from datetime import datetime
from pathlib import Path
project_root = str(Path(__file__).parent.parent / "Assets" )
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from OrmsgpackConsts import _OrmsgpackConsts
# ormsgpack在windows环境下会导致内存崩溃，使用猴子补丁注入到系统模块缓存
sys.modules["ormsgpack"] = _OrmsgpackConsts()
from typing import Annotated, List
from typing_extensions import TypedDict
from pydantic import SecretStr
from functools import partial, wraps
from langchain_core.messages import (HumanMessage, SystemMessage,ToolMessage,
                                     AIMessage, BaseMessage, RemoveMessage)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# ================== 工具 ==================
from DS_Utilities import (load_tools_from_config, TokenCounter, summarize_tool_message,
                          normalize_response_with_DSML, update_node_progress)
from LLMGenerate import Independent_LLM_Tools
from FileTools import  FileToolConfig
logger = logging.getLogger(__name__)
_mcp_client: MultiServerMCPClient | None = None
mcp_session_ctxs: list = []

# ================== 人工终止标记（进程内全局，按 task_uuid 隔离）==================
# 为什么不用 LangGraph 的 update_state：
# 图正在 astream 运行时，Pregel 循环已在启动阶段加载 checkpoint，
# 之后每个 superstep 都用它内存中的 channel 写新 checkpoint，
# 外部 update_state 写入的 force_termination 会被运行中的循环直接覆盖。
# 因此改用进程内 flag，由节点入口主动读取，确定性 100%。
STOP_FLAGS: set[str] = set()


def set_stop_flag(task_uuid: str):
    if task_uuid:
        STOP_FLAGS.add(task_uuid)


def clear_stop_flag(task_uuid: str):
    STOP_FLAGS.discard(task_uuid)


def is_stopped(task_uuid: str) -> bool:
    return bool(task_uuid) and task_uuid in STOP_FLAGS


# ================== CLI 进程注册表 / 状态回写 ==================
# task_uuid -> {"proc": Popen, "command": str, "ds_dir": str, "started_at": str, "killed": bool, ...}
CLI_PROC_REGISTRY: dict = {}
# 由 AgentFrame 注入的状态回写钩子，避免 TA_Nodes 反向依赖 AgentFrame（循环导入）
_cli_status_hook = None


def set_cli_status_hook(fn):
    """注入钩子：fn(ds_dir: str, info: dict) -> None，用于把 CLI 运行状态写进 status.json"""
    global _cli_status_hook
    _cli_status_hook = fn


def _emit_cli_status(ds_dir: str, info: dict):
    if not _cli_status_hook or not ds_dir:
        return
    try:
        _cli_status_hook(ds_dir, info)
    except Exception as e:
        print(f"[CLI状态回写失败] {e}")
# 自定义合并规则：支持追加工具记录，也支持重置清空
def tool_run_list_reducer(existing: list, new_value):
    # 传入 "reset" 标记时，清空全部历史记录
    if new_value == "reset":
        return []
    # 传入列表时，批量追加到历史末尾
    if isinstance(new_value, list):
        combined =  existing + new_value
    # 传入单条记录时，追加单个元素
    return combined[-100:]
# project_history 专用 reducer：追加式更新，自动换行
def append_list_reducer(existing: list | None, new_value) -> list:
    # 整体替换
    if isinstance(new_value, tuple) and len(new_value) == 2 and new_value[0] == 'REPLACE':
        return list(new_value[1])
    # 旧值为空则初始化为空列表
    if existing is None:
        existing = []
    # 新值为空，不做修改
    if new_value is None:
        return existing
    # 批量追加：传入列表时用 extend
    if isinstance(new_value, list):
        existing.extend(new_value)
    # 单条追加：传入单个元素时用 append
    else:
        existing.append(new_value)
    return existing
# ================== 状态 ==================
class AgentState(TypedDict):
    task_uuid: str
    status: str
    root_workspace: str
    ds_agent_dir: str
    messages: Annotated[list, add_messages]
    project_history: Annotated[List,append_list_reducer]
    project_meta: str
    intent: str
    task_manage: str
    tool_run_list: Annotated[list, tool_run_list_reducer]
    outer_loop_count: int  # 外层主循环累计次数
    force_termination: bool
    token: int
# ==================通用节点==================
def flexible_node(state: AgentState, logger=None, ui_dialogger=None, node_name = '', node_config: dict = None):
    record = f'节点{node_name} :'
    logger.info(f">>>> 执行节点：【{node_name}】")
    logger.info(f"in：{state['messages'][-1]}")
    system_prompt = all_prompt.get(node_config["prompt_key"], '')
    # logger.info(f"system_prompt: {system_prompt}")
    interrupt_flag = node_config.get('interrupt', False)
    if node_config.get("use_tools", False):
        logger.info(f"使用带工具LLM")
        llm_with_tools = llm_dict_with_tools.get(node_name, llm_dict_with_tools['general']) # 限定暴露的工具范围
        response = llm_with_tools.invoke([SystemMessage(content=f"{system_prompt}\n{state['task_manage']}")] + state["messages"])
    else:
        response = llm.invoke([SystemMessage(content=f"{system_prompt}\n{state['task_manage']}")] + state["messages"],  tool_choice="none" )
    logger.info(f"out：{response}")
    if "<｜｜DSML｜｜tool_calls>" in response.content:
        response = normalize_response_with_DSML(response, logger,node_config.get("use_tools", False))
    if response.content and not interrupt_flag:
        ui_dialogger.append_log(f"**[{node_name}]Agent：**\n{response.content}")
    record += response.content
    # 处理invalid_tool_calls
    if  hasattr(response, "invalid_tool_calls") or  response.invalid_tool_calls:
        if response.invalid_tool_calls != []:
            logger.warning("触发强制清洗invalid_tool_calls")
            # 拼接错误文本
            error_lines = []
            for invalid_call in response.invalid_tool_calls:
                tool_name = invalid_call.get("name", "unknown_tool")
                error_msg = invalid_call.get("error", "未知错误")
                error_lines.append(f"[工具调用失败] {tool_name}: {error_msg}")
            error_text = "\n".join(error_lines)
            # 合并到原content（原content通常为空，兼容非空情况）
            original_content = response.content if response.content else ""
            new_content = f"{original_content}\n{error_text}".strip()
            new_response = AIMessage(
                    content=new_content,
                    tool_calls=response.tool_calls,  # 正常工具调用保留
                    invalid_tool_calls=[],  # 清空无效调用
                    response_metadata=response.response_metadata,
                    id=response.id)
            response = new_response
    # 有工具调用先走工具
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info("模型返回了 tool_calls，需要先执行工具")
        record += "  调用工具:"
        for toolcall in response.tool_calls:
            record += f" {toolcall['name']} 参数：{str(toolcall['args'])} "
        return {"messages": [response],
                "project_history":record}
    # 更新当前tokens总数
    usage = getattr(response, 'usage_metadata', None) or response.response_metadata.get('token_usage',{}) or response.response_metadata
    total_tokens = usage.get('total_tokens', None)
    if total_tokens:
        state["token"] = total_tokens
        logger.info(f"total_tokens：{total_tokens}")
    # 是否打断并要求输入
    if node_config.get("interrupt", False):
        # 处理中断逻辑（如 ask_question）
        question_text = response.content
        user_answer = interrupt(question_text)
        record += f"\n用户指令{user_answer}"
        return {"messages": [response, HumanMessage(content=user_answer)],
                "project_history":record}
    else:
        return {"messages": [response],
                "project_history":record}
# ================== 路由节点 ==================
def _as_text(v) -> str:
    """把 LLM 可能返回的任意结构（str / dict / list / None）安全地转成可写入的字符串"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, indent=2)
    except Exception:
        return str(v)


def router_node(state: AgentState, logger=None, ui_dialogger=None):
    logger.info(">>>> 执行节点：【router_node】")
    #保存history：
    proj_history = "# Project History\n\n" + ("="*30 + "Step" + "="*30 + "\n").join(state["project_history"][-100:])
    record = f'节点router :'
    project_meta = ''
    current_loop = state.get("outer_loop_count", 0) + 1
    MAX_OUTER_LOOPS = 20

    # 循环超限强制标记结束意图
    if current_loop > MAX_OUTER_LOOPS:
        return {
            "intent": "finish",
            "outer_loop_count": current_loop,
            "tool_run_list": 'reset'
        }
    # 主动结束意图
    if state["force_termination"]:
        return {
            "intent": "finish",
            "messages": [HumanMessage(content="**用户主动强制终止流程**")],
            "force_termination": False
        }
    system_prompt = all_prompt.get('router_node', '')
    # logger.info("system prompt: " + system_prompt)
    rectifed_resp = independent_llm.llm_json_rectify(f"{system_prompt}\n{state['task_manage']}", proj_history)
    logger.info(f"out：{rectifed_resp}")
    try:
        data = json.loads(rectifed_resp)
        # LLM 可能把字段返回成 dict / list，统一安全转成字符串，避免 f.write / AIMessage 崩溃
        intent = _as_text(data.get("intent", "ask_question")) or "ask_question"
        reason = _as_text(data.get("reason", ""))
        project_meta = _as_text(data.get("project_meta", ""))
        ui_dialogger.append_log(f"**[router]** intent=`{intent}`\n{reason}")
        logger.info(f"intent: {intent},reason: {reason}")
        record += f" intent: {intent},reason: {reason}"
    except Exception as e:
        intent = "ask_question"
        reason = '没能找到合适的方向'
        logger.info(f"{e}  intent: {intent}")
        record += f" intent: {intent},reason: {e}"
    if project_meta:
        try:
            meta_dir = state.get('root_workspace') or '.'
            os.makedirs(meta_dir, exist_ok=True)
            with open(os.path.join(meta_dir, "project_meta.md"), "w", encoding="utf-8") as f:
                f.write(project_meta)
        except Exception as e:
            logger.warning(f"写入 project_meta.md 失败：{e}")
        return {
            "messages": [AIMessage(content=reason)],
            "intent": intent,
            "outer_loop_count": current_loop,
            "tool_run_list": 'reset',
            "project_history": record,
            "project_meta": project_meta
        }
    else:
        return {
            "messages": [AIMessage(content=reason)],
            "intent": intent,
            "outer_loop_count": current_loop,
            "tool_run_list": 'reset',
            "project_history": record
        }
# ================== 条件路由函数 ==================
def route_by_intent(state: AgentState) -> str:
    intent = state["intent"]
    valid_intents = ALL_NODES_CONFIG["router_valid_intents"]
    if intent not in valid_intents:
        return "task_process"
    return intent
# =================任务执行节点,全工具兜底，LLM惯性===================
def task_process_node(state: AgentState,logger=None,ui_dialogger=None):
    logger.info(">>>> 执行节点：【task_process_node】")
    record = f'节点task_process :'
    logger.info(f"【task_process_node】in：{state['messages'][-1]}")
    # 检查是否出现工具调用死循环
    history_run_list = state.get("tool_run_list", [])  # 读取历史工具调用记录
    # 规则：历史记录中最后4次调用的工具组合，与本次完全一致
    trigger_warning = False
    if len(history_run_list) >= 5:
        last_5_runs = history_run_list[-5:]
        # 逐次比对工具列表（严格匹配顺序与工具名）
        trigger_warning = all(run == last_5_runs[-1] for run in last_5_runs)
    if trigger_warning:
        logger.warning("检测到工具调用死循环，触发强制打断工具调用")
        record += "\n检测到工具调用死循环，触发强制打断工具调用，并记录问题"
        response = AIMessage(content="检测到工具调用死循环触，发强制打断工具调用。应该复盘原因，在project_problems.md记录问题")
        return {"messages": [response],
                "tool_run_list": 'reset',
                "project_history": record}
    system_prompt = all_prompt.get('task_process_node','')
    llm_with_tools = llm_dict_with_tools['all'] #全工具接入
    response = llm_with_tools.invoke(
        [SystemMessage(f"{system_prompt}\n{state['task_manage']}")] + state["messages"]
    )
    ui_dialogger.append_log(f"**[task process]Agent：**\n{response.content}")
    usage = getattr(response, 'usage_metadata', None) or response.response_metadata.get('token_usage',                                                                              {}) or response.response_metadata
    total_tokens = usage.get('total_tokens', None)
    if total_tokens:
        state["token"] = total_tokens
        logger.info(f"total_tokens：{total_tokens}")
    logger.info(f"【task_process_node】out：{response}")
    if "<｜｜DSML｜｜tool_calls>" in response.content:
        response = normalize_response_with_DSML(response, logger)
    record += response.content
    # 有工具调用先走工具
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info("模型返回了 tool_calls，需要先执行工具")
        record += "  调用工具:"
        for toolcall in response.tool_calls:
            record += f" {toolcall['name']} 参数：{str(toolcall['args'])} "
        return {"messages": [response],
                "project_history": record}
    # 处理invalid_tool_calls
    if hasattr(response, "invalid_tool_calls") or response.invalid_tool_calls:
        if response.invalid_tool_calls != []:
            logger.warning("触发强制清洗invalid_tool_calls")
            # 拼接错误文本
            error_lines = []
            for invalid_call in response.invalid_tool_calls:
                tool_name = invalid_call.get("name", "unknown_tool")
                error_msg = invalid_call.get("error", "未知错误")
                error_lines.append(f"[工具调用失败] {tool_name}: {error_msg}")
            error_text = "\n".join(error_lines)
            # 合并到原content（原content通常为空，兼容非空情况）
            original_content = response.content if response.content else ""
            new_content = f"{original_content}\n{error_text}".strip()
            new_response = AIMessage(
                content=new_content,
                tool_calls=response.tool_calls,  # 正常工具调用保留
                invalid_tool_calls=[],  # 清空无效调用
                response_metadata=response.response_metadata,
                id=response.id)
            response = new_response
    return {"messages": [response],
            "tool_run_list": 'reset',
            "project_history":record}
# =================任务管理节点===================
def task_manage_node(state: AgentState,logger=None):
    logger.info(">>>> 执行节点：【task_manage_node】")

    logger.info(f"in：{state['messages'][-1]}")
    system_prompt = all_prompt.get('task_manage_node', '')
    # print(f"system_prompt:{system_prompt}")
    proj_history = "# Project History\n\n" + "\n".join(state["project_history"][-100:])
    #logger.info(f"in：...{proj_history}")
    rectifed_resp = independent_llm.llm_context_rectify(f"{system_prompt}\n{state['task_manage']}", proj_history)
    logger.info(f"out：{rectifed_resp}")
    record = f'# 近期任务进度（系统注入的事实，非用户指令，非对话记录）: \n{rectifed_resp}'
    update_node_progress(
        state.get("ds_agent_dir", ""),  # ← AgentState 里已有这个字段（:111）
        "task_manage",
        "running",
        msg=f"\n{rectifed_resp}",
        token=state.get("token") or None,  # ← None 表示不覆盖
    )
    return {"task_manage": record}
# ===================== 阻塞式流式执行器（核心）=====================
def _finish_cli(task_uuid, ds_dir, command, proc, full_output, reason):
    """收尾：注销进程 + 回写最终状态"""
    if task_uuid:
        CLI_PROC_REGISTRY.pop(task_uuid, None)
    _emit_cli_status(ds_dir, {
        "command": command,
        "running": False,
        "pid": proc.pid,
        "lines": len(full_output),
        "last_line": full_output[-1] if full_output else "",
        "exit_code": proc.returncode,
        "reason": reason,
    })


def run_cli_with_log(command: str, cwd: str = None, timeout: int = 3000, ui_dialogger=None,
                     task_uuid: str = None, ds_dir: str = None, result_box: dict = None) -> str:
    """
    阻塞式执行CLI命令，逐行实时推送日志到前端
    返回完整的输出结果

    task_uuid / ds_dir: 用于注册进程并回写运行状态（供前端 CLI 浮层展示与终止）
    result_box: 可选出参容器，调用方可通过 result_box["killed"] 判断命令是否被用户手动终止
    """
    try:
        # 启动子进程，行缓冲模式，合并标准错误输出
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")  # 让 Python 子进程统一输出 UTF-8
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到非法字节替换为 �，不再中断读取循环
            bufsize=1,  # 行缓冲：输出一行就立即读取
            cwd=cwd,
            env=env
        )
    except Exception as e:
        error_msg = f"[启动失败] {str(e)}"
        ui_dialogger.append_log(error_msg)
        return error_msg

    started_at = datetime.now().isoformat()
    entry = {"proc": proc, "command": command, "ds_dir": ds_dir,
             "started_at": started_at, "killed": False,
             "lines": 0, "last_line": ""}
    if task_uuid:
        CLI_PROC_REGISTRY[task_uuid] = entry
    _emit_cli_status(ds_dir, {"command": command, "running": True, "pid": proc.pid,
                              "started_at": started_at, "lines": 0, "last_line": ""})

    full_output = []
    # 阻塞逐行读取输出，读到一行就推给前端
    _last_emit = [0.0]
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n")
            full_output.append(line)
            ui_dialogger.append_log(line)
            entry["lines"] = len(full_output)
            entry["last_line"] = line
            # 节流回写，避免每行都写一次 status.json
            now = time.time()
            if now - _last_emit[0] > 0.6:
                _last_emit[0] = now
                _emit_cli_status(ds_dir, {"command": command, "running": True, "pid": proc.pid,
                                          "started_at": started_at,
                                          "lines": len(full_output), "last_line": line})
    except Exception as e:
        error_msg = f"[读取异常] {str(e)}"
        ui_dialogger.append_log(error_msg)
        proc.kill()
        _finish_cli(task_uuid, ds_dir, command, proc, full_output, "读取异常")
        if result_box is not None:
            result_box["killed"] = True
        return error_msg
    # 等待进程结束，处理超时
    # 读取循环结束后再确认一次是否被用户 kill（kill 发生在读循环期间）
    killed = bool(entry.get("killed"))
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        timeout_msg = f"[超时] 命令执行超过 {timeout} 秒，已终止"
        ui_dialogger.append_log(timeout_msg)
        full_output.append(timeout_msg)
        _finish_cli(task_uuid, ds_dir, command, proc, full_output, "超时终止")
        if result_box is not None:
            result_box["killed"] = True
            result_box["exit_code"] = proc.returncode
        return "\n".join(full_output)
    # 非0退出码标记
    if proc.returncode != 0:
        exit_msg = f"[退出码 {proc.returncode}] 命令执行异常"
        ui_dialogger.append_log(exit_msg)
        full_output.append(exit_msg)
    if killed:
        full_output.append("[用户终止] 命令已被用户手动终止")
    _finish_cli(task_uuid, ds_dir, command, proc, full_output,
                "用户终止" if killed else ("异常" if proc.returncode else "完成"))
    if result_box is not None:
        result_box["killed"] = killed
        result_box["exit_code"] = proc.returncode
    return "\n".join(full_output)
# ===================== CLI专用执行节点（阻塞式）=====================
def cli_executor_node(state: AgentState, logger=None, ui_dialogger=None, node_name = '', node_config: dict = None):
    """
    阻塞式CLI专用节点
    内部完成：命令生成 → 安全校验 → 阻塞执行+实时日志推送 → 结果返回
    """
    logger.info(f">>>> 执行节点：【{node_name}】")
    record = f''
    cwd = state['root_workspace'] #工作路径
    # 1. 模型生成命令
    system_prompt = """你是CLI参数生成器，根据最近的对话提取任务按照规范输出任务要素。
要求：
1. 只输出JSON格式的Dict型的命令要素，包括“parameters”和“reason”
2. 一次输出的JSON中只包含一组参数和一个理由，参数值只能是String类型，如果没有合适的参数可以给出空字典
3. 除了JSON，不要任何解释、说明、多余文字
4. 输出示例：
    {
        "parameters": {
            "demo_param1" : "String1",
            "demo_param2" : "String2",    
            },
        "reason": "简短解释给出的参数的理由或无法给出参数的理由"
    }
参数信息：
"""
    system_prompt += f"\n当前工作路径：{cwd}\n"
    system_prompt += all_prompt.get(node_config["prompt_key"], '')
    proj_history = "# Project History Steps:\n\n".join(state["project_history"][-100:])
    logger.info('system_prompt:  ' + system_prompt)
    rectifed_resp = independent_llm.llm_json_rectify(f"{system_prompt}\n{state['task_manage']}", proj_history)
    logger.info(f"out：{rectifed_resp}")
    command = ''
    try:
        data = json.loads(rectifed_resp)
        command = node_config['cli_template']
        cli_parameters = data.get('parameters',{})
        cli_reason = data.get('reason', '')
        if cli_parameters == {} and cli_reason:
            record += cli_reason
        elif cli_parameters == {} and cli_reason == '':
            record += '节点没有输出可执行的参数，也没给出这么做的理由'
        else:
            for param in cli_parameters:
                command = command.replace('%'+param +'%', cli_parameters[param])
                # 兜底：仍有未替换占位符就拒绝执行，避免拿模板硬跑
            leftover = re.findall(r'%[A-Za-z_][A-Za-z0-9_]*%', command)
            if leftover:
                record += f"命令仍存在未替换的占位符 {leftover}，已取消执行"
                logger.info(f"未替换占位符：{leftover}")
                command = ''
        logger.info(f"cli command：{command}")
    except Exception as e:
        logger.info(f"Error: {e}")
        record += f"Error: {e}"
    # 3. 阻塞执行 + 实时推送日志
    task_uuid = state.get("task_uuid", "")
    ds_dir = state.get("ds_agent_dir", "")
    if command:
        ui_dialogger.append_log(f"**[CLI]Agent：**\n执行 {command}")
        box: dict = {}
        full_output = run_cli_with_log(command, cwd=cwd, timeout=3000, ui_dialogger=ui_dialogger,
                                       task_uuid=task_uuid, ds_dir=ds_dir, result_box=box)
        record += '\n执行命令 ' + command + '\n执行Log:\n' + full_output
        # 用户手动终止 → 复用人工介入流程（节点自然结束，不抛异常）
        if box.get("killed"):
            record += "\n[用户终止] 该 CLI 命令已被用户手动终止，转入人工介入。"
            ui_dialogger.append_log("\n**[CLI] 已被用户终止，转入人工介入**\n")
            logger.info("CLI 被用户终止，跳转 human_intervene")
            return Command(goto="human_intervene",
                           update={"messages": [AIMessage(content=record)],
                                   "project_history": record})
    else:
        ui_dialogger.append_log(f"**[CLI]Agent：**\n未生成可执行的命令")
        record += "\n没有可执行的命令"
    logger.info(f"result：{record}")
    # 4. 返回完整结果更新全局状态
    return {
        "messages": [AIMessage(content=record)],
        "project_history": record
    }

# ================= 任务收尾节点（新增，优雅结束） =================
def finish_summary_node(state: AgentState, logger=None, ui_dialogger=None):
    """任务完成汇总节点，统一输出最终结论，优雅退出前汇总信息"""
    logger.info(">>>> 执行节点：【finish_summary_node】")
    record = f'节点finish_summary :'
    logger.info(f"in：{state['messages'][-1]}")
    # prompt = SystemMessage("请基于全部对话，整理最终完整结论。然后等待用户发起新任务。")
    # response = llm.invoke([prompt] + state["messages"], tool_choice="none" )
    system_prompt = "请基于项目历史对话，整理现阶段的最终完整结论。然后等待用户发起新任务。"
    proj_history = "# Project History\n\n" + ("=" * 30 + "Step" + "=" * 30 + "\n").join(state["project_history"][-100:])
    response = independent_llm.llm_context_rectify(f"{system_prompt}\n{state['task_manage']}", proj_history)
    ui_dialogger.append_log(f"**[task summary]Agent：**\n{response}")
    logger.info(f"out：{response}")
    # if "<｜｜DSML｜｜tool_calls>" in response.content:
    #     response = normalize_response_with_DSML(response, logger, False)
    record += response
    return {"messages": [response],
            "project_history":record}
# =================等待新任务节点（替代END） =================
def await_new_task_node(state: AgentState, logger=None):
    """任务完成后进入等待，接收新一轮用户指令"""
    logger.info(">>>> 执行节点：【await_new_task_node】")
    start_time = time.time()
    logger.info(f"in：{state['messages'][-1]}")
    interrupt_data = interrupt( "当前任务已完成，请输入新任务指令")
    # 记录 interrupt 结束时间
    end_time = time.time()
    logger.info(f"interrupt 等待时间：{end_time - start_time:.2f} 秒")
    # 用户下发新指令，追加消息
    record = f'用户指令 : {interrupt_data}'
    return {
        "messages": [HumanMessage(content=interrupt_data)],
        # 重置循环计数器，开启新一轮任务
        "outer_loop_count": 0,
        "tool_run_list": 'reset',
        "project_history": record
    }
# =================上下文压缩节点 =================
def compress_node(state: AgentState,logger=None,ui_dialogger=None):
    logger.info(">>>> 执行节点：【compress_node】")
    all_messages: List[BaseMessage] = state["messages"]
    project_history = state["project_history"]
    global_system_msg = all_messages[0]
    # 1. 全量统计token，判断是否触发压缩
    current_total_token = token_counter.count_all_tokens(all_messages)
    logger.info(f"token总数：{current_total_token}")
    if current_total_token < TRIGGER_TOKEN:
        logger.info("前文压缩没触发")
        return {"token": current_total_token}
    ui_dialogger.append_log("**Agent：**让我整理一下记忆...")
    logger.info("触发前文压缩")
    # 3. 业务消息整体切分
    split_x = int(len(project_history) * 0.6)
    reserved_history = project_history[split_x:]
    compressed_history = project_history[:split_x]
    # 4. 头部片段过滤，仅Human、AI,ToolMessage参与总结
    summary_source = "# Project History\n" + "\n".join(compressed_history)
    reserved_msg = HumanMessage(content="# Project History\n" + "\n## Step:\n".join(reserved_history))
    # 先清除所有消息
    all_ids_except_sys = [m.id for m in all_messages[1:] if getattr(m, "id", None)]
    delete_all_except_sys = [RemoveMessage(id=mid) for mid in all_ids_except_sys]
    summary_text = independent_llm.llm_messages_compress(f"{state['task_manage']}\n{summary_source}")
    summary_msg = HumanMessage(content="# Project Meta Info\n" +str(summary_text))
    # 6. 清除缓存
    for mid in all_ids_except_sys:
        token_counter.cache.pop(mid, None)
    # 6. 重组上下文：工程信息 + 摘要 + 后半段转化后的业务历史
    new_messages = [global_system_msg, summary_msg, reserved_msg]
    # logger.info("#"*50 +f"\n new_messages：{new_messages}")
    return {"messages": delete_all_except_sys + new_messages,
            "project_history": ('REPLACE',[summary_text] + reserved_history),
            "token": current_total_token}

# ====================工具接点 =========================
def tools_condition_with_log(state) -> str:
    """
    带日志输出的 tools_condition 路由函数。
    逻辑与 langgraph.prebuilt.tools_condition 完全一致，
    额外在每个关键节点输出 logger 信息，方便调试。
    """
    logger.info(">>>> 执行节点： 【tools_condition_with_log】")
    last_message = state["messages"][-1]
    logger.info(f"最后一条消息类型: {type(last_message).__name__}")

    if hasattr(last_message, "content"):
        content_preview = str(last_message.content)[:200]
        logger.info(f"最后一条消息内容预览: {content_preview}")
    # 内联 has_pending_tool_call 逻辑
    pending = hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)
    logger.info(f"pending_tool_call={pending}")

    if pending:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info(f"即将调用工具: {tool_names}")
        logger.info(f"工具调用参数: {[tc.get('args', {}) for tc in last_message.tool_calls]}")
        return "tools"
    return END


async def tool_node_with_counter(state: AgentState, logger=None, ui_dialogger=None):
    logger.info(">>>> 执行节点：【tool_node_with_counter】")
    logger.info(f"【tool_node_with_counter】in：{state['messages'][-1]}")
    # 1. 执行基础工具节点，得到工具执行结果
    result =await base_tool_node.ainvoke(state)
    logger.info(f"【tool_node_with_counter】out：{result}")
    # 2. 提取本次tool_calls
    tool_info = []
    if hasattr(state['messages'][-1], "tool_calls"):
        if state['messages'][-1].tool_calls:
            for tool_call in state['messages'][-1].tool_calls:
                tool_info.append([tool_call['name'],tool_call['args']])
                ui_dialogger.append_log(f"\n[tool]Agent：\n调用工具 {tool_call['name']}")
                for _m in result.get("messages", []):
                    if isinstance(_m, ToolMessage):
                        ui_dialogger.append_log(summarize_tool_message(_m)[:200].replace('--','').replace('#','') + '...\n', mode='temp')
    # 3. 读取历史工具调用记录
    history_run_list = state.get("tool_run_list", 'reset')
    # 更新工具调用历史：追加本次记录，返回完整新列表
    new_run_list = history_run_list + tool_info
    # 合并结果返回
    # 把 ToolMessage 的客观事实写进 project_history
    record_lines = ["工具执行 :"]
    # ainvoke 返回的 result 里 messages 已追加 ToolMessage
    for m in result.get("messages", []):
        if not isinstance(m, ToolMessage):
            continue
        record_lines.append(summarize_tool_message(m))
    record = "\n".join(record_lines) + "\n"
    return {
        **result,
        "tool_run_list": new_run_list,
        "project_history": record
    }

async def group_mcp_tools_by_scope(mcp_config_paths, mcp_tools_config_paths,
                                   local_names: set, logger=None) -> dict:
    """按该 MCP 的 tools_config.yaml 的 mcp_node 条目分组（配置里写的是服务器原始名）
    返回 {scope: [BaseTool, ...]}，工具对象统一为最终名字"""
    global _mcp_client, mcp_session_ctxs
    if not mcp_config_paths:
        return {}
    # ① 连接所有 MCP server，收集全部工具
    all_mcp_tools = []
    for cfg_path in mcp_config_paths:
        if not Path(cfg_path).exists():  # 防御：该 MCP 未提供分组清单
            print(f'未找到文件{cfg_path}')
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)                     # 必须是合法 JSON
        for server_name, server_cfg in cfg.items():
            _mcp_client = MultiServerMCPClient({server_name: server_cfg})
            session_ctx = _mcp_client.session(server_name)  # 长会话
            session = await session_ctx.__aenter__()  # 启动一次子进程
            mcp_session_ctxs.append(session_ctx)
            all_mcp_tools.extend(
                await load_mcp_tools(session))# 工具绑定该会话
    # ② 重名处理：与本地工具冲突的加 cfd_ 前缀，保留「原始名 -> 最终工具」映射
    name_map = {}
    for t in all_mcp_tools:
        if t.name in local_names:
            new_name = f"cfd_{t.name}"
            logger.warning(f"MCP 工具 {t.name} 与本地工具重名，重命名为 {new_name}")
            name_map[t.name] = t.model_copy(update={"name": new_name})
        else:
            name_map[t.name] = t
    # 按 tools_config.yaml 的 mcp scope 分组（配置里写的是服务器原始名）
    result = {}
    for cfg_path in mcp_tools_config_paths:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("tools", []):
            scope = entry.get("mcp_node")
            if not scope:
                continue                        # 跳过本地 module/func 条目
            names = entry["tools"]
            missing = [n for n in names if n not in name_map]
            if missing:
                raise RuntimeError(f"scope [{scope}] 未找到 MCP 工具: {missing}")
            result[scope] = [name_map[n] for n in names]
            for n in names:
                print(f"✅ 成功加载工具：{scope}.{name_map[n].name}")
    return result
# ===================== 人工介入：统一入口拦截 + 介入节点 =====================
def _is_async_callable(fn) -> bool:
    """判断目标函数是否为 async（穿透 partial 与 __wrapped__）"""
    f = fn
    seen = set()
    while True:
        inner = getattr(f, "__wrapped__", None)
        if inner is not None and id(inner) not in seen:
            seen.add(id(f))
            f = inner
            continue
        break
    return inspect.iscoroutinefunction(f)


def with_human_interrupt(fn):
    """
    节点入口统一检查人工终止标记。
    命中则「等当前节点跑完之后」再跳转到人工介入节点，不做抢占式中断。
    注意：CLI 节点不套此装饰器，CLI 的终止由用户通过 CLI 浮层的终止按钮决定。
    同时兼容同步节点与 async 节点（tools 节点是 async）。
    """
    def _tid(state) -> str:
        try:
            return state.get("task_uuid", "") or ""
        except Exception:
            return ""

    def _hit(state) -> bool:
        # 优先级：进程内 flag（主） > state 里的 force_termination（兼容旧逻辑）
        if is_stopped(_tid(state)):
            return True
        try:
            return bool(state.get("force_termination"))
        except Exception:
            return False

    def _abort(state):
        # 消费掉标记，避免跳到 human_intervene 后又被下一轮重复触发
        clear_stop_flag(_tid(state))
        logger.info("检测到人工终止标记，当前节点结束后跳转 human_intervene")
        return Command(goto="human_intervene", update={"force_termination": False})

    if _is_async_callable(fn):
        @wraps(fn)
        async def wrapper(state, *args, **kwargs):
            if _hit(state):
                return _abort(state)
            return await fn(state, *args, **kwargs)
    else:
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            if _hit(state):
                return _abort(state)
            return fn(state, *args, **kwargs)
    return wrapper


def human_intervene_node(state: AgentState, logger=None, ui_dialogger=None):
    """
    人工介入节点：注入 HumanMessage 标记本次介入，并等待用户给出下一步指令。
    复用既有的 interrupt() + /send 通道，前端输入框无需改动。
    """
    logger.info(">>>> 执行节点：【human_intervene_node】")
    # 进入介入节点即清空标记：避免用户在等待输入期间再次点终止，导致 resume 后又绕回来
    clear_stop_flag(state.get("task_uuid", "") or "")
    notice = "【人工介入】流程已暂停，请输入你的下一步指令（输入 /end 直接结束任务）。"
    # 注意：不要把 notice / 用户输入写进 ui_dialogger。
    # 外层 _run_TA_loop_body 已经统一写过一遍（Agent 提问 + User 回复），
    # 而本节点会因 interrupt() 重跑两次，在这里写会重复 2~3 次。

    # 先注入一条 HumanMessage，明确标记「本次是人工介入」
    intervene_msg = HumanMessage(
        content="【人工介入】用户已中断当前流程，请忽略之前未完成的步骤，严格按用户接下来的指令执行。")
    user_input = interrupt(notice)
    user_msg = HumanMessage(content=user_input)

    logger.info(f"人工介入指令：{user_input}")
    record = f"【人工介入】用户指令：{user_input}"

    if user_input.strip().lower() in ("/end", "exit", "结束", "end"):
        return Command(goto="finish_summary",
                       update={"messages": [intervene_msg, user_msg],
                               "project_history": record,
                               "force_termination": False})
    return Command(goto="router",
                   update={"messages": [intervene_msg, user_msg],
                           "project_history": record,
                           "force_termination": False})


def build_task_graph(logger, ui_dialogger):
    valid_intents = []
    # ================== 构建图 ==================
    builder = StateGraph(AgentState)
    builder.add_node("router", with_human_interrupt(partial(router_node, logger=logger, ui_dialogger = ui_dialogger)))
    builder.add_node("task_process", with_human_interrupt(partial(task_process_node, logger=logger, ui_dialogger = ui_dialogger)))
    builder.add_node("task_manage", with_human_interrupt(partial(task_manage_node, logger=logger)))
    builder.add_node("tools", with_human_interrupt(partial(tool_node_with_counter, logger=logger, ui_dialogger = ui_dialogger)))
    builder.add_node("finish_summary", partial(finish_summary_node, logger=logger, ui_dialogger = ui_dialogger))
    builder.add_node("await_new_task", partial(await_new_task_node, logger=logger))
    builder.add_node("compress_node", with_human_interrupt(partial(compress_node, logger=logger, ui_dialogger = ui_dialogger)))
    builder.add_node("human_intervene", partial(human_intervene_node, logger=logger, ui_dialogger=ui_dialogger))
    # if all_prompt.get('CLI', ''):
    #     # CLI 不套 with_human_interrupt：是否终止由用户在 CLI 浮层决定
    #     builder.add_node("CLI", partial(cli_executor_node, logger=logger, ui_dialogger=ui_dialogger))
    #     builder.add_edge("CLI", "task_manage")
    #     valid_intents.append("CLI")
    # 动态添加业务节点
    for node_cfg in ALL_NODES_CONFIG["nodes"]:
        if 'cli_template' in ALL_NODES_CONFIG["nodes"][node_cfg]:
            node_func = with_human_interrupt(
                partial(cli_executor_node, logger=logger, ui_dialogger=ui_dialogger, node_name=node_cfg,
                        node_config=ALL_NODES_CONFIG["nodes"][node_cfg]))
        else:
            node_func = with_human_interrupt(partial(flexible_node, logger=logger, ui_dialogger = ui_dialogger, node_name = node_cfg, node_config=ALL_NODES_CONFIG["nodes"][node_cfg]))
        builder.add_node(node_cfg, node_func)

    # 路由映射也动态生成
    valid_intents += (ALL_NODES_CONFIG["router_valid_intents"])
    route_map = {intent: intent for intent in valid_intents }
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_by_intent, route_map)

    # 为每个业务节点添加条件边（工具 or fallback）
    for node_cfg in ALL_NODES_CONFIG["nodes"]:
        if ALL_NODES_CONFIG["nodes"][node_cfg]['use_tools']:
            builder.add_conditional_edges(
                node_cfg,
                tools_condition_with_log,
                {
                    "tools": "tools",
                    END: ALL_NODES_CONFIG["nodes"][node_cfg].get("fallback_to", "task_manage")
                }
            )
        else: builder.add_edge( node_cfg, ALL_NODES_CONFIG["nodes"][node_cfg].get("fallback_to", "task_manage"))
        print(f"✅ 成功添加节点：{node_cfg}")
    builder.add_edge("tools", "task_manage")
    # 人工介入节点：正常情况靠 Command(goto=...) 跳转，此边仅作结构兜底
    # builder.add_edge("human_intervene", "router")
    #builder.add_conditional_edges("tools", lambda state: state.get("tool_back_node", "task_process"))
    #builder.add_conditional_edges("tools", lambda s: (logger.info(f"[tools] 回跳 -> {s.get('tool_back_node', 'task_process')}") or s.get("tool_back_node", "task_process")))
    builder.add_conditional_edges(
        "task_process",
        tools_condition_with_log,
        {
            "tools": "tools",
            END: "task_manage"
        }
    )
    #builder.add_edge("tech_support", "task_manage")
    builder.add_edge("finish_summary", 'await_new_task') #汇总完成结束
    builder.add_edge("await_new_task", "compress_node")
    builder.add_edge("task_manage", "compress_node")
    builder.add_edge("compress_node", "router")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
async def init_TA_Node(workspace_root, llm_name, plugins, mcps, cli_cmds,logger, ui_dialogger):
    global llm, llm_dict_with_tools, independent_llm, base_tool_node, token_counter, all_prompt, cli_dict, TRIGGER_TOKEN, ALL_NODES_CONFIG
    if workspace_root is not None:
        FileToolConfig.WORKSPACE_ROOT = workspace_root
    elif not FileToolConfig.WORKSPACE_ROOT:
        # 如果未传入且全局未设置，给出友好默认或报错
        raise ValueError("workspace_root must be provided or set via FileToolConfig.WORKSPACE_ROOT before init.")
    # ==================== 获取配置文件  ======================
    llm_config_path = Path(__file__).parent.parent / "LLM_config.json"
    prompt_list = [Path(__file__).parent.parent / "Assets" / "General" / "Prompt.yaml"]
    tools_config_paths= [Path(__file__).parent.parent / "Assets" / "General" / "tools_config.yaml"]
    nodes_config_paths = [Path(__file__).parent.parent / "Assets" / "General" / "nodes_config.yaml"]
    cli_config_path = Path(__file__).parent.parent / "CLI" / "CLI_COMMANDS.yaml"
    for plugin in plugins:
        tools_config_paths.append(Path(__file__).parent.parent / "Plugin" / plugin / "tools_config.yaml")
        nodes_config_paths.append(Path(__file__).parent.parent / "Plugin" / plugin / "nodes_config.yaml")
        prompt_list.append(Path(__file__).parent.parent / "Plugin" / plugin / "Prompt.yaml")
    mcp_config_paths = []
    mcp_tools_config_paths = []
    for mcp in mcps:
        nodes_config_paths.append(Path(__file__).parent.parent / "MCP" / mcp / "nodes_config.yaml")
        prompt_list.append(Path(__file__).parent.parent / "MCP" / mcp / "Prompt.yaml")
        mcp_config_paths.append(Path(__file__).parent.parent / "MCP" / mcp / "mcp_config.json")
        mcp_tools_config_paths.append(Path(__file__).parent.parent / "MCP" / mcp / "tools_config.yaml")

    # ==================== 初始化LLM  ======================
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config_dict = json.load(f)
    llm_config = llm_config_dict[llm_name]
    TRIGGER_TOKEN = llm_config.get('compress_trigger_token', 132768) #前文压缩触发
    token_counter = TokenCounter()
    api_key_env = os.getenv(llm_config.get('api_key', ''))
    api_type = llm_config.get('api_type', '')
    if not api_key_env:
        raise ValueError("API key not found in environment variable")
    api_key = SecretStr(api_key_env)
    if api_type == 'deepseek': # deepseek有专门的api
        llm = ChatDeepSeek(
            base_url=llm_config.get('base_url', None),
            model=llm_config.get('model', None),
            temperature=llm_config.get('temperature', 0.2),
            api_key=api_key,
            max_tokens = 4096,
            frequency_penalty =  0.7,
            presence_penalty = 0.6,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=60
        )
    else:
        llm = ChatOpenAI(
            base_url=llm_config.get('base_url', None),
            model=llm_config.get('model', None),
            temperature=llm_config.get('temperature', 0.2),
            api_key=api_key,
            max_tokens = 4096,
            frequency_penalty= 0.7,
            presence_penalty= 0.6,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=60
        )
    # ==================== 工具统一加载 ====================
    # 有些tool需要独立的LLM会话
    independent_llm = Independent_LLM_Tools(llm_config)
    # 初始全局prompt和节点配置
    all_prompt = {}
    ALL_NODES_CONFIG = {'nodes': {}, 'router_valid_intents': [], 'router_add_prompt':''}
    # 处理CLI命令白名单
    try:
        with open(cli_config_path, "r", encoding="utf-8") as f:
            cli_dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print('未加载到可用CLI命令')
        cli_dict = {}
    cli_router_prompt = ""
    for cli_cmd in cli_cmds:
        cli_router_prompt += cli_cmd + ' : ' + cli_dict['commands'][cli_cmd]['description']
        cli_param_description = "\n".join(f"参数名:{k}; 说明:{v}" for k,v in cli_dict['commands'][cli_cmd]['parameters'].items())
        cli_node_prompt = f'''
命令: {cli_cmd}
命令说明：{cli_dict['commands'][cli_cmd]['description']}
参数如下：
    {cli_param_description}
'''
        all_prompt.update({cli_cmd: cli_node_prompt})
        ALL_NODES_CONFIG['nodes'].update({cli_cmd:{'prompt_key': cli_cmd, 'use_tools': False, 'interrupt': False, 'cli_template':cli_dict['commands'][cli_cmd]['command_template'], 'fallback_to': 'task_process'}})
        ALL_NODES_CONFIG['router_valid_intents'].append(cli_cmd)
        ALL_NODES_CONFIG['router_add_prompt'] += f"- {cli_cmd}：{cli_dict['commands'][cli_cmd]['description']}\n"
    #  本地工具（含 plugin，不含mcp），返回 {node: [tools]}，默认 key 是 general
    tool_dict = load_tools_from_config(tools_config_paths,independent_llm)
    local_names = {
                (t.name if hasattr(t, "name") else type(t).__name__)
                for tools in tool_dict.values()
                for t in tools
                } #当前工具名，用于查重
    # MCP 工具按 scope 分组（连接 → 改名 → 分组）
    mcp_scope_dict = await group_mcp_tools_by_scope(
        mcp_config_paths, mcp_tools_config_paths, local_names, logger)
    # 统一合并：本地 + MCP 全进同一个 tool_dict
    tool_dict.update(mcp_scope_dict)
    # 全量一维去重（供 ToolNode / all 兜底）
    all_tools, seen = [], set()
    for tools in tool_dict.values():
        for t in tools:
            key = t.name if hasattr(t, "name") else type(t).__name__
            if key not in seen:
                seen.add(key)
                all_tools.append(t)
    # scope 预绑定 LLM（请求侧瘦身）+ all 兜底
    llm_dict_with_tools = {scope: llm.bind_tools(tools)
                           for scope, tools in tool_dict.items()}
    llm_dict_with_tools["all"] = llm.bind_tools(all_tools)
    print(f"llm_dict_with_tools {llm_dict_with_tools}")
    base_tool_node = ToolNode(tools=all_tools)  # 执行侧保持全量
    # ====================节点prompt统一加载=======================
    for nodes_config_path in nodes_config_paths:
        with open(nodes_config_path, "r", encoding="utf-8") as f:
            NODES_CONFIG = yaml.safe_load(f)
            ALL_NODES_CONFIG['nodes'].update(NODES_CONFIG['nodes'])
            for intent in NODES_CONFIG['router_valid_intents']:
                if intent not in ALL_NODES_CONFIG['router_valid_intents']:
                    ALL_NODES_CONFIG['router_valid_intents'].append(intent)
    print(ALL_NODES_CONFIG)
    for prompt_path in prompt_list:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = yaml.safe_load(f)
        for key, value in prompt.items():
            if key in all_prompt:
                all_prompt[key] = all_prompt[key] + value  # 或 + "\n" + value
            else:
                all_prompt[key] = value

    all_prompt.update({'router_node': all_prompt['router_node'].replace(
        "{router_valid_intents}", ", ".join(ALL_NODES_CONFIG["router_valid_intents"]))}) # 将节点加入可选方向
    all_prompt['router_node'] += ALL_NODES_CONFIG['router_add_prompt']
    for key, value in all_prompt.items():
        print(key + '\n' + value)
    graph = build_task_graph(logger, ui_dialogger)
    return graph, all_prompt, mcp_session_ctxs
# ================== 主运行逻辑【大幅修复中断处理】 ==================
if __name__ == "__main__":
    # 命令行直跑演示：python Assets/TA_Nodes.py [llm_name] [plugins]
    import sys
    from Logger import setup_logger
    # 工作区根目录：可通过环境变量 TA_WORKSPACE_ROOT 指定，默认使用项目下的 demo_workspace
    workspace_root = os.environ.get("TA_WORKSPACE_ROOT") or str(
        Path(__file__).parent.parent / "demo_workspace"
    )
    os.makedirs(workspace_root, exist_ok=True)
    llm_name = sys.argv[1] if len(sys.argv) > 1 else 'mimo_v2.5'
    plugins = sys.argv[2:] if len(sys.argv) > 2 else ['TA_SDK']
    mcps = ['ConfigurationDesk']
    FileToolConfig.WORKSPACE_ROOT = workspace_root
    logger = setup_logger(os.path.join(workspace_root, 'Agent.log'))
    graph = asyncio.run(init_TA_Node(workspace_root, llm_name, plugins ,mcps, logger))
    async def run_test(graph, workspace_root, logger):
        test_inputs = [
            "你好，先问我想干什么",
        ]

        for user_input in test_inputs:
            print("=" * 60)
            print(f"【初始用户输入】: {user_input}")
            print("=" * 60)

            input_payload = {
                "messages": [HumanMessage(content=user_input)],
                "ds_agent_dir": workspace_root,
                "root_workspace": workspace_root,
                "force_termination": False,
                "project_meta":'待补充'
            }
            config = {
                "recursion_limit": 300,
                "configurable": {
                    "thread_id": "test-session-001"
                }
            }
            while True:
                # 1. 执行一轮流（遇到中断会自动暂停，流正常结束）
                event_stream = graph.astream(input_payload, config=config, stream_mode="updates")
                input_payload = None

                async for event in event_stream:
                    for node_name, state_update in event.items():
                        # 跳过中断虚拟节点，不做日志解析
                        if node_name == "__interrupt__":
                            continue

                        print(f"\n#>> 执行完成节点：【{node_name}】")
                        logger.info(f"#>> 执行完成节点：【{node_name}】")
                        if "intent" in state_update:
                            print(f"    intent = {state_update['intent']}")
                            logger.info(f"    intent = {state_update['intent']}")
                        if "outer_loop_count" in state_update:
                            print(f"    外层循环数 = {state_update['outer_loop_count']}")
                            logger.info(f"    外层循环数 = {state_update['outer_loop_count']}")
                        if "tool_run_list" in state_update:
                            print(f"    工具调用 = {state_update['tool_run_list']}")
                            logger.info(f"    工具调用计数 = {state_update['tool_run_list']}")
                        if "messages" in state_update:
                            latest_msg = state_update["messages"][-1]
                            msg_type = type(latest_msg).__name__
                            preview = latest_msg.content[:200]
                            print(f"    最后消息 [{msg_type}] : {preview}...")
                            logger.info(f"    最后消息 [{msg_type}] : {latest_msg.content}...")

                # 2. 执行完一轮后，从状态快照中检查是否存在中断（官方标准方式）
                snapshot = graph.get_state(config)

                if snapshot.interrupts:
                    # 存在中断，取出中断数据，等待用户输入
                    print(str(snapshot.interrupts[0]))
                    print(type(snapshot.interrupts[0]))
                    interrupt_data = snapshot.interrupts[0].value
                    print("\n========== Agent需要你的输入 ==========")
                    print(f"Agent提问：{interrupt_data}")
                    logger.info(f"Agent提问：{interrupt_data}")
                    user_reply = input("请输入你的回答：")
                    logger.info(f"用户回答：{user_reply}")
                    # 可选：输入 exit 强制跳转结束
                    if user_reply.strip().lower() == "exit":
                        input_payload = Command(goto="finish_summary")
                    else:
                        input_payload = Command(resume=user_reply)
                    continue
                else:
                    # 无中断，流程真正执行完毕
                    break

            # 3. 输出最终结果
            msg_list = snapshot.values.get("messages", [])
            print("\n" + "-" * 50)
            if msg_list:
                print(f"【最终AI回复】: {msg_list[-1].content}\n")
            else:
                print("会话无消息输出")


    asyncio.run(run_test(graph, workspace_root, logger))