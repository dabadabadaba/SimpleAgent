import asyncio
import uuid
import os, sys, traceback, threading, json, yaml, subprocess, signal
from Assets.OrmsgpackConsts import _OrmsgpackConsts
# ormsgpack在windows环境下会导致内存崩溃，使用猴子补丁注入到系统模块缓存
sys.modules["ormsgpack"] = _OrmsgpackConsts()
from pathlib import Path
from datetime import datetime
from typing import TypedDict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, SecretStr
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
import Assets.TA_Nodes as TA_Node
from Assets.Logger import setup_logger, UI_dialogger
from Assets.DS_Utilities import update_node_progress, update_task_status, _write_json, _read_json


# ===================== 全局配置 =====================
LISTEN_PORT = 9988
DEFAULT_ROOT_WORKSPACE = "task_workspaces"
mcp_session_ctxs: list = []

if not os.path.exists(DEFAULT_ROOT_WORKSPACE):
    os.makedirs(DEFAULT_ROOT_WORKSPACE)

app = FastAPI(title="LangGraph Agent - 工作流管理平台")
checkpointer = MemorySaver()
running_tasks: dict[str, asyncio.Task] = {}
task_lock = asyncio.Lock()

# 任务元数据映射：task_uuid → {root_workspace, work_dir, ds_agent_dir, ...}
task_metadata: dict[str, dict] = {}
meta_lock = asyncio.Lock()

# 用户输入等待映射：task_uuid → {"event": threading.Event, "data": dict}
input_events: dict[str, dict] = {}
input_lock = asyncio.Lock()

# 内层 TA 图注册表：task_uuid → {"graph": CompiledGraph, "config": dict}
# 用于把人工终止标记写进「内层图」的 state（外层图的 state 内层看不到）
TA_GRAPH_REGISTRY: dict[str, dict] = {}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== 请求体模型 =====================
class TaskRequest(BaseModel):
    msg: str
    root_workspace: str = DEFAULT_ROOT_WORKSPACE
    llm_base: str = ""
    plugins: list = []
    mcps: str = ""
    cli_cmds: list = []

class StatusUpdateRequest(BaseModel):
    """上传 / 更新工作流状态的请求体"""
    status: str = None
    current_node: str = None
    extra: dict = None

class SendMessageRequest(BaseModel):
    """前端发送用户消息的请求体"""
    message: str

# ===================== 定义流程状态 =====================
class TaskState(TypedDict):
    task_uuid: str
    input_msg: str
    root_workspace: str
    work_dir: str
    ds_agent_dir: str         # ds_agent 子目录路径
    log_file_path: str
    force_termination: bool
    status: str
    llm_base: str
    plugins : list
    mcps: list
    cli_cmds: list

# ===================== 工具函数 =====================

def load_html(filename: str) -> str:
    """从文件加载 HTML 模板"""
    path = os.path.join(_BASE_DIR, filename)
    if not os.path.exists(path):
        return f"<h1>{filename} 未找到</h1>"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===================== ds_agent 状态管理函数 =====================
def init_ds_agent(work_dir: str, task_uuid: str, input_msg: str, root_workspace: str, log_file_path: str) -> str:
    """
    在工作目录下创建 ds_agent/ 子文件夹，初始化 status.json。
    返回 ds_agent_dir 路径。
    """
    ds_dir = os.path.join(work_dir, "ds_agent")
    os.makedirs(ds_dir, exist_ok=True)

    status = {
        "task_uuid": task_uuid,
        "input_msg": input_msg,
        "root_workspace": root_workspace,
        "log_file_path": log_file_path,
        "status": "created",
        "current_node": "",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "nodes": {},
        "extra": {}
    }
    _write_json(os.path.join(ds_dir, "status.json"), status)
    return ds_dir


def _cli_status_hook(ds_dir: str, info: dict):
    """注入给 TA_Nodes：把 CLI 运行状态写进 status.json 的 extra.cli，供前端浮层展示"""
    update_task_status(ds_dir, extra={"cli": info})


TA_Node.set_cli_status_hook(_cli_status_hook)


# ===================== LangGraph 节点 =====================
async def start_web_ui_node(state: TaskState) -> Command:
    if state["force_termination"] is True:
        _on_force_terminate(state, "start_web_ui")
        return Command(goto=END, update={"status": "force_terminated"})

    root_ws = state["root_workspace"] or DEFAULT_ROOT_WORKSPACE
    if not os.path.exists(root_ws):
        os.makedirs(root_ws)

    print(f"[WebUI Node] ROOT_WORKSPACE 就绪: {root_ws}")
    return Command(update={"root_workspace": root_ws, "status": "web_ui_ready"})


async def create_workspace_node(state: TaskState) -> Command:
    if state["force_termination"] is True:
        _on_force_terminate(state, "create_workspace")
        return Command(goto=END, update={"status": "force_terminated"})

    task_uid = state["task_uuid"]
    root_ws = state["root_workspace"]
    log_file_path = state["log_file_path"]
    work_dir = os.path.join(root_ws, 'temp__'+task_uid)
    os.makedirs(work_dir, exist_ok=True)

    # 初始化 ds_agent 子文件夹 + status.json
    ds_agent_dir = init_ds_agent(work_dir, task_uid, state["input_msg"], root_ws, log_file_path)
    update_node_progress(ds_agent_dir, "create_workspace", "running", msg="正在创建工作区...")

    # 更新元数据
    async with meta_lock:
        task_metadata[task_uid] = {
            "root_workspace": root_ws,
            "work_dir": work_dir,
            "ds_agent_dir": ds_agent_dir,
        }

    update_node_progress(ds_agent_dir, "create_workspace", "completed", msg="工作区创建完成")
    print(f"[CreateWorkspace] 任务目录: {work_dir}")
    return Command(update={
        "work_dir": work_dir,
        "ds_agent_dir": ds_agent_dir,
        "status": "running"
    })

async def run_TA_loop(state,ds_dir:str):
    global mcp_session_ctxs, logger, ui_dialogger
    log_file = state["log_file_path"]
    root_workspace = state["root_workspace"]
    llm_base = state.get("llm_base", "")
    plugins = state.get("plugins", [])
    mcps = state.get("mcps", [])
    cli_cmds = state.get("cli_cmds", [])
    logger = setup_logger(os.path.join(ds_dir, r'agent_detail.log'))
    ui_dialogger = UI_dialogger(log_file)
    TA_Node_graph, ALL_PROMPT, mcp_session_ctxs= await TA_Node.init_TA_Node(root_workspace, llm_base, plugins, mcps, cli_cmds, logger, ui_dialogger)
    system_prompt = ALL_PROMPT.get('global_system_prompt', '')
    system_prompt += '当前工作路径：'+ root_workspace
    print(system_prompt)
    task_uuid = state.get("task_uuid", "")
    input_msg = state.get("input_msg", "")
    project_history_seed = [f"## 用户初始需求\n{input_msg}"] if input_msg else []
    input_payload = {
        "messages": [SystemMessage(content=system_prompt),HumanMessage(content=input_msg)],
        "task_uuid": task_uuid,
        "ds_agent_dir": ds_dir,
        "root_workspace": root_workspace,
        "llm_base": llm_base,
        "plugins": plugins,
        "project_meta": '待补充',
        "project_history": project_history_seed,
        "task_manage": "",
        "force_termination": False,
        "token": 0
    }
    config = {
        "recursion_limit": 1300,
        "configurable": {
            # 每个任务独立线程，避免多任务共用同一份 checkpoint
            "thread_id": f"ta-{task_uuid}" if task_uuid else "ta-default"
        }
    }
    # 注册内层图：供 /terminate 把终止标记写进内层 state
    if task_uuid:
        TA_GRAPH_REGISTRY[task_uuid] = {"graph": TA_Node_graph, "config": config}
    try:
        await _run_TA_loop_body(TA_Node_graph, input_payload, config, state, ds_dir)
    finally:
        if task_uuid:
            TA_GRAPH_REGISTRY.pop(task_uuid, None)
            TA_Node.CLI_PROC_REGISTRY.pop(task_uuid, None)


async def _run_TA_loop_body(TA_Node_graph, input_payload, config, state, ds_dir: str):
    """内层图的主循环（原 run_TA_loop 的循环体）"""
    while True:
        try:
            # 1. 执行一轮流（遇到中断会自动暂停，流正常结束）
            event_stream = TA_Node_graph.astream(input_payload, config=config, stream_mode="updates")
            input_payload = None

            async for event in event_stream:
                for node_name, state_update in event.items():
                    # 跳过中断虚拟节点，不做日志解析
                    if node_name == "__interrupt__":
                        continue
                    # logger.info(f"\n>>>> 执行完成节点：【{node_name}】")
                    token = state_update.get('token', None)
                    # task_manage 的阶段进度直接进状态栏（不再依赖 messages 中转）
                    if node_name == "task_manage" and state_update.get("task_manage"):
                        update_node_progress(ds_dir, node_name, "running",
                                             msg=f"\n{state_update['task_manage']}",
                                             token=token or 0)
                    elif token:
                        update_node_progress(ds_dir, node_name, "running", token = token)
                    else:
                        update_node_progress(ds_dir, node_name, "running")
                    # # 节点执行轨迹写进 UI 日志（小字体，避免喧宾夺主）
                    # try:
                    #     ui_dialogger.append_log(
                    #         f"<span class='log-trace'>▸ {node_name}</span>", mode='temp')
                    # except Exception:
                    #     pass
        except GraphInterrupt:
            # 捕获主动中断异常，属于正常业务流程
            pass
        # 2. 执行完一轮后，从状态快照中检查是否存在中断（官方标准方式）
        snapshot = TA_Node_graph.get_state(config)

        if snapshot.interrupts:
            # 存在中断，取出中断数据，等待用户输入
            interrupt_data = snapshot.interrupts[0].value
            ui_dialogger.append_log( f"\n **Agent：**\n{interrupt_data}\n")
            logger.info(f"Agent提问：{interrupt_data}")
            task_uuid = state["task_uuid"]

            # 写入等待输入状态到 status.json（前端轮询检测）
            update_task_status(ds_dir, status="awaiting_input", extra={
                "awaiting_input": True,
                "interrupt_data": interrupt_data
            })

            # 等待前端通过 /send API 提交回复
            evt = threading.Event()
            reply_box = {}
            async with input_lock:
                input_events[task_uuid] = {"event": evt, "data": reply_box}

            # 阻塞等待，定期检查是否被终止
            while not evt.is_set():
                s = _read_json(os.path.join(ds_dir, "status.json"))
                if s and s.get("status") in ("force_terminated", "immediately_terminated"):
                    input_events.pop(task_uuid, None)
                    return
                await asyncio.sleep(0.5)

            async with input_lock:
                input_events.pop(task_uuid, None)

            user_reply = reply_box.get("reply", "")

            # 清除等待状态，恢复 running
            update_task_status(ds_dir, status="running", extra={"awaiting_input": False})

            logger.info(f"用户回答：{user_reply}")
            ui_dialogger.append_log(f"\n\n**User：**\n{user_reply}\n")
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

async def close_mcp_sessions() -> None:
    for ctx in mcp_session_ctxs:
        try:
            await ctx.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("关闭 MCP 会话失败: %s", exc)
    mcp_session_ctxs.clear()
    
async def do_work(state: TaskState) -> Command:
    ds_dir = state.get("ds_agent_dir", "")
    log_file = state.get("log_file_path", "")
    try:
        update_node_progress(ds_dir, "do_work", "running", msg="开始执行业务流程")
        update_task_status(ds_dir, status="running")
        # 在线程池中执行阻塞主体，保持事件循环可用
        try:
            await run_TA_loop(state, ds_dir)
        finally:
            await close_mcp_sessions()
        return Command(update={"status": "completed"})
    except Exception as e:
        full_error = traceback.format_exc()
        update_node_progress(ds_dir, "do_work", "error", msg=f"业务流程异常: {e}")
        update_task_status(ds_dir, status="error", extra={"error": str(e)})
        ui_dialogger.append_log( f"\n## 任务异常: {e} \n {full_error} ")
        print(f"\n任务异常: {e} \n {full_error} ")
        # 可以选择跳转到错误处理节点，或直接结束
        return Command(goto="router", update={"status": "error"})


def _on_force_terminate(state: TaskState, node_name: str):
    """统一处理强制终止：写日志 + 更新 ds_agent 状态"""
    ds_dir = state.get("ds_agent_dir", "")
    log_path = state.get("log_file_path", "")
    if ds_dir:
        update_node_progress(ds_dir, node_name, "skipped", msg=f"跳过节点: {node_name}（任务已终止）")
        update_task_status(ds_dir, status="force_terminated")
    if log_path and os.path.exists(log_path):
        try:
            ui_dialogger.append_log( "\n【手动标记终止】流程中途停止\n")
        except Exception as e:
            print(f"写入终止日志失败: {e}")
# ===================== 构建流程图 =====================
def build_single_task_graph():
    builder = StateGraph(TaskState)
    builder.add_node("start_web_ui", start_web_ui_node)
    builder.add_node("create_workspace", create_workspace_node)
    builder.add_node("do_work", do_work)
    builder.add_edge(START, "start_web_ui")
    builder.add_edge("start_web_ui", "create_workspace")
    builder.add_edge("create_workspace", "do_work")
    builder.add_edge("do_work", END)

    return builder.compile(checkpointer=checkpointer)


graph_instance = build_single_task_graph()

# ===================== 异步执行单个流程 =====================
async def exec_graph_task(task_uid: str, msg: str, root_workspace: str, llm_base: str,
                          plugins: list, mcps: str = "", cli_cmds: list = None):
    cfg = {"configurable": {"thread_id": task_uid}}
    log_dir = os.path.join(root_workspace, 'temp__' + task_uid)
    log_file = os.path.join(log_dir, r"agent.log")
    init_state = {
        "task_uuid": task_uid,
        "input_msg": msg,
        "root_workspace": root_workspace,
        "llm_base": llm_base,
        "plugins": plugins,
        "mcps": [mcps] if mcps else [],
        "cli_cmds": list(cli_cmds or []),
        "work_dir": "",
        "ds_agent_dir": "",
        "log_file_path": log_file,
        "force_termination": False,
        "status": "init"
    }
    try:
        await graph_instance.ainvoke(init_state, config=cfg)
    except asyncio.CancelledError:
        print(f"[{task_uid}] 任务被外部取消")
    except Exception as e:
        print(f"[{task_uid}] 任务异常: {e}")
        ui_dialogger.append_log(f"## {task_uid} 任务异常: {e} 任务终止")
    finally:
        TA_GRAPH_REGISTRY.pop(task_uid, None)
        TA_Node.CLI_PROC_REGISTRY.pop(task_uid, None)
        TA_Node.clear_stop_flag(task_uid)
        async with task_lock:
            running_tasks.pop(task_uid, None)


# ===================== 提交入口 =====================
async def spawn_task(msg: str, root_workspace: str, llm_base: str = "", plugins: list = None,
                     mcps: str = "", cli_cmds: list = None) -> str:
    task_uid = str(uuid.uuid4())
    bg_task = asyncio.create_task(
        exec_graph_task(task_uid, msg, root_workspace, llm_base,
                        plugins or [], mcps, cli_cmds or [])
    )
    async with task_lock:
        running_tasks[task_uid] = bg_task
    return task_uid


# ===================== HTTP 接口 =====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=load_html("index.html"))


@app.get("/info")
def server_info():
    return {
        "port": LISTEN_PORT,
        "default_workspace": DEFAULT_ROOT_WORKSPACE,
        "base_dir": _BASE_DIR,
    }


@app.get("/llm-config")
def list_llm_configs():
    """返回 LLM_config.json 中的配置 key 列表"""
    config_path = os.path.join(_BASE_DIR, "LLM_config.json")
    if not os.path.exists(config_path):
        return {"keys": []}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"keys": list(data.keys())}


@app.get("/plugins")
def list_plugins():
    """返回 Plugin 目录下的所有子文件夹名"""
    plugin_dir = os.path.join(_BASE_DIR, "Plugin")
    if not os.path.isdir(plugin_dir):
        return {"plugins": []}
    plugins = [d for d in os.listdir(plugin_dir)
               if os.path.isdir(os.path.join(plugin_dir, d))]
    return {"plugins": plugins}


@app.get("/mcps")
def list_mcps():
    """返回 MCP 目录下的所有子文件夹名"""
    mcp_dir = os.path.join(_BASE_DIR, "MCP")
    if not os.path.isdir(mcp_dir):
        return {"mcps": []}
    mcps = [d for d in os.listdir(mcp_dir)
            if os.path.isdir(os.path.join(mcp_dir, d))]
    return {"mcps": mcps}


@app.get("/cli-commands")
def list_cli_commands():
    """返回 CLI/CLI_COMMANDS.yaml 中 commands 下的所有命令名 + 描述"""
    cli_path = os.path.join(_BASE_DIR, "CLI", "CLI_COMMANDS.yaml")
    if not os.path.isfile(cli_path):
        return {"cli_commands": []}
    try:
        with open(cli_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"加载 CLI_COMMANDS.yaml 失败: {e}")
        return {"cli_commands": []}
    cmds = data.get("commands") or {}
    return {
        "cli_commands": [
            {"name": k, "description": (v or {}).get("description", "")}
            for k, v in cmds.items()
        ]
    }


@app.post("/submit")
async def submit_task(req: TaskRequest):
    if not req.msg.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    if not req.root_workspace.strip():
        raise HTTPException(status_code=400, detail="ROOT_WORKSPACE 不能为空")
    if not req.llm_base.strip():
        raise HTTPException(status_code=400, detail="请选择 LLM_Base 配置")
    task_uid = await spawn_task(req.msg, req.root_workspace, req.llm_base,
                                req.plugins, req.mcps, req.cli_cmds)
    return {
        "code": 0,
        "msg": "任务已后台启动",
        "task_uuid": task_uid,
        "root_workspace": req.root_workspace,
        "llm_base": req.llm_base,
        "plugins": req.plugins,
        "mcps": req.mcps,
        "cli_cmds": req.cli_cmds
    }


# ---------- 终止接口 ----------
@app.post("/terminate/{task_uuid}")
async def terminate_task(task_uuid: str):
    """
    请求中断（温柔模式）：设置进程内终止标记。
    当前节点跑完后，下一个节点入口的 with_human_interrupt 会检测到并跳转 human_intervene。

    注意：不能改用 graph.aupdate_state 写 force_termination ——
    图正在运行时，Pregel 循环用的是它启动时加载的内存 checkpoint，
    外部写入会在下一个 superstep 被覆盖，节点永远看不到。
    """
    if task_uuid not in running_tasks:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")

    TA_Node.set_stop_flag(task_uuid)          # ← 关键：进程内 flag，节点入口直接读

    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir:
        update_task_status(ds_dir, status="terminating",
                           extra={"terminate_requested": True})
    print(f"[{task_uuid}] 已设置人工终止标记，等待当前节点结束")
    return {"msg": "已请求中断，将在当前节点结束后暂停并等待你的指令"}


@app.post("/task/{task_uuid}/cli/kill")
async def kill_cli(task_uuid: str):
    """
    仅终止 CLI 子进程，不打断图。
    进程被杀 → 管道关闭 → 读取循环自然结束 → cli_executor_node 正常返回并转入人工介入。
    """
    entry = TA_Node.CLI_PROC_REGISTRY.get(task_uuid)
    if not entry:
        raise HTTPException(status_code=404, detail="当前没有正在执行的 CLI 命令")
    proc = entry.get("proc")
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=404, detail="CLI 进程已结束")

    entry["killed"] = True
    ds_dir = entry.get("ds_dir") or _resolve_ds_agent_dir(task_uuid)
    try:
        if os.name == "nt":
            # Windows: shell=True 时 Popen.terminate() 只杀 cmd.exe，孙进程会变孤儿
            # 必须 taskkill /T（杀进程树）/F（强制）
            subprocess.run(f"taskkill /PID {proc.pid} /T /F",
                           shell=True, capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        print(f"[CLI终止异常] {e}")
    if ds_dir:
        update_task_status(ds_dir, status="terminating", extra={
            "cli": {"command": entry.get("command", ""), "running": False,
                    "pid": proc.pid, "reason": "用户终止", "killing": True}
        })
    return {"msg": "已终止 CLI 命令，节点收尾后将转入人工介入"}


@app.post("/terminate_immediately/{task_uuid}")
async def terminate_immediately(task_uuid: str):
    """紧急兜底：直接取消整个任务 Task（会丢失当前节点结果，仅在无响应时使用）"""
    if task_uuid not in running_tasks:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    TA_Node.set_stop_flag(task_uuid)
    task = running_tasks[task_uuid]
    task.cancel()
    return {"msg": "已发送立即中断信号"}


# ---------- 任务日志 / 状态接口 ----------
def _resolve_ds_agent_dir(task_uuid: str) -> str | None:
    """根据 task_uuid 查找 ds_agent_dir"""
    meta = task_metadata.get(task_uuid)
    if meta and os.path.isdir(meta["ds_agent_dir"]):
        return meta["ds_agent_dir"]
    # 回退：扫描默认 workspace
    ws = DEFAULT_ROOT_WORKSPACE
    candidate = os.path.join(ws, task_uuid, "ds_agent")
    if os.path.isdir(candidate):
        return candidate
    return None


@app.get("/task/{task_uuid}/log")
async def get_task_log(task_uuid: str):
    """获取任务的完整日志"""
    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在或 ds_agent 未初始化")

    work_dir = os.path.dirname(ds_dir)
    log_path = os.path.join(work_dir, r"agent.log")
    if not os.path.exists(log_path):
        return {"task_uuid": task_uuid, "log": "(暂无日志)", "lines": 0}

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.count("\n") + (1 if content else 0)
    return {"task_uuid": task_uuid, "log": content, "lines": lines}


@app.get("/task/{task_uuid}/log/tail")
async def get_task_log_tail(task_uuid: str, lines: int = Query(default=50, ge=1, le=500)):
    """获取任务日志的尾部 N 行"""
    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在或 ds_agent 未初始化")

    work_dir = os.path.dirname(ds_dir)
    log_path = os.path.join(work_dir, r"agent.log")
    if not os.path.exists(log_path):
        return {"task_uuid": task_uuid, "log": "(暂无日志)", "lines": 0}

    with open(log_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    tail_lines = all_lines[-lines:]
    return {
        "task_uuid": task_uuid,
        "log": "".join(tail_lines),
        "lines": len(tail_lines),
        "total_lines": len(all_lines)
    }


# ===================== 日志窗口读取（供任务详情页分页加载） =====================
LOG_DEFAULT_WINDOW = 64 * 1024          # 默认窗口 64KB
LOG_MAX_WINDOW = 1024 * 1024            # 单次上限 1MB


def _log_file_path(task_uuid: str):
    """解析任务的 agent.log 路径；任务不存在返回 None"""
    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir is None:
        return None
    return os.path.join(os.path.dirname(ds_dir), r"agent.log")


@app.get("/task/{task_uuid}/log/window")
async def get_task_log_window(
    task_uuid: str,
    after: int = Query(default=-1, ge=-1, description="增量起点字节，-1 表示不用"),
    before: int = Query(default=-1, ge=-1, description="翻页终点字节，-1 表示不用"),
    limit: int = Query(default=LOG_DEFAULT_WINDOW, ge=1, le=LOG_MAX_WINDOW),
):
    """按字节窗口读取日志，供前端做「增量追加」与「加载更早」。

    - after  >= 0：返回 [after, EOF)              增量轮询，耗时与日志总长无关
    - before >= 0：返回 [before-limit, before)    向更早翻页
    - 都不传     ：返回末尾 limit 字节             首次加载

    窗口起点会对齐到换行边界；终点严格等于上一页的起点，不裁剪。
    """
    path = _log_file_path(task_uuid)
    if path is None:
        raise HTTPException(status_code=404, detail="任务不存在或 ds_agent 未初始化")
    if not os.path.exists(path):
        return {"log": "", "start": 0, "end": 0, "size": 0, "has_more": False}

    size = os.path.getsize(path)

    if after >= 0:
        start, end = min(after, size), size
    elif before >= 0:
        end = min(before, size)
        start = max(0, end - limit)
    else:
        end, start = size, max(0, size - limit)

    if end <= start:
        return {"log": "", "start": start, "end": end,
                "size": size, "has_more": start > 0}

    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    # 只把起点对齐到换行边界：被丢弃的首个半行会在下次向前翻页时补上。
    # 终点不能裁剪 —— 它正好等于已加载内容的起点，裁掉会让那一行的开头永久丢失。
    if start > 0:
        nl = raw.find(b"\n")
        if nl >= 0:
            raw = raw[nl + 1:]
            start += nl + 1

    return {
        "log": raw.decode("utf-8", errors="replace"),
        "start": start,
        "end": end,
        "size": size,
        "has_more": start > 0,
    }


@app.get("/task/{task_uuid}/status")
async def get_task_status(task_uuid: str):
    """获取工作流状态（读取 ds_agent/status.json）"""
    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在或 ds_agent 未初始化")

    status_path = os.path.join(ds_dir, "status.json")
    status = _read_json(status_path)
    if status is None:
        return {"task_uuid": task_uuid, "status": "unknown", "message": "status.json 未初始化"}

    # 追加运行中标记
    status["is_running"] = task_uuid in running_tasks
    return status


@app.post("/task/{task_uuid}/status")
async def post_task_status(task_uuid: str, req: StatusUpdateRequest):
    """
    上传 / 更新工作流状态。
    外部系统可通过此接口回写 ds_agent/status.json 中的字段。
    """
    ds_dir = _resolve_ds_agent_dir(task_uuid)
    if ds_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在或 ds_agent 未初始化")

    kwargs = {}
    if req.status is not None:
        kwargs["status"] = req.status
    if req.current_node is not None:
        kwargs["current_node"] = req.current_node
    if req.extra is not None:
        kwargs["extra"] = req.extra

    update_task_status(ds_dir, **kwargs)
    return {"code": 0, "msg": "状态已更新", "task_uuid": task_uuid}


@app.post("/task/{task_uuid}/send")
async def send_user_message(task_uuid: str, req: SendMessageRequest):
    """接收前端用户输入，恢复被 interrupt() 暂停的 Agent 任务"""
    async with input_lock:
        entry = input_events.get(task_uuid)
    if entry is None:
        raise HTTPException(status_code=400, detail="任务未等待用户输入")
    entry["data"]["reply"] = req.message
    entry["event"].set()
    return {"code": 0, "msg": "消息已发送", "task_uuid": task_uuid}


# ---------- 任务详情页面 ----------
@app.get("/task/{task_uuid}/detail", response_class=HTMLResponse)
async def task_detail_page(task_uuid: str):
    """任务详情页面"""
    return HTMLResponse(content=load_html("task_detail.html"))


# ---------- 文件系统浏览接口 ----------
@app.get("/drives")
def list_drives():
    """返回可用盘符（Windows）/ 根目录（Linux/Mac）"""
    if os.name == "nt":
        import string
        drives = []
        for letter in string.ascii_uppercase:
            p = f"{letter}:\\"
            if os.path.exists(p):
                drives.append({"path": p, "name": f"本地磁盘 ({letter}:)"})
        return {"drives": drives}
    else:
        return {"drives": [{"path": "/", "name": "根目录 /"}]}


@app.get("/browse")
def browse_directory(path: str = Query(default="")):
    """
    浏览指定路径下的子目录，返回绝对路径列表。
    前端通过此接口实现文件系统导航。
    """
    if not path or not os.path.exists(path):
        # 默认：Windows 返回盘符，其他返回 /
        if os.name == "nt":
            import string
            entries = []
            for letter in string.ascii_uppercase:
                p = f"{letter}:\\"
                if os.path.exists(p):
                    entries.append({"name": f"本地磁盘 ({letter}:)", "path": p, "is_dir": True})
            return {
                "current_path": None,
                "parent_path": None,
                "entries": entries,
                "is_root": True
            }
        else:
            path = "/"

    norm = os.path.normpath(os.path.abspath(path))
    try:
        items = os.listdir(norm)
    except PermissionError:
        return {
            "current_path": norm,
            "parent_path": os.path.dirname(norm) if norm != os.path.dirname(norm) else None,
            "entries": [],
            "error": "无访问权限",
            "is_root": False
        }

    entries = []
    for name in sorted(items, key=lambda x: x.lower()):
        full = os.path.join(norm, name)
        if os.path.isdir(full) and not name.startswith("."):
            entries.append({"name": name, "path": full, "is_dir": True})

    # 计算父目录：盘符根目录（如 C:\）的父级返回 None 回到盘符列表
    parent = os.path.dirname(norm)
    if os.name == "nt" and norm.endswith(":\\"):
        parent_path = None   # 回到盘符选择
        is_root = True
    elif norm == "/":
        parent_path = None
        is_root = True
    else:
        parent_path = parent
        is_root = False

    return {
        "current_path": norm,
        "parent_path": parent_path,
        "entries": entries,
        "is_root": is_root
    }


# ---------- 列出接口 ----------
@app.get("/list_workspaces")
def list_workspaces():
    ws = DEFAULT_ROOT_WORKSPACE
    if not os.path.exists(ws):
        return {"workspaces": []}
    dirs = [d for d in os.listdir(ws) if os.path.isdir(os.path.join(ws, d))]
    return {"root_workspace": ws, "workspaces": dirs}


@app.get("/list_tasks")
def list_tasks():
    active = list(running_tasks.keys())
    return {"code": 0, "active_running_tasks": active}


@app.get("/list_all_tasks")
def list_all_tasks():
    """列出所有任务（包括已结束的，来自 task_metadata）"""
    tasks = []
    for tid, meta in task_metadata.items():
        ds_dir = meta.get("ds_agent_dir", "")
        status_path = os.path.join(ds_dir, "status.json") if ds_dir else ""
        current_status = None
        if status_path and os.path.exists(status_path):
            s = _read_json(status_path)
            if s:
                current_status = s.get("status")
        tasks.append({
            "task_uuid": tid,
            "status": current_status or "unknown",
            "is_running": tid in running_tasks,
            "work_dir": meta.get("work_dir", ""),
        })
    return {"code": 0, "tasks": tasks}


# ===================== 启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    print(f"=" * 55)
    print(f"  LangGraph Agent 工作流管理平台")
    print(f"  主界面:     http://localhost:{LISTEN_PORT}")
    print(f"  API 文档:   http://localhost:{LISTEN_PORT}/docs")
    print(f"=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, access_log=False, log_level="critical")