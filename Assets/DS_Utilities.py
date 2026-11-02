import pandas as pd
from datetime import datetime
from charset_normalizer import from_path
from openpyxl import load_workbook
from pathlib import Path
import re, os, json, yaml, importlib, sys, tiktoken
project_root = str(Path(__file__).parent.parent / "Assets" )
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typing import List, Callable, Dict
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage

# ================== ToolMessage 提取规范 ==================
_TOOL_FULL_LIMIT = 500      # ≤ 500 字符：全量展示
_TOOL_EDGE_LIMIT = 200      # > 500 字符：保留前 200 + 后 200

# 关键字段：抽成 key=value 摘要，与头尾裁剪并用，不互斥
_TOOL_KEY_FIELDS = (
    "success", "ok", "status", "error", "error_code", "error_detail",
    "message", "detail", "reason",
    "path", "file", "file_path", "dir", "count", "total", "total_lines",
    "project_name", "application_name", "connected", "verified", "exit_code",
)

def _content_to_text(content) -> str:
    """ToolMessage.content 归一化：兼容 str / MCP content-block 列表 / None。
    必须先把多 block 拼完整，后面的长度判断才准确。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if "text" in b:
                    parts.append(str(b.get("text", "")))
                elif b.get("type") == "image":
                    parts.append("[image]")
                elif b.get("type") == "resource":
                    parts.append(f"[resource] {b.get('resource', '')}")
                else:
                    parts.append(json.dumps(b, ensure_ascii=False))
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return str(content)
def summarize_tool_message(msg: ToolMessage) -> str:
    """一条 ToolMessage → 一行审计摘要：状态 + 关键字段 + 原文（头尾采样）。"""
    text = _content_to_text(msg.content)          # ← 先归一化

    # 状态：用 LangChain 原生 status，不靠猜 JSON 里的 success
    flag = {"error": "❌", "success": "✅"}.get(getattr(msg, "status", None), "•")

    # 关键字段：在「完整文本」上解析（必须在裁剪之前）
    fields = ""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            fields = " | ".join(
                f"{k}={payload[k]}" for k in _TOOL_KEY_FIELDS if k in payload)
        elif isinstance(payload, list):
            fields = f"items={len(payload)}"
            if payload and isinstance(payload[0], dict):
                first = " | ".join(
                    f"{k}={payload[0][k]}" for k in _TOOL_KEY_FIELDS if k in payload[0])
                if first:
                    fields += f" | first: {first}"
    except Exception:
        pass                                       # 非 JSON：跳过字段，不影响原文兜底

    name = getattr(msg, "name", None) or "unknown_tool"
    parts = [f"{flag} {name}"]
    if fields:
        parts.append(fields)
    if text.strip():
        parts.append(f"结果: {_clip_result(text)}")   # ← 再裁剪
    return "  - " + " | ".join(parts) + '\n'
def _clip_result(text: str,
                 full_limit: int = _TOOL_FULL_LIMIT,
                 edge: int = _TOOL_EDGE_LIMIT) -> str:
    """≤ full_limit：原样返回；否则 前 edge + 省略说明 + 后 edge。"""
    n = len(text)
    if n <= full_limit:
        return text
    head, tail = text[:edge], text[-edge:]
    omitted = n - len(head) - len(tail)
    return f"{head}…（中间省略 {omitted} 字符）…{tail}"
# ======================================================

def update_node_progress(ds_agent_dir: str, node_name: str, node_status: str, msg: str = None, token: int | None = None):
    """
    更新 workflow 中某个节点的执行状态，同时可添加文本进度条目。
    node_status: "running" | "completed" | "error" | "skipped"
    msg: 可选的文本进度描述，会追加到 progress_entries 列表供前端展示。
    """
    status_path = os.path.join(ds_agent_dir, "status.json")
    status = _read_json(status_path)
    if status is None:
        return

    nodes = status.setdefault("nodes", {})
    now = datetime.now().isoformat()

    if node_name not in nodes:
        nodes[node_name] = {"status": node_status, "started_at": None, "completed_at": None}

    nodes[node_name]["status"] = node_status
    if node_status == "running":
        nodes[node_name]["started_at"] = now
    elif node_status in ("completed", "error", "skipped"):
        nodes[node_name]["completed_at"] = now

    # 更新当前进度文本（直接替换，不追加）
    if msg is not None:
        status["progress_text"] = msg
        status["progress_time"] = now

    status["current_node"] = node_name
    status["updated_at"] = now
    if token is not None:  # 不传 / 传 None 时保持原值，避免被默认值清零
        status["token"] = token
    _write_json(status_path, status)

def update_task_status(ds_agent_dir: str, **kwargs):
    """
    上传 / 更新工作流整体状态。支持更新 status, current_node, extra 等字段。
    """
    status_path = os.path.join(ds_agent_dir, "status.json")
    status = _read_json(status_path)
    if status is None:
        return

    if "extra" in kwargs and kwargs["extra"] is not None:
        existing_extra = status.get("extra", {})
        existing_extra.update(kwargs.pop("extra"))
        kwargs["extra"] = existing_extra

    status.update(kwargs)
    status["updated_at"] = datetime.now().isoformat()
    _write_json(status_path, status)

# 在 AgentFrame.py 中添加一个 JSON 序列化辅助函数
def _json_serializable(obj):
    """将对象转换为 JSON 可序列化的格式"""
    if hasattr(obj, 'value') and hasattr(obj, 'id'):
        # 处理 Interrupt 对象
        return {
            "value": obj.value,
            "id": obj.id
        }
    if hasattr(obj, '__dict__'):
        # 处理普通对象
        return obj.__dict__
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _json_serializable(value) for key, value in obj.items()}
    return str(obj)

def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: str, data: dict):
    """安全的 JSON 写入，处理不可序列化的对象"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_serializable)
    except TypeError as e:
        # 如果还是失败，尝试更激进的转换
        print(f"JSON 序列化失败: {e}")
        # 递归处理数据
        serializable_data = _json_serializable(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)


# 封装Token缓存类，替代全局变量，解决语法报错
class TokenCounter:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.cache: Dict[str, int] = {}

    @staticmethod
    def _content_to_str(content) -> str:
        """把 LangChain 消息 content 归一化为纯文本（兼容 str / 多模态 list / None）"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    # 多模态块：优先取 text 字段，没有就整体序列化
                    parts.append(str(block.get("text") or block))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def get_msg_token(self, msg: BaseMessage) -> int:
        if msg.id in self.cache:
            return self.cache[msg.id]
        length = len(self.enc.encode(self._content_to_str(msg.content)))
        self.cache[msg.id] = length
        return length

    def count_all_tokens(self, messages: List[BaseMessage]) -> int:
        return sum(self.get_msg_token(m) for m in messages)

    def clean_cache(self, remove_ids: List[str]):
        for mid in remove_ids:
            self.cache.pop(mid, None)

def abb_with_head_tail(text:str, len_limit = 120) -> str:
    """
    只留头尾省略法
    """
    return text[:int(len_limit*0.5)] + "...省略..." + text[-int(len_limit*0.3):] if len(text) > len_limit else text
def flatten_llm_message(msg: BaseMessage) -> BaseMessage:
    """
    处理单条消息：剥离所有工具调用元数据，仅保留纯文本语义
    - 带 tool_calls 的 AIMessage → 纯文本 AIMessage
    - ToolMessage → 纯文本 AIMessage
    - 其他普通消息 → 原样返回
    """
    # 情况1：携带工具调用的 AI 消息
    if isinstance(msg, AIMessage) and msg.tool_calls:
        base_content = msg.content if isinstance(msg.content, str) else ""

        tool_call_texts = []
        for tc in msg.tool_calls:
            tool_name = tc.get("name", "unknown_tool")
            tool_args = tc.get("args", {})
            tool_call_texts.append(
                f"[调用工具] {tool_name}\n参数：{json.dumps(tool_args, ensure_ascii=False)}"
            )

        full_content = base_content + "\n" + "\n".join(tool_call_texts) if base_content else "\n".join(tool_call_texts)
        # 仅保留 content，丢弃所有 tool_calls / additional_kwargs 等工具字段
        return AIMessage(content=full_content.strip())

    # 情况2：工具返回消息
    if isinstance(msg, ToolMessage):
        tool_name = getattr(msg, "name", "tool")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        flat_content = f"[工具执行结果] {tool_name}\n{content}"
        return AIMessage(content=flat_content)
    if isinstance(msg, HumanMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        flat_content = f"** 用户指令 ： {content} **"
        return HumanMessage(content=flat_content)
    # 情况3：普通消息直接返回
    return msg
def output_messages_json(messages,path):
    serializable_data = [msg.model_dump() for msg in messages]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    print("文件已生成：langgraph_messages.json")
def parse_dsml_tool_calls(dsml_text: str) -> List[Dict]:
    """
    解析 <｜｜DSML｜｜tool_calls> 标记文本，输出langchain风格tool_calls列表
    """
    invoke_pattern = re.compile(
        r'<｜｜DSML｜｜invoke name="(?P<tool_name>[^"]+)">\s*'
        r'<｜｜DSML｜｜parameter name="(?P<param_name>[^"]+)"[^>]*>(?P<param_value>.*?)</｜｜DSML｜｜parameter>\s*'
        r'</｜｜DSML｜｜invoke>',
        re.DOTALL
    )
    matches = invoke_pattern.findall(dsml_text)
    tool_calls = []
    for idx, (tool_name, param_name, param_value) in enumerate(matches):
        call_id = f"call_{idx}"
        args = {param_name: param_value}
        tool_calls.append({
            "id": call_id,
            "name": tool_name,
            "args": args
        })
    return tool_calls
def clean_DSML(text):
    """
    清洗文本：删除所有<｜｜DSML｜｜tool_calls>...</｜｜DSML｜｜tool_calls>标签及其内容
    输出langchain风格tool_calls列表
    Args:
        text: 原始文本
    Returns:
        清洗后的文本
    """
    # 删除tool_calls标签及其内容
    pattern = r'<｜｜DSML｜｜tool_calls>.*?</｜｜DSML｜｜tool_calls>'
    # 先提取所有匹配的内容
    removed_list = re.findall(pattern, text, flags=re.DOTALL)
    # 将多个匹配片段拼接成一个字符串，给下层正则解析
    removed_concat = "\n".join(removed_list)
    # 再删除匹配内容
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    # tool_calls列表
    tool_calls = parse_dsml_tool_calls(removed_concat)
    # 清理多余空白
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip(), tool_calls
def normalize_response_with_DSML(response: BaseMessage, logger, with_tool: True) -> BaseMessage:
    """
        清洗文本：删除所有<｜｜DSML｜｜tool_calls>...</｜｜DSML｜｜tool_calls>标签及其内容
        转化为正常的langgraph toolcalls
        Returns:
            正常化的response,with_tool控制是否包含完整tool calls
    """
    logger.warning("触发强制清洗DSML转化")
    logger.warning(f"对象{response.content}")
    cleaned_content, tool_calls = clean_DSML(response.content)
    if with_tool:
        response = response.model_copy(update={"content": cleaned_content, "tool_calls":tool_calls})
    else:
        response = response.model_copy(update={"content": cleaned_content})
    return response

def load_tools_from_config(config_list, llm) -> List[Callable]:
    """从YAML配置文件动态加载所有工具函数"""
    project_root = str(Path(__file__).parent.parent / "Assets" )
    # 使用 insert(0, path) 确保自定义路径的优先级最高，避免覆盖标准库路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"已将项目根目录添加到模块搜索路径: {project_root}")
    tools_config = []
    for config_path in config_list:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            for item in config["tools"]:
                if item not in tools_config:
                    tools_config.append(item)
    tools = {}
    for item in tools_config:
        module_name = item["module"]
        func_name = item["func"]
        node_name = item.get('node','general')
        tools.setdefault(node_name, [])
        try:
            # 动态导入模块
            mod = importlib.import_module(module_name)
            # 如果模块包含 LLM_CONFIG 类，注入 llm 实例
            if hasattr(mod, "LLM_CONFIG"):
                mod.LLM_CONFIG.llm = llm
                print(f"🔧 为 {module_name}.{func_name} 注入 llm 实例")
            # 获取工具函数
            tool_func = getattr(mod, func_name)
            tools[node_name].append(tool_func)
            print(f"✅ 成功加载工具：{module_name}.{func_name}")
        except ModuleNotFoundError:
            print(f"❌ 模块未找到: {module_name}")
        except AttributeError:
            print(f"❌ 函数未找到: {module_name}.{func_name}")
        except Exception as e:
            print(f"❌ 加载工具失败 {module_name}.{func_name}: {e}")
    return tools
def mapping_list_markdown(input_list: list, col_list: list):
    """
    将两层List和列名转换为Markdown表格字符串
    专用于TASDK .csv格式 mapping文件
    """
    # 1. 基础校验
    if not input_list or len(input_list[0]) != len(col_list):
        print("输入不匹配或数据为空")
        return None

    # 2. 生成表头 (例如: | 姓名 | 年龄 | 城市 |)
    header = "| " + " | ".join(str(c) for c in col_list) + " |"

    # 3. 生成分隔线 (例如: | --- | --- | --- |)
    separator = "| " + " | ".join("---" for _ in col_list) + " |"

    # 4. 生成数据行
    rows = []
    for row in input_list:
        # 将每个单元格转为字符串，防止 None 或数字类型报错
        row_str = " | ".join(str(cell) if cell is not None else "" for cell in row)
        rows.append(f"| {row_str} |")

    # 5. 拼接完整的 Markdown 表格
    markdown_table = "\n".join([header, separator] + rows)

    return markdown_table
def extract_excel_with_pandas(file_path, start_row=None, columns=None):
    """
    使用pandas读取CSV文件，提取指定行列数据并转为二维列表
    :param file_path: CSV文件路径
    :param start_row: 起始行号（业务行号，从1开始），None则读取全部行
    :param columns: 需要提取的列号列表（业务列号，从1开始），None则读取全部列
    :return: list[list] 二维数据列表，空值填充为空字符串
    """
    # 读取参数处理
    skiprows = start_row - 1 if start_row is not None else None
    usecols = [c - 1 for c in columns] if columns is not None else None

    df = pd.read_csv(
        file_path,
        header=None,
        usecols=usecols,
        skiprows=skiprows,
        encoding='utf-8'
    )
    df = df.fillna('')
    return df.to_numpy().tolist()
def generate_file_md(file_path: str, max_chars=30000):
    """
    根据文件路径生成包含文件内容的Markdown文本
    自动识别文件编码、匹配代码块语法、展示文件元信息，超长内容自动截断

    Args:
        file_path: 文件路径字符串
        max_chars: 内容最大字符上限，超出后截断，默认30000

    Returns:
        str: 组装完成的Markdown字符串；文件不存在时返回携带错误提示的Markdown文本
    输出Markdown包含内容：
        1. 文件名称一级标题
        2. 文件大小、检测编码信息备注
        3. 超长截断警告（触发时）
        4. 对应语言代码块包裹文件文本
    异常处理：
        - 文件不存在：返回错误提示MD
        - 无法识别编码/二进制文件：输出提示，不展示代码块
        - 权限不足、读取异常：在MD内嵌入异常信息
    """
    fp = Path(file_path)
    if not fp.exists():
        print(f"❌ 文件不存在: {file_path}")
        return f"❌ 文件不存在: {file_path}"
    # 后缀对应代码块语言
    ext = fp.suffix.lower()
    lang_map = {
        ".py": "python",
        ".txt": "",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml"
    }
    lang = lang_map.get(ext, "")
    # 自适应编码读取（整合原 read_file_as_text 逻辑）
    try:
        detect_result = from_path(file_path).best()
        if detect_result is None:
            return f"# {fp.name}\n> ❌ 无法识别文本编码，疑似二进制文件，无法展示内容"
        full_text = str(detect_result)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return f"❌ 文件不存在: {file_path}"
    except PermissionError:
        return f"# {fp.name}\n> ❌ 无文件读取权限"
    except Exception as e:
        return f"# {fp.name}\n> ❌ 文件读取异常：{str(e)}"
    truncated = False
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
        truncated = True
    filename = fp.name
    md_parts = [f"# {filename}"]
    if truncated:
        md_parts.append(f"\n> ⚠️ 文件内容过长，已截断至前 {max_chars} 字符")

    md_parts.append(f"\n#{lang}文件")
    md_parts.append(full_text)
    if truncated:
        print(f"⚠️ 内容超过限制，执行截断")
    md_content = "\n".join(md_parts)
    return md_content
def generate_excel_markdown(excel_path, max_row_area1=3, max_col_area1=30, max_row_area2=20, max_col_area2=2):
    '''
    处置xlsx文件，截取其中部分信息生成markdown
    '''
    def is_empty(val):
        """判断单元格是否为空"""
        if val is None:
            return True
        s = str(val).strip()
        return s == ""
    try:
        # 只读模式加载excel，data_only=True读取计算后的值
        wb = load_workbook(excel_path, read_only=True, data_only=True)
        md_lines = []
        md_lines.append("# Excel 文件数据概览报告")
        md_lines.append(f"> 源文件：{excel_path}")
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            md_lines.append(f"## Sheet：{sheet_name}")
            # ========== 区域1：1 ~ max_row_area1 行，1 ~ max_col_area1 列 ==========
            md_lines.append(f"### 区域1采样：1~{max_row_area1}行，1~{max_col_area1}列有效数据")
            area1_data = []
            for row in range(1, max_row_area1 + 1):
                for col in range(1, max_col_area1 + 1):
                    cell = ws.cell(row=row, column=col)
                    if not is_empty(cell.value):
                        val = str(cell.value).strip()
                        area1_data.append(f"({row},{col}) = `{val}`")
            if area1_data:
                for item in area1_data:
                    md_lines.append(f"- {item}")
            else:
                md_lines.append("- 无有效数据\n")
            # ========== 区域2：1 ~ max_row_area2 行，1 ~ max_col_area2 列 ==========
            md_lines.append(f"### 区域2采样：{max_row_area1+1}~{max_row_area2}行，1~{max_col_area2}列有效数据")
            area2_data = []
            for row in range(max_row_area1+1 , max_row_area2 + 1):
                for col in range(1, max_col_area2 + 1):
                    cell = ws.cell(row=row, column=col)
                    if not is_empty(cell.value):
                        val = str(cell.value).strip()
                        area2_data.append(f"({row},{col}) = `{val}`")
            if area2_data:
                for item in area2_data:
                    md_lines.append(f"- {item}")
            else:
                md_lines.append("- 无有效数据\n")
            md_lines.append("源文件行数列数都有可能超过采样范围\n")
            md_lines.append("---\n")
        wb.close()
        return "\n".join(md_lines)
    except Exception as e:
        return f"发生了错误：{e}"
def multi_file_read(file_path: str, max_chars=5000):
    '''
    综合获取所有文件的信息
    Args:
        file_path: 件路径
        max_chars: 字符数上线
    return: 文件内容markdown
    '''
    if file_path[-5:] == '.xlsx':
        md = generate_excel_markdown(file_path)
    elif file_path[-4:] == '.csv':
        md = extract_excel_with_pandas(file_path)
    else:
        md = generate_file_md(file_path)
    if len(md) > max_chars:
        md = md[:max_chars] + "\n> ⚠️ 内容超出长度限制，内容已截断"
    return md
if __name__ == "__main__":
    pass
