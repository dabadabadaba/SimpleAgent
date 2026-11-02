# FileTools.py
import traceback
import re, hashlib, ast
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator


# ================== 配置类 ==================
class FileToolConfig:
    MAX_FILE_SIZE_MB = 10  # 大文件保护阈值
    MAX_READ_LINES = 500  # 默认读取行数
    WORKSPACE_ROOT = None  # 限制访问根目录


# ================== 工具函数 ==================
def _get_absolute_path(file_path: str) -> Path:
    """获取绝对路径，并限制访问范围"""
    path = Path(file_path)
    if not path.is_absolute():
        # 相对于工作区根目录
        if FileToolConfig.WORKSPACE_ROOT:
            path = Path(FileToolConfig.WORKSPACE_ROOT) / path
        else:
            path = path.resolve()

    # 安全检查：防止路径遍历攻击
    if FileToolConfig.WORKSPACE_ROOT:
        root = Path(FileToolConfig.WORKSPACE_ROOT).resolve()
        if not str(path.resolve()).startswith(str(root)):
            return(f"禁止访问工作区外的文件: {path}")

    return path.resolve()

def _check_file_size(file_path: Path) -> bool:
    """检查文件大小是否超过限制"""
    if isinstance(file_path,str):
        return f"❌ check_file_size失败: {file_path}"
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > FileToolConfig.MAX_FILE_SIZE_MB:
            return False
    return True

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()
def _find_all_offsets(text: str, pattern: str) -> List[int]:
    """返回所有精确匹配的起始偏移量"""
    offsets = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets

def _offset_to_line(text: str, offset: int) -> int:
    """偏移量转行号（从1开始）"""
    return text.count('\n', 0, offset) + 1
# ================== 工具1: 读取文件（智能截断） ==================
class ReadFileInput(BaseModel):
    file_path: str = Field(description="文件路径（相对工作区或绝对路径）")
    start_line: Optional[int] = Field(default=1, description="起始行号（从1开始）")
    num_lines: Optional[int] = Field(default=500, description="读取行数，默认500行")
    encoding: str = Field(default="utf-8", description="文件编码")


@tool(args_schema=ReadFileInput)
def read_file_tool(
        file_path: str,
        start_line: int = 1,
        num_lines: int = 500,
        encoding: str = "utf-8"
) -> str:
    """
    读取文件内容（支持大文件分段读取）。

    使用场景：
    - 查看配置文件内容
    - 检查代码文件
    - 读取日志文件
    - 定位特定代码段

    注意：对于大文件，要分批读取。如果文件超过10MB，会自动截断只读取前500行
    """
    abs_path = _get_absolute_path(file_path)
    if isinstance(abs_path,str):
        return f"❌ Read中断: {abs_path}"
    # 检查文件是否存在
    if not abs_path.exists():
        return f"❌ 文件不存在: {abs_path}"

    if abs_path.is_dir():
        return f"❌ 路径是目录，不是文件: {abs_path}"

    # 检查文件大小
    if not _check_file_size(abs_path):
        # 大文件只读取前500行
        num_lines = min(num_lines, FileToolConfig.MAX_READ_LINES)
        size_warning = f"⚠️ 文件较大（>{FileToolConfig.MAX_FILE_SIZE_MB}MB），仅显示前{num_lines}行\n"
    else:
        size_warning = ""

    try:
        with open(abs_path, 'r', encoding=encoding) as f:
            # 跳到指定行
            for _ in range(start_line - 1):
                f.readline()

            # 读取指定行数
            lines = []
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

        content = ''.join(lines)

        # 统计总行数（快速估算）
        total_lines = 0
        with open(abs_path, 'r', encoding=encoding) as f:
            total_lines = sum(1 for _ in f)

        result = f"📄 文件: {abs_path}\n"
        result += f"文件hash: {_file_hash(abs_path)}\n"
        result += f"总行数: {total_lines} | 显示: {start_line}-{start_line + len(lines) - 1} 行\n"
        result += size_warning
        result += content
        if start_line + num_lines -1 > total_lines:
            result += f"\n... (已到达文件末尾（EOF），文件共{total_lines}行，要求读取的行数{start_line}-{start_line + num_lines - 1}已超出范围)"

        return result

    except UnicodeDecodeError:
        return f"❌ 文件编码错误，尝试其他编码: {abs_path}"
    except Exception as e:
        # print(traceback.format_exc())
        return f"❌ 读取文件失败: {str(e)}"


# ================== 工具2: 编辑文件（基于行号替换） ==================
class EditFileInput(BaseModel):
    file_path: str = Field(description="文件路径")
    old_content: str = Field(description="要替换的旧内容片段，必须精确匹配；建议包含足够上下文使其唯一")
    new_content: str = Field(description="替换后的新内容")
    expected_file_hash: Optional[str] = Field(
        default=None,
        description="读取文件时返回的文件哈希，用于防止文件在读取后被修改"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="可选：期望旧内容出现的起始行，仅用于校验/缩小范围"
    )
    end_line: Optional[int] = Field(
        default=None,
        description="可选：期望旧内容出现的结束行"
    )
    replace_all: bool = Field(default=False, description="匹配到多个时是否全部替换")
    dry_run: bool = Field(default=False, description="只预览，不实际写入")
    encoding: str = Field(default="utf-8", description="文件编码")

@tool(args_schema=EditFileInput)
def edit_file_tool(
    file_path: str,
    old_content: str,
    new_content: str,
    expected_file_hash: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    replace_all: bool = False,
    dry_run: bool = False,
    encoding: str = "utf-8"
) -> str:
    """
        编辑文件（基于行号的精准替换）。
        使用场景：
        - 修改配置文件
        - 更新代码片段
        - 修复特定代码段
        - 批量替换文本
        注意：部分修改有可能改变文件其他部分的行数，对于大行修改，因该分批修改。
    """
    abs_path = _get_absolute_path(file_path)
    if isinstance(abs_path, str):
        return f"❌ 编辑中断: {abs_path}"
    if not abs_path.exists():
        return f"❌ 文件不存在: {abs_path}"
    if abs_path.is_dir():
        return f"❌ 路径是目录，不是文件: {abs_path}"
    if not old_content:
        return "❌ old_content 不能为空"

    # 1. 文件版本校验
    if expected_file_hash:
        current_hash = _file_hash(abs_path)
        if current_hash != expected_file_hash:
            return (
                f"❌ 文件已变更，拒绝编辑。\n"
                f"当前哈希: {current_hash}\n"
                f"期望哈希: {expected_file_hash}\n"
                f"请重新读取文件后再操作。"
            )

    # 2. 读取最新文件内容
    try:
        with open(abs_path, 'r', encoding=encoding) as f:
            content = f.read()
    except Exception as e:
        return f"❌ 读取文件失败: {str(e)}"

    # 3. 内容定位
    offsets = _find_all_offsets(content, old_content)
    if not offsets:
        return "❌ 未找到指定的旧内容，文件可能已变化，请重新读取后再试。"

    # 4. 如果提供了行号范围，过滤匹配位置
    if start_line is not None or end_line is not None:
        start_line = start_line or 1
        end_line = end_line or content.count('\n') + 1
        valid_offsets = []
        for off in offsets:
            line = _offset_to_line(content, off)
            if start_line <= line <= end_line:
                valid_offsets.append(off)

        if not valid_offsets:
            return (
                f"❌ 指定的行号范围 {start_line}-{end_line} 内未找到旧内容。"
                f"文件可能已变化，请重新读取。"
            )
        offsets = valid_offsets

    # 5. 唯一性检查
    if len(offsets) > 1 and not replace_all:
        matched_lines = ", ".join(str(_offset_to_line(content, off)) for off in offsets)
        return (
            f"❌ 旧内容匹配到 {len(offsets)} 处（行号: {matched_lines}）。\n"
            f"请提供更长的上下文使其唯一，或设置 replace_all=True。"
        )

    # 6. dry-run 预览
    if dry_run:
        matched_lines = ", ".join(str(_offset_to_line(content, off)) for off in offsets)
        return (
            f"🔍 [dry-run] 将在 {len(offsets)} 处替换（行号: {matched_lines}），"
            f"未实际写入文件。"
        )

    # 7. 备份原文件
    backup_path = abs_path.with_suffix(abs_path.suffix + '.bak')
    try:
        backup_path.write_text(content, encoding=encoding)
    except Exception:
        pass  # 备份失败不阻断编辑，但可考虑返回警告

    # 8. 执行替换
    if len(offsets) == 1:
        new_file_content = (
            content[:offsets[0]] +
            new_content +
            content[offsets[0] + len(old_content):]
        )
    else:
        new_file_content = content.replace(old_content, new_content)

    # 9. 写入文件
    try:
        with open(abs_path, 'w', encoding=encoding) as f:
            f.write(new_file_content)
    except Exception as e:
        return f"❌ 写入文件失败: {str(e)}"

    # 10. 返回结果摘要
    matched_lines = ", ".join(str(_offset_to_line(content, off)) for off in offsets)
    result = f"✅ 文件编辑成功: {abs_path}\n"
    result += f"📝 替换位置: 行号 {matched_lines}，共 {len(offsets)} 处\n"
    result += f"🔄 旧内容:\n{old_content[:200]}"
    if len(old_content) > 200:
        result += "...\n"
    result += f"\n🔄 新内容:\n{new_content[:200]}"
    if len(new_content) > 200:
        result += "...\n"
    result += f"\n💾 原文件已备份至: {backup_path}"
    return result


# ================== 工具3: 文件搜索（grep） ==================
class GrepFileInput(BaseModel):
    pattern: str = Field(description="搜索模式（支持正则表达式）")
    file_path: str = Field(description="文件路径")
    context_lines: int = Field(default=2, description="显示上下文行数")
    case_sensitive: bool = Field(default=True, description="是否区分大小写")


@tool(args_schema=GrepFileInput)
def grep_file_tool(
        pattern: str,
        file_path: str,
        context_lines: int = 2,
        case_sensitive: bool = True
) -> str:
    """
    在文件中搜索文本（类似grep命令）。

    使用场景：
    - 查找函数定义
    - 搜索配置项
    - 定位错误日志
    - 查找特定字符串
    """
    abs_path = _get_absolute_path(file_path)
    if isinstance(abs_path,str):
        return f"❌ grep中断: {abs_path}"
    if not abs_path.exists():
        return f"❌ 文件不存在: {abs_path}"

    if abs_path.is_dir():
        return f"❌ 路径是目录，不是文件: {abs_path}"

    # 大文件保护
    if not _check_file_size(abs_path):
        return f"❌ 文件太大（>{FileToolConfig.MAX_FILE_SIZE_MB}MB），无法搜索"

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern_re = re.compile(pattern, flags)

        matches = []
        with open(abs_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, start=1):
            if pattern_re.search(line):
                # 收集上下文
                start = max(0, i - context_lines - 1)
                end = min(len(lines), i + context_lines)
                context = []
                for j in range(start, end):
                    prefix = ">" if j == i - 1 else " "
                    context.append(f"{j + 1:4d} {prefix} {lines[j].rstrip()}")
                matches.append({
                    "line_num": i,
                    "line": line.rstrip(),
                    "context": "\n".join(context)
                })

        if not matches:
            return f"🔍 未找到匹配项: {pattern}"

        result = f"🔍 搜索 '{pattern}' 在 {abs_path}\n"
        result += f"📊 找到 {len(matches)} 个匹配项：\n"
        for match in matches[:20]:  # 最多显示20个结果
            result += f"\n📍 第 {match['line_num']} 行:\n"
            result += match['context'] + "\n"

        if len(matches) > 20:
            result += f"\n... (还有 {len(matches) - 20} 个匹配项未显示)"

        return result

    except UnicodeDecodeError:
        return f"❌ 文件编码错误: {abs_path}"
    except Exception as e:
        return f"❌ 搜索失败: {str(e)}"


# ================== 工具4: 列出目录 ==================
class ListDirInput(BaseModel):
    dir_path: str = Field(default=".", description="目录路径")
    pattern: Optional[str] = Field(default=None, description="文件名过滤模式")
    max_items: int = Field(default=50, description="最大显示数量")


@tool(args_schema=ListDirInput)
def list_dir_tool(
        dir_path: str = ".",
        pattern: Optional[str] = None,
        max_items: int = 50
) -> str:
    """
    列出目录内容。

    使用场景：
    - 查看项目结构
    - 查找特定文件
    - 了解工作区布局
    """
    abs_path = _get_absolute_path(dir_path)
    if isinstance(abs_path,str):
        return f"❌ list_dir中断: {abs_path}"
    if not abs_path.exists():
        return f"❌ 目录不存在: {abs_path}"

    if not abs_path.is_dir():
        return f"❌ 路径不是目录: {abs_path}"

    try:
        items = []
        for item in abs_path.iterdir():
            # 过滤.log文件
            if item.suffix.lower() in ['.log']:
                continue
            # 过滤以 temp__ 开头的文件夹
            if item.is_dir() and item.name.startswith('temp__'):
                continue
            if pattern and not re.search(pattern, item.name, re.IGNORECASE):
                continue
            is_dir = item.is_dir()
            size = "" if is_dir else f" {item.stat().st_size / 1024:.1f}KB"
            items.append(f"{'📁' if is_dir else '📄'} {item.name}{size}")

        items.sort()
        total = len(items)
        items = items[:max_items]

        result = f"目录: {abs_path}\n"
        result += f"共 {total} 个文件"
        if total > max_items:
            result += f" (显示前 {max_items} 个)"
        result += "\n".join(items)

        if total > max_items:
            result += f"\n... (还有 {total - max_items} 个项目未显示)"

        return result

    except Exception as e:
        return f"❌ 列出目录失败: {str(e)}"


# ================== 工具5: 创建/写入文件 ==================
class WriteFileInput(BaseModel):
    file_path: str = Field(description="文件路径")
    content: str = Field(description="文件内容")
    encoding: str = Field(default="utf-8", description="文件编码")
    append: bool = Field(default=False, description="是否追加模式")


@tool(args_schema=WriteFileInput)
def write_file_tool(
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False
) -> str:
    """
    创建或写入文件。

    使用场景：
    - 创建新配置文件
    - 生成测试用例文件
    - 导出处理结果

    注意：对于大量内容的写入，应分批写入。如果文件存在且不追加，会覆盖原有内容！ 不支持写python脚本用于执行
    """
    abs_path = _get_absolute_path(file_path)

    # 创建目录（如果不存在）
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查文件大小（写入前预估）
    estimated_size = len(content.encode(encoding)) / (1024 * 1024)
    if estimated_size > FileToolConfig.MAX_FILE_SIZE_MB:
        return f"❌ 写入内容太大 ({estimated_size:.1f}MB)，超过限制"

    try:
        mode = 'a' if append else 'w'
        with open(abs_path, mode, encoding=encoding) as f:
            f.write(content)

        action = "追加" if append else "创建/覆写"
        size_kb = len(content.encode(encoding)) / 1024

        result = f"✅ 文件{action}成功: {abs_path}\n"
        result += f"📊 大小: {size_kb:.1f}KB\n"
        result += f"📝 内容预览:\n{content[:200]}"
        if len(content) > 200:
            result += "..."

        return result

    except Exception as e:
        return f"❌ 写入文件失败: {str(e)}"
# ================== 工具5: 创建文件夹 ==================
class CreateFolderInput(BaseModel):
    folder_path: str = Field(description="文件夹路径")

@tool(args_schema=CreateFolderInput)
def create_folder_tool(
        folder_path: str,
) -> str:
    """
    创建文件夹。
    比如创建文件夹放置交付成果或创建“to_delete”文件夹，放置需要删除的文件
    """
    abs_path = 'empty'
    try:
        abs_path = _get_absolute_path(folder_path)
        # 创建目录（如果不存在）
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f'创建成功：{abs_path}'
    except:
        return f'创建失败：{abs_path}'
# ================== 工具6: 读取Excel文件 ==================

class ReadExcelInput(BaseModel):
    file_path: str = Field(description="Excel文件路径（相对工作区或绝对路径）")
    sheet_name: Optional[int | str] = Field(default=None, description="工作表名称，None或0表示读取第一个工作表， 1表示第二个工作表")
    header_row: int = Field(default=0, description="表头所在行号（从0开始）")
    max_rows: int = Field(default=100, description="最大读取行数，防止内存溢出")

@tool(args_schema=ReadExcelInput)
def excel_read_tool(
        file_path: str,
        sheet_name: Optional[int | str] = 0,
        header_row: int = 0,
        max_rows: int = 100
) -> str:
    """
    读取Excel文件内容（支持xlsx/xls/csv）。
    使用场景：
    - 查看数据报表
    - 分析结构化数据
    - 读取配置文件
    注意：大文件会自动截断，只显示前max_rows行
    """
    abs_path = _get_absolute_path(file_path)
    if isinstance(abs_path,str):
        return f"❌ 中断: {abs_path}"
    # 检查文件是否存在
    if not abs_path.exists():
        return f"❌ 文件不存在: {abs_path}"
    if abs_path.is_dir():
        return f"❌ 路径是目录，不是文件: {abs_path}"

    # 检查文件扩展名
    valid_extensions = ['.xlsx', '.xls', '.csv']
    if not any(abs_path.suffix.lower() == ext for ext in valid_extensions):
        return f"❌ 不支持的文件格式: {abs_path.suffix}，仅支持 {valid_extensions}"

    # 检查文件大小
    if not _check_file_size(abs_path):
        return f"❌ 文件太大（>{FileToolConfig.MAX_FILE_SIZE_MB}MB），无法读取"

    try:
        # 读取文件
        if abs_path.suffix.lower() == '.csv':
            # CSV 只有单表，直接返回 DataFrame
            df = pd.read_csv(abs_path, nrows=max_rows, header=header_row)
            sheet_display = "CSV文件(单表)"
        else:
            raw_data = pd.read_excel(abs_path, sheet_name=sheet_name, nrows=max_rows, header=header_row)

            # 判断返回类型：字典 = 多工作表返回格式
            if isinstance(raw_data, dict):
                sheet_names = list(raw_data.keys())
                if len(sheet_names) == 1:
                    # 只有1个工作表，自动提取，兼容 sheet_name=None 的场景
                    df = raw_data[sheet_names[0]]
                    sheet_display = sheet_names[0]
                else:
                    # 多个工作表，提示用户指定
                    return (
                        f"❌ 检测到 {len(sheet_names)} 个工作表，请指定单个工作表\n"
                        f"📋 所有工作表: {sheet_names}\n"
                        f"💡 请将 sheet_name 参数设为具体名称或索引"
                    )
            else:
                # 单工作表直接返回 DataFrame
                df = raw_data
                sheet_display = sheet_name if sheet_name != 0 else "第一个工作表(索引0)"

        # 获取实际读取的行数和列数
        rows, cols = df.shape
        # 生成结果
        result = f"一下是文件总体情况\n"
        result += f"文件: {abs_path}\n"
        result += f"工作表: {sheet_display}\n"
        result += f"数据维度: {rows}行 × {cols}列\n"
        if rows >= max_rows:
            result += f"⚠️ 文件较大，仅显示前{max_rows}行\n"
        # 添加数据预览
        result += df.to_string(max_rows=20, max_cols=30)
        if rows > 20:
            result += f"\n... (还有 {rows - 20} 行数据未显示)"
        if cols > 30:
            result += f"\n... (还有 {cols - 30} 列数据未显示)"
        return result
    except Exception as e:
        return f"❌ 读取文件失败: {str(e)}"
class ReadExcelRowsColsInput(BaseModel):
    file_path: str = Field(description="Excel文件路径（相对工作区或绝对路径）")
    sheet_name: Optional[int | str] = Field(default=None, description="工作表名称，None或0表示读取第一个工作表，1表示第二个工作表")
    rows: Optional[list[int]] = Field(default=None, description="需要读取的行号列表（从0开始，不含表头行），一次读取不要超过10行, 只支持list，不支持string")
    cols: Optional[list[int | str]] = Field(default=None, description="需要读取的列，支持列索引(int)或列名(string)的列表，，一次读取不要超过10列")
    @validator('rows', 'cols', pre=True, always=True)
    def parse_str_list(cls, v):
        # 已经是 None 或 list 直接放行
        if v is None or isinstance(v, list):
            return v
        # 字符串格式则尝试解析为列表
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v.strip())
                if not isinstance(parsed, list):
                    raise ValueError(f"解析结果不是列表类型: {type(parsed)}")
                return parsed
            except Exception as e:
                raise ValueError(f"无法解析列表字符串: {v}，错误: {str(e)}")
        # 其他不支持类型直接抛出
        raise ValueError(f"不支持的参数类型: {type(v)}，仅支持列表或列表格式的字符串")

@tool(args_schema=ReadExcelRowsColsInput)
def excel_read_rows_cols_tool(
        file_path: str,
        sheet_name: Optional[int | str] = 0,
        rows: Optional[list[int]] = None,
        cols: Optional[list[int | str]] = None
) -> str:
    """
    读取Excel文件中指定行和指定列的数据。
    使用场景：
    - 精确提取特定行的数据
    - 只关注特定列的数据
    - 同时指定行和列进行数据筛选
    注意：返回结果是经过总结的关键信息
    """
    abs_path = _get_absolute_path(file_path)
    if isinstance(abs_path, str):
        return f"❌ 中断: {abs_path}"

    # 检查文件是否存在
    if not abs_path.exists():
        return f"❌ 文件不存在: {abs_path}"
    if abs_path.is_dir():
        return f"❌ 路径是目录，不是文件: {abs_path}"

    # 检查文件扩展名
    valid_extensions = ['.xlsx', '.xls', '.csv']
    if not any(abs_path.suffix.lower() == ext for ext in valid_extensions):
        return f"❌ 不支持的文件格式: {abs_path.suffix}，仅支持 {valid_extensions}"

    # 检查文件大小
    if not _check_file_size(abs_path):
        return f"❌ 文件太大（>{FileToolConfig.MAX_FILE_SIZE_MB}MB），无法读取"

    # 参数校验：rows 和 cols 不能同时为 None
    if rows is None and cols is None:
        return "❌ rows 和 cols 不能同时为空，请至少指定一个要筛选的维度"

    try:
        # 读取文件
        if abs_path.suffix.lower() == '.csv':
            df = pd.read_csv(abs_path)
            sheet_display = "CSV文件(单表)"
        else:
            raw_data = pd.read_excel(abs_path, sheet_name=sheet_name, header=None)

            if isinstance(raw_data, dict):
                sheet_names = list(raw_data.keys())
                if len(sheet_names) == 1:
                    df = raw_data[sheet_names[0]]
                    sheet_display = sheet_names[0]
                else:
                    return (
                        f"❌ 检测到 {len(sheet_names)} 个工作表，请指定单个工作表\n"
                        f"📋 所有工作表: {sheet_names}\n"
                        f"💡 请将 sheet_name 参数设为具体名称或索引"
                    )
            else:
                df = raw_data
                sheet_display = sheet_name if sheet_name != 0 else "第一个工作表(索引0)"

        total_rows, total_cols = df.shape
        original_shape = f"{total_rows}行 × {total_cols}列"

        # 筛选列
        if cols is not None:
            col_selectors = []
            for c in cols:
                if isinstance(c, int):
                    if c < 0 or c >= total_cols:
                        return f"❌ 列索引 {c} 超出范围，有效范围: 0 ~ {total_cols - 1}"
                    col_selectors.append(c)
                elif isinstance(c, str):
                    if c not in df.columns:
                        return f"❌ 列名 '{c}' 不存在，可用列名: {list(df.columns)}"
                    col_selectors.append(c)
                else:
                    return f"❌ 列参数类型错误: {c}，仅支持 int(列索引) 或 str(列名)"
            df = df.iloc[:, col_selectors] if all(isinstance(c, int) for c in col_selectors) else df[col_selectors]

        # 筛选行
        if rows is not None:
            for r in rows:
                if r < 0 or r >= total_rows:
                    return f"❌ 行索引 {r} 超出范围，有效范围: 0 ~ {total_rows - 1}"
            df = df.iloc[rows]

        # 获取筛选后的维度
        result_rows, result_cols = df.shape

        # 生成结果（不缩略，完整展示）
        result = f"📊 文件: {abs_path}\n"
        result += f"📑 工作表: {sheet_display}\n"
        result += f"📏 原始数据维度: {original_shape}\n"
        result += f"🔍 筛选后数据维度: {result_rows}行 × {result_cols}列\n"

        # 显示筛选条件
        filter_info = []
        if rows is not None:
            filter_info.append(f"指定行索引: {rows}")
        if cols is not None:
            filter_info.append(f"指定列: {cols}")
        result += f"🎯 筛选条件: {' | '.join(filter_info)}\n"
        excel_map = result
        # 完整输出所有数据，不做任何截断
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        excel_map += df.to_string(index=True)

        return excel_map
    except Exception as e:
        return f"❌ 读取文件失败: {str(e)}"
if __name__ == "__main__":
    # 自测示例（替换为本地实际路径后运行）
    pass