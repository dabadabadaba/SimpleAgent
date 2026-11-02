# GeneralTools.py
from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """获取当前日期时间，格式例为 2026年08月26日 16:31:56.123。"""
    now = datetime.now()
    return "当前时间：" + now.strftime("%Y年%m月%d日 %H:%M:%S.") + f"{now.microsecond // 1000:03d}"



if __name__ == "__main__":
    pass
  