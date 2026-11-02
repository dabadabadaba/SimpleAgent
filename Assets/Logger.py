import os
import logging
from logging.handlers import RotatingFileHandler
def setup_logger(LOG_FILE, LOG_MAX_SIZE = 5242880 , LOG_BACKUP_COUNT = 5): #LOG_MAX_SIZE默认5M
    """配置日志系统（带大小滚动，防止文件膨胀）"""
    logger = logging.getLogger("AI Agent")
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 确保日志目录存在
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 替换为按大小滚动的文件处理器
    # 当文件达到LOG_MAX_SIZE时，自动切割为 bench_agent.log.1、bench_agent.log.2...
    fh = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=LOG_MAX_SIZE,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)

    # 控制台处理器（调试用）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class UI_dialogger:
    """UI日志记录器，支持普通和临时模式"""

    def __init__(self, file_path: str):
        """
        初始化UI日志记录器

        Args:
            file_path: 日志文件路径
        """
        self.file_path = file_path

    def append_log(self, text: str, mode: str = 'normal'):
        """
        追加日志到文件

        Args:
            text: 要写入的文本内容
            mode: 写入模式，'normal'为普通模式，'temp'为临时模式（使用小字体）
        """
        if mode == 'temp':
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(f"\n<font size='1'>{text}</font>")
        else:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n" + text + "\n")

if __name__ == "__main__":
    # 自测示例
    logger = setup_logger("test.log")
    logger.info("测试日志输出")