"""
logger_setup.py

Cấu hình logger dùng chung cho pipeline đo tôm.
main.py gọi setup_logging() sau khi chuẩn bị OUTPUT_DIR để pipeline.log không
bị mất khi CLEAR_OUTPUT được bật. Các flow chỉ cần gọi get_logger().
"""

import logging
import sys
from pathlib import Path


_LOGGER_NAME = "pipeline"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _configure_stdout() -> None:
    """Đưa stdout về UTF-8 để log tiếng Việt không bị lỗi mã hóa."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def setup_logging(output_dir: str | Path) -> logging.Logger:
    """
    Khởi tạo logger pipeline tại output_dir/pipeline.log.

    Logger ghi đồng thời ra stdout và file log, không propagate lên root
    logger để tránh nhân đôi dòng log. Nếu logger đã có handler, hàm trả lại
    instance hiện có để không gắn trùng handler.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.propagate = False
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    _configure_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(output_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger khởi tạo - log file: {log_path}")
    return logger


def get_logger() -> logging.Logger:
    """Lấy logger pipeline đã được setup_logging() cấu hình trong main.py."""
    return logging.getLogger(_LOGGER_NAME)
