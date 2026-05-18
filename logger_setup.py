"""
logger_setup.py
Cấu hình logger dùng chung cho toàn bộ pipeline đo tôm.

Cách dùng trong mỗi module:
    from logger_setup import get_logger
    log = get_logger()

    log.info("Thông báo thường")
    log.warning("Cảnh báo")
    log.error("Lỗi")

Định dạng mỗi dòng log:
    2026-05-18T10:30:00 [INFO] [F1] Đã đọc ảnh: shrimp.jpg

Cấu trúc output:
    output/<timestamp>/pipeline.log   <- file log chính để API đọc
"""

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "pipeline"
_LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(run_dir: str) -> logging.Logger:
    """
    Khởi tạo logger lần đầu khi biết run_dir.
    Gọi một lần duy nhất từ main() trước khi tạo thread.

    Args:
        run_dir: Thư mục output của lần chạy này (vd: "output/2026-05-18_10-30-00").

    Returns:
        Logger đã được cấu hình.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:          # tránh thêm handler trùng khi test
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Handler 1: Console (stdout) - giống print cũ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2: File - pipeline.log ngay tại thư mục gốc của lần chạy
    log_path = Path(run_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger khởi tạo - log file: {log_path}")
    return logger


def get_logger() -> logging.Logger:
    """
    Lấy logger đã được setup (gọi sau setup_logging).
    Dùng trong tất cả các module con: pipeline, draw_utils, v.v.
    """
    return logging.getLogger(_LOGGER_NAME)