"""
logger_setup.py

Cấu hình logger dùng chung cho pipeline đo tôm.
"""

import logging
import sys
from pathlib import Path


_LOGGER_NAME = "pipeline"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def setup_logging(output_dir: str | Path) -> logging.Logger:
    """Khởi tạo logger pipeline tại OUTPUT_DIR/pipeline.log."""
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
    """Lấy logger pipeline đã được cấu hình bởi main.py."""
    return logging.getLogger(_LOGGER_NAME)