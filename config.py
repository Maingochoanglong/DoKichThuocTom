"""
config.py

Tham số cấu hình hệ thống đo tôm trên băng chuyền.
Các giá trị có thể chỉnh được đọc từ settings.json qua settings_loader.
"""

from settings_loader import load_setting


# Bảng 12 màu BGR dùng trực tiếp với OpenCV.
COLOR = [
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 128,   0),
    (128,   0, 255),
    (  0, 128, 255),
    (128, 255,   0),
    (255,   0, 128),
    (  0, 255, 128),
]

# Giới hạn queue giữa các flow. 0 là không giới hạn.
QUEUE_SIZE = 64

# Đường dẫn model.
MODEL_DET = "model/yolov8n_det_v6_openvino_model"
MODEL_SEG = "model/yolov8n_seg_v74_openvino_model"

# Thiết bị suy luận phụ thuộc phần cứng máy chủ.
DEVICE = "intel:gpu"


def load_config_values() -> dict:
    """Đọc toàn bộ config từ settings.json và tự ghi default nếu thiếu."""
    return {
        "INPUT_DIR": str(load_setting("INPUT_DIR", "input", section="config")),
        "OUTPUT_DIR": str(load_setting("OUTPUT_DIR", "output", section="config")),
        "CLEAR_OUTPUT": bool(load_setting("CLEAR_OUTPUT", False, section="config")),
        "CLEAR_INPUT": bool(load_setting("CLEAR_INPUT", False, section="config")),
        "CHUNK_MODE": bool(load_setting("CHUNK_MODE", False, section="config")),
        "SCALE": float(load_setting("SCALE", 1.0, section="config")),
        "CONF_DET": float(load_setting("CONF_DET", 0.5, section="config")),
        "CONF_SEG": float(load_setting("CONF_SEG", 0.5, section="config")),
        "BBOX_PAD": int(load_setting("BBOX_PAD", 5, section="config")),
        "TOUCH_THRESHOLD": float(load_setting("TOUCH_THRESHOLD", 10.0, section="config")),
        "TARGET_FPS": float(load_setting("TARGET_FPS", 0.0, section="config")),
        "CONVEYOR_VERTICAL": bool(load_setting("CONVEYOR_VERTICAL", False, section="config")),
        "SAVE": bool(load_setting("SAVE", True, section="config")),
        "IMG_EXTS": list(load_setting(
            "IMG_EXTS",
            [".bmp", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"],
            section="config",
        )),
        "VID_EXTS": list(load_setting(
            "VID_EXTS",
            [".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"],
            section="config",
        )),
    }


_CONFIG_VALUES = load_config_values()

INPUT_DIR = _CONFIG_VALUES["INPUT_DIR"]
OUTPUT_DIR = _CONFIG_VALUES["OUTPUT_DIR"]
CLEAR_OUTPUT = _CONFIG_VALUES["CLEAR_OUTPUT"]
CLEAR_INPUT = _CONFIG_VALUES["CLEAR_INPUT"]
CHUNK_MODE = _CONFIG_VALUES["CHUNK_MODE"]
SCALE = _CONFIG_VALUES["SCALE"]
CONF_DET = _CONFIG_VALUES["CONF_DET"]
CONF_SEG = _CONFIG_VALUES["CONF_SEG"]
BBOX_PAD = _CONFIG_VALUES["BBOX_PAD"]
TOUCH_THRESHOLD = _CONFIG_VALUES["TOUCH_THRESHOLD"]
TARGET_FPS = _CONFIG_VALUES["TARGET_FPS"]
CONVEYOR_VERTICAL = _CONFIG_VALUES["CONVEYOR_VERTICAL"]
SAVE = _CONFIG_VALUES["SAVE"]
IMG_EXTS = set(_CONFIG_VALUES["IMG_EXTS"])
VID_EXTS = set(_CONFIG_VALUES["VID_EXTS"])
