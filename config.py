"""
config.py

Tham số cấu hình hệ thống đo tôm trên băng chuyền.

Module này giữ các hằng số bất biến và hàm đọc cấu hình chỉnh được từ
settings.json. Không load giá trị chỉnh được ở module-level để pipeline luôn
nhận cấu hình mới nhất do main.py truyền vào.
"""

from typing import Any

from settings_loader import load_setting


# Section trong settings.json dùng cho cấu hình pipeline.
CONFIG_SECTION = "config"

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

# Số item tối đa trong queue giữa các flow xử lý.
QUEUE_SIZE = 16

# Đường dẫn model phát hiện tôm.
MODEL_DET = "model/yolov8n_det_v6_openvino_model"

# Đường dẫn model segmentation để tách mask tôm.
MODEL_SEG = "model/yolov8n_seg_v74_openvino_model"

# Thiết bị suy luận OpenVINO dùng khi tải model.
DEVICE = "intel:gpu"


def _load_config(key: str, default: Any) -> Any:
    """Đọc một key trong section config bằng settings_loader."""
    return load_setting(key, default, section=CONFIG_SECTION)


def load_config_values() -> dict[str, Any]:
    """
    Đọc toàn bộ cấu hình chỉnh được từ settings.json.

    Hàm trả về dict mới mỗi lần gọi để app.py có thể lấy cấu hình mới nhất.
    Nếu key thiếu hoặc sai kiểu, settings_loader sẽ ghi default vào file và
    trả về default đã ép kiểu theo giá trị mặc định truyền vào.
    """
    return {
        "INPUT_DIR": str(_load_config("INPUT_DIR", "input")),
        "OUTPUT_DIR": str(_load_config("OUTPUT_DIR", "output")),
        "CLEAR_OUTPUT": bool(_load_config("CLEAR_OUTPUT", False)),
        "CLEAR_INPUT": bool(_load_config("CLEAR_INPUT", False)),
        "CHUNK_MODE": bool(_load_config("CHUNK_MODE", False)),
        "SCALE": float(_load_config("SCALE", 1.0)),
        "CONF_DET": float(_load_config("CONF_DET", 0.5)),
        "CONF_SEG": float(_load_config("CONF_SEG", 0.5)),
        "BBOX_PAD": int(_load_config("BBOX_PAD", 5)),
        "TOUCH_THRESHOLD": float(_load_config("TOUCH_THRESHOLD", 10.0)),
        "TARGET_FPS": float(_load_config("TARGET_FPS", 0.0)),
        "CONVEYOR_VERTICAL": bool(_load_config("CONVEYOR_VERTICAL", False)),
        "SAVE": bool(_load_config("SAVE", True)),
        "IMG_EXTS": list(_load_config(
            "IMG_EXTS",
            [".bmp", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"],
        )),
        "VID_EXTS": list(_load_config(
            "VID_EXTS",
            [".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"],
        )),
    }
