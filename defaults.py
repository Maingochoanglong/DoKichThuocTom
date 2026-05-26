"""
defaults.py

Giá trị mặc định dùng chung cho cả backend và frontend.
Đồng bộ một nguồn truth duy nhất.
"""

DEFAULT_CONFIG = {
    "INPUT_DIR":         "input",
    "OUTPUT_DIR":        "output",
    "CLEAR_OUTPUT":      False,
    "CLEAR_INPUT":       False,
    "CHUNK_MODE":        False,
    "SCALE":             1.0,
    "CONF_DET":          0.5,
    "CONF_SEG":          0.5,
    "BBOX_PAD":          5,
    "TOUCH_THRESHOLD":   10.0,
    "TARGET_FPS":        0.0,
    "CONVEYOR_VERTICAL": False,
    "SAVE":              True,
    "IMG_EXTS": [".bmp", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"],
    "VID_EXTS": [".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"],
}

DEFAULT_SIZES = {
    "SIZE_RANGES": {
        "S": [12, 14],
        "M": [14, 17],
        "L": [18, 22],
    },
    "UNDERSIZE_LABEL": "Ngoại cỡ nhỏ",
    "OVERSIZE_LABEL":  "Ngoại cỡ lớn",
    "FALLBACK_LABEL":  "Ngoại cỡ",
}
