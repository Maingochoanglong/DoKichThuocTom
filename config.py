"""
config.py

Toàn bộ tham số cấu hình hệ thống đo tôm trên băng chuyền.

Phân loại hằng số:
    - Có thể chỉnh qua giao diện Flask hoặc settings.json:
        INPUT_DIR, OUTPUT_DIR, CLEAR_OUTPUT, CLEAR_INPUT, CHUNK_MODE,
        SCALE, CONF_DET, CONF_SEG, BBOX_PAD, TOUCH_THRESHOLD,
        TARGET_FPS, CONVEYOR_VERTICAL, SAVE, IMG_EXTS, VID_EXTS.
    - Cứng trong code (không lưu vào settings.json):
        COLOR, QUEUE_SIZE, MODEL_DET, MODEL_SEG, DEVICE.

Nếu key chưa có trong settings.json, load_setting() tự ghi giá trị mặc định
vào file để người dùng có thể chỉnh trực tiếp mà không cần tạo tay.

Quy ước:
    - Đường dẫn tương đối tính từ thư mục gốc dự án.
    - Ngưỡng CONF nằm trong khoảng 0.0 đến 1.0.
"""

from settings_loader import load_setting
from defaults import DEFAULT_CONFIG


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

# Giới hạn queue giữa các flow. 0 = không giới hạn.
QUEUE_SIZE = 64

# Đường dẫn model 
MODEL_DET = "model/yolov8n_det_v6_openvino_model"
MODEL_SEG = "model/yolov8n_seg_v74_openvino_model"

# Thiết bị suy luận phụ thuộc phần cứng máy chủ.
DEVICE = "intel:gpu"

# Thư mục input và output.
INPUT_DIR  = str( load_setting("INPUT_DIR",  DEFAULT_CONFIG["INPUT_DIR"],  section="config"))
OUTPUT_DIR = str( load_setting("OUTPUT_DIR", DEFAULT_CONFIG["OUTPUT_DIR"], section="config"))

# Tùy chọn xóa tự động.
CLEAR_OUTPUT = bool(load_setting("CLEAR_OUTPUT", DEFAULT_CONFIG["CLEAR_OUTPUT"], section="config"))
CLEAR_INPUT  = bool(load_setting("CLEAR_INPUT",  DEFAULT_CONFIG["CLEAR_INPUT"],  section="config"))

# Chế độ xử lý nhiều video liên tiếp như một băng chuyền.
CHUNK_MODE = bool(load_setting("CHUNK_MODE", DEFAULT_CONFIG["CHUNK_MODE"], section="config"))

# Hệ số quy đổi pixel -> mm.
SCALE = float(load_setting("SCALE", DEFAULT_CONFIG["SCALE"], section="config"))

# Tập hợp đuôi file
IMG_EXTS = set(load_setting(
    "IMG_EXTS",
    DEFAULT_CONFIG["IMG_EXTS"],
    section="config",
))
VID_EXTS = set(load_setting(
    "VID_EXTS",
    DEFAULT_CONFIG["VID_EXTS"],
    section="config",
))

# Ngưỡng tin cậy phát hiện và phân đoạn.
CONF_DET = float(load_setting("CONF_DET", DEFAULT_CONFIG["CONF_DET"],  section="config"))
CONF_SEG = float(load_setting("CONF_SEG", DEFAULT_CONFIG["CONF_SEG"],  section="config"))

# Padding bounding box (pixel).
BBOX_PAD = int(load_setting("BBOX_PAD", DEFAULT_CONFIG["BBOX_PAD"], section="config"))

# Ngưỡng khoảng cách tính là chạm vạch (pixel).
TOUCH_THRESHOLD = float(load_setting("TOUCH_THRESHOLD", DEFAULT_CONFIG["TOUCH_THRESHOLD"], section="config"))

# FPS mục tiêu khi lấy mẫu từ video. 0 = lấy tất cả frame.
TARGET_FPS = float(load_setting("TARGET_FPS", DEFAULT_CONFIG["TARGET_FPS"], section="config"))

# Hướng băng chuyền: True = dọc, False = ngang.
CONVEYOR_VERTICAL = bool(load_setting("CONVEYOR_VERTICAL", DEFAULT_CONFIG["CONVEYOR_VERTICAL"], section="config"))

# Lưu ảnh debug F3-F6.
SAVE = bool(load_setting("SAVE", DEFAULT_CONFIG["SAVE"], section="config"))