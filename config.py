# config.py
# Toàn bộ tham số cấu hình hệ thống đo tôm trên băng chuyền.
# Chỉnh sửa file này để thay đổi hành vi pipeline mà không cần sửa code.


# Bảng màu BGR dùng trực tiếp với OpenCV.
COLOR = [
    (255,   0,   0), # Xanh dương
    (  0, 255,   0), # Xanh lá
    (  0,   0, 255), # Đỏ
    (255, 255,   0), # Cyan
    (255,   0, 255), # Tím hồng
    (  0, 255, 255), # Vàng
    (255, 128,   0), # Cam
    (128,   0, 255), # Tím
    (  0, 128, 255), # Xanh da trời
    (128, 255,   0), # Xanh lá sáng
    (255,   0, 128), # Hồng đậm
    (  0, 255, 128), # Xanh ngọc
]


INPUT_DIR  = "input"
OUTPUT_DIR = "output"

# Xóa dữ liệu cũ trước mỗi lần chạy
# CLEAR_OUTPUT : True → xóa toàn bộ thư mục output trước khi bắt đầu pipeline.
# CLEAR_INPUT  : True → xóa file input sau khi F6 ghi JSON thành công.
CLEAR_OUTPUT = False
CLEAR_INPUT  = True

# True  = các file trong input/ là chunk liên tiếp của 1 băng chuyền
# False = mỗi file là 1 video độc lập (hành vi cũ)
CHUNK_MODE = False

MODEL_DET = "model/yolov8n_shrimp_v46_openvino_model"
MODEL_SEG = "model/yolov8n-seg_shrimp_v54_openvino_model"

SCALE = 1      # hệ số quy đổi pixel → mm — cập nhật sau khi hiệu chuẩn camera
               # Ví dụ: SCALE = 0.35 nghĩa là 1 pixel = 0.35 mm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

CONF_DET = 0.5
CONF_SEG = 0.5
BBOX_PAD = 10

REQUIRED_TOUCHES = 3
TOUCH_THRESHOLD  = 10

TARGET_FPS = 0

CONVEYOR_VERTICAL = False


# Lưu ảnh debug & link ảnh trong JSON 
# True : lưu toàn bộ ảnh debug (F3–F6) và ghi đường dẫn vào JSON output.
# False: chỉ xuất JSON kết quả số (track_id, pixel_length, real_length, size),
#        không lưu ảnh debug.
SAVE = True