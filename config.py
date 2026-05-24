# config.py
# Toàn bộ tham số cấu hình hệ thống đo tôm trên băng chuyền.
# Có thể chỉnh trực tiếp hoặc lưu từ giao diện Flask.
# Quy ước:
# - Các đường dẫn tương đối được tính từ thư mục gốc dự án.
# - Các ngưỡng CONF nằm trong khoảng 0.0 đến 1.0.
# - Các khoảng cách/chiều dài ghi bằng pixel hoặc mm sẽ được chú thích riêng.

from settings_loader import load_setting


# Bảng màu BGR dùng trực tiếp với OpenCV để vẽ track, mask, skeleton và ảnh debug.
# OpenCV dùng thứ tự BGR, không phải RGB.
COLOR = [
    (255, 0, 0),  # Xanh dương
    (0, 255, 0),  # Xanh lá
    (0, 0, 255),  # Đỏ
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Tím hồng
    (0, 255, 255),  # Vàng
    (255, 128, 0),  # Cam
    (128, 0, 255),  # Tím
    (0, 128, 255),  # Xanh da trời
    (128, 255, 0),  # Xanh lá sáng
    (255, 0, 128),  # Hồng đậm
    (0, 255, 128),  # Xanh ngọc
]

# Kích thước hàng đợi giữa các flow
QUEUE_SIZE = 32

# Thư mục chứa ảnh/video đầu vào. Upload từ giao diện web sẽ lưu vào đây.
INPUT_DIR = str(load_setting("INPUT_DIR", 'input'))

# Thư mục chứa kết quả chạy pipeline: pipeline.log, JSON kết quả, ảnh debug và thư mục run.
OUTPUT_DIR = str(load_setting("OUTPUT_DIR", 'output'))

# True: xóa toàn bộ dữ liệu cũ trong OUTPUT_DIR trước mỗi lần chạy pipeline.
# False: giữ lại các run cũ để xem lại, so sánh hoặc xuất CSV.
CLEAR_OUTPUT = bool(load_setting("CLEAR_OUTPUT", False))

# True: xóa file trong INPUT_DIR sau khi pipeline ghi JSON thành công.
# False: giữ file input để kiểm tra lại hoặc chạy lại pipeline.
CLEAR_INPUT = bool(load_setting("CLEAR_INPUT", False))

# True: coi các file video trong INPUT_DIR là các chunk liên tiếp của cùng một băng chuyền.
# False: xử lý mỗi ảnh/video như một nguồn độc lập.
CHUNK_MODE = bool(load_setting("CHUNK_MODE", False))

# Đường dẫn model phát hiện tôm. Có thể là file .pt hoặc thư mục OpenVINO model.
MODEL_DET = 'model/yolov8_det_v65_openvino_model'


# Đường dẫn model phân đoạn tôm để tạo mask thân tôm. Có thể là file .pt hoặc thư mục OpenVINO model.
# MODEL_SEG = 'model/yolov8n-seg_shrimp_openvino_model'
MODEL_SEG = 'model/yolov8_seg_v65_openvino_model'

# Hệ số quy đổi pixel -> mm.
# Ví dụ SCALE = 0.35 nghĩa là 1 pixel tương ứng 0.35 mm.
# Có thể cập nhật từ chức năng "Tính scale" trên giao diện.
SCALE = float(load_setting("SCALE", 1.0))

# Các định dạng ảnh được phép nạp vào INPUT_DIR hoặc upload qua giao diện.
IMG_EXTS = {'.bmp', '.heic', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

# Các định dạng video được phép nạp vào INPUT_DIR hoặc upload qua giao diện.
VID_EXTS = {'.avi', '.flv', '.m4v', '.mkv', '.mov', '.mp4', '.webm', '.wmv'}

# Ngưỡng tin cậy cho bước phát hiện tôm.
# Tăng giá trị để lọc chặt hơn, giảm giá trị để bắt nhiều đối tượng hơn.
CONF_DET = float(load_setting("CONF_DET", 0.5))

# Ngưỡng tin cậy cho bước phân đoạn mask thân tôm.
# Tăng giá trị để mask chắc hơn, giảm giá trị để mask rộng/nhạy hơn.
CONF_SEG = float(load_setting("CONF_SEG", 0.5))

DEVICE = 'intel:gpu'

# Số pixel nới rộng quanh bounding box trước khi cắt vùng tôm để xử lý tiếp.
# Giá trị lớn giúp tránh cắt cụt đầu/đuôi, nhưng quá lớn có thể kéo thêm nhiễu nền.
BBOX_PAD = int(load_setting("BBOX_PAD", 5))

# Số lần chạm vạch tham chiếu tối thiểu trước khi một track được xem là đủ điều kiện đo.
REQUIRED_TOUCHES = 3

# Sai số pixel khi kiểm tra tôm có chạm vạch tham chiếu hay không.
TOUCH_THRESHOLD = float(load_setting("TOUCH_THRESHOLD", 10.0))

# FPS mục tiêu khi lấy mẫu video.
# 0 nghĩa là xử lý toàn bộ frame; giá trị > 0 sẽ lấy mẫu theo FPS này để giảm tải.
TARGET_FPS = float(load_setting("TARGET_FPS", 0.0))

# True: băng chuyền chạy theo chiều dọc khung hình.
# False: băng chuyền chạy theo chiều ngang khung hình.
CONVEYOR_VERTICAL = bool(load_setting("CONVEYOR_VERTICAL", False))

# True: lưu ảnh debug F3-F6 và ghi đường dẫn ảnh vào JSON output.
# False: chỉ xuất JSON kết quả số, tiết kiệm dung lượng và thời gian ghi file.
SAVE = bool(load_setting("SAVE", True))
