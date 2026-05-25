"""
config.py

Toàn bộ tham số cấu hình hệ thống đo tôm trên băng chuyền.

Phân loại hằng số:
    - Có thể chỉnh qua giao diện Flask hoặc settings.json:
        INPUT_DIR, OUTPUT_DIR, CLEAR_OUTPUT, CLEAR_INPUT, CHUNK_MODE,
        SCALE, CONF_DET, CONF_SEG, BBOX_PAD, TOUCH_THRESHOLD,
        TARGET_FPS, CONVEYOR_VERTICAL, SAVE.
    - Cứng trong code (không lưu vào settings.json):
        COLOR, QUEUE_SIZE, MODEL_DET, MODEL_SEG, IMG_EXTS, VID_EXTS,
        DEVICE, REQUIRED_TOUCHES.

Nếu key chưa có trong settings.json, load_setting() tự ghi giá trị mặc định
vào file để người dùng có thể chỉnh trực tiếp mà không cần tạo tay.

Quy ước:
    - Đường dẫn tương đối tính từ thư mục gốc dự án.
    - Ngưỡng CONF nằm trong khoảng 0.0 đến 1.0.
    - Khoảng cách/chiều dài ghi bằng pixel hoặc mm sẽ được chú thích riêng.
"""

from settings_loader import load_setting


# Bảng 12 màu BGR dùng trực tiếp với OpenCV để phân biệt các track khi vẽ
# bounding box, mask overlay, skeleton và nhãn kết quả trên ảnh debug.
# OpenCV dùng thứ tự BGR (Blue-Green-Red), không phải RGB.
# Track ID 1 lấy màu index 0, track ID 2 lấy index 1, cuộn vòng sau 12 track.
COLOR = [
    (255,   0,   0),   # Xanh dương
    (  0, 255,   0),   # Xanh lá
    (  0,   0, 255),   # Đỏ
    (255, 255,   0),   # Cyan
    (255,   0, 255),   # Tím hồng
    (  0, 255, 255),   # Vàng
    (255, 128,   0),   # Cam
    (128,   0, 255),   # Tím
    (  0, 128, 255),   # Xanh da trời
    (128, 255,   0),   # Xanh lá sáng
    (255,   0, 128),   # Hồng đậm
    (  0, 255, 128),   # Xanh ngọc
]

# Giới hạn số phần tử tối đa của mỗi hàng đợi giữa các flow.
# 0 = không giới hạn (Python Queue mặc định), toàn bộ frame được đệm vào RAM.
# Đặt giá trị dương (ví dụ 32) để giảm RAM khi xử lý video rất dài,
# nhưng sẽ làm F1 bị block khi các flow sau chưa xử lý kịp.
QUEUE_SIZE = 0

# Thư mục chứa ảnh/video đầu vào. Upload từ giao diện web sẽ lưu vào đây.
# Đường dẫn tương đối được tính từ thư mục gốc dự án (BASE_DIR).
# Ví dụ: "input" -> BASE_DIR/input | "C:\Data\tom" -> dùng thẳng đường dẫn đó.
INPUT_DIR = str(load_setting("INPUT_DIR", "input", section="config"))

# Thư mục chứa kết quả chạy pipeline: pipeline.log, JSON kết quả, ảnh debug và thư mục run.
# Đường dẫn tương đối được tính từ thư mục gốc dự án (BASE_DIR).
# Ví dụ: "output" -> BASE_DIR/output | "D:\Results\tom" -> dùng thẳng đường dẫn đó.
OUTPUT_DIR = str(load_setting("OUTPUT_DIR", "output", section="config"))

# True: xóa toàn bộ dữ liệu cũ trong OUTPUT_DIR trước mỗi lần chạy pipeline.
# False: giữ lại các run cũ để xem lại, so sánh hoặc xuất CSV.
CLEAR_OUTPUT = bool(load_setting("CLEAR_OUTPUT", False, section="config"))

# True: xóa file trong INPUT_DIR sau khi pipeline ghi JSON thành công.
# False: giữ file input để kiểm tra lại hoặc chạy lại pipeline.
CLEAR_INPUT = bool(load_setting("CLEAR_INPUT", False, section="config"))

# True: coi các file video trong INPUT_DIR là các chunk liên tiếp của cùng một băng chuyền.
#       ByteTrack và danh sách track đã hoàn thành được giữ lại giữa các video.
# False: xử lý mỗi ảnh/video như một nguồn độc lập, reset tracker khi sang file mới.
CHUNK_MODE = bool(load_setting("CHUNK_MODE", False, section="config"))

# Đường dẫn model phát hiện tôm (YOLO detect).
# Có thể là file .pt (PyTorch) hoặc thư mục OpenVINO IR model (.xml + .bin).
# Hardcode vì người dùng không được chọn model qua giao diện.
# app.py chỉ đọc giá trị này để giữ nguyên khi validate config, không ghi vào settings.json.
# Muốn đổi model thì sửa trực tiếp trong file này.
MODEL_DET = "model/yolov8n_det_v6_openvino_model"

# Đường dẫn model phân đoạn tôm (YOLO segment) để tạo mask pixel-level thân tôm.
# Có thể là file .pt (PyTorch) hoặc thư mục OpenVINO IR model (.xml + .bin).
# Hardcode vì người dùng không được chọn model qua giao diện.
# app.py chỉ đọc giá trị này để giữ nguyên khi validate config, không ghi vào settings.json.
# Muốn đổi model thì sửa trực tiếp trong file này.
MODEL_SEG = "model/yolov8n_seg_v74_openvino_model"

# Hệ số quy đổi pixel -> mm: real_length_mm = pixel_length * SCALE.
# Ví dụ SCALE = 0.35 nghĩa là 1 pixel tương ứng 0.35 mm.
# Nên hiệu chuẩn lại mỗi khi thay đổi camera, độ phân giải hoặc khoảng cách lắp đặt.
# Có thể cập nhật tự động từ chức năng "Tính scale" trên giao diện.
SCALE = float(load_setting("SCALE", 1.0, section="config"))

# Tập hợp đuôi file ảnh (chữ thường) được phép nạp vào INPUT_DIR hoặc upload qua giao diện.
IMG_EXTS = {".bmp", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# Tập hợp đuôi file video (chữ thường) được phép nạp vào INPUT_DIR hoặc upload qua giao diện.
VID_EXTS = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}

# Ngưỡng tin cậy tối thiểu (0.0 – 1.0) để nhận một vật là tôm ở bước detect.
# Tăng giá trị để lọc chặt hơn (ít nhận nhầm), giảm để bắt nhiều đối tượng hơn (ít bỏ sót).
CONF_DET = float(load_setting("CONF_DET", 0.5, section="config"))

# Ngưỡng tin cậy tối thiểu (0.0 – 1.0) để tách thân tôm khỏi nền ở bước segment.
# Tăng giá trị để mask chắc/sạch hơn, giảm để mask rộng/nhạy hơn.
CONF_SEG = float(load_setting("CONF_SEG", 0.5, section="config"))

# Chỉ định thiết bị cho suy luận (ví dụ: "cpu", "cuda:0", "0", "npu" hoặc "npu:0").
# Cho phép người dùng chọn giữa CPU, GPU cụ thể, NPU Huawei Ascend hoặc các thiết bị
# tính toán khác để thực thi mô hình.
# Giá trị hiện tại "intel:gpu" theo định dạng OpenVINO (vendor:device),
# dùng khi model được xuất sang OpenVINO IR (.xml + .bin).
# Không lưu vào settings.json vì phụ thuộc phần cứng máy chủ.
DEVICE = "intel:gpu"

# Số pixel nới rộng mỗi cạnh của bounding box trước khi cắt vùng tôm để xử lý tiếp.
# Giá trị lớn giúp tránh cắt cụt đầu/đuôi tôm khi box khít,
# nhưng quá lớn có thể kéo thêm nhiễu nền vào vùng segment.
BBOX_PAD = int(load_setting("BBOX_PAD", 5, section="config"))

# Số vạch tham chiếu tối thiểu mà tôm phải chạm qua để được xem là đủ điều kiện đo.
# Phải bằng với số vạch mà get_lines() tạo ra (mặc định 3 vạch: trái, giữa, phải).
# Không lưu vào settings.json vì phải đồng bộ với logic get_lines().
REQUIRED_TOUCHES = 3

# Sai số pixel cho phép khi kiểm tra tâm bounding box có chạm vạch tham chiếu không.
# Tăng giá trị để dễ được tính là chạm hơn (ít bỏ sót track),
# giảm để chỉ tính chạm khi tâm rất gần vạch (chính xác hơn nhưng có thể bỏ sót).
TOUCH_THRESHOLD = float(load_setting("TOUCH_THRESHOLD", 10.0, section="config"))

# FPS mục tiêu khi lấy mẫu frame từ video (stride = round(video_fps / TARGET_FPS)).
# 0.0 = xử lý toàn bộ frame (stride = 1), phù hợp khi video đã có FPS thấp sẵn.
# Giá trị dương giúp giảm tải CPU/GPU khi video có FPS cao mà tôm di chuyển chậm.
TARGET_FPS = float(load_setting("TARGET_FPS", 0.0, section="config"))

# True: băng chuyền chạy theo chiều dọc (tôm vào từ trên/dưới khung hình).
#       Vạch tham chiếu là các đường ngang, tọa độ kiểm tra là cy của bounding box.
# False: băng chuyền chạy theo chiều ngang (tôm vào từ trái/phải khung hình).
#        Vạch tham chiếu là các đường dọc, tọa độ kiểm tra là cx của bounding box.
CONVEYOR_VERTICAL = bool(load_setting("CONVEYOR_VERTICAL", False, section="config"))

# True: lưu ảnh debug F3–F6 ra đĩa và ghi đường dẫn ảnh vào JSON output.
#       Tiêu tốn thêm dung lượng và thời gian ghi file, hữu ích khi cần kiểm tra.
# False: chỉ xuất JSON kết quả số (track_id, pixel_length, real_length_mm, size),
#        tiết kiệm tài nguyên khi chạy production liên tục.
SAVE = bool(load_setting("SAVE", True, section="config"))