"""
app.py

Máy chủ Flask cho giao diện web hệ thống đo chiều dài tôm tự động.

Các nhóm chức năng chính:
  - Quản lý file input  : upload, liệt kê, xoá file ảnh/video.
  - Điều khiển pipeline : khởi động, dừng, theo dõi trạng thái và log.
  - Cấu hình hệ thống  : đọc/ghi config (SCALE, CONF, BBOX_PAD, ...).
  - Phân loại kích cỡ  : đọc/ghi bảng SIZE_RANGES, nhãn ngoại cỡ.
  - Kết quả đo         : liệt kê theo run, lọc theo kích cỡ, xuất CSV/Excel.
  - Hiệu chuẩn scale   : nạp file CSV/Excel đo thực tế, tính SCALE bằng
                         hồi quy bình phương tối thiểu qua gốc toạ độ.
  - Phục vụ ảnh debug  : trả file ảnh F3–F6 dưới đường dẫn /outputs/<path>.
"""

import csv
import importlib.util
import io
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from settings_loader import clear_settings_errors, get_settings_errors, save_settings


# ===========================================================================
# Hằng số và cấu hình ứng dụng
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
SIZE_PATH = BASE_DIR / "size.py"
MAIN_PATH = BASE_DIR / "main.py"

# Danh sách khoá config người dùng có thể chỉnh qua API.
CONFIG_KEYS = [
    "INPUT_DIR",
    "OUTPUT_DIR",
    "CLEAR_OUTPUT",
    "CLEAR_INPUT",
    "CHUNK_MODE",
    "SCALE",
    "CONF_DET",
    "CONF_SEG",
    "BBOX_PAD",
    "TOUCH_THRESHOLD",
    "TARGET_FPS",
    "CONVEYOR_VERTICAL",
    "SAVE",
]

# Khoá nội bộ (đường dẫn model) – không cho phép sửa qua giao diện.
INTERNAL_CONFIG_KEYS = ["MODEL_DET", "MODEL_SEG"]

# Khoá nhận giá trị bool.
BOOL_CONFIG_KEYS = ["CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE", "CONVEYOR_VERTICAL", "SAVE"]

# Tiêu đề cột khi xuất kết quả ra CSV/Excel.
RESULT_EXPORT_HEADERS = ["run", "source_file", "track_id", "frame_idx", "pixel_length", "real_length_mm", "size"]

# Giới hạn kích thước file scale import (8 MB).
SCALE_IMPORT_MAX_BYTES = 8 * 1024 * 1024

# Tập hợp tên cột (sau khi chuẩn hoá) chứa giá trị mm thực tế.
SCALE_MM_COLUMNS = {"mm", "real_mm", "real_length_mm", "length_mm", "ground_truth_mm", "actual_mm", "do_dai_mm"}

# Tập hợp tên cột chứa source_stem (tên nguồn không có đuôi file).
SCALE_SOURCE_STEM_COLUMNS = {"source_stem", "stem", "ten_nguon", "nguon"}

# Tập hợp tên cột chứa source_file (tên file đầy đủ).
SCALE_SOURCE_FILE_COLUMNS = {"source_file", "file", "filename", "ten_file"}

# Tập hợp tên cột chứa track_id.
SCALE_TRACK_COLUMNS = {"track_id", "id", "shrimp_id", "tom_id"}


# ===========================================================================
# Khởi tạo Flask
# ===========================================================================

mimetypes.add_type("text/css; charset=utf-8", ".css")
mimetypes.add_type("application/javascript; charset=utf-8", ".js")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
app.json.ensure_ascii = False


# ===========================================================================
# Lọc log Werkzeug – tắt spam từ endpoint /api/pipeline/status
# ===========================================================================

class _StatusLogFilter(logging.Filter):
    """Lọc bỏ dòng log Werkzeug sinh ra do polling /api/pipeline/status."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/pipeline/status" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_StatusLogFilter())


# ===========================================================================
# Trạng thái pipeline toàn cục (được bảo vệ bởi _pipeline_lock)
# ===========================================================================

_pipeline_lock = threading.Lock()
_pipeline_process: subprocess.Popen | None = None
_pipeline_started_at: float | None = None
_pipeline_ended_at: float | None = None
_pipeline_returncode: int | None = None


# ===========================================================================
# Tiện ích nội bộ – tải module động
# ===========================================================================

def _load_module(path: Path, name: str):
    """
    Tải động một module Python từ đường dẫn tuyệt đối.

    Mỗi lần gọi tạo ra một tên module duy nhất (dùng timestamp nanosecond)
    để tránh Python cache lại module cũ sau khi file thay đổi.

    Tham số:
        path  : Đường dẫn tuyệt đối đến file .py cần tải.
        name  : Tiền tố đặt tên module nội bộ.

    Trả về:
        Module đã được thực thi (exec_module).

    Ngoại lệ:
        RuntimeError : Khi importlib không tạo được spec hoặc loader.
    """
    spec = importlib.util.spec_from_file_location(f"_web_{name}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể tải module từ {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config():
    """
    Tải lại module config.py và xoá cache lỗi settings trước đó.

    Trả về:
        Module config đã được nạp mới nhất.
    """
    clear_settings_errors()
    return _load_module(CONFIG_PATH, "config")


def _load_size():
    """
    Tải lại module size.py và xoá cache lỗi settings trước đó.

    Trả về:
        Module size đã được nạp mới nhất.
    """
    clear_settings_errors()
    return _load_module(SIZE_PATH, "size")


# ===========================================================================
# Tiện ích đường dẫn
# ===========================================================================

def _workspace_path(value: str) -> Path:
    """
    Chuyển đổi chuỗi đường dẫn (tuyệt đối hoặc tương đối) thành Path tuyệt đối.

    Đường dẫn tương đối được tính từ BASE_DIR (thư mục gốc dự án).

    Tham số:
        value : Chuỗi đường dẫn cần chuyển đổi.

    Trả về:
        Path tuyệt đối đã được resolve.
    """
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (BASE_DIR / path).resolve()
    return resolved


def _configured_dir(name: str) -> Path:
    """
    Lấy đường dẫn tuyệt đối của một thư mục được khai báo trong config.

    Tham số:
        name : Tên thuộc tính trong config, ví dụ 'INPUT_DIR' hoặc 'OUTPUT_DIR'.

    Trả về:
        Path tuyệt đối tương ứng.
    """
    cfg = _load_config()
    return _workspace_path(str(getattr(cfg, name)))


def _input_dir() -> Path:
    """
    Trả về đường dẫn thư mục INPUT_DIR, tạo thư mục nếu chưa tồn tại.

    Trả về:
        Path tuyệt đối đến INPUT_DIR.
    """
    path = _configured_dir("INPUT_DIR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_dir() -> Path:
    """
    Trả về đường dẫn thư mục OUTPUT_DIR, tạo thư mục nếu chưa tồn tại.

    Trả về:
        Path tuyệt đối đến OUTPUT_DIR.
    """
    path = _configured_dir("OUTPUT_DIR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_path() -> Path:
    """
    Trả về đường dẫn file pipeline.log nằm trong OUTPUT_DIR.

    Trả về:
        Path tuyệt đối đến output/pipeline.log.
    """
    return _output_dir() / "pipeline.log"


# ===========================================================================
# Tiện ích đọc/ghi config
# ===========================================================================

def _config_values(include_internal: bool = False) -> dict[str, Any]:
    """
    Đọc toàn bộ giá trị config hiện tại từ config.py.

    Tham số:
        include_internal : Nếu True thì bao gồm cả MODEL_DET và MODEL_SEG.

    Trả về:
        Dict ánh xạ tên khoá -> giá trị tương ứng.
    """
    cfg = _load_config()
    keys = CONFIG_KEYS + (INTERNAL_CONFIG_KEYS if include_internal else [])
    return {key: getattr(cfg, key) for key in keys}


def _jsonable_config() -> dict[str, Any]:
    """
    Đọc config công khai (không bao gồm khoá nội bộ) dưới dạng dict JSON-safe.

    Trả về:
        Dict gồm các khoá trong CONFIG_KEYS với giá trị hiện tại.
    """
    return _config_values(include_internal=False)


def _with_settings_errors(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gắn thêm trường _settings_errors vào payload nếu có lỗi đọc settings.json.

    Tham số:
        payload : Dict phản hồi gốc cần bổ sung thông tin lỗi.

    Trả về:
        Dict gốc có thêm khoá '_settings_errors' khi tồn tại lỗi.
    """
    errors = get_settings_errors()
    if errors:
        payload["_settings_errors"] = errors
    return payload


def _config_response():
    """
    Tạo phản hồi JSON chứa config hiện tại và lỗi settings (nếu có).

    Trả về:
        Response JSON của Flask.
    """
    return jsonify(_with_settings_errors(_jsonable_config()))


def _sizes_response():
    """
    Tạo phản hồi JSON chứa bảng kích cỡ hiện tại và lỗi settings (nếu có).

    Trả về:
        Response JSON của Flask.
    """
    return jsonify(_with_settings_errors(_jsonable_sizes()))


def _allowed_suffixes() -> set[str]:
    """
    Lấy tập hợp đuôi file được phép upload (ảnh + video), viết thường.

    Trả về:
        Set các chuỗi như {'.jpg', '.mp4', ...}.
    """
    cfg = _load_config()
    return {str(x).lower() for x in cfg.IMG_EXTS | cfg.VID_EXTS}


def _normalize_bool(value: Any) -> bool:
    """
    Chuyển đổi giá trị kiểu bất kỳ sang bool.

    Nhận diện chuỗi '1', 'true', 'yes', 'on' (không phân biệt hoa thường)
    là True; các chuỗi khác là False.

    Tham số:
        value : Giá trị cần chuyển đổi.

    Trả về:
        True hoặc False.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(
    data: dict[str, Any],
    key: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """
    Lấy và kiểm tra giá trị float từ dict.

    Tham số:
        data      : Dict chứa khoá cần đọc.
        key       : Tên khoá.
        min_value : Giới hạn dưới (bao gồm). None nghĩa là không kiểm tra.
        max_value : Giới hạn trên (bao gồm). None nghĩa là không kiểm tra.

    Trả về:
        Giá trị float hợp lệ.

    Ngoại lệ:
        ValueError : Khi không chuyển đổi được hoặc vượt ngoài khoảng cho phép.
    """
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{key} phải >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} phải <= {max_value}")
    return value


def _coerce_int(data: dict[str, Any], key: str, min_value: int | None = None) -> int:
    """
    Lấy và kiểm tra giá trị int từ dict.

    Tham số:
        data      : Dict chứa khoá cần đọc.
        key       : Tên khoá.
        min_value : Giới hạn dưới (bao gồm). None nghĩa là không kiểm tra.

    Trả về:
        Giá trị int hợp lệ.

    Ngoại lệ:
        ValueError : Khi không chuyển đổi được hoặc nhỏ hơn min_value.
    """
    try:
        value = int(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số nguyên") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{key} phải >= {min_value}")
    return value


def _write_config(data: dict[str, Any]) -> None:
    """
    Ghi các khoá config công khai vào settings.json.

    Tham số:
        data : Dict chứa ít nhất tất cả các khoá trong CONFIG_KEYS.
    """
    save_settings("config", {key: data[key] for key in CONFIG_KEYS})


def _validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Kiểm tra và chuẩn hoá payload config nhận từ client.

    Merge payload với giá trị hiện tại (giữ nguyên khoá nội bộ),
    sau đó kiểm tra kiểu dữ liệu và ràng buộc giá trị cho từng khoá.

    Tham số:
        payload : Dict JSON từ request PUT /api/config.

    Trả về:
        Dict đã được chuẩn hoá, sẵn sàng truyền vào _write_config.

    Ngoại lệ:
        ValueError : Khi bất kỳ khoá nào không hợp lệ.
    """
    current = _config_values(include_internal=True)
    public_payload = {key: value for key, value in payload.items() if key in CONFIG_KEYS}
    data = {**current, **public_payload}

    for key in ["INPUT_DIR", "OUTPUT_DIR", *INTERNAL_CONFIG_KEYS]:
        value = str(data.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} không được để trống")
        data[key] = value

    _workspace_path(data["INPUT_DIR"])
    _workspace_path(data["OUTPUT_DIR"])

    for key in BOOL_CONFIG_KEYS:
        data[key] = _normalize_bool(data[key])

    data["SCALE"]           = _coerce_float(data, "SCALE",           min_value=0.00001)
    data["CONF_DET"]        = _coerce_float(data, "CONF_DET",        min_value=0, max_value=1)
    data["CONF_SEG"]        = _coerce_float(data, "CONF_SEG",        min_value=0, max_value=1)
    data["BBOX_PAD"]        = _coerce_int(  data, "BBOX_PAD",        min_value=0)
    data["TOUCH_THRESHOLD"] = _coerce_float(data, "TOUCH_THRESHOLD", min_value=0)
    data["TARGET_FPS"]      = _coerce_float(data, "TARGET_FPS",      min_value=0)
    return data


# ===========================================================================
# Tiện ích kích cỡ tôm
# ===========================================================================

def _jsonable_sizes() -> dict[str, Any]:
    """
    Đọc bảng phân loại kích cỡ hiện tại từ size.py dưới dạng dict JSON-safe.

    Trả về:
        Dict gồm:
            ranges         : {nhãn: [từ_mm, đến_mm], ...}
            undersize_label: Nhãn tôm nhỏ hơn cỡ nhỏ nhất.
            oversize_label : Nhãn tôm lớn hơn cỡ lớn nhất.
            fallback_label : Nhãn dự phòng khi không khớp cỡ nào.
    """
    sizes = _load_size()
    return {
        "ranges"         : {label: list(bounds) for label, bounds in sizes.SIZE_RANGES.items()},
        "undersize_label": sizes.UNDERSIZE_LABEL,
        "oversize_label" : sizes.OVERSIZE_LABEL,
        "fallback_label" : sizes.FALLBACK_LABEL,
    }


def _validate_sizes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Kiểm tra và chuẩn hoá payload bảng kích cỡ nhận từ client.

    Quy tắc kiểm tra:
        - Mỗi nhãn phải khác rỗng và không trùng nhau.
        - Mỗi khoảng phải có đúng hai phần tử [từ, đến] đều >= 0 và từ < đến.
        - Các khoảng sau khi sắp xếp không được chồng lên nhau.

    Tham số:
        payload : Dict JSON từ request PUT /api/config/sizes.

    Trả về:
        Dict đã chuẩn hoá với khoá 'ranges', 'undersize_label', 'oversize_label', 'fallback_label'.

    Ngoại lệ:
        ValueError : Khi vi phạm bất kỳ quy tắc nào ở trên.
    """
    raw_ranges = payload.get("ranges")
    if not isinstance(raw_ranges, dict):
        raise ValueError("Bảng phân loại phải là object")
    if not raw_ranges:
        return {
            "ranges"         : {},
            "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
            "oversize_label" : str(payload.get("oversize_label",  "")).strip() or "Ngoại cỡ lớn",
            "fallback_label" : str(payload.get("fallback_label",  "")).strip() or "Ngoại cỡ",
        }

    ranges: list[tuple[str, float, float]] = []
    labels = set()
    for label, bounds in raw_ranges.items():
        clean_label = str(label).strip()
        if not clean_label:
            raise ValueError("Tên cỡ không được trống")
        if clean_label in labels:
            raise ValueError(f"Cỡ {clean_label} bị trùng")
        labels.add(clean_label)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Cỡ {clean_label} phải có [từ, đến]")
        try:
            lo = float(bounds[0])
            hi = float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cỡ {clean_label} phải là số") from exc
        if lo < 0 or hi < 0:
            raise ValueError(f"Cỡ {clean_label} phải >= 0")
        if lo >= hi:
            raise ValueError(f"Cỡ {clean_label}: mốc từ phải nhỏ hơn mốc đến")
        ranges.append((clean_label, lo, hi))

    ranges.sort(key=lambda item: (item[1], item[2], item[0]))
    previous_label, previous_hi = ranges[0][0], ranges[0][2]
    for label, lo, hi in ranges[1:]:
        if lo < previous_hi:
            raise ValueError(f"Cỡ {previous_label} và {label} đang chồng lấp")
        previous_label, previous_hi = label, hi

    return {
        "ranges"         : {label: (lo, hi) for label, lo, hi in ranges},
        "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
        "oversize_label" : str(payload.get("oversize_label",  "")).strip() or "Ngoại cỡ lớn",
        "fallback_label" : str(payload.get("fallback_label",  "")).strip() or "Ngoại cỡ",
    }


def _write_sizes(data: dict[str, Any]) -> None:
    """
    Ghi bảng kích cỡ đã chuẩn hoá vào settings.json (section 'size').

    Tham số:
        data : Dict đầu ra của _validate_sizes_payload.
    """
    save_settings(
        "size",
        {
            "SIZE_RANGES"    : data["ranges"],
            "UNDERSIZE_LABEL": data["undersize_label"],
            "OVERSIZE_LABEL" : data["oversize_label"],
            "FALLBACK_LABEL" : data["fallback_label"],
        },
    )


# ===========================================================================
# Tiện ích theo dõi pipeline
# ===========================================================================

def _pipeline_running() -> bool:
    """
    Kiểm tra xem pipeline con có đang chạy hay không.

    Nếu process đã kết thúc, cập nhật _pipeline_returncode và _pipeline_ended_at.

    Trả về:
        True khi process đang chạy, False khi đã kết thúc hoặc chưa khởi động.
    """
    global _pipeline_process, _pipeline_ended_at, _pipeline_returncode
    if _pipeline_process is None:
        return False
    code = _pipeline_process.poll()
    if code is None:
        return True
    _pipeline_returncode = code
    if _pipeline_ended_at is None:
        _pipeline_ended_at = time.time()
    return False


def _watch_pipeline(process: subprocess.Popen) -> None:
    """
    Chạy trong thread riêng để chờ pipeline kết thúc và ghi lại returncode.

    Tham số:
        process : Đối tượng Popen của pipeline con.
    """
    global _pipeline_ended_at, _pipeline_returncode
    code = process.wait()
    with _pipeline_lock:
        _pipeline_returncode = code
        _pipeline_ended_at = time.time()


def _pipeline_status() -> dict[str, Any]:
    """
    Trả về trạng thái hiện tại của pipeline dưới dạng dict.

    Trả về:
        Dict gồm:
            running    : bool – pipeline đang chạy hay không.
            returncode : int | None – mã thoát (None khi đang chạy).
            started_at : float | None – Unix timestamp lúc khởi động.
            ended_at   : float | None – Unix timestamp lúc kết thúc.
    """
    running = _pipeline_running()
    return {
        "running"    : running,
        "returncode" : None if running else _pipeline_returncode,
        "started_at" : _pipeline_started_at,
        "ended_at"   : None if running else _pipeline_ended_at,
    }


# ===========================================================================
# Tiện ích file input
# ===========================================================================

def _safe_input_name(filename: str) -> str:
    """
    Làm sạch tên file upload, thay thế ký tự không an toàn bằng dấu gạch dưới.

    Tham số:
        filename : Tên file gốc từ client (có thể chứa ký tự đặc biệt).

    Trả về:
        Tên file an toàn, không có dấu chấm đầu dòng; fallback nếu rỗng.
    """
    raw_name = Path(str(filename).replace("\\", "/")).name.strip()
    secured  = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip(" .")
    while secured.startswith("."):
        secured = secured[1:]
    if not secured:
        secured = f"upload_{time.time_ns()}"
    return secured


def _unique_destination(directory: Path, filename: str) -> Path:
    """
    Tìm đường dẫn đích không trùng trong thư mục đích bằng cách thêm hậu tố số.

    Ví dụ: nếu 'video.mp4' đã tồn tại thì trả về 'video_1.mp4', 'video_2.mp4', ...

    Tham số:
        directory : Thư mục đích.
        filename  : Tên file mong muốn.

    Trả về:
        Path chưa tồn tại trong directory.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem   = candidate.stem
    suffix = candidate.suffix
    index  = 1
    while True:
        next_candidate = directory / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _file_payload(path: Path) -> dict[str, Any]:
    """
    Tạo dict mô tả ngắn gọn một file để trả về cho client.

    Tham số:
        path : Đường dẫn đến file.

    Trả về:
        Dict gồm name, size (bytes), mtime (Unix timestamp), suffix.
    """
    stat = path.stat()
    return {
        "name"  : path.name,
        "size"  : stat.st_size,
        "mtime" : stat.st_mtime,
        "suffix": path.suffix.lower(),
    }


def _relative_workspace_path(path: Path) -> str:
    """
    Chuyển Path tuyệt đối thành đường dẫn tương đối so với BASE_DIR.

    Trả về '.' nếu path trùng với BASE_DIR.

    Tham số:
        path : Path tuyệt đối cần chuyển đổi.

    Trả về:
        Chuỗi POSIX tương đối, ví dụ 'input' hoặc 'output/2026-05-25_10-00-00'.
    """
    path = path.resolve()
    if path == BASE_DIR:
        return "."
    return path.relative_to(BASE_DIR).as_posix()


def _pick_local_folder(initial: str = "") -> Path | None:
    """
    Mở hộp thoại chọn thư mục trên máy chủ bằng tkinter.

    Chỉ hoạt động khi Flask chạy trực tiếp trên máy có giao diện đồ hoạ.

    Tham số:
        initial : Đường dẫn ban đầu gợi ý cho hộp thoại.

    Trả về:
        Path tuyệt đối thư mục đã chọn, hoặc None nếu người dùng huỷ.

    Ngoại lệ:
        RuntimeError : Khi không thể import tkinter (môi trường headless).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Không mở được hộp thoại chọn đường dẫn trên máy này") from exc

    initial_path = Path(initial) if initial else BASE_DIR
    if not initial_path.is_absolute():
        initial_path = (BASE_DIR / initial_path).resolve()
    if initial_path.is_file():
        initial_dir = initial_path.parent
    elif initial_path.exists():
        initial_dir = initial_path
    else:
        initial_dir = BASE_DIR

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(initialdir=str(initial_dir), title="Chọn folder")
    finally:
        root.destroy()

    if not selected:
        return None
    return _workspace_path(selected)


# ===========================================================================
# Tiện ích kết quả đo
# ===========================================================================

def _run_dirs() -> list[Path]:
    """
    Liệt kê các thư mục run trong OUTPUT_DIR, sắp xếp mới nhất lên đầu.

    Trả về:
        Danh sách Path đến từng thư mục run (là thư mục con trực tiếp).
    """
    output_dir = _output_dir()
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _result_json_files(run_dir: Path) -> list[Path]:
    """
    Tìm tất cả file *_results.json trong một thư mục run.

    Tham số:
        run_dir : Đường dẫn đến thư mục run (ví dụ output/2026-05-25_10-00-00).

    Trả về:
        Danh sách Path đến các file JSON kết quả, sắp xếp theo tên.
    """
    return sorted(run_dir.glob("*/*_results.json"))


def _image_url(raw_path: str | None) -> str | None:
    """
    Chuyển đường dẫn file ảnh trên disk thành URL /outputs/<path> phục vụ qua Flask.

    Kiểm tra file phải nằm trong OUTPUT_DIR và thực sự tồn tại trên disk.

    Tham số:
        raw_path : Đường dẫn tuyệt đối hoặc tương đối đến file ảnh.

    Trả về:
        Chuỗi URL như '/outputs/2026-05-25_10-00-00/stem/F6_Result_F45_234px.jpg',
        hoặc None nếu nằm ngoài OUTPUT_DIR hoặc file không tồn tại.
    """
    if not raw_path:
        return None
    output_root = _output_dir().resolve()
    candidate   = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(output_root)
    except ValueError:
        return None
    if not candidate.exists():
        return None
    return f"/outputs/{relative.as_posix()}"


def _normalize_images(images: dict[str, Any] | None) -> dict[str, Any]:
    """
    Chuyển đổi dict đường dẫn ảnh thành dict URL /outputs/<path>.

    Hỗ trợ cả giá trị đơn (str) lẫn danh sách (list[str]).
    Loại bỏ các URL None (file không tồn tại hoặc nằm ngoài OUTPUT_DIR).

    Tham số:
        images : Dict từ trường 'images' trong JSON kết quả.

    Trả về:
        Dict cùng cấu trúc với giá trị là URL hoặc danh sách URL.
    """
    if not images:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in images.items():
        if isinstance(value, list):
            urls = [url for item in value if (url := _image_url(item))]
            normalized[key] = urls
        else:
            normalized[key] = _image_url(value)
    return normalized


def _results_for_run(run_name: str | None = None) -> dict[str, Any]:
    """
    Đọc toàn bộ kết quả của một lần chạy từ các file *_results.json.

    Nếu run_name là None hoặc không tìm thấy, dùng run mới nhất.

    Tham số:
        run_name : Tên thư mục run (ví dụ '2026-05-25_10-00-00'). None = run mới nhất.

    Trả về:
        Dict gồm:
            run     : Tên run đã chọn (str | None).
            sources : Danh sách dict mô tả từng file nguồn và danh sách tôm.
    """
    runs     = _run_dirs()
    selected = None
    if run_name:
        for run_dir in runs:
            if run_dir.name == run_name:
                selected = run_dir
                break
    elif runs:
        selected = runs[0]

    if selected is None:
        return {"run": None, "sources": []}

    sources = []
    for json_file in _result_json_files(selected):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shrimps = []
        for item in data.get("shrimps", []):
            shrimp = dict(item)
            shrimp["images"] = _normalize_images(shrimp.get("images"))
            shrimps.append(shrimp)
        sources.append(
            {
                "source_file"   : data.get("source_file",  json_file.parent.name),
                "source_stem"   : data.get("source_stem",  json_file.parent.name),
                "processed_at"  : data.get("processed_at"),
                "scale_mm_per_px": data.get("scale_mm_per_px"),
                "shrimps"       : shrimps,
            }
        )
    return {"run": selected.name, "sources": sources}


# ===========================================================================
# Tiện ích xuất CSV/Excel
# ===========================================================================

def _result_export_rows(data: dict[str, Any]) -> list[list[Any]]:
    """
    Chuyển kết quả đo thành danh sách hàng để xuất CSV hoặc Excel.

    Hàng đầu tiên là tiêu đề (RESULT_EXPORT_HEADERS).

    Tham số:
        data : Dict trả về bởi _results_for_run.

    Trả về:
        Danh sách các hàng, mỗi hàng là list giá trị tương ứng với RESULT_EXPORT_HEADERS.
    """
    rows = [RESULT_EXPORT_HEADERS]
    for source in data["sources"]:
        for shrimp in source["shrimps"]:
            rows.append(
                [
                    data["run"],
                    source["source_file"],
                    shrimp.get("track_id"),
                    shrimp.get("frame_idx"),
                    shrimp.get("pixel_length"),
                    shrimp.get("real_length_mm"),
                    shrimp.get("size"),
                ]
            )
    return rows


def _export_filename(data: dict[str, Any], suffix: str) -> str:
    """
    Tạo tên file xuất kết quả dựa trên tên run, ví dụ 'shrimp_results_2026-05-25.csv'.

    Tham số:
        data   : Dict từ _results_for_run có khoá 'run'.
        suffix : Phần mở rộng file, ví dụ 'csv' hoặc 'xlsx'.

    Trả về:
        Tên file chuỗi an toàn cho Content-Disposition header.
    """
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data["run"] or "empty"))
    return f"shrimp_results_{run_name}.{suffix}"


def _excel_column_name(index: int) -> str:
    """
    Chuyển chỉ số cột (1-based) thành tên cột Excel kiểu chữ cái (A, B, ..., Z, AA, ...).

    Tham số:
        index : Chỉ số cột bắt đầu từ 1.

    Trả về:
        Chuỗi chữ cái đại diện cột, ví dụ 1 -> 'A', 27 -> 'AA'.
    """
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value: Any, row_index: int, column_index: int) -> str:
    """
    Tạo XML tag <c> cho một ô trong file XLSX.

    Số nguyên/thực được viết dạng <v>; chuỗi dùng inlineStr.

    Tham số:
        value        : Giá trị ô (None, số, hoặc chuỗi).
        row_index    : Chỉ số hàng (1-based).
        column_index : Chỉ số cột (1-based).

    Trả về:
        Chuỗi XML fragment cho ô đó.
    """
    cell_ref = f"{_excel_column_name(column_index)}{row_index}"
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{xml_escape(str(value))}</t></is></c>'


def _xlsx_bytes(rows: list[list[Any]]) -> bytes:
    """
    Tạo nội dung file XLSX thuần Python (không cần thư viện ngoài) từ danh sách hàng.

    File XLSX tạo ra gồm một sheet duy nhất tên 'Results'.

    Tham số:
        rows : Danh sách hàng, mỗi hàng là list giá trị (tiêu đề + dữ liệu).

    Trả về:
        Bytes của file XLSX hợp lệ, sẵn sàng ghi hoặc trả về qua HTTP.
    """
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(
            _xlsx_cell(value, row_index, column_index)
            for column_index, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_index}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Results" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": sheet_xml,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


# ===========================================================================
# Tiện ích nạp file scale (CSV/Excel)
# ===========================================================================

def _fold_column_name(value: Any) -> str:
    """
    Chuẩn hoá tên cột: bỏ dấu Unicode, thay ký tự đặc biệt bằng '_', viết thường.

    Dùng để so khớp tên cột không phân biệt dấu và hoa thường.

    Tham số:
        value : Tên cột gốc từ file CSV/Excel.

    Trả về:
        Chuỗi chuẩn hoá, ví dụ 'Độ dài mm' -> 'do_dai_mm'.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def _parse_positive_mm(value: Any) -> float | None:
    """
    Trích xuất số thực dương (mm) từ một giá trị ô bất kỳ.

    Chấp nhận cả dấu phẩy lẫn dấu chấm thập phân, loại bỏ ký tự đơn vị.

    Tham số:
        value : Giá trị ô từ file CSV/Excel, ví dụ '152.3', '152,3 mm'.

    Trả về:
        Số float dương nếu hợp lệ, None nếu không tìm thấy hoặc <= 0.
    """
    text  = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    return number if number > 0 else None


def _decode_csv_bytes(raw: bytes) -> str:
    """
    Giải mã bytes CSV thử lần lượt các encoding phổ biến.

    Thứ tự thử: utf-8-sig -> utf-8 -> cp1258 -> latin-1.
    Fallback cuối dùng utf-8 với errors='replace' để không bao giờ raise.

    Tham số:
        raw : Nội dung file CSV dạng bytes.

    Trả về:
        Chuỗi Unicode đã giải mã.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_csv_table(raw: bytes) -> list[list[str]]:
    """
    Đọc file CSV từ bytes và trả về bảng 2 chiều (list of list of str).

    Tự động phát hiện dấu phân cách (,  ;  tab) từ 4096 byte đầu.

    Tham số:
        raw : Nội dung file CSV dạng bytes.

    Trả về:
        Danh sách hàng, mỗi hàng là list các ô đã strip khoảng trắng.
    """
    text   = _decode_csv_bytes(raw)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), dialect)]


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """
    Đọc bảng chuỗi chia sẻ (sharedStrings.xml) trong file XLSX.

    Tham số:
        archive : ZipFile đang mở của file XLSX.

    Trả về:
        Danh sách chuỗi theo thứ tự chỉ số, dùng để tra cứu ô kiểu 's'.
    """
    try:
        raw_xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root    = ET.fromstring(raw_xml)
    strings = []
    for item in root.iter():
        if item.tag.endswith("}si") or item.tag == "si":
            strings.append(
                "".join(node.text or "" for node in item.iter()
                        if node.tag.endswith("}t") or node.tag == "t")
            )
    return strings


def _xlsx_column_index(cell_ref: str) -> int:
    """
    Chuyển tham chiếu ô Excel (ví dụ 'B3') thành chỉ số cột 0-based.

    Tham số:
        cell_ref : Chuỗi tham chiếu ô, ví dụ 'A1', 'BC10'.

    Trả về:
        Chỉ số cột 0-based (A=0, B=1, ...).
    """
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0
    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    """
    Lấy giá trị văn bản của một phần tử <c> trong worksheet XLSX.

    Hỗ trợ kiểu ô: inlineStr (t='inlineStr'), shared string (t='s'), số thuần.

    Tham số:
        cell           : Phần tử XML <c> cần đọc.
        shared_strings : Bảng chuỗi chia sẻ từ _xlsx_shared_strings.

    Trả về:
        Giá trị dạng chuỗi của ô, rỗng nếu ô trống.
    """
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter()
            if node.tag.endswith("}t") or node.tag == "t"
        )
    value_node = None
    for node in cell:
        if node.tag.endswith("}v") or node.tag == "v":
            value_node = node
            break
    raw_value = value_node.text if value_node is not None else ""
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            return ""
    return str(raw_value or "")


def _parse_xlsx_table(raw: bytes) -> list[list[str]]:
    """
    Đọc sheet đầu tiên của file XLSX từ bytes và trả về bảng 2 chiều.

    Không phụ thuộc openpyxl – tự phân tích XML bên trong ZIP.

    Tham số:
        raw : Nội dung file XLSX dạng bytes.

    Trả về:
        Danh sách hàng, mỗi hàng là list các ô đã strip khoảng trắng.

    Ngoại lệ:
        ValueError : Khi không tìm thấy sheet1 trong file XLSX.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        try:
            sheet_xml = archive.read("xl/worksheets/sheet1.xml")
        except KeyError as exc:
            raise ValueError("Không tìm thấy sheet đầu tiên trong file XLSX") from exc

        shared_strings = _xlsx_shared_strings(archive)
        root = ET.fromstring(sheet_xml)
        rows = []
        for row in root.iter():
            if not (row.tag.endswith("}row") or row.tag == "row"):
                continue
            cells: dict[int, str] = {}
            max_index = -1
            for cell in row:
                if not (cell.tag.endswith("}c") or cell.tag == "c"):
                    continue
                col_index          = _xlsx_column_index(cell.attrib.get("r", ""))
                cells[col_index]   = _xlsx_cell_text(cell, shared_strings).strip()
                max_index          = max(max_index, col_index)
            if max_index >= 0:
                rows.append([cells.get(index, "") for index in range(max_index + 1)])
        return rows


def _scale_import_table(raw: bytes, filename: str) -> list[list[str]]:
    """
    Điều phối đọc file scale: chọn CSV hoặc XLSX dựa vào đuôi file.

    Tham số:
        raw      : Nội dung file dạng bytes.
        filename : Tên file gốc (chỉ dùng để lấy đuôi .csv/.xlsx).

    Trả về:
        Bảng 2 chiều (list of list of str).

    Ngoại lệ:
        ValueError : Khi đuôi file không phải .csv hoặc .xlsx.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return _parse_csv_table(raw)
    if suffix == ".xlsx":
        return _parse_xlsx_table(raw)
    raise ValueError("Chỉ hỗ trợ file CSV hoặc XLSX")


def _header_index(header: list[str], aliases: set[str]) -> int | None:
    """
    Tìm chỉ số cột đầu tiên trong header khớp với một trong các alias.

    So khớp sau khi chuẩn hoá tên cột bằng _fold_column_name.

    Tham số:
        header  : Hàng tiêu đề từ bảng CSV/Excel.
        aliases : Tập hợp tên cột đã chuẩn hoá cần tìm.

    Trả về:
        Chỉ số (0-based) của cột khớp đầu tiên, hoặc None nếu không tìm thấy.
    """
    folded = [_fold_column_name(cell) for cell in header]
    for index, name in enumerate(folded):
        if name in aliases:
            return index
    return None


def _normalize_track_id(value: Any) -> str:
    """
    Chuẩn hoá track_id từ file CSV/Excel về dạng chuỗi số nguyên nếu có thể.

    Ví dụ: '1.0' -> '1', '2' -> '2', 'abc' -> 'abc'.

    Tham số:
        value : Giá trị ô track_id gốc.

    Trả về:
        Chuỗi chuẩn hoá của track_id.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _scale_import_records(table: list[list[str]]) -> list[dict[str, Any]]:
    """
    Phân tích bảng CSV/Excel thành danh sách bản ghi mm thực tế.

    Hỗ trợ hai chế độ:
        - Có header: nhận diện cột mm, source_stem, source_file, track_id.
        - Không header: đọc tuần tự cột đầu tiên làm giá trị mm.

    Tham số:
        table : Bảng 2 chiều từ _scale_import_table.

    Trả về:
        Danh sách dict, mỗi dict có ít nhất khoá 'real_length_mm' (float dương).
        Có thể có thêm 'source_stem', 'source_file', 'track_id'.
    """
    rows = [row for row in table if any(str(cell).strip() for cell in row)]
    if not rows:
        return []

    header        = rows[0]
    mm_col        = _header_index(header, SCALE_MM_COLUMNS)
    source_stem_col  = _header_index(header, SCALE_SOURCE_STEM_COLUMNS)
    source_file_col  = _header_index(header, SCALE_SOURCE_FILE_COLUMNS)
    track_col     = _header_index(header, SCALE_TRACK_COLUMNS)
    has_header    = mm_col is not None

    if not has_header:
        records = []
        for row in rows:
            mm = _parse_positive_mm(row[0] if row else "")
            if mm is not None:
                records.append({"real_length_mm": mm})
        return records

    records = []
    for row in rows[1:]:
        mm = _parse_positive_mm(row[mm_col] if mm_col < len(row) else "")
        if mm is None:
            continue
        record: dict[str, Any] = {"real_length_mm": mm}
        if source_stem_col is not None and source_stem_col < len(row):
            record["source_stem"] = str(row[source_stem_col]).strip()
        if source_file_col is not None and source_file_col < len(row):
            record["source_file"] = str(row[source_file_col]).strip()
        if track_col is not None and track_col < len(row):
            record["track_id"] = _normalize_track_id(row[track_col])
        records.append(record)
    return records


def _scale_import_measurements(
    records: list[dict[str, Any]],
    ordered_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Ghép các bản ghi mm từ file scale với danh sách kết quả đo hiện tại.

    Chiến lược ghép:
        1. Nếu bản ghi có source_stem + track_id: khớp theo khoá chính xác.
        2. Nếu bản ghi chỉ có mm: ghép tuần tự với các hàng kết quả còn trống.

    Tham số:
        records      : Danh sách bản ghi từ _scale_import_records.
        ordered_rows : Danh sách hàng kết quả theo đúng thứ tự hiển thị trên UI.

    Trả về:
        Tuple gồm:
            measurements : Danh sách dict đã ghép, cùng thứ tự với ordered_rows.
            warnings     : Danh sách cảnh báo khi bỏ qua một số bản ghi.
    """
    warnings: list[str] = []
    measurements_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_by_key = {
        (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip()): row
        for row in ordered_rows
    }

    sequential_records = []
    for record in records:
        source_stem = str(record.get("source_stem") or "").strip()
        source_file = str(record.get("source_file") or "").strip()
        track_id    = str(record.get("track_id")    or "").strip()

        if not source_stem and source_file:
            source_stem = Path(source_file).stem

        if source_stem and track_id:
            key = (source_stem, track_id)
            if key in ordered_by_key:
                source_row = ordered_by_key[key]
                measurements_by_key[key] = {
                    "source_file"   : source_row.get("source_file", source_file),
                    "source_stem"   : source_stem,
                    "track_id"      : track_id,
                    "real_length_mm": record["real_length_mm"],
                }
            else:
                warnings.append(f"Bỏ qua {source_stem} ID {track_id}: không có trong kết quả hiện tại")
        else:
            sequential_records.append(record)

    if sequential_records:
        available_rows = [
            row for row in ordered_rows
            if (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip())
            not in measurements_by_key
        ]
        for row, record in zip(available_rows, sequential_records):
            key = (
                str(row.get("source_stem") or "").strip(),
                str(row.get("track_id")    or "").strip(),
            )
            measurements_by_key[key] = {
                "source_file"   : row.get("source_file", ""),
                "source_stem"   : key[0],
                "track_id"      : key[1],
                "real_length_mm": record["real_length_mm"],
            }
        if len(sequential_records) > len(available_rows):
            warnings.append(
                f"Bỏ qua {len(sequential_records) - len(available_rows)} "
                f"dòng mm vì nhiều hơn số dòng kết quả"
            )

    measurements = [
        measurements_by_key[
            (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip())
        ]
        for row in ordered_rows
        if (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip())
        in measurements_by_key
    ]
    return measurements, warnings


# ===========================================================================
# Tiện ích hiệu chuẩn SCALE
# ===========================================================================

def _find_run_dir(run_name: str | None) -> Path | None:
    """
    Tìm thư mục run theo tên trong OUTPUT_DIR.

    Nếu run_name là None, trả về run mới nhất.

    Tham số:
        run_name : Tên thư mục run, hoặc None để lấy run mới nhất.

    Trả về:
        Path đến thư mục run, hoặc None nếu không có run nào.
    """
    for run_dir in _run_dirs():
        if run_name is None or run_dir.name == run_name:
            return run_dir
    return None


def _calibration_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Xây dựng chỉ mục hiệu chuẩn (source_stem, track_id) -> thông tin tôm.

    Đọc tất cả file *_results.json trong run_dir để lấy pixel_length.

    Tham số:
        run_dir : Đường dẫn thư mục run.

    Trả về:
        Dict ánh xạ (source_stem, track_id) -> dict gồm source_file,
        source_stem, track_id, pixel_length.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for json_file in _result_json_files(run_dir):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_stem = str(data.get("source_stem") or json_file.parent.name)
        source_file = str(data.get("source_file") or source_stem)
        for shrimp in data.get("shrimps", []):
            track_id = str(shrimp.get("track_id"))
            index[(source_stem, track_id)] = {
                "source_stem"  : source_stem,
                "source_file"  : source_file,
                "track_id"     : shrimp.get("track_id"),
                "pixel_length" : shrimp.get("pixel_length"),
            }
    return index


def _least_squares_mm_per_px(
    samples: list[dict[str, Any]],
) -> tuple[float, float, list[dict[str, Any]]]:
    """
    Tính hệ số SCALE (mm/px) bằng hồi quy tuyến tính bình phương tối thiểu qua gốc.

    Mô hình: real_length_mm = SCALE x pixel_length  (không có hệ số tự do).

    SCALE tối ưu = sum(pixel * real) / sum(pixel^2).

    Tham số:
        samples : Danh sách dict có khoá 'pixel_length' và 'real_length_mm'.

    Trả về:
        Tuple gồm:
            scale          : Giá trị SCALE tính được (mm/px).
            rmse_mm        : Sai số toàn phương trung bình (mm).
            enriched       : Danh sách samples có thêm 'fitted_mm' và 'residual_mm'.

    Ngoại lệ:
        ValueError : Khi không có mẫu hoặc tổng pixel^2 bằng 0.
    """
    if not samples:
        raise ValueError("Cần ít nhất 1 mẫu hợp lệ để tính scale")

    sum_xx = sum(s["pixel_length"] ** 2 for s in samples)
    sum_xy = sum(s["pixel_length"] * s["real_length_mm"] for s in samples)
    if sum_xx == 0:
        raise ValueError("pixel_length phải lớn hơn 0")

    scale          = sum_xy / sum_xx
    enriched       = []
    squared_errors = []
    for s in samples:
        fitted_mm    = s["pixel_length"] * scale
        residual_mm  = s["real_length_mm"] - fitted_mm
        squared_errors.append(residual_mm ** 2)
        entry = dict(s)
        entry["fitted_mm"]   = round(fitted_mm,   6)
        entry["residual_mm"] = round(residual_mm, 6)
        enriched.append(entry)

    rmse_mm = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return scale, rmse_mm, enriched


# ===========================================================================
# Route: trang chủ và health check
# ===========================================================================

@app.get("/")
def index():
    """Phục vụ trang HTML giao diện chính (index.html)."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """
    Kiểm tra sức khoẻ của server và trả về trạng thái pipeline.

    Trả về JSON gồm ok, url giao diện, url static và pipeline_status.
    """
    return jsonify(
        {
            "ok"    : True,
            "ui"    : "/",
            "static": "/static/",
            "status": _pipeline_status(),
        }
    )


# ===========================================================================
# Route: quản lý file input
# ===========================================================================

@app.get("/api/files/input")
def list_input_files():
    """
    Liệt kê tất cả file trong INPUT_DIR.

    Trả về JSON: {"files": [{name, size, mtime, suffix}, ...]}.
    """
    input_dir = _input_dir()
    files = [_file_payload(path) for path in sorted(input_dir.iterdir()) if path.is_file()]
    return jsonify({"files": files})


@app.post("/api/files/upload")
def upload_files():
    """
    Nhận file upload từ form multipart và lưu vào INPUT_DIR.

    Từ chối file có định dạng nằm ngoài IMG_EXTS | VID_EXTS.
    Tự động đổi tên nếu trùng với file đã có.

    Trả về JSON: {"saved": [...], "rejected": [...]}.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Chưa chọn file"}), 400

    allowed   = _allowed_suffixes()
    input_dir = _input_dir()
    saved     = []
    rejected  = []
    for file in files:
        original = file.filename or ""
        suffix   = Path(original).suffix.lower()
        if suffix not in allowed:
            rejected.append({"name": original, "reason": "Định dạng không hỗ trợ"})
            continue
        filename    = _safe_input_name(original)
        destination = _unique_destination(input_dir, filename)
        file.save(destination)
        saved.append(_file_payload(destination))
    return jsonify({"saved": saved, "rejected": rejected})


@app.delete("/api/files/input/<path:filename>")
def delete_input_file(filename: str):
    """
    Xoá một file khỏi INPUT_DIR theo tên.

    Kiểm tra path traversal trước khi xoá. Không báo lỗi nếu file không tồn tại.

    Tham số URL:
        filename : Tên file cần xoá.

    Trả về JSON: {"ok": true}.
    """
    input_dir = _input_dir().resolve()
    target    = (input_dir / filename).resolve()
    try:
        target.relative_to(input_dir)
    except ValueError:
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    if target.exists() and target.is_file():
        target.unlink()
    return jsonify({"ok": True})


# ===========================================================================
# Route: duyệt thư mục (dùng cho hộp thoại chọn đường dẫn)
# ===========================================================================

@app.get("/api/filesystem/directories")
def list_directories():
    """
    Liệt kê các thư mục con của một đường dẫn, dùng để duyệt cây thư mục trên UI.

    Query param:
        path : Đường dẫn cần liệt kê (mặc định '.').

    Trả về JSON gồm current, absolute_path, parent, directories.
    """
    raw_path = request.args.get("path") or "."
    try:
        current = _workspace_path(raw_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not current.exists():
        current = current.parent if current.parent.exists() else BASE_DIR
    if current.is_file():
        current = current.parent
    current = current.resolve()

    directories = []
    try:
        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400

    for child in children:
        if child.is_dir():
            directories.append(
                {
                    "name"         : child.name,
                    "path"         : _relative_workspace_path(child),
                    "absolute_path": str(child.resolve()),
                }
            )

    parent = None
    if current != BASE_DIR:
        parent = _relative_workspace_path(current.parent)

    return jsonify(
        {
            "current"      : _relative_workspace_path(current),
            "absolute_path": str(current),
            "parent"       : parent,
            "directories"  : directories,
        }
    )


# ===========================================================================
# Route: điều khiển pipeline
# ===========================================================================

@app.post("/api/pipeline/run")
def run_pipeline():
    """
    Khởi động pipeline đo tôm bằng cách chạy main.py trong subprocess mới.

    Xoá sạch pipeline.log cũ trước khi bắt đầu.
    Trả lỗi 409 nếu pipeline đang chạy.

    Trả về JSON: {"ok": true, "status": {...}}.
    """
    global _pipeline_process, _pipeline_started_at, _pipeline_ended_at, _pipeline_returncode
    with _pipeline_lock:
        if _pipeline_running():
            return jsonify({"error": "Pipeline đang chạy"}), 409
        _output_dir().mkdir(parents=True, exist_ok=True)
        _log_path().write_text("", encoding="utf-8")
        _pipeline_started_at  = time.time()
        _pipeline_ended_at    = None
        _pipeline_returncode  = None
        _pipeline_process = subprocess.Popen(
            [sys.executable, str(MAIN_PATH)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        threading.Thread(target=_watch_pipeline, args=(_pipeline_process,), daemon=True).start()
    return jsonify({"ok": True, "status": _pipeline_status()})


@app.post("/api/pipeline/stop")
def stop_pipeline():
    """
    Gửi tín hiệu terminate đến subprocess pipeline đang chạy.

    Không làm gì nếu pipeline không chạy.

    Trả về JSON: {"ok": true}.
    """
    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_running():
            _pipeline_process.terminate()
            return jsonify({"ok": True})
    return jsonify({"ok": True, "message": "Pipeline không chạy"})


@app.get("/api/pipeline/status")
def pipeline_status():
    """
    Trả về trạng thái hiện tại của pipeline.

    Trả về JSON: {running, returncode, started_at, ended_at}.
    """
    return jsonify(_pipeline_status())


@app.get("/api/pipeline/log")
def pipeline_log():
    """
    Đọc nội dung mới của pipeline.log từ vị trí offset.

    Dùng cho long-polling từ client; trả về đoạn log mới kể từ lần đọc trước.

    Query param:
        offset : Vị trí byte bắt đầu đọc (mặc định 0).

    Trả về JSON: {content, offset, size}.
    """
    try:
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        offset = 0
    offset = max(0, offset)
    path   = _log_path()
    if not path.exists():
        return jsonify({"content": "", "offset": 0, "size": 0})
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        data        = stream.read()
        next_offset = stream.tell()
    return jsonify({"content": data.decode("utf-8", errors="replace"), "offset": next_offset, "size": size})


# ===========================================================================
# Route: cấu hình hệ thống
# ===========================================================================

@app.get("/api/config")
def get_config():
    """
    Đọc cấu hình hiện tại từ config.py và trả về dạng JSON.

    Trả về JSON config công khai (không bao gồm MODEL_DET, MODEL_SEG).
    """
    return _config_response()


@app.put("/api/config")
def put_config():
    """
    Lưu cấu hình mới vào settings.json.

    Body JSON: dict với các khoá trong CONFIG_KEYS.
    Trả lỗi 400 kèm thông điệp khi dữ liệu không hợp lệ.

    Trả về JSON config đã cập nhật (giống GET /api/config).
    """
    try:
        data = _validate_config_payload(request.get_json(force=True, silent=True) or {})
        _write_config(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _config_response()


@app.post("/api/config/pick-path")
def pick_config_path():
    """
    Mở hộp thoại chọn thư mục trên máy chủ và trả về đường dẫn đã chọn.

    Chỉ hỗ trợ khoá INPUT_DIR và OUTPUT_DIR, chế độ 'folder'.

    Body JSON: {"key": "INPUT_DIR"|"OUTPUT_DIR", "mode": "folder"}.
    Trả về JSON: {"path": "...", "cancelled": bool}.
    """
    payload = request.get_json(force=True, silent=True) or {}
    key     = str(payload.get("key")  or "").strip()
    mode    = str(payload.get("mode") or "folder").strip()
    if key not in {"INPUT_DIR", "OUTPUT_DIR"}:
        return jsonify({"error": "Hằng số config không hợp lệ"}), 400
    if mode != "folder":
        return jsonify({"error": "Kiểu chọn đường dẫn không hợp lệ"}), 400

    current = str(_jsonable_config().get(key, ""))
    try:
        selected = _pick_local_folder(current)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if selected is None:
        return jsonify({"path": None, "cancelled": True})

    return jsonify({"path": _relative_workspace_path(selected), "cancelled": False})


# ===========================================================================
# Route: nạp file scale và hiệu chuẩn
# ===========================================================================

@app.post("/api/calibrate/import-measurements")
def import_scale_measurements():
    """
    Nạp file CSV/Excel chứa độ dài thực tế (mm) và ghép với kết quả đo hiện tại.

    Form data:
        file : File CSV hoặc XLSX (tối đa 8 MB).
        run  : Tên run cần ghép.
        rows : JSON string danh sách hàng kết quả theo thứ tự hiển thị.

    Trả về JSON: {measurements, count, expected_count, warnings}.
    """
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Chưa chọn file scale"}), 400

    run_name = str(request.form.get("run") or "").strip()
    if not run_name:
        return jsonify({"error": "Chưa chọn run để nạp file scale"}), 400
    if _find_run_dir(run_name) is None:
        return jsonify({"error": f"Không tìm thấy run {run_name}"}), 404

    try:
        ordered_rows = json.loads(request.form.get("rows") or "[]")
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Danh sách dòng kết quả không hợp lệ: {exc}"}), 400
    if not isinstance(ordered_rows, list) or not ordered_rows:
        return jsonify({"error": "Chưa có dòng kết quả để ghép file scale"}), 400
    ordered_rows = [row for row in ordered_rows if isinstance(row, dict)]
    if not ordered_rows:
        return jsonify({"error": "Danh sách dòng kết quả không hợp lệ"}), 400

    raw = file.read(SCALE_IMPORT_MAX_BYTES + 1)
    if len(raw) > SCALE_IMPORT_MAX_BYTES:
        return jsonify({"error": "File scale vượt quá giới hạn 8 MB"}), 400

    try:
        table   = _scale_import_table(raw, file.filename)
        records = _scale_import_records(table)
    except (OSError, UnicodeError, zipfile.BadZipFile, ET.ParseError, ValueError) as exc:
        return jsonify({"error": f"Không đọc được file scale: {exc}"}), 400

    if not records:
        return jsonify({"error": "Không tìm thấy cột/dòng mm hợp lệ trong file scale"}), 400

    measurements, warnings = _scale_import_measurements(records, ordered_rows)
    if not measurements:
        return jsonify({"error": "Không ghép được dòng mm nào với kết quả hiện tại", "warnings": warnings}), 400

    return jsonify(
        {
            "measurements"  : measurements,
            "count"         : len(measurements),
            "expected_count": len(ordered_rows),
            "warnings"      : warnings,
        }
    )


@app.post("/api/calibrate")
def calibrate_scale():
    """
    Tính hệ số SCALE mới từ danh sách mẫu (pixel_length, real_length_mm) và lưu vào settings.json.

    Dùng hồi quy bình phương tối thiểu qua gốc: real = SCALE x pixel.

    Body JSON:
        run          : Tên run chứa dữ liệu pixel_length.
        measurements : [{source_stem, track_id, real_length_mm}, ...].

    Trả về JSON: {scale, method, formula, rmse_mm, count, samples, errors, config}.
    """
    payload      = request.get_json(force=True, silent=True) or {}
    run_name     = str(payload.get("run") or "").strip()
    measurements = payload.get("measurements")
    if not run_name:
        return jsonify({"error": "Chưa chọn run để tính scale"}), 400
    if not isinstance(measurements, list) or not measurements:
        return jsonify({"error": "Chưa nhập độ dài thực tế ở cột mm"}), 400

    run_dir = _find_run_dir(run_name)
    if run_dir is None:
        return jsonify({"error": f"Không tìm thấy run {run_name}"}), 404

    index   = _calibration_index(run_dir)
    samples = []
    errors  = []
    for item in measurements:
        if not isinstance(item, dict):
            continue
        source_stem = str(item.get("source_stem") or "").strip()
        track_id    = str(item.get("track_id")    or "").strip()
        try:
            real_length_mm = float(item.get("real_length_mm"))
        except (TypeError, ValueError):
            errors.append(f"{source_stem} ID {track_id}: mm không hợp lệ")
            continue
        if real_length_mm <= 0:
            errors.append(f"{source_stem} ID {track_id}: mm phải lớn hơn 0")
            continue

        record = index.get((source_stem, track_id))
        if record is None:
            errors.append(f"{source_stem} ID {track_id}: không tìm thấy trong JSON")
            continue
        try:
            pixel_length = float(record["pixel_length"])
        except (TypeError, ValueError):
            errors.append(f"{source_stem} ID {track_id}: pixel_length không hợp lệ")
            continue
        if pixel_length <= 0:
            errors.append(f"{source_stem} ID {track_id}: pixel_length phải lớn hơn 0")
            continue

        samples.append(
            {
                "source_stem"   : source_stem,
                "source_file"   : record["source_file"],
                "track_id"      : record["track_id"],
                "pixel_length"  : pixel_length,
                "real_length_mm": real_length_mm,
            }
        )

    if not samples:
        message = "Không có mẫu hợp lệ để tính scale"
        if errors:
            message += ": " + "; ".join(errors[:3])
        return jsonify({"error": message, "errors": errors}), 400

    try:
        new_scale, rmse_mm, samples = _least_squares_mm_per_px(samples)
    except ValueError as exc:
        return jsonify({"error": str(exc), "errors": errors}), 400

    config_data = _validate_config_payload({"SCALE": round(new_scale, 6)})
    _write_config(config_data)
    return jsonify(
        {
            "scale"  : config_data["SCALE"],
            "method" : "least_squares_origin",
            "formula": "real_length_mm = scale * pixel_length",
            "rmse_mm": round(rmse_mm, 6),
            "count"  : len(samples),
            "samples": samples,
            "errors" : errors,
            "config" : _with_settings_errors(_jsonable_config()),
        }
    )


# ===========================================================================
# Route: bảng kích cỡ tôm
# ===========================================================================

@app.get("/api/config/sizes")
def get_sizes():
    """
    Đọc bảng phân loại kích cỡ từ size.py và trả về dạng JSON.

    Trả về JSON: {ranges, undersize_label, oversize_label, fallback_label}.
    """
    return _sizes_response()


@app.put("/api/config/sizes")
def put_sizes():
    """
    Lưu bảng phân loại kích cỡ mới vào settings.json.

    Body JSON: {ranges: {nhãn: [từ, đến]}, undersize_label, oversize_label, fallback_label}.
    Trả lỗi 400 khi dữ liệu vi phạm quy tắc (trùng nhãn, chồng khoảng, ...).

    Trả về JSON bảng kích cỡ đã cập nhật (giống GET /api/config/sizes).
    """
    try:
        data = _validate_sizes_payload(request.get_json(force=True, silent=True) or {})
        _write_sizes(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _sizes_response()


# ===========================================================================
# Route: danh sách run và kết quả đo
# ===========================================================================

@app.get("/api/results/runs")
def result_runs():
    """
    Liệt kê tất cả các lần chạy (run) trong OUTPUT_DIR.

    Trả về JSON: {"runs": [{name, mtime, source_count, shrimp_count}, ...]}.
    """
    runs = []
    for run_dir in _run_dirs():
        files       = _result_json_files(run_dir)
        shrimp_count = 0
        for json_file in files:
            try:
                data          = json.loads(json_file.read_text(encoding="utf-8"))
                shrimp_count += len(data.get("shrimps", []))
            except (OSError, json.JSONDecodeError):
                continue
        runs.append(
            {
                "name"        : run_dir.name,
                "mtime"       : run_dir.stat().st_mtime,
                "source_count": len(files),
                "shrimp_count": shrimp_count,
            }
        )
    return jsonify({"runs": runs})


@app.get("/api/results")
def results():
    """
    Trả về toàn bộ kết quả đo của một run.

    Query param:
        run : Tên run (tuỳ chọn, mặc định run mới nhất).

    Trả về JSON: {run, sources: [{source_file, shrimps: [...]}]}.
    """
    return jsonify(_results_for_run(request.args.get("run")))


@app.get("/api/results/export-csv")
def export_csv():
    """
    Xuất kết quả đo của một run dưới dạng file CSV để tải về.

    Query param:
        run : Tên run (tuỳ chọn, mặc định run mới nhất).

    Trả về Response với Content-Disposition: attachment; filename=shrimp_results_<run>.csv.
    """
    data   = _results_for_run(request.args.get("run"))
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(_result_export_rows(data))
    return Response(
        buffer.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'csv')}"},
    )


@app.get("/api/results/export-excel")
def export_excel():
    """
    Xuất kết quả đo của một run dưới dạng file XLSX để tải về.

    Query param:
        run : Tên run (tuỳ chọn, mặc định run mới nhất).

    Trả về Response XLSX với Content-Disposition: attachment; filename=shrimp_results_<run>.xlsx.
    """
    data = _results_for_run(request.args.get("run"))
    return Response(
        _xlsx_bytes(_result_export_rows(data)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'xlsx')}"},
    )


# ===========================================================================
# Route: phục vụ ảnh debug
# ===========================================================================

@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    """
    Phục vụ file ảnh debug (F3–F6) từ OUTPUT_DIR.

    Kiểm tra path traversal: chỉ phục vụ file nằm trong OUTPUT_DIR.

    Tham số URL:
        filename : Đường dẫn tương đối so với OUTPUT_DIR.

    Trả về file tĩnh hoặc lỗi 400 nếu đường dẫn không hợp lệ.
    """
    output_dir = _output_dir().resolve()
    target     = (output_dir / filename).resolve()
    try:
        target.relative_to(output_dir)
    except ValueError:
        return jsonify({"error": "Đường dẫn không hợp lệ"}), 400
    return send_from_directory(output_dir, filename)


# ===========================================================================
# Điểm khởi động
# ===========================================================================

if __name__ == "__main__":
    host  = os.environ.get("HOST", "127.0.0.1")
    port  = int(os.environ.get("PORT", "3000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Shrimp Measure UI: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)