"""
settings_loader.py

Đọc và ghi cài đặt từ/vào settings.json.

Cơ chế tự phục hồi
    Nếu settings.json chưa tồn tại, bị xóa, hoặc key cần đọc chưa có,
    load_setting() sẽ tự ghi giá trị mặc định vào đúng section trong file
    để lần sau người dùng có thể chỉnh trực tiếp mà không cần tạo tay.

    Nếu file bị lỗi JSON (nội dung hỏng), hàm ghi lỗi vào _SETTINGS_ERRORS
    và trả về default mà KHÔNG xóa nội dung file cũ để tránh mất dữ liệu
    người dùng đã chỉnh tay.

Cách dùng trong config.py (section mặc định là "config"):
    SCALE = load_setting("SCALE", 1.0)

Cách dùng trong size.py:
    SIZE_RANGES = load_setting("SIZE_RANGES", {}, section="size")
"""

import json
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

_SETTINGS_ERRORS: dict[str, str] = {}

# Sentinel để phân biệt "không tìm thấy key" với "giá trị là None".
_MISSING = object()


# ---- API công khai ----
def load_setting(key: str, default: Any, section: str = "config") -> Any:
    """
    Đọc một giá trị từ settings.json.

    Nếu file chưa tồn tại hoặc key không có trong file, hàm sẽ:
      1. Ghi giá trị mặc định vào file dưới section chỉ định
         (tạo mới nếu file chưa có, bổ sung nếu key chưa có).
      2. Trả về default.

    Nếu file bị lỗi JSON, hàm ghi lỗi vào _SETTINGS_ERRORS và trả về
    default mà không xóa nội dung file cũ.

    Tham số:
        key:     Tên cài đặt (phân biệt hoa thường).
        default: Giá trị mặc định khi không tìm thấy hoặc khi lỗi.
        section: Tên nhóm trong settings.json để ghi default vào.
                 Mặc định là "config" để tương thích với config.py.

    Trả về:
        Giá trị đọc được (đã ép kiểu theo default) hoặc default.
    """
    if not SETTINGS_PATH.exists():
        _write_default(key, default, section)
        _SETTINGS_ERRORS.pop(key, None)
        return default

    try:
        settings = _read_settings()
        value = _find_value(settings, key)
        if value is _MISSING:
            _write_default(key, default, section)
            _SETTINGS_ERRORS.pop(key, None)
            return default
        value = _same_type(value, default)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        _SETTINGS_ERRORS[key] = (
            f"Lỗi đọc settings.json key '{key}', đang dùng giá trị mặc định: {exc}"
        )
        return default

    _SETTINGS_ERRORS.pop(key, None)
    return value


def clear_settings_errors() -> None:
    """Xóa toàn bộ lỗi đọc settings đã tích lũy (gọi trước mỗi lần reload config)."""
    _SETTINGS_ERRORS.clear()


def get_settings_errors() -> list[str]:
    """Trả về danh sách mô tả lỗi đọc settings chưa được xử lý."""
    return list(_SETTINGS_ERRORS.values())


def save_settings(section: str, values: dict[str, Any]) -> None:
    """
    Ghi một nhóm cài đặt vào settings.json.

    Nếu file đã tồn tại, chỉ section được chỉ định bị ghi đè;
    các section khác giữ nguyên.

    Tham số:
        section: Tên nhóm ("config", "size", ...).
        values:  Dict key->value sẽ ghi vào settings[section].

    Ném:
        ValueError nếu không đọc hoặc ghi được file.
    """
    try:
        settings = _read_settings() if SETTINGS_PATH.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Không đọc được settings.json để lưu chính thức: {exc}") from exc

    settings[section] = _jsonable(values)
    try:
        SETTINGS_PATH.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"Không ghi được settings.json: {exc}") from exc

    _SETTINGS_ERRORS.clear()


# ---- Hàm nội bộ ----
def _write_default(key: str, default: Any, section: str) -> None:
    """
    Ghi giá trị mặc định của key vào settings.json dưới section chỉ định.

    Quy tắc ghi:
      - Nếu file tồn tại nhưng bị lỗi JSON: bỏ qua, không ghi đè để tránh
        làm mất dữ liệu người dùng đã chỉnh tay.
      - Nếu key đã có trong section: không ghi đè giá trị hiện có.
      - Nếu file chưa tồn tại: tạo mới rồi ghi.
    """
    try:
        settings = _read_settings() if SETTINGS_PATH.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        # File hỏng: không ghi đè, người dùng tự sửa.
        return

    sec = settings.setdefault(section, {})
    if not isinstance(sec, dict):
        settings[section] = {}
        sec = settings[section]

    if key in sec:
        # Giá trị đã tồn tại: không cần ghi thêm.
        return

    sec[key] = _jsonable(default)
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # Lỗi ghi: câm lặng, caller vẫn nhận được default.
        pass


def _read_settings() -> dict[str, Any]:
    """Đọc toàn bộ settings.json. Ném ValueError nếu root không phải object JSON."""
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("Gốc settings.json phải là object JSON")
    return settings


def _find_value(settings: dict[str, Any], key: str) -> Any:
    """
    Tìm key trong settings.json.

    Ưu tiên tìm ở top-level trước, sau đó duyệt qua từng section.
    Trả về _MISSING nếu không tìm thấy ở đâu.
    """
    if key in settings:
        return settings[key]
    for section in settings.values():
        if isinstance(section, dict) and key in section:
            return section[key]
    return _MISSING


def _same_type(value: Any, default: Any) -> Any:
    """Ép kiểu value về kiểu của default. Ném ValueError nếu không thể chuyển đổi."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            raise ValueError("Giá trị không đúng dạng tuple/list")
        return tuple(_same_type(item, default[i]) for i, item in enumerate(value))
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError("Giá trị không đúng dạng object")
        sample = next(iter(default.values()), None)
        return {str(k): _same_type(v, sample) for k, v in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    """Chuyển đổi value thành kiểu JSON-serializable."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value