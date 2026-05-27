"""
settings_loader.py

Đọc và ghi cài đặt từ settings.json.

Nếu file bị thiếu, hỏng, hoặc root không phải object JSON thì hệ thống
tạo lại file rỗng, dùng tham số mặc định, và lưu cảnh báo cho giao diện.
Đây là nơi duy nhất của app chính biết đường dẫn settings.json và cách
phục hồi file cấu hình.
"""

import json
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

_SETTING_WARNINGS: list[str] = []
_RECOVERY_WARNING = (
    "Không đọc được settings.json, đã chuyển sang tham số mặc định và tạo lại settings.json."
)


def read_setting() -> dict[str, Any]:
    """
    Đọc toàn bộ settings.json.

    Trả về dict cấu hình nếu file đọc được. Nếu file thiếu, JSON hỏng hoặc
    root không phải object, hàm tạo lại file rỗng, ghi cảnh báo nội bộ và
    trả về dict rỗng để caller tiếp tục dùng default.
    """
    settings = _read_settings_file()
    if settings is None:
        return _recover_settings_file()
    return settings


def preflight_settings() -> dict[str, Any]:
    """
    Kiểm tra settings.json trước khi chạy pipeline.

    Nếu file thiếu hoặc hỏng, hàm chỉ phục hồi file và trả cờ recovered để
    caller dừng pipeline, cho người dùng kiểm tra lại cấu hình trước khi đo.
    """
    settings = _read_settings_file()
    recovered = settings is None
    if recovered:
        _recover_settings_file()
    return {
        "ok": True,
        "recovered": recovered,
        "warnings": pull_setting_warnings() if recovered else [],
    }


def load_setting(key: str, default: Any, section: str = "config") -> Any:
    """
    Đọc một giá trị theo đúng section.

    Nếu thiếu hoặc sai kiểu thì ghi default vào settings.json và trả về default.
    Giá trị hợp lệ được ép kiểu theo kiểu của default trước khi trả về.
    """
    settings = read_setting()
    section_values = _ensure_section(settings, section)

    if key not in section_values:
        section_values[key] = _jsonable(default)
        _safe_write_settings_file(settings)
        return default

    try:
        return _same_type(section_values[key], default)
    except (TypeError, ValueError, IndexError) as exc:
        _push_warning(
            f"Giá trị {section}.{key} trong settings.json không hợp lệ, "
            f"đã chuyển sang tham số mặc định: {exc}"
        )
        section_values[key] = _jsonable(default)
        _safe_write_settings_file(settings)
        return default


def save_setting(section: str, values: dict[str, Any]) -> None:
    """
    Ghi một section vào settings.json.

    Nếu file hiện tại còn đọc được, các section khác được giữ nguyên. Nếu file
    hỏng hoặc thiếu, loader phục hồi file rỗng rồi ghi section đang lưu.
    """
    settings = _read_settings_file()
    if settings is None:
        settings = _recover_settings_file()

    settings[section] = _jsonable(values)
    try:
        _write_settings_file(settings)
    except OSError as exc:
        raise ValueError(f"Không ghi được settings.json: {exc}") from exc


def pull_setting_warnings() -> list[str]:
    """
    Lấy và xóa các cảnh báo settings đang chờ hiển thị.

    Flask API gọi hàm này để đưa cảnh báo phục hồi settings.json về UI đúng
    một lần, tránh toast lặp lại vô hạn.
    """
    warnings = list(_SETTING_WARNINGS)
    _SETTING_WARNINGS.clear()
    return warnings


def _read_settings_file() -> dict[str, Any] | None:
    """Đọc file settings.json, trả None nếu file thiếu hoặc không hợp lệ."""
    if not SETTINGS_PATH.exists():
        return None

    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    if not isinstance(settings, dict):
        return None
    return settings


def _recover_settings_file() -> dict[str, Any]:
    """Tạo lại settings.json rỗng và ghi cảnh báo phục hồi nội bộ."""
    _push_warning(_RECOVERY_WARNING)
    settings: dict[str, Any] = {}
    _safe_write_settings_file(settings)
    return settings


def _ensure_section(settings: dict[str, Any], section: str) -> dict[str, Any]:
    """Đảm bảo section tồn tại và là object trước khi đọc hoặc ghi key."""
    section_values = settings.get(section)
    if not isinstance(section_values, dict):
        section_values = {}
        settings[section] = section_values
    return section_values


def _push_warning(message: str) -> None:
    """Thêm warning vào hàng đợi nếu nội dung chưa tồn tại."""
    if message not in _SETTING_WARNINGS:
        _SETTING_WARNINGS.append(message)


def _write_settings_file(settings: dict[str, Any]) -> None:
    """Ghi toàn bộ dict settings xuống settings.json bằng UTF-8."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_write_settings_file(settings: dict[str, Any]) -> None:
    """Ghi settings nhưng chỉ lưu warning nếu hệ thống file báo lỗi."""
    try:
        _write_settings_file(settings)
    except OSError as exc:
        _push_warning(f"Không ghi được settings.json: {exc}")


def _same_type(value: Any, default: Any) -> Any:
    """
    Ép value về kiểu của default.

    Hàm hỗ trợ các kiểu đang dùng trong config và size: bool, int, float,
    str, list, tuple và dict. Nếu không ép được, caller sẽ dùng default.
    """
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

    if isinstance(default, list):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Giá trị không đúng dạng list")
        if not default:
            return list(value)
        if len(default) == 2 and len(value) != len(default):
            raise ValueError("Giá trị list không đúng số phần tử")
        if len(default) == 2 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in default
        ):
            return [float(item) for item in value]
        sample = default[0]
        return [_same_type(item, sample) for item in value]

    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            raise ValueError("Giá trị không đúng dạng tuple/list")
        return tuple(_same_type(item, default[i]) for i, item in enumerate(value))

    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError("Giá trị không đúng dạng object")
        sample = next(iter(default.values()), None)
        if sample is None:
            return dict(value)
        return {str(k): _same_type(v, sample) for k, v in value.items()}

    return value


def _jsonable(value: Any) -> Any:
    """Chuyển value thành kiểu ghi được vào JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value
