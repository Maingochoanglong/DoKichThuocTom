"""
settings_loader.py

Đọc và ghi cài đặt từ settings.json.
Nếu settings.json bị thiếu, hỏng, hoặc không phải object JSON thì hệ thống
tạo lại file, dùng tham số mặc định, và lưu cảnh báo để giao diện hiển thị.
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
    """Đọc toàn bộ settings.json, tự tạo lại object rỗng nếu file không đọc được."""
    if not SETTINGS_PATH.exists():
        _push_warning(_RECOVERY_WARNING)
        _safe_write_settings({})
        return {}

    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict):
            raise ValueError("Gốc settings.json phải là object JSON")
        return settings
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _push_warning(_RECOVERY_WARNING)
        _safe_write_settings({})
        return {}


def load_setting(key: str, default: Any, section: str = "config") -> Any:
    """
    Đọc một giá trị trong settings.json theo đúng section.
    Nếu thiếu hoặc sai kiểu thì ghi default vào file và trả về default.
    """
    settings = read_setting()
    sec = settings.get(section)
    if not isinstance(sec, dict):
        sec = {}
        settings[section] = sec

    if key not in sec:
        sec[key] = _jsonable(default)
        _safe_write_settings(settings)
        return default

    try:
        return _same_type(sec[key], default)
    except (TypeError, ValueError, IndexError) as exc:
        _push_warning(
            f"Giá trị {section}.{key} trong settings.json không hợp lệ, "
            f"đã chuyển sang tham số mặc định: {exc}"
        )
        sec[key] = _jsonable(default)
        _safe_write_settings(settings)
        return default


def save_setting(section: str, values: dict[str, Any]) -> None:
    """Ghi một section vào settings.json và giữ các section khác nếu còn đọc được."""
    settings = read_setting()
    settings[section] = _jsonable(values)
    try:
        _write_settings(settings)
    except OSError as exc:
        raise ValueError(f"Không ghi được settings.json: {exc}") from exc


def pull_setting_warnings() -> list[str]:
    """Lấy và xóa các cảnh báo phục hồi settings đang chờ hiển thị."""
    warnings = list(_SETTING_WARNINGS)
    _SETTING_WARNINGS.clear()
    return warnings


def _push_warning(message: str) -> None:
    if message not in _SETTING_WARNINGS:
        _SETTING_WARNINGS.append(message)


def _write_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_write_settings(settings: dict[str, Any]) -> None:
    try:
        _write_settings(settings)
    except OSError as exc:
        _push_warning(f"Không ghi được settings.json: {exc}")


def _same_type(value: Any, default: Any) -> Any:
    """Ép kiểu value về kiểu của default."""
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