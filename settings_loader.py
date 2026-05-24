import json
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
_SETTINGS_ERRORS: dict[str, str] = {}


def load_setting(key: str, default: Any) -> Any:
    if not SETTINGS_PATH.exists():
        return default

    try:
        settings = _read_settings()
        value = _find_value(settings, key, default)
        value = _same_type(value, default)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        _SETTINGS_ERRORS[key] = f"Lỗi đọc settings.json key '{key}', đang dùng giá trị mặc định: {exc}"
        return default

    _SETTINGS_ERRORS.pop(key, None)
    return value


def clear_settings_errors() -> None:
    _SETTINGS_ERRORS.clear()


def get_settings_errors() -> list[str]:
    return list(_SETTINGS_ERRORS.values())


def save_settings(section: str, values: dict[str, Any]) -> None:
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


def _read_settings() -> dict[str, Any]:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("gốc settings.json phải là object")
    return settings


def _find_value(settings: dict[str, Any], key: str, default: Any) -> Any:
    if key in settings:
        return settings[key]

    for section in settings.values():
        if isinstance(section, dict) and key in section:
            return section[key]

    return default


def _same_type(value: Any, default: Any) -> Any:
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
            raise ValueError("giá trị không đúng dạng tuple/list")
        return tuple(_same_type(item, default[index]) for index, item in enumerate(value))
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError("giá trị không đúng dạng object")
        sample = next(iter(default.values()), None)
        return {str(item_key): _same_type(item_value, sample) for item_key, item_value in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
