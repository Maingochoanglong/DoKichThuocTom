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
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from settings_loader import get_settings_errors, save_settings_section


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
SIZE_PATH = BASE_DIR / "size.py"
MAIN_PATH = BASE_DIR / "main.py"


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


class _StatusLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/pipeline/status" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_StatusLogFilter())

_pipeline_lock = threading.Lock()
_pipeline_process: subprocess.Popen | None = None
_pipeline_started_at: float | None = None
_pipeline_ended_at: float | None = None
_pipeline_returncode: int | None = None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"_web_{name}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config():
    return _load_module(CONFIG_PATH, "config")


def _load_size():
    return _load_module(SIZE_PATH, "size")


def _workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (BASE_DIR / path).resolve()
    if resolved != BASE_DIR and BASE_DIR not in resolved.parents:
        raise ValueError("Đường dẫn phải nằm trong thư mục dự án")
    return resolved


def _configured_dir(name: str) -> Path:
    cfg = _load_config()
    return _workspace_path(str(getattr(cfg, name)))


def _input_dir() -> Path:
    path = _configured_dir("INPUT_DIR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_dir() -> Path:
    path = _configured_dir("OUTPUT_DIR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_path() -> Path:
    return _output_dir() / "pipeline.log"


def _ensure_runtime_dirs() -> None:
    _input_dir()
    _output_dir()


def _config_values(include_internal: bool = False) -> dict[str, Any]:
    cfg = _load_config()
    keys = [
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
    if include_internal:
        keys[5:5] = ["MODEL_DET", "MODEL_SEG"]
    return {key: getattr(cfg, key) for key in keys}


def _jsonable_config() -> dict[str, Any]:
    return _config_values(include_internal=False)


def _with_settings_errors(section: str, payload: dict[str, Any]) -> dict[str, Any]:
    errors = get_settings_errors(section)
    if errors:
        payload["_settings_errors"] = errors
    return payload


def _allowed_suffixes() -> set[str]:
    cfg = _load_config()
    return {str(x).lower() for x in cfg.IMG_EXTS | cfg.VID_EXTS}


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(data: dict[str, Any], key: str, min_value: float | None = None, max_value: float | None = None) -> float:
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
    try:
        value = int(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số nguyên") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{key} phải >= {min_value}")
    return value


def _write_config(data: dict[str, Any]) -> None:
    save_settings_section(
        "config",
        {
            "INPUT_DIR": data["INPUT_DIR"],
            "OUTPUT_DIR": data["OUTPUT_DIR"],
            "CLEAR_OUTPUT": data["CLEAR_OUTPUT"],
            "CLEAR_INPUT": data["CLEAR_INPUT"],
            "CHUNK_MODE": data["CHUNK_MODE"],
            "SCALE": data["SCALE"],
            "CONF_DET": data["CONF_DET"],
            "CONF_SEG": data["CONF_SEG"],
            "BBOX_PAD": data["BBOX_PAD"],
            "TOUCH_THRESHOLD": data["TOUCH_THRESHOLD"],
            "TARGET_FPS": data["TARGET_FPS"],
            "CONVEYOR_VERTICAL": data["CONVEYOR_VERTICAL"],
            "SAVE": data["SAVE"],
        },
    )


def _validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    current = _config_values(include_internal=True)
    public_payload = {key: value for key, value in payload.items() if key not in {"MODEL_DET", "MODEL_SEG"}}
    data = {**current, **public_payload}

    for key in ["INPUT_DIR", "OUTPUT_DIR", "MODEL_DET", "MODEL_SEG"]:
        value = str(data.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} không được để trống")
        data[key] = value

    _workspace_path(data["INPUT_DIR"])
    _workspace_path(data["OUTPUT_DIR"])

    for key in ["CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE", "CONVEYOR_VERTICAL", "SAVE"]:
        data[key] = _normalize_bool(data[key])

    data["SCALE"] = _coerce_float(data, "SCALE", min_value=0.000001)
    data["CONF_DET"] = _coerce_float(data, "CONF_DET", min_value=0, max_value=1)
    data["CONF_SEG"] = _coerce_float(data, "CONF_SEG", min_value=0, max_value=1)
    data["BBOX_PAD"] = _coerce_int(data, "BBOX_PAD", min_value=0)
    data["TOUCH_THRESHOLD"] = _coerce_float(data, "TOUCH_THRESHOLD", min_value=0)
    data["TARGET_FPS"] = _coerce_float(data, "TARGET_FPS", min_value=0)
    return data


def _jsonable_sizes() -> dict[str, Any]:
    sizes = _load_size()
    return {
        "ranges": {label: list(bounds) for label, bounds in sizes.SIZE_RANGES.items()},
        "undersize_label": sizes.UNDERSIZE_LABEL,
        "oversize_label": sizes.OVERSIZE_LABEL,
        "fallback_label": sizes.FALLBACK_LABEL,
    }


def _validate_sizes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_ranges = payload.get("ranges")
    if not isinstance(raw_ranges, dict) or not raw_ranges:
        raise ValueError("Bảng phân loại không được trống")

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
        "ranges": {label: (lo, hi) for label, lo, hi in ranges},
        "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
        "oversize_label": str(payload.get("oversize_label", "")).strip() or "Ngoại cỡ lớn",
        "fallback_label": str(payload.get("fallback_label", "")).strip() or "Ngoại cỡ",
    }


def _write_sizes(data: dict[str, Any]) -> None:
    save_settings_section(
        "size",
        {
            "SIZE_RANGES": data["ranges"],
            "UNDERSIZE_LABEL": data["undersize_label"],
            "OVERSIZE_LABEL": data["oversize_label"],
            "FALLBACK_LABEL": data["fallback_label"],
        },
    )


def _pipeline_running() -> bool:
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
    global _pipeline_ended_at, _pipeline_returncode
    code = process.wait()
    with _pipeline_lock:
        _pipeline_returncode = code
        _pipeline_ended_at = time.time()


def _pipeline_status() -> dict[str, Any]:
    running = _pipeline_running()
    return {
        "running": running,
        "returncode": None if running else _pipeline_returncode,
        "started_at": _pipeline_started_at,
        "ended_at": None if running else _pipeline_ended_at,
    }


def _safe_input_name(filename: str) -> str:
    raw_name = Path(str(filename).replace("\\", "/")).name.strip()
    secured = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip(" .")
    while secured.startswith("."):
        secured = secured[1:]
    if not secured:
        secured = f"upload_{time.time_ns()}"
    return secured


def _unique_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = directory / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _file_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "suffix": path.suffix.lower(),
    }


def _relative_workspace_path(path: Path) -> str:
    path = path.resolve()
    if path == BASE_DIR:
        return "."
    return path.relative_to(BASE_DIR).as_posix()


def _pick_local_path(mode: str, initial: str = "") -> Path | None:
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
        if mode == "folder":
            selected = filedialog.askdirectory(initialdir=str(initial_dir), title="Chọn folder")
        else:
            selected = filedialog.askopenfilename(initialdir=str(initial_dir), title="Chọn file")
    finally:
        root.destroy()

    if not selected:
        return None
    return _workspace_path(selected)


def _run_dirs() -> list[Path]:
    output_dir = _output_dir()
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _result_json_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*_results.json"))


def _image_url(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    output_root = _output_dir().resolve()
    candidate = Path(raw_path)
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
    if not images:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in images.items():
        if isinstance(value, list):
            normalized[key] = [_image_url(item) for item in value if _image_url(item)]
        else:
            normalized[key] = _image_url(value)
    return normalized


def _results_for_run(run_name: str | None = None) -> dict[str, Any]:
    runs = _run_dirs()
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
                "source_file": data.get("source_file", json_file.parent.name),
                "source_stem": data.get("source_stem", json_file.parent.name),
                "processed_at": data.get("processed_at"),
                "scale_mm_per_px": data.get("scale_mm_per_px"),
                "shrimps": shrimps,
            }
        )
    return {"run": selected.name, "sources": sources}


def _find_run_dir(run_name: str | None) -> Path | None:
    for run_dir in _run_dirs():
        if run_name is None or run_dir.name == run_name:
            return run_dir
    return None


def _calibration_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
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
                "source_stem": source_stem,
                "source_file": source_file,
                "track_id": shrimp.get("track_id"),
                "pixel_length": shrimp.get("pixel_length"),
            }
    return index


def _least_squares_mm_per_px(samples: list[dict[str, Any]]) -> tuple[float, float, float, list[dict[str, Any]]]:
    if len(samples) < 2:
        raise ValueError("Cần ít nhất 2 mẫu hợp lệ để tính scale theo y = m x + b")

    n = len(samples)
    sum_x = sum(sample["pixel_length"] for sample in samples)
    sum_y = sum(sample["real_length_mm"] for sample in samples)
    sum_xx = sum(sample["pixel_length"] ** 2 for sample in samples)
    sum_xy = sum(sample["pixel_length"] * sample["real_length_mm"] for sample in samples)
    denominator = n * sum_xx - sum_x ** 2
    if denominator == 0:
        raise ValueError("Cần ít nhất 2 mẫu có pixel_length khác nhau để tính scale theo y = m x + b")

    scale = (n * sum_xy - sum_x * sum_y) / denominator
    intercept_mm = (sum_y - scale * sum_x) / n
    enriched_samples = []
    squared_errors = []
    for sample in samples:
        fitted_mm = sample["pixel_length"] * scale + intercept_mm
        residual_mm = sample["real_length_mm"] - fitted_mm
        squared_errors.append(residual_mm ** 2)
        enriched_sample = dict(sample)
        enriched_sample["fitted_mm"] = round(fitted_mm, 6)
        enriched_sample["residual_mm"] = round(residual_mm, 6)
        enriched_samples.append(enriched_sample)

    rmse_mm = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return scale, intercept_mm, rmse_mm, enriched_samples


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/ui")
def ui():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "ui": "/",
            "static": "/static/",
            "status": _pipeline_status(),
        }
    )


@app.get("/api/files/input")
def list_input_files():
    input_dir = _input_dir()
    files = [_file_payload(path) for path in sorted(input_dir.iterdir()) if path.is_file()]
    return jsonify({"files": files})


@app.post("/api/files/upload")
def upload_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Chưa chọn file"}), 400

    allowed = _allowed_suffixes()
    input_dir = _input_dir()
    saved = []
    rejected = []
    for file in files:
        original = file.filename or ""
        suffix = Path(original).suffix.lower()
        if suffix not in allowed:
            rejected.append({"name": original, "reason": "Định dạng không hỗ trợ"})
            continue
        filename = _safe_input_name(original)
        destination = _unique_destination(input_dir, filename)
        file.save(destination)
        saved.append(_file_payload(destination))
    return jsonify({"saved": saved, "rejected": rejected})


@app.delete("/api/files/input/<path:filename>")
def delete_input_file(filename: str):
    input_dir = _input_dir().resolve()
    target = (input_dir / filename).resolve()
    try:
        target.relative_to(input_dir)
    except ValueError:
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    if target.exists() and target.is_file():
        target.unlink()
    return jsonify({"ok": True})


@app.get("/api/filesystem/directories")
def list_directories():
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
                    "name": child.name,
                    "path": _relative_workspace_path(child),
                    "absolute_path": str(child.resolve()),
                }
            )

    parent = None
    if current != BASE_DIR:
        parent = _relative_workspace_path(current.parent)

    return jsonify(
        {
            "current": _relative_workspace_path(current),
            "absolute_path": str(current),
            "parent": parent,
            "directories": directories,
        }
    )


@app.post("/api/pipeline/run")
def run_pipeline():
    global _pipeline_process, _pipeline_started_at, _pipeline_ended_at, _pipeline_returncode
    with _pipeline_lock:
        if _pipeline_running():
            return jsonify({"error": "Pipeline đang chạy"}), 409
        _output_dir().mkdir(parents=True, exist_ok=True)
        _log_path().write_text("", encoding="utf-8")
        _pipeline_started_at = time.time()
        _pipeline_ended_at = None
        _pipeline_returncode = None
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
    with _pipeline_lock:
        if _pipeline_process is not None and _pipeline_running():
            _pipeline_process.terminate()
            return jsonify({"ok": True})
    return jsonify({"ok": True, "message": "Pipeline không chạy"})


@app.get("/api/pipeline/status")
def pipeline_status():
    return jsonify(_pipeline_status())


@app.get("/api/pipeline/log")
def pipeline_log():
    try:
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        offset = 0
    offset = max(0, offset)
    path = _log_path()
    if not path.exists():
        return jsonify({"content": "", "offset": 0, "size": 0})
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
        next_offset = stream.tell()
    return jsonify({"content": data.decode("utf-8", errors="replace"), "offset": next_offset, "size": size})


@app.get("/api/config")
def get_config():
    return jsonify(_with_settings_errors("config", _jsonable_config()))


@app.put("/api/config")
def put_config():
    try:
        data = _validate_config_payload(request.get_json(force=True, silent=True) or {})
        _write_config(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_with_settings_errors("config", _jsonable_config()))


@app.post("/api/config/pick-path")
def pick_config_path():
    payload = request.get_json(force=True, silent=True) or {}
    key = str(payload.get("key") or "").strip()
    mode = str(payload.get("mode") or "file").strip()
    if key not in {"INPUT_DIR", "OUTPUT_DIR"}:
        return jsonify({"error": "Hằng số config không hợp lệ"}), 400
    if mode not in {"file", "folder"}:
        return jsonify({"error": "Kiểu chọn đường dẫn không hợp lệ"}), 400

    current = str(_jsonable_config().get(key, ""))
    try:
        selected = _pick_local_path(mode, current)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if selected is None:
        return jsonify({"path": None, "cancelled": True})

    if key in {"INPUT_DIR", "OUTPUT_DIR"} and selected.is_file():
        selected = selected.parent
    return jsonify({"path": _relative_workspace_path(selected), "cancelled": False})


@app.patch("/api/config/scale")
def patch_scale():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        data = _validate_config_payload({"SCALE": payload.get("scale")})
        _write_config(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_with_settings_errors("config", _jsonable_config()))


@app.post("/api/calibrate")
def calibrate_scale():
    payload = request.get_json(force=True, silent=True) or {}
    run_name = str(payload.get("run") or "").strip()
    measurements = payload.get("measurements")
    if not run_name:
        return jsonify({"error": "Chưa chọn run để tính scale"}), 400
    if not isinstance(measurements, list) or not measurements:
        return jsonify({"error": "Chưa nhập độ dài thực tế ở cột mm"}), 400

    run_dir = _find_run_dir(run_name)
    if run_dir is None:
        return jsonify({"error": f"Không tìm thấy run {run_name}"}), 404

    index = _calibration_index(run_dir)
    samples = []
    errors = []
    for item in measurements:
        if not isinstance(item, dict):
            continue
        source_stem = str(item.get("source_stem") or "").strip()
        track_id = str(item.get("track_id") or "").strip()
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

        sample_scale = real_length_mm / pixel_length
        samples.append(
            {
                "source_stem": source_stem,
                "source_file": record["source_file"],
                "track_id": record["track_id"],
                "pixel_length": pixel_length,
                "real_length_mm": real_length_mm,
                "scale": sample_scale,
            }
        )

    if not samples:
        message = "Không có mẫu hợp lệ để tính scale"
        if errors:
            message += ": " + "; ".join(errors[:3])
        return jsonify({"error": message, "errors": errors}), 400

    try:
        new_scale, intercept_mm, rmse_mm, samples = _least_squares_mm_per_px(samples)
    except ValueError as exc:
        return jsonify({"error": str(exc), "errors": errors}), 400

    config_data = _validate_config_payload({"SCALE": round(new_scale, 6)})
    _write_config(config_data)
    return jsonify(
        {
            "scale": config_data["SCALE"],
            "intercept_mm": round(intercept_mm, 6),
            "method": "least_squares_linear",
            "formula": "real_length_mm = scale * pixel_length + intercept_mm",
            "rmse_mm": round(rmse_mm, 6),
            "count": len(samples),
            "samples": samples,
            "errors": errors,
            "config": _with_settings_errors("config", _jsonable_config()),
        }
    )


@app.get("/api/config/sizes")
def get_sizes():
    return jsonify(_with_settings_errors("size", _jsonable_sizes()))


@app.put("/api/config/sizes")
def put_sizes():
    try:
        data = _validate_sizes_payload(request.get_json(force=True, silent=True) or {})
        _write_sizes(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_with_settings_errors("size", _jsonable_sizes()))


@app.get("/api/results/runs")
def result_runs():
    runs = []
    for run_dir in _run_dirs():
        files = _result_json_files(run_dir)
        shrimp_count = 0
        for json_file in files:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                shrimp_count += len(data.get("shrimps", []))
            except (OSError, json.JSONDecodeError):
                pass
        runs.append(
            {
                "name": run_dir.name,
                "mtime": run_dir.stat().st_mtime,
                "source_count": len(files),
                "shrimp_count": shrimp_count,
            }
        )
    return jsonify({"runs": runs})


@app.get("/api/results")
def results():
    return jsonify(_results_for_run(request.args.get("run")))


@app.get("/api/results/export-csv")
def export_csv():
    data = _results_for_run(request.args.get("run"))
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["run", "source_file", "track_id", "frame_idx", "pixel_length", "real_length_mm", "size"])
    for source in data["sources"]:
        for shrimp in source["shrimps"]:
            writer.writerow(
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
    filename = f"shrimp_results_{data['run'] or 'empty'}.csv"
    return Response(
        buffer.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    output_dir = _output_dir().resolve()
    target = (output_dir / filename).resolve()
    try:
        target.relative_to(output_dir)
    except ValueError:
        return jsonify({"error": "Đường dẫn không hợp lệ"}), 400
    return send_from_directory(output_dir, filename)


if __name__ == "__main__":
    _ensure_runtime_dirs()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "3000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Shrimp Measure UI: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
