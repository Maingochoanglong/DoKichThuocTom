"""
app.py

Giao diện web Flask cho hệ thống đo tôm.
app.py dùng chung settings_loader, config và size với pipeline vì các module
này đều nằm cùng cấp thư mục dự án.
"""

import csv
import io
import json
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
from openpyxl import Workbook, load_workbook

from config import load_config_values
from settings_loader import pull_setting_warnings, read_setting, save_setting
from size import load_size_values


BASE_DIR = Path(__file__).resolve().parent

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

# Các key cho phép chỉnh qua giao diện web.
CONFIG_KEYS = [
    "INPUT_DIR", "OUTPUT_DIR", "CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE",
    "SCALE", "CONF_DET", "CONF_SEG", "BBOX_PAD", "TOUCH_THRESHOLD",
    "TARGET_FPS", "CONVEYOR_VERTICAL", "SAVE",
]
BOOL_KEYS = {"CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE", "CONVEYOR_VERTICAL", "SAVE"}

RESULT_COLS    = ["run", "source_file", "track_id", "frame_idx", "pixel_length", "real_length_mm", "size"]
MAX_SCALE_BYTE = 4 * 1024 * 1024   # 4 MB giới hạn file scale import


def _get_config() -> dict:
    """Lấy config qua config.load_config_values(), nơi đọc settings bằng settings_loader."""
    return load_config_values()


def _save_config(values: dict) -> None:
    """
    Lưu config qua settings_loader.
    Giữ IMG_EXTS, VID_EXTS và key người dùng thêm tay trong section config.
    """
    current = read_setting().get("config", {})
    if not isinstance(current, dict):
        current = {}
    merged  = {**current, **{k: values[k] for k in CONFIG_KEYS if k in values}}
    save_setting("config", merged)


def _get_sizes() -> dict:
    """Lấy bảng kích cỡ qua size.load_size_values(), nơi đọc settings bằng settings_loader."""
    values = load_size_values()
    return {
        "ranges": {k: list(v) for k, v in values["SIZE_RANGES"].items()},
        "undersize_label": values["UNDERSIZE_LABEL"],
        "oversize_label":  values["OVERSIZE_LABEL"],
        "fallback_label":  values["FALLBACK_LABEL"],
    }


# Kiểm tra dữ liệu đầu vào 

def _validate_config(raw: dict) -> dict:
    """Kiểm tra và chuẩn hóa payload config từ client."""
    current = _get_config()
    data    = {**current, **{k: raw[k] for k in CONFIG_KEYS if k in raw}}

    for k in ["INPUT_DIR", "OUTPUT_DIR"]:
        v = str(data.get(k, "")).strip()
        if not v:
            raise ValueError(f"{k} không được để trống")
        data[k] = v

    for k in BOOL_KEYS:
        v = data[k]
        data[k] = (v.strip().lower() in {"1", "true", "yes", "on"}
                   if isinstance(v, str) else bool(v))

    for k, lo, hi in [
        ("SCALE",           1e-5, None),
        ("CONF_DET",        0.0,  1.0),
        ("CONF_SEG",        0.0,  1.0),
        ("TOUCH_THRESHOLD", 0.0,  None),
        ("TARGET_FPS",      0.0,  None),
    ]:
        try:
            v = float(data[k])
        except Exception:
            raise ValueError(f"{k} phải là số thực")
        if lo is not None and v < lo:
            raise ValueError(f"{k} phải >= {lo}")
        if hi is not None and v > hi:
            raise ValueError(f"{k} phải <= {hi}")
        data[k] = v

    try:
        v = int(data["BBOX_PAD"])
        if v < 0:
            raise ValueError
        data["BBOX_PAD"] = v
    except (ValueError, TypeError):
        raise ValueError("BBOX_PAD phải là số nguyên không âm")

    return data


def _validate_sizes(raw: dict) -> dict:
    """Kiểm tra và chuẩn hóa payload bảng kích cỡ từ client."""
    ranges_raw = raw.get("ranges", {})
    if not isinstance(ranges_raw, dict):
        raise ValueError("ranges phải là object")

    result: list[tuple[str, float, float]] = []
    labels: set[str] = set()

    for lbl, bounds in ranges_raw.items():
        lbl = str(lbl).strip()
        if not lbl:
            raise ValueError("Tên cỡ không được để trống")
        if lbl in labels:
            raise ValueError(f"Cỡ '{lbl}' bị trùng")
        labels.add(lbl)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Cỡ '{lbl}' cần [từ, đến]")
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except Exception:
            raise ValueError(f"Cỡ '{lbl}': giá trị phải là số")
        if lo < 0 or hi < 0:
            raise ValueError(f"Cỡ '{lbl}': không được âm")
        if lo >= hi:
            raise ValueError(f"Cỡ '{lbl}': mốc đầu phải nhỏ hơn mốc cuối")
        result.append((lbl, lo, hi))

    result.sort(key=lambda x: x[1])
    for i in range(1, len(result)):
        a, _, a_hi = result[i - 1]
        b, b_lo, _ = result[i]
        if b_lo < a_hi:
            raise ValueError(f"Cỡ '{a}' và '{b}' bị chồng lấp")

    return {
        "ranges":          {lbl: [lo, hi] for lbl, lo, hi in result},
        "undersize_label": str(raw.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
        "oversize_label":  str(raw.get("oversize_label",  "")).strip() or "Ngoại cỡ lớn",
        "fallback_label":  str(raw.get("fallback_label",  "")).strip() or "Ngoại cỡ",
    }


# Tiện ích đường dẫn 

def _abs(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()


def _input_dir() -> Path:
    p = _abs(_get_config()["INPUT_DIR"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _output_dir() -> Path:
    p = _abs(_get_config()["OUTPUT_DIR"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path() -> Path:
    return _output_dir() / "pipeline.log"


# Tiện ích file input

def _allowed_ext() -> set[str]:
    cfg = _get_config()
    return {e.lower() for e in [*cfg["IMG_EXTS"], *cfg["VID_EXTS"]]}


def _safe_name(filename: str) -> str:
    name = Path(str(filename).replace("\\", "/")).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip(" .")
    return name or f"upload_{time.time_ns()}"


def _unique_dest(directory: Path, name: str) -> Path:
    p = directory / name
    stem, suffix, i = Path(name).stem, Path(name).suffix, 1
    while p.exists():
        p = directory / f"{stem}_{i}{suffix}"
        i += 1
    return p


def _file_info(p: Path) -> dict:
    s = p.stat()
    return {"name": p.name, "size": s.st_size, "mtime": s.st_mtime, "suffix": p.suffix.lower()}


# Trạng thái pipeline

_lock:    threading.Lock       = threading.Lock()
_proc:    subprocess.Popen | None = None
_running: bool                 = False
_t_start: float | None        = None
_t_end:   float | None        = None
_retcode: int | None          = None   # 0 = thành công, != 0 = lỗi


def _watch_pipeline(proc: subprocess.Popen) -> None:
    """Chạy trong thread riêng, chờ subprocess kết thúc và cập nhật trạng thái."""
    global _running, _t_end, _retcode
    code = proc.wait()
    with _lock:
        _retcode = code
        _running = False
        _t_end   = time.time()


def _status() -> dict:
    return {
        "running":    _running,
        "returncode": None if _running else _retcode,
        "started_at": _t_start,
        "ended_at":   None if _running else _t_end,
    }


# Tiện ích kết quả

def _run_dirs() -> list[Path]:
    d = _output_dir()
    return sorted(
        [p for p in d.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _json_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*_results.json"))


def _selected_run_dir(run_name: str | None) -> Path | None:
    runs = _run_dirs()
    if run_name:
        return next((r for r in runs if r.name == run_name), None)
    return runs[0] if runs else None


def _read_result_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _image_url(raw: str | None) -> str | None:
    if not raw:
        return None
    root = _output_dir().resolve()
    p    = Path(raw)
    p    = p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    return f"/outputs/{rel.as_posix()}" if p.exists() else None


def _norm_images(images: dict | None) -> dict:
    if not images:
        return {}
    return {
        k: [u for x in v if (u := _image_url(x))] if isinstance(v, list) else _image_url(v)
        for k, v in images.items()
    }


def _results_for_run(run_name: str | None) -> dict:
    sel = _selected_run_dir(run_name)
    if sel is None:
        return {"run": None, "sources": []}

    sources = []
    for jf in _json_files(sel):
        d = _read_result_json(jf)
        if d is None:
            continue
        sources.append({
            "source_file":     d.get("source_file",    jf.parent.name),
            "source_stem":     d.get("source_stem",    jf.parent.name),
            "processed_at":    d.get("processed_at"),
            "scale_mm_per_px": d.get("scale_mm_per_px"),
            "shrimps": [
                dict(s, images=_norm_images(s.get("images")))
                for s in d.get("shrimps", [])
            ],
        })
    return {"run": sel.name, "sources": sources}


def _export_rows(data: dict) -> list[list]:
    rows = [RESULT_COLS]
    for src in data["sources"]:
        for s in src["shrimps"]:
            rows.append([
                data["run"], src["source_file"],
                s.get("track_id"), s.get("frame_idx"),
                s.get("pixel_length"), s.get("real_length_mm"), s.get("size"),
            ])
    return rows


# Excel 

def _xlsx_bytes(rows: list[list]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# Đọc file scale (1 cột mm, khớp tuần tự) 

def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_positive(cell: Any) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(cell or "").strip())
    if not m:
        return None
    n = float(m.group().replace(",", "."))
    return n if n > 0 else None


def _read_col_csv(raw: bytes) -> list[float]:
    """Đọc cột đầu tiên của CSV, bỏ qua dòng header không phải số."""
    text = _decode_bytes(raw)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [
        n
        for row in csv.reader(io.StringIO(text), dialect)
        if row and (n := _parse_positive(row[0])) is not None
    ]


def _read_col_xlsx(raw: bytes) -> list[float]:
    """Đọc cột đầu tiên của XLSX, bỏ qua dòng header không phải số."""
    try:
        wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception as e:
        raise ValueError("File XLSX không hợp lệ") from e
    try:
        ws = wb.active
        if ws is None:
            raise ValueError("Không tìm thấy sheet trong XLSX")
        values = []
        for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
            n = _parse_positive(row[0] if row else None)
            if n is not None:
                values.append(n)
        return values
    finally:
        wb.close()


# Routes 

@app.get("/")
def index():
    return render_template("index.html")



# File input 

@app.get("/api/files/input")
def list_input():
    d = _input_dir()
    return jsonify({
        "files": [_file_info(p) for p in sorted(d.iterdir()) if p.is_file()]
    })


@app.post("/api/files/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Chưa chọn file"}), 400
    allowed  = _allowed_ext()
    d        = _input_dir()
    saved, rejected = [], []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in allowed:
            rejected.append({"name": f.filename, "reason": "Định dạng không hỗ trợ"})
            continue
        dest = _unique_dest(d, _safe_name(f.filename))
        f.save(dest)
        saved.append(_file_info(dest))
    return jsonify({"saved": saved, "rejected": rejected})


@app.delete("/api/files/input/<path:filename>")
def delete_input(filename: str):
    d = _input_dir().resolve()
    t = (d / filename).resolve()
    try:
        t.relative_to(d)
    except ValueError:
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    if t.is_file():
        t.unlink()
    return jsonify({"ok": True})


# Pipeline

@app.post("/api/pipeline/run")
def run_pipeline():
    """
    Kích hoạt backend bằng subprocess.
    main.py sẽ tự đọc config từ settings.json khi khởi chạy.
    Không truyền bất kỳ tham số nào qua Python API.
    """
    global _proc, _running, _t_start, _t_end, _retcode
    warnings: list[str] = []
    with _lock:
        if _running:
            return jsonify({"error": "Pipeline đang chạy"}), 409
        _get_config()
        _get_sizes()
        warnings = pull_setting_warnings()
        _output_dir().mkdir(parents=True, exist_ok=True)
        _log_path().write_text("", encoding="utf-8")
        _t_start, _t_end, _retcode = time.time(), None, None
        _running = True
        _proc    = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "main.py")],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        threading.Thread(
            target=_watch_pipeline,
            args=(_proc,),
            daemon=True,
            name="pipeline-watcher",
        ).start()
    return jsonify({"ok": True, "status": _status(), "warnings": warnings})


@app.get("/api/pipeline/status")
def pipeline_status():
    return jsonify(_status())


@app.get("/api/pipeline/log")
def pipeline_log():
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        offset = 0
    p = _log_path()
    if not p.exists():
        return jsonify({"content": "", "offset": 0, "size": 0})
    size = p.stat().st_size
    if offset > size:
        offset = 0
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read()
        nxt  = f.tell()
    return jsonify({
        "content": data.decode("utf-8", errors="replace"),
        "offset":  nxt,
        "size":    size,
    })


# Cấu hình

@app.get("/api/config")
def get_config():
    return jsonify({
        "config": _get_config(),
        "warnings": pull_setting_warnings(),
    })


@app.put("/api/config")
def put_config():
    try:
        data = _validate_config(request.get_json(force=True, silent=True) or {})
        _save_config(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "config": _get_config(),
        "warnings": pull_setting_warnings(),
    })


# Kích cỡ 

@app.get("/api/config/sizes")
def get_sizes():
    return jsonify({
        "sizes": _get_sizes(),
        "warnings": pull_setting_warnings(),
    })


@app.put("/api/config/sizes")
def put_sizes():
    try:
        data = _validate_sizes(request.get_json(force=True, silent=True) or {})
        save_setting("size", {
            "SIZE_RANGES":     data["ranges"],
            "UNDERSIZE_LABEL": data["undersize_label"],
            "OVERSIZE_LABEL":  data["oversize_label"],
            "FALLBACK_LABEL":  data["fallback_label"],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "sizes": _get_sizes(),
        "warnings": pull_setting_warnings(),
    })


# Kết quả 

@app.get("/api/results/runs")
def result_runs():
    runs = []
    for rd in _run_dirs():
        count = 0
        for jf in _json_files(rd):
            d = _read_result_json(jf)
            if d is not None:
                count += len(d.get("shrimps", []))
        runs.append({
            "name":         rd.name,
            "mtime":        rd.stat().st_mtime,
            "shrimp_count": count,
        })
    return jsonify({"runs": runs})


@app.get("/api/results")
def results():
    return jsonify(_results_for_run(request.args.get("run")))


@app.get("/api/results/export-csv")
def export_csv():
    data = _results_for_run(request.args.get("run"))
    buf  = io.StringIO()
    csv.writer(buf).writerows(_export_rows(data))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data["run"] or "empty"))
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=shrimp_{name}.csv"},
    )


@app.get("/api/results/export-excel")
def export_excel():
    data = _results_for_run(request.args.get("run"))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data["run"] or "empty"))
    return Response(
        _xlsx_bytes(_export_rows(data)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=shrimp_{name}.xlsx"},
    )


# Hiệu chuẩn scale 

@app.post("/api/calibrate/import-measurements")
def import_measurements():
    """
    Nhận file CSV hoặc XLSX có 1 cột giá trị mm.
    Khớp tuần tự với danh sách tôm đang hiển thị trên màn hình.
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Chưa chọn file"}), 400
    run_name = str(request.form.get("run") or "").strip()
    if not run_name:
        return jsonify({"error": "Chưa chọn run"}), 400
    try:
        ordered = json.loads(request.form.get("rows") or "[]")
    except Exception:
        return jsonify({"error": "Danh sách tôm không hợp lệ"}), 400
    if not isinstance(ordered, list) or not ordered:
        return jsonify({"error": "Danh sách tôm trống"}), 400

    raw = f.read(MAX_SCALE_BYTE + 1)
    if len(raw) > MAX_SCALE_BYTE:
        return jsonify({"error": "File vượt quá 4 MB"}), 400

    suffix = Path(f.filename).suffix.lower()
    try:
        if suffix == ".csv":
            mm_values = _read_col_csv(raw)
        elif suffix == ".xlsx":
            mm_values = _read_col_xlsx(raw)
        else:
            return jsonify({"error": "Chỉ hỗ trợ CSV hoặc XLSX"}), 400
    except Exception as e:
        return jsonify({"error": f"Không đọc được file: {e}"}), 400

    if not mm_values:
        return jsonify({"error": "Không tìm thấy giá trị mm hợp lệ trong file"}), 400

    measurements = [
        {
            "source_file":    str(row.get("source_file", "")),
            "source_stem":    str(row.get("source_stem", "")),
            "track_id":       str(row.get("track_id",    "")),
            "real_length_mm": mm,
        }
        for row, mm in zip(ordered, mm_values)
        if isinstance(row, dict)
    ]
    warnings = []
    if len(mm_values) != len(ordered):
        warnings.append(
            f"File có {len(mm_values)} giá trị, danh sách có {len(ordered)} tôm. "
            f"Đã điền {len(measurements)} dòng."
        )
    return jsonify({
        "measurements":   measurements,
        "count":          len(measurements),
        "expected_count": len(ordered),
        "warnings":       warnings,
    })


@app.post("/api/calibrate")
def calibrate():
    """
    Tính SCALE bằng hồi quy tuyến tính qua gốc tọa độ:
        real_length_mm = SCALE x pixel_length
        SCALE = sum(pixel * real) / sum(pixel^2)
    Lưu SCALE vào settings.json qua settings_loader.
    """
    payload  = request.get_json(force=True, silent=True) or {}
    run_name = str(payload.get("run") or "").strip()
    meas     = payload.get("measurements")
    if not run_name:
        return jsonify({"error": "Chưa chọn run"}), 400
    if not isinstance(meas, list) or not meas:
        return jsonify({"error": "Chưa có dữ liệu đo thực tế"}), 400

    run_dir = _selected_run_dir(run_name)
    if run_dir is None:
        return jsonify({"error": f"Không tìm thấy run '{run_name}'"}), 404

    # Bảng tra pixel_length từ file JSON kết quả của backend
    px_index: dict[tuple[str, str], float] = {}
    for jf in _json_files(run_dir):
        d = _read_result_json(jf)
        if d is None:
            continue
        stem = str(d.get("source_stem") or jf.parent.name)
        for s in d.get("shrimps", []):
            px_index[(stem, str(s.get("track_id")))] = float(s.get("pixel_length", 0))

    samples, errors = [], []
    for item in meas:
        if not isinstance(item, dict):
            continue
        stem = str(item.get("source_stem") or "").strip()
        tid  = str(item.get("track_id")    or "").strip()
        try:
            real_mm = float(item["real_length_mm"])
        except Exception:
            errors.append(f"{stem} ID {tid}: mm không hợp lệ")
            continue
        if real_mm <= 0:
            errors.append(f"{stem} ID {tid}: mm phải > 0")
            continue
        px = px_index.get((stem, tid))
        if px is None or px <= 0:
            errors.append(f"{stem} ID {tid}: không tìm thấy pixel_length hợp lệ")
            continue
        samples.append({
            "source_stem":    stem,
            "track_id":       tid,
            "pixel_length":   px,
            "real_length_mm": real_mm,
        })

    if not samples:
        return jsonify({"error": "Không có mẫu hợp lệ", "errors": errors}), 400

    sum_xx = sum(s["pixel_length"] ** 2 for s in samples)
    sum_xy = sum(s["pixel_length"] * s["real_length_mm"] for s in samples)
    if sum_xx == 0:
        return jsonify({"error": "pixel_length bằng 0"}), 400

    scale = sum_xy / sum_xx
    rmse  = (
        sum((s["pixel_length"] * scale - s["real_length_mm"]) ** 2 for s in samples)
        / len(samples)
    ) ** 0.5

    try:
        data = _validate_config({"SCALE": round(scale, 6)})
        _save_config(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "scale":   round(scale, 6),
        "rmse_mm": round(rmse,  6),
        "count":   len(samples),
        "errors":  errors,
        "config":  _get_config(),
    })


# Phục vụ ảnh debug 

@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    root   = _output_dir().resolve()
    target = (root/filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return jsonify({"error": "Đường dẫn không hợp lệ"}), 400
    return send_from_directory(root, filename)


# Khởi động

if __name__ == "__main__":
    host  = os.environ.get("HOST","127.0.0.1")
    port  = int(os.environ.get("PORT","3000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Shrimp Measure UI: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
