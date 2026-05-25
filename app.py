"""
app.py

Máy chủ Flask cho giao diện web hệ thống đo chiều dài tôm.

Phạm vi trách nhiệm của Flask:
    - Quản lý file input  : upload, liệt kê, xóa.
    - Điều khiển pipeline : khởi động bằng thread, theo dõi trạng thái, đọc log.
    - Cấu hình hệ thống  : đọc/ghi config qua settings.json.
    - Phân loại kích cỡ  : đọc/ghi SIZE_RANGES và nhãn ngoại cỡ.
    - Kết quả đo         : liệt kê run, lọc, xuất CSV/Excel.
    - Hiệu chuẩn scale   : nạp file 1 cột, tính SCALE bằng y = mx.
    - Phục vụ ảnh debug  : trả file từ OUTPUT_DIR qua /outputs/<path>.

Flask KHÔNG chạy subprocess — pipeline chạy trong threading.Thread gọi main.main().
"""

import csv
import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from settings_loader import clear_settings_errors, get_settings_errors, save_settings


# ============================================================
# Hằng số ứng dụng
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CONFIG_KEYS = [
    "INPUT_DIR", "OUTPUT_DIR", "CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE",
    "SCALE", "CONF_DET", "CONF_SEG", "BBOX_PAD", "TOUCH_THRESHOLD",
    "TARGET_FPS", "CONVEYOR_VERTICAL", "SAVE",
]

BOOL_CONFIG_KEYS = {"CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE", "CONVEYOR_VERTICAL", "SAVE"}

RESULT_EXPORT_HEADERS = [
    "run", "source_file", "track_id", "frame_idx",
    "pixel_length", "real_length_mm", "size",
]

SCALE_IMPORT_MAX_BYTES = 8 * 1024 * 1024


# ============================================================
# Flask
# ============================================================

mimetypes.add_type("text/css; charset=utf-8",              ".css")
mimetypes.add_type("application/javascript; charset=utf-8", ".js")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
app.json.ensure_ascii = False

logging.getLogger("werkzeug").addFilter(
    type("_F", (logging.Filter,), {
        "filter": lambda self, r: "/api/pipeline/status" not in r.getMessage()
    })()
)


# ============================================================
# Trạng thái pipeline (thread-based)
# ============================================================

_pipeline_lock    = threading.Lock()
_pipeline_thread: threading.Thread | None = None
_pipeline_started_at: float | None = None
_pipeline_ended_at:   float | None = None
_pipeline_success:    bool  | None = None   # None = đang chạy


def _pipeline_worker() -> None:
    """Chạy main.main() trong thread riêng, bắt SystemExit."""
    global _pipeline_ended_at, _pipeline_success

    # Đặt lại logger để tạo file log mới ở OUTPUT_DIR hiện tại.
    log = logging.getLogger("pipeline")
    for handler in log.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        log.removeHandler(handler)

    try:
        import main as _main_module
        _main_module.main()
        _pipeline_success = True
    except SystemExit as exc:
        _pipeline_success = (int(exc.code or 0) == 0)
    except Exception:
        _pipeline_success = False
    finally:
        _pipeline_ended_at = time.time()


def _pipeline_running() -> bool:
    return _pipeline_thread is not None and _pipeline_thread.is_alive()


def _pipeline_status() -> dict[str, Any]:
    running = _pipeline_running()
    returncode: int | None = None
    if not running and _pipeline_success is not None:
        returncode = 0 if _pipeline_success else 1
    return {
        "running"    : running,
        "returncode" : returncode,
        "started_at" : _pipeline_started_at,
        "ended_at"   : None if running else _pipeline_ended_at,
    }


# ============================================================
# Tiện ích module / đường dẫn
# ============================================================

def _load_config():
    """Nạp lại config.py, xóa cache lỗi settings trước đó."""
    import importlib.util, time as _t
    clear_settings_errors()
    spec = importlib.util.spec_from_file_location(
        f"_cfg_{_t.time_ns()}", BASE_DIR / "config.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể nạp config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_size():
    """Nạp lại size.py, xóa cache lỗi settings trước đó."""
    import importlib.util, time as _t
    clear_settings_errors()
    spec = importlib.util.spec_from_file_location(
        f"_sz_{_t.time_ns()}", BASE_DIR / "size.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Không thể nạp size.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _workspace_path(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else BASE_DIR / p).resolve()


def _configured_dir(attr: str) -> Path:
    return _workspace_path(str(getattr(_load_config(), attr)))


def _input_dir() -> Path:
    p = _configured_dir("INPUT_DIR")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _output_dir() -> Path:
    p = _configured_dir("OUTPUT_DIR")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path() -> Path:
    return _output_dir() / "pipeline.log"


# ============================================================
# Tiện ích config
# ============================================================

def _jsonable_config() -> dict[str, Any]:
    cfg = _load_config()
    result: dict[str, Any] = {key: getattr(cfg, key) for key in CONFIG_KEYS}
    result["IMG_EXTS"] = sorted(cfg.IMG_EXTS)
    result["VID_EXTS"] = sorted(cfg.VID_EXTS)
    return result


def _with_errors(payload: dict) -> dict:
    errors = get_settings_errors()
    if errors:
        payload["_settings_errors"] = errors
    return payload


def _config_response():
    return jsonify(_with_errors(_jsonable_config()))


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(data: dict, key: str,
                  min_val: float | None = None,
                  max_val: float | None = None) -> float:
    try:
        v = float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số") from exc
    if min_val is not None and v < min_val:
        raise ValueError(f"{key} phải >= {min_val}")
    if max_val is not None and v > max_val:
        raise ValueError(f"{key} phải <= {max_val}")
    return v


def _coerce_int(data: dict, key: str, min_val: int | None = None) -> int:
    try:
        v = int(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} phải là số nguyên") from exc
    if min_val is not None and v < min_val:
        raise ValueError(f"{key} phải >= {min_val}")
    return v


def _validate_config_payload(payload: dict) -> dict:
    """Merge payload với config hiện tại, kiểm tra kiểu và ràng buộc."""
    cfg = _load_config()
    data: dict[str, Any] = {key: getattr(cfg, key) for key in CONFIG_KEYS}
    data.update({k: v for k, v in payload.items() if k in CONFIG_KEYS})

    for key in ("INPUT_DIR", "OUTPUT_DIR"):
        data[key] = str(data.get(key, "")).strip()
        if not data[key]:
            raise ValueError(f"{key} không được để trống")

    for key in BOOL_CONFIG_KEYS:
        data[key] = _normalize_bool(data[key])

    data["SCALE"]           = _coerce_float(data, "SCALE",           min_val=0.00001)
    data["CONF_DET"]        = _coerce_float(data, "CONF_DET",        min_val=0, max_val=1)
    data["CONF_SEG"]        = _coerce_float(data, "CONF_SEG",        min_val=0, max_val=1)
    data["BBOX_PAD"]        = _coerce_int(  data, "BBOX_PAD",        min_val=0)
    data["TOUCH_THRESHOLD"] = _coerce_float(data, "TOUCH_THRESHOLD", min_val=0)
    data["TARGET_FPS"]      = _coerce_float(data, "TARGET_FPS",      min_val=0)
    return data


# ============================================================
# Tiện ích kích cỡ
# ============================================================

def _jsonable_sizes() -> dict[str, Any]:
    s = _load_size()
    return {
        "ranges"         : {label: list(bounds) for label, bounds in s.SIZE_RANGES.items()},
        "undersize_label": s.UNDERSIZE_LABEL,
        "oversize_label" : s.OVERSIZE_LABEL,
        "fallback_label" : s.FALLBACK_LABEL,
    }


def _validate_sizes_payload(payload: dict) -> dict:
    raw = payload.get("ranges")
    if not isinstance(raw, dict):
        raise ValueError("Bảng phân loại phải là object")

    if not raw:
        return {
            "ranges"         : {},
            "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
            "oversize_label" : str(payload.get("oversize_label",  "")).strip() or "Ngoại cỡ lớn",
            "fallback_label" : str(payload.get("fallback_label",  "")).strip() or "Ngoại cỡ",
        }

    ranges: list[tuple[str, float, float]] = []
    labels: set[str] = set()
    for label, bounds in raw.items():
        lbl = str(label).strip()
        if not lbl:
            raise ValueError("Tên cỡ không được trống")
        if lbl in labels:
            raise ValueError(f"Cỡ {lbl} bị trùng")
        labels.add(lbl)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Cỡ {lbl} phải có [từ, đến]")
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cỡ {lbl} phải là số") from exc
        if lo < 0 or hi < 0:
            raise ValueError(f"Cỡ {lbl} phải >= 0")
        if lo >= hi:
            raise ValueError(f"Cỡ {lbl}: mốc từ phải nhỏ hơn mốc đến")
        ranges.append((lbl, lo, hi))

    ranges.sort(key=lambda x: (x[1], x[2]))
    prev_lbl, prev_hi = ranges[0][0], ranges[0][2]
    for lbl, lo, hi in ranges[1:]:
        if lo < prev_hi:
            raise ValueError(f"Cỡ {prev_lbl} và {lbl} đang chồng lấp")
        prev_lbl, prev_hi = lbl, hi

    return {
        "ranges"         : {lbl: (lo, hi) for lbl, lo, hi in ranges},
        "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
        "oversize_label" : str(payload.get("oversize_label",  "")).strip() or "Ngoại cỡ lớn",
        "fallback_label" : str(payload.get("fallback_label",  "")).strip() or "Ngoại cỡ",
    }


# ============================================================
# Tiện ích file input
# ============================================================

def _allowed_suffixes() -> set[str]:
    cfg = _load_config()
    return {s.lower() for s in cfg.IMG_EXTS | cfg.VID_EXTS}


def _safe_filename(filename: str) -> str:
    raw = Path(str(filename).replace("\\", "/")).name.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(" .")
    while safe.startswith("."):
        safe = safe[1:]
    return safe or f"upload_{time.time_ns()}"


def _unique_dest(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for i in range(1, 10_000):
        next_ = directory / f"{stem}_{i}{suffix}"
        if not next_.exists():
            return next_
    return directory / f"{stem}_{time.time_ns()}{suffix}"


def _file_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size,
            "mtime": stat.st_mtime, "suffix": path.suffix.lower()}


# ============================================================
# Tiện ích kết quả đo
# ============================================================

def _run_dirs() -> list[Path]:
    d = _output_dir()
    return sorted(
        [p for p in d.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    ) if d.exists() else []


def _result_json_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*_results.json"))


def _image_url(raw: str | None) -> str | None:
    if not raw:
        return None
    root = _output_dir().resolve()
    p = (Path(raw) if Path(raw).is_absolute() else (BASE_DIR / raw)).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    return f"/outputs/{rel.as_posix()}" if p.exists() else None


def _normalize_images(images: dict | None) -> dict:
    if not images:
        return {}
    out: dict[str, Any] = {}
    for k, v in images.items():
        if isinstance(v, list):
            out[k] = [u for item in v if (u := _image_url(item))]
        else:
            out[k] = _image_url(v)
    return out


def _results_for_run(run_name: str | None = None) -> dict[str, Any]:
    runs = _run_dirs()
    selected: Path | None = None
    if run_name:
        selected = next((r for r in runs if r.name == run_name), None)
    elif runs:
        selected = runs[0]
    if selected is None:
        return {"run": None, "sources": []}

    sources = []
    for jf in _result_json_files(selected):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shrimps = []
        for item in data.get("shrimps", []):
            s = dict(item)
            s["images"] = _normalize_images(s.get("images"))
            shrimps.append(s)
        sources.append({
            "source_file"    : data.get("source_file",   jf.parent.name),
            "source_stem"    : data.get("source_stem",   jf.parent.name),
            "processed_at"   : data.get("processed_at"),
            "scale_mm_per_px": data.get("scale_mm_per_px"),
            "shrimps"        : shrimps,
        })
    return {"run": selected.name, "sources": sources}


# ============================================================
# Tiện ích xuất CSV / Excel
# ============================================================

def _export_rows(data: dict) -> list[list[Any]]:
    rows = [RESULT_EXPORT_HEADERS]
    for src in data["sources"]:
        for s in src["shrimps"]:
            rows.append([data["run"], src["source_file"],
                         s.get("track_id"), s.get("frame_idx"),
                         s.get("pixel_length"), s.get("real_length_mm"), s.get("size")])
    return rows


def _export_filename(data: dict, ext: str) -> str:
    run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data["run"] or "empty"))
    return f"shrimp_results_{run}.{ext}"


def _col_letter(i: int) -> str:
    name = ""
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def _xlsx_cell(val: Any, row: int, col: int) -> str:
    ref = f"{_col_letter(col)}{row}"
    if val is None:
        return f'<c r="{ref}"/>'
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return f'<c r="{ref}"><v>{val}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(str(val))}</t></is></c>'


def _xlsx_bytes(rows: list[list[Any]]) -> bytes:
    sheet_rows = "".join(
        f'<row r="{ri}">' +
        "".join(_xlsx_cell(v, ri, ci) for ci, v in enumerate(row, 1)) +
        "</row>"
        for ri, row in enumerate(rows, 1)
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
        "xl/worksheets/sheet1.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{sheet_rows}</sheetData>"
            "</worksheet>"
        ),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ============================================================
# Tiện ích nạp file scale (1 cột CSV/Excel)
# ============================================================

def _decode_csv(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_csv_table(raw: bytes) -> list[list[str]]:
    text = _decode_csv(raw)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text), dialect)]


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        return [
            "".join(n.text or "" for n in si.iter() if n.tag.endswith("}t") or n.tag == "t")
            for si in root.iter()
            if si.tag.endswith("}si") or si.tag == "si"
        ]
    except KeyError:
        return []


def _xlsx_col_idx(ref: str) -> int:
    m = re.match(r"[A-Z]+", ref.upper())
    if not m:
        return 0
    idx = 0
    for ch in m.group(0):
        idx = idx * 26 + ord(ch) - 64
    return idx - 1


def _xlsx_cell_text(cell: ET.Element, shared: list[str]) -> str:
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        return "".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t") or n.tag == "t")
    vnode = next((n for n in cell if n.tag.endswith("}v") or n.tag == "v"), None)
    raw = vnode.text if vnode is not None else ""
    if ctype == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return ""
    return str(raw or "")


def _parse_xlsx_table(raw: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        try:
            sheet_xml = zf.read("xl/worksheets/sheet1.xml")
        except KeyError as exc:
            raise ValueError("Không tìm thấy sheet1 trong XLSX") from exc
        shared = _xlsx_shared_strings(zf)
        root = ET.fromstring(sheet_xml)
        rows = []
        for row_el in root.iter():
            if not (row_el.tag.endswith("}row") or row_el.tag == "row"):
                continue
            cells: dict[int, str] = {}
            max_col = -1
            for c in row_el:
                if not (c.tag.endswith("}c") or c.tag == "c"):
                    continue
                col = _xlsx_col_idx(c.attrib.get("r", ""))
                cells[col] = _xlsx_cell_text(c, shared).strip()
                max_col = max(max_col, col)
            if max_col >= 0:
                rows.append([cells.get(i, "") for i in range(max_col + 1)])
        return rows


def _table_from_file(raw: bytes, filename: str) -> list[list[str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return _parse_csv_table(raw)
    if suffix == ".xlsx":
        return _parse_xlsx_table(raw)
    raise ValueError("Chỉ hỗ trợ CSV hoặc XLSX")


def _parse_positive_mm(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    v = float(m.group(0))
    return v if v > 0 else None


def _mm_column_values(table: list[list[str]]) -> list[float]:
    """
    Đọc cột đầu tiên của bảng, bỏ qua dòng không phải số dương (header, rỗng).
    Trả về danh sách giá trị mm theo đúng thứ tự dòng.
    """
    values = []
    for row in table:
        if not row:
            continue
        mm = _parse_positive_mm(row[0])
        if mm is not None:
            values.append(mm)
    return values


# ============================================================
# Tiện ích hiệu chuẩn SCALE
# ============================================================

def _calibration_index(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Xây dựng {(source_stem, track_id): {pixel_length, ...}} từ JSON kết quả."""
    index: dict[tuple[str, str], dict] = {}
    for jf in _result_json_files(run_dir):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = str(data.get("source_stem") or jf.parent.name)
        src  = str(data.get("source_file") or stem)
        for s in data.get("shrimps", []):
            tid = str(s.get("track_id"))
            index[(stem, tid)] = {
                "source_stem" : stem,
                "source_file" : src,
                "track_id"    : s.get("track_id"),
                "pixel_length": s.get("pixel_length"),
            }
    return index


def _least_squares_scale(samples: list[dict]) -> tuple[float, float, list[dict]]:
    """
    Hồi quy tuyến tính qua gốc: real_mm = scale * pixel.
    SCALE = sum(pixel * real) / sum(pixel^2).
    Trả về (scale, rmse_mm, samples_enriched).
    """
    if not samples:
        raise ValueError("Cần ít nhất 1 mẫu hợp lệ")
    sx2 = sum(s["pixel_length"] ** 2 for s in samples)
    sxy = sum(s["pixel_length"] * s["real_length_mm"] for s in samples)
    if sx2 == 0:
        raise ValueError("pixel_length phải lớn hơn 0")
    scale = sxy / sx2
    enriched, sq_errors = [], []
    for s in samples:
        fitted   = s["pixel_length"] * scale
        residual = s["real_length_mm"] - fitted
        sq_errors.append(residual ** 2)
        enriched.append({**s,
                         "fitted_mm"  : round(fitted,   6),
                         "residual_mm": round(residual, 6)})
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5
    return scale, rmse, enriched


# ============================================================
# Route: trang chủ
# ============================================================

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "status": _pipeline_status()})


# ============================================================
# Route: quản lý file input
# ============================================================

@app.get("/api/files/input")
def list_input_files():
    files = [_file_payload(p) for p in sorted(_input_dir().iterdir()) if p.is_file()]
    return jsonify({"files": files})


@app.post("/api/files/upload")
def upload_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Chưa chọn file"}), 400
    allowed   = _allowed_suffixes()
    input_dir = _input_dir()
    saved, rejected = [], []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in allowed:
            rejected.append({"name": f.filename, "reason": "Định dạng không hỗ trợ"})
            continue
        dest = _unique_dest(input_dir, _safe_filename(f.filename or ""))
        f.save(dest)
        saved.append(_file_payload(dest))
    return jsonify({"saved": saved, "rejected": rejected})


@app.delete("/api/files/input/<path:filename>")
def delete_input_file(filename: str):
    root   = _input_dir().resolve()
    target = (root / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    if target.is_file():
        target.unlink()
    return jsonify({"ok": True})


# ============================================================
# Route: điều khiển pipeline
# ============================================================

@app.post("/api/pipeline/run")
def run_pipeline():
    global _pipeline_thread, _pipeline_started_at, _pipeline_ended_at, _pipeline_success
    with _pipeline_lock:
        if _pipeline_running():
            return jsonify({"error": "Pipeline đang chạy"}), 409
        _output_dir().mkdir(parents=True, exist_ok=True)
        _log_path().write_text("", encoding="utf-8")
        _pipeline_started_at = time.time()
        _pipeline_ended_at   = None
        _pipeline_success    = None
        _pipeline_thread = threading.Thread(
            target=_pipeline_worker, daemon=True, name="pipeline"
        )
        _pipeline_thread.start()
    return jsonify({"ok": True, "status": _pipeline_status()})


@app.get("/api/pipeline/status")
def pipeline_status():
    return jsonify(_pipeline_status())


@app.get("/api/pipeline/log")
def pipeline_log():
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        offset = 0
    path = _log_path()
    if not path.exists():
        return jsonify({"content": "", "offset": 0, "size": 0})
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
        next_offset = f.tell()
    return jsonify({"content": data.decode("utf-8", errors="replace"),
                    "offset": next_offset, "size": size})


# ============================================================
# Route: cấu hình hệ thống
# ============================================================

@app.get("/api/config")
def get_config():
    return _config_response()


@app.put("/api/config")
def put_config():
    try:
        data = _validate_config_payload(request.get_json(force=True, silent=True) or {})
        save_settings("config", {k: data[k] for k in CONFIG_KEYS})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _config_response()


# ============================================================
# Route: bảng kích cỡ
# ============================================================

@app.get("/api/config/sizes")
def get_sizes():
    return jsonify(_with_errors(_jsonable_sizes()))


@app.put("/api/config/sizes")
def put_sizes():
    try:
        data = _validate_sizes_payload(request.get_json(force=True, silent=True) or {})
        save_settings("size", {
            "SIZE_RANGES"    : data["ranges"],
            "UNDERSIZE_LABEL": data["undersize_label"],
            "OVERSIZE_LABEL" : data["oversize_label"],
            "FALLBACK_LABEL" : data["fallback_label"],
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_with_errors(_jsonable_sizes()))


# ============================================================
# Route: kết quả đo
# ============================================================

@app.get("/api/results/runs")
def result_runs():
    runs = []
    for rd in _run_dirs():
        count = 0
        for jf in _result_json_files(rd):
            try:
                count += len(json.loads(jf.read_text(encoding="utf-8")).get("shrimps", []))
            except (OSError, json.JSONDecodeError):
                pass
        runs.append({
            "name"        : rd.name,
            "mtime"       : rd.stat().st_mtime,
            "source_count": len(_result_json_files(rd)),
            "shrimp_count": count,
        })
    return jsonify({"runs": runs})


@app.get("/api/results")
def results():
    return jsonify(_results_for_run(request.args.get("run")))


@app.get("/api/results/export-csv")
def export_csv():
    data   = _results_for_run(request.args.get("run"))
    buf    = io.StringIO()
    csv.writer(buf).writerows(_export_rows(data))
    return Response(
        buf.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'csv')}"},
    )


@app.get("/api/results/export-excel")
def export_excel():
    data = _results_for_run(request.args.get("run"))
    return Response(
        _xlsx_bytes(_export_rows(data)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'xlsx')}"},
    )


# ============================================================
# Route: hiệu chuẩn scale
# ============================================================

@app.post("/api/calibrate/import-measurements")
def import_scale_measurements():
    """
    Nhận file CSV/Excel 1 cột giá trị mm thực tế (theo đúng thứ tự tôm trên màn hình).
    Header nếu có sẽ bị bỏ qua vì không phải số.
    """
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Chưa chọn file scale"}), 400

    run_name = str(request.form.get("run") or "").strip()
    if not run_name:
        return jsonify({"error": "Chưa chọn run"}), 400

    try:
        ordered_rows: list[dict] = json.loads(request.form.get("rows") or "[]")
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Danh sách dòng không hợp lệ: {exc}"}), 400
    if not ordered_rows:
        return jsonify({"error": "Không có dòng kết quả để ghép"}), 400

    raw = file.read(SCALE_IMPORT_MAX_BYTES + 1)
    if len(raw) > SCALE_IMPORT_MAX_BYTES:
        return jsonify({"error": "File vượt quá giới hạn 8 MB"}), 400

    try:
        table     = _table_from_file(raw, file.filename)
        mm_values = _mm_column_values(table)
    except (ValueError, zipfile.BadZipFile, ET.ParseError, OSError) as exc:
        return jsonify({"error": f"Không đọc được file: {exc}"}), 400

    if not mm_values:
        return jsonify({"error": "Không tìm thấy giá trị mm hợp lệ trong cột đầu tiên"}), 400

    measurements = [
        {
            "source_file"   : row.get("source_file", ""),
            "source_stem"   : row.get("source_stem", ""),
            "track_id"      : row.get("track_id",    ""),
            "real_length_mm": mm,
        }
        for row, mm in zip(ordered_rows, mm_values)
    ]

    warnings = []
    if len(mm_values) < len(ordered_rows):
        warnings.append(
            f"File có {len(mm_values)} giá trị, màn hình có {len(ordered_rows)} tôm "
            f"— chỉ điền {len(measurements)} dòng đầu."
        )

    return jsonify({
        "measurements"  : measurements,
        "count"         : len(measurements),
        "expected_count": len(ordered_rows),
        "warnings"      : warnings,
    })


@app.post("/api/calibrate")
def calibrate_scale():
    """
    Tính SCALE mới từ danh sách {source_stem, track_id, real_length_mm}.
    Dùng hồi quy bình phương tối thiểu qua gốc: real_mm = SCALE * pixel.
    """
    payload      = request.get_json(force=True, silent=True) or {}
    run_name     = str(payload.get("run") or "").strip()
    measurements = payload.get("measurements")

    if not run_name:
        return jsonify({"error": "Chưa chọn run"}), 400
    if not isinstance(measurements, list) or not measurements:
        return jsonify({"error": "Chưa nhập mm thực tế"}), 400

    run_dir = next((r for r in _run_dirs() if r.name == run_name), None)
    if run_dir is None:
        return jsonify({"error": f"Không tìm thấy run {run_name}"}), 404

    index   = _calibration_index(run_dir)
    samples, errors = [], []

    for item in measurements:
        if not isinstance(item, dict):
            continue
        stem = str(item.get("source_stem") or "").strip()
        tid  = str(item.get("track_id")    or "").strip()
        try:
            real_mm = float(item["real_length_mm"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{stem} ID {tid}: mm không hợp lệ")
            continue
        if real_mm <= 0:
            errors.append(f"{stem} ID {tid}: mm phải > 0")
            continue
        rec = index.get((stem, tid))
        if rec is None:
            errors.append(f"{stem} ID {tid}: không có trong JSON kết quả")
            continue
        try:
            px = float(rec["pixel_length"])
        except (TypeError, ValueError):
            errors.append(f"{stem} ID {tid}: pixel_length không hợp lệ")
            continue
        if px <= 0:
            errors.append(f"{stem} ID {tid}: pixel_length phải > 0")
            continue
        samples.append({
            "source_stem"   : stem,
            "source_file"   : rec["source_file"],
            "track_id"      : rec["track_id"],
            "pixel_length"  : px,
            "real_length_mm": real_mm,
        })

    if not samples:
        msg = "Không có mẫu hợp lệ" + (": " + "; ".join(errors[:3]) if errors else "")
        return jsonify({"error": msg, "errors": errors}), 400

    try:
        new_scale, rmse, samples = _least_squares_scale(samples)
    except ValueError as exc:
        return jsonify({"error": str(exc), "errors": errors}), 400

    data = _validate_config_payload({"SCALE": round(new_scale, 6)})
    save_settings("config", {k: data[k] for k in CONFIG_KEYS})

    return jsonify({
        "scale"  : data["SCALE"],
        "method" : "least_squares_origin",
        "formula": "real_mm = scale * pixel",
        "rmse_mm": round(rmse, 6),
        "count"  : len(samples),
        "samples": samples,
        "errors" : errors,
        "config" : _with_errors(_jsonable_config()),
    })


# ============================================================
# Route: phục vụ ảnh debug
# ============================================================

@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    root   = _output_dir().resolve()
    target = (root / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return jsonify({"error": "Đường dẫn không hợp lệ"}), 400
    return send_from_directory(root, filename)


# ============================================================
# Khởi động
# ============================================================

if __name__ == "__main__":
    host  = os.environ.get("HOST", "127.0.0.1")
    port  = int(os.environ.get("PORT", "3000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Shrimp Measure UI: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)