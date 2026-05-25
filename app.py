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


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
SIZE_PATH = BASE_DIR / "size.py"
MAIN_PATH = BASE_DIR / "main.py"

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
INTERNAL_CONFIG_KEYS = ["MODEL_DET", "MODEL_SEG"]
BOOL_CONFIG_KEYS = ["CLEAR_OUTPUT", "CLEAR_INPUT", "CHUNK_MODE", "CONVEYOR_VERTICAL", "SAVE"]
RESULT_EXPORT_HEADERS = ["run", "source_file", "track_id", "frame_idx", "pixel_length", "real_length_mm", "size"]
SCALE_IMPORT_MAX_BYTES = 8 * 1024 * 1024
SCALE_MM_COLUMNS = {"mm", "real_mm", "real_length_mm", "length_mm", "ground_truth_mm", "actual_mm", "do_dai_mm"}
SCALE_SOURCE_STEM_COLUMNS = {"source_stem", "stem", "ten_nguon", "nguon"}
SCALE_SOURCE_FILE_COLUMNS = {"source_file", "file", "filename", "ten_file"}
SCALE_TRACK_COLUMNS = {"track_id", "id", "shrimp_id", "tom_id"}


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
    clear_settings_errors()
    return _load_module(CONFIG_PATH, "config")


def _load_size():
    clear_settings_errors()
    return _load_module(SIZE_PATH, "size")


def _workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (BASE_DIR / path).resolve()
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


def _config_values(include_internal: bool = False) -> dict[str, Any]:
    cfg = _load_config()
    keys = CONFIG_KEYS + (INTERNAL_CONFIG_KEYS if include_internal else [])
    return {key: getattr(cfg, key) for key in keys}


def _jsonable_config() -> dict[str, Any]:
    return _config_values(include_internal=False)


def _with_settings_errors(payload: dict[str, Any]) -> dict[str, Any]:
    errors = get_settings_errors()
    if errors:
        payload["_settings_errors"] = errors
    return payload


def _config_response():
    return jsonify(_with_settings_errors(_jsonable_config()))


def _sizes_response():
    return jsonify(_with_settings_errors(_jsonable_sizes()))


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
    save_settings("config", {key: data[key] for key in CONFIG_KEYS})


def _validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
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

    data["SCALE"] = _coerce_float(data, "SCALE", min_value=0.00001)
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
    if not isinstance(raw_ranges, dict):
        raise ValueError("Bảng phân loại phải là object")
    if not raw_ranges:
        return {
            "ranges": {},
            "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
            "oversize_label": str(payload.get("oversize_label", "")).strip() or "Ngoại cỡ lớn",
            "fallback_label": str(payload.get("fallback_label", "")).strip() or "Ngoại cỡ",
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
        "ranges": {label: (lo, hi) for label, lo, hi in ranges},
        "undersize_label": str(payload.get("undersize_label", "")).strip() or "Ngoại cỡ nhỏ",
        "oversize_label": str(payload.get("oversize_label", "")).strip() or "Ngoại cỡ lớn",
        "fallback_label": str(payload.get("fallback_label", "")).strip() or "Ngoại cỡ",
    }


def _write_sizes(data: dict[str, Any]) -> None:
    save_settings(
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


def _pick_local_folder(initial: str = "") -> Path | None:
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
            urls = []
            for item in value:
                url = _image_url(item)
                if url:
                    urls.append(url)
            normalized[key] = urls
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


def _result_export_rows(data: dict[str, Any]) -> list[list[Any]]:
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
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(data["run"] or "empty"))
    return f"shrimp_results_{run_name}.{suffix}"


def _excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value: Any, row_index: int, column_index: int) -> str:
    cell_ref = f"{_excel_column_name(column_index)}{row_index}"
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{xml_escape(str(value))}</t></is></c>'


def _xlsx_bytes(rows: list[list[Any]]) -> bytes:
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, row_index, column_index) for column_index, value in enumerate(row, start=1))
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


def _fold_column_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def _parse_positive_mm(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    return number if number > 0 else None


def _decode_csv_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_csv_table(raw: bytes) -> list[list[str]]:
    text = _decode_csv_bytes(raw)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), dialect)]


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw_xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw_xml)
    strings = []
    for item in root.iter():
        if item.tag.endswith("}si") or item.tag == "si":
            strings.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t") or node.tag == "t"))
    return strings


def _xlsx_column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0
    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t") or node.tag == "t")

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
                col_index = _xlsx_column_index(cell.attrib.get("r", ""))
                cells[col_index] = _xlsx_cell_text(cell, shared_strings).strip()
                max_index = max(max_index, col_index)
            if max_index >= 0:
                rows.append([cells.get(index, "") for index in range(max_index + 1)])
        return rows


def _scale_import_table(raw: bytes, filename: str) -> list[list[str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return _parse_csv_table(raw)
    if suffix == ".xlsx":
        return _parse_xlsx_table(raw)
    raise ValueError("Chỉ hỗ trợ file CSV hoặc XLSX")


def _header_index(header: list[str], aliases: set[str]) -> int | None:
    folded = [_fold_column_name(cell) for cell in header]
    for index, name in enumerate(folded):
        if name in aliases:
            return index
    return None


def _normalize_track_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _scale_import_records(table: list[list[str]]) -> list[dict[str, Any]]:
    rows = [row for row in table if any(str(cell).strip() for cell in row)]
    if not rows:
        return []

    header = rows[0]
    mm_col = _header_index(header, SCALE_MM_COLUMNS)
    source_stem_col = _header_index(header, SCALE_SOURCE_STEM_COLUMNS)
    source_file_col = _header_index(header, SCALE_SOURCE_FILE_COLUMNS)
    track_col = _header_index(header, SCALE_TRACK_COLUMNS)
    has_header = mm_col is not None

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
    warnings = []
    measurements_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_by_key = {
        (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip()): row
        for row in ordered_rows
    }

    sequential_records = []
    for record in records:
        source_stem = str(record.get("source_stem") or "").strip()
        source_file = str(record.get("source_file") or "").strip()
        track_id = str(record.get("track_id") or "").strip()

        if not source_stem and source_file:
            source_stem = Path(source_file).stem

        if source_stem and track_id:
            key = (source_stem, track_id)
            if key in ordered_by_key:
                source_row = ordered_by_key[key]
                measurements_by_key[key] = {
                    "source_file": source_row.get("source_file", source_file),
                    "source_stem": source_stem,
                    "track_id": track_id,
                    "real_length_mm": record["real_length_mm"],
                }
            else:
                warnings.append(f"Bỏ qua {source_stem} ID {track_id}: không có trong kết quả hiện tại")
        else:
            sequential_records.append(record)

    if sequential_records:
        available_rows = [
            row for row in ordered_rows
            if (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip()) not in measurements_by_key
        ]
        for row, record in zip(available_rows, sequential_records):
            key = (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip())
            measurements_by_key[key] = {
                "source_file": row.get("source_file", ""),
                "source_stem": key[0],
                "track_id": key[1],
                "real_length_mm": record["real_length_mm"],
            }
        if len(sequential_records) > len(available_rows):
            warnings.append(f"Bỏ qua {len(sequential_records) - len(available_rows)} dòng mm vì nhiều hơn số dòng kết quả")

    measurements = [
        measurements_by_key[(str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip())]
        for row in ordered_rows
        if (str(row.get("source_stem") or "").strip(), str(row.get("track_id") or "").strip()) in measurements_by_key
    ]
    return measurements, warnings


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


def _least_squares_mm_per_px(samples: list[dict[str, Any]]) -> tuple[float, float, list[dict[str, Any]]]:
    if not samples:
        raise ValueError("Cần ít nhất 1 mẫu hợp lệ để tính scale")

    sum_xx = sum(sample["pixel_length"] ** 2 for sample in samples)
    sum_xy = sum(sample["pixel_length"] * sample["real_length_mm"] for sample in samples)
    if sum_xx == 0:
        raise ValueError("pixel_length phải lớn hơn 0")

    scale = sum_xy / sum_xx
    enriched_samples = []
    squared_errors = []
    for sample in samples:
        fitted_mm = sample["pixel_length"] * scale
        residual_mm = sample["real_length_mm"] - fitted_mm
        squared_errors.append(residual_mm ** 2)
        enriched_sample = dict(sample)
        enriched_sample["fitted_mm"] = round(fitted_mm, 6)
        enriched_sample["residual_mm"] = round(residual_mm, 6)
        enriched_samples.append(enriched_sample)

    rmse_mm = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return scale, rmse_mm, enriched_samples


@app.get("/")
def index():
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
    return _config_response()


@app.put("/api/config")
def put_config():
    try:
        data = _validate_config_payload(request.get_json(force=True, silent=True) or {})
        _write_config(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _config_response()


@app.post("/api/config/pick-path")
def pick_config_path():
    payload = request.get_json(force=True, silent=True) or {}
    key = str(payload.get("key") or "").strip()
    mode = str(payload.get("mode") or "folder").strip()
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


@app.post("/api/calibrate/import-measurements")
def import_scale_measurements():
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
        table = _scale_import_table(raw, file.filename)
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
            "measurements": measurements,
            "count": len(measurements),
            "expected_count": len(ordered_rows),
            "warnings": warnings,
        }
    )


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

        samples.append(
            {
                "source_stem": source_stem,
                "source_file": record["source_file"],
                "track_id": record["track_id"],
                "pixel_length": pixel_length,
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
            "scale": config_data["SCALE"],
            "method": "least_squares_origin",
            "formula": "real_length_mm = scale * pixel_length",
            "rmse_mm": round(rmse_mm, 6),
            "count": len(samples),
            "samples": samples,
            "errors": errors,
            "config": _with_settings_errors(_jsonable_config()),
        }
    )


@app.get("/api/config/sizes")
def get_sizes():
    return _sizes_response()


@app.put("/api/config/sizes")
def put_sizes():
    try:
        data = _validate_sizes_payload(request.get_json(force=True, silent=True) or {})
        _write_sizes(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _sizes_response()


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
                continue
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
    writer.writerows(_result_export_rows(data))
    return Response(
        buffer.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'csv')}"},
    )


@app.get("/api/results/export-excel")
def export_excel():
    data = _results_for_run(request.args.get("run"))
    return Response(
        _xlsx_bytes(_result_export_rows(data)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={_export_filename(data, 'xlsx')}"},
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
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "3000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Shrimp Measure UI: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
