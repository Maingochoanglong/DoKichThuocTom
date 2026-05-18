from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file

import config
import size

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / config.INPUT_DIR
OUTPUT_DIR = BASE_DIR / config.OUTPUT_DIR
ALLOWED_EXTS = {ext.lower() for ext in (config.IMG_EXTS | config.VID_EXTS)}


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/files/upload")
def upload_files():
    files = request.files.getlist("files")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in ALLOWED_EXTS:
            errors.append({"filename": name, "error": "unsupported_format"})
            continue
        path = INPUT_DIR / name
        f.save(path)
        saved.append({"filename": name, "size": path.stat().st_size})
    return jsonify({"saved": saved, "errors": errors})


@app.get("/api/files/input")
def list_input_files():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {"filename": p.name, "size": p.stat().st_size}
        for p in sorted(INPUT_DIR.iterdir())
        if p.is_file()
    ]
    return jsonify(data)


@app.delete("/api/files/input/<path:filename>")
def delete_input_file(filename: str):
    path = INPUT_DIR / Path(filename).name
    if not path.exists():
        return jsonify({"error": "not_found"}), 404
    path.unlink()
    return jsonify({"deleted": path.name})


@app.post("/api/pipeline/run")
def run_pipeline():
    def _run():
        subprocess.run(["python", "main.py"], cwd=BASE_DIR, check=False)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.get("/api/results/runs")
def list_runs():
    runs = sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()], reverse=True) if OUTPUT_DIR.exists() else []
    return jsonify(runs)


def _results_files(run_dir: str):
    run_path = OUTPUT_DIR / run_dir
    return sorted(run_path.glob("**/*_results.json")) if run_path.exists() else []


@app.get("/api/results/sources")
def list_sources():
    run_dir = request.args.get("run_dir", "")
    files = _results_files(run_dir)
    return jsonify([p.parent.name for p in files])


@app.get("/api/results/shrimps")
def list_shrimps():
    run_dir = request.args.get("run_dir", "")
    source = request.args.get("source", "")
    all_items: list[dict[str, Any]] = []
    for fp in _results_files(run_dir):
        if source and fp.parent.name != source:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for row in data.get("shrimps", []):
            row["source_stem"] = data.get("source_stem", fp.parent.name)
            all_items.append(row)
    return jsonify(all_items)


@app.get("/api/results/image")
def get_image():
    raw_path = request.args.get("path", "")
    norm = raw_path.replace("\\", os.sep)
    full = (BASE_DIR / norm).resolve()
    if not str(full).startswith(str(BASE_DIR.resolve())) or not full.exists():
        return jsonify({"error": "not_found"}), 404
    return send_file(full)


@app.get("/api/config")
def get_config():
    keys = ["SCALE", "CONF_DET", "CONF_SEG", "BBOX_PAD", "TOUCH_THRESHOLD", "TARGET_FPS", "SAVE", "CONVEYOR_VERTICAL", "CHUNK_MODE", "CLEAR_OUTPUT", "CLEAR_INPUT"]
    return jsonify({k: getattr(config, k) for k in keys})


def _update_py_constant(file_path: Path, key: str, value: Any):
    text = file_path.read_text(encoding="utf-8")
    replacement = f"{key} = {repr(value)}"
    updated = re.sub(rf"^{re.escape(key)}\s*=.*$", replacement, text, flags=re.M)
    file_path.write_text(updated, encoding="utf-8")


@app.put("/api/config")
def put_config():
    payload = request.get_json(force=True)
    for key, value in payload.items():
        _update_py_constant(BASE_DIR / "config.py", key, value)
    importlib.reload(config)
    return jsonify({"ok": True})


@app.post("/api/calibrate")
def calibrate():
    body = request.get_json(force=True)
    entries = body.get("entries", [])
    if not entries:
        return jsonify({"error": "entries_empty"}), 400

    run_dir = request.args.get("run_dir", "")
    files = {p.parent.name: p for p in _results_files(run_dir)}

    scales_detail = []
    for e in entries:
        source_stem = e["source_stem"]
        track_id = int(e["track_id"])
        real_mm = float(e["real_length_mm"])
        fp = files.get(source_stem)
        if fp is None:
            return jsonify({"error": f'source_not_found:{source_stem}' }), 404
        data = json.loads(fp.read_text(encoding="utf-8"))
        shrimp = next((s for s in data.get("shrimps", []) if int(s.get("track_id", -1)) == track_id), None)
        if shrimp is None:
            return jsonify({"error": f'track_not_found:{source_stem}:{track_id}'}), 404
        pixel = float(shrimp.get("pixel_length", 0))
        if pixel <= 0:
            return jsonify({"error": f'pixel_zero:{track_id}'}), 400
        scales_detail.append(real_mm / pixel)

    new_scale = sum(scales_detail) / len(scales_detail)
    _update_py_constant(BASE_DIR / "config.py", "SCALE", round(new_scale, 6))
    importlib.reload(config)
    return jsonify({"scale": round(new_scale, 6), "num_samples": len(scales_detail), "scales_detail": [round(x, 6) for x in scales_detail]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
