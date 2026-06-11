"""
evaluate.py

Đánh giá độ chính xác đo chiều dài tôm bằng cách so sánh
kết quả trong run mới nhất với số đo thực tế từ file CSV.

Định dạng CSV cố định:
    mm
    150.2
    143.5
    ...
"""

import csv
import json
import math
from pathlib import Path

# ─── CẤU HÌNH ───────────────────────────────────────────────
OUTPUT_DIR = r"output"          # folder output
CSV_FILE   = r"D:\DoKichThuocTom\test\thang\thang.csv"      # file csv số đo thực tế
# ────────────────────────────────────────────────────────────


def _latest_run(output_dir: Path) -> Path:
    runs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise RuntimeError(f"Không có run nào trong '{output_dir}'")
    return runs[0]


def _read_run_shrimps(run_dir: Path) -> list[dict]:
    shrimps = []
    for jf in sorted(run_dir.glob("*/*_results.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stem = data.get("source_stem", jf.parent.name)
        for s in data.get("shrimps", []):
            shrimps.append({
                "source_stem" : stem,
                "source_file" : data.get("source_file", stem),
                "track_id"    : s.get("track_id"),
                "frame_idx"   : s.get("frame_idx"),
                "predicted_mm": float(s.get("real_length_mm", 0)),
                "size"        : s.get("size", ""),
            })
    return shrimps


def _read_csv(csv_path: Path) -> list[float]:
    raw = csv_path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("File CSV trống")
    if "mm" not in rows[0]:
        raise ValueError(f"Không tìm thấy cột 'mm' trong CSV, chỉ thấy: {list(rows[0].keys())}")

    values = []
    for i, row in enumerate(rows, start=2):
        cell = str(row["mm"] or "").strip().replace(",", ".")
        try:
            v = float(cell)
            if v <= 0:
                raise ValueError
            values.append(v)
        except ValueError:
            print(f"  Cảnh báo: dòng {i} giá trị '{cell}' không hợp lệ, bỏ qua")

    return values


def _metrics(errors: list[float], actuals: list[float]) -> dict:
    n    = len(errors)
    mae  = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    mape = sum(abs(e) / a * 100 for e, a in zip(errors, actuals) if a > 0) / n
    return {"n": n, "mae": mae, "rmse": rmse, "mape": mape}


def _error_distribution(errors: list[float]) -> None:
    buckets = [
        ("< 1 mm"  , lambda e: abs(e) <  1.0),
        ("1 - 3 mm", lambda e: 1.0 <= abs(e) <  3.0),
        ("3 - 5 mm", lambda e: 3.0 <= abs(e) <  5.0),
        (">= 5 mm" , lambda e: abs(e) >= 5.0),
    ]
    n = len(errors)
    print("  Phân bố sai số:")
    for label, cond in buckets:
        count = sum(1 for e in errors if cond(e))
        bar   = "█" * int(count / n * 30)
        print(f"    {label:10}  {count:>4} ({count / n * 100:5.1f}%)  {bar}")


def _report_by_source(pairs: list[dict]) -> None:
    groups: dict[str, list] = {}
    for p in pairs:
        groups.setdefault(p["source_stem"], []).append(p)

    if len(groups) <= 1:
        return

    print("\nKết quả theo từng file:")
    print(f"  {'Source':<25} {'N':>4}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}")
    print(f"  {'-'*25} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for stem, group in sorted(groups.items()):
        errors  = [g["predicted_mm"] - g["actual_mm"] for g in group]
        actuals = [g["actual_mm"] for g in group]
        m = _metrics(errors, actuals)
        print(
            f"  {stem:<25} {m['n']:>4}"
            f"  {m['mae']:>8.3f}  {m['rmse']:>8.3f}  {m['mape']:>7.2f}%"
        )


def _print_detail(pairs: list[dict]) -> None:
    print(f"\n  {'Source':<20} {'ID':>4} {'Dự đoán':>10} {'Thực tế':>10} {'Lệch':>8} {'% Lệch':>8}")
    print(f"  {'-'*20} {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for p in pairs:
        err = p["predicted_mm"] - p["actual_mm"]
        pct = err / p["actual_mm"] * 100 if p["actual_mm"] > 0 else 0
        print(
            f"  {p['source_stem']:<20} {str(p['track_id']):>4}"
            f" {p['predicted_mm']:>10.2f} {p['actual_mm']:>10.2f}"
            f" {err:>+8.2f} {pct:>+7.2f}%"
        )


def evaluate(output_dir: str, csv_file: str) -> None:
    root     = Path(output_dir)
    csv_path = Path(csv_file)

    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy OUTPUT_DIR: {root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV_FILE: {csv_path}")

    run_dir   = _latest_run(root)
    shrimps   = _read_run_shrimps(run_dir)
    mm_values = _read_csv(csv_path)

    if not shrimps:
        print("Không có dữ liệu tôm trong run mới nhất.")
        return
    if not mm_values:
        print("Không có giá trị hợp lệ trong CSV.")
        return

    n_pairs = min(len(shrimps), len(mm_values))
    pairs   = [
        {**shrimps[i], "actual_mm": mm_values[i]}
        for i in range(n_pairs)
    ]

    print(f"Run      : {run_dir.name}")
    print(f"CSV      : {csv_path.name}")
    print(f"Tôm run  : {len(shrimps)}  |  CSV  : {len(mm_values)}  |  Đánh giá  : {n_pairs}")

    if len(shrimps) != len(mm_values):
        print(f"  Cảnh báo: số tôm và số giá trị CSV không bằng nhau, chỉ đánh giá {n_pairs} cặp đầu")

    print()

    errors  = [p["predicted_mm"] - p["actual_mm"] for p in pairs]
    actuals = [p["actual_mm"] for p in pairs]
    m       = _metrics(errors, actuals)

    print("Kết quả tổng thể:")
    print(f"  Số mẫu  : {m['n']}")
    print(f"  MAE     : {m['mae']:.3f} mm   (sai số trung bình)")
    print(f"  RMSE    : {m['rmse']:.3f} mm   (sai số bình phương trung bình)")
    print(f"  MAPE    : {m['mape']:.2f}%     (sai số phần trăm trung bình)")
    print()
    _error_distribution(errors)
    _report_by_source(pairs)

    print("\nChi tiết từng con (sắp xếp theo sai số giảm dần):")
    _print_detail(pairs)


if __name__ == "__main__":
    evaluate(OUTPUT_DIR, CSV_FILE)