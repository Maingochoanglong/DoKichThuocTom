"""
compare_runs.py

So sánh 2 run mới nhất trong OUTPUT_DIR.
Chỉ liệt kê các file có kết quả khác nhau giữa hai run.
"""

import json
from pathlib import Path

# ─── CẤU HÌNH ───────────────────────────────────────────────
OUTPUT_DIR = r"C:\shrimp\output"   # điền đường dẫn folder output
# ────────────────────────────────────────────────────────────


def _find_two_latest_runs(output_dir: Path) -> tuple[Path, Path]:
    runs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(runs) < 2:
        raise RuntimeError(f"Cần ít nhất 2 run trong '{output_dir}', hiện có {len(runs)}")
    return runs[0], runs[1]


def _read_json_files(run_dir: Path) -> dict[str, dict]:
    """Đọc tất cả *_results.json trong run, trả về dict keyed theo source_file."""
    result = {}
    for jf in run_dir.glob("*/*_results.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        key = data.get("source_file", jf.parent.name)
        result[key] = data
    return result


def _shrimp_key(shrimp: dict) -> tuple:
    """Dùng để so sánh từng con tôm, bỏ qua ảnh debug."""
    return (
        shrimp.get("track_id"),
        shrimp.get("frame_idx"),
        shrimp.get("pixel_length"),
        shrimp.get("real_length_mm"),
        shrimp.get("size"),
    )


def _compare_source(name: str, a: dict, b: dict) -> list[str]:
    """So sánh kết quả của một source file giữa 2 run, trả về danh sách điểm khác."""
    diffs = []

    shrimps_a = {s["track_id"]: s for s in a.get("shrimps", [])}
    shrimps_b = {s["track_id"]: s for s in b.get("shrimps", [])}

    only_in_a = set(shrimps_a) - set(shrimps_b)
    only_in_b = set(shrimps_b) - set(shrimps_a)
    common    = set(shrimps_a) & set(shrimps_b)

    if only_in_a:
        ids = sorted(only_in_a)
        diffs.append(f"  Chỉ có trong run mới hơn  : ID {ids}")

    if only_in_b:
        ids = sorted(only_in_b)
        diffs.append(f"  Chỉ có trong run cũ hơn   : ID {ids}")

    for tid in sorted(common):
        ka = _shrimp_key(shrimps_a[tid])
        kb = _shrimp_key(shrimps_b[tid])
        if ka != kb:
            sa, sb = shrimps_a[tid], shrimps_b[tid]
            diffs.append(
                f"  ID {tid:>3}  pixel {sa.get('pixel_length')} → {sb.get('pixel_length')}"
                f"   mm {sa.get('real_length_mm')} → {sb.get('real_length_mm')}"
                f"   size {sa.get('size')} → {sb.get('size')}"
            )

    return diffs


def compare_runs(output_dir: str) -> None:
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")

    run_new, run_old = _find_two_latest_runs(root)

    print(f"Run mới hơn : {run_new.name}")
    print(f"Run cũ hơn  : {run_old.name}")
    print()

    data_new = _read_json_files(run_new)
    data_old = _read_json_files(run_old)

    all_sources = sorted(set(data_new) | set(data_old))

    if not all_sources:
        print("Không tìm thấy file JSON trong cả 2 run.")
        return

    found_diff = False

    for name in all_sources:
        in_new = name in data_new
        in_old = name in data_old

        if in_new and not in_old:
            print(f"[+] {name}  (chỉ có trong run mới hơn)")
            found_diff = True
            continue

        if in_old and not in_new:
            print(f"[-] {name}  (chỉ có trong run cũ hơn)")
            found_diff = True
            continue

        diffs = _compare_source(name, data_new[name], data_old[name])
        if diffs:
            print(f"[~] {name}")
            for line in diffs:
                print(line)
            found_diff = True

    if not found_diff:
        print("Hai run giống nhau hoàn toàn.")


if __name__ == "__main__":
    compare_runs(OUTPUT_DIR)