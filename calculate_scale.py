import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"

# Giữ tên biến theo ghi chú ban đầu; các giá trị 150, 195... được dùng trực tiếp là mm.
GROUND_TRUTH_CM_DEFAULT: dict[int, float] = {
    1: 150,
    2: 195,
    3: 125,
    4: 180,
    5: 210,
    6: 150,
    7: 140,
    8: 200,
    9: 195,
    10: 160,
    11: 155,
    12: 180,
    13: 200,
    14: 160,
    15: 155,
    16: 155,
    17: 150,
    18: 130,
    19: 170,
    20: 160,
    21: 155,
    22: 155,
    23: 150,
    24: 155,
    25: 153,
    26: 155,
    27: 153,
}


def read_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("settings.json phải là object")
    return settings


def output_dir() -> Path:
    settings = read_settings()
    output_value = settings.get("config", {}).get("OUTPUT_DIR", "output")
    path = Path(str(output_value))
    return path if path.is_absolute() else BASE_DIR / path


def find_run_dir(run_name: str | None = None) -> Path:
    root = output_dir()
    if run_name:
        run_dir = root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy run: {run_name}")
        return run_dir

    runs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not runs:
        raise FileNotFoundError(f"Không có run trong {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def result_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    sample_index = 1
    for json_path in sorted(run_dir.glob("*/*_results.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Bỏ qua {json_path}: {exc}")
            continue

        for shrimp in data.get("shrimps", []):
            try:
                pixel_length = float(shrimp["pixel_length"])
            except (KeyError, TypeError, ValueError):
                continue
            if pixel_length <= 0:
                continue

            records.append(
                {
                    "sample_index": sample_index,
                    "source_file": data.get("source_file", json_path.parent.name),
                    "source_stem": data.get("source_stem", json_path.parent.name),
                    "track_id": shrimp.get("track_id"),
                    "track_id_int": int_or_none(shrimp.get("track_id")),
                    "pixel_length": pixel_length,
                    "real_length_mm": shrimp.get("real_length_mm"),
                }
            )
            sample_index += 1
    return records


def attach_ground_truth(records: list[dict[str, Any]], ground_truth: dict[int, float]) -> list[dict[str, Any]]:
    track_counts = Counter(record["track_id_int"] for record in records if record["track_id_int"] is not None)
    samples = []
    for record in records:
        track_id = record["track_id_int"]
        truth_key = None
        if track_id in ground_truth and track_counts[track_id] == 1:
            truth_key = track_id
        elif record["sample_index"] in ground_truth:
            truth_key = record["sample_index"]
        elif track_id in ground_truth:
            truth_key = track_id

        if truth_key is None:
            continue

        sample = dict(record)
        sample["truth_key"] = truth_key
        sample["ground_truth_mm"] = float(ground_truth[truth_key])
        samples.append(sample)
    return samples


def least_squares_scale(samples: list[dict[str, Any]]) -> tuple[float, float]:
    if not samples:
        raise ValueError("Cần ít nhất 1 mẫu hợp lệ để tính scale")

    sum_xx = sum(sample["pixel_length"] ** 2 for sample in samples)
    sum_xy = sum(sample["pixel_length"] * sample["ground_truth_mm"] for sample in samples)
    if sum_xx == 0:
        raise ValueError("pixel_length phải lớn hơn 0")

    scale = sum_xy / sum_xx
    mse = 0.0
    for sample in samples:
        fitted_mm = scale * sample["pixel_length"]
        sample["fitted_mm"] = fitted_mm
        sample["residual_mm"] = fitted_mm - sample["ground_truth_mm"]
        mse += sample["residual_mm"] ** 2
    return scale, math.sqrt(mse / len(samples))


def save_scale(scale: float) -> None:
    settings = read_settings()
    config = settings.setdefault("config", {})
    if not isinstance(config, dict):
        raise ValueError("settings.json mục config phải là object")
    config["SCALE"] = round(scale, 6)
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_scale_result(run_dir: Path, scale: float, rmse_mm: float, samples: list[dict[str, Any]]) -> None:
    print(f"Run: {run_dir.name}")
    print(f"Số mẫu dùng tính scale: {len(samples)}")
    print(f"SCALE = {scale:.6f} mm/px")
    print(f"RMSE fit = {rmse_mm:.3f} mm")
    print("Công thức fit: y = SCALE * pixel_length")
    print()
    print("key | source_file | track_id | pixel | ground_truth_mm | fitted_mm | err_mm")
    for sample in samples:
        print(
            f"{sample['truth_key']:>3} | "
            f"{sample['source_file']} | "
            f"{sample['track_id']} | "
            f"{sample['pixel_length']:.1f} | "
            f"{sample['ground_truth_mm']:.2f} | "
            f"{sample['fitted_mm']:.2f} | "
            f"{sample['residual_mm']:+.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính SCALE từ JSON output và ground truth.")
    parser.add_argument("--run", help="Tên run trong thư mục output. Bỏ trống thì dùng run mới nhất.")
    parser.add_argument("--save", action="store_true", help="Ghi SCALE vào settings.json sau khi tính.")
    args = parser.parse_args()

    run_dir = find_run_dir(args.run)
    samples = attach_ground_truth(result_records(run_dir), GROUND_TRUTH_CM_DEFAULT)
    if not samples:
        raise ValueError("Không có mẫu nào khớp ground truth")

    scale, rmse_mm = least_squares_scale(samples)
    print_scale_result(run_dir, scale, rmse_mm, samples)
    if args.save:
        save_scale(scale)
        print(f"\nĐã ghi SCALE = {round(scale, 6)} vào settings.json")
    else:
        print("\nChưa ghi settings.json. Nếu muốn lưu SCALE, chạy thêm --save")


if __name__ == "__main__":
    main()
