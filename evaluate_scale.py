import argparse
import math
import sys
from typing import Any

from calculate_scale import GROUND_TRUTH_CM_DEFAULT, attach_ground_truth, find_run_dir, read_settings, result_records


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def current_scale() -> float:
    config = read_settings().get("config", {})
    if not isinstance(config, dict):
        raise ValueError("settings.json mục config phải là object")
    return float(config.get("SCALE", 1.0))


def valid_samples(run_name: str | None = None, use_json_mm: bool = False) -> tuple[str, str, list[dict[str, Any]]]:
    run_dir = find_run_dir(run_name)
    scale = current_scale()
    method = "real_length_mm trong JSON" if use_json_mm else f"settings.json: y = {scale:.6f} * pixel"
    samples = []
    for sample in attach_ground_truth(result_records(run_dir), GROUND_TRUTH_CM_DEFAULT):
        predicted_mm = number_or_none(sample.get("real_length_mm")) if use_json_mm else sample["pixel_length"] * scale
        if predicted_mm is None:
            continue
        sample["predicted_mm"] = predicted_mm
        sample["error_mm"] = predicted_mm - sample["ground_truth_mm"]
        sample["abs_error_mm"] = abs(sample["error_mm"])
        sample["relative_error_percent"] = sample["error_mm"] / sample["ground_truth_mm"] * 100
        samples.append(sample)
    return run_dir.name, method, samples


def evaluate(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        raise ValueError("Không có mẫu hợp lệ để đánh giá")

    n = len(samples)
    mae = sum(sample["abs_error_mm"] for sample in samples) / n
    rmse = math.sqrt(sum(sample["error_mm"] ** 2 for sample in samples) / n)
    mape = sum(sample["abs_error_mm"] / sample["ground_truth_mm"] for sample in samples) * 100 / n
    return {"n": n, "mae": mae, "rmse": rmse, "mape": mape}


def print_evaluation(run_name: str, method: str, metrics: dict[str, float], samples: list[dict[str, Any]]) -> None:
    print(f"Run: {run_name}")
    print(f"ŷ lấy từ: {method}")
    print(f"Số mẫu đánh giá: {int(metrics['n'])}")
    print(f"MAE  = {metrics['mae']:.3f} mm")
    print(f"RMSE = {metrics['rmse']:.3f} mm")
    print(f"MAPE = {metrics['mape']:.3f} %")
    print()
    print("key | source_file | track_id | y_hat_mm | y_mm | err_mm | err_%")
    for sample in samples:
        print(
            f"{sample['truth_key']:>3} | "
            f"{sample['source_file']} | "
            f"{sample['track_id']} | "
            f"{sample['predicted_mm']:.2f} | "
            f"{sample['ground_truth_mm']:.2f} | "
            f"{sample['error_mm']:+.2f} | "
            f"{sample['relative_error_percent']:+.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá MAE, RMSE, MAPE từ JSON output sau khi chạy scale.")
    parser.add_argument("--run", help="Tên run trong thư mục output. Bỏ trống thì dùng run mới nhất.")
    parser.add_argument("--json-mm", action="store_true", help="Đánh giá real_length_mm đang có sẵn trong JSON thay vì tính từ settings.json.")
    args = parser.parse_args()

    run_name, method, samples = valid_samples(args.run, args.json_mm)
    print_evaluation(run_name, method, evaluate(samples), samples)


if __name__ == "__main__":
    main()
