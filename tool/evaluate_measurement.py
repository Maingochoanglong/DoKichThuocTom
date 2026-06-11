"""
Tính MAE, RMSE, MAPE và bảng độ lệch giữa kết quả đo JSON với CSV thực tế.

CSV cần có một cột tên là mm. JSON có thể là một file kết quả hoặc một thư mục
chứa các file *_results.json của pipeline.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any


# Sửa hai đường dẫn này theo dữ liệu cần đánh giá.
OUTPUT_JSON_PATH = r"D:\DoKichThuocTom\output\video2"
GROUND_TRUTH_CSV_PATH = r"D:\DoKichThuocTom\test\video2\video2.csv"


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text.strip())
    if not path_text.strip():
        raise ValueError("Chưa cấu hình đường dẫn file hoặc thư mục.")
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw, 0, 1, f"Không đọc được mã hóa của {path}")


def _read_actual_mm(csv_path: Path) -> list[float]:
    text = _read_text(csv_path)

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows:
        raise ValueError("File CSV trống.")
    if "mm" not in rows[0]:
        raise ValueError(f"Không tìm thấy cột 'mm' trong CSV. Các cột hiện có: {list(rows[0].keys())}")

    values: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        cell = str(row.get("mm", "")).strip().replace(",", ".")
        if not cell:
            continue
        try:
            value = float(cell)
        except ValueError as exc:
            raise ValueError(f"Dòng {row_index} trong CSV có giá trị mm không hợp lệ: {cell}") from exc
        if value <= 0:
            raise ValueError(f"Dòng {row_index} trong CSV phải có mm lớn hơn 0: {cell}")
        values.append(value)

    if not values:
        raise ValueError("CSV không có giá trị mm hợp lệ.")
    return values


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*_results.json"))
        if files:
            return files
        return sorted(path.rglob("*.json"))
    raise FileNotFoundError(f"Không tìm thấy đường dẫn output JSON: {path}")


def _number_from_item(item: dict[str, Any]) -> float | None:
    for key in ("mm", "real_length_mm", "measured_mm", "length_mm", "predicted_mm"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(str(value).strip().replace(",", "."))
        except ValueError:
            return None
    return None


def _extract_predictions_from_data(data: Any, source_name: str) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    if isinstance(data, dict) and isinstance(data.get("shrimps"), list):
        source_stem = str(data.get("source_stem") or source_name)
        for shrimp in data["shrimps"]:
            if not isinstance(shrimp, dict):
                continue
            predicted_mm = _number_from_item(shrimp)
            if predicted_mm is None:
                continue
            predictions.append({
                "source": source_stem,
                "id": shrimp.get("track_id", ""),
                "predicted_mm": predicted_mm,
            })
        return predictions

    if isinstance(data, list):
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            predicted_mm = _number_from_item(item)
            if predicted_mm is None:
                continue
            predictions.append({
                "source": str(item.get("source_stem") or item.get("source") or source_name),
                "id": item.get("track_id", item.get("id", index)),
                "predicted_mm": predicted_mm,
            })
        return predictions

    if isinstance(data, dict):
        predicted_mm = _number_from_item(data)
        if predicted_mm is not None:
            predictions.append({
                "source": str(data.get("source_stem") or data.get("source") or source_name),
                "id": data.get("track_id", data.get("id", "")),
                "predicted_mm": predicted_mm,
            })

    return predictions


def _read_predicted_mm(output_path: Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for json_path in _json_files(output_path):
        data = json.loads(_read_text(json_path))
        predictions.extend(_extract_predictions_from_data(data, json_path.stem))

    if not predictions:
        raise ValueError("Không tìm thấy giá trị mm đo được trong JSON.")
    return predictions


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = [row["deviation_mm"] for row in rows]
    actuals = [row["actual_mm"] for row in rows]
    n = len(rows)
    mae = sum(abs(error) for error in errors) / n
    rmse = math.sqrt(sum(error * error for error in errors) / n)
    mape = sum(abs(error) / actual * 100 for error, actual in zip(errors, actuals)) / n
    return {"n": n, "mae": mae, "rmse": rmse, "mape": mape}


def _format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def line(values: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    print(line(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        print(line(row))


def _build_compare_rows(predictions: list[dict[str, Any]], actuals: list[float]) -> list[dict[str, Any]]:
    count = min(len(predictions), len(actuals))
    rows: list[dict[str, Any]] = []

    for index in range(count):
        predicted_mm = predictions[index]["predicted_mm"]
        actual_mm = actuals[index]
        deviation_mm = predicted_mm - actual_mm
        deviation_percent = deviation_mm / actual_mm * 100
        rows.append({
            "index": index + 1,
            "source": predictions[index]["source"],
            "id": predictions[index]["id"],
            "predicted_mm": predicted_mm,
            "actual_mm": actual_mm,
            "deviation_mm": deviation_mm,
            "deviation_percent": deviation_percent,
        })

    return rows


def evaluate(output_json_path: str, ground_truth_csv_path: str) -> None:
    output_path = _resolve_path(output_json_path)
    csv_path = _resolve_path(ground_truth_csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    predictions = _read_predicted_mm(output_path)
    actuals = _read_actual_mm(csv_path)
    rows = _build_compare_rows(predictions, actuals)

    if not rows:
        raise ValueError("Không có cặp dữ liệu hợp lệ để đánh giá.")

    if len(predictions) != len(actuals):
        print(
            f"Cảnh báo: JSON có {len(predictions)} giá trị, CSV có {len(actuals)} giá trị. "
            f"Chỉ đánh giá {len(rows)} cặp đầu tiên."
        )
        print()

    metric_values = _metrics(rows)
    print("Bảng chỉ số tổng hợp")
    _print_table(
        ["Số mẫu", "MAE (mm)", "RMSE (mm)", "MAPE (%)"],
        [[
            str(int(metric_values["n"])),
            _format_float(metric_values["mae"], 3),
            _format_float(metric_values["rmse"], 3),
            _format_float(metric_values["mape"], 2),
        ]],
    )

    print()
    print("Bảng độ lệch từng mẫu")
    _print_table(
        ["STT", "Nguồn", "ID", "Đo được (mm)", "Thực tế (mm)", "Độ lệch (mm)", "% lệch"],
        [
            [
                str(row["index"]),
                str(row["source"]),
                str(row["id"]),
                _format_float(row["predicted_mm"]),
                _format_float(row["actual_mm"]),
                _format_float(row["deviation_mm"]),
                _format_float(row["deviation_percent"]),
            ]
            for row in rows
        ],
    )


if __name__ == "__main__":
    evaluate(OUTPUT_JSON_PATH, GROUND_TRUTH_CSV_PATH)
