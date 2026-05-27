"""
size.py

Định nghĩa bảng phân loại kích cỡ tôm theo chiều dài thực tế.
Các giá trị chỉnh được đọc từ settings.json qua settings_loader. Hàm phân loại
nhận cấu hình kích cỡ qua tham số để không phụ thuộc trạng thái module-level.
"""

from typing import Any

from settings_loader import load_setting


# Section trong settings.json dùng cho phân loại kích cỡ.
SIZE_SECTION = "size"


def _load_size(key: str, default: Any) -> Any:
    """Đọc một key trong section size bằng settings_loader."""
    return load_setting(key, default, section=SIZE_SECTION)


def load_size_values() -> dict[str, Any]:
    """
    Đọc toàn bộ cấu hình phân loại kích cỡ từ settings.json.

    Nếu section hoặc key thiếu, settings_loader sẽ ghi default vào file. Hàm
    trả `SIZE_RANGES` dưới dạng dict nhãn -> tuple float để caller dùng trực
    tiếp cho phân loại hoặc hiển thị.
    """
    raw_ranges = _load_size(
        "SIZE_RANGES",
        {
            "S": [120, 140],
            "M": [140, 170],
            "L": [180, 220],
        },
    )

    return {
        "SIZE_RANGES": {
            str(label): (float(bounds[0]), float(bounds[1]))
            for label, bounds in raw_ranges.items()
        },
        "UNDERSIZE_LABEL": str(_load_size("UNDERSIZE_LABEL", "Ngoại cỡ nhỏ")),
        "OVERSIZE_LABEL": str(_load_size("OVERSIZE_LABEL", "Ngoại cỡ lớn")),
        "FALLBACK_LABEL": str(_load_size("FALLBACK_LABEL", "Ngoại cỡ")),
    }


def classify_size(real_length: float, size_cfg: dict[str, Any]) -> str:
    """
    Phân loại tôm theo chiều dài thực tế.

    Khoảng phân loại dùng dạng nửa mở: lo <= real_length < hi. Nếu chiều dài
    nhỏ hơn toàn bộ bảng thì trả nhãn nhỏ, nếu lớn hơn hoặc bằng mốc cuối thì
    trả nhãn lớn. Khi bảng rỗng hoặc không khớp khoảng nào, hàm trả nhãn dự
    phòng trong size_cfg.
    """
    size_ranges = size_cfg["SIZE_RANGES"]
    undersize_label = size_cfg["UNDERSIZE_LABEL"]
    oversize_label = size_cfg["OVERSIZE_LABEL"]
    fallback_label = size_cfg["FALLBACK_LABEL"]

    for size_label, (lo, hi) in size_ranges.items():
        if lo <= real_length < hi:
            return size_label

    if size_ranges:
        min_length = min(lo for lo, _ in size_ranges.values())
        max_length = max(hi for _, hi in size_ranges.values())
        if real_length < min_length:
            return undersize_label
        if real_length >= max_length:
            return oversize_label

    return fallback_label
