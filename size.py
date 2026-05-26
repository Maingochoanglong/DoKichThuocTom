"""
size.py

Định nghĩa bảng phân loại kích cỡ tôm theo chiều dài thực tế.
Các giá trị chỉnh được đọc từ settings.json qua settings_loader.
"""

from settings_loader import load_setting


def load_size_values() -> dict:
    """Đọc toàn bộ cấu hình kích cỡ và tự ghi default nếu thiếu."""
    raw_ranges = load_setting(
        "SIZE_RANGES",
        {
            "S": [12, 14],
            "M": [14, 17],
            "L": [18, 22],
        },
        section="size",
    )

    return {
        "SIZE_RANGES": {
            str(label): (float(bounds[0]), float(bounds[1]))
            for label, bounds in raw_ranges.items()
        },
        "UNDERSIZE_LABEL": str(load_setting("UNDERSIZE_LABEL", "Ngoại cỡ nhỏ", section="size")),
        "OVERSIZE_LABEL": str(load_setting("OVERSIZE_LABEL", "Ngoại cỡ lớn", section="size")),
        "FALLBACK_LABEL": str(load_setting("FALLBACK_LABEL", "Ngoại cỡ", section="size")),
    }


_SIZE_VALUES = load_size_values()

# Bảng phân loại kích cỡ: nhãn -> khoảng [từ mm, đến mm).
SIZE_RANGES: dict[str, tuple[float, float]] = _SIZE_VALUES["SIZE_RANGES"]

# Nhãn cho tôm nhỏ hơn khoảng đầu tiên trong bảng.
UNDERSIZE_LABEL = _SIZE_VALUES["UNDERSIZE_LABEL"]

# Nhãn cho tôm lớn hơn khoảng cuối cùng trong bảng.
OVERSIZE_LABEL = _SIZE_VALUES["OVERSIZE_LABEL"]

# Nhãn dự phòng khi chiều dài không khớp với khoảng nào.
FALLBACK_LABEL = _SIZE_VALUES["FALLBACK_LABEL"]


def classify_size(real_length: float) -> str:
    """
    Phân loại tôm theo chiều dài thực tế.
    Khoảng phân loại dùng dạng nửa mở: lo <= real_length < hi.
    """
    for size_label, (lo, hi) in SIZE_RANGES.items():
        if lo <= real_length < hi:
            return size_label

    if SIZE_RANGES:
        min_length = min(lo for lo, _ in SIZE_RANGES.values())
        max_length = max(hi for _, hi in SIZE_RANGES.values())
        if real_length < min_length:
            return UNDERSIZE_LABEL
        if real_length >= max_length:
            return OVERSIZE_LABEL

    return FALLBACK_LABEL
