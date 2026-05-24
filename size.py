"""
size.py
Định nghĩa các hằng số và hàm phân loại kích cỡ tôm theo chiều dài thực (mm).
"""

from settings_loader import load_setting


# Bảng phân loại kích cỡ (từ mm đến mm)
SIZE_RANGES: dict[str, tuple[float, float]] = {
    str(label): (float(bounds[0]), float(bounds[1]))
    for label, bounds in load_setting(
        "SIZE_RANGES",
        {},
    ).items()
}

# Các nhãn ngoại cỡ để người dùng có thể tùy chỉnh.
UNDERSIZE_LABEL = str(load_setting("UNDERSIZE_LABEL", 'Ngoại cỡ nhỏ'))
OVERSIZE_LABEL = str(load_setting("OVERSIZE_LABEL", 'Ngoại cỡ lớn'))
FALLBACK_LABEL = str(load_setting("FALLBACK_LABEL", 'Ngoại cỡ'))


def classify_size(real_length: float) -> str:
    """
    Phân loại tôm theo chiều dài thực (mm).
    """
    for size_label, (lo, hi) in SIZE_RANGES.items():
        if lo <= real_length < hi:
            return size_label

    if SIZE_RANGES:
        min_length = min(lo for lo, hi in SIZE_RANGES.values())
        max_length = max(hi for lo, hi in SIZE_RANGES.values())

        if real_length < min_length:
            return UNDERSIZE_LABEL
        if real_length >= max_length:
            return OVERSIZE_LABEL

    return FALLBACK_LABEL
