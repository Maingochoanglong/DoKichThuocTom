"""
size.py

Định nghĩa các hằng số và hàm phân loại kích cỡ tôm theo chiều dài thực (mm).

Bảng phân loại và các nhãn ngoại cỡ đều đọc từ settings.json (section "size").
Nếu key chưa có trong file, giá trị mặc định sẽ được tự động ghi lại vào file.
"""

from settings_loader import load_setting


# Bảng phân loại kích cỡ: nhãn -> (từ mm, đến mm).
SIZE_RANGES: dict[str, tuple[float, float]] = {
    str(label): (float(bounds[0]), float(bounds[1]))
    for label, bounds in load_setting(
        "SIZE_RANGES",
        {},
        section="size",
    ).items()
}

# Nhãn cho tôm nhỏ hơn khoảng đầu tiên trong bảng.
UNDERSIZE_LABEL = str(load_setting("UNDERSIZE_LABEL", "Ngoại cỡ nhỏ", section="size"))

# Nhãn cho tôm lớn hơn khoảng cuối cùng trong bảng.
OVERSIZE_LABEL = str(load_setting("OVERSIZE_LABEL", "Ngoại cỡ lớn", section="size"))

# Nhãn dự phòng khi chiều dài không khớp với khoảng nào (bảng rỗng hoặc ngoài tất cả khoảng).
FALLBACK_LABEL = str(load_setting("FALLBACK_LABEL", "Ngoại cỡ", section="size"))


def classify_size(real_length: float) -> str:
    """
    Phân loại tôm theo chiều dài thực (mm).

    Duyệt qua SIZE_RANGES theo thứ tự khai báo. Khoảng [lo, hi) là nửa mở
    (lo <= real_length < hi).

    Tham số:
        real_length: Chiều dài thực của tôm tính bằng mm.

    Trả về:
        Nhãn kích cỡ (ví dụ "S", "M", "L") hoặc nhãn ngoại cỡ tương ứng.
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