"""
size.py
Định nghĩa các hằng số và hàm phân loại kích cỡ tôm theo chiều dài thực (mm).
"""

# Bảng phân loại kích cỡ (từ mm đến mm)
SIZE_RANGES: dict[str, tuple[float, float]] = {
    "S": (100, 125), # Tôm nhỏ: 100-125 mm
    "M": (125, 160), # Tôm vừa: 125-160 mm
    "L": (160, 200), # Tôm lớn: 160-200 mm
}

# Các nhãn ngoại cỡ để người dùng có thể dễ dàng tùy chỉnh
UNDERSIZE_LABEL = "Ngoại cỡ nhỏ"  # Dùng khi tôm nhỏ hơn kích thước tối thiểu
OVERSIZE_LABEL  = "Ngoại cỡ lớn"  # Dùng khi tôm lớn hơn kích thước tối đa
FALLBACK_LABEL  = "Ngoại cỡ"      # Dùng khi rơi vào khoảng trống giữa các size (nếu có)

def classify_size(real_length: float) -> str:
    """
    Phân loại tôm theo chiều dài thực (mm).
    """
    for size_label, (lo, hi) in SIZE_RANGES.items():
        if lo <= real_length < hi:
            return size_label
            
    # Nếu không khớp với khoảng nào, phân loại là ngoại cỡ nhỏ / lớn
    if SIZE_RANGES:
        min_length = min(lo for lo, hi in SIZE_RANGES.values())
        max_length = max(hi for lo, hi in SIZE_RANGES.values())
        
        if real_length < min_length:
            return UNDERSIZE_LABEL
        elif real_length >= max_length:
            return OVERSIZE_LABEL
            
    return FALLBACK_LABEL

