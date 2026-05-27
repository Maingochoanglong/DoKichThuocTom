"""
flow_utils.py

Các helper dùng chung cho các flow đo chiều dài tôm.
Các hàm trong file này không ghi file và không giữ trạng thái toàn cục.
"""

from pathlib import Path
from typing import Sequence

import numpy as np


def ensure_dir(path: Path) -> None:
    """Tạo thư mục đích nếu chưa tồn tại."""
    path.mkdir(parents=True, exist_ok=True)


def get_lines(frame_dim: int) -> dict[int, int]:
    """
    Tính 3 vạch tham chiếu cách đều quanh tâm frame.

    frame_dim là width khi băng chuyền ngang hoặc height khi băng chuyền dọc.
    Kết quả dùng key 0, 1, 2 để F3 kiểm tra tôm đã đi qua đủ vạch.
    """
    center = frame_dim // 2
    gap_px = int(0.1 * frame_dim)
    return {0: center - gap_px, 1: center, 2: center + gap_px}


def box_touches_line(center: float, line_coord: int, threshold: int) -> bool:
    """Kiểm tra tâm bbox có nằm trong ngưỡng chạm vạch tham chiếu không."""
    return abs(center - line_coord) <= threshold


def get_masked_image(
    frame: np.ndarray,
    box_xyxy: Sequence[float] | np.ndarray,
    pad: int,
) -> np.ndarray | None:
    """
    Tạo ảnh nền xám và chỉ giữ vùng bbox đã mở rộng bằng pad.

    Hàm trả None nếu bbox sau khi clamp theo biên ảnh bị rỗng. Ảnh trả về có
    cùng kích thước và dtype với frame, dùng làm input sạch hơn cho model sau.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    out = np.full_like(frame, 114)
    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out
