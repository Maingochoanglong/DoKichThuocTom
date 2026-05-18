"""
flow_utils.py
Các hàm tiện ích hỗ trợ cho toàn bộ hệ thống đo chiều dài tôm trên băng chuyền.

  ensure_dir       - Tạo thư mục nếu chưa tồn tại
  get_lines        - Tính 3 vạch tham chiếu (dọc hoặc ngang) để xác nhận tôm trôi qua
  box_touches_line - Kiểm tra tâm bounding box của tôm có chạm vạch không
  get_masked_image - Tạo ảnh nền đen, chỉ giữ lại vùng bounding box chứa tôm
"""

import numpy as np
from pathlib import Path

from config import BBOX_PAD


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_lines(frame_dim: int) -> dict[int, int]:
    # Tính toạ độ 3 vạch tham chiếu cách đều nhau dựa trên kích thước frame.
    # Truyền width  -> 3 vạch dọc,  tôm chạy ngang qua.
    # Truyền height -> 3 vạch ngang, tôm chạy dọc qua.
    center = frame_dim // 2
    gap_px = int(0.1 * frame_dim)
    return {0: center - gap_px, 1: center, 2: center + gap_px}


def box_touches_line(center: float, line_coord: int, threshold: int) -> bool:
    # Kiểm tra tâm bounding box (cx hoặc cy) có nằm trong ngưỡng chạm vạch không.
    return abs(center - line_coord) <= threshold


def get_masked_image(
    frame: np.ndarray,
    box_xyxy: np.ndarray | list,
    pad: int = BBOX_PAD,
) -> np.ndarray | None:
    # Tạo ảnh nền đen cùng kích thước frame, chỉ giữ lại vùng bounding box (+ padding).
    # Trả về None nếu vùng crop rỗng sau khi clamp về biên ảnh.
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    out = np.zeros_like(frame)
    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out