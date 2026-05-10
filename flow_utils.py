"""
flow_utils.py
Các hàm tiện ích hỗ trợ cho toàn bộ hệ thống đo chiều dài tôm trên băng chuyền.

  ensure_dir       – Tạo thư mục lưu trữ
  get_lines        – Tính 3 vạch tham chiếu (dọc hoặc ngang) để xác nhận tôm trôi qua
  box_touches_line – Kiểm tra tâm bounding box của tôm có chạm vạch không
  get_masked_image – Che nền đen bên ngoài vùng tôm, giữ lại phần thân cần đo
"""


import os
from pathlib import Path
from config import BBOX_PAD

import numpy as np

def ensure_dir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def get_lines(size: int) -> dict[int, int]:
    # Tính toạ độ 3 vạch tham chiếu dựa trên kích thước frame.
    # Truyền width  -> 3 vạch dọc  cách đều nhau, tôm chạy ngang qua.
    # Truyền height -> 3 vạch ngang cách đều nhau, tôm chạy dọc qua.
    center = size // 2
    gap    = int(0.1 * size)
    return {0: center - gap, 1: center, 2: center + gap}


def box_touches_line(center: float, line_coord: int, threshold: int) -> bool:
    # Kiểm tra tâm bounding box (cx hoặc cy) có nằm trong ngưỡng chạm vạch không.
    return abs(center - line_coord) <= threshold


def get_masked_image(frame: np.ndarray, box_xyxy, pad = BBOX_PAD) -> np.ndarray | None:
    # Trả về bản sao frame với toàn bộ vùng ngoài bounding box (+ padding) bị tô đen.
    # Trả về None nếu vùng crop không hợp lệ (diện tích bằng 0).
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    out = np.zeros_like(frame)
    out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out