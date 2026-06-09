"""
skeleton_utils.py

Tìm đường dài nhất trên skeleton bằng BFS 2 lần.
File này không ghi dữ liệu ra đĩa, chỉ nhận skeleton nhị phân và trả lại mask
đường đi dài nhất cùng độ dài pixel.
"""

from math import sqrt
from collections import deque

import numpy as np


DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

SQRT = sqrt(2.0)


def _bfs(
    skeleton: np.ndarray,
    start: tuple[int, int],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    """
    Duyệt skeleton từ start bằng BFS trên 8 hướng.

    Trả về điểm xa nhất gặp được và map parent để caller dựng lại đường đi từ
    điểm xa nhất về điểm xuất phát.
    """
    h, w = skeleton.shape
    visited = np.zeros((h, w), dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    q = deque([start])
    visited[start] = True
    farthest = start

    while q:
        y, x = q.popleft()
        farthest = (y, x)

        for dy, dx in DIRECTIONS:
            ny, nx = y + dy, x + dx
            inside = 0 <= ny < h and 0 <= nx < w
            if inside and skeleton[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                parent[(ny, nx)] = (y, x)
                q.append((ny, nx))

    return farthest, parent


def find_longest_path(skeleton: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Tìm đường dài nhất trên skeleton bằng cách chạy BFS 2 lần.

    Tham số skeleton là ảnh nhị phân hoặc bool, trong đó pixel True là phần xương tôm.
    Hàm trả về path_mask đánh dấu đường dài nhất và total_length là số pixel trên đường đi,
    mỗi bước tính 1 px bất kể hướng ngang, dọc và căn 2 cho hướng chéo. Nếu skeleton rỗng thì trả về
    mask rỗng và 0.0.
    """
    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=bool), 0.0

    start: tuple[int, int] = tuple(np.argwhere(skeleton)[0])
    point_a, _ = _bfs(skeleton, start)
    point_b, parent = _bfs(skeleton, point_a)

    path_mask = np.zeros_like(skeleton, dtype=bool)
    total_length = 0.0
    node = point_b

    while node in parent:
        path_mask[node] = True
        prev = parent[node]
        dy = abs(node[0] - prev[0])
        dx = abs(node[1] - prev[1])
        total_length += SQRT if dy == 1 and dx == 1 else 1.0
        node = prev

    path_mask[node] = True
    return path_mask, total_length