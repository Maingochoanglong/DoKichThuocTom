"""
skeleton_utils.py

Tính chiều dài tôm từ skeleton (medial axis) đã được tính sẵn ở F5.
Dùng BFS 2 lần trên skeleton để tìm đường đi dài nhất giữa 2 đầu mút,
từ đó suy ra chiều dài thân tôm tính bằng pixel.
"""
import math
from collections import deque

import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
SQRT2 = math.sqrt(2)


def _bfs(skeleton: np.ndarray, start: tuple) -> tuple[tuple, dict]:
    h, w    = skeleton.shape
    visited = np.zeros((h, w), dtype=bool)
    parent  = {}

    q = deque([start])
    visited[start] = True
    farthest = start

    while q:
        y, x = q.popleft()
        farthest = (y, x)

        for dy, dx in DIRECTIONS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and not visited[ny, nx]:
                visited[ny, nx]    = True
                parent[(ny, nx)]   = (y, x)
                q.append((ny, nx))

    return farthest, parent


def find_longest_path(skeleton: np.ndarray) -> tuple[np.ndarray, float]:
    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=bool), 0.0

    start        = tuple(np.argwhere(skeleton)[0])
    pt_A, _      = _bfs(skeleton, start)   # lần 1: tìm đầu mút A
    pt_B, parent = _bfs(skeleton, pt_A)    # lần 2: tìm đầu mút B

    path_mask    = np.zeros_like(skeleton, dtype=bool)
    total_length = 0.0
    node         = pt_B

    while node in parent:
        path_mask[node] = True
        prev = parent[node]
        dy   = abs(node[0] - prev[0])
        dx   = abs(node[1] - prev[1])
        total_length += SQRT2 if dy == 1 and dx == 1 else 1.0
        node = prev

    path_mask[node] = True   # đánh dấu điểm đầu pt_A
    return path_mask, total_length