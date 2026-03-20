import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# CONFIG
INPUT_DIR  = "input"
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"
SCALE      = 1   # mm/pixel

IMG_EXTS   = {".jpg", ".jpeg", ".png"}

COLORS = [
    (  0,   0, 255),   # đỏ
    (  0, 255,   0),   # xanh lá
    (255,   0,   0),   # xanh dương
    (  0, 200, 255),   # cam
    (255,   0, 200),   # hồng
]

# HÀM BFS
def bfs(skeleton, start):
    h, w     = skeleton.shape
    visited  = {start: None}
    queue    = deque([start])
    farthest = start
    while queue:
        y, x = queue.popleft()
        farthest = (y, x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w \
                        and skeleton[ny, nx] and (ny, nx) not in visited:
                    visited[(ny, nx)] = (y, x)
                    queue.append((ny, nx))
    return farthest, visited

# MAIN
import os
from pathlib import Path

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chạy mô hình YOLO
model      = YOLO(MODEL_PATH)

# Lấy danh địa chỉ ảnh trong thư mục
img_paths  = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in IMG_EXTS]

print(f"[OK] Tim thay {len(img_paths)} anh trong '{INPUT_DIR}/'")

for img_path in img_paths:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[!] Bo qua (khong doc duoc): {img_path.name}")
        continue

    # h: độ cao ảnh đơn vị px
    # w: độ rộng ảnh đơn vị px
    results = model(image, verbose=False, retina_masks=True)[0]
    canvas  = image.copy()

    if results.masks is None:
        print(f"  {img_path.name}: khong co mask")
        cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), canvas)
        continue

    for i, mask_tensor in enumerate(results.masks.data):

        # 1. Lấy mask
        mask_u8 = (mask_tensor.numpy() > 0.5).astype(np.uint8) * 255

        # 2. Thinning đến skeleton
        skeleton = cv2.ximgproc.thinning(mask_u8) > 0

        # 3. Lấy 1 điểm bất kỳ trên skeleton
        ys, xs = np.where(skeleton)
        if len(ys) == 0:
            continue
        start = (ys[0], xs[0])

        # 4. BFS lần 1: tìm điểm A xa nhất
        pt_A, _ = bfs(skeleton, start)

        # 5. BFS lần 2: từ A tìm đường dài nhất đến B
        pt_B, visited = bfs(skeleton, pt_A)

        # 6. Truy ngược đường đi A đến B
        path, node = [], pt_B
        while node is not None:
            path.append(node)
            node = visited[node]
        path.reverse()

        # 7. Vẽ đường + ghi pixel lên ảnh
        clr = COLORS[i % len(COLORS)]
        for y, x in path:
            cv2.circle(canvas, (x, y), 2, clr, -1)

        mid = path[len(path) // 2]
        cv2.putText(canvas, f"#{i+1}  {len(path)}px  {len(path)*SCALE:.1f}mm",
                    (mid[1] + 6, mid[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"#{i+1}  {len(path)}px  {len(path)*SCALE:.1f}mm",
                    (mid[1] + 6, mid[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr,   1, cv2.LINE_AA)

        print(f"  {img_path.name}  con #{i+1}: {len(path)} px = {len(path)*SCALE:.2f} mm")

    # Lưu ảnh kết quả
    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), canvas)
    print(f"   Da luu: {out_path}")
