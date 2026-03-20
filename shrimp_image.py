import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import os
from pathlib import Path

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

model     = YOLO(MODEL_PATH)
img_paths = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in IMG_EXTS]

print(f"[OK] Tim thay {len(img_paths)} anh trong '{INPUT_DIR}/'")

for img_path in sorted(img_paths):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[!] Bo qua (khong doc duoc): {img_path.name}")
        continue

    results = model(image, verbose=False, retina_masks=True)[0]

    # 3 canvas riêng biệt
    canvas_result   = image.copy()
    canvas_mask     = image.copy()
    canvas_skeleton = image.copy()

    if results.masks is None:
        print(f"  {img_path.name}: khong co mask")
        cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), canvas_result)
        continue

    for i, mask_tensor in enumerate(results.masks.data):

        # 1. Lấy mask
        # mask_tensor là tensor PyTorch kiểu float32, giá trị từ 0.0 đến 1.0
        # .numpy()        : chuyển tensor sang numpy để OpenCV xử lý được
        # > 0.5           : chuyển float sang bool (True = tôm, False = nền)
        # .astype(uint8)  : chuyển bool sang số nguyên (True=1, False=0)
        # * 255           : chuyển sang ảnh trắng đen (255 = tôm, 0 = nền)
        #                   vì cv2.ximgproc.thinning chỉ nhận uint8 với giá trị 0 hoặc 255
        mask_u8  = (mask_tensor.numpy() > 0.5).astype(np.uint8) * 255

        # 2. Thinning den skeleton
        # cv2.ximgproc.thinning trả về uint8 (0 hoặc 255)
        # > 0 : chuyển sang bool vì hàm bfs kiểm tra skeleton[ny, nx] theo kiểu bool
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

        clr = COLORS[i % len(COLORS)]

        # 7. Vẽ mask lên canvas_mask
        overlay = canvas_mask.copy()
        overlay[mask_u8 > 127] = clr
        cv2.addWeighted(overlay, 0.4, canvas_mask, 0.6, 0, canvas_mask)

        # 8. Vẽ skeleton lên canvas_skeleton
        skel_ys, skel_xs = np.where(skeleton)
        for sy, sx in zip(skel_ys.tolist(), skel_xs.tolist()):
            cv2.circle(canvas_skeleton, (sx, sy), 1, clr, -1)

        # 9. Vẽ đường + ghi pixel lên canvas_result
        for y, x in path:
            cv2.circle(canvas_result, (x, y), 2, clr, -1)

        mid = path[len(path) // 2]
        cv2.putText(canvas_result, f"#{i+1}  {len(path)}px  {len(path)*SCALE:.1f}mm",
                    (mid[1] + 6, mid[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas_result, f"#{i+1}  {len(path)}px  {len(path)*SCALE:.1f}mm",
                    (mid[1] + 6, mid[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1, cv2.LINE_AA)

        print(f"  {img_path.name}  con #{i+1}: {len(path)} px = {len(path)*SCALE:.2f} mm")

    # Lưu 3 ảnh với tên khác nhau
    stem = img_path.stem
    ext  = img_path.suffix

    cv2.imwrite(str(Path(OUTPUT_DIR) / f"{stem}_result{ext}"),   canvas_result)
    cv2.imwrite(str(Path(OUTPUT_DIR) / f"{stem}_mask{ext}"),     canvas_mask)
    cv2.imwrite(str(Path(OUTPUT_DIR) / f"{stem}_skeleton{ext}"), canvas_skeleton)
    print(f"  Da luu: {stem}_result{ext}, {stem}_mask{ext}, {stem}_skeleton{ext}")