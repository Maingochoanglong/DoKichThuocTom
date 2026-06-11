"""
rotate_video.py

Xoay video 90 độ ngược chiều kim đồng hồ (phải sang trái).
Gắn địa chỉ video vào hằng số INPUT_VIDEO rồi chạy.
"""

import cv2
from pathlib import Path

# ─── CẤU HÌNH ───────────────────────────────────────────────
INPUT_VIDEO = r"D:\DoKichThuocTom\input\IMG_0133.mp4"   # gắn đường dẫn vào đây
# ────────────────────────────────────────────────────────────

SUFFIX_NAME = "xoay_trai_90"
ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE


def rotate_video(input_path: str) -> str:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {src}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path   = src.parent / f"{src.stem}_{SUFFIX_NAME}{src.suffix}"

    # Sau khi xoay, width và height hoán đổi cho nhau
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (height, width))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được file output: {out_path}")

    print(f"Input  : {src.name}  ({width}x{height}  {fps:.1f}fps  {total} frames)")
    print(f"Output : {out_path.name}  ({height}x{width})")
    print(f"Đang xoay", end="", flush=True)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.rotate(frame, ROTATE_CODE))
        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"\rĐang xoay  {frame_idx}/{total}  ({pct:.1f}%)", end="", flush=True)

    cap.release()
    writer.release()

    print(f"\rHoàn tất   {frame_idx} frames  →  {out_path.name}          ")
    return str(out_path)


if __name__ == "__main__":
    rotate_video(INPUT_VIDEO)