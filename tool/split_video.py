"""
split_video.py

Tách video thành N phần bằng nhau.
Gắn địa chỉ video và số lần tách vào hằng số rồi chạy.
"""

import cv2
from pathlib import Path

# ─── CẤU HÌNH ───────────────────────────────────────────────
INPUT_VIDEO  = r"C:\videos\ten_video.mp4"   # đường dẫn video
NUM_CHUNKS   = 4                             # số phần muốn tách
# ────────────────────────────────────────────────────────────


def split_video(input_path: str, num_chunks: int) -> list[str]:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")
    if num_chunks < 2:
        raise ValueError("Số phần tách phải >= 2")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {src}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chunk_size = total // num_chunks          # số frame mỗi phần
    remainder  = total % num_chunks           # frame dư dồn vào phần cuối

    print(f"Input   : {src.name}")
    print(f"Kích thước : {width}x{height}  {fps:.1f}fps  {total} frames")
    print(f"Tách thành : {num_chunks} phần  (~{chunk_size} frames/phần)\n")

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    out_paths = []

    for chunk_idx in range(num_chunks):
        out_name = src.parent / f"{src.stem}_chunk_{chunk_idx + 1}{src.suffix}"
        out_paths.append(str(out_name))

        # Phần cuối lấy thêm số frame dư
        frames_in_chunk = chunk_size + (remainder if chunk_idx == num_chunks - 1 else 0)

        writer = cv2.VideoWriter(str(out_name), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Không tạo được file: {out_name}")

        print(f"  Chunk {chunk_idx + 1}/{num_chunks}  →  {out_name.name}", end="", flush=True)

        for i in range(frames_in_chunk):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            if (i + 1) % 100 == 0:
                pct = (i + 1) / frames_in_chunk * 100
                print(f"\r  Chunk {chunk_idx + 1}/{num_chunks}  →  {out_name.name}  ({pct:.1f}%)", end="", flush=True)

        writer.release()
        print(f"\r  Chunk {chunk_idx + 1}/{num_chunks}  →  {out_name.name}  hoàn tất  ({frames_in_chunk} frames)")

    cap.release()
    print(f"\nĐã tách xong {num_chunks} phần.")
    return out_paths


if __name__ == "__main__":
    split_video(INPUT_VIDEO, NUM_CHUNKS)