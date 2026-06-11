import cv2
from pathlib import Path

from ultralytics import YOLO


# Cấu hình
INPUT_VIDEO_DIR = r"C:\Users\RongDiBo\Pictures\b"
OUTPUT_FRAME_DIR = r"tool\d"
MODEL_DET = r"model\yolov8n_det_v6.pt"

# 0 = tự động lấy theo tổng số frame của từng video
# 1 = lấy 1 frame tốt nhất
# 5 = lấy 5 frame tốt nhất
NUM_FRAMES_PER_VIDEO = 1


def _get_video_files(video_dir: Path) -> list[Path]:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in video_exts
    )


def _get_max_conf(results) -> float | None:
    if not results or results[0].boxes is None or results[0].boxes.conf is None:
        return None

    confs = results[0].boxes.conf
    if len(confs) == 0:
        return None

    return float(confs.max().item())


def _save_best_frames_for_video(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    num_frames: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Không mở được video: {video_path}")
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Nếu NUM_FRAMES_PER_VIDEO = 0 thì lấy mặc định bằng tổng số frame của video
    if num_frames == 0:
        num_frames = total_video_frames

    if num_frames < 1:
        print(f"Không đọc được tổng số frame của video: {video_path.name}")
        cap.release()
        return 0

    print(f"Video {video_path.name} có {total_video_frames} frame")
    print(f"Sẽ lấy tối đa {num_frames} frame tốt nhất")

    best_items = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False)
        max_conf = _get_max_conf(results)

        if max_conf is not None:
            best_items.append((max_conf, frame_idx, frame.copy()))
            best_items.sort(key=lambda item: item[0], reverse=True)

            if len(best_items) > num_frames:
                best_items.pop()

        frame_idx += 1

    cap.release()

    if not best_items:
        print(f"Không tìm thấy detection trong video: {video_path.name}")
        return 0

    saved_count = 0
    total_digits = max(3, len(str(len(best_items))))

    for rank, (best_conf, best_frame_idx, best_frame) in enumerate(best_items, start=1):
        if len(best_items) == 1:
            output_path = output_dir / f"{video_path.stem}.jpg"
        else:
            output_path = output_dir / f"{video_path.stem}_{rank:0{total_digits}d}.jpg"

        if not cv2.imwrite(str(output_path), best_frame):
            raise RuntimeError(f"Không lưu được frame: {output_path}")

        print(
            f"Đã lưu {output_path.name} từ frame {best_frame_idx}, "
            f"conf={best_conf:.4f}"
        )
        saved_count += 1

    return saved_count


def main() -> None:
    if not INPUT_VIDEO_DIR:
        raise ValueError("Chưa cấu hình INPUT_VIDEO_DIR.")
    if not OUTPUT_FRAME_DIR:
        raise ValueError("Chưa cấu hình OUTPUT_FRAME_DIR.")
    if not MODEL_DET:
        raise ValueError("Chưa cấu hình MODEL_DET.")

    # Cho phép NUM_FRAMES_PER_VIDEO = 0
    if NUM_FRAMES_PER_VIDEO < 0:
        raise ValueError("NUM_FRAMES_PER_VIDEO phải >= 0.")

    video_dir = Path(INPUT_VIDEO_DIR)
    output_dir = Path(OUTPUT_FRAME_DIR)

    if not video_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục video: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"INPUT_VIDEO_DIR không phải thư mục: {video_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = _get_video_files(video_dir)
    if not video_files:
        print(f"Không tìm thấy video trong thư mục: {video_dir}")
        return

    model = YOLO(MODEL_DET, task="detect")

    total_saved = 0
    for video_path in video_files:
        print(f"\nĐang xử lý: {video_path.name}")
        total_saved += _save_best_frames_for_video(
            model,
            video_path,
            output_dir,
            NUM_FRAMES_PER_VIDEO,
        )

    print(f"\nHoàn tất. Đã lưu {total_saved} frame từ {len(video_files)} video.")


if __name__ == "__main__":
    main()