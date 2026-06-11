from pathlib import Path
import subprocess
import imageio_ffmpeg

# =========================
# HẰNG SỐ CẤU HÌNH
# =========================
INPUT_FOLDER = r"C:\Users\RongDiBo\Pictures\b"
OUTPUT_FOLDER = r"D:\DoAnTotNghiep\tool\a"

INPUT_EXTENSIONS = [".mov", ".avi", ".mkv", ".mp4"]
OUTPUT_EXTENSION = ".mp4"


def convert_video(input_path: Path, output_path: Path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),

        # Lấy video đầu tiên và audio nếu có
        "-map", "0:v:0",
        "-map", "0:a?",

        # Chỉ ép video sang H.264, còn lại để mặc định
        "-c:v", "libx264",

        # Giữ nguyên audio gốc
        "-c:a", "copy",

        str(output_path)
    ]

    print(f"Đang chuyển sang H.264: {input_path.name} -> {output_path.name}")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode == 0 and output_path.exists():
        print(f"Xong: {output_path}")
    else:
        print(f"Lỗi khi chuyển: {input_path}")
        print(result.stderr)


def main():
    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)

    if not input_dir.exists():
        print(f"Thư mục đầu vào không tồn tại: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        file for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in INPUT_EXTENSIONS
    ]

    if not video_files:
        print("Không tìm thấy video nào trong thư mục đầu vào.")
        return

    for video_file in video_files:
        output_file = output_dir / f"{video_file.stem}{OUTPUT_EXTENSION}"
        convert_video(video_file, output_file)

    print("Hoàn tất chuyển đổi toàn bộ video.")


if __name__ == "__main__":
    main()