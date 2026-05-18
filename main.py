"""
main.py
Entry point của hệ thống đo chiều dài tôm tự động trên băng chuyền.
"""

import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Queue

import openvino as ov
os.makedirs("openvino_cache", exist_ok=True)
_original_core = ov.Core

class CachedCore(_original_core):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_property({"CACHE_DIR": "openvino_cache"})

ov.Core = CachedCore
from ultralytics import YOLO

from config import (
    CLEAR_INPUT, CLEAR_OUTPUT,
    INPUT_DIR, MODEL_DET, MODEL_SEG, OUTPUT_DIR, SAVE,
)
from logger_setup import get_logger, setup_logging
from pipeline import (
    flow1_read_input, flow2_detect_track, flow3_touch_logic,
    flow4_segment, flow5_longest_path, flow6_save_results,
)


# Xóa output cũ
def _clear_output_dir(output_dir: str) -> None:
    """Xóa toàn bộ nội dung trong thư mục output trước mỗi lần chạy mới."""
    p = Path(output_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# Wrapper chạy thread an toàn
def _safe_thread(
    target,
    args: tuple,
    thread_name: str,
    error_event: threading.Event,
    error_info: list,
    error_lock: threading.Lock,
    all_queues: list[Queue],
) -> None:
    """
    Bọc hàm target trong try/except.
    Khi lỗi xảy ra:
      1. Lưu thông tin lỗi (tên luồng, exception, traceback).
      2. Set error_event để báo hiệu cho các luồng khác.
      3. Đẩy None sentinel vào TẤT CẢ queue để unblock các luồng đang chờ.
    """
    try:
        target(*args)
    except Exception as exc:
        tb = traceback.format_exc()
        with error_lock:
            error_info.append((thread_name, exc, tb))
        error_event.set()
        for q in all_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass


# Main
def main() -> None:
    t_start = time.perf_counter()

    # Xóa output cũ nếu CLEAR_OUTPUT=True
    if CLEAR_OUTPUT:
        _clear_output_dir(OUTPUT_DIR)

    # Tạo thư mục output theo timestamp: output/<timestamp>/
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_dir   = str(Path(OUTPUT_DIR) / timestamp)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Khởi tạo logger — file pipeline.log nằm ngay tại run_dir
    log = setup_logging(run_dir)

    log.info("Đang tải mô hình 1 — Phát hiện (Detect)...")
    model_det = YOLO(MODEL_DET, task="detect")

    log.info("Đang tải mô hình 2 — Phân đoạn (Segment)...")
    model_seg = YOLO(MODEL_SEG, task="segment")

    log.info("Tải mô hình hoàn tất.")
    log.info(f"SAVE={SAVE}  CLEAR_INPUT={CLEAR_INPUT}  CLEAR_OUTPUT={CLEAR_OUTPUT}")
    log.info(f"Input  : {INPUT_DIR}/")
    log.info(f"Output : {run_dir}/")

    flow_times: dict[str, float] = {}
    error_event = threading.Event()
    error_info: list[tuple] = []
    error_lock = threading.Lock()

    # Queues
    q_f1_f2 = Queue()
    q_f2_f3 = Queue()
    q_f3_f4 = Queue()
    q_f4_f5 = Queue()
    q_f5_f6 = Queue()
    all_queues = [q_f1_f2, q_f2_f3, q_f3_f4, q_f4_f5, q_f5_f6]

    # Pipeline threads
    thread_defs = [
        ("F1", flow1_read_input,   (q_f1_f2, flow_times, run_dir)),
        ("F2", flow2_detect_track, (model_det, q_f1_f2, q_f2_f3, flow_times)),
        ("F3", flow3_touch_logic,  (q_f2_f3, q_f3_f4, flow_times)),
        ("F4", flow4_segment,      (model_seg, q_f3_f4, q_f4_f5, flow_times)),
        ("F5", flow5_longest_path, (q_f4_f5, q_f5_f6, flow_times)),
        ("F6", flow6_save_results, (q_f5_f6, flow_times)),
    ]

    threads = []
    for name, target, args in thread_defs:
        t = threading.Thread(
            target=_safe_thread,
            args=(target, args, name, error_event, error_info, error_lock, all_queues),
            name=name, daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Kiểm tra lỗi
    if error_info:
        sep = "=" * 60
        log.error(sep)
        log.error(f"LỖI PIPELINE — {len(error_info)} luồng gặp sự cố")
        log.error(sep)
        for name, exc, tb in error_info:
            log.error(f"[{name}] {type(exc).__name__}: {exc}")
            for line in tb.strip().splitlines():
                log.error(f"  {line}")
        log.error("Pipeline dừng do lỗi. Kiểm tra chi tiết ở trên.")
        sys.exit(1)

    # Báo cáo thời gian
    elapsed  = time.perf_counter() - t_start
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    labels = {
        "F1": "Đọc input (Ảnh/Video)  ",
        "F2": "Phát hiện & Theo dõi   ",
        "F3": "Kiểm tra chạm đường    ",
        "F4": "Phân đoạn (Segment)    ",
        "F5": "Tính skeleton & BFS    ",
        "F6": "Lưu kết quả & JSON     ",
    }
    sep = "=" * 52
    log.info(sep)
    log.info("THỜI GIAN THỰC THI TỪNG LUỒNG")
    log.info("-" * 52)
    for key in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        val = flow_times.get(key)
        if val is not None:
            log.info(f"{key}  {labels[key]}  {val:>8.2f} s")
    log.info("-" * 52)
    log.info(f"TỔNG CỘNG                           {elapsed:>8.2f} s  ({time_str})")
    log.info(sep)
    log.info(f"Kết quả đã lưu tại: {run_dir}/")

if __name__ == "__main__":
    main()