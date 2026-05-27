"""
main.py

Entry point của hệ thống đo chiều dài tôm tự động trên băng chuyền.
File này tải model, tạo queue giữa 6 flow, chạy mỗi flow trong một thread và
trả exit code 0 hoặc 1 cho app.py khi chạy dưới dạng subprocess.
"""

import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Full, Queue

import openvino as ov


BASE_DIR = Path(__file__).resolve().parent


def _setup_openvino_cache(cache_dir: Path) -> None:
    """
    Cấu hình cache OpenVINO trước khi Ultralytics tạo ov.Core.

    Hàm monkey patch ov.Core để mọi core mới đều dùng CACHE_DIR cố định trong
    thư mục dự án, giúp lần tải model sau nhanh hơn.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    original_core = ov.Core

    class CachedCore(original_core):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_property({"CACHE_DIR": str(cache_dir)})

    ov.Core = CachedCore


_setup_openvino_cache(BASE_DIR / "openvino_cache")

from ultralytics import YOLO

from config import MODEL_DET, MODEL_SEG, QUEUE_SIZE, load_config_values
from logger_setup import setup_logging
from pipeline import (
    flow1_read_input, flow2_detect_track, flow3_touch_logic,
    flow4_segment, flow5_longest_path, flow6_save_results,
)
from size import load_size_values


def _force_put_sentinel(q: Queue) -> None:
    """
    Ép đưa sentinel None vào queue ngay cả khi queue đang đầy.

    Khi một flow lỗi, helper này bỏ bớt item cũ nếu cần để các flow đang chờ
    queue có thể nhận None và thoát.
    """
    while True:
        try:
            q.put_nowait(None)
            return
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass


def _clear_output_dir(output_dir: str) -> bool:
    """
    Xóa toàn bộ output_dir rồi tạo lại thư mục.

    Trả về True nếu thư mục đã tồn tại trước khi xóa, để main() có thể log
    rõ việc CLEAR_OUTPUT đã thực sự dọn output cũ.
    """
    p = Path(output_dir)
    existed = p.exists()
    if existed:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return existed


def _safe_thread(
    target,
    args: tuple,
    thread_name: str,
    error_info: list,
    error_lock: threading.Lock,
    all_queues: list[Queue],
) -> None:
    """
    Chạy một flow và chuyển lỗi thành trạng thái pipeline có thể báo cáo.

    Nếu target ném exception, hàm lưu tên thread, exception và traceback vào
    error_info, sau đó gửi sentinel None tới toàn bộ queue để các flow khác
    không bị treo khi join thread.
    """
    try:
        target(*args)
    except Exception as exc:
        tb = traceback.format_exc()
        with error_lock:
            error_info.append((thread_name, exc, tb))
        for q in all_queues:
            _force_put_sentinel(q)


def _report_pipeline_errors(log, error_info: list[tuple]) -> None:
    """Ghi toàn bộ exception của các flow vào pipeline.log."""
    sep = "=" * 60
    log.error(f"\n{sep}")
    log.error(f"  LỖI PIPELINE - {len(error_info)} luồng gặp sự cố")
    log.error(sep)
    for name, exc, tb in error_info:
        log.error(f"\n  [{name}] {type(exc).__name__}: {exc}")
        log.error(f"  {'-' * 56}")
        for line in tb.strip().splitlines():
            log.error(f"    {line}")
    log.error(f"\n{sep}")
    log.error("Pipeline dừng do lỗi. Kiểm tra chi tiết ở trên.")


def _report_flow_times(log, flow_times: dict[str, float], elapsed: float) -> None:
    """Ghi bảng thời gian từng flow và tổng thời gian chạy pipeline."""
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
    log.info(f"\n{sep}")
    log.info("  THỜI GIAN THỰC THI TỪNG LUỒNG")
    log.info(f"{'-' * 52}")
    for key in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        val = flow_times.get(key)
        if val is not None:
            log.info(f"  {key}  {labels[key]}  {val:>8.2f} s")
    log.info(f"{'-' * 52}")
    log.info(f"  TỔNG CỘNG                    {elapsed:>8.2f} s  ({time_str})")
    log.info(f"{sep}")


def main() -> int:
    """
    Chạy pipeline hoàn chỉnh và trả exit code.

    Luồng chính gồm dọn output nếu cần, setup logger, tải hai model YOLO, tạo
    queue F1 đến F6, chạy thread, gom lỗi và ghi thời gian. Trả 0 khi thành
    công, trả 1 nếu bất kỳ flow nào ném exception.
    """
    t_start = time.perf_counter()
    cfg = load_config_values()
    size_cfg = load_size_values()
    output_dir = cfg["OUTPUT_DIR"]
    output_cleared = _clear_output_dir(output_dir) if cfg["CLEAR_OUTPUT"] else False

    log = setup_logging(output_dir)
    if output_cleared:
        log.info(f"Đã xóa output cũ: {output_dir}/")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_dir = str(Path(output_dir) / timestamp)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    log.info("Đang tải mô hình 1 - Phát hiện (Detect)...")
    model_det = YOLO(MODEL_DET, task="detect")

    log.info("Đang tải mô hình 2 - Phân đoạn (Segment)...")
    model_seg = YOLO(MODEL_SEG, task="segment")

    log.info("Tải mô hình hoàn tất.")
    log.info(
        f"SAVE={cfg['SAVE']}  "
        f"CLEAR_INPUT={cfg['CLEAR_INPUT']}  "
        f"CLEAR_OUTPUT={cfg['CLEAR_OUTPUT']}"
    )
    log.info(f"Input  : {cfg['INPUT_DIR']}/")
    log.info(f"Output : {run_dir}/")

    flow_times: dict[str, float] = {}
    error_info: list[tuple] = []
    error_lock = threading.Lock()

    q_f1_f2 = Queue(maxsize=QUEUE_SIZE)
    q_f2_f3 = Queue(maxsize=QUEUE_SIZE)
    q_f3_f4 = Queue(maxsize=QUEUE_SIZE)
    q_f4_f5 = Queue(maxsize=QUEUE_SIZE)
    q_f5_f6 = Queue(maxsize=QUEUE_SIZE)
    all_queues = [q_f1_f2, q_f2_f3, q_f3_f4, q_f4_f5, q_f5_f6]

    thread_defs = [
        ("F1", flow1_read_input,   (q_f1_f2, flow_times, run_dir, cfg)),
        ("F2", flow2_detect_track, (model_det, q_f1_f2, q_f2_f3, flow_times, cfg)),
        ("F3", flow3_touch_logic,  (q_f2_f3, q_f3_f4, flow_times, cfg)),
        ("F4", flow4_segment,      (model_seg, q_f3_f4, q_f4_f5, flow_times, cfg)),
        ("F5", flow5_longest_path, (q_f4_f5, q_f5_f6, flow_times, cfg)),
        ("F6", flow6_save_results, (q_f5_f6, flow_times, cfg, size_cfg)),
    ]

    threads = []
    for name, target, args in thread_defs:
        t = threading.Thread(
            target=_safe_thread,
            args=(target, args, name, error_info, error_lock, all_queues),
            name=name,
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if error_info:
        _report_pipeline_errors(log, error_info)
        return 1

    elapsed = time.perf_counter() - t_start
    _report_flow_times(log, flow_times, elapsed)
    log.info(f"Kết quả đã lưu tại: {run_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
