"""
pipeline.py

Logic cốt lõi của pipeline đo tôm gồm 6 flow F1 đến F6.
Các flow giao tiếp qua Queue bằng item dict và dùng sentinel None để báo hết
dữ liệu. File này đọc input, chạy detect/track/segment, tính pixel length,
quy đổi sang mm, phân loại kích cỡ và ghi JSON kết quả.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full

import cv2
import numpy as np
import supervision as sv
from skimage.morphology import medial_axis

from config import DEVICE
from draw_utils import draw_f6_result, save_f3_debug, save_f4_debug, save_f5_debug
from flow_utils import box_touches_line, get_lines, get_masked_image
from logger_setup import get_logger
from size import classify_size
from skeleton_utils import find_longest_path


QUEUE_WAIT_SECONDS = 0.2


def _stop_requested(stop_event) -> bool:
    """Kiểm tra tín hiệu dừng chung của pipeline."""
    return stop_event is not None and stop_event.is_set()


def _put_queue(q, item, stop_event) -> bool:
    """Đẩy item vào queue nhưng thoát sớm nếu pipeline đã báo dừng."""
    while not _stop_requested(stop_event):
        try:
            q.put(item, timeout=QUEUE_WAIT_SECONDS)
            return True
        except Full:
            pass
    return False


def _get_queue(q, stop_event):
    """Lấy item từ queue nhưng không chờ vô hạn khi pipeline đã báo dừng."""
    while True:
        if _stop_requested(stop_event):
            return None
        try:
            return q.get(timeout=QUEUE_WAIT_SECONDS)
        except Empty:
            pass


# F1: Đọc input
def flow1_read_input(
    q_f1_f2,
    flow_times: dict[str, float],
    run_dir: str,
    cfg: dict,
    stop_event,
) -> None:
    """
    Đọc ảnh hoặc video từ INPUT_DIR và đẩy frame sang F2.

    Ảnh tạo một item có type `image`, frame_idx 0 và không có line tham chiếu.
    Video tạo nhiều item type `video`, mỗi item có frame, frame_idx, lines và
    run_dir. Khi quét xong hoặc INPUT_DIR không tồn tại, hàm gửi sentinel None
    vào q_f1_f2 và ghi thời gian vào flow_times["F1"].
    """
    log = get_logger()
    start_time = time.perf_counter()
    input_dir = cfg["INPUT_DIR"]
    target_fps = cfg["TARGET_FPS"]
    conveyor_vertical = cfg["CONVEYOR_VERTICAL"]
    line_gap_ratio = cfg["LINE_GAP_RATIO"]
    img_exts = set(cfg["IMG_EXTS"])
    vid_exts = set(cfg["VID_EXTS"])
    input_path = Path(input_dir)
    stop_logged = False

    def _should_stop() -> bool:
        nonlocal stop_logged
        if not _stop_requested(stop_event):
            return False
        if not stop_logged:
            log.info("[F1] Nhận tín hiệu dừng do flow khác lỗi, ngừng đọc input.")
            stop_logged = True
        return True

    if not input_path.exists():
        log.warning(f"[F1] Thư mục '{input_dir}' không tồn tại!")
        _put_queue(q_f1_f2, None, stop_event)
        flow_times["F1"] = time.perf_counter() - start_time
        return

    def _push_image(fpath: Path) -> bool:
        if _should_stop():
            return False
        image = cv2.imread(str(fpath))
        if image is None:
            log.warning(f"[F1] Bỏ qua '{fpath.name}' - lỗi đọc ảnh")
            return True
        pushed = _put_queue(q_f1_f2, {
            "type": "image",
            "path": fpath,
            "source_file": fpath.name,
            "source_stem": fpath.stem,
            "frame": image,
            "frame_idx": 0,
            "lines": {},
            "run_dir": run_dir,
        }, stop_event)
        if not pushed:
            _should_stop()
            return False
        log.info(f"[F1] Đã đọc ảnh: {fpath.name}")
        return True

    def _push_video(fpath: Path) -> bool:
        if _should_stop():
            return False
        try:
            info = sv.VideoInfo.from_video_path(str(fpath))
        except Exception:
            log.warning(f"[F1] Bỏ qua '{fpath.name}' - lỗi đọc video")
            return True

        fps = info.fps or 30.0
        step = max(1, round(fps / target_fps)) if target_fps > 0 else 1
        frame_dim = info.height if conveyor_vertical else info.width
        lines = get_lines(frame_dim, line_gap_ratio)

        log.info(
            f"[F1] Đọc video: {fpath.name} | "
            f"{info.width}x{info.height} | lines={dict(lines)}"
        )
        for i, frame in enumerate(sv.get_video_frames_generator(str(fpath), stride=step)):
            if _should_stop():
                return False
            pushed = _put_queue(q_f1_f2, {
                "type": "video",
                "path": fpath,
                "source_file": fpath.name,
                "source_stem": fpath.stem,
                "frame": frame,
                "frame_idx": i * step + 1,
                "lines": lines,
                "run_dir": run_dir,
            }, stop_event)
            if not pushed:
                _should_stop()
                return False
        log.info(f"[F1] Đọc xong: {fpath.name}")
        return True

    def _process_file(fpath: Path) -> bool:
        suffix = fpath.suffix.lower()
        if suffix in img_exts:
            return _push_image(fpath)
        elif suffix in vid_exts:
            return _push_video(fpath)
        return True

    log.info(f"[F1] Quét thư mục '{input_dir}'")
    for fpath in sorted(input_path.iterdir()):
        if _should_stop() or not _process_file(fpath):
            break

    _put_queue(q_f1_f2, None, stop_event)
    flow_times["F1"] = time.perf_counter() - start_time
    if _stop_requested(stop_event):
        log.info(f"[F1] Dừng sớm  |  {flow_times['F1']:.2f}s")
    else:
        log.info(f"[F1] Hoàn tất  |  {flow_times['F1']:.2f}s")


# F2: Phát hiện & bám vết
def flow2_detect_track(model_det, q_f1_f2, q_f2_f3, flow_times: dict[str, float], cfg: dict, stop_event) -> None:
    """
    Chạy YOLO detect và gán track_id trước khi chuyển sang F3.

    Với ảnh tĩnh, mỗi detection hợp lệ tạo một item riêng có masked_img,
    orig_img, track_id và debug_images rỗng. Với video, hàm dùng ByteTrack để
    cập nhật tracker_id và chỉ chuyển các frame có detection. Khi nhận sentinel
    None từ F1, hàm gửi None sang q_f2_f3.
    """
    log = get_logger()
    start_time = time.perf_counter()
    bbox_pad = cfg["BBOX_PAD"]
    chunk_mode = cfg["CHUNK_MODE"]
    conf_det = cfg["CONF_DET"]
    tracker: sv.ByteTrack | None = None
    current_video = None

    while True:
        item = _get_queue(q_f1_f2, stop_event)
        if item is None:
            _put_queue(q_f2_f3, None, stop_event)
            break

        results = model_det.predict(
            source=item["frame"],
            verbose=False,
            conf=conf_det,
            device=DEVICE,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if item["type"] == "image":
            if len(detections) > 0:
                log.info(f"[F2] {item['source_file']}: phát hiện {len(detections)} con tôm")
                for i, box_xyxy in enumerate(detections.xyxy):
                    masked_img = get_masked_image(item["frame"], box_xyxy, pad=bbox_pad)
                    if masked_img is not None:
                        pushed = _put_queue(q_f2_f3, {
                            "type": "image",
                            "source_file": item["source_file"],
                            "source_stem": item["source_stem"],
                            "track_id": i + 1,
                            "frame_idx": 0,
                            "masked_img": masked_img,
                            "orig_img": item["frame"],
                            "lines": {},
                            "debug_images": {},
                            "run_dir": item["run_dir"],
                        }, stop_event)
                        if not pushed:
                            break

        elif item["type"] == "video":
            if item["path"] != current_video:
                if tracker is None or not chunk_mode:
                    tracker = sv.ByteTrack()
                    log.info(f"[F2] Tracker mới cho: {item['path'].name}")
                current_video = item["path"]

            detections = tracker.update_with_detections(detections)
            if len(detections) > 0:
                item["detections"] = detections
                if not _put_queue(q_f2_f3, item, stop_event):
                    break

    flow_times["F2"] = time.perf_counter() - start_time
    log.info(f"[F2] Hoàn tất  |  {flow_times['F2']:.2f}s")


# F3: Kiểm tra chạm vạch
def flow3_touch_logic(q_f2_f3, q_f3_f4, flow_times: dict[str, float], cfg: dict, stop_event) -> None:
    """
    Chọn frame tốt nhất trước khi segmentation.

    Ảnh tĩnh được chuyển thẳng sang F4. Với video, hàm theo dõi từng track,
    ghi nhận các vạch đã chạm, lưu frame có bbox lớn nhất và chỉ gửi sang F4
    khi track chạm đủ 3 vạch. Khi video đổi hoặc nhận sentinel None, các track
    còn hoạt động được flush sang F4 nếu có frame tốt nhất.
    """
    log = get_logger()
    start_time = time.perf_counter()
    bbox_pad = cfg["BBOX_PAD"]
    chunk_mode = cfg["CHUNK_MODE"]
    conveyor_vertical = cfg["CONVEYOR_VERTICAL"]
    save_debug = cfg["SAVE"]
    touch_threshold = cfg["TOUCH_THRESHOLD"]
    required_touches = cfg["REQUIRE_TOUCH"]
    active_tracks = {}
    completed_tracks = set()
    current_video = None
    current_lines = {}
    current_run_dir = ""

    def has_required_touches(track_data: dict, lines: dict) -> bool:
        """Kiểm tra track đã chạm đủ số vạch tham chiếu hiện có chưa."""
        required_count = min(required_touches, len(lines))
        return len(track_data["lines_touched"]) >= required_count

    def flush_track_to_f4(
        track_id: int,
        track_data: dict,
        source_stem: str,
        source_file: str,
        lines: dict,
        run_dir: str,
    ) -> None:
        if lines and not has_required_touches(track_data, lines):
            return
        if track_data["best_frame"] is None:
            return
        masked = get_masked_image(track_data["best_frame"], track_data["best_box_xyxy"], bbox_pad)
        if masked is None:
            return

        debug_images: dict = {}
        if save_debug and track_data["touch_records"]:
            paths = save_f3_debug(
                run_dir=run_dir,
                source_stem=source_stem,
                track_id=track_id,
                touch_records=track_data["touch_records"],
                best_frame_idx=track_data["best_frame_idx"],
                best_area=track_data["best_area"],
                masked_img=masked,
            )
            debug_images.update(paths)

        _put_queue(q_f3_f4, {
            "type": "video",
            "source_file": source_file,
            "source_stem": source_stem,
            "track_id": track_id,
            "frame_idx": track_data["best_frame_idx"],
            "masked_img": masked,
            "orig_img": track_data["best_frame"],
            "lines": lines,
            "debug_images": debug_images,
            "run_dir": run_dir,
        }, stop_event)

    def flush_all_active_tracks() -> None:
        if current_video is None:
            return
        source_stem = current_video.stem
        source_file = current_video.name
        for track_id, track_data in list(active_tracks.items()):
            flush_track_to_f4(
                track_id,
                track_data,
                source_stem,
                source_file,
                current_lines,
                current_run_dir,
            )

    while True:
        item = _get_queue(q_f2_f3, stop_event)

        if item is None:
            if not _stop_requested(stop_event):
                flush_all_active_tracks()
            _put_queue(q_f3_f4, None, stop_event)
            break

        if item["type"] == "image":
            if not _put_queue(q_f3_f4, item, stop_event):
                break
            continue

        video_path = item["path"]
        frame_idx = item["frame_idx"]
        lines = item["lines"]
        detections = item["detections"]

        if video_path != current_video:
            if not chunk_mode:
                flush_all_active_tracks()
                active_tracks.clear()
                completed_tracks.clear()
            current_video = video_path
            current_lines = lines
            current_run_dir = item["run_dir"]

        centers = detections.get_anchors_coordinates(sv.Position.CENTER)

        for box_xyxy, track_id, area, (cx, cy) in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.area,
            centers,
        ):

            track_id = int(track_id)
            area = float(area)
            coord = float(cy) if conveyor_vertical else float(cx)

            if track_id in completed_tracks:
                continue

            if track_id not in active_tracks:
                active_tracks[track_id] = {
                    "lines_touched": set(),
                    "touch_records": [] if save_debug else None,
                    "best_box_xyxy": None,
                    "best_frame": None,
                    "best_frame_idx": None,
                    "best_area": 0.0,
                }

            track_data = active_tracks[track_id]

            for line_id, line_pos in lines.items():
                if line_id in track_data["lines_touched"]:
                    continue
                if not box_touches_line(coord, line_pos, touch_threshold):
                    continue

                track_data["lines_touched"].add(line_id)

                if save_debug and track_data["touch_records"] is not None:
                    track_data["touch_records"].append({
                        "frame": item["frame"],
                        "box_xyxy": box_xyxy.copy(),
                        "area": area,
                        "frame_idx": frame_idx,
                        "line_id": line_id,
                    })

                if area > track_data["best_area"]:
                    track_data["best_box_xyxy"] = box_xyxy.copy()
                    track_data["best_frame"] = item["frame"]
                    track_data["best_frame_idx"] = frame_idx
                    track_data["best_area"] = area

            if has_required_touches(track_data, lines):
                flush_track_to_f4(
                    track_id,
                    track_data,
                    video_path.stem,
                    video_path.name,
                    lines,
                    item["run_dir"],
                )
                completed_tracks.add(track_id)
                del active_tracks[track_id]

    flow_times["F3"] = time.perf_counter() - start_time
    log.info(f"[F3] Hoàn tất  |  {flow_times['F3']:.2f}s")


# F4: Phân đoạn
def flow4_segment(model_seg, q_f3_f4, q_f4_f5, flow_times: dict[str, float], cfg: dict, stop_event) -> None:
    """
    Chạy YOLO segment trên masked_img và tạo crop mask cho F5.

    Hàm chọn detection có bbox lớn nhất, giữ component mask lớn nhất nếu mask
    có nhiều vùng rời rạc, crop mask theo bbox đã pad và cập nhật item bằng
    `crop_mask`, `crop_box`, `cx_label`, `cy_label`. Nếu không tìm thấy mask,
    item bị bỏ qua và lỗi được ghi log.
    """
    log = get_logger()
    start_time = time.perf_counter()
    bbox_pad = cfg["BBOX_PAD"]
    conf_seg = cfg["CONF_SEG"]
    save_debug = cfg["SAVE"]

    while True:
        item = _get_queue(q_f3_f4, stop_event)
        if item is None:
            _put_queue(q_f4_f5, None, stop_event)
            break

        results = model_seg.predict(
            source=item["masked_img"],
            verbose=False,
            conf=conf_seg,
            retina_masks=True,
            device=DEVICE,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0 or detections.mask is None:
            log.warning(f"[F4] {item['source_stem']} ID {item['track_id']}: không tìm thấy mask")
            continue

        xyxy = detections.xyxy
        box_areas = (
            (xyxy[:, 2] - xyxy[:, 0]) *
            (xyxy[:, 3] - xyxy[:, 1])
        )
        best_idx = int(box_areas.argmax())
        best_det = detections[[best_idx]]
        seg_xyxy = best_det.xyxy[0]
        mask_full = best_det.mask[0].astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full)
        if num_labels > 2:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask_full = (labels == largest_label).astype(np.uint8)

        h, w = mask_full.shape
        x1 = max(0, int(seg_xyxy[0]) - bbox_pad)
        y1 = max(0, int(seg_xyxy[1]) - bbox_pad)
        x2 = min(w, int(seg_xyxy[2]) + bbox_pad)
        y2 = min(h, int(seg_xyxy[3]) + bbox_pad)
        crop_mask = mask_full[y1:y2, x1:x2]

        if save_debug:
            paths = save_f4_debug(item, mask_full, seg_xyxy, crop_mask)
            item["debug_images"].update(paths)

        annot_cx, annot_cy = best_det.get_anchors_coordinates(sv.Position.CENTER)[0]

        item.pop("masked_img")
        item.update({
            "crop_mask": crop_mask,
            "crop_box": (x1, y1, x2, y2),
            "cx_label": annot_cx,
            "cy_label": annot_cy,
        })
        if not _put_queue(q_f4_f5, item, stop_event):
            break
        log.info(f"[F4] {item['source_stem']} ID {item['track_id']} phân đoạn xong")

    flow_times["F4"] = time.perf_counter() - start_time
    log.info(f"[F4] Hoàn tất  |  {flow_times['F4']:.2f}s")


# F5: Tìm đường dài nhất trên skeleton
def flow5_longest_path(q_f4_f5, q_f5_f6, flow_times: dict[str, float], cfg: dict, stop_event) -> None:
    """
    Tính skeleton và chiều dài pixel của tôm.

    Hàm dùng medial_axis trên crop_mask, gọi find_longest_path() để lấy
    path_mask và pixel_length, lưu ảnh debug F5 nếu SAVE bật, rồi chuyển item
    đã cập nhật sang F6. Sentinel None được chuyển tiếp sang q_f5_f6.
    """
    log = get_logger()
    start_time = time.perf_counter()
    save_debug = cfg["SAVE"]

    while True:
        item = _get_queue(q_f4_f5, stop_event)
        if item is None:
            _put_queue(q_f5_f6, None, stop_event)
            break

        skeleton = medial_axis(item["crop_mask"], rng=42)
        path_mask, pixel_length = find_longest_path(skeleton)

        if save_debug:
            paths = save_f5_debug(item, skeleton, path_mask)
            item["debug_images"].update(paths)

        item.update({
            "path_mask": path_mask,
            "pixel_length": pixel_length,
        })
        if not _put_queue(q_f5_f6, item, stop_event):
            break

    flow_times["F5"] = time.perf_counter() - start_time
    log.info(f"[F5] Hoàn tất  |  {flow_times['F5']:.2f}s")


# F6: Lưu kết quả
def flow6_save_results(
    q_f5_f6,
    flow_times: dict[str, float],
    cfg: dict,
    size_cfg: dict,
    stop_event,
) -> None:
    """
    Ghi kết quả cuối ra JSON theo từng source.

    Mỗi item từ F5 được quy đổi pixel_length sang real_length_mm bằng cfg,
    phân loại bằng size_cfg và gom vào output của source_stem hiện tại.
    Khi chuyển sang source mới hoặc nhận sentinel None, hàm ghi file
    `<source_stem>_results.json`. Nếu SAVE bật, ảnh F6 được tạo và các đường
    dẫn debug được đưa vào trường `images`. Nếu CLEAR_INPUT bật, input tương
    ứng được xóa sau khi ghi JSON thành công.
    """
    log = get_logger()
    start_time = time.perf_counter()
    clear_input = cfg["CLEAR_INPUT"]
    input_dir = cfg["INPUT_DIR"]
    save_debug = cfg["SAVE"]
    scale = cfg["SCALE"]

    json_data: dict[str, dict] = {}
    stem_to_file: dict[str, str] = {}
    stem_to_run_dir: dict[str, str] = {}
    prev_stem: str | None = None

    def _flush_json(stem: str) -> None:
        """Ghi JSON của một source và xóa input tương ứng nếu CLEAR_INPUT bật."""
        if stem not in json_data:
            return
        out_dir = Path(stem_to_run_dir[stem]) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data[stem], f, ensure_ascii=False, indent=2)
        log.info(f"[F6] JSON -> {json_path}  ({len(json_data[stem]['shrimps'])} tôm)")

        if clear_input and stem in stem_to_file:
            input_file = Path(input_dir) / stem_to_file[stem]
            try:
                if input_file.exists():
                    input_file.unlink()
                    log.info(f"[F6] Đã xóa input: {stem_to_file[stem]}")
            except Exception as e:
                log.warning(f"[F6] Không xóa được input '{stem_to_file[stem]}': {e}")

    while True:
        item = _get_queue(q_f5_f6, stop_event)
        if item is None:
            break

        pixel_length = item["pixel_length"]
        real_length_mm = round(pixel_length * scale, 2)
        size_label = classify_size(real_length_mm, size_cfg)
        stem = item["source_stem"]

        if prev_stem is not None and stem != prev_stem:
            _flush_json(prev_stem)
            json_data.pop(prev_stem, None)
            stem_to_file.pop(prev_stem, None)
            stem_to_run_dir.pop(prev_stem, None)

        prev_stem = stem

        if stem not in stem_to_file:
            stem_to_file[stem] = item["source_file"]
            stem_to_run_dir[stem] = item["run_dir"]

        if stem not in json_data:
            json_data[stem] = {
                "source_file": item["source_file"],
                "source_stem": stem,
                "processed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "scale_mm_per_px": scale,
                "shrimps": [],
            }

        shrimp_entry: dict = {
            "track_id": item["track_id"],
            "frame_idx": item["frame_idx"],
            "pixel_length": round(pixel_length, 1),
            "real_length_mm": real_length_mm,
            "size": size_label,
        }

        if save_debug:
            f6_path = draw_f6_result(item, cfg)
            item["debug_images"]["f6_result"] = f6_path
            shrimp_entry["images"] = item["debug_images"]

        json_data[stem]["shrimps"].append(shrimp_entry)

        log.info(
            f"[F6] {stem} ID {item['track_id']:>3} | "
            f"frame {item['frame_idx']:>4} | "
            f"{round(pixel_length, 1):>7.1f} px | "
            f"{real_length_mm:>7.2f} mm | "
            f"size={size_label}"
        )

    if prev_stem is not None:
        _flush_json(prev_stem)

    flow_times["F6"] = time.perf_counter() - start_time
    log.info(f"[F6] Hoàn tất  |  {flow_times['F6']:.2f}s")
