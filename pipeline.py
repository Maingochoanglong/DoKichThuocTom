"""
pipeline.py
Logic cốt lõi của 6 luồng (F1 - F6).

  F1 - Đọc ảnh/video từ INPUT_DIR, đẩy từng frame vào queue.
  F2 - Phát hiện tôm (YOLO detect) + ByteTrack tracking.
  F3 - Xác nhận tôm chạm đủ vạch tham chiếu, chọn frame tốt nhất.
  F4 - Phân đoạn thân tôm (YOLO segment), lấy mask chính xác.
  F5 - Tính skeleton (medial axis) + BFS tìm đường dài nhất, trả về pixel_length.
  F6 - Quy đổi sang mm, phân loại kích cỡ, xuất JSON kết quả.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from skimage.morphology import medial_axis

from config import (
    BBOX_PAD, CHUNK_MODE, CLEAR_INPUT, CONF_DET, CONF_SEG, CONVEYOR_VERTICAL,
    IMG_EXTS, INPUT_DIR,
    REQUIRED_TOUCHES, SAVE, SCALE, TARGET_FPS, TOUCH_THRESHOLD, VID_EXTS,
)
from draw_utils import draw_f6_result, save_f3_debug, save_f4_debug, save_f5_debug
from flow_utils import box_touches_line, get_lines, get_masked_image
from logger_setup import get_logger
from size import classify_size
from skeleton_utils import find_longest_path


# F1: Đọc input
def flow1_read_input(
    q_f1_f2,
    flow_times: dict,
    run_dir: str,
) -> None:
    """
    Đọc ảnh/video từ INPUT_DIR và đẩy từng frame vào q_f1_f2.
    Quét một lần, xử lý hết rồi gửi sentinel None.
    Mỗi item dict đều chứa 'run_dir' để các flow sau ghi kết quả đúng thư mục.
    """
    log        = get_logger()
    start_time = time.perf_counter()
    input_path = Path(INPUT_DIR)

    if not input_path.exists():
        log.warning(f"[F1] Thư mục '{INPUT_DIR}' không tồn tại!")
        q_f1_f2.put(None)
        flow_times["F1"] = time.perf_counter() - start_time
        return

    def _push_image(fpath: Path) -> None:
        image = cv2.imread(str(fpath))
        if image is None:
            log.warning(f"[F1] Bỏ qua '{fpath.name}' - lỗi đọc ảnh")
            return
        q_f1_f2.put({
            "type"        : "image",
            "path"        : fpath,
            "source_file" : fpath.name,
            "source_stem" : fpath.stem,
            "frame"       : image,
            "frame_idx"   : 0,
            "lines"       : {},
            "run_dir"     : run_dir,
        })
        log.info(f"[F1] Đã đọc ảnh: {fpath.name}")

    def _push_video(fpath: Path) -> None:
        try:
            info = sv.VideoInfo.from_video_path(str(fpath))
        except Exception:
            log.warning(f"[F1] Bỏ qua '{fpath.name}' - lỗi đọc video")
            return

        fps       = info.fps or 30.0
        step      = max(1, round(fps / TARGET_FPS)) if TARGET_FPS > 0 else 1
        frame_dim = info.height if CONVEYOR_VERTICAL else info.width
        lines     = get_lines(frame_dim)

        log.info(f"[F1] Đọc video: {fpath.name} | {info.width}x{info.height} | lines={dict(lines)}")
        for i, frame in enumerate(sv.get_video_frames_generator(str(fpath), stride=step)):
            q_f1_f2.put({
                "type"        : "video",
                "path"        : fpath,
                "source_file" : fpath.name,
                "source_stem" : fpath.stem,
                "frame"       : frame,
                "frame_idx"   : i * step + 1,
                "lines"       : lines,
                "run_dir"     : run_dir,
            })
        log.info(f"[F1] Đọc xong: {fpath.name}")

    def _process_file(fpath: Path) -> None:
        suffix = fpath.suffix.lower()
        if suffix in IMG_EXTS:
            _push_image(fpath)
        elif suffix in VID_EXTS:
            _push_video(fpath)

    log.info(f"[F1] Quét thư mục '{INPUT_DIR}'")
    for fpath in sorted(input_path.iterdir()):
        _process_file(fpath)

    q_f1_f2.put(None)
    flow_times["F1"] = time.perf_counter() - start_time
    log.info(f"[F1] Hoàn tất  |  {flow_times['F1']:.2f}s")


# F2: Phát hiện & bám vết
def flow2_detect_track(model_det, q_f1_f2, q_f2_f3, flow_times: dict) -> None:
    """
    Chạy YOLO detect trên từng frame.
    - Ảnh tĩnh: gán track_id tăng dần theo thứ tự detection, gửi thẳng sang F3.
    - Video: cập nhật ByteTrack, chỉ gửi frame có ít nhất một detection.
    """
    log           = get_logger()
    start_time    = time.perf_counter()
    tracker       = sv.ByteTrack() if CHUNK_MODE else None
    current_video = None

    while True:
        item = q_f1_f2.get()
        if item is None:
            q_f2_f3.put(None)
            break

        results = model_det.predict(
            source=item["frame"],
            verbose=False, conf=CONF_DET, device="intel:gpu",
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if item["type"] == "image":
            if len(detections) > 0:
                log.info(f"[F2] {item['source_file']}: phát hiện {len(detections)} con tôm")
                for i, box_xyxy in enumerate(detections.xyxy):
                    masked_img = get_masked_image(item["frame"], box_xyxy, pad=BBOX_PAD)
                    if masked_img is not None:
                        q_f2_f3.put({
                            "type"        : "image",
                            "source_file" : item["source_file"],
                            "source_stem" : item["source_stem"],
                            "track_id"    : i + 1,
                            "frame_idx"   : 0,
                            "masked_img"  : masked_img,
                            "orig_img"    : item["frame"],
                            "lines"       : {},
                            "debug_images": {},
                            "run_dir"     : item["run_dir"],
                        })

        elif item["type"] == "video":
            if item["path"] != current_video:
                if not CHUNK_MODE:
                    tracker = sv.ByteTrack()
                current_video = item["path"]
                log.info(f"[F2] Tracker mới cho: {current_video.name}")

            detections = tracker.update_with_detections(detections)
            if len(detections) > 0:
                item["detections"] = detections
                q_f2_f3.put(item)

    flow_times["F2"] = time.perf_counter() - start_time
    log.info(f"[F2] Hoàn tất  |  {flow_times['F2']:.2f}s")


# F3: Kiểm tra chạm vạch
def flow3_touch_logic(q_f2_f3, q_f3_f4, flow_times: dict) -> None:
    """
    - Ảnh tĩnh: chuyển thẳng sang F4 không qua kiểm tra vạch.
    - Video: xác nhận tôm chạm đủ REQUIRED_TOUCHES vạch rồi mới gửi F4.
    """
    log              = get_logger()
    start_time       = time.perf_counter()
    active_tracks    = {}
    completed_tracks = set()
    current_video    = None
    current_lines    = {}
    current_run_dir  = ""

    def flush_track_to_f4(
        track_id: int, track_data: dict,
        source_stem: str, source_file: str, lines: dict,
        run_dir: str,
    ):
        if track_data["best_frame"] is None:
            return
        masked = get_masked_image(track_data["best_frame"], track_data["best_box_xyxy"], BBOX_PAD)
        if masked is None:
            return

        debug_images: dict = {}
        if SAVE and track_data["touch_records"]:
            paths = save_f3_debug(
                run_dir=run_dir,
                source_stem=source_stem, track_id=track_id,
                touch_records=track_data["touch_records"],
                best_frame_idx=track_data["best_frame_idx"],
                best_area=track_data["best_area"],
                masked_img=masked,
            )
            debug_images.update(paths)

        q_f3_f4.put({
            "type"        : "video",
            "source_file" : source_file,
            "source_stem" : source_stem,
            "track_id"    : track_id,
            "frame_idx"   : track_data["best_frame_idx"],
            "masked_img"  : masked,
            "orig_img"    : track_data["best_frame"],
            "lines"       : lines,
            "debug_images": debug_images,
            "run_dir"     : run_dir,
        })

    def flush_all_active_tracks():
        if current_video is None:
            return
        for track_id, track_data in active_tracks.items():
            flush_track_to_f4(
                track_id, track_data,
                current_video.stem, current_video.name, current_lines,
                current_run_dir,
            )

    while True:
        item = q_f2_f3.get()

        if item is None:
            flush_all_active_tracks()
            q_f3_f4.put(None)
            break

        if item["type"] == "image":
            q_f3_f4.put(item)
            continue

        video_path = item["path"]
        frame_idx  = item["frame_idx"]
        lines      = item["lines"]
        detections = item["detections"]

        if video_path != current_video:
            if not CHUNK_MODE:
                flush_all_active_tracks()
                active_tracks.clear()
                completed_tracks.clear()
            current_video   = video_path
            current_lines   = lines
            current_run_dir = item["run_dir"]

        centers = detections.get_anchors_coordinates(sv.Position.CENTER)

        for box_xyxy, track_id, area, (cx, cy) in zip(
                detections.xyxy, detections.tracker_id, detections.area, centers):

            track_id = int(track_id)
            area     = float(area)
            coord    = float(cy) if CONVEYOR_VERTICAL else float(cx)

            if track_id in completed_tracks:
                continue

            if track_id not in active_tracks:
                active_tracks[track_id] = {
                    "lines_touched" : set(),
                    "touch_records" : [] if SAVE else None,
                    "best_box_xyxy" : None,
                    "best_frame"    : None,
                    "best_frame_idx": None,
                    "best_area"     : 0.0,
                }

            track_data = active_tracks[track_id]

            for line_id, line_pos in lines.items():
                if line_id in track_data["lines_touched"]:
                    continue
                if not box_touches_line(coord, line_pos, TOUCH_THRESHOLD):
                    continue

                track_data["lines_touched"].add(line_id)

                if SAVE and track_data["touch_records"] is not None:
                    track_data["touch_records"].append({
                        "frame"    : item["frame"],
                        "box_xyxy" : box_xyxy.copy(),
                        "area"     : area,
                        "frame_idx": frame_idx,
                        "line_id"  : line_id,
                    })

                if area > track_data["best_area"]:
                    track_data["best_box_xyxy"]  = box_xyxy.copy()
                    track_data["best_frame"]      = item["frame"]
                    track_data["best_frame_idx"]  = frame_idx
                    track_data["best_area"]       = area

            if len(track_data["lines_touched"]) == REQUIRED_TOUCHES:
                flush_track_to_f4(track_id, track_data, video_path.stem, video_path.name, lines, item["run_dir"])
                completed_tracks.add(track_id)
                del active_tracks[track_id]

    flow_times["F3"] = time.perf_counter() - start_time
    log.info(f"[F3] Hoàn tất  |  {flow_times['F3']:.2f}s")


# F4: Phân đoạn
def flow4_segment(model_seg, q_f3_f4, q_f4_f5, flow_times: dict) -> None:
    """
    Chạy YOLO segment trên vùng ảnh đã che (masked_img).
    Giữ lại mask lớn nhất, loại bỏ các vùng rời rạc nhỏ nếu có,
    rồi crop mask theo bounding box để giảm kích thước truyền sang F5.
    """
    log        = get_logger()
    start_time = time.perf_counter()

    while True:
        item = q_f3_f4.get()
        if item is None:
            q_f4_f5.put(None)
            break

        results = model_seg.predict(
            source=item["masked_img"],
            verbose=False, conf=CONF_SEG,
            retina_masks=True, device="intel:gpu",
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0 or detections.mask is None:
            log.warning(f"[F4] {item['source_stem']} ID {item['track_id']}: không tìm thấy mask")
            continue

        best_det  = detections[[detections.area.argmax()]]
        seg_xyxy  = best_det.xyxy[0]
        mask_full = best_det.mask[0].astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full)
        if num_labels > 2:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask_full = (labels == largest_label).astype(np.uint8)

        h, w = mask_full.shape
        x1 = max(0, int(seg_xyxy[0]) - BBOX_PAD)
        y1 = max(0, int(seg_xyxy[1]) - BBOX_PAD)
        x2 = min(w, int(seg_xyxy[2]) + BBOX_PAD)
        y2 = min(h, int(seg_xyxy[3]) + BBOX_PAD)
        crop_mask = mask_full[y1:y2, x1:x2]

        if SAVE:
            paths = save_f4_debug(item, mask_full.copy(), seg_xyxy.copy(), crop_mask.copy())
            item["debug_images"].update(paths)

        annot_cx, annot_cy = best_det.get_anchors_coordinates(sv.Position.CENTER)[0]

        item.pop("masked_img")
        item.update({
            "crop_mask": crop_mask,
            "crop_box" : (x1, y1, x2, y2),
            "cx_label" : annot_cx,
            "cy_label" : annot_cy,
        })
        q_f4_f5.put(item)
        log.info(f"[F4] {item['source_stem']} ID {item['track_id']} phân đoạn xong")

    flow_times["F4"] = time.perf_counter() - start_time
    log.info(f"[F4] Hoàn tất  |  {flow_times['F4']:.2f}s")


# F5: Tìm đường dài nhất trên skeleton
def flow5_longest_path(q_f4_f5, q_f5_f6, flow_times: dict) -> None:
    """
    Tính medial axis (skeleton) của crop_mask, sau đó dùng BFS 2 lần
    để tìm đường dài nhất trên skeleton, kết quả là pixel_length.
    """
    log        = get_logger()
    start_time = time.perf_counter()

    while True:
        item = q_f4_f5.get()
        if item is None:
            q_f5_f6.put(None)
            break

        skeleton = medial_axis(item["crop_mask"], rng=42)
        path_mask, pixel_length = find_longest_path(skeleton)

        if SAVE:
            paths = save_f5_debug(item, skeleton, path_mask)
            item["debug_images"].update(paths)

        item.update({
            "path_mask"   : path_mask,
            "pixel_length": pixel_length,
        })
        q_f5_f6.put(item)

    flow_times["F5"] = time.perf_counter() - start_time
    log.info(f"[F5] Hoàn tất  |  {flow_times['F5']:.2f}s")


# F6: Lưu kết quả
def flow6_save_results(q_f5_f6, flow_times: dict) -> None:
    """
    Xuất kết quả cuối:
      • JSON : output/<source_stem>/<source_stem>_results.json (luôn ghi)

    JSON được ghi khi chuyển sang source mới hoặc pipeline kết thúc.
    Sau khi ghi JSON thành công, xóa file input tương ứng.

    Cấu trúc JSON:
    {
      "source_file"      : "video.mp4",
      "source_stem"      : "video",
      "processed_at"     : "2026-05-01T10:30:00",
      "scale_mm_per_px"  : 1.0,
      "shrimps": [
        {
          "track_id"       : 1,
          "frame_idx"      : 45,
          "pixel_length"   : 234.5,
          "real_length_mm" : 234.5,
          "size"           : "L",
          "images": {                          // chỉ có khi SAVE=True
            "f3_best"   : "output/.../F3_Best_...",
            "f3_touches": ["output/.../F3_Touch_..."],
            "f4_seg"    : "output/.../F4_Seg_...",
            "f4_mask"   : "output/.../F4_Mask_...",
            "f5_skel"   : "output/.../F5_Skel_...",
            "f5_bfs"    : "output/.../F5_BFS_...",
            "f6_result" : "output/.../F6_Result_..."
          }
        },
        ...
      ]
    }
    """
    log        = get_logger()
    start_time = time.perf_counter()

    json_data: dict[str, dict] = {}
    stem_to_file: dict[str, str] = {}
    current_run_dir: str | None = None
    prev_stem: str | None = None

    def _flush_json(stem: str) -> None:
        if stem not in json_data:
            return
        out_dir = Path(current_run_dir) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data[stem], f, ensure_ascii=False, indent=2)
        log.info(f"[F6] JSON -> {json_path}  ({len(json_data[stem]['shrimps'])} tôm)")

        if CLEAR_INPUT and stem in stem_to_file:
            input_file = Path(INPUT_DIR) / stem_to_file[stem]
            try:
                if input_file.exists():
                    input_file.unlink()
                    log.info(f"[F6] Đã xóa input: {stem_to_file[stem]}")
            except Exception as e:
                log.warning(f"[F6] Không xóa được input '{stem_to_file[stem]}': {e}")

    while True:
        item = q_f5_f6.get()
        if item is None:
            break

        pixel_length   = item["pixel_length"]
        real_length_mm = round(pixel_length * SCALE, 2)
        size_label     = classify_size(real_length_mm)
        stem           = item["source_stem"]

        if prev_stem is not None and stem != prev_stem:
            _flush_json(prev_stem)
            json_data.pop(prev_stem, None)
            stem_to_file.pop(prev_stem, None)

        prev_stem = stem

        if stem not in stem_to_file:
            stem_to_file[stem] = item["source_file"]
            current_run_dir = item["run_dir"]

        if stem not in json_data:
            json_data[stem] = {
                "source_file"    : item["source_file"],
                "source_stem"    : stem,
                "processed_at"   : datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "scale_mm_per_px": SCALE,
                "shrimps"        : [],
            }

        shrimp_entry: dict = {
            "track_id"      : item["track_id"],
            "frame_idx"     : item["frame_idx"],
            "pixel_length"  : round(pixel_length, 1),
            "real_length_mm": real_length_mm,
            "size"          : size_label,
        }

        if SAVE:
            f6_path = draw_f6_result(item)
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