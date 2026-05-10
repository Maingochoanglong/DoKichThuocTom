"""
draw_utils.py
Các hàm vẽ & xuất ảnh debug, theo dõi quá trình đo đạc tôm trên băng chuyền.

Thay đổi so với phiên bản cũ:
  - Tất cả hàm save* và draw_f6_result đều trả về dict chứa đường dẫn
    tới các file ảnh đã lưu, để F6 đưa vào JSON output.
  - Đơn vị đo cố định là mm.
  - _id_dir nhận run_dir (thư mục output theo timestamp) thay vì dùng OUTPUT_DIR.
"""


from pathlib import Path

import cv2
import numpy as np

from config import COLOR, CONVEYOR_VERTICAL, SCALE
from flow_utils import ensure_dir


def get_track_color(track_id: int) -> tuple[int, int, int]:
    return COLOR[(track_id - 1) % len(COLOR)]


def _id_dir(run_dir: str, source_stem: str, track_id: int) -> Path:
    p = Path(run_dir) / source_stem / f"ID{track_id}"
    ensure_dir(p)
    return p


# F3 
def save_f3_debug(
    run_dir: str,
    source_stem: str, track_id: int,
    touch_records: list[dict],
    best_frame_idx: int, best_area: float,
    masked_img: np.ndarray,
) -> dict:
    """
    Lưu ảnh khung chạm vạch và ảnh masked tốt nhất.

    Returns:
        {
            "f3_best"   : str,        # đường dẫn ảnh F3_Best
            "f3_touches": [str, ...]  # danh sách đường dẫn ảnh F3_Touch
        }
    """
    out_dir     = _id_dir(run_dir, source_stem, track_id)
    track_color = get_track_color(track_id)
    touch_paths: list[str] = []

    for rec in touch_records:
        canvas = rec["frame"].copy()
        bx1, by1, bx2, by2 = map(int, rec["box_xyxy"])
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), track_color, 2)
        cv2.putText(
            canvas, f"ID:{track_id}  {rec['area']:.0f}px",
            (bx1, max(by1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2, cv2.LINE_AA,
        )
        p = str(out_dir / f"F3_Touch_F{rec['frame_idx']}_{rec['area']:.0f}px.jpg")
        cv2.imwrite(p, canvas)
        touch_paths.append(p)

    best_path = str(out_dir / f"F3_Best_F{best_frame_idx}_{best_area:.0f}px.jpg")
    cv2.imwrite(best_path, masked_img)

    print(f"[F3-debug] ID {track_id}: đã lưu {len(touch_records)} khung chạm vạch")
    return {"f3_best": best_path, "f3_touches": touch_paths}


# F4 
def save_f4_debug(
    item: dict,
    mask_full: np.ndarray,
    seg_xyxy: np.ndarray,
    crop_mask: np.ndarray,
) -> dict:
    """
    Lưu ảnh overlay phân đoạn và crop-mask nhị phân.

    Returns:
        {"f4_seg": str, "f4_mask": str}
    """
    out_dir     = _id_dir(item["run_dir"], item["source_stem"], item["track_id"])
    tid, fidx   = item["track_id"], item["frame_idx"]
    track_color = get_track_color(tid)

    seg_vis = item["masked_img"].copy()
    overlay = seg_vis.copy()
    overlay[mask_full > 0] = track_color
    cv2.addWeighted(overlay, 0.4, seg_vis, 0.6, 0, seg_vis)
    sx1, sy1, sx2, sy2 = map(int, seg_xyxy)
    cv2.rectangle(seg_vis, (sx1, sy1), (sx2, sy2), track_color, 2)
    cv2.putText(seg_vis, f"ID:{tid}",
                (sx1, max(sy1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2, cv2.LINE_AA)

    ch, cw   = crop_mask.shape
    crop_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    crop_vis[crop_mask > 0] = (255, 255, 255)

    seg_path  = str(out_dir / f"F4_Seg_F{fidx}.jpg")
    mask_path = str(out_dir / f"F4_Mask_F{fidx}.jpg")
    cv2.imwrite(seg_path,  seg_vis)
    cv2.imwrite(mask_path, crop_vis)

    print(f"[F4-debug] ID {tid}: đã lưu ảnh phân đoạn + mask vùng cắt")
    return {"f4_seg": seg_path, "f4_mask": mask_path}


# F5 
def save_f5_debug(
    item: dict,
    skeleton: np.ndarray,
    path_mask: np.ndarray,
) -> dict:
    """
    Lưu ảnh trục xương Medial Axis và đường BFS dài nhất.

    Returns:
        {"f5_skel": str, "f5_bfs": str}
    """
    out_dir        = _id_dir(item["run_dir"], item["source_stem"], item["track_id"])
    tid, fidx      = item["track_id"], item["frame_idx"]
    skel_h, skel_w = skeleton.shape
    track_color    = get_track_color(tid)

    ma_vis = np.zeros((skel_h, skel_w, 3), dtype=np.uint8)
    ma_vis[skeleton > 0] = (255, 255, 255)

    bfs_vis = np.zeros((skel_h, skel_w, 3), dtype=np.uint8)
    bfs_vis[path_mask > 0] = track_color

    skel_path = str(out_dir / f"F5_Skel_F{fidx}.jpg")
    bfs_path  = str(out_dir / f"F5_BFS_F{fidx}.jpg")
    cv2.imwrite(skel_path, ma_vis)
    cv2.imwrite(bfs_path,  bfs_vis)

    print(f"[F5-debug] ID {tid}: đã lưu trục giữa (Skel) + đường BFS dài nhất")
    return {"f5_skel": skel_path, "f5_bfs": bfs_path}


# F6 
def draw_f6_result(item: dict) -> str:
    """
    Vẽ ảnh kết quả cuối: đường tham chiếu, overlay tôm, skeleton, nhãn đo.
    Nhãn chiều dài đơn vị mm (cố định).

    Returns:
        str  — đường dẫn file ảnh F6_Result đã lưu.
    """
    pixel_length = item["pixel_length"]
    real_length  = pixel_length * SCALE
    tid, fidx    = item["track_id"], item["frame_idx"]
    out_dir      = _id_dir(item["run_dir"], item["source_stem"], tid)

    canvas           = item["orig_img"].copy()
    h_frame, w_frame = canvas.shape[:2]
    clr              = get_track_color(tid)

    if item.get("lines"):
        for lid, lv in item["lines"].items():
            if CONVEYOR_VERTICAL:
                cv2.line(canvas, (0, lv), (w_frame, lv), COLOR[5], 1)
                cv2.putText(canvas, str(lid), (4, lv - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[5], 1)
            else:
                cv2.line(canvas, (lv, 0), (lv, h_frame), COLOR[5], 1)
                cv2.putText(canvas, str(lid), (lv + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[5], 1)

    x1, y1, x2, y2 = item["crop_box"]
    roi       = canvas[y1:y2, x1:x2]
    crop_mask = item["crop_mask"]
    path_mask = item["path_mask"]

    if crop_mask.shape[:2] != roi.shape[:2]:
        crop_mask = cv2.resize(crop_mask,
                               (roi.shape[1], roi.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        path_mask = cv2.resize(path_mask.astype(np.uint8),
                               (roi.shape[1], roi.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = roi.copy()
    overlay[crop_mask > 0] = clr
    cv2.addWeighted(overlay, 0.4, roi, 0.6, 0, roi)
    canvas[y1:y2, x1:x2]                = roi
    canvas[y1:y2, x1:x2][path_mask > 0] = (255, 255, 255)

    # Nhãn: "672.4px  235.3mm  [ID:1]"
    txt    = f"{pixel_length:.1f}px  {real_length:.1f}mm  [ID:{tid}]"
    tx, ty = int(item["cx_label"]), max(int(item["cy_label"]) - 10, 20)
    cv2.putText(canvas, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2, cv2.LINE_AA)

    out_path = str(out_dir / f"F6_Result_F{fidx}_{pixel_length:.0f}px.jpg")
    cv2.imwrite(out_path, canvas)
    print(f"[F6] ID {tid}: kết quả cuối đã lưu -> {Path(out_path).name}")
    return out_path