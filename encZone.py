import argparse
from collections import defaultdict, deque
import pandas as pd
import logging
import math



from tqdm import tqdm
import time as tm

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv
from copy import deepcopy
import random
from datetime import datetime
from pathlib import Path
from boxmot import StrongSort

import os, json, yaml
from dotenv import load_dotenv

# right after the other imports:
load_dotenv(dotenv_path=".env.bns")

VIDEO_PATH      = os.getenv("VIDEO_PATH")
OUTPUT_PATH     = os.getenv("OUTPUT_PATH")
ZONE_CHECK_PNG     = os.getenv("ZONE_CHECK_PNG")
MODEL_WEIGHTS   = os.getenv("MODEL_WEIGHTS")
ENC_ZONE_CONFIG = os.getenv("ENC_ZONE_CONFIG")
DEVICE          = os.getenv("DEVICE", "cpu")



...
# map tracker_id → BGR colour tuple
future_colors: dict[int, tuple[int,int,int]] = {}

# Show live preview while processing?
DISPLAY = False     # set True if you ever want to see the window again

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)   # ERROR or CRITICAL

SOURCE = np.array([[1281,971], [2309,971], [6090,2160], [-2243,2160]])

#Storing trajectory coordinates
csv_rows = []
# ─── NEW: store TTC events ───
ttc_rows = []                         # rows for TTC-event CSV
id_to_class = {}                      # tracker_id → latest class name

TARGET_WIDTH = 50
TARGET_HEIGHT = 130

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)
def load_zones(path: str) -> tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        data = yaml.safe_load(f) if ext in {".yml", ".yaml"} else json.load(f)

    left  = np.asarray(data["left_zone"],  np.int32)
    right = np.asarray(data["right_zone"], np.int32)
    return left, right
def blend_zone(frame: np.ndarray,
               mask:  np.ndarray,
               colour: tuple[int, int, int],
               alpha:  float = 0.35) -> None:
    """
    In-place α-blend of `colour` wherever mask == 255.
    `colour` is BGR, 0‒255. Alpha 0‒1.
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = colour
    frame[:] = np.where(mask[..., None] == 255,
                        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0),
                        frame)
# ---------- FILTER RIDERS (person riding bicycle / motorcycle) ----------
def _iou_xyxy(box_a, box_b) -> float:
    """
    Fast scalar IoU between two [x1,y1,x2,y2] boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def filter_rider_persons(det: sv.Detections,
                         iou_thr: float = 0.50) -> sv.Detections:
    """
    Discard *person* boxes that almost certainly belong to a
    motorcycle / bicycle rider.

    • IoU > iou_thr   OR   person-centre inside vehicle box
    • Person appears vertically **above** vehicle centre
    """
    if len(det) == 0:
        return det                       # nothing to do

    cls   = det.class_id
    boxes = det.xyxy

    person_idx   = np.where(cls == 0)[0]             # 0 = person (COCO)
    vehicle_idx  = np.where(np.isin(cls, [1, 3]))[0] # 1=bicycle, 3=motorcycle

    if person_idx.size == 0 or vehicle_idx.size == 0:
        return det                                   # no pairing possible

    keep = np.ones(len(det), dtype=bool)

    for p in person_idx:
        px1, py1, px2, py2 = boxes[p]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2  # person centre

        for v in vehicle_idx:
            vx1, vy1, vx2, vy2 = boxes[v]
            v_cy = (vy1 + vy2) / 2                    # vehicle centre-Y

            iou_ok      = _iou_xyxy(boxes[p], boxes[v]) > iou_thr
            centre_inside = (vx1 <= pcx <= vx2) and (vy1 <= pcy <= vy2)
            above_ok    = pcy < v_cy                  # rider sits above centre

            if (iou_ok or centre_inside) and above_ok:
                keep[p] = False                       # drop this person
                break

    return det[keep]

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        self.m_inv = cv2.getPerspectiveTransform(target, source)  # Add this for inverse

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m_inv)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--no_blend_zones",
        action="store_true",
        help="Disable the translucent curb-lane overlays drawn by blend_zone()"
    )
    parser.add_argument(
        "--source_video_path",
        required=False,
        help="Path to the source video file",
        type=str,
        default=VIDEO_PATH  # Hardcoded path
    )
    parser.add_argument(
        "--target_video_path",
        required=False,
        help="Path to the target video file (output)",
        type=str,
        default=OUTPUT_PATH  # Hardcoded path
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    # 🚀 NEW ARGUMENTS
    parser.add_argument(
        "--num_future_predictions",
        default=10,
        help="Number of future points to predict per vehicle",
        type=int
    )
    parser.add_argument(
        "--future_prediction_interval",
        default=0.1,
        help="Time interval (seconds) between future predictions",
        type=float
    )
    parser.add_argument(
        "--ttc_threshold",
        default=1.0,
        help="Only show TTC if it's ≤ this value (in seconds)",
        type=float
    )
    parser.add_argument("--zones_file", default=ENC_ZONE_CONFIG,
                        help="YAML/JSON file with curb-lane polygons")
    parser.add_argument("--dump_zones_png",
                        help="Write a PNG showing lane polygons over the first video frame, then exit")
    # Tracker backend ----------------------------------------------------
    parser.add_argument(
        "--tracker",
        choices=["strongsort", "bytetrack"],
        default="strongsort",
        help="Tracking backend to use (default: strongsort)",
    )
    parser.add_argument(
        "--segment_speed",
        action="store_true",
        help="Enable segment-based speed calibration (default: off)"
    )

    return parser.parse_args()
# Kalman Filter related
kf_states = dict()

def sv_to_boxmot(det: sv.Detections) -> np.ndarray:
    """Return Nx6 array  [x1 y1 x2 y2 conf cls] for StrongSORT."""
    if not len(det):
        return np.empty((0, 6), dtype=np.float32)
    return np.hstack((det.xyxy,
                      det.confidence[:, None],
                      det.class_id[:, None])).astype(np.float32)
def create_kalman_filter(dt: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    # state = [x, y, vx, vy], measurement = [x, y]
    kf.transitionMatrix = np.array([
        [1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], np.float32)
    kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    return kf


future_coordinates = defaultdict(list)  # Store future (x, y) points

# Predict future positions (constant velocity assumption)
def predict_future_positions_kf(kf, delta_t=0.5, num_predictions=4):
    future_positions = []
    current_state = deepcopy(kf.statePost.flatten())  # [x, y, vx, vy]
    x, y, vx, vy = current_state
    # print(f"Tracker #{tracker_id} → vx={vx:.2f}, vy={vy:.2f}")

    for step in range(1, num_predictions + 1):
        t = step * delta_t
        x_future = x + vx * t
        y_future = y + vy * t
        future_positions.append((x_future, y_future))
    return future_positions

def line_side(pt, a, b):
    # +1 if pt is on left of AB, −1 on right, 0 on the line
    return np.sign((b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0]))

def load_segments(path:str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    entry = np.asarray(data["segment_entry"], np.int32)
    exit_  = np.asarray(data["segment_exit"],  np.int32)
    return entry, exit_



if __name__ == "__main__":
    args = parse_arguments()
    # ── load zone polygons (always) ─────────────────────────────
    LEFT_CNT, RIGHT_CNT = load_zones(args.zones_file)

    # ── load segment lines only if needed ──────────────────────
    if args.segment_speed:
        ENTRY_LINE, EXIT_LINE = load_segments(args.zones_file)
        segment_state = {}
        segment_results = []
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO(MODEL_WEIGHTS or "yolov8l.pt")
    model.to(DEVICE)

    CLIP_SECONDS = 20  # process only the first 20 s
    clip_frames = min(int(video_info.fps * CLIP_SECONDS),
                      video_info.total_frames)  # safety if the video is shorter

    total_frames = video_info.total_frames  # already provided by sv.get_video_info(...)
    bar = tqdm(total=clip_frames, desc="Processing video", unit="frame")
    t0 = tm.time()

    ttc_event_count = 0  # live counter

    last_seen_frame = {}  # tracker_id → last frame index where it was detected
    MAX_AGE_SECONDS = 1  # how long to keep a “dead” track around
    MAX_AGE_FRAMES = int(video_info.fps * MAX_AGE_SECONDS)

    # ─── ENCROACHMENT ZONES (pixel coordinates, clockwise) ───
    LEFT_ZONE_POLY, RIGHT_ZONE_POLY = load_zones(args.zones_file)

    # ─── 2  PolygonZone uses the plain (N,2) shape ───
    left_zone = sv.PolygonZone(polygon=LEFT_ZONE_POLY)
    right_zone = sv.PolygonZone(polygon=RIGHT_ZONE_POLY)

    # ─── 3  Masks need the OpenCV contour shape (N,1,2) ───
    LEFT_CNT = LEFT_ZONE_POLY.reshape((-1, 1, 2))
    RIGHT_CNT = RIGHT_ZONE_POLY.reshape((-1, 1, 2))

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Mask creation (after LEFT_CNT / RIGHT_CNT are defined)
    # ─────────────────────────────────────────────────────────────────────────────
    need_masks = (not args.no_blend_zones) or args.dump_zones_png

    if need_masks:
        H, W = video_info.resolution_wh[::-1]
        mask_left = np.zeros((H, W), np.uint8)
        mask_right = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask_left, [LEFT_CNT], 255)
        cv2.fillPoly(mask_right, [RIGHT_CNT], 255)
    else:
        mask_left = mask_right = None  # names stay defined

    if args.dump_zones_png:
        cap = cv2.VideoCapture(args.source_video_path)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame for zone preview")

        # make translucent overlays
        preview = frame.copy()
        cv2.fillPoly(preview, [LEFT_CNT], (0, 255, 0))
        cv2.fillPoly(preview, [RIGHT_CNT], (0, 0, 255))
        frame = cv2.addWeighted(preview, 0.35, frame, 0.65, 0, frame)

        # draw crisp outlines
        cv2.polylines(frame, [LEFT_CNT], True, (0, 255, 0), 2)
        cv2.polylines(frame, [RIGHT_CNT], True, (0, 0, 255), 2)

        cv2.imwrite(ZONE_CHECK_PNG, frame)
        print(f"[zone-preview] saved → {args.dump_zones_png}")
        exit(0)

    # ─── PARAMETERS ───
    ENCROACH_SECS = 1  # seconds inside zone
    MOVE_THRESH_METRES = 1.0  # “barely moved” in world metres

    if args.tracker == "strongsort":
        strong_sort = StrongSort(
            reid_weights=Path("mobilenetv2_x1_4_dukemtmcreid.pt"),
            device=0 if DEVICE != "cpu" else DEVICE,
            half=False,
            max_age=300,
        )
    else:  # ByteTrack
        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps,
            track_activation_threshold=args.confidence_threshold,
        )

    thickness = 2
    text_scale = 1
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_LEFT,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 3,
        position=sv.Position.CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    ttc_labels = defaultdict(list)
    frame_idx = 0
    enc_state = {}  # tracker_id → {t0, X0, Y0}
    enc_events = []  # list of flagged encroachments for CSV
    # ─── state holders (place once, before the loop) ─────────────────────────
    enc_active_ids = set()  # IDs still in-zone and already flagged
    enc_id_to_zone_side = {}  # tracker_id → "left" | "right"

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            bar.update(1)  # advance by one frame
            # say you have a frame counter:
            frame_idx += 1
            if frame_idx >= video_info.fps * CLIP_SECONDS:
                break
            # ─── CLEAR OLD TTC LABELS ───
            ttc_labels.clear()
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            # ✂ Ignore riders so that only true pedestrians remain
            #detections = filter_rider_persons(detections, iou_thr=0.50)
            if args.tracker == "strongsort":
                tracks = strong_sort.update(
                    sv_to_boxmot(detections),
                    frame,  # BGR uint8
                )
                if tracks.size:
                    detections = sv.Detections(
                        xyxy=tracks[:, :4],
                        confidence=tracks[:, 5],
                        class_id=tracks[:, 6].astype(int),
                        tracker_id=tracks[:, 4].astype(int),
                    )
                else:  # StrongSORT has no active tracks
                    detections = sv.Detections(
                        xyxy=np.empty((0, 4), dtype=float),
                        confidence=np.empty(0, dtype=float),
                        class_id=np.empty(0, dtype=int),
                        tracker_id=np.empty(0, dtype=int),
                    )
            else:  # ByteTrack
                detections = byte_track.update_with_detections(detections=detections)

            # ─── FIND WHICH DETECTIONS ARE INSIDE EITHER CURB-SIDE ZONE ───
            in_left = left_zone.trigger(detections)
            in_right = right_zone.trigger(detections)
            in_zone = in_left | in_right  # boolean mask, same length as detections

            # Decide live colour: green until encroached ≥ 30 s, then red
            left_enc = any(side == "left" for side in enc_id_to_zone_side.values())
            right_enc = any(side == "right" for side in enc_id_to_zone_side.values())

            if not args.no_blend_zones:
                blend_zone(frame, mask_left,
                           (0, 255, 0) if not left_enc else (0, 0, 255))
                blend_zone(frame, mask_right,
                           (0, 255, 0) if not right_enc else (0, 0, 255))

            # ─── PROCESS MEMBERSHIP FOR EACH VEHICLE ───
            for det_idx, inside in enumerate(in_zone):
                tid = int(detections.tracker_id[det_idx])

                # current WORLD-metre position from Kalman filter
                if tid not in kf_states:
                    continue  # should not happen, but be safe
                Xi_m, Yi_m, *_ = kf_states[tid].statePost.flatten()

                if inside:
                    # first time the vehicle touches the curb-side lane
                    if tid not in enc_state:
                        enc_state[tid] = dict(t0=frame_idx, X0=Xi_m, Y0=Yi_m)
                    else:
                        s = enc_state[tid]
                        dt = (frame_idx - s['t0']) / video_info.fps  # seconds inside
                        dist = math.hypot(Xi_m - s['X0'], Yi_m - s['Y0'])  # metres travelled

                        # ---- inside the “flag encroachment” block ----
                        if dt >= ENCROACH_SECS and dist < MOVE_THRESH_METRES:
                            cls_id = int(detections.class_id[det_idx])
                            cls_name = model.names[cls_id]  # e.g. "car", "bus"

                            enc_events.append({
                                'tracker_id': tid,
                                'class_id': cls_id,
                                'class_name': cls_name,
                                'zone': 'left' if in_left[det_idx] else 'right',
                                't_entry_s': s['t0'] / video_info.fps,
                                't_flag_s': frame_idx / video_info.fps,
                                'd_move_m': round(dist, 2)
                            })
                            # optional: draw label here
                            enc_active_ids.add(tid)
                            enc_id_to_zone_side[tid] = 'left' if in_left[det_idx] else 'right'

                else:
                    enc_state.pop(tid, None)  # existing line
                    enc_active_ids.discard(tid)  # new
                    enc_id_to_zone_side.pop(tid, None)  # new
            active_ids = set(detections.tracker_id.tolist())
            # ─── 2) Update last_seen for any currently detected IDs ───
            # safely extract a list of IDs (or empty list if none)
            if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                tracker_ids = detections.tracker_id.tolist()
            else:
                tracker_ids = []

            for tid in tracker_ids:
                last_seen_frame[tid] = frame_idx



            points = detections.get_anchors_coordinates(
                anchor=sv.Position.CENTER
            )
            points = view_transformer.transform_points(points=points).astype(np.float32)
            # print("DETECTED IDS:", detections.tracker_id.tolist())
            for det_idx, (tracker_id, [x, y]) in enumerate(zip(detections.tracker_id, points)):
                dt = 1 / video_info.fps
                if tracker_id not in kf_states:
                    kf = create_kalman_filter(dt)
                    kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
                    kf_states[tracker_id] = kf
                else:
                    kf = kf_states[tracker_id]
                    # if fps can vary, you can still update:
                    kf.transitionMatrix[0, 2] = dt
                    kf.transitionMatrix[1, 3] = dt

                kf.predict()
                kf.correct(np.array([[x], [y]], np.float32))
                Xf, Yf, Vx, Vy = kf.statePost.flatten()
                if args.segment_speed:
                    # ⇢ cur_p  : current pixel centre   (x_px, y_px)
                    # ⇢ Xf, Yf : current BEV metres     (already available)
                    # tracker_id: current ID

                    bbox = detections.xyxy[det_idx].astype(float)
                    p_cur = ((bbox[0] + bbox[2]) * 0.5,
                             (bbox[1] + bbox[3]) * 0.5)

                    # --------- entry / exit with sign-flip test ----------
                    prev_side = segment_state.setdefault(
                        tracker_id,
                        dict(p_prev=p_cur, side_prev=line_side(p_cur, *ENTRY_LINE))
                    )['side_prev']
                    curr_side = line_side(p_cur, *ENTRY_LINE)

                    # 1) crossed ENTRY?
                    st = segment_state[tracker_id]
                    if 't0' not in st and prev_side * curr_side < 0:
                        st.update(t0=frame_idx, x0m=Xf, y0m=Yf)

                    # 2) crossed EXIT?
                    prev_exit = line_side(st['p_prev'], *EXIT_LINE)
                    curr_exit = line_side(p_cur, *EXIT_LINE)
                    if 't0' in st and 't_exit' not in st and prev_exit * curr_exit < 0:
                        t1 = frame_idx
                        dt = (t1 - st['t0']) / video_info.fps
                        if dt:  # guard /0
                            dx = Xf - st['x0m']
                            dy = Yf - st['y0m']
                            v_ms = math.hypot(dx, dy) / dt
                            segment_results.append([
                                tracker_id, st['t0'], t1,
                                round(math.hypot(dx, dy), 2),
                                round(dt, 3),
                                round(v_ms, 2),
                                round(v_ms * 3.6, 2)
                            ])
                        st['t_exit'] = t1

                    # update memory for next frame
                    st['p_prev'] = p_cur
                    st['side_prev'] = curr_side

                # 2) UPDATE “last seen” for all detections
                # safely extract a list of IDs (or empty list if none)
                if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                    tracker_ids = detections.tracker_id.tolist()
                else:
                    tracker_ids = []

                for tid in tracker_ids:
                    last_seen_frame[tid] = frame_idx
                # print("ACTIVE IDS:", active_ids)

                # --- Compute TTC between this vehicle and others (two-way traffic) ---
                # tag this track’s direction
                direction = "downward" if Vy > 0 else "upward"

                # ─── PARAMS ───
                COLLISION_DISTANCE = 2 # m, treat as contact if centres < 2 m apart

                for other_id, other_kf in kf_states.items():
                    if other_id == tracker_id:
                        continue
                    if frame_idx - last_seen_frame.get(other_id, 0) > MAX_AGE_FRAMES:
                        continue

                    # State vectors
                    Xi, Yi, Vxi, Vyi = kf.statePost.flatten()
                    Xj, Yj, Vxj, Vyj = other_kf.statePost.flatten()

                    # Relative motion (j relative to i)
                    rx0, ry0 = Xj - Xi, Yj - Yi  # initial separation (m)
                    vx, vy = Vxj - Vxi, Vyj - Vyi  # relative velocity (m/s)

                    denom = vx * vx + vy * vy
                    if denom == 0:  # same velocity → never converge
                        continue

                    t_star = -(rx0 * vx + ry0 * vy) / denom  # time of closest approach (s)

                    if 0 < t_star <= args.ttc_threshold:  # within look-ahead window
                        # Separation at that instant
                        dx = rx0 + vx * t_star
                        dy = ry0 + vy * t_star
                        d_closest = math.hypot(dx, dy)

                        if d_closest <= COLLISION_DISTANCE:
                            follower_class = id_to_class.get(tracker_id, "unknown")
                            leader_class = id_to_class.get(other_id, "unknown")
                            rel_speed = math.hypot(vx, vy)  # m/s

                            ttc_rows.append([
                                frame_idx,
                                tracker_id, follower_class,
                                other_id, leader_class,
                                round(d_closest, 2),  # min distance
                                round(rel_speed, 2),  # relative speed
                                round(t_star, 2)  # TTC
                            ])

                            ttc_labels[tracker_id] = [f"TTC->#{other_id}:{t_star:.1f}s"]
                            ttc_event_count += 1
                # record raw coords for history
                coordinates[tracker_id].append((x, y))


                # 5) Debug print
                # print(f'vx={Vx:.2f}, vy={Vy:.2f}')


                future_positions = predict_future_positions_kf(
                    kf,
                    delta_t=args.future_prediction_interval,
                    num_predictions=args.num_future_predictions
                )

                # Transform future (meter) points to pixel coordinates
                future_positions_array = np.array(future_positions, dtype=np.float32)
                predicted_pixels = view_transformer.inverse_transform_points(future_positions_array)

                # Save predicted pixel positions
                future_coordinates[tracker_id] = predicted_pixels.tolist()
                # if not (0 <= Xf <= TARGET_WIDTH and 0 <= Yf <= TARGET_HEIGHT-1):
                #     # vehicle has left → drop its KF state and future preds
                #     kf_states.pop(tracker_id, None)
                #     future_coordinates.pop(tracker_id, None)
                #     continue
            to_remove = []
            for tid in list(kf_states.keys()):
                last_seen = last_seen_frame.get(tid, None)
                if last_seen is None or (frame_idx - last_seen) > MAX_AGE_FRAMES:
                    to_remove.append(tid)
            for tid in to_remove:
                kf_states.pop(tid, None)
                future_coordinates.pop(tid, None)
                last_seen_frame.pop(tid, None)
            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    (x_start, y_start) = coordinates[tracker_id][0]
                    (x_end, y_end) = coordinates[tracker_id][-1]

                    dx = x_end - x_start
                    dy = y_end - y_start

                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time # meters per second
                    det_idx = np.where(detections.tracker_id == tracker_id)[0][0]

                    class_id = int(detections.class_id[det_idx])  # numeric
                    class_name = result.names[class_id]  # "car", "bus", …
                    id_to_class[tracker_id] = class_name  # keep class for every active ID
                    conf = float(detections.confidence[det_idx])  # 0-1 range

                    # append to CSV (speed already km/h from earlier speed*3.6 conversion)
                    csv_rows.append([
                        frame_idx,  # frame number
                        int(tracker_id),  # vehicle id
                        class_name,  # readable class
                        conf,  # confidence
                        round(speed * 3.6, 2)  # speed km/h
                    ])




                    label = f"#{tracker_id} {int(speed * 3.6)} km/h"
                    if tracker_id in ttc_labels:
                        label += " | " + ttc_labels[tracker_id][0]
                    labels.append(label)
            if args.segment_speed:
                # ─── Segment overlays ─────────────────────────────────────────────
                ENTRY_COLOR = (0, 255, 255)  # BGR  yellow
                EXIT_COLOR = (0, 0, 255)  # BGR  red
                THICKNESS = 3  # pixels

                # draw the plain lines
                cv2.line(frame, tuple(ENTRY_LINE[0]), tuple(ENTRY_LINE[1]),
                         ENTRY_COLOR, THICKNESS)
                cv2.arrowedLine(frame, tuple(EXIT_LINE[0]), tuple(EXIT_LINE[1]),
                                EXIT_COLOR, THICKNESS, tipLength=0.05)  # arrow shows flow


                # optional text labels centred on each line
                def _midpt(a, b):
                    return (int((a[0] + b[0]) * 0.5), int((a[1] + b[1]) * 0.5))


                cv2.putText(frame, "ENTRY", _midpt(*ENTRY_LINE),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ENTRY_COLOR, 2, cv2.LINE_AA)

                cv2.putText(frame, "EXIT", _midpt(*EXIT_LINE),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, EXIT_COLOR, 2, cv2.LINE_AA)
                # ──────────────────────────────────────────────────────────────────

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            # ─── draw future trajectories as DOTS with per-ID colours ───
            h, w = annotated_frame.shape[:2]

            for tid in list(future_coordinates.keys()):
                # 1) drop stale tracks
                if tid not in active_ids:
                    future_coordinates.pop(tid, None)
                    continue

                # 2) keep only points inside the frame
                in_bounds = [
                    (int(x), int(y))
                    for x, y in future_coordinates[tid]
                    if 0 <= x < w and 0 <= y < h
                ]
                if not in_bounds:  # nothing to plot
                    continue

                # colour cache (one colour per track ID)
                colour = future_colors.setdefault(
                    tid,
                    (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))
                )

                # 3) draw each future point as a filled circle
                for cx, cy in in_bounds:
                    cv2.circle(
                        annotated_frame,
                        (cx, cy),
                        radius=4,  # dot size; tweak as needed
                        color=colour,
                        thickness=-1,  # –1 ⇒ filled
                        lineType=cv2.LINE_AA
                    )
            # for det_idx in range(len(detections)):
            #     tid = int(detections.tracker_id[det_idx])
            #     x1, y1, x2, y2 = detections.xyxy[det_idx].astype(int)
            #
            #     colour = (0, 0, 255) if tid in enc_active_ids else (255, 255, 0)
            #     annotated_frame=cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            # then continue with box_annotator and label_annotator as before
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(annotated_frame)
            # Resize frame for display only
            display_frame = cv2.resize(annotated_frame, (1920, 1080))  # Width x Height
            # ----- after drawing on `frame_out` -------
            if DISPLAY:
                cv2.imshow("Preview", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if DISPLAY:
            cv2.destroyAllWindows()

        # ➡ progress-bar update goes here
        bar.set_postfix(events=ttc_event_count)
        bar.update(1)

        df = pd.DataFrame(
            csv_rows,
            columns=["frame", "vehicle_id", "vehicle_class", "confidence", "speed_km_h"]
        )

        out_dir = Path("results")  # any folder you like
        out_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")  # 10052025_214500
        csv_file = out_dir / f"vehicle_metrics_{timestamp}.csv"

        # ─── NEW: write TTC-event CSV ───
        ttc_df = pd.DataFrame(
            ttc_rows,
            columns=[
                "frame",
                "follower_id", "follower_class",
                "leader_id", "leader_class",
                "closing_distance_m",
                "relative_velocity_m_s",
                "ttc_s"  # ← NEW
            ]
        )
        ttc_csv_file = out_dir / f"ttc_events_{timestamp}.csv"
        ttc_df.to_csv(ttc_csv_file, index=False)

        enc_csv_file = out_dir / f"enc_events_{timestamp}.csv"
        pd.DataFrame(enc_events).to_csv(enc_csv_file, index=False)
        print(f"[encroachment] {len(enc_events)} events saved → encroachments.csv")

        if args.segment_speed and segment_results:
            pd.DataFrame(
                segment_results,
                columns=["vehicle_id", "frame_entry", "frame_exit", "distance_m",
                         "time_s", "speed_m_s", "speed_km_h"]
            ).to_csv(out_dir / f"segment_speeds_{timestamp}.csv", index=False)

        bar.close()
        print(f"Total TTC events logged: {ttc_event_count}")

        elapsed = tm.time() - t0
        fps = frame_idx / elapsed  # frame_idx == processed frames
        print(f"Done: {frame_idx} frames in {elapsed:.1f}s  ({fps:.2f} FPS)")