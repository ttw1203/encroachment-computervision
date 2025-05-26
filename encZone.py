import argparse
from collections import defaultdict, deque
import pandas as pd
import logging
import math
from typing import List, Tuple, Dict, Optional

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

VIDEO_PATH = os.getenv("VIDEO_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
ZONE_CHECK_PNG = os.getenv("ZONE_CHECK_PNG")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS")
ENC_ZONE_CONFIG = os.getenv("ENC_ZONE_CONFIG")
DEVICE = os.getenv("DEVICE", "cpu")

# ============ OPTIMIZED STRUCTURES FOR ENHANCED TTC ============
# Store vehicle dimensions for each tracker ID
vehicle_dims: Dict[int, Tuple[float, float]] = {}  # tracker_id -> (width_m, height_m)
vehicle_dims_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))  # for smoothing

# Cache for trajectory predictions to avoid recalculation
trajectory_cache: Dict[int, Tuple[List[Tuple[float, float]], float]] = {}  # tracker_id -> (positions, last_update_time)


# ============ SIMPLIFIED AND OPTIMIZED UTILITY FUNCTIONS ============

def quick_box_distance_check(x1: float, y1: float, x2: float, y2: float,
                             threshold: float = 10.0) -> bool:
    """Quick distance check to avoid expensive calculations"""
    return math.hypot(x2 - x1, y2 - y1) < threshold


def simplified_box_overlap(box1_center: Tuple[float, float], box1_dims: Tuple[float, float],
                           box2_center: Tuple[float, float], box2_dims: Tuple[float, float],
                           safety_margin: float = 0.5) -> bool:
    """
    Simplified axis-aligned bounding box overlap check.
    Assumes boxes are roughly aligned with direction of travel.
    """
    # Add safety margin to dimensions
    w1, h1 = box1_dims[0] + safety_margin, box1_dims[1] + safety_margin
    w2, h2 = box2_dims[0] + safety_margin, box2_dims[1] + safety_margin

    # Check if boxes overlap
    dx = abs(box1_center[0] - box2_center[0])
    dy = abs(box1_center[1] - box2_center[1])

    return dx < (w1 + w2) / 2 and dy < (h1 + h2) / 2


def predict_future_positions_optimized(kf, delta_t: float = 0.5,
                                       num_predictions: int = 6,
                                       max_distance: float = 30.0) -> List[Tuple[float, float, float]]:
    """
    Optimized trajectory prediction - fewer points, simpler calculation.
    Returns list of (x, y, time) tuples.
    """
    positions = []
    x, y, vx, vy = kf.statePost.flatten()

    # Skip if vehicle is nearly stationary
    speed = math.hypot(vx, vy)
    if speed < 0.5:  # Less than 0.5 m/s
        return []

    for step in range(1, num_predictions + 1):
        t = step * delta_t
        x_future = x + vx * t
        y_future = y + vy * t

        # Check distance limit
        dist = math.hypot(x_future - x, y_future - y)
        if dist > max_distance:
            break

        positions.append((x_future, y_future, t))

    return positions


def calculate_simple_ttc(kf1, kf2, vehicle1_dims: Tuple[float, float],
                         vehicle2_dims: Tuple[float, float]) -> Optional[Dict]:
    """
    Simplified TTC calculation using current states and velocities.
    Much faster than full trajectory envelope intersection.
    """
    # Get states
    x1, y1, vx1, vy1 = kf1.statePost.flatten()
    x2, y2, vx2, vy2 = kf2.statePost.flatten()

    # Quick distance check
    current_dist = math.hypot(x2 - x1, y2 - y1)
    if current_dist > 20.0:  # Ignore vehicles more than 20m apart
        return None

    # Relative motion
    dvx = vx2 - vx1
    dvy = vy2 - vy1
    rel_speed = math.hypot(dvx, dvy)

    # Skip if relative speed is too low
    if rel_speed < 0.5:
        return None

    # Simple TTC calculation based on closest approach
    dx = x2 - x1
    dy = y2 - y1

    # Project when they'll be closest
    dot_product = dx * dvx + dy * dvy
    if dot_product >= 0:  # Moving away from each other
        return None

    # Time to closest approach
    ttc = -dot_product / (rel_speed ** 2)

    if ttc <= 0 or ttc > 3.0:  # Only consider next 3 seconds
        return None

    # Position at closest approach
    x1_future = x1 + vx1 * ttc
    y1_future = y1 + vy1 * ttc
    x2_future = x2 + vx2 * ttc
    y2_future = y2 + vy2 * ttc

    # Check if boxes would overlap at closest approach
    if simplified_box_overlap((x1_future, y1_future), vehicle1_dims,
                              (x2_future, y2_future), vehicle2_dims):
        # Determine collision type based on relative angle
        angle = abs(math.atan2(vy2 - vy1, vx2 - vx1) - math.atan2(vy1, vx1))
        angle = min(angle, 2 * math.pi - angle)  # Normalize to [0, π]

        collision_type = "rear_end" if angle < math.pi / 4 else "intersection"

        return {
            'ttc': ttc,
            'type': collision_type,
            'distance': current_dist,
            'rel_speed': rel_speed,
            'intersection_x': (x1_future + x2_future) / 2,
            'intersection_y': (y1_future + y2_future) / 2
        }

    return None


# ============ EXISTING FUNCTIONS (keep most unchanged) ============
# map tracker_id → BGR colour tuple
future_colors: dict[int, tuple[int, int, int]] = {}

# Show live preview while processing?
DISPLAY = False  # set True if you ever want to see the window again

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)  # ERROR or CRITICAL

SOURCE = np.array([[1281, 971], [2309, 971], [6090, 2160], [-2243, 2160]])

# Storing trajectory coordinates
csv_rows = []
# ─── NEW: store TTC events ───
ttc_rows = []  # rows for TTC-event CSV
id_to_class = {}  # tracker_id → latest class name

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


# [Include all the existing helper functions unchanged]
def load_zones(path: str) -> tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        data = yaml.safe_load(f) if ext in {".yml", ".yaml"} else json.load(f)

    left = np.asarray(data["left_zone"], np.int32)
    right = np.asarray(data["right_zone"], np.int32)
    return left, right


def blend_zone(frame: np.ndarray,
               mask: np.ndarray,
               colour: tuple[int, int, int],
               alpha: float = 0.35) -> None:
    """
    In-place α-blend of `colour` wherever mask == 255.
    `colour` is BGR, 0‒255. Alpha 0‒1.
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = colour
    frame[:] = np.where(mask[..., None] == 255,
                        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0),
                        frame)


def _iou_xyxy(box_a, box_b) -> float:
    """
    Fast scalar IoU between two [x1,y1,x2,y2] boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
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
        return det  # nothing to do

    cls = det.class_id
    boxes = det.xyxy

    person_idx = np.where(cls == 0)[0]  # 0 = person (COCO)
    vehicle_idx = np.where(np.isin(cls, [1, 3]))[0]  # 1=bicycle, 3=motorcycle

    if person_idx.size == 0 or vehicle_idx.size == 0:
        return det  # no pairing possible

    keep = np.ones(len(det), dtype=bool)

    for p in person_idx:
        px1, py1, px2, py2 = boxes[p]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2  # person centre

        for v in vehicle_idx:
            vx1, vy1, vx2, vy2 = boxes[v]
            v_cy = (vy1 + vy2) / 2  # vehicle centre-Y

            iou_ok = _iou_xyxy(boxes[p], boxes[v]) > iou_thr
            centre_inside = (vx1 <= pcx <= vx2) and (vy1 <= pcy <= vy2)
            above_ok = pcy < v_cy  # rider sits above centre

            if (iou_ok or centre_inside) and above_ok:
                keep[p] = False  # drop this person
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
    # 🚀 OPTIMIZED ARGUMENTS
    parser.add_argument(
        "--num_future_predictions",
        default=6,  # Reduced from 30 to 6 for performance
        help="Number of future points to predict per vehicle",
        type=int
    )
    parser.add_argument(
        "--future_prediction_interval",
        default=0.5,  # Increased from 0.1 to 0.5 for fewer calculations
        help="Time interval (seconds) between future predictions",
        type=float
    )
    parser.add_argument(
        "--ttc_threshold",
        default=3.0,
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
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    return kf


future_coordinates = defaultdict(list)  # Store future (x, y) points

if __name__ == "__main__":
    args = parse_arguments()

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

    # Note: We maintain "ghost" tracks briefly to handle temporary occlusions,
    # but TTC calculations are restricted to currently visible vehicles only
    MAX_AGE_SECONDS = 0.5  # Reduced from 1.0 to minimize ghost tracks
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
    MOVE_THRESH_METRES = 1.0  # "barely moved" in world metres

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

    # ============ PERFORMANCE TRACKING ============
    ttc_calc_times = []
    frame_times = []

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            frame_start_time = tm.time()

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
            # ✂ Ignore riders so that only true pedestrians remain
            detections = filter_rider_persons(detections, iou_thr=0.50)

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

            # ============ OPTIMIZED: BATCH TRANSFORM VEHICLE DIMENSIONS ============
            if len(detections) > 0:
                # Batch transform all box corners at once
                all_corners = []
                det_indices = []

                for i, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = box
                    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    all_corners.extend(corners)
                    det_indices.extend([i] * 4)

                if all_corners:
                    all_corners = np.array(all_corners)
                    all_corners_world = view_transformer.transform_points(all_corners)

                    # Process dimensions for each detection
                    for i, tracker_id in enumerate(detections.tracker_id):
                        start_idx = i * 4
                        corners_world = all_corners_world[start_idx:start_idx + 4]

                        # Calculate dimensions
                        width_m = np.linalg.norm(corners_world[1] - corners_world[0])
                        height_m = np.linalg.norm(corners_world[3] - corners_world[0])

                        # Store with smoothing
                        vehicle_dims_history[tracker_id].append((width_m, height_m))

                        if len(vehicle_dims_history[tracker_id]) > 0:
                            avg_width = np.mean([d[0] for d in vehicle_dims_history[tracker_id]])
                            avg_height = np.mean([d[1] for d in vehicle_dims_history[tracker_id]])
                            vehicle_dims[tracker_id] = (avg_width, avg_height)

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

                        # ---- inside the "flag encroachment" block ----
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

            # Update Kalman filters
            for tracker_id, [x, y] in zip(detections.tracker_id, points):
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

                # 2) UPDATE "last seen" for all detections
                # safely extract a list of IDs (or empty list if none)
                if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                    tracker_ids = detections.tracker_id.tolist()
                else:
                    tracker_ids = []

                for tid in tracker_ids:
                    last_seen_frame[tid] = frame_idx

                # ============ OPTIMIZED TTC CALCULATION ============
                # IMPORTANT: Only calculate TTC between vehicles that are BOTH currently visible
                # This prevents "ghost collisions" with vehicles that have left the frame
                ttc_start_time = tm.time()

                # Get vehicle dimensions
                my_dims = vehicle_dims.get(tracker_id, (2.0, 4.5))  # Default car size

                # Create a sorted list of nearby vehicles for efficient checking
                nearby_vehicles = []
                my_x, my_y = Xf, Yf

                # CRITICAL FIX: Only check TTC with vehicles currently visible in frame
                current_frame_ids = set(detections.tracker_id.tolist())

                for other_id, other_kf in kf_states.items():
                    if other_id == tracker_id:
                        continue
                    # Only consider vehicles that are currently detected
                    if other_id not in current_frame_ids:
                        continue

                    other_x, other_y = other_kf.statePost[0, 0], other_kf.statePost[1, 0]
                    dist = math.hypot(other_x - my_x, other_y - my_y)

                    if dist < 20.0:  # Only consider vehicles within 20m
                        nearby_vehicles.append((dist, other_id, other_kf))

                # Sort by distance and check only closest vehicles
                nearby_vehicles.sort(key=lambda x: x[0])

                for _, other_id, other_kf in nearby_vehicles[:5]:  # Check only 5 closest
                    # Get other vehicle dimensions
                    other_dims = vehicle_dims.get(other_id, (2.0, 4.5))

                    # Calculate simplified TTC
                    ttc_result = calculate_simple_ttc(
                        kf_states[tracker_id], other_kf,
                        my_dims, other_dims
                    )

                    if ttc_result and ttc_result['ttc'] <= args.ttc_threshold:
                        follower_class = id_to_class.get(tracker_id, "unknown")
                        leader_class = id_to_class.get(other_id, "unknown")

                        # Determine lane type based on edge distance
                        # Simplified calculation
                        lane_type = "same_lane" if ttc_result['type'] == "rear_end" else "cross_lane"

                        ttc_rows.append([
                            frame_idx,
                            tracker_id, follower_class,
                            other_id, leader_class,
                            round(ttc_result['distance'], 2),
                            round(ttc_result['rel_speed'], 2),
                            round(ttc_result['ttc'], 2),
                            ttc_result['type'],
                            round(ttc_result['intersection_x'], 2),
                            round(ttc_result['intersection_y'], 2),
                            lane_type
                        ])

                        # Double-check both vehicles are visible before showing TTC
                        if tracker_id in current_frame_ids and other_id in current_frame_ids:
                            ttc_labels[tracker_id] = [f"TTC->#{other_id}:{ttc_result['ttc']:.1f}s"]
                        ttc_event_count += 1
                        break  # Only show one TTC per vehicle

                ttc_calc_times.append(tm.time() - ttc_start_time)

                # record raw coords for history
                coordinates[tracker_id].append((x, y))

                # ============ SIMPLIFIED FUTURE VISUALIZATION ============
                # Only calculate for visualization if needed
                if len(active_ids) < 10:  # Only visualize for small number of vehicles
                    future_positions = predict_future_positions_optimized(
                        kf_states[tracker_id],
                        delta_t=args.future_prediction_interval,
                        num_predictions=args.num_future_predictions
                    )

                    if future_positions:
                        future_points = [(x, y) for x, y, _ in future_positions]
                        future_positions_array = np.array(future_points, dtype=np.float32)
                        predicted_pixels = view_transformer.inverse_transform_points(future_positions_array)
                        future_coordinates[tracker_id] = predicted_pixels.tolist()

            # Clean up stale tracks
            to_remove = []
            for tid in list(kf_states.keys()):
                last_seen = last_seen_frame.get(tid, None)
                if last_seen is None or (frame_idx - last_seen) > MAX_AGE_FRAMES:
                    to_remove.append(tid)
            for tid in to_remove:
                kf_states.pop(tid, None)
                future_coordinates.pop(tid, None)
                last_seen_frame.pop(tid, None)
                vehicle_dims.pop(tid, None)
                vehicle_dims_history.pop(tid, None)
                trajectory_cache.pop(tid, None)  # Clear cache

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
                    speed = distance / time  # meters per second
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
                for i, (cx, cy) in enumerate(in_bounds):
                    # Make dots progressively smaller/fainter for future points
                    radius = max(2, 4 - i)
                    cv2.circle(
                        annotated_frame,
                        (cx, cy),
                        radius=radius,
                        color=colour,
                        thickness=-1,  # –1 ⇒ filled
                        lineType=cv2.LINE_AA
                    )

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

            # Performance tracking
            frame_times.append(tm.time() - frame_start_time)

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

        # ============ PERFORMANCE REPORT ============
        print("\n=== Performance Report ===")
        print(f"Average frame processing time: {np.mean(frame_times):.3f}s")
        print(f"Average TTC calculation time: {np.mean(ttc_calc_times) if ttc_calc_times else 0:.4f}s")
        print(f"Max frame time: {np.max(frame_times):.3f}s")
        print(f"Processing FPS: {1 / np.mean(frame_times):.2f}")

        df = pd.DataFrame(
            csv_rows,
            columns=["frame", "vehicle_id", "vehicle_class", "confidence", "speed_km_h"]
        )

        out_dir = Path("results")  # any folder you like
        out_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")  # 10052025_214500
        csv_file = out_dir / f"vehicle_metrics_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # ─── ENHANCED TTC CSV with intersection details ───
        ttc_df = pd.DataFrame(
            ttc_rows,
            columns=[
                "frame",
                "follower_id", "follower_class",
                "leader_id", "leader_class",
                "closing_distance_m",
                "relative_velocity_m_s",
                "ttc_s",
                "collision_type",  # 'rear_end' or 'intersection'
                "intersection_x",  # conflict point coordinates
                "intersection_y",
                "lane_type"  # 'same_lane' or 'cross_lane'
            ]
        )
        ttc_csv_file = out_dir / f"ttc_events_{timestamp}.csv"
        ttc_df.to_csv(ttc_csv_file, index=False)

        enc_csv_file = out_dir / f"enc_events_{timestamp}.csv"
        pd.DataFrame(enc_events).to_csv(enc_csv_file, index=False)
        print(f"[encroachment] {len(enc_events)} events saved → {enc_csv_file}")

        bar.close()
        print(f"Total TTC events logged: {ttc_event_count}")
        print(f"Enhanced TTC results saved → {ttc_csv_file}")

        elapsed = tm.time() - t0
        fps = frame_idx / elapsed  # frame_idx == processed frames
        print(f"Done: {frame_idx} frames in {elapsed:.1f}s  ({fps:.2f} FPS)")