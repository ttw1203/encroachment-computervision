"""Object detection and tracking functionality."""
from typing import Optional, Dict, Tuple
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from boxmot import StrongSort


class DetectionTracker:
    """Manages object detection and tracking."""

    def __init__(self, model_weights: str, device: str, tracker_type: str = "strongsort",
                 confidence_threshold: float = 0.3, video_fps: float = 30.0):
        """Initialize detection and tracking components."""
        self.model = YOLO(model_weights)
        self.model.to(device)
        self.device = device
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.video_fps = video_fps

        # Initialize tracker
        if tracker_type == "strongsort":
            self.tracker = StrongSort(
                reid_weights=Path("mobilenetv2_x1_4_dukemtmcreid.pt"),
                device=0 if device != "cpu" else device,
                half=False,
                max_age=300,
            )
        else:  # ByteTrack
            self.tracker = sv.ByteTrack(
                frame_rate=video_fps,
                track_activation_threshold=confidence_threshold,
            )

    def detect_and_track(self, frame: np.ndarray,
                         polygon_zone: Optional[sv.PolygonZone] = None,
                         iou_threshold: float = 0.7) -> sv.Detections:
        """Run detection and tracking on a frame."""
        # Run detection
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter by confidence
        detections = detections[detections.confidence > self.confidence_threshold]

        # Apply polygon zone filter if provided
        if polygon_zone:
            detections = detections[polygon_zone.trigger(detections)]

        # Apply NMS
        detections = detections.with_nms(threshold=iou_threshold)

        # Apply tracking
        if self.tracker_type == "strongsort":
            tracks = self.tracker.update(
                sv_to_boxmot(detections),
                frame,
            )
            if tracks.size:
                detections = sv.Detections(
                    xyxy=tracks[:, :4],
                    confidence=tracks[:, 5],
                    class_id=tracks[:, 6].astype(int),
                    tracker_id=tracks[:, 4].astype(int),
                )
            else:
                detections = sv.Detections(
                    xyxy=np.empty((0, 4), dtype=float),
                    confidence=np.empty(0, dtype=float),
                    class_id=np.empty(0, dtype=int),
                    tracker_id=np.empty(0, dtype=int),
                )
        else:  # ByteTrack
            detections = self.tracker.update_with_detections(detections=detections)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID."""
        return self.model.names[class_id]


def sv_to_boxmot(det: sv.Detections) -> np.ndarray:
    """Convert supervision detections to boxmot format."""
    if not len(det):
        return np.empty((0, 6), dtype=np.float32)
    return np.hstack((det.xyxy,
                      det.confidence[:, None],
                      det.class_id[:, None])).astype(np.float32)


def filter_rider_persons(det: sv.Detections, iou_thr: float = 0.50) -> sv.Detections:
    """Discard person boxes that belong to motorcycle/bicycle riders."""
    if len(det) == 0:
        return det

    cls = det.class_id
    boxes = det.xyxy

    person_idx = np.where(cls == 0)[0]  # 0 = person (COCO)
    vehicle_idx = np.where(np.isin(cls, [1, 3]))[0]  # 1=bicycle, 3=motorcycle

    if person_idx.size == 0 or vehicle_idx.size == 0:
        return det

    keep = np.ones(len(det), dtype=bool)

    for p in person_idx:
        px1, py1, px2, py2 = boxes[p]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2

        for v in vehicle_idx:
            vx1, vy1, vx2, vy2 = boxes[v]
            v_cy = (vy1 + vy2) / 2

            iou_ok = _iou_xyxy(boxes[p], boxes[v]) > iou_thr
            centre_inside = (vx1 <= pcx <= vx2) and (vy1 <= pcy <= vy2)
            above_ok = pcy < v_cy

            if (iou_ok or centre_inside) and above_ok:
                keep[p] = False
                break

    return det[keep]


def _iou_xyxy(box_a, box_b) -> float:
    """Fast scalar IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union