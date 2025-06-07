"""Zone management for encroachment detection and visualization."""
from typing import Dict, Tuple, List, Optional, Set
import cv2
import numpy as np
import supervision as sv
import math


class ZoneManager:
    """Manages polygonal zones and encroachment detection."""

    def __init__(self, left_zone_poly: np.ndarray, right_zone_poly: np.ndarray,
                 video_fps: float, encroach_secs: float = 1.0,
                 move_thresh_metres: float = 1.0):
        """Initialize zone manager with polygon definitions."""
        self.left_zone_poly = left_zone_poly
        self.right_zone_poly = right_zone_poly
        self.video_fps = video_fps
        self.encroach_secs = encroach_secs
        self.move_thresh_metres = move_thresh_metres

        # Create PolygonZone objects
        self.left_zone = sv.PolygonZone(polygon=left_zone_poly)
        self.right_zone = sv.PolygonZone(polygon=right_zone_poly)

        # Simplified state tracking - use lists instead of nested dicts
        self.enc_tracker_ids: List[int] = []
        self.enc_entry_frames: List[int] = []
        self.enc_entry_positions: List[Tuple[float, float]] = []
        self.enc_events: List[Dict] = []
        self.enc_active_ids: Set[int] = set()
        self.enc_id_to_zone_side: Dict[int, str] = {}

        # Initialize masks as None (created on demand)
        self._mask_left = None
        self._mask_right = None

        # Cache for zone trigger results
        self._cached_detections_hash = None
        self._cached_trigger_results = None
        self._zones_changed = True  # Track if zone state changed for drawing optimization

    def create_masks(self, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create zone masks for visualization."""
        H, W = frame_shape
        self._mask_left = np.zeros((H, W), np.uint8)
        self._mask_right = np.zeros((H, W), np.uint8)

        # Convert to OpenCV contour format
        left_cnt = self.left_zone_poly.reshape((-1, 1, 2))
        right_cnt = self.right_zone_poly.reshape((-1, 1, 2))

        cv2.fillPoly(self._mask_left, [left_cnt], 255)
        cv2.fillPoly(self._mask_right, [right_cnt], 255)

        return self._mask_left, self._mask_right

    def _hash_detections(self, detections: sv.Detections) -> int:
        """Create a simple hash of detection positions for caching."""
        if len(detections.xyxy) == 0:
            return hash(())

        # Use center points of bounding boxes for hash
        centers = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2, \
                 (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
        return hash((tuple(centers[0]), tuple(centers[1])))

    def _get_zone_triggers_cached(self, detections: sv.Detections) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get zone trigger results with caching."""
        det_hash = self._hash_detections(detections)

        if (self._cached_detections_hash == det_hash and
            self._cached_trigger_results is not None):
            return self._cached_trigger_results

        # Compute fresh results
        in_left = self.left_zone.trigger(detections)
        in_right = self.right_zone.trigger(detections)
        in_zone = in_left | in_right

        # Cache results
        self._cached_detections_hash = det_hash
        self._cached_trigger_results = (in_left, in_right, in_zone)

        return in_left, in_right, in_zone

    def check_encroachment(self, detections: sv.Detections, frame_idx: int,
                          kf_states: Dict) -> List[Dict]:
        """Check for encroachment events and update state."""
        # Use cached zone triggers
        in_left, in_right, in_zone = self._get_zone_triggers_cached(detections)

        new_events = []
        zones_changed_this_frame = False

        for det_idx, inside in enumerate(in_zone):
            tid = int(detections.tracker_id[det_idx])

            # Get current world position from Kalman filter
            if tid not in kf_states:
                continue

            state_vector = kf_states[tid].statePost.flatten()
            Xi_m, Yi_m = state_vector[0], state_vector[1]

            if inside:
                # Find existing entry or create new one
                try:
                    enc_idx = self.enc_tracker_ids.index(tid)
                    # Existing entry - check encroachment conditions
                    dt = (frame_idx - self.enc_entry_frames[enc_idx]) / self.video_fps
                    X0, Y0 = self.enc_entry_positions[enc_idx]
                    dist = math.hypot(Xi_m - X0, Yi_m - Y0)

                    # Check encroachment conditions
                    if (dt >= self.encroach_secs and
                        dist < self.move_thresh_metres and
                        tid not in self.enc_active_ids):

                        cls_id = int(detections.class_id[det_idx])
                        zone_side = 'left' if in_left[det_idx] else 'right'
                        event = {
                            'tracker_id': tid,
                            'class_id': cls_id,
                            'zone': zone_side,
                            't_entry_s': self.enc_entry_frames[enc_idx] / self.video_fps,
                            't_flag_s': frame_idx / self.video_fps,
                            'd_move_m': round(dist, 2)
                        }

                        new_events.append(event)
                        self.enc_events.append(event)
                        self.enc_active_ids.add(tid)
                        self.enc_id_to_zone_side[tid] = zone_side
                        zones_changed_this_frame = True

                except ValueError:
                    # First time entering zone - add to tracking lists
                    self.enc_tracker_ids.append(tid)
                    self.enc_entry_frames.append(frame_idx)
                    self.enc_entry_positions.append((Xi_m, Yi_m))
            else:
                # Vehicle left zone - remove from tracking
                try:
                    enc_idx = self.enc_tracker_ids.index(tid)
                    self.enc_tracker_ids.pop(enc_idx)
                    self.enc_entry_frames.pop(enc_idx)
                    self.enc_entry_positions.pop(enc_idx)
                    zones_changed_this_frame = True
                except ValueError:
                    pass  # Not in tracking list

                if tid in self.enc_active_ids:
                    self.enc_active_ids.discard(tid)
                    self.enc_id_to_zone_side.pop(tid, None)
                    zones_changed_this_frame = True

        # Update zones changed flag
        if zones_changed_this_frame:
            self._zones_changed = True

        return new_events

    def blend_zone(self, frame: np.ndarray, mask: np.ndarray,
                   colour: Tuple[int, int, int], alpha: float = 0.35) -> None:
        """In-place alpha blend of colour wherever mask == 255."""
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[:] = colour
        frame[:] = np.where(mask[..., None] == 255,
                           cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0),
                           frame)

    def draw_zones(self, frame: np.ndarray, blend: bool = True) -> np.ndarray:
        """Draw zones on frame with appropriate colors."""
        # Skip blending when zones haven't changed
        if blend and self._mask_left is not None and not self._zones_changed:
            return frame

        if blend and (self._mask_left is not None):
            # Determine colors based on encroachment state
            left_enc = any(side == "left" for side in self.enc_id_to_zone_side.values())
            right_enc = any(side == "right" for side in self.enc_id_to_zone_side.values())

            self.blend_zone(frame, self._mask_left,
                           (0, 0, 255) if left_enc else (0, 255, 0))
            self.blend_zone(frame, self._mask_right,
                           (0, 0, 255) if right_enc else (0, 255, 0))

            # Reset zones changed flag after drawing
            self._zones_changed = False

        return frame

    def get_encroachment_events(self) -> List[Dict]:
        """Get all encroachment events."""
        return self.enc_events