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

        # State tracking
        self.enc_state: Dict[int, Dict] = {}  # tracker_id → {t0, X0, Y0}
        self.enc_events: List[Dict] = []
        self.enc_active_ids: Set[int] = set()
        self.enc_id_to_zone_side: Dict[int, str] = {}

        # Initialize masks as None (created on demand)
        self._mask_left = None
        self._mask_right = None

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

    def check_encroachment(self, detections: sv.Detections, frame_idx: int,
                      kf_states: Dict) -> List[Dict]:
        """Check for encroachment events and update state."""
        # Find which detections are inside zones
        in_left = self.left_zone.trigger(detections)
        in_right = self.right_zone.trigger(detections)
        in_zone = in_left | in_right

        new_events = []

        for det_idx, inside in enumerate(in_zone):
            tid = int(detections.tracker_id[det_idx])

            # Get current world position from Kalman filter
            if tid not in kf_states:
                continue

            state_vector = kf_states[tid].statePost.flatten()
            Xi_m, Yi_m = state_vector[0], state_vector[1]

            if inside:
                if tid not in self.enc_state:
                    # First time entering zone
                    self.enc_state[tid] = dict(t0=frame_idx, X0=Xi_m, Y0=Yi_m)
                else:
                    s = self.enc_state[tid]
                    dt = (frame_idx - s['t0']) / self.video_fps
                    dist = math.hypot(Xi_m - s['X0'], Yi_m - s['Y0'])

                    # Check encroachment conditions
                    if (dt >= self.encroach_secs and
                        dist < self.move_thresh_metres and
                        tid not in self.enc_active_ids):

                        cls_id = int(detections.class_id[det_idx])
                        event = {
                            'tracker_id': tid,
                            'class_id': cls_id,
                            'zone': 'left' if in_left[det_idx] else 'right',
                            't_entry_s': s['t0'] / self.video_fps,
                            't_flag_s': frame_idx / self.video_fps,
                            'd_move_m': round(dist, 2)
                        }

                        new_events.append(event)
                        self.enc_events.append(event)
                        self.enc_active_ids.add(tid)
                        self.enc_id_to_zone_side[tid] = event['zone']
            else:
                # Vehicle left zone
                if tid in self.enc_active_ids:
                    # Find the event to update
                    for event in reversed(self.enc_events):
                        if event['tracker_id'] == tid and 't_exit_s' not in event:
                            event['t_exit_s'] = frame_idx / self.video_fps
                            break

                # Clean up the vehicle's current state regardless
                self.enc_state.pop(tid, None)
                self.enc_active_ids.discard(tid)
                self.enc_id_to_zone_side.pop(tid, None)

        return new_events

    def handle_tracker_removal(self, tracker_id: int, exit_timestamp: float):
        """Handle tracker removal by updating any active encroachment events with exit timestamp.
        
        Args:
            tracker_id: The ID of the tracker being removed
            exit_timestamp: The timestamp (in seconds) when the tracker was removed
        """
        if tracker_id in self.enc_active_ids:
            # Remove from active encroachment IDs
            self.enc_active_ids.discard(tracker_id)
            
            # Find the most recent encroachment event for this tracker and add exit timestamp
            for event in reversed(self.enc_events):
                if event['tracker_id'] == tracker_id and 't_exit_s' not in event:
                    event['t_exit_s'] = exit_timestamp
                    break
        
        # Clean up any remaining state for this tracker
        self.enc_state.pop(tracker_id, None)
        self.enc_id_to_zone_side.pop(tracker_id, None)

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
        if blend and (self._mask_left is not None):
            # Determine colors based on encroachment state
            left_enc = any(side == "left" for side in self.enc_id_to_zone_side.values())
            right_enc = any(side == "right" for side in self.enc_id_to_zone_side.values())

            self.blend_zone(frame, self._mask_left,
                           (0, 0, 255) if left_enc else (0, 255, 0))
            self.blend_zone(frame, self._mask_right,
                           (0, 0, 255) if right_enc else (0, 255, 0))

        return frame

    def get_encroachment_events(self) -> List[Dict]:
        """Get all encroachment events."""
        return self.enc_events