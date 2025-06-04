"""Event processing for TTC and segment speed calculations."""
from typing import Dict, List, Tuple, Optional, Callable
import math
import numpy as np
from collections import defaultdict
from src.geometry_and_transforms import line_side


class EventProcessor:
    """Processes various event types including TTC and segment speeds."""

    def __init__(self, video_fps: float, collision_distance: float = 2.0,
                 calibration_func: Optional[Callable[[float], float]] = None):
        """Initialize event processor.

        Args:
            video_fps: Frames per second of the video
            collision_distance: Distance threshold for collision detection (meters)
            calibration_func: Optional function to calibrate speeds (takes raw speed, returns calibrated)
        """
        self.video_fps = video_fps
        self.collision_distance = collision_distance
        self.calibration_func = calibration_func
        self.segment_state: Dict = {}
        self.segment_results: List = []
        self.ttc_rows: List = []
        self.id_to_class: Dict[int, str] = {}

    def calculate_ttc(self, tracker_id: int, kf_states: Dict,
                      last_seen_frame: Dict, current_frame: int,
                      max_age_frames: int, ttc_threshold: float) -> Optional[Dict]:
        """Calculate Time-to-Collision for a vehicle with others."""
        if tracker_id not in kf_states:
            return None

        kf = kf_states[tracker_id]
        Xi, Yi, Vxi, Vyi = kf.statePost.flatten()

        ttc_events = []

        for other_id, other_kf in kf_states.items():
            if other_id == tracker_id:
                continue

            # Skip if other vehicle hasn't been seen recently
            if current_frame - last_seen_frame.get(other_id, 0) > max_age_frames:
                continue

            # Get other vehicle's state
            Xj, Yj, Vxj, Vyj = other_kf.statePost.flatten()

            # Relative motion
            rx0, ry0 = Xj - Xi, Yj - Yi
            vx, vy = Vxj - Vxi, Vyj - Vyi

            denom = vx * vx + vy * vy
            if denom == 0:
                continue

            t_star = -(rx0 * vx + ry0 * vy) / denom

            if 0 < t_star <= ttc_threshold:
                # Calculate closest approach distance
                dx = rx0 + vx * t_star
                dy = ry0 + vy * t_star
                d_closest = math.hypot(dx, dy)

                if d_closest <= self.collision_distance:
                    rel_speed = math.hypot(vx, vy)
                    ttc_events.append({
                        'other_id': other_id,
                        'd_closest': d_closest,
                        'rel_speed': rel_speed,
                        't_star': t_star
                    })

        return ttc_events[0] if ttc_events else None

    def process_segment_speed(self, tracker_id: int, current_pos: Tuple[float, float],
                              world_pos: Tuple[float, float], frame_idx: int,
                              entry_line: np.ndarray, exit_line: np.ndarray) -> Optional[Dict]:
        """Process segment-based speed measurement."""
        p_cur = current_pos
        Xf, Yf = world_pos

        # Initialize state if needed
        if tracker_id not in self.segment_state:
            self.segment_state[tracker_id] = {
                'p_prev': p_cur,
                'side_prev': line_side(p_cur, *entry_line)
            }

        st = self.segment_state[tracker_id]
        prev_side = st['side_prev']
        curr_side = line_side(p_cur, *entry_line)

        # Check entry crossing
        if 't0' not in st and prev_side * curr_side < 0:
            st.update(t0=frame_idx, x0m=Xf, y0m=Yf)

        # Check exit crossing
        prev_exit = line_side(st['p_prev'], *exit_line)
        curr_exit = line_side(p_cur, *exit_line)

        result = None
        if 't0' in st and 't_exit' not in st and prev_exit * curr_exit < 0:
            t1 = frame_idx
            dt = (t1 - st['t0']) / self.video_fps

            if dt > 0:
                dx = Xf - st['x0m']
                dy = Yf - st['y0m']
                v_ms = math.hypot(dx, dy) / dt

                # Apply calibration if function is available
                if self.calibration_func is not None:
                    v_ms = self.calibration_func(v_ms)

                result = {
                    'vehicle_id': tracker_id,
                    'frame_entry': st['t0'],
                    'frame_exit': t1,
                    'distance_m': round(math.hypot(dx, dy), 2),
                    'time_s': round(dt, 3),
                    'speed_m_s': round(v_ms, 2),
                    'speed_km_h': round(v_ms * 3.6, 2)
                }
                self.segment_results.append(list(result.values()))

            st['t_exit'] = t1

        # Update state for next frame
        st['p_prev'] = p_cur
        st['side_prev'] = curr_side

        return result