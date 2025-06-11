"""Enhanced event processor with integrated Kalman filter TTC safeguards."""
from typing import Dict, List, Tuple, Optional, Callable
import math
import numpy as np
from dataclasses import dataclass
from src.geometry_and_transforms import line_side


@dataclass
class TTCEvent:
    """Represents a validated TTC event with metadata."""
    frame_idx: int
    follower_id: int
    leader_id: int
    ttc_seconds: float
    closest_distance: float
    relative_speed: float
    confidence_score: float
    relative_angle: float
    event_id: str
    kalman_eligible: bool  # NEW: Track if both vehicles passed Kalman safeguards


class EnhancedTTCProcessor:
    """
    Enhanced TTC processor with integrated Kalman filter safeguards.
    Works with KalmanFilterManager to prevent false events from initialization.
    """

    def __init__(self, config, kalman_manager=None):
        """Initialize enhanced TTC processor with Kalman integration."""
        self.config = config
        self.kalman_manager = kalman_manager  # Reference to KalmanFilterManager
        self.video_fps = config.get('video_fps', 30.0)

        # TTC Configuration
        self.ttc_threshold_on = config.get('TTC_THRESHOLD_ON', 1.5)
        self.ttc_threshold_off = config.get('TTC_THRESHOLD_OFF', 2.5)
        self.collision_distance_on = config.get('COLLISION_DISTANCE_ON', 1.5)
        self.collision_distance_off = config.get('COLLISION_DISTANCE_OFF', 2.5)
        self.ttc_persistence_frames = config.get('TTC_PERSISTENCE_FRAMES', 3)
        self.min_confidence_for_ttc = config.get('MIN_CONFIDENCE_FOR_TTC', 0.4)
        self.ttc_min_relative_angle = config.get('TTC_MIN_RELATIVE_ANGLE', 10)
        self.ttc_max_relative_angle = config.get('TTC_MAX_RELATIVE_ANGLE', 150)
        self.cleanup_timeout_frames = config.get('TTC_CLEANUP_TIMEOUT_FRAMES', 90)

        # Vehicle dimensions for AABB collision detection
        self.vehicle_dimensions = config.get('VEHICLE_DIMENSIONS', {
            'car': {'length': 4.5, 'width': 1.8},
            'truck': {'length': 8.0, 'width': 2.5},
            'bus': {'length': 12.0, 'width': 2.5},
            'motorcycle': {'length': 2.0, 'width': 0.8},
            'bicycle': {'length': 1.8, 'width': 0.6},
            'person': {'length': 0.6, 'width': 0.4},
            'rickshaw': {'length': 3.0, 'width': 1.2},
            'default': {'length': 4.0, 'width': 1.8}
        })

        # State tracking dictionaries
        self.ttc_active_states: Dict[Tuple[int, int], bool] = {}
        self.ttc_streak_counters: Dict[Tuple[int, int], int] = {}
        self.ttc_event_history: Dict[str, int] = {}
        self.tracker_last_seen: Dict[int, int] = {}

        # Event storage with enhanced metadata
        self.validated_ttc_events: List[TTCEvent] = []
        self.id_to_class: Dict[int, str] = {}

        # Statistics tracking
        self.kalman_filtered_events = 0  # Count of events prevented by Kalman safeguards
        self.total_event_attempts = 0    # Total TTC calculations attempted

        # Debug mode
        self.debug_mode = config.get('ENABLE_TTC_DEBUG', False)

    def process_ttc_events(self, detections, kf_states: Dict, frame_idx: int,
                           detector_tracker) -> List[TTCEvent]:
        """Main TTC processing pipeline with integrated Kalman safeguards."""
        current_events = []

        # Update tracker activity
        if len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                self.tracker_last_seen[tracker_id] = frame_idx
                class_id = int(detections.class_id[i])
                self.id_to_class[tracker_id] = detector_tracker.get_class_name(class_id)

        # Process all tracker pairs
        active_trackers = list(kf_states.keys())

        for i, tracker_i in enumerate(active_trackers):
            for j, tracker_j in enumerate(active_trackers[i + 1:], i + 1):
                self.total_event_attempts += 1
                pair_key = (tracker_i, tracker_j)

                # INTEGRATED SAFEGUARD: Check Kalman eligibility FIRST (with null check)
                kalman_eligible_i = False
                kalman_eligible_j = False

                if self.kalman_manager is not None:
                    kalman_eligible_i = self.kalman_manager.is_ttc_eligible(tracker_i, frame_idx)
                    kalman_eligible_j = self.kalman_manager.is_ttc_eligible(tracker_j, frame_idx)
                else:
                    # Fallback: allow all trackers if no Kalman manager
                    kalman_eligible_i = True
                    kalman_eligible_j = True

                # Skip TTC calculation if either tracker fails Kalman safeguards
                if not (kalman_eligible_i and kalman_eligible_j):
                    self.kalman_filtered_events += 1

                    if self.debug_mode and self.kalman_manager is not None:
                        reasons = []
                        if not kalman_eligible_i:
                            status_i = self.kalman_manager.get_ttc_eligibility_status(tracker_i, frame_idx)
                            reasons.append(f"Tracker {tracker_i}: {status_i.get('reason', 'unknown')}")
                        if not kalman_eligible_j:
                            status_j = self.kalman_manager.get_ttc_eligibility_status(tracker_j, frame_idx)
                            reasons.append(f"Tracker {tracker_j}: {status_j.get('reason', 'unknown')}")

                        print(f"[TTC SAFEGUARD] Pair ({tracker_i}, {tracker_j}) filtered: {'; '.join(reasons)}")

                    continue

                # Get detection indices and confidences
                det_i_conf, det_j_conf = self._get_detection_confidences(
                    detections, tracker_i, tracker_j)

                # Apply remaining 5-stage filtering pipeline
                event = self._apply_ttc_filters(
                    pair_key, kf_states, frame_idx, det_i_conf, det_j_conf)

                if event:
                    # Mark event as Kalman-eligible
                    event.kalman_eligible = True
                    current_events.append(event)
                    self.validated_ttc_events.append(event)

        # Cleanup inactive tracker pairs
        self._cleanup_inactive_pairs(frame_idx)

        return current_events

    def _apply_ttc_filters(self, pair_key: Tuple[int, int], kf_states: Dict,
                          frame_idx: int, conf_i: float, conf_j: float) -> Optional[TTCEvent]:
        """Apply the 5-stage TTC filtering pipeline (Kalman check already passed)."""
        tracker_i, tracker_j = pair_key

        # Get Kalman states
        if tracker_i not in kf_states or tracker_j not in kf_states:
            return None

        kf_i = kf_states[tracker_i]
        kf_j = kf_states[tracker_j]

        # Extract state vectors (x, y, vx, vy) in real-world coordinates
        state_i = kf_i.statePost.flatten()
        state_j = kf_j.statePost.flatten()

        xi, yi, vxi, vyi = state_i
        xj, yj, vxj, vyj = state_j

        # Calculate basic TTC parameters
        rx0, ry0 = xj - xi, yj - yi
        vx, vy = vxj - vxi, vyj - vyi

        denom = vx * vx + vy * vy
        if denom == 0:
            return None

        t_star = -(rx0 * vx + ry0 * vy) / denom

        if t_star <= 0:
            return None

        # Calculate closest approach distance
        dx = rx0 + vx * t_star
        dy = ry0 + vy * t_star
        d_closest = math.hypot(dx, dy)

        # FILTER 1: Hysteresis Logic
        if not self._apply_hysteresis_filter(pair_key, t_star, d_closest):
            return None

        # FILTER 2: Persistence Threshold
        if not self._apply_persistence_filter(pair_key, frame_idx):
            return None

        # FILTER 3: Confidence-Based Filtering
        if not self._apply_confidence_filter(conf_i, conf_j):
            return None

        # FILTER 4: Relative Angle Filtering
        rel_angle = self._apply_relative_angle_filter(vxi, vyi, vxj, vyj, d_closest)
        if rel_angle is None:
            return None

        # FILTER 5: Vehicle Dimension Collision (AABB)
        if not self._apply_vehicle_dimension_filter(
            tracker_i, tracker_j, xi, yi, vxi, vyi, xj, yj, vxj, vyj, t_star):
            return None

        # All filters passed - create validated TTC event
        rel_speed = math.hypot(vx, vy)
        confidence_score = (conf_i + conf_j) / 2.0
        event_id = f"{frame_idx}_{min(tracker_i, tracker_j)}_{max(tracker_i, tracker_j)}"

        return TTCEvent(
            frame_idx=frame_idx,
            follower_id=tracker_i,
            leader_id=tracker_j,
            ttc_seconds=t_star,
            closest_distance=d_closest,
            relative_speed=rel_speed,
            confidence_score=confidence_score,
            relative_angle=rel_angle,
            event_id=event_id,
            kalman_eligible=True  # Always true if we reach this point
        )

    # [Include all the existing filter methods from the previous implementation]
    def _apply_hysteresis_filter(self, pair_key: Tuple[int, int],
                                ttc: float, distance: float) -> bool:
        """FILTER 1: Hysteresis Logic"""
        current_active = self.ttc_active_states.get(pair_key, False)

        if not current_active:
            if ttc <= self.ttc_threshold_on and distance <= self.collision_distance_on:
                self.ttc_active_states[pair_key] = True
                self.ttc_streak_counters[pair_key] = 1
                return True
        else:
            if ttc > self.ttc_threshold_off or distance > self.collision_distance_off:
                self.ttc_active_states[pair_key] = False
                self.ttc_streak_counters[pair_key] = 0
                return False
            else:
                return True

        return False

    def _apply_persistence_filter(self, pair_key: Tuple[int, int], frame_idx: int) -> bool:
        """FILTER 2: Persistence Threshold"""
        if not self.ttc_active_states.get(pair_key, False):
            return False

        current_streak = self.ttc_streak_counters.get(pair_key, 0)

        if current_streak >= self.ttc_persistence_frames:
            return True
        else:
            self.ttc_streak_counters[pair_key] = current_streak + 1
            return False

    def _apply_confidence_filter(self, conf_i: float, conf_j: float) -> bool:
        """FILTER 3: Confidence-Based Filtering"""
        return (conf_i >= self.min_confidence_for_ttc and
                conf_j >= self.min_confidence_for_ttc)

    def _apply_relative_angle_filter(self, vxi: float, vyi: float,
                                   vxj: float, vyj: float, d_closest: float) -> Optional[float]:
        """FILTER 4: Relative Angle Filtering"""
        if d_closest < 0.5:
            return 0.0

        speed_i = math.hypot(vxi, vyi)
        speed_j = math.hypot(vxj, vyj)

        if speed_i < 0.1 or speed_j < 0.1:
            return None

        vxi_norm, vyi_norm = vxi / speed_i, vyi / speed_i
        vxj_norm, vyj_norm = vxj / speed_j, vyj / speed_j

        dot_product = vxi_norm * vxj_norm + vyi_norm * vyj_norm
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = math.acos(abs(dot_product))
        angle_deg = math.degrees(angle_rad)

        if (angle_deg >= self.ttc_min_relative_angle and
            angle_deg <= self.ttc_max_relative_angle):
            return angle_deg
        else:
            return None

    def _apply_vehicle_dimension_filter(self, tracker_i: int, tracker_j: int,
                                      xi: float, yi: float, vxi: float, vyi: float,
                                      xj: float, yj: float, vxj: float, vyj: float,
                                      t_star: float) -> bool:
        """FILTER 5: Vehicle Dimension Collision (AABB Overlap)"""
        class_i = self.id_to_class.get(tracker_i, 'car')
        class_j = self.id_to_class.get(tracker_j, 'car')

        default_dims = {'length': 4.0, 'width': 1.8}
        dims_i = self.vehicle_dimensions.get(class_i,
                                           self.vehicle_dimensions.get('default', default_dims))
        dims_j = self.vehicle_dimensions.get(class_j,
                                           self.vehicle_dimensions.get('default', default_dims))

        center_i_future = (xi + vxi * t_star, yi + vyi * t_star)
        center_j_future = (xj + vxj * t_star, yj + vyj * t_star)

        box_i = self._construct_aabb(center_i_future, dims_i)
        box_j = self._construct_aabb(center_j_future, dims_j)

        return self._aabb_overlap(box_i, box_j)

    def _construct_aabb(self, center: Tuple[float, float],
                       dimensions: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Construct axis-aligned bounding box."""
        cx, cy = center
        half_width = dimensions['width'] / 2.0
        half_length = dimensions['length'] / 2.0
        return (cx - half_width, cx + half_width, cy - half_length, cy + half_length)

    def _aabb_overlap(self, box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> bool:
        """Check if two axis-aligned bounding boxes overlap."""
        x1_min, x1_max, y1_min, y1_max = box1
        x2_min, x2_max, y2_min, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or
                   y1_max < y2_min or y1_min > y2_max)

    def _get_detection_confidences(self, detections, tracker_i: int,
                                 tracker_j: int) -> Tuple[float, float]:
        """Get detection confidences for two trackers."""
        conf_i = conf_j = 0.0

        if len(detections) > 0:
            for idx, tid in enumerate(detections.tracker_id):
                if tid == tracker_i:
                    conf_i = float(detections.confidence[idx])
                elif tid == tracker_j:
                    conf_j = float(detections.confidence[idx])

        return conf_i, conf_j

    def _cleanup_inactive_pairs(self, current_frame: int) -> None:
        """Cleanup inactive tracker pairs to prevent memory leaks."""
        pairs_to_remove = []

        for pair_key in list(self.ttc_active_states.keys()):
            tracker_i, tracker_j = pair_key

            last_seen_i = self.tracker_last_seen.get(tracker_i, 0)
            last_seen_j = self.tracker_last_seen.get(tracker_j, 0)

            if (current_frame - last_seen_i > self.cleanup_timeout_frames or
                current_frame - last_seen_j > self.cleanup_timeout_frames):
                pairs_to_remove.append(pair_key)

        for pair_key in pairs_to_remove:
            self.ttc_active_states.pop(pair_key, None)
            self.ttc_streak_counters.pop(pair_key, None)

        old_events = [event_id for event_id, last_frame in self.ttc_event_history.items()
                     if current_frame - last_frame > self.cleanup_timeout_frames]
        for event_id in old_events:
            self.ttc_event_history.pop(event_id, None)

    def get_debug_info(self) -> Dict:
        """Get debug information including Kalman safeguard statistics."""
        if not self.debug_mode:
            return {}

        return {
            'active_pairs': len(self.ttc_active_states),
            'persistent_pairs': sum(1 for count in self.ttc_streak_counters.values()
                                  if count >= self.ttc_persistence_frames),
            'total_events': len(self.validated_ttc_events),
            'active_trackers': len(self.tracker_last_seen),
            'kalman_filtered_events': self.kalman_filtered_events,
            'total_event_attempts': self.total_event_attempts,
            'kalman_filter_effectiveness': (self.kalman_filtered_events / max(self.total_event_attempts, 1)) * 100
        }

    def export_events_for_csv(self) -> List[List]:
        """Export validated TTC events in CSV format with Kalman eligibility."""
        csv_rows = []

        for event in self.validated_ttc_events:
            follower_class = self.id_to_class.get(event.follower_id, "unknown")
            leader_class = self.id_to_class.get(event.leader_id, "unknown")

            csv_rows.append([
                event.frame_idx,
                event.follower_id,
                follower_class,
                event.leader_id,
                leader_class,
                round(event.closest_distance, 2),
                round(event.relative_speed, 2),
                round(event.ttc_seconds, 2),
                round(event.confidence_score, 3),
                round(event.relative_angle, 1),
                event.kalman_eligible  # NEW: Include safeguard status
            ])

        return csv_rows


# Updated EventProcessor for backward compatibility
class EventProcessor:
    """Enhanced EventProcessor with integrated Kalman safeguards."""

    def __init__(self, video_fps: float, collision_distance: float = 2.0,
                 calibration_func: Optional[Callable[[float], float]] = None,
                 ttc_config: Optional[Dict] = None, kalman_manager=None):
        """Initialize with enhanced TTC processing and Kalman integration."""
        self.video_fps = video_fps
        self.collision_distance = collision_distance
        self.calibration_func = calibration_func

        # Legacy attributes
        self.segment_state: Dict = {}
        self.segment_results: List = []
        self.ttc_rows: List = []
        self.id_to_class: Dict[int, str] = {}

        # Enhanced TTC processor with Kalman integration
        config = ttc_config or self._get_default_ttc_config()
        config['video_fps'] = video_fps
        self.ttc_processor = EnhancedTTCProcessor(config, kalman_manager)

    def _get_default_ttc_config(self) -> Dict:
        """Get default TTC configuration for urban traffic."""
        return {
            'TTC_THRESHOLD_ON': 1.5,
            'TTC_THRESHOLD_OFF': 2.5,
            'COLLISION_DISTANCE_ON': 1.5,
            'COLLISION_DISTANCE_OFF': 2.5,
            'TTC_PERSISTENCE_FRAMES': 3,
            'MIN_CONFIDENCE_FOR_TTC': 0.4,
            'TTC_MIN_RELATIVE_ANGLE': 10,
            'TTC_MAX_RELATIVE_ANGLE': 150,
            'TTC_CLEANUP_TIMEOUT_FRAMES': 90,
            'ENABLE_TTC_DEBUG': False,
            'VEHICLE_DIMENSIONS': {
                'car': {'length': 4.5, 'width': 1.8},
                'truck': {'length': 8.0, 'width': 2.5},
                'bus': {'length': 12.0, 'width': 2.5},
                'motorcycle': {'length': 2.0, 'width': 0.8},
                'bicycle': {'length': 1.8, 'width': 0.6},
                'person': {'length': 0.6, 'width': 0.4},
                'rickshaw': {'length': 3.0, 'width': 1.2},
                'default': {'length': 4.0, 'width': 1.8}
            }
        }

    def calculate_ttc(self, tracker_id: int, kf_states: Dict,
                      last_seen_frame: Dict, current_frame: int,
                      max_age_frames: int, ttc_threshold: float,
                      detections=None, detector_tracker=None) -> Optional[Dict]:
        """Enhanced TTC calculation with integrated Kalman safeguards."""
        if detections is not None and detector_tracker is not None:
            # Use enhanced processing with Kalman integration
            events = self.ttc_processor.process_ttc_events(
                detections, kf_states, current_frame, detector_tracker)

            # Find event for this tracker_id
            for event in events:
                if event.follower_id == tracker_id:
                    return {
                        'other_id': event.leader_id,
                        'd_closest': event.closest_distance,
                        'rel_speed': event.relative_speed,
                        't_star': event.ttc_seconds
                    }

        # Fallback to legacy calculation if enhanced data not available
        return self._legacy_calculate_ttc(
            tracker_id, kf_states, last_seen_frame, current_frame,
            max_age_frames, ttc_threshold)

    def _legacy_calculate_ttc(self, tracker_id: int, kf_states: Dict,
                             last_seen_frame: Dict, current_frame: int,
                             max_age_frames: int, ttc_threshold: float) -> Optional[Dict]:
        """Legacy TTC calculation for backward compatibility."""
        if tracker_id not in kf_states:
            return None

        kf = kf_states[tracker_id]
        Xi, Yi, Vxi, Vyi = kf.statePost.flatten()

        for other_id, other_kf in kf_states.items():
            if other_id == tracker_id:
                continue

            if current_frame - last_seen_frame.get(other_id, 0) > max_age_frames:
                continue

            Xj, Yj, Vxj, Vyj = other_kf.statePost.flatten()
            rx0, ry0 = Xj - Xi, Yj - Yi
            vx, vy = Vxj - Vxi, Vyj - Vyi

            denom = vx * vx + vy * vy
            if denom == 0:
                continue

            t_star = -(rx0 * vx + ry0 * vy) / denom

            if 0 < t_star <= ttc_threshold:
                dx = rx0 + vx * t_star
                dy = ry0 + vy * t_star
                d_closest = math.hypot(dx, dy)

                if d_closest <= self.collision_distance:
                    rel_speed = math.hypot(vx, vy)
                    return {
                        'other_id': other_id,
                        'd_closest': d_closest,
                        'rel_speed': rel_speed,
                        't_star': t_star
                    }

        return None

    def process_segment_speed(self, tracker_id: int, current_pos: Tuple[float, float],
                              world_pos: Tuple[float, float], frame_idx: int,
                              entry_line: np.ndarray, exit_line: np.ndarray) -> Optional[Dict]:
        """Process segment-based speed measurement (unchanged)."""
        p_cur = current_pos
        Xf, Yf = world_pos

        if tracker_id not in self.segment_state:
            self.segment_state[tracker_id] = {
                'p_prev': p_cur,
                'side_prev': line_side(p_cur, *entry_line)
            }

        st = self.segment_state[tracker_id]
        prev_side = st['side_prev']
        curr_side = line_side(p_cur, *entry_line)

        if 't0' not in st and prev_side * curr_side < 0:
            st.update(t0=frame_idx, x0m=Xf, y0m=Yf)

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

        st['p_prev'] = p_cur
        st['side_prev'] = curr_side

        return result