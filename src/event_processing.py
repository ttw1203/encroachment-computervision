"""Enhanced event processor with integrated Kalman filter TTC safeguards and performance optimizations."""
from typing import Dict, List, Tuple, Optional, Callable, Set
import math
import numpy as np
from dataclasses import dataclass


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
    Enhanced TTC processor with integrated Kalman filter safeguards and performance optimizations.
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

        # Performance optimization parameters
        self.max_ttc_distance = config.get('MAX_TTC_DISTANCE', 30.0)  # Skip distant pairs
        self.nearby_distance_threshold = config.get('NEARBY_DISTANCE_THRESHOLD', 50.0)
        self.frame_skip_interval = config.get('TTC_FRAME_SKIP_INTERVAL', 3)  # Process every N frames
        self.cleanup_batch_size = config.get('CLEANUP_BATCH_SIZE', 10)  # Batch cleanup operations

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

        # OPTIMIZATION: Use Set for boolean tracking instead of Dict
        self.ttc_active_pairs: Set[Tuple[int, int]] = set()
        self.ttc_streak_counters: Dict[Tuple[int, int], int] = {}
        self.ttc_event_history: Dict[str, int] = {}
        self.tracker_last_seen: Dict[int, int] = {}

        # Event storage with enhanced metadata
        self.validated_ttc_events: List[TTCEvent] = []
        self.id_to_class: Dict[int, str] = {}

        # OPTIMIZATION: Cache vehicle dimensions to avoid repeated lookups
        self.dimension_cache: Dict[int, Dict[str, float]] = {}

        # Performance tracking
        self.frame_skip_counter = 0
        self.cleanup_counter = 0
        self.kalman_filtered_events = 0  # Count of events prevented by Kalman safeguards
        self.total_event_attempts = 0    # Total TTC calculations attempted

        # Debug mode
        self.debug_mode = config.get('ENABLE_TTC_DEBUG', False)

    def are_vehicles_nearby(self, state_i: np.ndarray, state_j: np.ndarray,
                          max_distance: float = None) -> bool:
        """Check if two vehicles are within reasonable distance for TTC calculation."""
        if max_distance is None:
            max_distance = self.nearby_distance_threshold

        xi, yi = state_i[0], state_i[1]
        xj, yj = state_j[0], state_j[1]
        return math.hypot(xj - xi, yj - yi) < max_distance

    def process_ttc_events(self, detections, kf_states: Dict, frame_idx: int,
                           detector_tracker) -> List[TTCEvent]:
        """Main TTC processing pipeline with integrated Kalman safeguards and performance optimizations."""

        # OPTIMIZATION: Reduce TTC calculation frequency
        self.frame_skip_counter = (self.frame_skip_counter + 1) % self.frame_skip_interval
        if self.frame_skip_counter != 0:
            return []  # Skip this frame for performance

        current_events = []

        # Update tracker activity
        if len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                self.tracker_last_seen[tracker_id] = frame_idx
                class_id = int(detections.class_id[i])
                class_name = detector_tracker.get_class_name(class_id)
                self.id_to_class[tracker_id] = class_name

                # OPTIMIZATION: Cache vehicle dimensions
                if tracker_id not in self.dimension_cache:
                    self.dimension_cache[tracker_id] = self.vehicle_dimensions.get(
                        class_name, self.vehicle_dimensions.get('default', {'length': 4.0, 'width': 1.8})
                    )

        # Process all tracker pairs
        active_trackers = list(kf_states.keys())

        for i, tracker_i in enumerate(active_trackers):
            for j, tracker_j in enumerate(active_trackers[i + 1:], i + 1):
                self.total_event_attempts += 1
                pair_key = (tracker_i, tracker_j)

                # Get Kalman states for early distance check
                if tracker_i not in kf_states or tracker_j not in kf_states:
                    continue

                kf_i = kf_states[tracker_i]
                kf_j = kf_states[tracker_j]
                state_i = kf_i.statePost.flatten()
                state_j = kf_j.statePost.flatten()

                # OPTIMIZATION: Early spatial filtering - skip distant pairs
                if not self.are_vehicles_nearby(state_i, state_j):
                    continue

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

        # OPTIMIZATION: Batch cleanup operations instead of per-frame
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_batch_size:
            self._cleanup_inactive_pairs(frame_idx)
            self.cleanup_counter = 0

        return current_events

    def _apply_ttc_filters(self, pair_key: Tuple[int, int], kf_states: Dict,
                          frame_idx: int, conf_i: float, conf_j: float) -> Optional[TTCEvent]:
        """Apply the 5-stage TTC filtering pipeline with early exit optimizations."""
        tracker_i, tracker_j = pair_key

        # Get Kalman states
        kf_i = kf_states[tracker_i]
        kf_j = kf_states[tracker_j]

        # Extract state vectors (x, y, vx, vy) in real-world coordinates
        state_i = kf_i.statePost.flatten()
        state_j = kf_j.statePost.flatten()

        xi, yi, vxi, vyi = state_i
        xj, yj, vxj, vyj = state_j

        # OPTIMIZATION: Early exit for distant vehicles
        distance = math.hypot(xj - xi, yj - yi)
        if distance > self.max_ttc_distance:
            return None

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

        # FILTER 5: Vehicle Dimension Collision (AABB) - with optimized lookups
        if not self._apply_vehicle_dimension_filter_optimized(
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

    def _apply_hysteresis_filter(self, pair_key: Tuple[int, int],
                                ttc: float, distance: float) -> bool:
        """FILTER 1: Hysteresis Logic - optimized with Set operations"""
        is_currently_active = pair_key in self.ttc_active_pairs

        if not is_currently_active:
            if ttc <= self.ttc_threshold_on and distance <= self.collision_distance_on:
                self.ttc_active_pairs.add(pair_key)
                self.ttc_streak_counters[pair_key] = 1
                return True
        else:
            if ttc > self.ttc_threshold_off or distance > self.collision_distance_off:
                self.ttc_active_pairs.discard(pair_key)
                self.ttc_streak_counters.pop(pair_key, None)
                return False
            else:
                return True

        return False

    def _apply_persistence_filter(self, pair_key: Tuple[int, int], frame_idx: int) -> bool:
        """FILTER 2: Persistence Threshold"""
        if pair_key not in self.ttc_active_pairs:
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

    def _apply_vehicle_dimension_filter_optimized(self, tracker_i: int, tracker_j: int,
                                                xi: float, yi: float, vxi: float, vyi: float,
                                                xj: float, yj: float, vxj: float, vyj: float,
                                                t_star: float) -> bool:
        """FILTER 5: Vehicle Dimension Collision (AABB Overlap) - optimized with cached lookups"""

        # OPTIMIZATION: Use cached dimensions
        dims_i = self.dimension_cache.get(tracker_i)
        dims_j = self.dimension_cache.get(tracker_j)

        # Fallback if not cached
        if dims_i is None:
            class_i = self.id_to_class.get(tracker_i, 'car')
            dims_i = self.vehicle_dimensions.get(class_i, self.vehicle_dimensions.get('default', {'length': 4.0, 'width': 1.8}))
            self.dimension_cache[tracker_i] = dims_i

        if dims_j is None:
            class_j = self.id_to_class.get(tracker_j, 'car')
            dims_j = self.vehicle_dimensions.get(class_j, self.vehicle_dimensions.get('default', {'length': 4.0, 'width': 1.8}))
            self.dimension_cache[tracker_j] = dims_j

        center_i_future = (xi + vxi * t_star, yi + vyi * t_star)
        center_j_future = (xj + vxj * t_star, yj + vyj * t_star)

        # OPTIMIZATION: Simplified AABB overlap with early exit
        return self._aabb_overlap_optimized(center_i_future, dims_i, center_j_future, dims_j)

    def _aabb_overlap_optimized(self, center_i: Tuple[float, float], dims_i: Dict[str, float],
                              center_j: Tuple[float, float], dims_j: Dict[str, float]) -> bool:
        """Optimized AABB overlap check with early exit conditions."""
        cx_i, cy_i = center_i
        cx_j, cy_j = center_j

        half_width_i = dims_i['width'] / 2.0
        half_length_i = dims_i['length'] / 2.0
        half_width_j = dims_j['width'] / 2.0
        half_length_j = dims_j['length'] / 2.0

        # Early exit checks - if centers are too far apart, no collision possible
        dx = abs(cx_j - cx_i)
        dy = abs(cy_j - cy_i)

        if dx > (half_width_i + half_width_j) or dy > (half_length_i + half_length_j):
            return False

        return True

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
        """Cleanup inactive tracker pairs to prevent memory leaks - batched operations."""
        pairs_to_remove = []

        # Check active pairs
        for pair_key in list(self.ttc_active_pairs):
            tracker_i, tracker_j = pair_key

            last_seen_i = self.tracker_last_seen.get(tracker_i, 0)
            last_seen_j = self.tracker_last_seen.get(tracker_j, 0)

            if (current_frame - last_seen_i > self.cleanup_timeout_frames or
                current_frame - last_seen_j > self.cleanup_timeout_frames):
                pairs_to_remove.append(pair_key)

        # Batch removal operations
        for pair_key in pairs_to_remove:
            self.ttc_active_pairs.discard(pair_key)
            self.ttc_streak_counters.pop(pair_key, None)

        # Cleanup old events
        old_events = [event_id for event_id, last_frame in self.ttc_event_history.items()
                     if current_frame - last_frame > self.cleanup_timeout_frames]
        for event_id in old_events:
            self.ttc_event_history.pop(event_id, None)

        # Cleanup dimension cache for removed trackers
        active_tracker_ids = set(self.tracker_last_seen.keys())
        cache_keys_to_remove = [tid for tid in self.dimension_cache.keys()
                               if tid not in active_tracker_ids]
        for tid in cache_keys_to_remove:
            self.dimension_cache.pop(tid, None)

    def get_debug_info(self) -> Dict:
        """Get debug information including Kalman safeguard statistics and performance metrics."""
        if not self.debug_mode:
            return {}

        return {
            'active_pairs': len(self.ttc_active_pairs),
            'persistent_pairs': sum(1 for count in self.ttc_streak_counters.values()
                                  if count >= self.ttc_persistence_frames),
            'total_events': len(self.validated_ttc_events),
            'active_trackers': len(self.tracker_last_seen),
            'kalman_filtered_events': self.kalman_filtered_events,
            'total_event_attempts': self.total_event_attempts,
            'kalman_filter_effectiveness': (self.kalman_filtered_events / max(self.total_event_attempts, 1)) * 100,
            'dimension_cache_size': len(self.dimension_cache),
            'frame_skip_interval': self.frame_skip_interval,
            'max_ttc_distance': self.max_ttc_distance
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
    """Enhanced EventProcessor with integrated Kalman safeguards and performance optimizations."""

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
        """Get default TTC configuration for urban traffic with performance optimizations."""
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
            # Performance optimization parameters
            'MAX_TTC_DISTANCE': 30.0,
            'NEARBY_DISTANCE_THRESHOLD': 50.0,
            'TTC_FRAME_SKIP_INTERVAL': 3,
            'CLEANUP_BATCH_SIZE': 10,
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
        # [Keep existing implementation]
        pass