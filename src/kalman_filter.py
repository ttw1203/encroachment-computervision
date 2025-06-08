"""Enhanced Kalman filter with configurable initialization uncertainty and TTC safeguards."""
from typing import Dict, List, Tuple, Optional, Callable
import cv2
import numpy as np
from copy import deepcopy
import math
from collections import deque
import logging


class KalmanFilterManager:
    """Manages Kalman filters for multiple tracked objects with enhanced stability and TTC safeguards."""

    def __init__(self, speed_smoothing_window: int = 5, max_acceleration: float = 5.0,
                 min_speed_threshold: float = 0.1, initial_velocity_frames: int = 2,
                 initial_velocity_uncertainty: float = 5.0, position_uncertainty: float = 0.6,
                 ttc_burn_in_frames: int = 10, ttc_min_velocity: float = 0.3,
                 ttc_min_confidence: float = 0.75):
        """Initialize the Kalman filter manager with stability parameters and TTC safeguards.

        Args:
            speed_smoothing_window: Number of frames to average for speed smoothing
            max_acceleration: Maximum allowed acceleration in m/s²
            min_speed_threshold: Minimum speed threshold in m/s below which velocity is set to 0
            initial_velocity_frames: Number of frames to use for initial velocity calculation (2 or 3)
            initial_velocity_uncertainty: Initial velocity uncertainty in m/s for new tracks
            position_uncertainty: Initial position uncertainty in meters for new tracks
            ttc_burn_in_frames: Number of frames to skip TTC calculation for new tracks
            ttc_min_velocity: Minimum velocity magnitude for TTC calculation
            ttc_min_confidence: Minimum track confidence for TTC calculation
        """
        # Existing attributes
        self.kf_states: Dict[int, cv2.KalmanFilter] = {}
        self.speed_history: Dict[int, deque] = {}
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        self.speed_smoothing_window = speed_smoothing_window
        self.max_acceleration = max_acceleration
        self.min_speed_threshold = min_speed_threshold
        self.direction_history: Dict[int, deque] = {}

        # Initial velocity calculation
        self.initial_positions: Dict[int, List[Tuple[float, float, int]]] = {}
        self.initial_velocity_frames = initial_velocity_frames
        self.initial_velocity_applied: Dict[int, bool] = {}

        # NEW: Enhanced initialization parameters
        self.initial_velocity_uncertainty = initial_velocity_uncertainty  # High uncertainty for new tracks
        self.position_uncertainty = position_uncertainty  # Position uncertainty

        # NEW: TTC safeguard parameters
        self.ttc_burn_in_frames = ttc_burn_in_frames  # Skip TTC for new tracks
        self.ttc_min_velocity = ttc_min_velocity  # Minimum velocity for TTC
        self.ttc_min_confidence = ttc_min_confidence  # Minimum confidence for TTC

        # NEW: Track metadata for TTC safeguards
        self.track_creation_frame: Dict[int, int] = {}  # Track when each tracker was created
        self.track_confidence: Dict[int, float] = {}  # Track confidence scores
        self.track_detection_count: Dict[int, int] = {}  # Number of detections per track

    def create_kalman_filter(self, dt: float, tracker_id: int) -> cv2.KalmanFilter:
        """Create a new Kalman filter with enhanced initialization uncertainty."""
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

        # Enhanced process noise with configurable initial uncertainty
        pos_noise = self.position_uncertainty ** 2
        vel_noise = self.initial_velocity_uncertainty ** 2

        kf.processNoiseCov = np.array([
            [pos_noise, 0,        pos_noise, 0],
            [0,         pos_noise, 0,        pos_noise],
            [pos_noise, 0,        vel_noise, 0],
            [0,         pos_noise, 0,        vel_noise]
        ], np.float32)

        # Measurement noise (trust detections moderately)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0

        # CRITICAL: Set high initial covariance for velocity uncertainty
        kf.errorCovPost = np.array([
            [pos_noise, 0,        0,         0],
            [0,         pos_noise, 0,         0],
            [0,         0,        vel_noise, 0],
            [0,         0,        0,         vel_noise]
        ], np.float32)

        return kf

    def update_or_create(self, tracker_id: int, x: float, y: float, dt: float,
                        frame_idx: int, confidence: float = 0.5) -> cv2.KalmanFilter:
        """Update existing Kalman filter or create new one with enhanced initialization."""

        # Track initial positions for new trackers
        if tracker_id not in self.initial_positions:
            self.initial_positions[tracker_id] = []

        # Store position if we haven't applied initial velocity yet
        if tracker_id not in self.initial_velocity_applied:
            self.initial_positions[tracker_id].append((x, y, frame_idx))

        if tracker_id not in self.kf_states:
            # Initialize new tracker with enhanced uncertainty
            kf = self.create_kalman_filter(dt, tracker_id)
            kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
            self.kf_states[tracker_id] = kf
            self.speed_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.direction_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.last_positions[tracker_id] = (x, y)
            self.initial_velocity_applied[tracker_id] = False

            # NEW: Initialize track metadata
            self.track_creation_frame[tracker_id] = frame_idx
            self.track_confidence[tracker_id] = confidence
            self.track_detection_count[tracker_id] = 1

        else:
            kf = self.kf_states[tracker_id]

            # Update track metadata
            self.track_confidence[tracker_id] = confidence
            self.track_detection_count[tracker_id] += 1

            # Apply initial velocity if we have enough frames and haven't done so yet
            if not self.initial_velocity_applied[tracker_id]:
                if len(self.initial_positions[tracker_id]) >= self.initial_velocity_frames:
                    self._apply_initial_velocity(tracker_id, dt)
                    self.initial_velocity_applied[tracker_id] = True
                    # Clean up stored positions
                    del self.initial_positions[tracker_id]

            # Continue with existing update logic
            kf.transitionMatrix[0, 2] = dt
            kf.transitionMatrix[1, 3] = dt
            prediction = kf.predict()
            measurement = np.array([[x], [y]], np.float32)
            kf.correct(measurement)
            self._apply_velocity_constraints(tracker_id, dt)
            self.last_positions[tracker_id] = (x, y)

        return kf

    def is_ttc_eligible(self, tracker_id: int, current_frame: int) -> bool:
        """Check if tracker is eligible for TTC calculation based on safeguards."""
        if tracker_id not in self.track_creation_frame:
            return False

        # Safeguard 1: Burn-in period check
        frames_since_creation = current_frame - self.track_creation_frame[tracker_id]
        if frames_since_creation < self.ttc_burn_in_frames:
            return False

        # Safeguard 2: Track confidence check
        if self.track_confidence.get(tracker_id, 0.0) < self.ttc_min_confidence:
            return False

        # Safeguard 3: Minimum detections check (track maturity)
        if self.track_detection_count.get(tracker_id, 0) < 5:
            return False

        # Safeguard 4: Velocity magnitude check
        if tracker_id in self.kf_states:
            state = self.kf_states[tracker_id].statePost.flatten()
            _, _, vx, vy = state
            velocity_magnitude = math.hypot(vx, vy)
            if velocity_magnitude < self.ttc_min_velocity:
                return False

        return True

    def get_ttc_eligibility_status(self, tracker_id: int, current_frame: int) -> Dict[str, any]:
        """Get detailed TTC eligibility status for debugging."""
        if tracker_id not in self.track_creation_frame:
            return {"eligible": False, "reason": "tracker_not_found"}

        frames_since_creation = current_frame - self.track_creation_frame[tracker_id]
        confidence = self.track_confidence.get(tracker_id, 0.0)
        detection_count = self.track_detection_count.get(tracker_id, 0)

        velocity_magnitude = 0.0
        if tracker_id in self.kf_states:
            state = self.kf_states[tracker_id].statePost.flatten()
            _, _, vx, vy = state
            velocity_magnitude = math.hypot(vx, vy)

        status = {
            "eligible": self.is_ttc_eligible(tracker_id, current_frame),
            "frames_since_creation": frames_since_creation,
            "burn_in_required": self.ttc_burn_in_frames,
            "confidence": confidence,
            "min_confidence_required": self.ttc_min_confidence,
            "detection_count": detection_count,
            "velocity_magnitude": velocity_magnitude,
            "min_velocity_required": self.ttc_min_velocity
        }

        # Determine specific reason if not eligible
        if not status["eligible"]:
            if frames_since_creation < self.ttc_burn_in_frames:
                status["reason"] = "burn_in_period"
            elif confidence < self.ttc_min_confidence:
                status["reason"] = "low_confidence"
            elif detection_count < 5:
                status["reason"] = "immature_track"
            elif velocity_magnitude < self.ttc_min_velocity:
                status["reason"] = "low_velocity"
            else:
                status["reason"] = "unknown"

        return status

    def _apply_initial_velocity(self, tracker_id: int, dt: float) -> None:
        """Calculate and apply initial velocity with improved uncertainty handling."""
        positions = self.initial_positions[tracker_id]

        if len(positions) < 2:
            return

        # Calculate velocity based on configuration
        if self.initial_velocity_frames == 2 or len(positions) == 2:
            x1, y1, frame1 = positions[0]
            x2, y2, frame2 = positions[1]
            time_diff = (frame2 - frame1) * dt
        else:  # Use 3 frames
            if len(positions) >= 3:
                x1, y1, frame1 = positions[1]
                x2, y2, frame2 = positions[2]
                time_diff = (frame2 - frame1) * dt
            else:
                x1, y1, frame1 = positions[0]
                x2, y2, frame2 = positions[1]
                time_diff = (frame2 - frame1) * dt

        # Validate time difference
        if time_diff <= 0.001:
            return

        # Check for extreme position jumps
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance > 100.0:
            logging.warning(
                f"Tracker {tracker_id}: Extreme position jump detected ({distance:.1f}m), skipping initial velocity")
            return

        vx_init = (x2 - x1) / time_diff
        vy_init = (y2 - y1) / time_diff

        # Validate calculated velocities
        if math.isnan(vx_init) or math.isnan(vy_init) or math.isinf(vx_init) or math.isinf(vy_init):
            logging.warning(f"Tracker {tracker_id}: Invalid initial velocity calculated, skipping")
            return

        speed_init = math.hypot(vx_init, vy_init)

        # Apply constraints before setting
        if speed_init < self.min_speed_threshold:
            return

        # Check for unrealistic speeds
        max_initial_speed = 50.0  # m/s (180 km/h)
        if speed_init > max_initial_speed:
            scale = max_initial_speed / speed_init
            vx_init *= scale
            vy_init *= scale
            speed_init = max_initial_speed

        # Apply to Kalman filter
        kf = self.kf_states[tracker_id]
        kf.statePost[2, 0] = vx_init
        kf.statePost[3, 0] = vy_init

        # Update error covariance to reflect improved velocity knowledge
        # Reduce velocity uncertainty after initial velocity is calculated
        improved_vel_uncertainty = self.initial_velocity_uncertainty * 0.3  # 30% of original
        kf.errorCovPost[2, 2] = improved_vel_uncertainty ** 2
        kf.errorCovPost[3, 3] = improved_vel_uncertainty ** 2

        # Initialize speed history
        self.speed_history[tracker_id].append(speed_init)

        # Initialize direction history
        if abs(vx_init) > 0.01 or abs(vy_init) > 0.01:
            initial_direction = math.atan2(vy_init, vx_init)
            self.direction_history[tracker_id].append(initial_direction)

    # [Rest of the existing methods remain unchanged: _apply_velocity_constraints,
    # _stabilize_direction, apply_speed_calibration, _calculate_ema_speed,
    # get_smoothed_velocity, predict_future_positions, get_state, remove_tracker, get_all_states]

    def _apply_velocity_constraints(self, tracker_id: int, dt: float):
        """Apply physical constraints to velocity to prevent unrealistic values."""
        kf = self.kf_states[tracker_id]
        state = kf.statePost.flatten()
        x, y, vx, vy = state

        # Calculate current speed
        current_speed = math.hypot(vx, vy)

        # Store in history for smoothing
        self.speed_history[tracker_id].append(current_speed)

        # Apply minimum speed threshold
        if current_speed < self.min_speed_threshold:
            kf.statePost[2, 0] = 0.0
            kf.statePost[3, 0] = 0.0
            return

        # Check for maximum acceleration constraint
        if len(self.speed_history[tracker_id]) > 1:
            prev_speed = self.speed_history[tracker_id][-2]
            acceleration = (current_speed - prev_speed) / dt

            if abs(acceleration) > self.max_acceleration:
                # Limit the speed change
                max_speed_change = self.max_acceleration * dt
                if acceleration > 0:
                    new_speed = prev_speed + max_speed_change
                else:
                    new_speed = prev_speed - max_speed_change

                # Scale velocity components
                if current_speed > 0:
                    scale = new_speed / current_speed
                    kf.statePost[2, 0] = vx * scale
                    kf.statePost[3, 0] = vy * scale

        # Apply direction stabilization
        self._stabilize_direction(tracker_id)

    def _stabilize_direction(self, tracker_id: int):
        """Stabilize direction to prevent sudden reversals."""
        kf = self.kf_states[tracker_id]
        state = kf.statePost.flatten()
        x, y, vx, vy = state

        if abs(vx) > 0.01 or abs(vy) > 0.01:
            current_direction = math.atan2(vy, vx)
            self.direction_history[tracker_id].append(current_direction)

            if len(self.direction_history[tracker_id]) >= 3:
                directions = list(self.direction_history[tracker_id])
                angle_diffs = []
                for i in range(1, len(directions)):
                    diff = directions[i] - directions[i-1]
                    diff = math.atan2(math.sin(diff), math.cos(diff))
                    angle_diffs.append(abs(diff))

                if angle_diffs[-1] > math.pi / 2:
                    median_direction = np.median(directions[:-1])
                    speed = math.hypot(vx, vy)
                    kf.statePost[2, 0] = speed * math.cos(median_direction)
                    kf.statePost[3, 0] = speed * math.sin(median_direction)

    def apply_speed_calibration(self, tracker_id: int, calibration_func: Callable[[float], float]) -> None:
        """Apply speed calibration with smoothing to prevent fluctuations."""
        if tracker_id not in self.kf_states:
            return

        kf = self.kf_states[tracker_id]
        state = kf.statePost.flatten()
        x, y, vx, vy = state

        raw_speed = math.hypot(vx, vy)

        if raw_speed > self.min_speed_threshold:
            if len(self.speed_history[tracker_id]) > 0:
                smoothed_speed = self._calculate_ema_speed(tracker_id, raw_speed)
                calibrated_speed = calibration_func(smoothed_speed)

                if smoothed_speed > 0:
                    scale_factor = calibrated_speed / smoothed_speed
                    damping_factor = 0.7
                    smoothed_scale = 1.0 + (scale_factor - 1.0) * damping_factor
                    kf.statePost[2, 0] = vx * smoothed_scale
                    kf.statePost[3, 0] = vy * smoothed_scale

    def _calculate_ema_speed(self, tracker_id: int, current_speed: float, alpha: float = 0.3) -> float:
        """Calculate exponential moving average of speed."""
        history = list(self.speed_history[tracker_id])
        if not history:
            return current_speed

        ema = history[0]
        for speed in history[1:]:
            ema = alpha * speed + (1 - alpha) * ema

        ema = alpha * current_speed + (1 - alpha) * ema
        return ema

    def get_smoothed_velocity(self, tracker_id: int) -> Tuple[float, float]:
        """Get smoothed velocity components for display/output."""
        if tracker_id not in self.kf_states:
            return 0.0, 0.0

        kf = self.kf_states[tracker_id]
        state = kf.statePost.flatten()
        _, _, vx, vy = state

        if math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy):
            return 0.0, 0.0

        speed = math.hypot(vx, vy)
        if len(self.speed_history[tracker_id]) > 0:
            display_speed = self._calculate_ema_speed(tracker_id, speed)
            if speed > 0 and not math.isnan(display_speed) and not math.isinf(display_speed):
                scale = display_speed / speed
                return vx * scale, vy * scale

        return vx, vy

    def predict_future_positions(self, tracker_id: int, delta_t: float = 0.5,
                               num_predictions: int = 4) -> List[Tuple[float, float]]:
        """Predict future positions using smoothed velocity."""
        if tracker_id not in self.kf_states:
            return []

        kf = self.kf_states[tracker_id]
        future_positions = []
        current_state = deepcopy(kf.statePost.flatten())
        x, y, _, _ = current_state

        vx, vy = self.get_smoothed_velocity(tracker_id)

        for step in range(1, num_predictions + 1):
            t = step * delta_t
            x_future = x + vx * t
            y_future = y + vy * t
            future_positions.append((x_future, y_future))

        return future_positions

    def get_state(self, tracker_id: int) -> Optional[np.ndarray]:
        """Get current state vector for a tracker ID."""
        if tracker_id in self.kf_states:
            return self.kf_states[tracker_id].statePost.flatten()
        return None

    def remove_tracker(self, tracker_id: int):
        """Remove a tracker's Kalman filter and associated history."""
        self.kf_states.pop(tracker_id, None)
        self.speed_history.pop(tracker_id, None)
        self.direction_history.pop(tracker_id, None)
        self.last_positions.pop(tracker_id, None)
        self.initial_positions.pop(tracker_id, None)
        self.initial_velocity_applied.pop(tracker_id, None)
        # Clean up new metadata
        self.track_creation_frame.pop(tracker_id, None)
        self.track_confidence.pop(tracker_id, None)
        self.track_detection_count.pop(tracker_id, None)

    def get_all_states(self) -> Dict[int, cv2.KalmanFilter]:
        """Get all Kalman filter states."""
        return self.kf_states