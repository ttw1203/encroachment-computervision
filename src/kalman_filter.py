"""Kalman filter for state estimation and future position prediction with enhanced stability."""
from typing import Dict, List, Tuple, Optional, Callable
import cv2
import numpy as np
from copy import deepcopy
import math
from collections import deque


class KalmanFilterManager:
    """Manages Kalman filters for multiple tracked objects with enhanced stability."""

    def __init__(self, speed_smoothing_window: int = 5, max_acceleration: float = 5.0,
                 min_speed_threshold: float = 0.1):
        """Initialize the Kalman filter manager with stability parameters.

        Args:
            speed_smoothing_window: Number of frames to average for speed smoothing
            max_acceleration: Maximum allowed acceleration in m/s² (prevents unrealistic jumps)
            min_speed_threshold: Minimum speed threshold in m/s below which velocity is set to 0
        """
        self.kf_states: Dict[int, cv2.KalmanFilter] = {}
        self.speed_history: Dict[int, deque] = {}  # Store historical speeds for smoothing
        self.last_positions: Dict[int, Tuple[float, float]] = {}  # Store last known positions
        self.speed_smoothing_window = speed_smoothing_window
        self.max_acceleration = max_acceleration
        self.min_speed_threshold = min_speed_threshold
        self.direction_history: Dict[int, deque] = {}  # Store direction history

    def create_kalman_filter(self, dt: float) -> cv2.KalmanFilter:
        """Create a new Kalman filter with given time step and tuned parameters."""
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

        # Tuned noise parameters for better stability
        # Reduced process noise for smoother velocity estimates
        kf.processNoiseCov = np.array([
            [0.01, 0,    0.01, 0],     # position process noise
            [0,    0.01, 0,    0.01],
            [0.01, 0,    0.1,  0],     # velocity process noise (higher)
            [0,    0.01, 0,    0.1]
        ], np.float32)

        # Increased measurement noise to trust predictions more
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0

        return kf

    def update_or_create(self, tracker_id: int, x: float, y: float, dt: float) -> cv2.KalmanFilter:
        """Update existing Kalman filter or create new one with stability checks."""
        if tracker_id not in self.kf_states:
            # Initialize new tracker
            kf = self.create_kalman_filter(dt)
            kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
            self.kf_states[tracker_id] = kf
            self.speed_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.direction_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.last_positions[tracker_id] = (x, y)
        else:
            kf = self.kf_states[tracker_id]
            # Update transition matrix if dt varies
            kf.transitionMatrix[0, 2] = dt
            kf.transitionMatrix[1, 3] = dt

            # Predict next state
            prediction = kf.predict()

            # Apply measurement
            measurement = np.array([[x], [y]], np.float32)
            kf.correct(measurement)

            # Apply velocity constraints after correction
            self._apply_velocity_constraints(tracker_id, dt)

            # Update position history
            self.last_positions[tracker_id] = (x, y)

        return kf

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

        if abs(vx) > 0.01 or abs(vy) > 0.01:  # Only if moving
            current_direction = math.atan2(vy, vx)
            self.direction_history[tracker_id].append(current_direction)

            if len(self.direction_history[tracker_id]) >= 3:
                # Check for sudden direction changes
                directions = list(self.direction_history[tracker_id])

                # Calculate angular differences
                angle_diffs = []
                for i in range(1, len(directions)):
                    diff = directions[i] - directions[i-1]
                    # Normalize to [-pi, pi]
                    diff = math.atan2(math.sin(diff), math.cos(diff))
                    angle_diffs.append(abs(diff))

                # If recent angle change is too large (>90 degrees), use averaged direction
                if angle_diffs[-1] > math.pi / 2:
                    # Use median direction from history
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

        # Calculate raw speed magnitude
        raw_speed = math.hypot(vx, vy)

        # Only calibrate if speed is above threshold
        if raw_speed > self.min_speed_threshold:
            # Get smoothed speed from history
            if len(self.speed_history[tracker_id]) > 0:
                # Use exponential moving average for smoother results
                smoothed_speed = self._calculate_ema_speed(tracker_id, raw_speed)

                # Apply calibration to smoothed speed
                calibrated_speed = calibration_func(smoothed_speed)

                # Calculate scaling factor
                if smoothed_speed > 0:
                    scale_factor = calibrated_speed / smoothed_speed

                    # Apply scaling with damping to prevent sudden changes
                    damping_factor = 0.7  # Adjust between 0-1 (higher = more damping)
                    smoothed_scale = 1.0 + (scale_factor - 1.0) * damping_factor

                    # Update velocity components
                    kf.statePost[2, 0] = vx * smoothed_scale
                    kf.statePost[3, 0] = vy * smoothed_scale

    def _calculate_ema_speed(self, tracker_id: int, current_speed: float, alpha: float = 0.3) -> float:
        """Calculate exponential moving average of speed."""
        history = list(self.speed_history[tracker_id])
        if not history:
            return current_speed

        # EMA calculation
        ema = history[0]
        for speed in history[1:]:
            ema = alpha * speed + (1 - alpha) * ema

        # Include current speed in EMA
        ema = alpha * current_speed + (1 - alpha) * ema
        return ema

    def get_smoothed_velocity(self, tracker_id: int) -> Tuple[float, float]:
        """Get smoothed velocity components for display/output."""
        if tracker_id not in self.kf_states:
            return 0.0, 0.0

        kf = self.kf_states[tracker_id]
        state = kf.statePost.flatten()
        _, _, vx, vy = state

        # Apply additional smoothing for display
        speed = math.hypot(vx, vy)
        if len(self.speed_history[tracker_id]) > 0:
            display_speed = self._calculate_ema_speed(tracker_id, speed)
            if speed > 0:
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

        # Use smoothed velocity for predictions
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

    def get_all_states(self) -> Dict[int, cv2.KalmanFilter]:
        """Get all Kalman filter states."""
        return self.kf_states