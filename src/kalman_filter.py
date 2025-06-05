"""Kalman filter for state estimation and future position prediction with enhanced stability."""
from typing import Dict, List, Tuple, Optional, Callable
import cv2
import numpy as np
from copy import deepcopy
import math
from collections import deque
import logging


class KalmanFilterManager:
    """Manages Kalman filters for multiple tracked objects with enhanced stability."""

    def __init__(self, speed_smoothing_window: int = 5, max_acceleration: float = 5.0,
                 min_speed_threshold: float = 0.1, initial_velocity_frames: int = 2):
        """Initialize the Kalman filter manager with stability parameters.

        Args:
            speed_smoothing_window: Number of frames to average for speed smoothing
            max_acceleration: Maximum allowed acceleration in m/s² (prevents unrealistic jumps)
            min_speed_threshold: Minimum speed threshold in m/s below which velocity is set to 0
            initial_velocity_frames: Number of frames to use for initial velocity calculation (2 or 3)
        """
        # Existing attributes
        self.kf_states: Dict[int, cv2.KalmanFilter] = {}
        self.speed_history: Dict[int, deque] = {}  # Store historical speeds for smoothing
        self.last_positions: Dict[int, Tuple[float, float]] = {}  # Store last known positions
        self.speed_smoothing_window = speed_smoothing_window
        self.max_acceleration = max_acceleration
        self.min_speed_threshold = min_speed_threshold
        self.direction_history: Dict[int, deque] = {}  # Store direction history

        # New attributes for initial velocity calculation
        self.initial_positions: Dict[int, List[Tuple[float, float, int]]] = {}  # tracker_id -> [(x, y, frame_idx), ...]
        self.initial_velocity_frames = initial_velocity_frames  # Number of frames to use (2 or 3)
        self.initial_velocity_applied: Dict[int, bool] = {}  # Track if initial velocity was set

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
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0

        return kf

    def update_or_create(self, tracker_id: int, x: float, y: float, dt: float, frame_idx: int) -> cv2.KalmanFilter:
        """Update existing Kalman filter or create new one with stability checks."""

        # Track initial positions for new trackers
        if tracker_id not in self.initial_positions:
            self.initial_positions[tracker_id] = []

        # Store position if we haven't applied initial velocity yet
        if tracker_id not in self.initial_velocity_applied:
            self.initial_positions[tracker_id].append((x, y, frame_idx))

        if tracker_id not in self.kf_states:
            # Initialize new tracker (existing code)
            kf = self.create_kalman_filter(dt)
            kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
            self.kf_states[tracker_id] = kf
            self.speed_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.direction_history[tracker_id] = deque(maxlen=self.speed_smoothing_window)
            self.last_positions[tracker_id] = (x, y)
            self.initial_velocity_applied[tracker_id] = False
        else:
            kf = self.kf_states[tracker_id]

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

    def _apply_initial_velocity(self, tracker_id: int, dt: float) -> None:
        """Calculate and apply initial velocity based on first few detections."""
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

        # Validate time difference to prevent division by zero or very small values
        if time_diff <= 0.001:  # Less than 1ms
            return

        # Check for extreme position jumps (likely detection errors)
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance > 100.0:  # More than 100m jump between frames
            logging.warning(
                f"Tracker {tracker_id}: Extreme position jump detected ({distance:.1f}m), skipping initial velocity")
            return

        vx_init = (x2 - x1) / time_diff
        vy_init = (y2 - y1) / time_diff

        # Validate calculated velocities
        if math.isnan(vx_init) or math.isnan(vy_init) or math.isinf(vx_init) or math.isinf(vy_init):
            logging.warning(f"Tracker {tracker_id}: Invalid initial velocity calculated, skipping")
            return

        # Calculate initial speed
        speed_init = math.hypot(vx_init, vy_init)

        # Apply constraints before setting
        if speed_init < self.min_speed_threshold:
            return

        # Check for unrealistic speeds
        max_initial_speed = 50.0  # m/s (180 km/h)
        if speed_init > max_initial_speed:
            # Scale down to maximum
            scale = max_initial_speed / speed_init
            vx_init *= scale
            vy_init *= scale
            speed_init = max_initial_speed

        # Apply to Kalman filter
        kf = self.kf_states[tracker_id]
        kf.statePost[2, 0] = vx_init
        kf.statePost[3, 0] = vy_init

        # Initialize speed history with this value
        self.speed_history[tracker_id].append(speed_init)

        # Initialize direction history
        if abs(vx_init) > 0.01 or abs(vy_init) > 0.01:
            initial_direction = math.atan2(vy_init, vx_init)
            self.direction_history[tracker_id].append(initial_direction)

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

        # Validate velocity components
        if math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy):
            return 0.0, 0.0

        # Apply additional smoothing for display
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
        # Clean up new attributes
        self.initial_positions.pop(tracker_id, None)
        self.initial_velocity_applied.pop(tracker_id, None)

    def get_all_states(self) -> Dict[int, cv2.KalmanFilter]:
        """Get all Kalman filter states."""
        return self.kf_states