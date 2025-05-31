"""Kalman filter for state estimation and future position prediction."""
from typing import Dict, List, Tuple, Optional  # Added Optional
import cv2
import numpy as np
from copy import deepcopy


class KalmanFilterManager:
    """Manages Kalman filters for multiple tracked objects."""

    def __init__(self):
        """Initialize the Kalman filter manager."""
        self.kf_states: Dict[int, cv2.KalmanFilter] = {}

    def create_kalman_filter(self, dt: float) -> cv2.KalmanFilter:
        """Create a new Kalman filter with given time step."""
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

    def update_or_create(self, tracker_id: int, x: float, y: float, dt: float) -> cv2.KalmanFilter:
        """Update existing Kalman filter or create new one."""
        if tracker_id not in self.kf_states:
            kf = self.create_kalman_filter(dt)
            kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
            self.kf_states[tracker_id] = kf
        else:
            kf = self.kf_states[tracker_id]
            # Update transition matrix if dt varies
            kf.transitionMatrix[0, 2] = dt
            kf.transitionMatrix[1, 3] = dt

            kf.predict()
            kf.correct(np.array([[x], [y]], np.float32))

        return kf

    def predict_future_positions(self, tracker_id: int, delta_t: float = 0.5,
                               num_predictions: int = 4) -> List[Tuple[float, float]]:
        """Predict future positions using constant velocity assumption."""
        if tracker_id not in self.kf_states:
            return []

        kf = self.kf_states[tracker_id]
        future_positions = []
        current_state = deepcopy(kf.statePost.flatten())  # [x, y, vx, vy]
        x, y, vx, vy = current_state

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
        """Remove a tracker's Kalman filter."""
        self.kf_states.pop(tracker_id, None)

    def get_all_states(self) -> Dict[int, cv2.KalmanFilter]:
        """Get all Kalman filter states."""
        return self.kf_states