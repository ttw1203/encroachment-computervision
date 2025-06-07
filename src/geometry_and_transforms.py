"""Geometric transformations and utilities."""
import cv2
import numpy as np
from typing import Tuple, Optional


class ViewTransformer:
    """Handles perspective transformation between image and bird's eye view."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Initialize transformer with source and target points."""
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Store source and target for cache validation
        self._source = source.copy()
        self._target = target.copy()

        # Cache transformation matrices
        self._m: Optional[np.ndarray] = None
        self._m_inv: Optional[np.ndarray] = None

        # Initialize matrices
        self._compute_matrices()

    def _compute_matrices(self) -> None:
        """Compute and cache transformation matrices."""
        self._m = cv2.getPerspectiveTransform(self._source, self._target)
        self._m_inv = cv2.getPerspectiveTransform(self._target, self._source)

    @property
    def m(self) -> np.ndarray:
        """Get forward transformation matrix."""
        if self._m is None:
            self._compute_matrices()
        return self._m

    @property
    def m_inv(self) -> np.ndarray:
        """Get inverse transformation matrix."""
        if self._m_inv is None:
            self._compute_matrices()
        return self._m_inv

    def update_points(self, source: np.ndarray, target: np.ndarray) -> None:
        """Update transformation points and invalidate cache if changed."""
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Check if points have changed
        if not (np.array_equal(source, self._source) and np.array_equal(target, self._target)):
            self._source = source.copy()
            self._target = target.copy()
            # Invalidate cache
            self._m = None
            self._m_inv = None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from source to target coordinate system."""
        # Early exit for empty arrays
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from target back to source coordinate system."""
        # Early exit for empty arrays
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m_inv)
        return transformed_points.reshape(-1, 2)


def line_side(pt: Tuple[float, float], a: Tuple[float, float],
              b: Tuple[float, float]) -> int:
    """Determine which side of line AB the point is on.

    Returns:
        +1 if pt is on left of AB, -1 on right, 0 on the line
    """
    return np.sign((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]))