"""Geometric transformations and utilities."""
import cv2
import numpy as np
from typing import Tuple


class ViewTransformer:
    """Handles perspective transformation between image and bird's eye view."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Initialize transformer with source and target points."""
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        self.m_inv = cv2.getPerspectiveTransform(target, source)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from source to target coordinate system."""
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from target back to source coordinate system."""
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