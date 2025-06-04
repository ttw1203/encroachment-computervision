"""Visual annotation utilities for drawing on frames."""
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import supervision as sv
import random
from collections import defaultdict


class AnnotationManager:
    """Manages visual annotations on video frames."""

    def __init__(self, thickness: int = 2, text_scale: float = 1.0,
                 trace_length_seconds: float = 3.0, video_fps: float = 30.0):
        """Initialize annotation components."""
        self.thickness = thickness
        self.text_scale = text_scale

        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.TOP_LEFT,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=int(video_fps * trace_length_seconds),
            position=sv.Position.CENTER,
        )

        # Color cache for future trajectories
        self.future_colors: Dict[int, Tuple[int, int, int]] = {}

        # Counting line visualization state
        self.counting_line_last_cross_time_sec = -float('inf')
        self.counting_line_highlight_duration_sec = 0.5
        self.counting_line_default_color = (255, 255, 0)  # cyan
        self.counting_line_crossed_color = (0, 255, 255)  # yellow

    def signal_counting_line_cross(self, current_time_seconds: float) -> None:
        """Signal that a vehicle has crossed the counting line.

        Args:
            current_time_seconds: Current time in seconds from video start
        """
        self.counting_line_last_cross_time_sec = current_time_seconds

    def draw_counting_line(self, frame: np.ndarray, line_coords: np.ndarray,
                          current_time_seconds: float) -> np.ndarray:
        """Draw the counting line with color change on recent crossing.

        Args:
            frame: Current video frame
            line_coords: Line coordinates as numpy array [[x1, y1], [x2, y2]]
            current_time_seconds: Current time in seconds from video start

        Returns:
            Frame with counting line drawn
        """
        if line_coords is None or len(line_coords) != 2:
            return frame

        # Determine line color based on recent crossing
        time_since_crossing = current_time_seconds - self.counting_line_last_cross_time_sec

        if time_since_crossing < self.counting_line_highlight_duration_sec:
            line_color = self.counting_line_crossed_color
            line_thickness = self.thickness + 2
        else:
            line_color = self.counting_line_default_color
            line_thickness = self.thickness

        # Draw the counting line
        cv2.line(
            frame,
            tuple(line_coords[0].astype(int)),
            tuple(line_coords[1].astype(int)),
            line_color,
            line_thickness,
            lineType=cv2.LINE_AA
        )

        # Add label
        midpoint = ((line_coords[0] + line_coords[1]) / 2).astype(int)
        cv2.putText(
            frame,
            "COUNTING LINE",
            (midpoint[0] - 60, midpoint[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_live_vehicle_counts(self, frame: np.ndarray, incoming_count: int,
                                outgoing_count: int) -> np.ndarray:
        """Draw live vehicle counter on the video frame.

        Args:
            frame: Current video frame
            incoming_count: Total incoming vehicle count
            outgoing_count: Total outgoing vehicle count

        Returns:
            Frame with live counter display
        """
        # Counter background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Counter text
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3

        # Title
        cv2.putText(frame, "VEHICLE COUNTER", (20, 35),
                   font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Incoming count
        cv2.putText(frame, f"Incoming: {incoming_count}", (20, 70),
                   font, font_scale - 0.1, (0, 255, 0), thickness, cv2.LINE_AA)

        # Outgoing count
        cv2.putText(frame, f"Outgoing: {outgoing_count}", (20, 105),
                   font, font_scale - 0.1, (0, 0, 255), thickness, cv2.LINE_AA)

        return frame

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections,
                       labels: List[str], future_coordinates: Dict[int, List],
                       active_ids: set) -> np.ndarray:
        """Apply all annotations to a frame."""
        annotated = frame.copy()

        # Draw traces
        annotated = self.trace_annotator.annotate(
            scene=annotated, detections=detections
        )

        # Draw future trajectories
        annotated = self.draw_future_trajectories(
            annotated, future_coordinates, active_ids
        )

        # Draw boxes and labels
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated

    def draw_future_trajectories(self, frame: np.ndarray,
                                 future_coordinates: Dict[int, List],
                                 active_ids: set) -> np.ndarray:
        """Draw future trajectory predictions as dots."""
        h, w = frame.shape[:2]

        for tid in list(future_coordinates.keys()):
            # Skip inactive tracks
            if tid not in active_ids:
                continue

            # Keep only in-bounds points
            in_bounds = [
                (int(x), int(y))
                for x, y in future_coordinates[tid]
                if 0 <= x < w and 0 <= y < h
            ]

            if not in_bounds:
                continue

            # Get or create color for this track
            colour = self.future_colors.setdefault(
                tid,
                (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))
            )

            # Draw future points
            for cx, cy in in_bounds:
                cv2.circle(
                    frame,
                    (cx, cy),
                    radius=4,
                    color=colour,
                    thickness=-1,  # filled
                    lineType=cv2.LINE_AA
                )

        return frame

    def draw_segment_lines(self, frame: np.ndarray, entry_line: np.ndarray,
                           exit_line: np.ndarray) -> np.ndarray:
        """Draw segment entry/exit lines for speed measurement."""
        ENTRY_COLOR = (0, 255, 255)  # yellow
        EXIT_COLOR = (0, 0, 255)  # red
        THICKNESS = 3

        # Draw lines
        cv2.line(frame, tuple(entry_line[0]), tuple(entry_line[1]),
                 ENTRY_COLOR, THICKNESS)
        cv2.arrowedLine(frame, tuple(exit_line[0]), tuple(exit_line[1]),
                        EXIT_COLOR, THICKNESS, tipLength=0.05)

        # Add labels
        def _midpt(a, b):
            return (int((a[0] + b[0]) * 0.5), int((a[1] + b[1]) * 0.5))

        cv2.putText(frame, "ENTRY", _midpt(*entry_line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ENTRY_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "EXIT", _midpt(*exit_line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, EXIT_COLOR, 2, cv2.LINE_AA)

        return frame