"""Visual annotation utilities for drawing on frames."""
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import supervision as sv
import random


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

        # Double-line counting visualization state
        self.line_a_last_cross_time_sec = -float('inf')
        self.line_b_last_cross_time_sec = -float('inf')
        self.line_highlight_duration_sec = 0.5

        # Line colors
        self.line_a_default_color = (255, 0, 0)    # blue
        self.line_a_crossed_color = (0, 255, 255)  # yellow
        self.line_b_default_color = (0, 255, 0)    # green
        self.line_b_crossed_color = (0, 255, 255)  # yellow

    def signal_line_a_cross(self, current_time_seconds: float) -> None:
        """Signal that a vehicle has crossed Line A.

        Args:
            current_time_seconds: Current time in seconds from video start
        """
        self.line_a_last_cross_time_sec = current_time_seconds

    def signal_line_b_cross(self, current_time_seconds: float) -> None:
        """Signal that a vehicle has crossed Line B.

        Args:
            current_time_seconds: Current time in seconds from video start
        """
        self.line_b_last_cross_time_sec = current_time_seconds

    def draw_counting_lines(self, frame: np.ndarray,
                           line_a_coords: Optional[np.ndarray],
                           line_b_coords: Optional[np.ndarray],
                           current_time_seconds: float) -> np.ndarray:
        """Draw both counting lines with color changes on recent crossings.

        Args:
            frame: Current video frame
            line_a_coords: Line A coordinates as numpy array [[x1, y1], [x2, y2]]
            line_b_coords: Line B coordinates as numpy array [[x1, y1], [x2, y2]]
            current_time_seconds: Current time in seconds from video start

        Returns:
            Frame with counting lines drawn
        """
        # Draw Line A
        if line_a_coords is not None and len(line_a_coords) == 2:
            time_since_crossing_a = current_time_seconds - self.line_a_last_cross_time_sec

            if time_since_crossing_a < self.line_highlight_duration_sec:
                line_a_color = self.line_a_crossed_color
                line_a_thickness = self.thickness + 2
            else:
                line_a_color = self.line_a_default_color
                line_a_thickness = self.thickness

            # Draw Line A
            cv2.line(
                frame,
                tuple(line_a_coords[0].astype(int)),
                tuple(line_a_coords[1].astype(int)),
                line_a_color,
                line_a_thickness,
                lineType=cv2.LINE_AA
            )

            # Add Line A label
            midpoint_a = ((line_a_coords[0] + line_a_coords[1]) / 2).astype(int)
            cv2.putText(
                frame,
                "LINE A",
                (midpoint_a[0] - 30, midpoint_a[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_a_color,
                2,
                cv2.LINE_AA
            )

        # Draw Line B
        if line_b_coords is not None and len(line_b_coords) == 2:
            time_since_crossing_b = current_time_seconds - self.line_b_last_cross_time_sec

            if time_since_crossing_b < self.line_highlight_duration_sec:
                line_b_color = self.line_b_crossed_color
                line_b_thickness = self.thickness + 2
            else:
                line_b_color = self.line_b_default_color
                line_b_thickness = self.thickness

            # Draw Line B
            cv2.line(
                frame,
                tuple(line_b_coords[0].astype(int)),
                tuple(line_b_coords[1].astype(int)),
                line_b_color,
                line_b_thickness,
                lineType=cv2.LINE_AA
            )

            # Add Line B label
            midpoint_b = ((line_b_coords[0] + line_b_coords[1]) / 2).astype(int)
            cv2.putText(
                frame,
                "LINE B",
                (midpoint_b[0] - 30, midpoint_b[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_b_color,
                2,
                cv2.LINE_AA
            )

        return frame

    def draw_live_double_line_counts(self, frame: np.ndarray,
                                   a_to_b_incoming: int, a_to_b_outgoing: int,
                                   b_to_a_incoming: int, b_to_a_outgoing: int) -> np.ndarray:
        """Draw live double-line vehicle counter on the video frame.

        Args:
            frame: Current video frame
            a_to_b_incoming: A→B sequence incoming count
            a_to_b_outgoing: A→B sequence outgoing count
            b_to_a_incoming: B→A sequence incoming count
            b_to_a_outgoing: B→A sequence outgoing count

        Returns:
            Frame with live counter display
        """
        # Counter background (larger to accommodate more data)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Counter text
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Title
        cv2.putText(frame, "DOUBLE-LINE VEHICLE COUNTER", (20, 30),
                   font, font_scale + 0.1, text_color, thickness, cv2.LINE_AA)

        # A→B sequence counts
        cv2.putText(frame, f"A->B Incoming: {a_to_b_incoming}", (20, 55),
                   font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"A->B Outgoing: {a_to_b_outgoing}", (20, 75),
                   font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        # B→A sequence counts
        cv2.putText(frame, f"B->A Incoming: {b_to_a_incoming}", (20, 100),
                   font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"B->A Outgoing: {b_to_a_outgoing}", (20, 120),
                   font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

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
    
    def draw_analysis_lines(self, frame: np.ndarray, entry_line_px: np.ndarray, flow_line_px: np.ndarray, exit_line_px: np.ndarray) -> np.ndarray:
        """
        Draws the analysis segment lines (entry, flow, exit) on the frame.
        These lines are used for Space Mean Speed and Flow calculations.
        """
        # Define colors and styles for the lines
        ENTRY_COLOR = (255, 255, 0)   # Cyan
        FLOW_COLOR = (0, 255, 0)      # Green
        EXIT_COLOR = (255, 0, 255)    # Magenta
        THICKNESS = 2
        FONT_SCALE = 0.6
        TEXT_COLOR = (255, 255, 255)  # White

        # Draw Entry Line
        cv2.line(frame, tuple(entry_line_px[0]), tuple(entry_line_px[1]), ENTRY_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, "Analysis Entry", (entry_line_px[0][0] + 10, entry_line_px[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

        # Draw Flow Line (for vehicle counting)
        cv2.line(frame, tuple(flow_line_px[0]), tuple(flow_line_px[1]), FLOW_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, "Flow Count Line", (flow_line_px[0][0] + 10, flow_line_px[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

        # Draw Exit Line
        cv2.line(frame, tuple(exit_line_px[0]), tuple(exit_line_px[1]), EXIT_COLOR, THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, "Analysis Exit", (exit_line_px[0][0] + 10, exit_line_px[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

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