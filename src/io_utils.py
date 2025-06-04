"""Input/output utilities for file operations."""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class IOManager:
    """Manages file I/O operations."""

    def __init__(self, output_dir: str = "results"):
        """Initialize IO manager."""
        self.output_dir = Path(output_dir)

        # Create directory with parent directories if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        # Log where files will be saved
        print(f"[IOManager] Results will be saved to: {self.output_dir.absolute()}")

    def save_vehicle_metrics(self, csv_rows: List[List]) -> Path:
        """Save vehicle metrics to CSV."""
        df = pd.DataFrame(
            csv_rows,
            columns=["frame", "vehicle_id", "vehicle_class", "confidence", "speed_km_h"]
        )
        csv_file = self.output_dir / f"vehicle_metrics_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def save_double_line_vehicle_counts(self, processed_counts_list: List[Dict]) -> Path:
        """Save double-line vehicle counting results to CSV.

        Args:
            processed_counts_list: List of dictionaries with double-line counting data

        Returns:
            Path to saved CSV file
        """
        df = pd.DataFrame(processed_counts_list)

        # Ensure all required columns exist
        required_columns = [
            'vehicle_class',
            'a_to_b_incoming',
            'a_to_b_outgoing',
            'b_to_a_incoming',
            'b_to_a_outgoing',
            'avg_speed_a_to_b_incoming_kmh',
            'avg_speed_a_to_b_outgoing_kmh',
            'avg_speed_b_to_a_incoming_kmh',
            'avg_speed_b_to_a_outgoing_kmh'
        ]

        for col in required_columns:
            if col not in df.columns:
                if 'avg_speed' in col:
                    df[col] = 0.0
                else:
                    df[col] = 0

        # Reorder columns
        df = df[required_columns]

        csv_file = self.output_dir / f"double_line_vehicle_counts_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"[IOManager] Double-line vehicle counts saved to: {csv_file}")
        return csv_file

    def save_ttc_events(self, ttc_rows: List[List]) -> Path:
        """Save TTC events to CSV."""
        df = pd.DataFrame(
            ttc_rows,
            columns=[
                "frame",
                "follower_id", "follower_class",
                "leader_id", "leader_class",
                "closing_distance_m",
                "relative_velocity_m_s",
                "ttc_s"
            ]
        )
        csv_file = self.output_dir / f"ttc_events_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def save_encroachment_events(self, enc_events: List[Dict]) -> Path:
        """Save encroachment events to CSV."""
        csv_file = self.output_dir / f"enc_events_{self.timestamp}.csv"
        pd.DataFrame(enc_events).to_csv(csv_file, index=False)
        return csv_file

    def save_segment_speeds(self, segment_results: List[List]) -> Path:
        """Save segment-based speed measurements to CSV."""
        df = pd.DataFrame(
            segment_results,
            columns=["vehicle_id", "frame_entry", "frame_exit", "distance_m",
                     "time_s", "speed_m_s", "speed_km_h"]
        )
        csv_file = self.output_dir / f"segment_speeds_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    @staticmethod
    def dump_zones_png(video_path: str, output_path: str,
                       left_zone: np.ndarray, right_zone: np.ndarray) -> None:
        """Save a preview image showing zone overlays."""
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            raise RuntimeError("Cannot read first frame for zone preview")

        # Convert to OpenCV contour format
        left_cnt = left_zone.reshape((-1, 1, 2))
        right_cnt = right_zone.reshape((-1, 1, 2))

        # Make translucent overlays
        preview = frame.copy()
        cv2.fillPoly(preview, [left_cnt], (0, 255, 0))
        cv2.fillPoly(preview, [right_cnt], (0, 0, 255))
        frame = cv2.addWeighted(preview, 0.35, frame, 0.65, 0, frame)

        # Draw crisp outlines
        cv2.polylines(frame, [left_cnt], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_cnt], True, (0, 0, 255), 2)

        cv2.imwrite(output_path, frame)
        print(f"[zone-preview] saved → {output_path}")