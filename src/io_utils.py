"""Input/output utilities for file operations."""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import threading


class IOManager:
    """Manages file I/O operations with optimized batching."""

    def __init__(self, output_dir: str = "results", batch_size: int = 100):
        """Initialize IO manager with batching capabilities."""
        self.output_dir = Path(output_dir)

        # Create directory with parent directories if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.batch_size = batch_size

        # Batching buffers
        self._vehicle_metrics_buffer: List[List] = []
        self._ttc_events_buffer: List[List] = []
        self._enhanced_ttc_buffer: List[List] = []
        self._encroachment_buffer: List[Dict] = []
        self._segment_speeds_buffer: List[List] = []

        # Thread lock for buffer operations
        self._buffer_lock = threading.Lock()

        # Log where files will be saved
        print(f"[IOManager] Results will be saved to: {self.output_dir.absolute()}")
        print(f"[IOManager] Batch size set to: {batch_size}")

    def add_vehicle_metric(self, frame: int, vehicle_id: str, vehicle_class: str,
                          confidence: float, speed_km_h: float) -> None:
        """Add single vehicle metric to buffer."""
        with self._buffer_lock:
            self._vehicle_metrics_buffer.append([frame, vehicle_id, vehicle_class, confidence, speed_km_h])
            if len(self._vehicle_metrics_buffer) >= self.batch_size:
                self._flush_vehicle_metrics()

    def add_ttc_event(self, frame: int, follower_id: str, follower_class: str,
                      leader_id: str, leader_class: str, closing_distance_m: float,
                      relative_velocity_m_s: float, ttc_s: float) -> None:
        """Add single TTC event to buffer."""
        with self._buffer_lock:
            self._ttc_events_buffer.append([
                frame, follower_id, follower_class, leader_id, leader_class,
                closing_distance_m, relative_velocity_m_s, ttc_s
            ])
            if len(self._ttc_events_buffer) >= self.batch_size:
                self._flush_ttc_events()

    def add_enhanced_ttc_event(self, frame: int, follower_id: str, follower_class: str,
                              leader_id: str, leader_class: str, closing_distance_m: float,
                              relative_velocity_m_s: float, ttc_s: float, confidence_score: float,
                              relative_angle_deg: float, kalman_eligible: bool) -> None:
        """Add single enhanced TTC event to buffer."""
        with self._buffer_lock:
            self._enhanced_ttc_buffer.append([
                frame, follower_id, follower_class, leader_id, leader_class,
                closing_distance_m, relative_velocity_m_s, ttc_s, confidence_score,
                relative_angle_deg, kalman_eligible
            ])
            if len(self._enhanced_ttc_buffer) >= self.batch_size:
                self._flush_enhanced_ttc_events()

    def add_encroachment_event(self, event: Dict) -> None:
        """Add single encroachment event to buffer."""
        with self._buffer_lock:
            self._encroachment_buffer.append(event)
            if len(self._encroachment_buffer) >= self.batch_size:
                self._flush_encroachment_events()

    def add_segment_speed(self, vehicle_id: str, frame_entry: int, frame_exit: int,
                         distance_m: float, time_s: float, speed_m_s: float, speed_km_h: float) -> None:
        """Add single segment speed measurement to buffer."""
        with self._buffer_lock:
            self._segment_speeds_buffer.append([
                vehicle_id, frame_entry, frame_exit, distance_m, time_s, speed_m_s, speed_km_h
            ])
            if len(self._segment_speeds_buffer) >= self.batch_size:
                self._flush_segment_speeds()

    def _flush_vehicle_metrics(self) -> Optional[Path]:
        """Flush vehicle metrics buffer to file."""
        if not self._vehicle_metrics_buffer:
            return None

        df = pd.DataFrame(
            self._vehicle_metrics_buffer,
            columns=["frame", "vehicle_id", "vehicle_class", "confidence", "speed_km_h"]
        )
        csv_file = self.output_dir / f"vehicle_metrics_{self.timestamp}.csv"

        # Append to existing file or create new one
        df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)

        # Clear buffer
        self._vehicle_metrics_buffer.clear()
        return csv_file

    def _flush_ttc_events(self) -> Optional[Path]:
        """Flush TTC events buffer to file."""
        if not self._ttc_events_buffer:
            return None

        df = pd.DataFrame(
            self._ttc_events_buffer,
            columns=[
                "frame", "follower_id", "follower_class", "leader_id", "leader_class",
                "closing_distance_m", "relative_velocity_m_s", "ttc_s"
            ]
        )
        csv_file = self.output_dir / f"ttc_events_{self.timestamp}.csv"

        # Append to existing file or create new one
        df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)

        # Clear buffer
        self._ttc_events_buffer.clear()
        return csv_file

    def _flush_enhanced_ttc_events(self) -> Optional[Path]:
        """Flush enhanced TTC events buffer to file."""
        if not self._enhanced_ttc_buffer:
            return None

        df = pd.DataFrame(
            self._enhanced_ttc_buffer,
            columns=[
                "frame", "follower_id", "follower_class", "leader_id", "leader_class",
                "closing_distance_m", "relative_velocity_m_s", "ttc_s", "confidence_score",
                "relative_angle_deg", "kalman_eligible"
            ]
        )
        csv_file = self.output_dir / f"enhanced_ttc_events_{self.timestamp}.csv"

        # Append to existing file or create new one
        df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)

        # Clear buffer
        self._enhanced_ttc_buffer.clear()
        return csv_file

    def _flush_encroachment_events(self) -> Optional[Path]:
        """Flush encroachment events buffer to file."""
        if not self._encroachment_buffer:
            return None

        csv_file = self.output_dir / f"enc_events_{self.timestamp}.csv"
        df = pd.DataFrame(self._encroachment_buffer)

        # Append to existing file or create new one
        df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)

        # Clear buffer
        self._encroachment_buffer.clear()
        return csv_file

    def _flush_segment_speeds(self) -> Optional[Path]:
        """Flush segment speeds buffer to file."""
        if not self._segment_speeds_buffer:
            return None

        df = pd.DataFrame(
            self._segment_speeds_buffer,
            columns=["vehicle_id", "frame_entry", "frame_exit", "distance_m",
                     "time_s", "speed_m_s", "speed_km_h"]
        )
        csv_file = self.output_dir / f"segment_speeds_{self.timestamp}.csv"

        # Append to existing file or create new one
        df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)

        # Clear buffer
        self._segment_speeds_buffer.clear()
        return csv_file

    def flush_all_buffers(self) -> None:
        """Flush all buffers to files."""
        with self._buffer_lock:
            self._flush_vehicle_metrics()
            self._flush_ttc_events()
            self._flush_enhanced_ttc_events()
            self._flush_encroachment_events()
            self._flush_segment_speeds()
            print("[IOManager] All buffers flushed to disk")

    # Legacy methods for backward compatibility
    def save_vehicle_metrics(self, csv_rows: List[List]) -> Path:
        """Save vehicle metrics to CSV (legacy method)."""
        if not csv_rows:
            return None

        df = pd.DataFrame(
            csv_rows,
            columns=["frame", "vehicle_id", "vehicle_class", "confidence", "speed_km_h"]
        )
        csv_file = self.output_dir / f"vehicle_metrics_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def save_enhanced_ttc_events(self, enhanced_ttc_rows: List[List]) -> Path:
        """Save enhanced TTC events with additional filtering metrics (legacy method)."""
        if not enhanced_ttc_rows:
            return None

        df = pd.DataFrame(
            enhanced_ttc_rows,
            columns=[
                "frame", "follower_id", "follower_class", "leader_id", "leader_class",
                "closing_distance_m", "relative_velocity_m_s", "ttc_s", "confidence_score",
                "relative_angle_deg", "kalman_eligible"
            ]
        )
        csv_file = self.output_dir / f"enhanced_ttc_events_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"[IOManager] Enhanced TTC events saved to: {csv_file}")
        return csv_file

    def save_double_line_vehicle_counts(self, processed_counts_list: List[Dict]) -> Path:
        """Save vehicle counting results to CSV (compatible with both double-line and single-passage counting)."""
        if not processed_counts_list:
            print("[IOManager] No counting data to save")
            return None

        df = pd.DataFrame(processed_counts_list)

        # Check if this is single-passage counting data (new format)
        if 'incoming_count' in df.columns:
            # Single-passage counting format
            required_columns = [
                'vehicle_class', 'incoming_count', 'outgoing_count', 'total_count',
                'avg_speed_incoming_kmh', 'avg_speed_outgoing_kmh'
            ]

            for col in required_columns:
                if col not in df.columns:
                    if 'avg_speed' in col:
                        df[col] = 0.0
                    elif 'total_count' in col:
                        df[col] = df.get('incoming_count', 0) + df.get('outgoing_count', 0)
                    else:
                        df[col] = 0

            df = df[required_columns]
            csv_file = self.output_dir / f"vehicle_counts_{self.timestamp}.csv"
            file_type = "vehicle counts"

        else:
            # Legacy double-line counting format
            required_columns = [
                'vehicle_class', 'a_to_b_incoming', 'a_to_b_outgoing',
                'b_to_a_incoming', 'b_to_a_outgoing', 'avg_speed_a_to_b_incoming_kmh',
                'avg_speed_a_to_b_outgoing_kmh', 'avg_speed_b_to_a_incoming_kmh',
                'avg_speed_b_to_a_outgoing_kmh'
            ]

            for col in required_columns:
                if col not in df.columns:
                    if 'avg_speed' in col:
                        df[col] = 0.0
                    else:
                        df[col] = 0

            df = df[required_columns]
            csv_file = self.output_dir / f"double_line_vehicle_counts_{self.timestamp}.csv"
            file_type = "double-line vehicle counts"

        df.to_csv(csv_file, index=False)
        print(f"[IOManager] {file_type.title()} saved to: {csv_file}")
        return csv_file

    def save_ttc_events(self, ttc_rows: List[List]) -> Path:
        """Save TTC events to CSV (legacy method)."""
        if not ttc_rows:
            return None

        df = pd.DataFrame(
            ttc_rows,
            columns=[
                "frame", "follower_id", "follower_class", "leader_id", "leader_class",
                "closing_distance_m", "relative_velocity_m_s", "ttc_s"
            ]
        )
        csv_file = self.output_dir / f"ttc_events_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def save_encroachment_events(self, enc_events: List[Dict]) -> Path:
        """Save encroachment events to CSV (legacy method)."""
        if not enc_events:
            return None

        csv_file = self.output_dir / f"enc_events_{self.timestamp}.csv"
        pd.DataFrame(enc_events).to_csv(csv_file, index=False)
        return csv_file

    def save_segment_speeds(self, segment_results: List[List]) -> Path:
        """Save segment-based speed measurements to CSV (legacy method)."""
        if not segment_results:
            return None

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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush all buffers."""
        self.flush_all_buffers()