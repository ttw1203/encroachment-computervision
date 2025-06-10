"""Ground Truth Vehicle Clip Extractor

This script extracts 10-second video clips for ground truth vehicles and processes them
with the vehicle tracking pipeline.
"""

import os
import sys
import csv
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime

# Import configuration
from src.config import Config
from src.io_utils import IOManager


class GroundTruthExtractor:
    """Extract and process video clips for ground truth vehicles."""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def parse_timestamp(self, timestamp_hms: str) -> float:
        """Convert HH:MM:SS timestamp to seconds.

        Args:
            timestamp_hms: Time in HH:MM:SS format

        Returns:
            Time in seconds
        """
        try:
            parts = timestamp_hms.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid timestamp format: {timestamp_hms}")

            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds

        except Exception as e:
            self.logger.error(f"Error parsing timestamp {timestamp_hms}: {e}")
            raise

    def format_timestamp_for_filename(self, timestamp_hms: str) -> str:
        """Convert HH:MM:SS to HH-MM-SS for filename."""
        return timestamp_hms.replace(':', '-')

    def read_ground_truth_csv(self, csv_path: str) -> List[Dict]:
        """Read ground truth data from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of vehicle data dictionaries
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        vehicles = []

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                # Validate headers
                required_headers = {'vehicle_id', 'timestamp_hms', 'ground_truth_speed_kmph'}
                if not required_headers.issubset(reader.fieldnames):
                    missing = required_headers - set(reader.fieldnames)
                    raise ValueError(f"Missing required columns: {missing}")

                for row in reader:
                    # Parse and validate data
                    vehicle_data = {
                        'vehicle_id': row['vehicle_id'].strip(),
                        'timestamp_hms': row['timestamp_hms'].strip(),
                        'ground_truth_speed_kmph': float(row['ground_truth_speed_kmph']),
                        'timestamp_seconds': self.parse_timestamp(row['timestamp_hms'].strip())
                    }

                    # Optional: vehicle class if provided
                    if 'vehicle_class' in row:
                        vehicle_data['vehicle_class'] = row['vehicle_class'].strip()
                    else:
                        vehicle_data['vehicle_class'] = 'vehicle'  # default

                    vehicles.append(vehicle_data)

        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            raise

        self.logger.info(f"Loaded {len(vehicles)} vehicles from CSV")
        return vehicles

    def get_video_info(self, video_path: str) -> Tuple[float, int, float]:
        """Get video information (duration, frame count, fps).

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (duration_seconds, total_frames, fps)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            return duration, total_frames, fps

        finally:
            cap.release()

    def extract_clip_ffmpeg(self, input_video: str, output_path: str,
                            start_time: float, duration: float) -> bool:
        """Extract video clip using ffmpeg.

        Args:
            input_video: Path to input video
            output_path: Path to output clip
            start_time: Start time in seconds
            duration: Clip duration in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-ss', str(start_time),
                '-i', input_video,
                '-t', str(duration),
                '-c', 'copy',  # Copy codec for speed
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"ffmpeg error: {result.stderr}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error extracting clip with ffmpeg: {e}")
            return False

    def extract_clip_opencv(self, input_video: str, output_path: str,
                            start_time: float, duration: float, fps: float) -> bool:
        """Extract video clip using OpenCV (fallback method).

        Args:
            input_video: Path to input video
            output_path: Path to output clip
            start_time: Start time in seconds
            duration: Clip duration in seconds
            fps: Video frame rate

        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(input_video)

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Create output writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)

            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Extract frames
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()

            return True

        except Exception as e:
            self.logger.error(f"Error extracting clip with OpenCV: {e}")
            return False

    def process_vehicle_clip(self, clip_path: str, vehicle_data: Dict,
                             output_dir: str) -> bool:
        """Process a single vehicle clip with the tracking pipeline.

        Args:
            clip_path: Path to the extracted clip
            vehicle_data: Vehicle information dictionary
            output_dir: Directory for results

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create vehicle-specific output directory
            vehicle_output_dir = os.path.join(output_dir, vehicle_data['vehicle_id'])
            os.makedirs(vehicle_output_dir, exist_ok=True)

            # Prepare output video path
            output_video = os.path.join(
                vehicle_output_dir,
                f"{vehicle_data['vehicle_id']}_tracked.mp4"
            )

            # Build command for main.py
            cmd = [
                sys.executable,
                'main.py',
                '--source_video_path', clip_path,
                '--target_video_path', output_video,
                '--no_blend_zones',  # As requested
                '--segment_speed',  # As requested
                '--confidence_threshold', '0.55',
                '--iou_threshold', '0.3',
                '--detector_model', 'rf_detr',
                '--env_file', '.env.boardbazar.kaggle',
                '--tracker', 'bytetrack'
            ]

            # Set environment for output directory
            env = os.environ.copy()
            env['RESULTS_OUTPUT_DIR'] = vehicle_output_dir

            # Run the tracking pipeline
            self.logger.info(f"Processing {vehicle_data['vehicle_id']}...")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode != 0:
                self.logger.error(f"Tracking pipeline error: {result.stderr}")
                return False

            # Post-process segment results to add vehicle class
            self.update_segment_results(vehicle_output_dir, vehicle_data)

            # Add ground truth speed to results
            self.add_ground_truth_info(vehicle_output_dir, vehicle_data)

            return True

        except Exception as e:
            self.logger.error(f"Error processing clip: {e}")
            return False

    def update_segment_results(self, output_dir: str, vehicle_data: Dict):
        """Update segment_results.csv with vehicle class information.

        Args:
            output_dir: Directory containing results
            vehicle_data: Vehicle information dictionary
        """
        segment_file = os.path.join(output_dir, 'segment_results.csv')

        if not os.path.exists(segment_file):
            self.logger.warning(f"Segment results not found for {vehicle_data['vehicle_id']}")
            return

        try:
            # Read existing data
            rows = []
            with open(segment_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)

                # Add vehicle_class column if not present
                if 'vehicle_class' not in header:
                    header.append('vehicle_class')

                rows.append(header)

                for row in reader:
                    # Add vehicle class to each row
                    if len(row) < len(header):
                        row.append(vehicle_data.get('vehicle_class', 'vehicle'))
                    rows.append(row)

            # Write updated data
            with open(segment_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

        except Exception as e:
            self.logger.error(f"Error updating segment results: {e}")

    def add_ground_truth_info(self, output_dir: str, vehicle_data: Dict):
        """Add ground truth information to results.

        Args:
            output_dir: Directory containing results
            vehicle_data: Vehicle information dictionary
        """
        gt_file = os.path.join(output_dir, 'ground_truth_info.csv')

        try:
            with open(gt_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['vehicle_id', 'timestamp_hms', 'ground_truth_speed_kmph', 'vehicle_class'])
                writer.writerow([
                    vehicle_data['vehicle_id'],
                    vehicle_data['timestamp_hms'],
                    vehicle_data['ground_truth_speed_kmph'],
                    vehicle_data.get('vehicle_class', 'vehicle')
                ])

        except Exception as e:
            self.logger.error(f"Error writing ground truth info: {e}")

    def process_all_vehicles(self, csv_path: str, video_path: str,
                             output_base_dir: str, window_before: float = 5.0,
                             window_after: float = 5.0):
        """Process all vehicles from the CSV file.

        Args:
            csv_path: Path to ground truth CSV
            video_path: Path to source video
            output_base_dir: Base directory for outputs
            window_before: Seconds before timestamp
            window_after: Seconds after timestamp
        """
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video information
        video_duration, _, fps = self.get_video_info(video_path)
        self.logger.info(f"Video duration: {video_duration:.1f}s, FPS: {fps}")

        # Read ground truth data
        vehicles = self.read_ground_truth_csv(csv_path)

        # Create output directories
        clips_dir = os.path.join(output_base_dir, 'clips')
        results_dir = os.path.join(output_base_dir, 'results')
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Process each vehicle
        successful = 0
        failed = 0

        for vehicle in vehicles:
            try:
                self.logger.info(f"\nProcessing {vehicle['vehicle_id']} at {vehicle['timestamp_hms']}")

                # Calculate clip boundaries
                center_time = vehicle['timestamp_seconds']
                start_time = max(0, center_time - window_before)
                end_time = min(video_duration, center_time + window_after)
                duration = end_time - start_time

                # Check if timestamp is valid
                if center_time > video_duration:
                    self.logger.error(f"Timestamp {vehicle['timestamp_hms']} exceeds video duration")
                    failed += 1
                    continue

                # Extract clip
                clip_filename = f"{vehicle['vehicle_id']}_{self.format_timestamp_for_filename(vehicle['timestamp_hms'])}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)

                # Try ffmpeg first, fallback to OpenCV
                if self.extract_clip_ffmpeg(video_path, clip_path, start_time, duration):
                    self.logger.info(f"Extracted clip: {clip_filename}")
                elif self.extract_clip_opencv(video_path, clip_path, start_time, duration, fps):
                    self.logger.info(f"Extracted clip (OpenCV): {clip_filename}")
                else:
                    self.logger.error(f"Failed to extract clip for {vehicle['vehicle_id']}")
                    failed += 1
                    continue

                # Process with tracking pipeline
                if self.process_vehicle_clip(clip_path, vehicle, results_dir):
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                self.logger.error(f"Error processing {vehicle['vehicle_id']}: {e}")
                failed += 1

        # Summary
        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"Processing complete!")
        self.logger.info(f"Successful: {successful}/{len(vehicles)}")
        self.logger.info(f"Failed: {failed}/{len(vehicles)}")
        self.logger.info(f"Clips saved to: {clips_dir}")
        self.logger.info(f"Results saved to: {results_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract and process ground truth vehicle clips"
    )

    parser.add_argument(
        "csv_path",
        help="Path to ground truth CSV file"
    )

    parser.add_argument(
        "--video_path",
        help="Path to source video (overrides env)",
        default=None
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory for clips and results",
        default="ground_truth_output"
    )

    parser.add_argument(
        "--window_before",
        type=float,
        default=5.0,
        help="Seconds before timestamp (default: 5.0)"
    )

    parser.add_argument(
        "--window_after",
        type=float,
        default=5.0,
        help="Seconds after timestamp (default: 5.0)"
    )

    parser.add_argument(
        "--env_file",
        help="Environment file to use (default: from config.py)",
        default=None
    )

    args = parser.parse_args()

    # Initialize configuration
    if args.env_file:
        config = Config(env_path=args.env_file)
    else:
        config = Config()  # Uses default from config.py

    # Override video path if provided
    if args.video_path:
        video_path = args.video_path
    else:
        video_path = config.VIDEO_PATH

    if not video_path:
        print("Error: No video path specified. Use --video_path or set VIDEO_PATH in environment.")
        sys.exit(1)

    # Create extractor and process
    extractor = GroundTruthExtractor(config)

    try:
        extractor.process_all_vehicles(
            args.csv_path,
            video_path,
            args.output_dir,
            args.window_before,
            args.window_after
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()