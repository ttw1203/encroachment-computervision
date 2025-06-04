"""Main script for running the vehicle tracking and analysis pipeline with enhanced stability."""
import argparse
import logging
from collections import defaultdict, deque
from tqdm import tqdm
import time as tm
import cv2
import numpy as np
import supervision as sv
import math
from typing import Dict, Set, Optional
from dataclasses import dataclass

# Import our modules
from src.config import Config
from src.detection_and_tracking import DetectionTracker, filter_rider_persons
from src.kalman_filter import KalmanFilterManager
from src.zone_management import ZoneManager
from src.geometry_and_transforms import ViewTransformer, line_side
from src.annotators import AnnotationManager
from src.event_processing import EventProcessor
from src.io_utils import IOManager


@dataclass
class VehicleState:
    """Tracks the state of a vehicle in the double-line counting system."""
    tracker_id: int
    class_name: str
    line_a_cross_time: Optional[float] = None
    line_b_cross_time: Optional[float] = None
    line_a_cross_direction: Optional[str] = None
    line_b_cross_direction: Optional[str] = None
    last_position: Optional[tuple] = None

    def reset(self):
        """Reset the vehicle state for re-counting."""
        self.line_a_cross_time = None
        self.line_b_cross_time = None
        self.line_a_cross_direction = None
        self.line_b_cross_direction = None


class DoubleLineVehicleCounter:
    """Manages double-line vehicle counting with A→B and B→A sequences."""

    def __init__(self, video_fps: float, time_window_seconds: float = 30.0):
        """Initialize the double-line counter.

        Args:
            video_fps: Video frame rate
            time_window_seconds: Maximum time allowed between line crossings
        """
        self.video_fps = video_fps
        self.time_window_seconds = time_window_seconds

        # Vehicle state tracking
        self.vehicle_states: Dict[int, VehicleState] = {}

        # Counting results by direction sequence and vehicle class
        self.count_data = {
            "a_to_b": defaultdict(lambda: {
                "incoming": 0, "outgoing": 0,
                "total_speed_incoming": 0.0, "count_for_speed_incoming": 0,
                "total_speed_outgoing": 0.0, "count_for_speed_outgoing": 0
            }),
            "b_to_a": defaultdict(lambda: {
                "incoming": 0, "outgoing": 0,
                "total_speed_incoming": 0.0, "count_for_speed_incoming": 0,
                "total_speed_outgoing": 0.0, "count_for_speed_outgoing": 0
            })
        }

        # Visual feedback tracking
        self.line_a_last_cross_time = -float('inf')
        self.line_b_last_cross_time = -float('inf')

    def update_vehicle_position(self, tracker_id: int, class_name: str,
                               current_position: tuple, current_time: float,
                               line_a_coords: np.ndarray, line_b_coords: np.ndarray,
                               current_speed: float, min_speed_threshold: float) -> bool:
        """Update vehicle position and check for line crossings.

        Args:
            tracker_id: Vehicle tracker ID
            class_name: Vehicle class name
            current_position: Current (x, y) position
            current_time: Current time in seconds
            line_a_coords: Line A coordinates
            line_b_coords: Line B coordinates
            current_speed: Current vehicle speed in m/s
            min_speed_threshold: Minimum speed threshold

        Returns:
            True if a counting event occurred
        """
        # Get or create vehicle state
        if tracker_id not in self.vehicle_states:
            self.vehicle_states[tracker_id] = VehicleState(tracker_id, class_name)

        vehicle_state = self.vehicle_states[tracker_id]
        vehicle_state.class_name = class_name  # Update in case it changed

        # Initialize crossing flag
        crossing_occurred = False

        # Check for line crossings if we have a previous position
        if vehicle_state.last_position is not None:
            # Check Line A crossing
            if self._check_line_crossing(vehicle_state.last_position, current_position, line_a_coords):
                direction = self._get_crossing_direction(vehicle_state.last_position, current_position)
                if direction:
                    vehicle_state.line_a_cross_time = current_time
                    vehicle_state.line_a_cross_direction = direction
                    self.line_a_last_cross_time = current_time
                    crossing_occurred = True
                    print(f"[DOUBLE-LINE] Vehicle {tracker_id} ({class_name}) crossed Line A: {direction}")

            # Check Line B crossing
            if self._check_line_crossing(vehicle_state.last_position, current_position, line_b_coords):
                direction = self._get_crossing_direction(vehicle_state.last_position, current_position)
                if direction:
                    vehicle_state.line_b_cross_time = current_time
                    vehicle_state.line_b_cross_direction = direction
                    self.line_b_last_cross_time = current_time
                    crossing_occurred = True
                    print(f"[DOUBLE-LINE] Vehicle {tracker_id} ({class_name}) crossed Line B: {direction}")

            # Check for completed sequences
            if self._check_completed_sequences(vehicle_state, current_time, current_speed, min_speed_threshold):
                crossing_occurred = True

        # Update position for next frame
        vehicle_state.last_position = current_position

        return crossing_occurred

    def _check_line_crossing(self, prev_pos: tuple, curr_pos: tuple, line_coords: np.ndarray) -> bool:
        """Check if vehicle crossed a line using change of side method."""
        if len(line_coords) != 2:
            return False

        L1, L2 = line_coords[0], line_coords[1]
        prev_side = line_side(prev_pos, L1, L2)
        curr_side = line_side(curr_pos, L1, L2)

        # Crossing occurred if both sides are non-zero and different
        return (prev_side != 0 and curr_side != 0 and prev_side != curr_side)

    def _get_crossing_direction(self, prev_pos: tuple, curr_pos: tuple) -> Optional[str]:
        """Determine crossing direction based on y-coordinate movement."""
        if curr_pos[1] > prev_pos[1]:
            return "incoming"  # Moving down (y increases)
        elif curr_pos[1] < prev_pos[1]:
            return "outgoing"  # Moving up (y decreases)
        return None

    def _check_completed_sequences(self, vehicle_state: VehicleState, current_time: float,
                                  current_speed: float, min_speed_threshold: float) -> bool:
        """Check for completed A→B or B→A sequences and update counts."""
        if (vehicle_state.line_a_cross_time is None or
            vehicle_state.line_b_cross_time is None):
            return False

        time_diff = abs(vehicle_state.line_a_cross_time - vehicle_state.line_b_cross_time)

        # Check if crossings are within time window
        if time_diff > self.time_window_seconds:
            return False

        sequence_type = None
        direction = None

        # Determine sequence type and direction
        if (vehicle_state.line_a_cross_time < vehicle_state.line_b_cross_time and
            vehicle_state.line_a_cross_direction == vehicle_state.line_b_cross_direction):
            # A→B sequence
            sequence_type = "a_to_b"
            direction = vehicle_state.line_a_cross_direction

        elif (vehicle_state.line_b_cross_time < vehicle_state.line_a_cross_time and
              vehicle_state.line_a_cross_direction == vehicle_state.line_b_cross_direction):
            # B→A sequence
            sequence_type = "b_to_a"
            direction = vehicle_state.line_b_cross_direction

        if sequence_type and direction:
            # Update count
            self.count_data[sequence_type][vehicle_state.class_name][direction] += 1

            # Update speed tracking if vehicle has meaningful speed
            if current_speed >= min_speed_threshold:
                speed_kmh = current_speed * 3.6
                self.count_data[sequence_type][vehicle_state.class_name][f'total_speed_{direction}'] += speed_kmh
                self.count_data[sequence_type][vehicle_state.class_name][f'count_for_speed_{direction}'] += 1

            print(f"[DOUBLE-LINE] Vehicle {vehicle_state.tracker_id} ({vehicle_state.class_name}) "
                  f"completed {sequence_type.upper()} sequence: {direction}")

            # Reset vehicle state for potential re-counting
            vehicle_state.reset()
            return True

        return False

    def cleanup_old_states(self, active_tracker_ids: Set[int], current_time: float):
        """Remove states for inactive vehicles and expired crossings."""
        to_remove = []

        for tracker_id, vehicle_state in self.vehicle_states.items():
            # Remove if vehicle is no longer active
            if tracker_id not in active_tracker_ids:
                to_remove.append(tracker_id)
                continue

            # Remove if single line crossing is too old
            if (vehicle_state.line_a_cross_time is not None and
                current_time - vehicle_state.line_a_cross_time > self.time_window_seconds):
                vehicle_state.line_a_cross_time = None
                vehicle_state.line_a_cross_direction = None

            if (vehicle_state.line_b_cross_time is not None and
                current_time - vehicle_state.line_b_cross_time > self.time_window_seconds):
                vehicle_state.line_b_cross_time = None
                vehicle_state.line_b_cross_direction = None

        for tracker_id in to_remove:
            del self.vehicle_states[tracker_id]

    def get_total_counts(self) -> Dict[str, int]:
        """Get total counts across all vehicle classes."""
        totals = {
            "a_to_b_incoming": 0, "a_to_b_outgoing": 0,
            "b_to_a_incoming": 0, "b_to_a_outgoing": 0
        }

        for sequence_type, class_data in self.count_data.items():
            for class_name, counts in class_data.items():
                totals[f"{sequence_type}_incoming"] += counts["incoming"]
                totals[f"{sequence_type}_outgoing"] += counts["outgoing"]

        return totals

    def get_processed_counts_for_export(self) -> list:
        """Get processed counting data for CSV export."""
        processed_counts = []

        # Process each vehicle class
        all_classes = set()
        for sequence_data in self.count_data.values():
            all_classes.update(sequence_data.keys())

        for class_name in all_classes:
            a_to_b_data = self.count_data["a_to_b"][class_name]
            b_to_a_data = self.count_data["b_to_a"][class_name]

            # Calculate average speeds
            def calc_avg_speed(total_speed, count):
                return total_speed / count if count > 0 else 0.0

            processed_counts.append({
                "vehicle_class": class_name,
                "a_to_b_incoming": a_to_b_data["incoming"],
                "a_to_b_outgoing": a_to_b_data["outgoing"],
                "b_to_a_incoming": b_to_a_data["incoming"],
                "b_to_a_outgoing": b_to_a_data["outgoing"],
                "avg_speed_a_to_b_incoming_kmh": round(calc_avg_speed(
                    a_to_b_data["total_speed_incoming"], a_to_b_data["count_for_speed_incoming"]), 2),
                "avg_speed_a_to_b_outgoing_kmh": round(calc_avg_speed(
                    a_to_b_data["total_speed_outgoing"], a_to_b_data["count_for_speed_outgoing"]), 2),
                "avg_speed_b_to_a_incoming_kmh": round(calc_avg_speed(
                    b_to_a_data["total_speed_incoming"], b_to_a_data["count_for_speed_incoming"]), 2),
                "avg_speed_b_to_a_outgoing_kmh": round(calc_avg_speed(
                    b_to_a_data["total_speed_outgoing"], b_to_a_data["count_for_speed_outgoing"]), 2)
            })

        # Add summary row if there are any vehicles
        if processed_counts:
            totals = self.get_total_counts()

            # Calculate overall average speeds
            def calc_overall_avg(sequence_type, direction):
                total_speed = sum(data[f"total_speed_{direction}"]
                                for data in self.count_data[sequence_type].values())
                total_count = sum(data[f"count_for_speed_{direction}"]
                                for data in self.count_data[sequence_type].values())
                return total_speed / total_count if total_count > 0 else 0.0

            processed_counts.append({
                "vehicle_class": "TOTAL",
                "a_to_b_incoming": totals["a_to_b_incoming"],
                "a_to_b_outgoing": totals["a_to_b_outgoing"],
                "b_to_a_incoming": totals["b_to_a_incoming"],
                "b_to_a_outgoing": totals["b_to_a_outgoing"],
                "avg_speed_a_to_b_incoming_kmh": round(calc_overall_avg("a_to_b", "incoming"), 2),
                "avg_speed_a_to_b_outgoing_kmh": round(calc_overall_avg("a_to_b", "outgoing"), 2),
                "avg_speed_b_to_a_incoming_kmh": round(calc_overall_avg("b_to_a", "incoming"), 2),
                "avg_speed_b_to_a_outgoing_kmh": round(calc_overall_avg("b_to_a", "outgoing"), 2)
            })

        return processed_counts


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )

    # Initialize config to get defaults
    config = Config()

    parser.add_argument(
        "--no_blend_zones",
        action="store_true",
        help="Disable the translucent curb-lane overlays drawn by blend_zone()"
    )
    parser.add_argument(
        "--initial_velocity_frames",
        default=2,
        type=int,
        choices=[2, 3],
        help="Number of frames to use for initial velocity calculation (default: 2)"
    )
    parser.add_argument(
        "--source_video_path",
        required=False,
        help="Path to the source video file",
        type=str,
        default=config.VIDEO_PATH
    )
    parser.add_argument(
        "--target_video_path",
        required=False,
        help="Path to the target video file (output)",
        type=str,
        default=config.OUTPUT_PATH
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.6,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.4,
        help="IOU threshold for the model",
        type=float
    )
    parser.add_argument(
        "--num_future_predictions",
        default=config.DEFAULT_NUM_FUTURE_PREDICTIONS,
        help="Number of future points to predict per vehicle",
        type=int
    )
    parser.add_argument(
        "--future_prediction_interval",
        default=config.DEFAULT_FUTURE_PREDICTION_INTERVAL,
        help="Time interval (seconds) between future predictions",
        type=float
    )
    parser.add_argument(
        "--ttc_threshold",
        default=config.DEFAULT_TTC_THRESHOLD,
        help="Only show TTC if it's ≤ this value (in seconds)",
        type=float
    )
    parser.add_argument(
        "--zones_file",
        default=config.ENC_ZONE_CONFIG,
        help="YAML/JSON file with curb-lane polygons"
    )
    parser.add_argument(
        "--dump_zones_png",
        help="Write a PNG showing lane polygons over the first video frame, then exit"
    )
    parser.add_argument(
        "--tracker",
        choices=["strongsort", "bytetrack"],
        default="strongsort",
        help="Tracking backend to use (default: strongsort)",
    )
    parser.add_argument(
        "--segment_speed",
        action="store_true",
        help="Enable segment-based speed calibration (default: off)"
    )
    parser.add_argument(
        "--display_basic_info",
        action="store_true",
        help="Display basic info (ID, Class, Confidence) instead of speed/TTC"
    )
    parser.add_argument(
        "--detector_model",
        choices=["yolo", "rf_detr"],
        default="yolo",
        help="Object detection model to use (default: yolo)"
    )
    parser.add_argument(
        "--advanced_counting",
        action="store_true",
        help="Enable advanced double-line vehicle counting (default: off)"
    )
    # New stability parameters
    parser.add_argument(
        "--speed_smoothing_window",
        default=5,
        type=int,
        help="Number of frames to use for speed smoothing (default: 5)"
    )
    parser.add_argument(
        "--max_acceleration",
        default=5.0,
        type=float,
        help="Maximum allowed acceleration in m/s² (default: 5.0)"
    )
    parser.add_argument(
        "--min_speed_threshold",
        default=0.1,
        type=float,
        help="Minimum speed threshold in m/s below which velocity is set to 0 (default: 0.1)"
    )

    return parser.parse_args()


def validate_detection_consistency(detections: sv.Detections, previous_detections: dict[int, np.ndarray],
                                 max_pixel_jump: float = 300.0) -> sv.Detections:
    """Validate detections to prevent ID switches and large position jumps."""
    if len(detections) == 0:
        return detections

    valid_mask = np.ones(len(detections), dtype=bool)

    for idx, tracker_id in enumerate(detections.tracker_id):
        if tracker_id in previous_detections:
            # Check for unrealistic position jumps
            current_center = detections.get_anchors_coordinates(sv.Position.CENTER)[idx]
            prev_center = previous_detections[tracker_id]

            distance = np.linalg.norm(current_center - prev_center)

            if distance > max_pixel_jump:
                # Mark as invalid - likely an ID switch or detection error
                valid_mask[idx] = False
                logging.warning(f"Tracker {tracker_id}: Large position jump detected ({distance:.1f} pixels)")

    return detections[valid_mask]


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Initialize configuration
    config = Config()

    # Check advanced counting configuration
    advanced_counting_enabled = (args.advanced_counting or
                               config.ENABLE_ADVANCED_VEHICLE_COUNTING)

    if advanced_counting_enabled:
        # Check if both counting lines are available
        if (config.COUNTING_LINE_A_COORDS is None or
            config.COUNTING_LINE_B_COORDS is None):
            print("[ERROR] Advanced counting enabled but counting lines not properly configured.")
            print("Please check counting_line_a and counting_line_b in your zone configuration file.")
            return

        print("[INFO] Advanced double-line vehicle counting enabled")
        print(f"[INFO] Line A: {config.COUNTING_LINE_A_COORDS}")
        print(f"[INFO] Line B: {config.COUNTING_LINE_B_COORDS}")
    else:
        print("[INFO] Advanced vehicle counting disabled")

    # Validate RF-DETR configuration if selected
    if args.detector_model == "rf_detr":
        try:
            config.validate_rf_detr_config()
            print(f"[RF-DETR] Using {config.RF_DETR_MODEL_TYPE} model with {config.RF_DETR_VARIANT} variant")
            if config.RF_DETR_MODEL_TYPE == "custom":
                print(f"[RF-DETR] Custom model: {config.RF_DETR_MODEL_PATH}")
                print(f"[RF-DETR] Custom classes: {config.RF_DETR_CUSTOM_CLASSES_PATH}")
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] RF-DETR configuration error: {e}")
            return

    # Set up logging
    logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

    # Define master calibration function based on configuration
    model_type = config.SPEED_CALIBRATION_MODEL_TYPE.lower()

    if model_type in ["linear", "ransac_linear"]:
        def calibration_func(raw_speed_m_s: float) -> float:
            return config.SPEED_CALIBRATION_MODEL_A * raw_speed_m_s + config.SPEED_CALIBRATION_MODEL_B

    elif model_type == "poly2":
        # Parse polynomial coefficients
        try:
            coeffs = [float(x.strip()) for x in config.SPEED_CALIBRATION_POLY_COEFFS.split(',')]
            if len(coeffs) != 3:
                logging.warning(f"Invalid number of coefficients for poly2 model. Expected 3, got {len(coeffs)}. Using no calibration.")
                calibration_func = lambda x: x
            else:
                def calibration_func(raw_speed_m_s: float) -> float:
                    return coeffs[0] + coeffs[1] * raw_speed_m_s + coeffs[2] * (raw_speed_m_s**2)
        except Exception as e:
            logging.warning(f"Error parsing poly2 coefficients: {e}. Using no calibration.")
            calibration_func = lambda x: x

    elif model_type == "poly3":
        # Parse polynomial coefficients
        try:
            coeffs = [float(x.strip()) for x in config.SPEED_CALIBRATION_POLY_COEFFS.split(',')]
            if len(coeffs) != 4:
                logging.warning(f"Invalid number of coefficients for poly3 model. Expected 4, got {len(coeffs)}. Using no calibration.")
                calibration_func = lambda x: x
            else:
                def calibration_func(raw_speed_m_s: float) -> float:
                    return coeffs[0] + coeffs[1] * raw_speed_m_s + coeffs[2] * (raw_speed_m_s**2) + coeffs[3] * (raw_speed_m_s**3)
        except Exception as e:
            logging.warning(f"Error parsing poly3 coefficients: {e}. Using no calibration.")
            calibration_func = lambda x: x

    else:
        # Default: no calibration
        logging.warning(f"Unknown calibration model type: {model_type}. Using no calibration.")
        calibration_func = lambda x: x

    # Load zone configurations
    LEFT_ZONE_POLY, RIGHT_ZONE_POLY = Config.load_zones(args.zones_file)

    # Handle zone preview if requested
    if args.dump_zones_png:
        IOManager.dump_zones_png(
            args.source_video_path,
            args.dump_zones_png,
            LEFT_ZONE_POLY,
            RIGHT_ZONE_POLY
        )
        return

    # Load segment configurations if needed
    if args.segment_speed:
        ENTRY_LINE, EXIT_LINE = Config.load_segments(args.zones_file)

    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    # Calculate clip frames
    clip_frames = min(
        int(video_info.fps * config.CLIP_SECONDS),
        video_info.total_frames
    )

    # Initialize components
    SOURCE, TARGET = config.get_source_target_points()

    # Detection and tracking
    detector_tracker = DetectionTracker(
        model_weights=config.MODEL_WEIGHTS,
        device=config.DEVICE,
        tracker_type=args.tracker,
        confidence_threshold=args.confidence_threshold,
        video_fps=video_info.fps,
        detector_model=args.detector_model,
        rf_detr_config=config.get_rf_detr_config() if args.detector_model == "rf_detr" else None
    )

    # Kalman filter manager with stability parameters
    kf_manager = KalmanFilterManager(
        speed_smoothing_window=args.speed_smoothing_window,
        max_acceleration=args.max_acceleration,
        min_speed_threshold=args.min_speed_threshold,
        initial_velocity_frames=args.initial_velocity_frames
    )

    # Zone management
    zone_manager = ZoneManager(
        LEFT_ZONE_POLY,
        RIGHT_ZONE_POLY,
        video_info.fps,
        config.ENCROACH_SECS,
        config.MOVE_THRESH_METRES
    )

    # Create masks if needed
    if not args.no_blend_zones:
        zone_manager.create_masks((video_info.resolution_wh[1], video_info.resolution_wh[0]))

    # Geometry and transforms
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Annotation manager
    annotation_manager = AnnotationManager(
        thickness=2,
        text_scale=1,
        trace_length_seconds=3.0,
        video_fps=video_info.fps
    )

    # Event processor with calibration function
    event_processor = EventProcessor(
        video_fps=video_info.fps,
        collision_distance=config.COLLISION_DISTANCE,
        calibration_func=calibration_func
    )

    # IO manager using environment variable
    io_manager = IOManager(output_dir=config.RESULTS_OUTPUT_DIR)

    # Initialize double-line vehicle counter if enabled
    double_line_counter = None
    if advanced_counting_enabled:
        double_line_counter = DoubleLineVehicleCounter(
            video_fps=video_info.fps,
            time_window_seconds=30.0
        )

    # State variables
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    future_coordinates = defaultdict(list)
    csv_rows = []
    ttc_labels = defaultdict(list)
    ttc_event_count = 0

    last_seen_frame = {}
    previous_detections = {}  # Store previous detection positions
    MAX_AGE_FRAMES = int(video_info.fps * config.MAX_AGE_SECONDS)

    # Progress bar
    bar = tqdm(total=clip_frames, desc="Processing video", unit="frame")
    t0 = tm.time()

    # Main processing loop
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            if frame_idx >= clip_frames:
                break

            bar.update(1)
            ttc_labels.clear()

            # Calculate current frame time for visual feedback
            current_frame_time_sec = frame_idx / video_info.fps

            # Detect and track
            detections = detector_tracker.detect_and_track(
                frame,
                polygon_zone,
                args.iou_threshold
            )

            # Filter riders if needed
            detections = filter_rider_persons(detections, iou_thr=0.30)

            # Validate detection consistency
            detections = validate_detection_consistency(detections, previous_detections)

            # Update previous detections
            if len(detections) > 0:
                centers = detections.get_anchors_coordinates(sv.Position.CENTER)
                for idx, tid in enumerate(detections.tracker_id):
                    previous_detections[tid] = centers[idx]

            # Check encroachment
            new_enc_events = zone_manager.check_encroachment(
                detections,
                frame_idx,
                kf_manager.get_all_states()
            )

            # Update encroachment events with class names
            for event in new_enc_events:
                event['class_name'] = detector_tracker.get_class_name(event['class_id'])

            # Draw zones
            frame = zone_manager.draw_zones(frame, not args.no_blend_zones)

            # Update tracking state
            active_ids = set(detections.tracker_id.tolist() if len(detections) else [])

            # Update last seen frame
            for tid in active_ids:
                last_seen_frame[tid] = frame_idx

            # Process each detection
            points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            points = view_transformer.transform_points(points=points).astype(np.float32)

            # Double-line vehicle counting logic
            crossing_detected = False

            for det_idx, (tracker_id, [x, y]) in enumerate(zip(detections.tracker_id, points)):
                # Update Kalman filter
                dt = 1 / video_info.fps
                kf = kf_manager.update_or_create(tracker_id, x, y, dt, frame_idx)

                # Apply speed calibration with stability
                kf_manager.apply_speed_calibration(tracker_id, calibration_func)

                # Get current state (now with calibrated and stabilized velocity)
                Xf, Yf, _, _ = kf.statePost.flatten()

                # Get smoothed velocity for display
                Vx_smooth, Vy_smooth = kf_manager.get_smoothed_velocity(tracker_id)

                # Double-line vehicle counting logic
                if advanced_counting_enabled:
                    # Get current representative point in pixel coordinates (bottom-center of bounding box)
                    bbox = detections.xyxy[det_idx]
                    current_point_px = (
                        (bbox[0] + bbox[2]) / 2,  # center x
                        bbox[3]  # bottom y
                    )

                    # Get vehicle class
                    class_id = int(detections.class_id[det_idx])
                    class_name = detector_tracker.get_class_name(class_id)

                    # Get current speed
                    current_speed = math.hypot(Vx_smooth, Vy_smooth)

                    # Update double-line counter
                    if double_line_counter.update_vehicle_position(
                        tracker_id, class_name, current_point_px, current_frame_time_sec,
                        config.COUNTING_LINE_A_COORDS, config.COUNTING_LINE_B_COORDS,
                        current_speed, args.min_speed_threshold
                    ):
                        crossing_detected = True

                # Process segment speed if enabled
                if args.segment_speed:
                    bbox = detections.xyxy[det_idx].astype(float)
                    p_cur = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

                    event_processor.process_segment_speed(
                        tracker_id, p_cur, (Xf, Yf), frame_idx,
                        ENTRY_LINE, EXIT_LINE
                    )

                # Store coordinates for visualization
                coordinates[tracker_id].append((x, y))

                # Update class mapping
                class_id = int(detections.class_id[det_idx])
                class_name = detector_tracker.get_class_name(class_id)
                event_processor.id_to_class[tracker_id] = class_name

                # Calculate TTC using smoothed velocities
                ttc_event = event_processor.calculate_ttc(
                    tracker_id,
                    kf_manager.get_all_states(),
                    last_seen_frame,
                    frame_idx,
                    MAX_AGE_FRAMES,
                    args.ttc_threshold
                )

                if ttc_event:
                    follower_class = event_processor.id_to_class.get(tracker_id, "unknown")
                    leader_class = event_processor.id_to_class.get(ttc_event['other_id'], "unknown")

                    event_processor.ttc_rows.append([
                        frame_idx,
                        tracker_id, follower_class,
                        ttc_event['other_id'], leader_class,
                        round(ttc_event['d_closest'], 2),
                        round(ttc_event['rel_speed'], 2),
                        round(ttc_event['t_star'], 2)
                    ])

                    ttc_labels[tracker_id] = [f"TTC->#{ttc_event['other_id']}:{ttc_event['t_star']:.1f}s"]
                    ttc_event_count += 1

                # Predict future positions using smoothed velocity
                future_positions = kf_manager.predict_future_positions(
                    tracker_id,
                    args.future_prediction_interval,
                    args.num_future_predictions
                )

                # Transform to pixel coordinates
                if future_positions:
                    future_positions_array = np.array(future_positions, dtype=np.float32)
                    predicted_pixels = view_transformer.inverse_transform_points(future_positions_array)
                    future_coordinates[tracker_id] = predicted_pixels.tolist()

            # Signal line crossings for visual feedback if advanced counting enabled
            if advanced_counting_enabled and crossing_detected:
                # Update annotation manager with crossing times
                annotation_manager.signal_line_a_cross(double_line_counter.line_a_last_cross_time)
                annotation_manager.signal_line_b_cross(double_line_counter.line_b_last_cross_time)

            # Clean up old tracks
            to_remove = []
            for tid in list(kf_manager.get_all_states().keys()):
                last_seen = last_seen_frame.get(tid, None)
                if last_seen is None or (frame_idx - last_seen) > MAX_AGE_FRAMES:
                    to_remove.append(tid)

            for tid in to_remove:
                kf_manager.remove_tracker(tid)
                future_coordinates.pop(tid, None)
                last_seen_frame.pop(tid, None)
                previous_detections.pop(tid, None)

            # Clean up double-line counter state if enabled
            if advanced_counting_enabled:
                double_line_counter.cleanup_old_states(active_ids, current_frame_time_sec)

            # Generate labels and save metrics using smoothed speeds
            labels = []

            for det_idx, tracker_id in enumerate(detections.tracker_id):
                class_id = int(detections.class_id[det_idx])
                class_name = detector_tracker.get_class_name(class_id)
                confidence = float(detections.confidence[det_idx])

                if args.display_basic_info:
                    # Basic info display mode
                    label = f"ID:{tracker_id} {class_name} ({confidence:.2f})"
                else:
                    # Speed and TTC display mode
                    if tracker_id in kf_manager.get_all_states():
                        vx_smooth, vy_smooth = kf_manager.get_smoothed_velocity(tracker_id)
                        speed_ms = math.hypot(vx_smooth, vy_smooth)

                        # Only save metrics if speed is meaningful
                        if speed_ms >= args.min_speed_threshold:
                            csv_rows.append([
                                frame_idx,
                                int(tracker_id),
                                class_name,
                                confidence,
                                round(speed_ms * 3.6, 2)  # Convert to km/h
                            ])

                        # Create speed label
                        if speed_ms < args.min_speed_threshold:
                            label = f"#{tracker_id} 0 km/h"
                        else:
                            label = f"#{tracker_id} {int(speed_ms * 3.6)} km/h"

                        # Add TTC info if available
                        if tracker_id in ttc_labels:
                            label += " | " + ttc_labels[tracker_id][0]
                    else:
                        # Fallback for trackers without Kalman state
                        label = f"#{tracker_id}"

                labels.append(label)

            # Draw segment lines if enabled
            if args.segment_speed:
                frame = annotation_manager.draw_segment_lines(frame, ENTRY_LINE, EXIT_LINE)

            # Draw counting lines if advanced counting enabled
            if advanced_counting_enabled:
                frame = annotation_manager.draw_counting_lines(
                    frame,
                    config.COUNTING_LINE_A_COORDS,
                    config.COUNTING_LINE_B_COORDS,
                    current_frame_time_sec
                )

            # Annotate frame
            annotated_frame = annotation_manager.annotate_frame(
                frame, detections, labels, future_coordinates, active_ids
            )

            # Draw live double-line vehicle counter if advanced counting enabled
            if advanced_counting_enabled:
                totals = double_line_counter.get_total_counts()
                annotated_frame = annotation_manager.draw_live_double_line_counts(
                    annotated_frame,
                    totals["a_to_b_incoming"],
                    totals["a_to_b_outgoing"],
                    totals["b_to_a_incoming"],
                    totals["b_to_a_outgoing"]
                )

            # Write frame
            sink.write_frame(annotated_frame)

            # Display if enabled
            if config.DISPLAY:
                display_frame = cv2.resize(annotated_frame, (1920, 1080))
                cv2.imshow("Preview", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if config.DISPLAY:
            cv2.destroyAllWindows()

    # Save results
    bar.close()
    print(f"Total TTC events logged: {ttc_event_count}")

    # Save all CSV files
    io_manager.save_vehicle_metrics(csv_rows)
    io_manager.save_ttc_events(event_processor.ttc_rows)
    io_manager.save_encroachment_events(zone_manager.get_encroachment_events())

    if args.segment_speed and event_processor.segment_results:
        io_manager.save_segment_speeds(event_processor.segment_results)

    # Save double-line vehicle counting results if enabled
    if advanced_counting_enabled:
        processed_counts = double_line_counter.get_processed_counts_for_export()
        io_manager.save_double_line_vehicle_counts(processed_counts)

        totals = double_line_counter.get_total_counts()
        print(f"Double-line counting results saved:")
        print(f"  A→B: {totals['a_to_b_incoming']} incoming, {totals['a_to_b_outgoing']} outgoing")
        print(f"  B→A: {totals['b_to_a_incoming']} incoming, {totals['b_to_a_outgoing']} outgoing")

    # Print summary
    elapsed = tm.time() - t0
    fps = frame_idx / elapsed
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({fps:.2f} FPS)")


if __name__ == "__main__":
    main()