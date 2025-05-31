"""Main script for running the vehicle tracking and analysis pipeline."""
import argparse
import logging
from collections import defaultdict, deque
from tqdm import tqdm
import time as tm
import cv2
import numpy as np
import supervision as sv
import math

# Import our modules
from src.config import Config
from src.detection_and_tracking import DetectionTracker, filter_rider_persons
from src.kalman_filter import KalmanFilterManager
from src.zone_management import ZoneManager
from src.geometry_and_transforms import ViewTransformer
from src.annotators import AnnotationManager
from src.event_processing import EventProcessor
from src.io_utils import IOManager


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
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
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

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Initialize configuration
    config = Config()

    # Set up logging
    logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

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
        video_fps=video_info.fps
    )

    # Kalman filter manager
    kf_manager = KalmanFilterManager()

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

    # Event processor
    event_processor = EventProcessor(
        video_fps=video_info.fps,
        collision_distance=config.COLLISION_DISTANCE
    )

    # IO manager
    io_manager = IOManager()

    # IO manager using environment variable
    io_manager = IOManager(output_dir=config.RESULTS_OUTPUT_DIR)

    # State variables
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    future_coordinates = defaultdict(list)
    csv_rows = []
    ttc_labels = defaultdict(list)
    ttc_event_count = 0

    last_seen_frame = {}
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

            # Detect and track
            detections = detector_tracker.detect_and_track(
                frame,
                polygon_zone,
                args.iou_threshold
            )

            # Filter riders if needed
            # detections = filter_rider_persons(detections, iou_thr=0.50)

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

            for det_idx, (tracker_id, [x, y]) in enumerate(zip(detections.tracker_id, points)):
                # Update Kalman filter
                dt = 1 / video_info.fps
                kf = kf_manager.update_or_create(tracker_id, x, y, dt)

                # Get current state
                Xf, Yf, Vx, Vy = kf.statePost.flatten()

                # Process segment speed if enabled
                if args.segment_speed:
                    bbox = detections.xyxy[det_idx].astype(float)
                    p_cur = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

                    event_processor.process_segment_speed(
                        tracker_id, p_cur, (Xf, Yf), frame_idx,
                        ENTRY_LINE, EXIT_LINE
                    )

                # Calculate speed
                coordinates[tracker_id].append((x, y))

                if len(coordinates[tracker_id]) >= video_info.fps / 2:
                    (x_start, y_start) = coordinates[tracker_id][0]
                    (x_end, y_end) = coordinates[tracker_id][-1]

                    dx = x_end - x_start
                    dy = y_end - y_start
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time  # meters per second

                    class_id = int(detections.class_id[det_idx])
                    class_name = detector_tracker.get_class_name(class_id)
                    conf = float(detections.confidence[det_idx])

                    # Update class mapping
                    event_processor.id_to_class[tracker_id] = class_name

                    # Save metrics
                    csv_rows.append([
                        frame_idx,
                        int(tracker_id),
                        class_name,
                        conf,
                        round(speed * 3.6, 2)
                    ])

                # Calculate TTC
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

                # Predict future positions
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

            # Generate labels
            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    (x_start, y_start) = coordinates[tracker_id][0]
                    (x_end, y_end) = coordinates[tracker_id][-1]
                    dx = x_end - x_start
                    dy = y_end - y_start
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time

                    label = f"#{tracker_id} {int(speed * 3.6)} km/h"
                    if tracker_id in ttc_labels:
                        label += " | " + ttc_labels[tracker_id][0]
                    labels.append(label)

            # Draw segment lines if enabled
            if args.segment_speed:
                frame = annotation_manager.draw_segment_lines(frame, ENTRY_LINE, EXIT_LINE)

            # Annotate frame
            annotated_frame = annotation_manager.annotate_frame(
                frame, detections, labels, future_coordinates, active_ids
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

    # Print summary
    elapsed = tm.time() - t0
    fps = frame_idx / elapsed
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({fps:.2f} FPS)")


if __name__ == "__main__":
    main()