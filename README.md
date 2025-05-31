# Encroachment Computer Vision System

A comprehensive computer vision system for real-time vehicle tracking, speed estimation, encroachment detection, and collision prediction using YOLO object detection and advanced tracking algorithms.

## Features

- **Multi-Object Tracking**: Supports both StrongSORT and ByteTrack algorithms  
- **Speed Estimation**: Calculate vehicle speeds using perspective transformation  
- **Encroachment Detection**: Monitor vehicles entering and dwelling in restricted zones  
- **Collision Prediction**: Time-to-Collision (TTC) calculation between vehicles  
- **Kalman Filtering**: Smooth tracking and future position prediction  
- **Zone Management**: Define and monitor custom polygonal zones  
- **Segment-Based Speed**: Calibrated speed measurement between entry/exit lines  
- **Visual Annotations**: Real-time visualization with tracks, speeds, and predictions  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/encroachment-computervision.git
cd encroachment-computervision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights:
- YOLO weights (default: `yolov8l.pt`)  
- ReID weights for StrongSORT: `mobilenetv2_x1_4_dukemtmcreid.pt`

## Configuration

### Environment Variables

Create a `.env.bns` file in the project root:

```env
# Video paths
VIDEO_PATH=path/to/input/video.mp4
OUTPUT_PATH=path/to/output/video.mp4
ZONE_CHECK_PNG=path/to/zone_preview.png

# Model configuration
MODEL_WEIGHTS=yolov8l.pt
DEVICE=cuda  # or cpu

# Zone configuration
ENC_ZONE_CONFIG=zones_bns.yaml

# Output directory for CSV results
RESULTS_OUTPUT_DIR=D:/analysis/results
```

### Zone Configuration

Define detection zones in a YAML file (e.g., `zones_bns.yaml`):

```yaml
# Polygonal zones (pixel coordinates)
left_zone:
  - [100, 200]
  - [150, 200]
  - [150, 400]
  - [100, 400]

right_zone:
  - [500, 200]
  - [550, 200]
  - [550, 400]
  - [500, 400]

# Optional: Segment lines for speed measurement
segment_entry:
  - [0, 300]
  - [800, 300]

segment_exit:
  - [0, 500]
  - [800, 500]
```

## Usage

### Basic Usage

```bash
python main.py
```

### Command Line Options

```bash
# Use ByteTrack instead of StrongSORT
python main.py --tracker bytetrack

# Disable zone visualization overlays
python main.py --no_blend_zones

# Enable segment-based speed measurement
python main.py --segment_speed

# Adjust detection confidence
python main.py --confidence_threshold 0.5

# Set TTC threshold (seconds)
python main.py --ttc_threshold 2.0

# Preview zones and exit
python main.py --dump_zones_png preview.png
```

### Full Example

```bash
python main.py \
  --source_video_path input.mp4 \
  --target_video_path output.mp4 \
  --zones_file custom_zones.yaml \
  --tracker strongsort \
  --confidence_threshold 0.4 \
  --num_future_predictions 15 \
  --segment_speed
```

## Output Files

The system generates timestamped CSV files in the configured output directory:

- **`vehicle_metrics_TIMESTAMP.csv`**: Frame-by-frame vehicle tracking data  
  Columns: `frame`, `vehicle_id`, `vehicle_class`, `confidence`, `speed_km_h`

- **`ttc_events_TIMESTAMP.csv`**: Time-to-collision events  
  Columns: `frame`, `follower_id`, `follower_class`, `leader_id`, `leader_class`, `closing_distance_m`, `relative_velocity_m_s`, `ttc_s`

- **`enc_events_TIMESTAMP.csv`**: Encroachment violations  
  Columns: `tracker_id`, `class_id`, `class_name`, `zone`, `t_entry_s`, `t_flag_s`, `d_move_m`

- **`segment_speeds_TIMESTAMP.csv`**: Segment-based speed measurements (if enabled)  
  Columns: `vehicle_id`, `frame_entry`, `frame_exit`, `distance_m`, `time_s`, `speed_m_s`, `speed_km_h`

## Project Structure

```
project_root/
├── src/                      
│   ├── __init__.py
│   ├── config.py            
│   ├── detection_and_tracking.py
│   ├── kalman_filter.py     
│   ├── zone_management.py   
│   ├── geometry_and_transforms.py  
│   ├── annotators.py        
│   ├── event_processing.py  
│   └── io_utils.py          
├── main.py                  
├── requirements.txt         
├── zones_bns.yaml          
└── .env.bns                
```

## Key Components

### Detection and Tracking

- YOLO-based object detection  
- Multi-object tracking with identity preservation  
- Configurable confidence and IOU thresholds  

### Kalman Filtering

- 4-state Kalman filter (position + velocity)  
- Smooth trajectory estimation  
- Future position prediction for trajectory visualization  

### Zone Management

- Polygonal zone definition  
- Encroachment detection with time and movement thresholds  
- Real-time zone state visualization  

### Event Processing

- Time-to-Collision (TTC) calculation  
- Segment-based speed measurement  
- Relative velocity analysis  

### Performance

- Processes video at ~30 FPS on NVIDIA GPU  
- Supports multiple simultaneous tracked objects  
- Real-time visualization with minimal overhead  

## Requirements

- Python 3.8+  
- CUDA-capable GPU (recommended)  
- See `requirements.txt` for package dependencies  

## License

MIT License

## Acknowledgments

- YOLO by Ultralytics  
- Supervision library for computer vision utilities  
- StrongSORT and ByteTrack for object tracking  
