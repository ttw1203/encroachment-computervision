# Encroachment Computer Vision System

A comprehensive computer vision system for real-time vehicle tracking, speed estimation, encroachment detection, and collision prediction using YOLO/RF-DETR object detection with advanced tracking algorithms.

![Version](https://img.shields.io/badge/version-v2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🚀 Features

### Core Detection & Tracking
- **Multi-Model Support**: YOLO (YOLOv8/v11) and RF-DETR detection models
- **Advanced Tracking**: StrongSORT and ByteTrack algorithms with ID consistency validation
- **Multi-Object Tracking**: Supports 15+ vehicle classes optimized for urban traffic

### Speed & Motion Analysis  
- **Calibrated Speed Estimation**: Linear, polynomial (2nd/3rd degree), and RANSAC models
- **Enhanced Stability**: Exponential smoothing, acceleration limits, direction filtering
- **Kalman Filtering**: 4-state filter with adaptive noise modeling and future prediction
- **Initial Velocity Estimation**: Realistic speed display from 2nd-3rd detection frames

### Safety & Collision Detection
- **Enhanced TTC (Time-to-Collision)**: 5-stage filtering pipeline with hysteresis logic
- **Persistence Filtering**: Requires sustained conditions over multiple frames  
- **Confidence-Based Filtering**: Validates detection quality before TTC calculations
- **Vehicle Dimension Collision**: AABB overlap prediction using real vehicle dimensions
- **Relative Angle Filtering**: Validates approach angles (10°-150° configurable)

### Zone Management & Monitoring
- **Encroachment Detection**: Custom polygonal zones with time/movement thresholds
- **Advanced Vehicle Counting**: Single-passage double-line counting system
- **Segment-Based Speed**: Entry/exit line speed measurement with calibration
- **Visual Feedback**: Real-time zone highlighting and line crossing indicators

### Enhanced Output & Analysis
- **Comprehensive CSV Exports**: Vehicle metrics, TTC events, encroachment violations
- **Ground Truth Processing**: Automated clip extraction and validation tools
- **Performance Diagnostics**: Speed stability analysis and regression tools
- **SAHI Integration**: Slicing-aided inference for improved small object detection

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for video processing)

## 🔧 Installation

1. **Clone repository:**
```bash
git clone https://github.com/yourusername/encroachment-computervision.git
cd encroachment-computervision
```

2. **Install dependencies:**
```bash
pip install -r requirements_strict.txt
```

3. **Download model weights:**
- YOLO: `yolov8l.pt` or custom trained weights
- StrongSORT ReID: `mobilenetv2_x1_4_dukemtmcreid.pt`
- RF-DETR: Custom trained `.pt` files or use pretrained

4. **Optional RF-DETR support:**
```bash
pip install rfdetr>=1.0.0
```

## ⚙️ Configuration

### Environment Setup
Create `.env.mirpur` (or custom env file):

```env
# Video Configuration
VIDEO_PATH=/path/to/input/video.mp4
OUTPUT_PATH=/path/to/output/video.mp4
RESULTS_OUTPUT_DIR=/path/to/results

# Model Configuration  
MODEL_WEIGHTS=path/to/yolo/weights.pt
DEVICE=cuda:0

# Zone Configuration
ENC_ZONE_CONFIG=zones-mirpur.yaml

# Speed Calibration (Linear Model)
SPEED_CALIBRATION_MODEL_TYPE=linear
SPEED_CALIBRATION_MODEL_A=1.0
SPEED_CALIBRATION_MODEL_B=0.0

# Enhanced TTC Configuration
TTC_THRESHOLD_ON=1.0
TTC_THRESHOLD_OFF=1.5
TTC_PERSISTENCE_FRAMES=5
MIN_CONFIDENCE_FOR_TTC=0.4

# RF-DETR Configuration (Optional)
RF_DETR_MODEL_PATH=/path/to/custom/model.pt
RF_DETR_MODEL_TYPE=custom
RF_DETR_CUSTOM_CLASSES_PATH=custom_classes.json
RF_DETR_RESOLUTION=560
RF_DETR_VARIANT=base
```

### Zone Configuration (YAML)
```yaml
# Detection zones (pixel coordinates)
left_zone:
  - [1531, 780]
  - [1778, 785]
  - [837, 2160]
  - [-219, 2160]

right_zone:
  - [2092, 790]
  - [2427, 797]
  - [4618, 2160]
  - [3628, 2160]

# Double-line counting system
counting_line_a:
  - [912, 1170]
  - [3066, 1122]

counting_line_b:
  - [100, 1800]
  - [3790, 1635]

# Segment speed measurement
segment_entry: [[1977, 1137], [2800, 1152]]
segment_exit: [[2019, 1466], [3342, 1490]]

# Perspective transformation
source_points:
  - [1666, 783]
  - [2222, 793]
  - [5574, 2878]
  - [-843, 2761]

target_width: 19
target_height: 78
```

## 🚦 Usage

### Basic Usage
```bash
python main.py
```

### Advanced Options
```bash
# Use RF-DETR detection model
python main.py --detector_model rf_detr

# Enable advanced vehicle counting
python main.py --advanced_counting

# Use ByteTrack instead of StrongSORT
python main.py --tracker bytetrack

# Enable segment-based speed measurement
python main.py --segment_speed

# Custom configuration
python main.py \
  --env_file .env.custom \
  --confidence_threshold 0.4 \
  --ttc_threshold_on 1.0 \
  --ttc_persistence_frames 5 \
  --enable_ttc_debug
```

### RF-DETR Specific Usage
```bash
# Custom RF-DETR model
python main.py \
  --detector_model rf_detr \
  --env_file .env.custom

# Pretrained COCO RF-DETR
python main.py \
  --detector_model rf_detr \
  --model coco
```

### Utility Scripts
```bash
# Preview zones before processing
python main.py --dump_zones_png preview.png

# Extract ground truth clips
python helper-scripts/ground_truth_extractor.py ground_truth.csv

# Analyze speed stability
python helper-scripts/speed_diagnostic.py results/vehicle_metrics.csv

# SAHI batch inference with RF-DETR
python helper-scripts/sahi_rfdetr_advanced.py \
  --train /path/to/train \
  --output /path/to/output \
  --model /path/to/custom.pt
```

## 📊 Output Files

The system generates timestamped CSV files:

- **`vehicle_metrics_TIMESTAMP.csv`**: Frame-by-frame tracking data
- **`enhanced_ttc_events_TIMESTAMP.csv`**: Validated TTC events with filtering metrics
- **`enc_events_TIMESTAMP.csv`**: Encroachment violations with timing
- **`vehicle_counts_TIMESTAMP.csv`**: Single-passage counting results
- **`segment_speeds_TIMESTAMP.csv`**: Calibrated segment-based speeds

### Enhanced TTC Events Schema
```csv
frame,follower_id,follower_class,leader_id,leader_class,closing_distance_m,relative_velocity_m_s,ttc_s,confidence_score,relative_angle_deg
1500,101,car,87,car,2.1,5.2,1.8,0.87,45.2
```

## 🏗️ Project Structure

```
project_root/
├── src/
│   ├── config.py                 # Enhanced configuration management
│   ├── detection_and_tracking.py # Multi-model detection (YOLO/RF-DETR)
│   ├── kalman_filter.py          # Enhanced stability & initial velocity
│   ├── event_processing.py       # 5-stage TTC filtering pipeline
│   ├── zone_management.py        # Encroachment & counting zones
│   ├── annotators.py             # Visual annotations & feedback
│   └── io_utils.py               # Enhanced CSV exports
├── helper-scripts/
│   ├── sahi_rfdetr_advanced.py   # SAHI batch inference
│   ├── ground_truth_extractor.py # Automated validation clips
│   ├── speed_diagnostic.py       # Performance analysis
│   └── regression_analysis.py    # Calibration model fitting
├── main.py                       # Main processing pipeline
├── vehicle_dimensions.json       # Vehicle dimension database
└── zones-mirpur.yaml            # Zone configuration example
```

## 🔬 Key Technical Features

### Enhanced TTC Processing Pipeline
1. **Hysteresis Logic**: Separate ON/OFF thresholds prevent toggling
2. **Persistence Filtering**: Requires sustained conditions (3-5 frames)
3. **Confidence Filtering**: Validates detection quality (≥40%)
4. **Angle Filtering**: Approach angle validation (10°-150°)
5. **AABB Collision**: Real vehicle dimension overlap prediction

### Speed Calibration Models
- **Linear**: `y = Ax + B`
- **Polynomial 2nd**: `y = a₀ + a₁x + a₂x²`
- **Polynomial 3rd**: `y = a₀ + a₁x + a₂x² + a₃x³`
- **RANSAC Linear**: Robust linear fitting for outlier handling

### RF-DETR Integration
- **Model Support**: Base (29M) and Large (128M) parameter variants
- **Custom Classes**: JSON-based class mapping for fine-tuned models
- **Resolution Flexibility**: Configurable input resolution (divisible by 56)
- **Seamless Pipeline**: Full compatibility with all tracking features

## 📈 Performance

- **Processing Speed**: ~25-30 FPS on NVIDIA T4 GPU
- **Detection Accuracy**: 60%+ mAP with RF-DETR models
- **Memory Efficiency**: Optimized for long video sequences
- **Urban Traffic**: Tested on dense traffic scenarios (Dhaka, Bangladesh)

## 🔧 Configuration Examples

### Urban Dense Traffic (Dhaka)
```env
TTC_THRESHOLD_ON=1.0
TTC_THRESHOLD_OFF=1.5
TTC_PERSISTENCE_FRAMES=5
MIN_CONFIDENCE_FOR_TTC=0.4
TTC_MIN_RELATIVE_ANGLE=10
TTC_MAX_RELATIVE_ANGLE=150
SPEED_CALIBRATION_MODEL_TYPE=poly3
```

### Highway/Sparse Traffic
```env
TTC_THRESHOLD_ON=2.0
TTC_THRESHOLD_OFF=3.0
TTC_PERSISTENCE_FRAMES=3
MIN_CONFIDENCE_FOR_TTC=0.6
SPEED_CALIBRATION_MODEL_TYPE=linear
```

## 📋 Version History

- **v2.0.0** - RF-DETR integration, enhanced TTC pipeline, single-passage counting
- **v1.0.0** - Speed calibration models, Kalman stability, initial velocity estimation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO** by Ultralytics for object detection
- **RF-DETR** by Roboflow for transformer-based detection
- **Supervision** library for computer vision utilities
- **StrongSORT & ByteTrack** for multi-object tracking algorithms