# Real-Time Traffic Analysis and Safety Assessment System

This project is a comprehensive computer vision pipeline designed for in-depth, real-time analysis of traffic dynamics from a video feed. The primary goal is to move beyond simple object detection and provide a multi-faceted understanding of traffic behavior. The system quantifies vehicle movement, identifies potentially hazardous situations, and analyzes traffic flow characteristics like speed and volume.

This is achieved by detecting and tracking individual vehicles and then transforming their pixel-space movements into a calibrated, real-world coordinate system. In this "bird's-eye view," the system performs a suite of analyses to measure speed, detect illegal stopping (encroachment), predict potential collisions (Time-to-Collision), and calculate key traffic engineering metrics like Space Mean Speed (SMS) and vehicle flow rates.

![Version](https://img.shields.io/badge/version-v2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

***

## 📋 Core Methodology

The strength of this system lies in its robust, step-by-step processing pipeline that converts raw video pixels into actionable traffic insights.

### 1. Detection and Tracking
The process begins by identifying all vehicles in the current video frame using an object detection model (the system supports both YOLO and RF-DETR). Immediately after detection, a tracking algorithm (StrongSORT or ByteTrack) assigns a unique and persistent ID to each vehicle. This ensures that a specific car can be followed frame-by-frame, which is fundamental for all subsequent analysis.

### 2. Perspective Transformation (Bird's-Eye View)
A critical step is the geometric transformation of vehicle positions. The system converts the coordinates of each vehicle's bounding box from the camera's 2D perspective into a 2D top-down view, often called a Bird's-Eye View (BEV). This is achieved using a perspective transformation matrix defined by mapping four points in the image to their known real-world coordinates.

**Why is this important?** The BEV allows the system to perform all calculations in real-world units (meters) instead of pixels. This makes it possible to accurately measure vehicle speed in km/h, distances between vehicles in meters, and define analysis zones with real-world dimensions.

### 3. Motion State Estimation with Kalman Filtering
For each tracked vehicle, a **Kalman Filter** is employed to estimate its true state—position (x, y) and velocity (Vx, Vy) in the BEV. The Kalman Filter is a powerful algorithm that produces a smoothed, optimal estimate of the vehicle's state by considering past measurements and predicting future ones.

This system's implementation includes several enhancements for stability and accuracy:
* **Initial Velocity Estimation:** It intelligently calculates a vehicle's initial speed based on its first few detected positions, avoiding a "zero-speed" start.
* **Motion Constraints:** It applies physical constraints, such as a maximum plausible acceleration, to prevent erratic speed spikes caused by detection jitter.
* **Speed Smoothing:** It uses an Exponential Moving Average (EMA) to smooth the calculated speed over a configurable window of frames, providing a more stable and realistic speed reading.
* **Speed Calibration:** It applies a calibration model (e.g., linear, polynomial) to correct for any systemic inaccuracies in the perspective transform, ensuring the final speed measurements are as accurate as possible.

### 4. Event Detection and Traffic Analysis
With a stable, real-world understanding of each vehicle's motion, the system performs its core analyses:

#### 🚗 Encroachment Detection
* **Methodology:** Users can define any number of polygonal "no-stopping" zones in the `zones.yaml` file. The system continuously monitors these zones. If a vehicle enters a zone and its movement distance remains below a defined threshold (`MOVE_THRESH_METRES`) for a specified duration (`ENCROACH_SECS`), an **encroachment event** is logged. This is effective for detecting illegally parked or stopped vehicles in critical areas like bus lanes or intersections.

#### 💥 Time-to-Collision (TTC) Analysis
* **Methodology:** To ensure that TTC warnings are both timely and reliable, the system uses a sophisticated **5-stage filtering pipeline** before flagging an event. For any pair of vehicles, a TTC is only reported if *all* of the following conditions are met:
    1.  **Hysteresis Filter:** The calculated TTC and distance must first fall *below* an "ON" threshold and will only be dismissed when they rise *above* a higher "OFF" threshold. This prevents flickering alerts.
    2.  **Persistence Filter:** The threat condition must be sustained for a minimum number of consecutive frames.
    3.  **Confidence Filter:** The detection confidence scores of both vehicles must be above a minimum threshold.
    4.  **Angle Filter:** The relative angle of approach between the two vehicles must be within a plausible range (e.g., not heading in parallel or opposite directions).
    5.  **AABB Collision Filter:** Using a database of real-world vehicle dimensions, the system projects the axis-aligned bounding boxes (AABBs) of both vehicles forward in time to the point of closest approach. An event is only triggered if these boxes would actually overlap.

#### 📊 Traffic Flow and Speed Analysis
* **Methodology:** The system uses a virtual "analysis segment" defined in the BEV to calculate high-level traffic metrics. This segment is defined by an `entry_line`, an `exit_line`, and a `flow_line` at its midpoint.
    * **Vehicle Counting & Flow:** As vehicles cross the `flow_line`, they are counted. At the end of a configurable time interval (e.g., 5 minutes), the system reports the total vehicle count (flow) for that period.
    * **Space Mean Speed (SMS):** The system records the exact frame a vehicle crosses the `entry_line` and the `exit_line`. By averaging the travel time for all vehicles to cross the known length of the segment, it calculates the SMS—a key metric representing the average speed of traffic over a stretch of road.
    * **Time Mean Speed (TMS):** The instantaneous speed of every vehicle is logged in `vehicle_metrics.csv`. This data can be used to calculate the TMS, which is the average speed of all vehicles passing a single point over a period of time.

## ✨ Key Features

* **Multi-Model Support:** Flexibility to use YOLO or RF-DETR for object detection.
* **Advanced Tracking:** Choice between StrongSORT and ByteTrack for robust vehicle tracking.
* **Real-World Metrics:** All calculations (speed, distance) are performed in a calibrated bird's-eye view, yielding measurements in meters and km/h.
* **Robust TTC Calculation:** A 5-stage filtering pipeline minimizes false positive collision warnings.
* **Configurable Zone Management:** Define custom polygonal zones for encroachment detection using a simple YAML file.
* **Comprehensive Traffic Analysis:** Automatically calculates and exports vehicle counts, traffic flow, and Space Mean Speed (SMS).
* **Advanced Motion Stabilization:** Employs a Kalman Filter with initial velocity estimation, acceleration limits, and speed smoothing for highly stable and accurate speed readings.
* **Detailed Data Export:** Generates multiple timestamped CSV files for in-depth analysis of vehicle metrics, encroachment events, TTC events, and overall traffic flow.
* **Visualization:** Produces an annotated output video showing bounding boxes, tracker IDs, speeds, and the status of analysis zones for intuitive visual feedback.

## ⚙️ Configuration

The entire system is controlled via a central environment file and a zones configuration file.

1.  **Environment File (`.env.*`)**: Create a file like `.env.mirpur` to control all system parameters.
    ```env
    # --- Paths ---
    VIDEO_PATH=/path/to/video.mp4
    OUTPUT_PATH=/path/to/output.mp4
    RESULTS_OUTPUT_DIR=results/
    ENC_ZONE_CONFIG=src/data/zones-mirpur.yaml # Path to your zones file

    # --- Models ---
    DETECTOR_MODEL=yolo # 'yolo' or 'rf_detr'
    MODEL_WEIGHTS=yolov8l.pt
    DEVICE=cuda:0

    # --- Core Analysis Parameters ---
    ANALYSIS_INTERVAL_MINUTES=5
    SMS_SEGMENT_LENGTH=40.0 # Length in meters of the Space Mean Speed segment
    SPEED_CALIBRATION_MODEL_TYPE=linear # linear, poly2, poly3, piecewise
    SPEED_CALIBRATION_MODEL_A=1.0
    SPEED_CALIBRATION_MODEL_B=0.0

    # --- Encroachment ---
    ENCROACH_SECS=3.0 # Time in seconds to trigger an event
    MOVE_THRESH_METRES=1.0 # Max distance moved to be considered stopped

    # --- TTC Safeguards ---
    TTC_THRESHOLD_ON=1.5
    TTC_THRESHOLD_OFF=2.5
    TTC_PERSISTENCE_FRAMES=3
    MIN_CONFIDENCE_FOR_TTC=0.4
    TTC_MIN_RELATIVE_ANGLE=10
    TTC_MAX_RELATIVE_ANGLE=150
    ```

2.  **Zone Configuration (`zones-mirpur.yaml`)**: Define all geometric elements in this YAML file. Coordinates are pixel values from the source video frame.
    ```yaml
    # Defines the region for perspective transformation
    source_points:
      - [1666, 783]
      - [2222, 793]
      - [5574, 2878]
      - [-843, 2761]
    target_width: 15  # Real-world width of the source area in meters
    target_height: 55 # Real-world height of the source area in meters

    # Polygonal zones for encroachment detection
    left_zone:
      - [1531, 780]
      - [1778, 785]
      # ... more points
    right_zone:
      - [2092, 790]
      - [2427, 797]
      # ... more points

    # Lines for vehicle counting (used by the advanced_counting feature)
    counting_line_a:
      - [912, 1170]
      - [3066, 1122]
    counting_line_b:
      - [100, 1800]
      - [3790, 1635]
    ```

## 🚀 Usage

The pipeline is executed from `main.py` with various command-line arguments to customize a run.

**Basic execution (uses parameters from the `.env` file):**
```bash
python main.py --env_file .env.mirpur
```

**Advanced execution with overrides:**
```bash
python main.py \
  --env_file .env.mirpur \
  --source_video_path /data/another_video.mp4 \
  --detector_model rf_detr \
  --tracker bytetrack \
  --confidence_threshold 0.4 \
  --advanced_counting \
  --enable_ttc_debug
```

### Key Arguments:
* `--env_file`: **(Required)** Path to your environment configuration file.
* `--detector_model`: Choose between `yolo` and `rf_detr`.
* `--tracker`: Choose between `strongsort` and `bytetrack`.
* `--advanced_counting`: Enables the double-line vehicle counting system.
* `--enable_ttc_debug`: Enables verbose console output for the TTC filtering pipeline.
* `--dump_zones_png <filename.png>`: Does not process the video. Instead, saves a single frame with the defined zones drawn on it for easy verification.

## 📊 Output Analysis

The system generates a set of timestamped CSV files in the specified `RESULTS_OUTPUT_DIR`.

* **`traffic_analysis_TIMESTAMP.csv`**: The high-level summary. Each row represents one analysis interval.
    * `time_interval`: The start and end time of the interval (e.g., "00:00-05:00").
    * `flow_incoming_veh_per_interval`: Total vehicles counted moving in the primary direction.
    * `sms_incoming_kmh`: The calculated Space Mean Speed for the interval.
    * `encroachment_detected`: `1` if any encroachment occurred during the interval, otherwise `0`.

* **`vehicle_metrics_TIMESTAMP.csv`**: Frame-by-frame data for every tracked vehicle.
    * `frame`, `vehicle_id`, `vehicle_class`, `confidence`, `speed_km_h`.

* **`enhanced_ttc_events_TIMESTAMP.csv`**: A log of every validated TTC event that passed the 5-stage filter.
    * Includes follower/leader IDs, distance, relative speed, TTC in seconds, confidence, and relative angle.

* **`enc_events_TIMESTAMP.csv`**: A log of every encroachment violation.
    * Includes tracker ID, zone, entry time, and total time spent stationary.

* **`vehicle_counts_TIMESTAMP.csv`**: Detailed vehicle counts by class (generated when `--advanced_counting` is used).

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-repo/encroachment-computervision.git](https://github.com/your-repo/encroachment-computervision.git)
    cd encroachment-computervision
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For RF-DETR support, ensure `rfdetr` is installed: `pip install rfdetr>=1.0.0`*

3.  **Download Model Weights:**
    * Place your desired model weights (e.g., `yolov8l.pt`) in the project directory or provide the full path in your `.env` file.

## 🏗️ Project Structure
```
/
├── main.py                     # Main execution script
├── requirements.txt
├── src/
│   ├── config.py                 # Configuration loader from .env files
│   ├── detection_and_tracking.py # Manages detection and tracking models
│   ├── kalman_filter.py          # Enhanced Kalman filter for state estimation
│   ├── geometry_and_transforms.py# Handles perspective transformation (BEV)
│   ├── event_processing.py       # Core logic for TTC, SMS, and flow analysis
│   ├── zone_management.py        # Handles encroachment zone logic
│   ├── annotators.py             # Draws visualizations on the output video
│   ├── io_utils.py               # Manages all CSV file outputs
│   └── data/                     # Contains config files, class maps, etc.
│       ├── .env.mirpur           # Example environment configuration
│       ├── zones-mirpur.yaml     # Example zone configuration
│       └── vehicle_dimensions.json # Real-world dimensions for AABB check
└── results/                      # Default output directory for CSVs and videos