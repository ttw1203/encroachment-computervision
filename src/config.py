"""Enhanced configuration management with TTC safeguard parameters."""
import os
import json
import yaml
from typing import Tuple, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv


class Config:
    """Centralized configuration management with enhanced TTC and Kalman parameters."""

    def __init__(self, env_path: str = ".env.mirpur.vastai"):
        """Initialize configuration from environment file."""
        load_dotenv(dotenv_path=env_path)

        # Existing configuration (unchanged)
        self.VIDEO_PATH = os.getenv("VIDEO_PATH")
        self.OUTPUT_PATH = os.getenv("OUTPUT_PATH")
        self.ZONE_CHECK_PNG = os.getenv("ZONE_CHECK_PNG")
        self.RESULTS_OUTPUT_DIR = os.getenv("RESULTS_OUTPUT_DIR", "results")
        self.MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "yolov8l.pt")
        self.DEVICE = os.getenv("DEVICE", "cpu")

        # RF-DETR Configuration (unchanged)
        self.RF_DETR_MODEL_PATH = os.getenv("RF_DETR_MODEL_PATH")
        self.RF_DETR_MODEL_TYPE = os.getenv("RF_DETR_MODEL_TYPE", "coco")
        self.RF_DETR_CUSTOM_CLASSES_PATH = os.getenv("RF_DETR_CUSTOM_CLASSES_PATH")
        self.RF_DETR_RESOLUTION = int(os.getenv("RF_DETR_RESOLUTION", "560"))
        self.RF_DETR_VARIANT = os.getenv("RF_DETR_VARIANT", "base")

        # Zone configuration (unchanged)
        self.ENC_ZONE_CONFIG = os.getenv("ENC_ZONE_CONFIG")
        self.ENABLE_ADVANCED_VEHICLE_COUNTING = os.getenv("ENABLE_ADVANCED_VEHICLE_COUNTING", "False").lower() == "true"
        self.COUNTING_LINE_A_COORDS = self.load_counting_line_coords(self.ENC_ZONE_CONFIG, "counting_line_a")
        self.COUNTING_LINE_B_COORDS = self.load_counting_line_coords(self.ENC_ZONE_CONFIG, "counting_line_b")

        # Speed calibration configuration (unchanged)
        self.SPEED_CALIBRATION_MODEL_TYPE = os.getenv("SPEED_CALIBRATION_MODEL_TYPE", "linear")
        self.SPEED_CALIBRATION_MODEL_A = float(os.getenv("SPEED_CALIBRATION_MODEL_A", "1.0"))
        self.SPEED_CALIBRATION_MODEL_B = float(os.getenv("SPEED_CALIBRATION_MODEL_B", "0.0"))
        self.SPEED_CALIBRATION_POLY_COEFFS = os.getenv("SPEED_CALIBRATION_POLY_COEFFS", "0.0,1.0")

        # ============================================
        # ENHANCED KALMAN FILTER INITIALIZATION PARAMETERS
        # ============================================

        # Initial uncertainty parameters for new tracks
        self.INITIAL_VELOCITY_UNCERTAINTY = float(os.getenv("INITIAL_VELOCITY_UNCERTAINTY", "5.0"))  # m/s
        self.INITIAL_POSITION_UNCERTAINTY = float(os.getenv("INITIAL_POSITION_UNCERTAINTY", "0.5"))  # meters

        # Velocity calculation parameters
        self.INITIAL_VELOCITY_FRAMES = int(os.getenv("INITIAL_VELOCITY_FRAMES", "2"))  # 2 or 3 frames

        # ============================================
        # TTC SAFEGUARD PARAMETERS
        # ============================================

        # Burn-in period - skip TTC for new tracks
        self.TTC_BURN_IN_FRAMES = int(os.getenv("TTC_BURN_IN_FRAMES", "10"))

        # Velocity threshold for TTC calculation
        self.TTC_MIN_VELOCITY = float(os.getenv("TTC_MIN_VELOCITY", "0.3"))  # m/s

        # Track confidence threshold for TTC
        self.TTC_MIN_TRACK_CONFIDENCE = float(os.getenv("TTC_MIN_TRACK_CONFIDENCE", "0.75"))

        # Minimum detections required before TTC is enabled
        self.TTC_MIN_DETECTIONS = int(os.getenv("TTC_MIN_DETECTIONS", "5"))

        # Speed stabilization parameters (unchanged)
        self.SPEED_SMOOTHING_WINDOW = int(os.getenv("SPEED_SMOOTHING_WINDOW", "5"))
        self.MAX_ACCELERATION = float(os.getenv("MAX_ACCELERATION", "5.0"))
        self.MIN_SPEED_THRESHOLD = float(os.getenv("MIN_SPEED_THRESHOLD", "0.1"))

        # ============================================
        # ENHANCED TTC PROCESSING CONFIGURATION
        # ============================================

        # Hysteresis thresholds for TTC activation/deactivation
        self.TTC_THRESHOLD_ON = float(os.getenv("TTC_THRESHOLD_ON", "1.5"))        # seconds
        self.TTC_THRESHOLD_OFF = float(os.getenv("TTC_THRESHOLD_OFF", "2.5"))      # seconds

        # Collision distance thresholds for activation/deactivation
        self.COLLISION_DISTANCE_ON = float(os.getenv("COLLISION_DISTANCE_ON", "1.5"))   # meters
        self.COLLISION_DISTANCE_OFF = float(os.getenv("COLLISION_DISTANCE_OFF", "2.5"))  # meters

        # Persistence filtering - require sustained conditions
        self.TTC_PERSISTENCE_FRAMES = int(os.getenv("TTC_PERSISTENCE_FRAMES", "3"))

        # Confidence filtering - minimum detection confidence for TTC evaluation
        self.MIN_CONFIDENCE_FOR_TTC = float(os.getenv("MIN_CONFIDENCE_FOR_TTC", "0.4"))

        # Relative angle filtering - approach angle constraints
        self.TTC_MIN_RELATIVE_ANGLE = float(os.getenv("TTC_MIN_RELATIVE_ANGLE", "10"))   # degrees
        self.TTC_MAX_RELATIVE_ANGLE = float(os.getenv("TTC_MAX_RELATIVE_ANGLE", "150"))  # degrees

        # Cleanup and memory management
        self.TTC_CLEANUP_TIMEOUT_FRAMES = int(os.getenv("TTC_CLEANUP_TIMEOUT_FRAMES", "90"))

        # Debug mode for TTC processing
        self.ENABLE_TTC_DEBUG = os.getenv("ENABLE_TTC_DEBUG", "False").lower() == "true"

        # Vehicle dimensions for AABB collision detection
        self.VEHICLE_DIMENSIONS = self._load_vehicle_dimensions()

        # Default parameters (unchanged)
        self.CLIP_SECONDS = 25
        self.DISPLAY = False
        self.ENCROACH_SECS = 1
        self.MOVE_THRESH_METRES = 1.0
        self.DEFAULT_NUM_FUTURE_PREDICTIONS = 10
        self.DEFAULT_FUTURE_PREDICTION_INTERVAL = 0.1
        self.DEFAULT_TTC_THRESHOLD = 1.0
        self.MAX_AGE_SECONDS = 0
        self.COLLISION_DISTANCE = 2.0

    def get_kalman_config(self) -> Dict[str, Any]:
        """Get Kalman filter configuration with TTC safeguards."""
        return {
            'speed_smoothing_window': self.SPEED_SMOOTHING_WINDOW,
            'max_acceleration': self.MAX_ACCELERATION,
            'min_speed_threshold': self.MIN_SPEED_THRESHOLD,
            'initial_velocity_frames': self.INITIAL_VELOCITY_FRAMES,
            'initial_velocity_uncertainty': self.INITIAL_VELOCITY_UNCERTAINTY,
            'position_uncertainty': self.INITIAL_POSITION_UNCERTAINTY,
            'ttc_burn_in_frames': self.TTC_BURN_IN_FRAMES,
            'ttc_min_velocity': self.TTC_MIN_VELOCITY,
            'ttc_min_confidence': self.TTC_MIN_TRACK_CONFIDENCE
        }

    def validate_kalman_config(self) -> bool:
        """Validate Kalman filter and TTC safeguard configuration."""
        errors = []

        # Validate initialization parameters
        if not (0.1 <= self.INITIAL_VELOCITY_UNCERTAINTY <= 20.0):
            errors.append("INITIAL_VELOCITY_UNCERTAINTY should be between 0.1 and 20.0 m/s")

        if not (0.01 <= self.INITIAL_POSITION_UNCERTAINTY <= 5.0):
            errors.append("INITIAL_POSITION_UNCERTAINTY should be between 0.01 and 5.0 meters")

        if self.INITIAL_VELOCITY_FRAMES not in [2, 3]:
            errors.append("INITIAL_VELOCITY_FRAMES must be 2 or 3")

        # Validate TTC safeguard parameters
        if not (1 <= self.TTC_BURN_IN_FRAMES <= 60):
            errors.append("TTC_BURN_IN_FRAMES should be between 1 and 60 frames")

        if not (0.1 <= self.TTC_MIN_VELOCITY <= 5.0):
            errors.append("TTC_MIN_VELOCITY should be between 0.1 and 5.0 m/s")

        if not (0.0 <= self.TTC_MIN_TRACK_CONFIDENCE <= 1.0):
            errors.append("TTC_MIN_TRACK_CONFIDENCE should be between 0.0 and 1.0")

        if not (1 <= self.TTC_MIN_DETECTIONS <= 20):
            errors.append("TTC_MIN_DETECTIONS should be between 1 and 20")

        # Validate existing TTC parameters
        if self.TTC_THRESHOLD_ON >= self.TTC_THRESHOLD_OFF:
            errors.append("TTC_THRESHOLD_ON must be less than TTC_THRESHOLD_OFF")

        if self.COLLISION_DISTANCE_ON >= self.COLLISION_DISTANCE_OFF:
            errors.append("COLLISION_DISTANCE_ON must be less than COLLISION_DISTANCE_OFF")

        if errors:
            print("Kalman/TTC Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def print_kalman_config_summary(self) -> None:
        """Print a summary of Kalman and TTC safeguard configuration."""
        print("=== Enhanced Kalman Filter Configuration ===")
        print(f"Initialization Parameters:")
        print(f"  Velocity Uncertainty: {self.INITIAL_VELOCITY_UNCERTAINTY:.1f} m/s")
        print(f"  Position Uncertainty: {self.INITIAL_POSITION_UNCERTAINTY:.1f} m")
        print(f"  Initial Velocity Frames: {self.INITIAL_VELOCITY_FRAMES}")
        print(f"TTC Safeguards:")
        print(f"  Burn-in Period: {self.TTC_BURN_IN_FRAMES} frames")
        print(f"  Min Velocity: {self.TTC_MIN_VELOCITY:.1f} m/s")
        print(f"  Min Confidence: {self.TTC_MIN_TRACK_CONFIDENCE:.2f}")
        print(f"  Min Detections: {self.TTC_MIN_DETECTIONS}")
        print(f"Stability Parameters:")
        print(f"  Speed Smoothing: {self.SPEED_SMOOTHING_WINDOW} frames")
        print(f"  Max Acceleration: {self.MAX_ACCELERATION:.1f} m/s²")
        print(f"  Min Speed Threshold: {self.MIN_SPEED_THRESHOLD:.1f} m/s")
        print("=" * 45)

    def _load_vehicle_dimensions(self) -> Dict[str, Dict[str, float]]:
        """Load vehicle dimensions configuration from environment or use defaults."""
        # Try to load from environment file first
        dimensions_env = os.getenv("VEHICLE_DIMENSIONS_JSON")
        if dimensions_env:
            try:
                return json.loads(dimensions_env)
            except json.JSONDecodeError:
                print("Warning: Invalid VEHICLE_DIMENSIONS_JSON, using defaults")

        # Try to load from separate JSON file
        dimensions_file = os.getenv("VEHICLE_DIMENSIONS_FILE")
        if dimensions_file and os.path.exists(dimensions_file):
            try:
                with open(dimensions_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load {dimensions_file}, using defaults")

        # Default dimensions optimized for Dhaka traffic (meters)
        return {
            'car': {'length': 4.5, 'width': 1.8},
            'truck': {'length': 8.0, 'width': 2.5},
            'bus': {'length': 12.0, 'width': 2.5},
            'motorcycle': {'length': 2.0, 'width': 0.8},
            'bicycle': {'length': 1.8, 'width': 0.6},
            'person': {'length': 0.6, 'width': 0.4},
            'rickshaw': {'length': 3.0, 'width': 1.2},
            'cng': {'length': 3.5, 'width': 1.5},        # CNG auto-rickshaw
            'van': {'length': 5.5, 'width': 2.0},
            'microbus': {'length': 6.0, 'width': 2.0},
            'default': {'length': 4.0, 'width': 1.8}
        }

    def get_ttc_config(self) -> Dict[str, Any]:
        """Get TTC processing configuration dictionary."""
        return {
            'TTC_THRESHOLD_ON': self.TTC_THRESHOLD_ON,
            'TTC_THRESHOLD_OFF': self.TTC_THRESHOLD_OFF,
            'COLLISION_DISTANCE_ON': self.COLLISION_DISTANCE_ON,
            'COLLISION_DISTANCE_OFF': self.COLLISION_DISTANCE_OFF,
            'TTC_PERSISTENCE_FRAMES': self.TTC_PERSISTENCE_FRAMES,
            'MIN_CONFIDENCE_FOR_TTC': self.MIN_CONFIDENCE_FOR_TTC,
            'TTC_MIN_RELATIVE_ANGLE': self.TTC_MIN_RELATIVE_ANGLE,
            'TTC_MAX_RELATIVE_ANGLE': self.TTC_MAX_RELATIVE_ANGLE,
            'TTC_CLEANUP_TIMEOUT_FRAMES': self.TTC_CLEANUP_TIMEOUT_FRAMES,
            'ENABLE_TTC_DEBUG': self.ENABLE_TTC_DEBUG,
            'VEHICLE_DIMENSIONS': self.VEHICLE_DIMENSIONS,
            'video_fps': 30.0  # Will be updated with actual video FPS
        }

    def validate_ttc_config(self) -> bool:
        """Validate TTC configuration parameters."""
        errors = []

        # Validate thresholds
        if self.TTC_THRESHOLD_ON >= self.TTC_THRESHOLD_OFF:
            errors.append("TTC_THRESHOLD_ON must be less than TTC_THRESHOLD_OFF")

        if self.COLLISION_DISTANCE_ON >= self.COLLISION_DISTANCE_OFF:
            errors.append("COLLISION_DISTANCE_ON must be less than COLLISION_DISTANCE_OFF")

        # Validate ranges
        if not (0.1 <= self.TTC_THRESHOLD_ON <= 10.0):
            errors.append("TTC_THRESHOLD_ON should be between 0.1 and 10.0 seconds")

        if not (0.1 <= self.COLLISION_DISTANCE_ON <= 10.0):
            errors.append("COLLISION_DISTANCE_ON should be between 0.1 and 10.0 meters")

        if not (0.0 <= self.MIN_CONFIDENCE_FOR_TTC <= 1.0):
            errors.append("MIN_CONFIDENCE_FOR_TTC should be between 0.0 and 1.0")

        if not (0 <= self.TTC_MIN_RELATIVE_ANGLE <= 180):
            errors.append("TTC_MIN_RELATIVE_ANGLE should be between 0 and 180 degrees")

        if not (0 <= self.TTC_MAX_RELATIVE_ANGLE <= 180):
            errors.append("TTC_MAX_RELATIVE_ANGLE should be between 0 and 180 degrees")

        if self.TTC_MIN_RELATIVE_ANGLE >= self.TTC_MAX_RELATIVE_ANGLE:
            errors.append("TTC_MIN_RELATIVE_ANGLE must be less than TTC_MAX_RELATIVE_ANGLE")

        if errors:
            print("TTC Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def print_ttc_config_summary(self) -> None:
        """Print a summary of TTC configuration for debugging."""
        print("=== Enhanced TTC Configuration ===")
        print(f"Hysteresis Thresholds:")
        print(f"  TTC ON/OFF: {self.TTC_THRESHOLD_ON:.1f}s / {self.TTC_THRESHOLD_OFF:.1f}s")
        print(f"  Distance ON/OFF: {self.COLLISION_DISTANCE_ON:.1f}m / {self.COLLISION_DISTANCE_OFF:.1f}m")
        print(f"Filtering:")
        print(f"  Persistence: {self.TTC_PERSISTENCE_FRAMES} frames")
        print(f"  Min Confidence: {self.MIN_CONFIDENCE_FOR_TTC:.2f}")
        print(f"  Angle Range: {self.TTC_MIN_RELATIVE_ANGLE:.0f}° - {self.TTC_MAX_RELATIVE_ANGLE:.0f}°")
        print(f"Debug Mode: {'Enabled' if self.ENABLE_TTC_DEBUG else 'Disabled'}")
        print(f"Vehicle Classes: {len(self.VEHICLE_DIMENSIONS)} defined")
        print("=" * 35)

    # Existing methods (unchanged)
    @staticmethod
    def load_counting_line_coords(path: str, line_key: str) -> Optional[np.ndarray]:
        """Load counting line coordinates from YAML or JSON file."""
        if not path or not os.path.exists(path):
            return None

        try:
            ext = os.path.splitext(path)[1].lower()
            with open(path, "r") as f:
                data = yaml.safe_load(f) if ext in {".yml", ".yaml"} else json.load(f)

            if line_key not in data:
                return None

            counting_line = np.asarray(data[line_key], dtype=np.int32)

            if counting_line.shape != (2, 2):
                raise ValueError(f"{line_key} must have shape (2, 2), got {counting_line.shape}")

            return counting_line

        except Exception as e:
            print(f"Warning: Failed to load {line_key} from {path}: {e}")
            return None

    def get_rf_detr_config(self) -> Dict[str, Any]:
        """Get RF-DETR configuration dictionary."""
        config = {
            'model_path': self.RF_DETR_MODEL_PATH,
            'model_type': self.RF_DETR_MODEL_TYPE,
            'resolution': self.RF_DETR_RESOLUTION,
            'variant': self.RF_DETR_VARIANT
        }

        if self.RF_DETR_MODEL_TYPE == 'custom':
            config['classes_path'] = self.RF_DETR_CUSTOM_CLASSES_PATH

        return config

    def validate_rf_detr_config(self) -> bool:
        """Validate RF-DETR configuration."""
        if self.RF_DETR_RESOLUTION % 56 != 0:
            raise ValueError(f"RF-DETR resolution ({self.RF_DETR_RESOLUTION}) must be divisible by 56")

        if self.RF_DETR_VARIANT not in ['base', 'large']:
            raise ValueError(f"RF-DETR variant must be 'base' or 'large', got: {self.RF_DETR_VARIANT}")

        if self.RF_DETR_MODEL_TYPE == 'custom':
            if not self.RF_DETR_MODEL_PATH:
                raise ValueError("RF_DETR_MODEL_PATH must be specified for custom models")

            if not self.RF_DETR_CUSTOM_CLASSES_PATH:
                raise ValueError("RF_DETR_CUSTOM_CLASSES_PATH must be specified for custom models")

            if not os.path.exists(self.RF_DETR_MODEL_PATH):
                raise FileNotFoundError(f"RF-DETR model not found: {self.RF_DETR_MODEL_PATH}")

            if not os.path.exists(self.RF_DETR_CUSTOM_CLASSES_PATH):
                raise FileNotFoundError(f"RF-DETR classes file not found: {self.RF_DETR_CUSTOM_CLASSES_PATH}")

        return True

    @staticmethod
    def load_rf_detr_custom_classes(classes_path: str) -> Dict[str, str]:
        """Load custom class mapping from JSON file."""
        try:
            with open(classes_path, 'r') as f:
                class_map = json.load(f)

            if not isinstance(class_map, dict):
                raise ValueError("Custom classes file must contain a JSON object")

            return {str(k): str(v) for k, v in class_map.items()}

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in custom classes file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading custom classes file: {e}")

    @staticmethod
    def load_zones(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load zone polygons from YAML or JSON file."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r") as f:
            data = yaml.safe_load(f) if ext in {".yml", ".yaml"} else json.load(f)

        left = np.asarray(data["left_zone"], np.int32)
        right = np.asarray(data["right_zone"], np.int32)
        return left, right

    @staticmethod
    def load_segments(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load segment entry/exit lines from configuration file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        entry = np.asarray(data["segment_entry"], np.int32)
        exit_ = np.asarray(data["segment_exit"], np.int32)
        return entry, exit_

    def get_source_target_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get source and target points for view transformation."""
        with open(self.ENC_ZONE_CONFIG, "r") as f:
            data = yaml.safe_load(f)

        SOURCE = np.array(data.get("source_points", [[1281, 971], [2309, 971], [6090, 2160], [-2243, 2160]]))
        TARGET_WIDTH = data.get("target_width", 50)
        TARGET_HEIGHT = data.get("target_height", 130)

        TARGET = np.array([
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ])

        return SOURCE, TARGET