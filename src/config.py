"""Configuration management for loading environment variables and zone configurations."""
import os
import json
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv


class Config:
    """Centralized configuration management."""

    def __init__(self, env_path: str = ".env.mirpur"):
        """Initialize configuration from environment file."""
        load_dotenv(dotenv_path=env_path)

        # Video paths
        self.VIDEO_PATH = os.getenv("VIDEO_PATH")
        self.OUTPUT_PATH = os.getenv("OUTPUT_PATH")
        self.ZONE_CHECK_PNG = os.getenv("ZONE_CHECK_PNG")

        # Output directory
        self.RESULTS_OUTPUT_DIR = os.getenv("RESULTS_OUTPUT_DIR", "results")

        # Model configuration
        self.MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "yolov8l.pt")
        self.DEVICE = os.getenv("DEVICE", "cpu")

        # RF-DETR Configuration
        self.RF_DETR_MODEL_PATH = os.getenv("RF_DETR_MODEL_PATH")
        self.RF_DETR_MODEL_TYPE = os.getenv("RF_DETR_MODEL_TYPE", "coco")  # 'custom' or 'coco'
        self.RF_DETR_CUSTOM_CLASSES_PATH = os.getenv("RF_DETR_CUSTOM_CLASSES_PATH")
        self.RF_DETR_RESOLUTION = int(os.getenv("RF_DETR_RESOLUTION", "560"))
        self.RF_DETR_VARIANT = os.getenv("RF_DETR_VARIANT", "base")  # 'base' or 'large'

        # Zone configuration
        self.ENC_ZONE_CONFIG = os.getenv("ENC_ZONE_CONFIG")

        # Advanced vehicle counting configuration
        self.ENABLE_ADVANCED_VEHICLE_COUNTING = os.getenv("ENABLE_ADVANCED_VEHICLE_COUNTING", "False").lower() == "true"

        # Load counting line coordinates (A and B)
        self.COUNTING_LINE_A_COORDS = self.load_counting_line_coords(self.ENC_ZONE_CONFIG, "counting_line_a")
        self.COUNTING_LINE_B_COORDS = self.load_counting_line_coords(self.ENC_ZONE_CONFIG, "counting_line_b")

        # Speed calibration configuration
        self.SPEED_CALIBRATION_MODEL_TYPE = os.getenv("SPEED_CALIBRATION_MODEL_TYPE", "linear")

        # Linear / RANSAC-Linear model coefficients
        self.SPEED_CALIBRATION_MODEL_A = float(os.getenv("SPEED_CALIBRATION_MODEL_A", "1.0"))
        self.SPEED_CALIBRATION_MODEL_B = float(os.getenv("SPEED_CALIBRATION_MODEL_B", "0.0"))

        # Polynomial model coefficients (stored as string, will be parsed in main.py)
        self.SPEED_CALIBRATION_POLY_COEFFS = os.getenv("SPEED_CALIBRATION_POLY_COEFFS", "0.0,1.0")

        # Speed stabilization parameters
        self.SPEED_SMOOTHING_WINDOW = int(os.getenv("SPEED_SMOOTHING_WINDOW", "5"))
        self.MAX_ACCELERATION = float(os.getenv("MAX_ACCELERATION", "5.0"))
        self.MIN_SPEED_THRESHOLD = float(os.getenv("MIN_SPEED_THRESHOLD", "0.1"))

        # Default parameters
        self.CLIP_SECONDS = 0
        self.DISPLAY = False

        # Encroachment parameters
        self.ENCROACH_SECS = 30
        self.MOVE_THRESH_METRES = 1.0

        # Future prediction defaults
        self.DEFAULT_NUM_FUTURE_PREDICTIONS = 10
        self.DEFAULT_FUTURE_PREDICTION_INTERVAL = 0.1
        self.DEFAULT_TTC_THRESHOLD = 1.0

        # Tracking parameters
        self.MAX_AGE_SECONDS = 3.0

        # TTC parameters
        self.COLLISION_DISTANCE = 2.0  # meters

    @staticmethod
    def load_counting_line_coords(path: str, line_key: str) -> Optional[np.ndarray]:
        """Load counting line coordinates from YAML or JSON file.

        Args:
            path: Path to the configuration file
            line_key: Key name for the line (e.g., 'counting_line_a', 'counting_line_b')

        Returns:
            NumPy array of shape (2, 2) with line endpoints [[x1, y1], [x2, y2]]
            or None if the line is not defined in the configuration file.
        """
        if not path or not os.path.exists(path):
            return None

        try:
            ext = os.path.splitext(path)[1].lower()
            with open(path, "r") as f:
                data = yaml.safe_load(f) if ext in {".yml", ".yaml"} else json.load(f)

            if line_key not in data:
                return None

            counting_line = np.asarray(data[line_key], dtype=np.int32)

            # Validate shape
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

        # Add custom classes path if using custom model
        if self.RF_DETR_MODEL_TYPE == 'custom':
            config['classes_path'] = self.RF_DETR_CUSTOM_CLASSES_PATH

        return config

    def validate_rf_detr_config(self) -> bool:
        """Validate RF-DETR configuration."""
        # Check resolution is divisible by 56
        if self.RF_DETR_RESOLUTION % 56 != 0:
            raise ValueError(f"RF-DETR resolution ({self.RF_DETR_RESOLUTION}) must be divisible by 56")

        # Check variant is valid
        if self.RF_DETR_VARIANT not in ['base', 'large']:
            raise ValueError(f"RF-DETR variant must be 'base' or 'large', got: {self.RF_DETR_VARIANT}")

        # Check custom model configuration
        if self.RF_DETR_MODEL_TYPE == 'custom':
            if not self.RF_DETR_MODEL_PATH:
                raise ValueError("RF_DETR_MODEL_PATH must be specified for custom models")

            if not self.RF_DETR_CUSTOM_CLASSES_PATH:
                raise ValueError("RF_DETR_CUSTOM_CLASSES_PATH must be specified for custom models")

            # Check if files exist
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

            # Validate format - should be {id: name} mapping
            if not isinstance(class_map, dict):
                raise ValueError("Custom classes file must contain a JSON object")

            # Convert all keys to strings for consistency
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