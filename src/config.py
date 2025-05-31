"""Configuration management for loading environment variables and zone configurations."""
import os
import json
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any
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

        # Zone configuration
        self.ENC_ZONE_CONFIG = os.getenv("ENC_ZONE_CONFIG")

        # Default parameters
        self.CLIP_SECONDS = 20
        self.DISPLAY = False

        # Encroachment parameters
        self.ENCROACH_SECS = 1.0
        self.MOVE_THRESH_METRES = 1.0

        # Future prediction defaults
        self.DEFAULT_NUM_FUTURE_PREDICTIONS = 10
        self.DEFAULT_FUTURE_PREDICTION_INTERVAL = 0.1
        self.DEFAULT_TTC_THRESHOLD = 1.0

        # Tracking parameters
        self.MAX_AGE_SECONDS = 1.0

        # TTC parameters
        self.COLLISION_DISTANCE = 2.0  # meters

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