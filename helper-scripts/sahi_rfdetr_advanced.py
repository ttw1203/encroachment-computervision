"""
Advanced SAHI RF-DETR Batch Inference Script

This script performs batch inference using RF-DETR models with SAHI (Slicing Aided Hyper Inference)
for improved small object detection. It processes images from train/val/test splits and generates
YOLO-format annotations.

Requirements:
- Python 3.9+
- rfdetr package: pip install rfdetr
- sahi package: pip install sahi
- supervision package: pip install supervision

RF-DETR Model Weights:
- Use 'coco' for pretrained COCO weights (recommended for initial testing)
- For custom weights, ensure they are RF-DETR format (.pt files from RF-DETR training)
- Common RF-DETR checkpoint files: model_ema.pt, checkpoint.pt

Troubleshooting:
- If loading custom weights fails, use --force-coco flag
- Ensure RF-DETR installation: pip install rfdetr
- Check image directories contain valid image files
- For DETR-related errors, verify checkpoint format compatibility

Example Usage:
    # Using COCO pretrained weights
    python sahi_rfdetr_advanced.py --train /path/to/train --val /path/to/val --output /path/to/output --model coco

    # Using custom RF-DETR weights
    python sahi_rfdetr_advanced.py --train /path/to/train --output /path/to/output --model /path/to/model_ema.pt

    # Force COCO weights if custom loading fails
    python sahi_rfdetr_advanced.py --train /path/to/train --output /path/to/output --force-coco

Author: Modified for RF-DETR integration
"""

import os
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from tqdm import tqdm
import yaml
import argparse
from typing import List, Dict, Tuple, Optional
import supervision as sv

# RF-DETR specific imports
try:
    from rfdetr import RFDETRBase, RFDETRLarge
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Warning: RF-DETR not installed. Install with: pip install rfdetr")

# Import OverlapFilter with fallback for different supervision versions
try:
    from supervision import OverlapFilter
except ImportError:
    try:
        from supervision.detection.overlap_filter import OverlapFilter
    except ImportError:
        # For very old versions, we'll handle this in the slicer creation
        OverlapFilter = None


class RFDETRWrapper:
    """Custom wrapper for RF-DETR to work with SAHI"""

    def __init__(self, model_path: str, model_size: str = 'base',
                 confidence_threshold: float = 0.25, device: str = 'cuda:0'):
        """
        Initialize RF-DETR wrapper.

        Args:
            model_path: Path to RF-DETR weights file or 'coco' for pretrained
            model_size: 'base' or 'large'
            confidence_threshold: Minimum confidence score for detections
            device: Device to run inference on
        """
        if not RFDETR_AVAILABLE:
            raise ImportError("RF-DETR is not installed. Install with: pip install rfdetr")

        self.confidence_threshold = confidence_threshold
        self.device = device

        # Initialize RF-DETR model with weights
        if model_path and model_path != 'coco' and Path(model_path).exists():
            try:
                print(f"Attempting to load custom weights from: {model_path}")
                # Validate checkpoint format before loading
                self._validate_checkpoint(model_path)

                # Load custom weights
                if model_size.lower() == 'large':
                    self.model = RFDETRLarge(pretrain_weights=model_path)
                else:
                    self.model = RFDETRBase(pretrain_weights=model_path)
                print("Successfully loaded custom weights")

            except Exception as e:
                print(f"Error loading custom weights: {e}")
                print("Falling back to COCO pretrained weights...")

                # Fallback to COCO weights
                if model_size.lower() == 'large':
                    self.model = RFDETRLarge()
                else:
                    self.model = RFDETRBase()
        else:
            # Use default COCO pretrained weights
            print("Using COCO pretrained weights")
            if model_size.lower() == 'large':
                self.model = RFDETRLarge()
            else:
                self.model = RFDETRBase()

            if model_path and model_path != 'coco':
                print(f"Warning: Model path {model_path} not found, using COCO pretrained weights")

    def _validate_checkpoint(self, checkpoint_path: str):
        """
        Validate RF-DETR checkpoint format.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            ValueError: If checkpoint format is invalid
        """
        import torch

        try:
            # Load checkpoint to inspect its structure
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Check if it's a proper RF-DETR checkpoint
            if isinstance(checkpoint, dict):
                # Check for expected keys in RF-DETR checkpoint
                expected_keys = ['model', 'epoch', 'optimizer']
                if 'model' in checkpoint:
                    model_state = checkpoint['model']
                    if isinstance(model_state, dict):
                        # Check for RF-DETR specific keys
                        detr_keys = ['class_embed.bias', 'class_embed.weight', 'bbox_embed']
                        has_detr_keys = any(key in str(model_state.keys()) for key in detr_keys)
                        if not has_detr_keys:
                            raise ValueError("Checkpoint doesn't appear to be an RF-DETR model")
                    else:
                        raise ValueError("Invalid model state format in checkpoint")
                else:
                    # Might be a direct state_dict
                    detr_keys = ['class_embed.bias', 'class_embed.weight', 'bbox_embed']
                    has_detr_keys = any(key in str(checkpoint.keys()) for key in detr_keys)
                    if not has_detr_keys:
                        raise ValueError("Checkpoint doesn't contain RF-DETR model keys")
            else:
                raise ValueError("Checkpoint is not in dictionary format")

        except Exception as e:
            if "DetectionModel" in str(e):
                raise ValueError(f"Checkpoint appears to be from a different framework. RF-DETR expects PyTorch .pt files. Error: {e}")
            else:
                raise ValueError(f"Error validating checkpoint: {e}")

    def predict(self, image_slice: np.ndarray) -> sv.Detections:
        """
        Run inference on image slice and return supervision Detections.

        Args:
            image_slice: Image slice as numpy array

        Returns:
            supervision Detections object
        """
        # Convert numpy array to PIL Image if needed
        from PIL import Image
        if isinstance(image_slice, np.ndarray):
            # Convert BGR to RGB
            image_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image_slice

        # Run RF-DETR inference
        detections = self.model.predict(image_pil, threshold=self.confidence_threshold)

        return detections


class AdvancedSAHIRFDETR:
    def __init__(self, model_path: str, model_size: str = 'base',
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = 'cuda:0', use_sahi_integration: bool = False,
                 disable_sahi: bool = False):
        """
        Initialize SAHI with RF-DETR model with advanced features.

        Args:
            model_path: Path to RF-DETR weights file (.pt) or 'coco' for pretrained
            model_size: 'base' or 'large' for RF-DETR model size
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on
            use_sahi_integration: Whether to use SAHI's built-in integration (experimental)
            disable_sahi: Disable SAHI completely and use regular inference only
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_path = model_path
        self.model_size = model_size
        self.use_sahi_integration = use_sahi_integration
        self.disable_sahi = disable_sahi

        if disable_sahi:
            print("SAHI is disabled - using regular inference only")

        # Initialize model based on integration method
        if use_sahi_integration and not disable_sahi:
            # Try using SAHI's built-in RT-DETR support (experimental)
            try:
                self.model = AutoDetectionModel.from_pretrained(
                    model_type='rtdetr',
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device
                )
                print("Using SAHI's built-in RT-DETR integration")
            except Exception as e:
                print(f"SAHI RT-DETR integration failed: {e}")
                print("Falling back to custom wrapper...")
                self.model = RFDETRWrapper(model_path, model_size, confidence_threshold, device)
        else:
            # Use custom RF-DETR wrapper
            self.model = RFDETRWrapper(model_path, model_size, confidence_threshold, device)

        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_counts': {},
            'processing_times': []
        }

        # Check supervision version compatibility
        self._check_supervision_version()

    def _check_supervision_version(self):
        """Check supervision version and warn about potential compatibility issues."""
        try:
            import supervision as sv
            version = sv.__version__
            major, minor = map(int, version.split('.')[:2])

            if major == 0 and minor < 21:
                print(f"Warning: supervision version {version} is quite old.")
                print("Some features may not work correctly. Consider upgrading:")
                print("pip install --upgrade supervision")
            elif major == 0 and minor >= 24:
                print(f"Using supervision version {version}")
            else:
                print(f"Using supervision version {version}")

        except Exception as e:
            print(f"Could not check supervision version: {e}")
            print("This may cause parameter compatibility issues.")

    def process_batch(self, config: Dict):
        """Process batch of images with configuration dictionary."""
        # Extract configuration
        splits = ['train', 'val', 'test']
        slice_height = config.get('slice_height', 640)
        slice_width = config.get('slice_width', 640)
        overlap_height_ratio = config.get('overlap_height_ratio', 0.2)
        overlap_width_ratio = config.get('overlap_width_ratio', 0.2)
        save_visualization = config.get('save_visualization', False)
        save_crops = config.get('save_crops', False)
        output_dir = Path(config['output_dir'])

        # Process each split
        for split in splits:
            if split not in config or not config[split]:
                print(f"Skipping {split} split - no directory specified")
                continue

            image_dir = Path(config[split])
            if not image_dir.exists():
                print(f"Warning: {split} directory {image_dir} does not exist")
                continue

            print(f"\nProcessing {split} split...")
            self._process_split(
                image_dir, output_dir, split, slice_height, slice_width,
                overlap_height_ratio, overlap_width_ratio,
                save_visualization, save_crops
            )

        # Save dataset files
        self._save_dataset_files(output_dir, config)

        # Save statistics
        self._save_statistics(output_dir)

        # Print summary
        self._print_summary()

    def _process_split(self, image_dir: Path, output_dir: Path, split: str,
                       slice_height: int, slice_width: int,
                       overlap_height_ratio: float, overlap_width_ratio: float,
                       save_visualization: bool, save_crops: bool):
        """Process a single split (train/val/test)."""

        # Validate overlap ratios
        if not (0.0 <= overlap_height_ratio < 1.0):
            print(f"Warning: overlap_height_ratio {overlap_height_ratio} out of range [0, 1). Setting to 0.2")
            overlap_height_ratio = 0.2

        if not (0.0 <= overlap_width_ratio < 1.0):
            print(f"Warning: overlap_width_ratio {overlap_width_ratio} out of range [0, 1). Setting to 0.2")
            overlap_width_ratio = 0.2

        # Create output structure for this split
        split_dir = output_dir / split
        labels_path = split_dir / 'labels'
        images_path = split_dir / 'images'
        vis_path = split_dir / 'visualizations' if save_visualization else None
        crops_path = split_dir / 'crops' if save_crops else None

        for path in [labels_path, images_path, vis_path, crops_path]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = self._get_image_files(image_dir)
        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"Found {len(image_files)} images to process in {split}")
        print(f"SAHI parameters: slice=({slice_width}x{slice_height}), overlap=({overlap_width_ratio:.2f},{overlap_height_ratio:.2f})")

        # Process each image
        for image_path in tqdm(image_files, desc=f"Processing {split}"):
            start_time = datetime.now()

            try:
                if self.disable_sahi:
                    # Skip SAHI entirely - use regular inference
                    print("SAHI disabled - using regular inference")
                    image = cv2.imread(str(image_path))

                    def callback(image_slice: np.ndarray) -> sv.Detections:
                        return self.model.predict(image_slice)

                    detections = callback(image)
                    result = self._convert_supervision_to_sahi_result(detections, image)

                elif self.use_sahi_integration and hasattr(self.model, 'model'):
                    # Use SAHI's get_sliced_prediction
                    result = get_sliced_prediction(
                        str(image_path),
                        self.model,
                        slice_height=slice_height,
                        slice_width=slice_width,
                        overlap_height_ratio=overlap_height_ratio,
                        overlap_width_ratio=overlap_width_ratio,
                        postprocess_type="NMS",
                        postprocess_match_threshold=self.iou_threshold,
                        verbose=0
                    )
                else:
                    # Use custom implementation with supervision SAHI
                    image = cv2.imread(str(image_path))

                    def callback(image_slice: np.ndarray) -> sv.Detections:
                        return self.model.predict(image_slice)

                    # Use supervision's InferenceSlicer with version-compatible parameters
                    slicer = None
                    detections = None

                    # Try different parameter combinations based on supervision version
                    try:
                        # Method 1: Latest supervision (>= 0.24.0) - overlap_wh expects ratios, not pixels
                        slicer = sv.InferenceSlicer(
                            callback=callback,
                            slice_wh=(slice_width, slice_height),
                            overlap_wh=(overlap_width_ratio, overlap_height_ratio),  # Use ratios, not pixels
                            iou_threshold=self.iou_threshold
                        )
                        print("Using InferenceSlicer with overlap_wh parameters (ratios)")

                    except (TypeError, ValueError) as e:
                        try:
                            # Method 2: Intermediate supervision (0.21.0-0.23.0)
                            slicer = sv.InferenceSlicer(
                                callback=callback,
                                slice_wh=(slice_width, slice_height),
                                overlap_ratio_wh=(overlap_width_ratio, overlap_height_ratio),
                                iou_threshold=self.iou_threshold
                            )
                            print("Using InferenceSlicer with overlap_ratio_wh parameters")

                        except (TypeError, ValueError) as e2:
                            try:
                                # Method 3: Basic supervision (older versions) - minimal parameters
                                slicer = sv.InferenceSlicer(
                                    callback=callback,
                                    slice_wh=(slice_width, slice_height)
                                )
                                print("Using basic InferenceSlicer parameters (no overlap)")

                            except Exception as e3:
                                print(f"All InferenceSlicer methods failed:")
                                print(f"  Method 1 error: {e}")
                                print(f"  Method 2 error: {e2}")
                                print(f"  Method 3 error: {e3}")
                                slicer = None

                    # Run inference
                    if slicer is not None:
                        try:
                            detections = slicer(image)
                            print(f"SAHI processing successful, found {len(detections)} detections")
                        except Exception as e:
                            print(f"SAHI inference failed: {e}")
                            detections = None

                    # Fallback to regular inference if SAHI fails
                    if detections is None:
                        print("Falling back to regular inference without SAHI...")
                        detections = callback(image)

                        # If regular inference also fails, create empty detections
                        if detections is None:
                            print("Warning: Both SAHI and regular inference failed, creating empty detections")
                            detections = sv.Detections.empty()

                    # Convert to SAHI-style result for compatibility
                    result = self._convert_supervision_to_sahi_result(detections, image)

                # Process results
                self._process_single_image(
                    image_path, result, labels_path, images_path,
                    vis_path, crops_path, save_visualization, save_crops
                )

                # Update statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                self.stats['processing_times'].append(processing_time)
                self.stats['total_images'] += 1
                if hasattr(result, 'object_prediction_list'):
                    self.stats['total_detections'] += len(result.object_prediction_list)
                elif hasattr(result, 'detections'):
                    self.stats['total_detections'] += len(result.detections)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    def _convert_supervision_to_sahi_result(self, detections: sv.Detections, image: np.ndarray):
        """Convert supervision detections to SAHI-style result for compatibility."""
        import copy

        class SAHIStyleResult:
            def __init__(self, detections, image_shape):
                self.object_prediction_list = []
                self.image_shape = image_shape

                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    class_id = detections.class_id[i] if detections.class_id is not None else 0
                    confidence = detections.confidence[i] if detections.confidence is not None else 1.0

                    # Create SAHI-style prediction object
                    prediction = SAHIPrediction(bbox, class_id, confidence)
                    self.object_prediction_list.append(prediction)

        class SAHIPrediction:
            def __init__(self, bbox, class_id, confidence):
                self.bbox = SAHIBBox(bbox)
                self.category = SAHICategory(class_id)
                self.score = SAHIScore(confidence)

            def __deepcopy__(self, memo):
                """Support for deepcopy operations needed by SAHI visualization."""
                return SAHIPrediction(
                    [self.bbox.minx, self.bbox.miny, self.bbox.maxx, self.bbox.maxy],
                    self.category.id,
                    self.score.value
                )

        class SAHIBBox:
            def __init__(self, bbox):
                self.minx, self.miny, self.maxx, self.maxy = bbox

            def __deepcopy__(self, memo):
                return SAHIBBox([self.minx, self.miny, self.maxx, self.maxy])

        class SAHICategory:
            def __init__(self, class_id):
                self.id = int(class_id)
                # Map class IDs to COCO class names (you may need to customize this)
                coco_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
                    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
                    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
                }
                self.name = coco_names.get(int(class_id), f'class_{class_id}')

            def __deepcopy__(self, memo):
                return SAHICategory(self.id)

        class SAHIScore:
            def __init__(self, confidence):
                self.value = float(confidence)

            def __deepcopy__(self, memo):
                return SAHIScore(self.value)

        return SAHIStyleResult(detections, image.shape[:2])

    def _process_single_image(self, image_path: Path, result, labels_path: Path,
                              images_path: Path, vis_path: Path = None,
                              crops_path: Path = None, save_vis: bool = False,
                              save_crops: bool = False):
        """Process single image results."""
        # Load image
        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]

        # Convert to YOLO format
        yolo_annotations = []
        predictions = getattr(result, 'object_prediction_list', [])

        for idx, prediction in enumerate(predictions):
            # Get bbox and class
            bbox = prediction.bbox
            x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

            # YOLO format conversion
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            class_id = prediction.category.id
            class_name = prediction.category.name
            confidence = prediction.score.value

            # Update class counts
            if class_name not in self.stats['class_counts']:
                self.stats['class_counts'][class_name] = 0
            self.stats['class_counts'][class_name] += 1

            # Add annotation
            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

            # Save crop if requested
            if save_crops and crops_path:
                self._save_crop(image, bbox, crops_path, image_path.stem,
                                class_name, idx, confidence)

        # Save label file
        label_file = labels_path / f"{image_path.stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        # Copy image
        output_image = images_path / image_path.name
        cv2.imwrite(str(output_image), image)

        # Save visualization if requested
        if save_vis and vis_path and hasattr(result, 'object_prediction_list'):
            try:
                # Try SAHI visualization first
                vis_image = visualize_object_predictions(
                    image=image,
                    object_prediction_list=result.object_prediction_list
                )['image']
                vis_file = vis_path / f"{image_path.stem}_vis.jpg"
                cv2.imwrite(str(vis_file), vis_image)

            except Exception as e:
                print(f"SAHI visualization failed for {image_path}: {e}")
                print("Attempting fallback visualization...")

                try:
                    # Fallback: Create simple supervision visualization
                    # Convert back to supervision detections for visualization
                    xyxy_list = []
                    class_ids = []
                    confidences = []

                    for pred in result.object_prediction_list:
                        xyxy_list.append([pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
                        class_ids.append(pred.category.id)
                        confidences.append(pred.score.value)

                    if xyxy_list:  # Only create detections if we have any
                        fallback_detections = sv.Detections(
                            xyxy=np.array(xyxy_list),
                            class_id=np.array(class_ids),
                            confidence=np.array(confidences)
                        )

                        # Create simple box and label annotators
                        box_annotator = sv.BoxAnnotator()
                        label_annotator = sv.LabelAnnotator()

                        # Create labels
                        labels = [f"{pred.category.name}: {pred.score.value:.2f}"
                                for pred in result.object_prediction_list]

                        # Annotate image
                        vis_image = box_annotator.annotate(image.copy(), fallback_detections)
                        vis_image = label_annotator.annotate(vis_image, fallback_detections, labels=labels)

                        vis_file = vis_path / f"{image_path.stem}_vis_fallback.jpg"
                        cv2.imwrite(str(vis_file), vis_image)
                        print(f"Fallback visualization saved: {vis_file}")
                    else:
                        print("No detections to visualize")

                except Exception as e2:
                    print(f"Fallback visualization also failed for {image_path}: {e2}")
                    print("Skipping visualization for this image")

    def _save_crop(self, image: np.ndarray, bbox, crops_path: Path,
                   image_name: str, class_name: str, idx: int, confidence: float):
        """Save cropped detection."""
        x_min = int(bbox.minx)
        y_min = int(bbox.miny)
        x_max = int(bbox.maxx)
        y_max = int(bbox.maxy)

        # Create class directory
        class_dir = crops_path / class_name
        class_dir.mkdir(exist_ok=True)

        # Crop and save
        crop = image[y_min:y_max, x_min:x_max]
        crop_name = f"{image_name}_{idx}_{confidence:.3f}.jpg"
        cv2.imwrite(str(class_dir / crop_name), crop)

    def _get_image_files(self, image_dir: Path) -> List[Path]:
        """Get all image files from directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        return sorted(image_files)

    def _save_dataset_files(self, output_dir: Path, config: Dict):
        """Save YOLO dataset configuration files with the requested format."""
        # Save classes.txt
        classes_file = output_dir / 'classes.txt'
        class_names = sorted(self.stats['class_counts'].keys())
        with open(classes_file, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Save data.yaml in the requested format
        yaml_content = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': len(class_names),
            'names': class_names  # List format as requested
        }

        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        # Also save a mapping file for reference
        class_mapping = {i: name for i, name in enumerate(class_names)}
        with open(output_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)

    def _save_statistics(self, output_dir: Path):
        """Save inference statistics."""
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0

        stats_dict = {
            'total_images': self.stats['total_images'],
            'total_detections': self.stats['total_detections'],
            'average_detections_per_image': self.stats['total_detections'] / max(1, self.stats['total_images']),
            'class_distribution': self.stats['class_counts'],
            'average_processing_time_seconds': avg_time,
            'total_processing_time_seconds': sum(self.stats['processing_times']),
            'model_info': {
                'model_path': str(self.model_path),
                'model_size': self.model_size,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'device': self.device
            }
        }

        with open(output_dir / 'inference_stats.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(
            f"Average detections per image: {self.stats['total_detections'] / max(1, self.stats['total_images']):.2f}")
        print("\nClass distribution:")
        for class_name, count in sorted(self.stats['class_counts'].items()):
            print(f"  {class_name}: {count}")

        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            print(f"\nAverage processing time per image: {avg_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='SAHI RF-DETR Batch Inference')
    parser.add_argument('--model', type=str, default='coco',
                        help='Path to RF-DETR model weights or "coco" for pretrained')
    parser.add_argument('--model-size', type=str, default='base', choices=['base', 'large'],
                        help='RF-DETR model size')
    parser.add_argument('--train', type=str, help='Training images directory')
    parser.add_argument('--val', type=str, help='Validation images directory')
    parser.add_argument('--test', type=str, help='Test images directory')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--slice-size', type=int, default=640, help='Slice size')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--visualize', action='store_true', help='Save visualizations')
    parser.add_argument('--save-crops', action='store_true', help='Save detection crops')
    parser.add_argument('--use-sahi-integration', action='store_true',
                        help='Use SAHI built-in RT-DETR integration (experimental)')
    parser.add_argument('--disable-sahi', action='store_true',
                        help='Disable SAHI completely and use regular inference only')
    parser.add_argument('--force-coco', action='store_true',
                        help='Force use of COCO pretrained weights even if custom path provided')

    args = parser.parse_args()

    # Validate that at least one split is provided
    if not any([args.train, args.val, args.test]):
        parser.error("At least one of --train, --val, or --test must be specified")

    # Validate overlap ratio
    if not (0.0 <= args.overlap < 1.0):
        parser.error(f"Overlap ratio must be between 0.0 and 1.0, got {args.overlap}")

    # Handle force COCO option
    if args.force_coco:
        model_path = 'coco'
        print("Forcing use of COCO pretrained weights")
    else:
        model_path = args.model

    # Configuration
    config = {
        'train': args.train,
        'val': args.val,
        'test': args.test,
        'output_dir': args.output,
        'slice_height': args.slice_size,
        'slice_width': args.slice_size,
        'overlap_height_ratio': args.overlap,
        'overlap_width_ratio': args.overlap,
        'save_visualization': args.visualize,
        'save_crops': args.save_crops
    }

    try:
        # Initialize and run
        processor = AdvancedSAHIRFDETR(
            model_path=model_path,
            model_size=args.model_size,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            use_sahi_integration=args.use_sahi_integration,
            disable_sahi=args.disable_sahi
        )

        processor.process_batch(config)

    except Exception as e:
        print(f"\nError during processing: {e}")
        print("\nTroubleshooting tips:")
        print("1. If using custom weights, ensure they are RF-DETR format (.pt files)")
        print("2. Try using --force-coco to use pretrained COCO weights")
        print("3. Check that RF-DETR is properly installed: pip install rfdetr")
        print("4. Verify your image directories exist and contain valid images")
        print(f"5. For RF-DETR specific issues, check: https://github.com/roboflow/rf-detr")
        raise


if __name__ == "__main__":
    main()