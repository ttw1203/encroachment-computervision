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
from typing import List, Dict, Tuple


class AdvancedSAHIYOLOv11:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cuda:0'):
        """
        Initialize SAHI with YOLOv11 model with advanced features.

        Args:
            model_path: Path to YOLOv11 weights file (.pt)
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on
        """
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_counts': {},
            'processing_times': []
        }

    def process_batch(self, config: Dict):
        """Process batch of images with configuration dictionary."""
        # Extract configuration
        image_dir = Path(config['image_dir'])
        output_dir = Path(config['output_dir'])
        slice_height = config.get('slice_height', 640)
        slice_width = config.get('slice_width', 640)
        overlap_height_ratio = config.get('overlap_height_ratio', 0.2)
        overlap_width_ratio = config.get('overlap_width_ratio', 0.2)
        save_visualization = config.get('save_visualization', False)
        save_crops = config.get('save_crops', False)

        # Create output structure
        labels_path = output_dir / 'labels'
        images_path = output_dir / 'images'
        vis_path = output_dir / 'visualizations' if save_visualization else None
        crops_path = output_dir / 'crops' if save_crops else None

        for path in [labels_path, images_path, vis_path, crops_path]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = self._get_image_files(image_dir)
        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for image_path in tqdm(image_files, desc="Processing"):
            start_time = datetime.now()

            # Run SAHI inference
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

            # Process results
            self._process_single_image(
                image_path, result, labels_path, images_path,
                vis_path, crops_path, save_visualization, save_crops
            )

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            self.stats['total_images'] += 1
            self.stats['total_detections'] += len(result.object_prediction_list)

        # Save dataset files
        self._save_dataset_files(output_dir)

        # Save statistics
        self._save_statistics(output_dir)

        # Print summary
        self._print_summary()

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
        for idx, prediction in enumerate(result.object_prediction_list):
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
        if save_vis and vis_path:
            vis_image = visualize_object_predictions(
                image=image,
                object_prediction_list=result.object_prediction_list
            )['image']
            vis_file = vis_path / f"{image_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_file), vis_image)

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

    def _save_dataset_files(self, output_dir: Path):
        """Save YOLO dataset configuration files."""
        # Save classes.txt
        classes_file = output_dir / 'classes.txt'
        class_names = sorted(self.stats['class_counts'].keys())
        with open(classes_file, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Save data.yaml
        yaml_content = {
            'path': str(output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'test': None,
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }

        with open(output_dir / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def _save_statistics(self, output_dir: Path):
        """Save inference statistics."""
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0

        stats_dict = {
            'total_images': self.stats['total_images'],
            'total_detections': self.stats['total_detections'],
            'average_detections_per_image': self.stats['total_detections'] / max(1, self.stats['total_images']),
            'class_distribution': self.stats['class_counts'],
            'average_processing_time_seconds': avg_time,
            'total_processing_time_seconds': sum(self.stats['processing_times'])
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
    parser = argparse.ArgumentParser(description='SAHI YOLOv11 Batch Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv11 model')
    parser.add_argument('--input', type=str, required=True, help='Input images directory')
    parser.add_argument('--output', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--slice-size', type=int, default=640, help='Slice size')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--visualize', action='store_true', help='Save visualizations')
    parser.add_argument('--save-crops', action='store_true', help='Save detection crops')

    args = parser.parse_args()

    # Configuration
    config = {
        'image_dir': args.input,
        'output_dir': args.output,
        'slice_height': args.slice_size,
        'slice_width': args.slice_size,
        'overlap_height_ratio': args.overlap,
        'overlap_width_ratio': args.overlap,
        'save_visualization': args.visualize,
        'save_crops': args.save_crops
    }

    # Initialize and run
    processor = AdvancedSAHIYOLOv11(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

    processor.process_batch(config)


if __name__ == "__main__":
    main()