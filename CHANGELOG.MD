# Changelog

## 1.0.0

### Added
- Speed calibration support using multiple models:
  - **Linear** (`y = Ax + B`)
  - **2nd-degree polynomial**
  - **3rd-degree polynomial**
  - **RANSAC-based linear regression**
- `.env.bns` config support for calibration parameters (`A`, `B`, polynomial coefficients)
- `apply_speed_calibration()` method in Kalman Filter module
- Calibration applied to both Kalman-based velocity and segment-based speeds
- CLI parameters for stabilization:
  - `--speed_smoothing_window`
  - `--max_acceleration`
  - `--min_speed_threshold`

### Enhanced
- Kalman Filter:
  - Lower process noise, higher measurement noise for stability
  - Adaptive noise modeling for position and velocity
- Speed smoothing via Exponential Moving Average (EMA)
- Damping mechanism in calibration to avoid noise amplification
- Physical constraints:
  - Max acceleration limit
  - Direction change suppression
  - Speed thresholding

### Fixed
- Prevented tracker ID switches using detection validation
- Smoothed out erratic speed spikes and sudden direction reversals

### Performance
- All stabilization and calibration steps optimized to maintain <5% extra runtime per frame

## [2.0.0] - 2025-06-04

### Added - RF-DETR Integration

#### 🚀 Major Features
- **RF-DETR Object Detection Model Support**: Full integration of RF-DETR as an alternative to YOLO
  - State-of-the-art transformer-based detection (60+ mAP on COCO)
  - Real-time performance (~25 FPS on NVIDIA T4)
  - End-to-end detection without NMS requirement
  - Support for both Base (29M) and Large (128M) parameter variants

#### 🎯 Custom Model Support
- **Fine-tuned Model Integration**: Complete support for custom-trained RF-DETR models
  - External JSON-based class mapping system
  - Configurable class names for custom datasets
  - Robust validation for custom model files and class mappings
  - Support for any number of custom classes (tested with 15-class traffic model)

#### ⚙️ Configuration Management
- **Enhanced Environment Configuration**: Extended configuration system for RF-DETR
  - `RF_DETR_MODEL_PATH`: Path to custom model checkpoint
  - `RF_DETR_MODEL_TYPE`: Model type selection ('custom' or 'coco')
  - `RF_DETR_CUSTOM_CLASSES_PATH`: Path to custom class mapping JSON
  - `RF_DETR_RESOLUTION`: Configurable input resolution (must be divisible by 56)
  - `RF_DETR_VARIANT`: Model size selection ('base' or 'large')

#### 🔧 Command Line Interface
- **New Detection Model Selection**: Added `--detector_model` argument
  - `--detector_model yolo`: Use YOLO detection (default, maintains backward compatibility)
  - `--detector_model rf_detr`: Use RF-DETR detection
  - Seamless switching between detection models without code changes

#### 🛡️ Error Handling & Validation
- **Comprehensive Validation System**: Robust error handling for RF-DETR integration
  - Model path and file existence validation
  - Custom class file format validation
  - Resolution constraint validation (divisible by 56)
  - Model variant validation
  - Graceful fallback on detection errors

#### 📊 Pipeline Compatibility
- **Full Feature Compatibility**: RF-DETR integrates seamlessly with all existing features
  - ✅ Multi-object tracking (StrongSORT and ByteTrack)
  - ✅ Speed estimation and Kalman filtering
  - ✅ Zone management and encroachment detection
  - ✅ Time-to-collision (TTC) calculations
  - ✅ Segment-based speed measurement
  - ✅ Future trajectory prediction
  - ✅ All visualization and annotation features
  - ✅ CSV output and metrics logging

#### 🔍 Testing & Validation
- **Comprehensive Test Suite**: Added validation tools for RF-DETR integration
  - `test_rf_detr_integration.py`: Complete integration validation script
  - Configuration validation tests
  - Model loading and inference tests
  - Class mapping verification tests
  - Detection pipeline compatibility tests

### Changed

#### 📦 Dependencies
- **Updated Requirements**: Added RF-DETR support
  - Added `rfdetr>=1.0.0` to requirements.txt
  - Maintained backward compatibility with existing YOLO dependencies

#### 🏗️ Architecture Improvements
- **Enhanced DetectionTracker Class**: Refactored for multi-model support
  - Conditional model initialization based on detector type
  - Unified detection interface for both YOLO and RF-DETR
  - Improved class name resolution system
  - Better error handling and logging

- **Improved Configuration Management**: Enhanced Config class
  - Added RF-DETR specific configuration methods
  - Better validation and error reporting
  - Debug output for configuration troubleshooting
  - Support for multiple environment file formats

#### 📝 Documentation
- **Enhanced README**: Comprehensive RF-DETR documentation
  - Installation instructions for RF-DETR
  - Configuration examples for custom models
  - Usage examples and command combinations
  - Performance comparison tables
  - Troubleshooting guide
  - Best practices for model selection

- **Configuration Examples**: Added example files
  - `.env.bns` template with RF-DETR variables
  - `custom_classes.json` example for 15-class models
  - Complete configuration documentation

### Technical Details

#### 🔌 Integration Architecture
- **Native Supervision Compatibility**: RF-DETR directly returns `supervision.Detections` objects
- **Consistent API Design**: Same detection interface as YOLO for seamless integration
- **Optimized Performance**: Minimal overhead for model switching
- **Memory Efficient**: Proper model cleanup and resource management

#### 🎛️ Configuration Schema
```bash
# RF-DETR Configuration Variables
RF_DETR_MODEL_PATH=path/to/custom/model.pth          # Custom model checkpoint
RF_DETR_MODEL_TYPE=custom|coco                       # Model type
RF_DETR_CUSTOM_CLASSES_PATH=custom_classes.json     # Class mapping file
RF_DETR_RESOLUTION=560                               # Input resolution (÷56)
RF_DETR_VARIANT=base|large                           # Model size variant
```

#### 📈 Performance Characteristics
| Model | mAP@0.5:0.95 | Speed (T4) | Parameters | Use Case |
|-------|--------------|------------|------------|----------|
| YOLOv8m | 50.6% | 6.3ms | 29M | Balanced performance |
| RF-DETR-B | 53.3% | 6.0ms | 29M | Better accuracy, similar speed |
| RF-DETR-L | 60.5%+ | ~9ms | 128M | Maximum accuracy |

#### 🔄 Migration Guide
**From YOLO-only to Multi-Model Support:**

1. **Update Dependencies**: `pip install rfdetr>=1.0.0`
2. **Configure RF-DETR**: Add RF-DETR variables to environment file
3. **Select Model**: Use `--detector_model rf_detr` command line argument
4. **Custom Models**: Prepare custom class mapping JSON file
5. **No Code Changes**: Existing pipeline code remains unchanged

### Backward Compatibility

#### ✅ Full Backward Compatibility Maintained
- **Default Behavior**: System defaults to YOLO if no detector specified
- **Existing Commands**: All existing command combinations work unchanged
- **Configuration**: Existing environment files work without modification
- **Output Format**: CSV outputs and file formats remain identical
- **Dependencies**: New dependencies are optional for YOLO-only usage

### Performance Impact

#### ⚡ Zero Performance Impact for YOLO Users
- **Conditional Loading**: RF-DETR dependencies only loaded when selected
- **Memory Efficiency**: No additional memory usage for YOLO workflows
- **Speed Maintained**: No performance regression for existing functionality

#### 🚀 Performance Benefits for RF-DETR Users
- **Higher Accuracy**: Up to 10% mAP improvement over comparable YOLO models
- **No NMS Overhead**: Eliminates post-processing bottleneck
- **Better Domain Adaptation**: Superior transfer learning capabilities
- **Consistent Latency**: More predictable inference times

### Testing

#### 🧪 Comprehensive Test Coverage
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Speed and accuracy benchmarks
- **Compatibility Tests**: Cross-platform validation
- **Error Handling Tests**: Robust failure mode testing

#### ✅ Validated Configurations
- **Hardware**: Tested on CPU and NVIDIA GPU configurations
- **Models**: Validated with both pre-trained and custom RF-DETR models
- **Datasets**: Tested with COCO and custom traffic datasets
- **Resolutions**: Validated across multiple input resolutions (448-728)

### Known Limitations

#### ⚠️ Current Constraints
- **Resolution Requirement**: RF-DETR resolution must be divisible by 56
- **Memory Usage**: Large variant requires more GPU memory than base
- **Model Format**: Only supports RF-DETR checkpoint (.pth) format
- **Class Mapping**: Custom models require external JSON class mapping

#### 🔮 Future Improvements
- **Automatic Class Extraction**: Extract class names from model metadata
- **Additional Model Variants**: Support for future RF-DETR model sizes
- **ONNX Support**: Export and deployment optimizations
- **Batch Processing**: Multi-image batch inference support

### Acknowledgments

#### 🙏 Credits
- **RF-DETR Team**: Roboflow team for developing and open-sourcing RF-DETR
- **Supervision Library**: Continued excellence in computer vision utilities
- **Community**: Feedback and testing.

---




---

## [Unreleased] - 2025-06-02

### Added
- **Initial Velocity Estimation**: Implemented in `KalmanFilterManager` to calculate velocity based on the first 2 or 3 frames.
- New configurable argument: `--initial_velocity_frames` (default: 2) to control initialization method.
- New private method `_apply_initial_velocity()` to calculate and validate initial speed and direction.
- Cleanup logic in `remove_tracker()` for new state variables.

### Changed
- `update_or_create()` now takes an extra `frame_idx` argument and applies initial velocity logic where applicable.
- Minor update in `main.py` to supply `frame_idx` to Kalman filter calls.

### Benefits
- Enables **realistic speed display** from the 2nd or 3rd detection frame.
- Enhances **robustness** to outliers with configurable speed validation.
- Maintains **modular design** and aligns with velocity smoothing and acceleration constraints.

### Notes
- Future improvements may include adaptive frame count selection and confidence-weighted initialization.


