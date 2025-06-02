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


