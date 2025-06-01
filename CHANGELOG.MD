# Changelog

## [Unreleased]

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

