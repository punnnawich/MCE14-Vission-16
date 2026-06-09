# Changelog — MCE14-Vission-16

---

## [Session 3] 2026-06-09 — YOLOv8 Ball Detector + Full GUI Pipeline

### Summary
Replaced HSV-based ball detection with YOLOv8n pretrained model (COCO sports ball, class 32).
Added prediction curve overlay, Trajectory Fit window, adaptive HSV calibrator,
depth background subtraction, clip recorder, and real-time calculation panel.

---

### New Files

| File | Description |
|------|-------------|
| `src/ball_detector.py` | Rewritten to use **YOLOv8n** (COCO class 32 "sports ball") + HSV red-colour confirmation |
| `src/hsv_calibrator.py` | Adaptive HSV calibrator — sample ball colour from ROI (press **c**), auto-compute thresholds, save to config.yaml |
| `src/clip_recorder.py` | Pre-throw ring-buffer clip recorder — auto-saves annotated video on every throw |
| `src/depth_background.py` | Median depth background model — compute foreground mask via depth difference (press **b**) |
| `src/ball_accuracy_test.py` | Standalone accuracy test utility for measuring systematic 3D position errors |
| `Robot/R2.ino` | Alternate robot firmware (R2 variant) |
| `src/models/` | Directory for YOLO model weights (excluded from git via .gitignore) |

---

### Modified Files

#### `src/ball_detector.py` — Full Rewrite (HSV → YOLOv8)
- **Removed**: HSV mask pipeline, MOG2 background subtractor, OpenCL GPU segmentation
- **Added**: `YOLO("yolov8n.pt")` inference (auto-downloads ~6 MB on first run)
- Detection flow: `YOLO bbox` → `HSV red-pixel ratio ≥ 12%` → `ellipse mask` → `find_ball_centroid` (unchanged)
- HSV thresholds (`lower1/upper1/lower2/upper2`) still used for red confirmation, still updated by `HSVCalibrator`
- `motion_mask = None` kept for backward compatibility with visualizer

#### `src/projectile_predictor.py` — Irondron Algorithm Alignment
- **X axis**: switched from quadratic to **linear fit** (`vx·t + x0`, matching Irondron `fit_parabolic_curve`)
- **Y axis**: switched to **Theil-Sen / RANSAC robust linear fit** (median of all pairwise slopes, handles stereo noise)
- **Z axis**: parabolic fit unchanged; landing solved at `z_catch = 0.25 m`
- New return keys: `vx`, `vy`, `n_pts`, `t_latest`, `coeff_x`, `coeff_y`, `coeff_z`, `t_land`, `t_start`

#### `src/vision_pipeline.py` — Major Additions
- **Adaptive HSV calibration** (MODULE B1): feeds `HSVCalibrator` during sampling, applies + saves on completion
- **Depth background** (MODULE B0): feeds `DepthBackground` each frame, passes `depth_fg_mask` to ball detector
- **Robot tracker disabled** (MODULE H): commented out to eliminate noisy console output
- **ClipRecorder gated**: `recorder.feed_frame()` only called after `is_calibrated = True`
- **Pre-calibration predictor guard**: `if not is_calibrated: released = False` prevents noise fill before SET ZERO
- **Prediction curve overlay**: `project_parabolic_curve()` rewritten — `t_latest → t_land`, 50 dense points, no drag factor
- **Trajectory Fit window**: always-open by default (`show_fit_plot = True`), toggle with **f**
- **Key bindings**: `c` = HSV calibrate, `b` = BG depth capture, `f` = toggle fit plot, `x`/`y` = axis lock test
- Automatic camera height calibration on SET ZERO (from `pos_camera_filtered`)

#### `src/debug_visualizer.py` — GUI Additions
- **`_draw_calc_panel()`**: right-side semi-transparent overlay showing:
  - BALL STATE: buffer count, X/Y/Z position
  - VELOCITY: Vx, Vy, Vz
  - PREDICTION: n_pts, t_land, target X/Y, CLAMPED flag
- **`draw_fit_plot()`**: 660×420 dark canvas with 3 subplots (X linear, Y Theil-Sen, Z parabola) with R² labels
- **`show_fit_plot_window()`**: renders Trajectory Fit in a separate OpenCV window
- **Prediction curve**: glow effect — 5 px dim blue (40,120,255) + 2 px bright cyan (0,220,255)
- **Landing marker**: r=14 circle + r=4 filled dot + tilted cross + coordinate label `(+31,-22)cm`
- "Waiting for throw..." message shown in fit plot before ball detected

#### `src/config.yaml`
- `system.headless`: `true` → `false` (GUI always visible)
- `hsv`: updated by adaptive calibration to match actual ball colour
- `background_depth` section added: `n_frames: 20`, `fg_thresh_mm: 150`, `dilate_px: 12`
- `calibration` section added: systematic x/y/z scale + offset corrections from `ball_accuracy_test.py`

#### `Robot/ROBOT_ALL.ino`
- Added READY protocol handshake with vision system
- ISR reduction, thread safety, IK precompute, watchdog improvements

---

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| YOLOv8n over custom-trained model | Pre-trained COCO sports ball works immediately, no training needed |
| HSV confirmation kept | Prevents YOLO false positives from non-red balls; reuses existing calibration UI |
| RANSAC Y-axis fit | Stereo depth Y noise is non-Gaussian; Theil-Sen median is robust where least-squares fails |
| Linear X-axis fit | Irondron algorithm alignment; horizontal motion is constant-velocity for short throws |
| `z_catch = 0.25 m` | Robot catching height is 25 cm from floor (Irondron uses 0.4 m) |
| `is_calibrated` guard on predictor | Stationary ball before SET ZERO filled buffer with stereo noise → false RELEASED detections |

---

## [Session 2] 2026-06-04 — Calibration, ESP32 Comms, Depth Background

### Changes
- Added `ClipRecorder` pre-throw ring buffer
- Added `DepthBackground` depth-based foreground masking
- Added `DataLogger` CSV export
- Added `LatencyProfiler` per-stage timing
- Added `ball_accuracy_test.py` for systematic calibration
- ESP32 UDP comms: READY protocol, REQUEST_POS, WAITING_FOR_CAMERA
- Robot tracker: gold → pink → green marker (HSV tuning)
- Automatic camera height calibration on SET ZERO

---

## [Session 1] 2026-05-30 — Initial Commit

### Changes
- Initial HSV + curve-fitting prediction system
- OAK-D Lite stereo camera integration (DepthAI v3 API)
- 3D ball tracking: Camera → World coordinate transform
- Projectile prediction: X/Y linear, Z parabola, z_catch = 0.25 m
- MedianFilter3D, ReleaseDetector, DebugVisualizer
- ESP32 UDP communication skeleton
- `plot_3d.py` real-time 3D trajectory viewer
