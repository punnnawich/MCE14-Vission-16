# ── Must be set BEFORE numpy/cv2/BLAS load ──────────────────────────────────
import os as _os
_N = str(_os.cpu_count() or 12)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, _N)
del _v, _N
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import yaml
import time
import os
import socket as stdlib_socket
import gc
from collections import deque
import depthai as dai

from ball_detector import BallDetector
from median_filter import MedianFilter3D
from release_detector import ReleaseDetector
from projectile_predictor import ProjectilePredictor
from robot_tracker import RobotTracker
from robot_comms import RobotComms
from debug_visualizer import DebugVisualizer
from latency_profiler import LatencyProfiler
from data_logger import DataLogger
from performance import init_performance, disable_gc, cleanup_performance, competition_mode
from clip_recorder import ClipRecorder
from depth_background import DepthBackground
from hsv_calibrator import HSVCalibrator

def load_config(config_path="config.yaml"):
    """
    Loads config.yaml configuration parameters.
    """
    if not os.path.exists(config_path):
        # Try local src/ folder fallback
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_camera_pipeline(config):
    """
    Creates and configures the DepthAI camera pipeline (depthai v3 API).
    """
    # Recovery: close any BOOTED device from a previous crashed session
    available = dai.Device.getAllAvailableDevices()
    for dev_info in available:
        if dev_info.state == dai.XLinkDeviceState.X_LINK_BOOTED:
            try:
                with dai.Device(dev_info) as d:
                    d.close()
            except Exception:
                pass
            import time
            time.sleep(1)

    pipeline = dai.Pipeline()
    
    camera_cfg = config.get("camera", {})
    width = camera_cfg.get("resolution_w", 640)
    height = camera_cfg.get("resolution_h", 360)
    fps = camera_cfg.get("fps", 30)
    preset_str = camera_cfg.get("preset_mode", "DEFAULT")

    # Map string preset to depthai PresetMode (v3 API)
    _PM = dai.node.StereoDepth.PresetMode
    preset_mode = getattr(_PM, preset_str.upper(), _PM.DEFAULT)

    # 1. RGB Camera (unified Camera node replaces ColorCamera)
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((width, height), type=dai.ImgFrame.Type.BGR888i, fps=fps)

    # 2. Stereo Depth Node — overload 2: autoCreateCameras with explicit fps
    # Passing Camera requestOutput() to build() overload 1 crashes the firmware:
    # requestOutput produces ISP-encoded frames, not the raw sensor data that
    # StereoDepth expects.  autoCreateCameras handles the internal linking
    # correctly; fps= sets the mono camera rate explicitly.
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.build(autoCreateCameras=True, presetMode=preset_mode)
    stereo.setExtendedDisparity(camera_cfg.get("extended_disparity", False))
    stereo.setLeftRightCheck(camera_cfg.get("left_right_check", True))
    use_subpixel = camera_cfg.get("subpixel", False)
    stereo.setSubpixel(use_subpixel)

    # Median filter for depth noise reduction.
    # HARDWARE LIMITATION: Subpixel mode multiplies disparity values by 32×,
    # pushing them beyond the median filter's maximum supported value (1024).
    # Auto-disable median filter when subpixel is active to suppress firmware errors.
    median_kernel = camera_cfg.get("median_filter", 7)
    if use_subpixel and median_kernel != 0:
        median_kernel = 0
    median_map = {
        0: dai.MedianFilter.MEDIAN_OFF,
        3: dai.MedianFilter.KERNEL_3x3,
        5: dai.MedianFilter.KERNEL_5x5,
        7: dai.MedianFilter.KERNEL_7x7,
    }
    stereo.initialConfig.setMedianFilter(median_map.get(median_kernel, dai.MedianFilter.KERNEL_7x7))
    
    # Align depth map to RGB camera perspective
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    # Depth output width must be a multiple of 16
    stereo.setOutputSize(width - (width % 16), height)

    # ── VPU On-Device Depth Post-Processing ──────────────────────────
    # These filters run on the Myriad X VPU chip (not CPU/GPU), reducing
    # depth noise BEFORE data reaches the host. Zero CPU cost.
    try:
        stereo_config = stereo.initialConfig.get()
        # Spatial filter: edge-preserving smoothing (reduces depth noise)
        stereo_config.postProcessing.spatialFilter.enable = camera_cfg.get("vpu_spatial_filter", True)
        stereo_config.postProcessing.spatialFilter.holeFillingRadius = 2
        stereo_config.postProcessing.spatialFilter.numIterations = 1
        # Temporal filter: uses previous frames to reduce depth flickering
        stereo_config.postProcessing.temporalFilter.enable = camera_cfg.get("vpu_temporal_filter", True)
        # Threshold filter: discard out-of-range depths on VPU
        stereo_config.postProcessing.thresholdFilter.minRange = camera_cfg.get("depth_min_mm", 200)
        stereo_config.postProcessing.thresholdFilter.maxRange = camera_cfg.get("depth_max_mm", 6000)
        stereo.initialConfig.set(stereo_config)
    except Exception:
        pass

    # 3. Create output queues directly (XLinkOut is automatic in v3)
    # Set to non-blocking with small maxSize to prevent queue backlog and device ping timeouts
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_rgb, q_depth

def get_synced_frames(q_rgb, q_depth):
    """
    Retrieves the latest synchronized pair of RGB and Depth frames from the queues.
    Ensures that sequence numbers match to prevent coordinate projection errors and jitter,
    while flushing old frames on the host to keep latency to an absolute minimum.
    """
    # 1. Flush RGB queue to get the latest frame
    last_rgb = None
    while True:
        f = q_rgb.tryGet()
        if f is None:
            break
        last_rgb = f

    # 2. Flush Depth queue to get the latest frame
    last_depth = None
    while True:
        f = q_depth.tryGet()
        if f is None:
            break
        last_depth = f

    # 3. If queues were empty, poll with timeout (prevents infinite blocking)
    if last_rgb is None:
        deadline = time.perf_counter() + 0.5  # 500ms timeout
        while last_rgb is None and time.perf_counter() < deadline:
            last_rgb = q_rgb.tryGet()
            if last_rgb is None:
                time.sleep(0.002)
        if last_rgb is None:
            return None, None

    if last_depth is None:
        deadline = time.perf_counter() + 0.5  # 500ms timeout
        while last_depth is None and time.perf_counter() < deadline:
            last_depth = q_depth.tryGet()
            if last_depth is None:
                time.sleep(0.002)
        if last_depth is None:
            return None, None

    # 4. Align frames by matching their sequence numbers (max 10 search attempts)
    for _ in range(10):
        seq_rgb = last_rgb.getSequenceNum()
        seq_depth = last_depth.getSequenceNum()

        if seq_rgb == seq_depth:
            return last_rgb, last_depth
        elif seq_rgb < seq_depth:
            # RGB is older, try to get a newer RGB frame (non-blocking)
            next_rgb = q_rgb.tryGet()
            if next_rgb is None:
                # Can't sync perfectly — return best available pair instead of blocking
                return last_rgb, last_depth
            last_rgb = next_rgb
        else:
            # Depth is older, try to get a newer Depth frame (non-blocking)
            next_depth = q_depth.tryGet()
            if next_depth is None:
                # Can't sync perfectly — return best available pair instead of blocking
                return last_rgb, last_depth
            last_depth = next_depth

    return last_rgb, last_depth

def project_parabolic_curve(prediction, pos_zero, R_ext, T_ext, camera_matrix):
    """
    Projects the fitted 3D trajectory (X linear, Y linear, Z parabola) from
    current time to impact onto 2D pixel coordinates for overlay on RGB feed.

    Uses the same linear/Theil-Sen/parabola model as the predictor — no drag
    correction applied here (model already accounts for all physics).
    """
    if prediction is None:
        return None

    coeff_x = np.array(prediction.get("coeff_x"))
    coeff_y = np.array(prediction.get("coeff_y"))
    coeff_z = np.array(prediction.get("coeff_z"))
    t_latest = prediction.get("t_latest", 0.0)
    t_land   = prediction.get("t_land",   0.0)

    if t_land <= t_latest:
        return None

    # Dense time steps from now to landing (50 points → smooth curve)
    t_steps = np.linspace(t_latest, t_land, 50)

    fx  = camera_matrix[0, 0]
    fy  = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]
    R_inv = R_ext.T

    cal_x = float(prediction.get("cal_x_scale", 1.0))
    cal_y = float(prediction.get("cal_y_scale", 1.0))

    curve_pixels = []
    for t in t_steps:
        # Evaluate 3D position in predictor world space (relative to pos_zero)
        x_rel = float(np.polyval(coeff_x, t)) * cal_x
        y_rel = float(np.polyval(coeff_y, t)) * cal_y
        z_rel = float(np.polyval(coeff_z, t))

        # Absolute world position
        pos_w = np.array([x_rel + pos_zero[0],
                          y_rel + pos_zero[1],
                          z_rel + pos_zero[2]])

        # World → Camera
        pos_c = R_inv @ (pos_w.reshape(3, 1) - T_ext)
        xc, yc, zc = pos_c.flatten()

        if zc > 0.05:
            u = int(xc * fx / zc + cx0)
            v = int(yc * fy / zc + cy0)
            if -200 <= u <= 1500 and -200 <= v <= 1200:
                curve_pixels.append((u, v))

    return curve_pixels

def project_workspace_boundary(pos_zero, z_catch, workspace_radius, R_ext, T_ext, camera_matrix):
    """
    Projects the 3D workspace boundary cylinder (radius 50cm) from floor (Z=0)
    to catching height (Z=z_catch) into 2D camera pixel coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, 60)
    R_inv = R_ext.T
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]
    
    def project_point(x_norm, y_norm, z_norm):
        # Absolute world coordinate
        x_world = x_norm + pos_zero[0]
        y_world = y_norm + pos_zero[1]
        z_world = z_norm + pos_zero[2]
        
        # Transform World to Camera space
        pos_w = np.array([x_world, y_world, z_world])
        pos_c = R_inv @ (pos_w.reshape(3, 1) - T_ext)
        xc, yc, zc = pos_c.flatten()
        
        # Project to pixel space
        if zc > 0.1:
            u = int(xc * fx / zc + cx0)
            v = int(yc * fy / zc + cy0)
            if -500 <= u <= 2000 and -500 <= v <= 2000:
                return (u, v)
        return None

    # 1. Project Base Circle (Z = 0)
    base_pixels = []
    for t in theta:
        pt = project_point(workspace_radius * np.cos(t), workspace_radius * np.sin(t), 0.0)
        if pt is not None:
            base_pixels.append(pt)
            
    # 2. Project Catch Circle (Z = z_catch)
    catch_pixels = []
    for t in theta:
        pt = project_point(workspace_radius * np.cos(t), workspace_radius * np.sin(t), z_catch)
        if pt is not None:
            catch_pixels.append(pt)
            
    # 3. Project 4 vertical pillars (at 0, 90, 180, 270 degrees)
    pillars = []
    for angle in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
        pt_base = project_point(workspace_radius * np.cos(angle), workspace_radius * np.sin(angle), 0.0)
        pt_catch = project_point(workspace_radius * np.cos(angle), workspace_radius * np.sin(angle), z_catch)
        if pt_base is not None and pt_catch is not None:
            pillars.append((pt_base, pt_catch))
            
    return {
        "base": base_pixels,
        "catch": catch_pixels,
        "pillars": pillars
    }

def main():
    # 0. Performance Optimization — CPU priority, GPU (OpenCL), threading
    init_performance()

    # 1. Load Configurations
    try:
        config = load_config()
    except Exception as e:
        print(f"[Main] Error loading config: {e}")
        return

    # 2. Initialize Core Components
    ball_detector = BallDetector(config)
    median_filter = MedianFilter3D(window_size=config.get("filter", {}).get("window_size", 5))
    rel_cfg = config.get("release", {})
    vel_t = rel_cfg.get("vel_threshold", 1.2)
    disp_t = rel_cfg.get("displacement_threshold_m", 0.10)
    release_detector = ReleaseDetector(vel_threshold=vel_t, displacement_threshold=disp_t, skin_cfg=rel_cfg)
    predictor = ProjectilePredictor(config)
    robot_tracker = RobotTracker(config)   # HSV Gold color tracker
    
    comms = RobotComms(config)
    comms.start()  # Starts UDP transmitter and heartbeat background thread
    
    profiler = LatencyProfiler()
    visualizer = DebugVisualizer(config)
    
    logger = DataLogger()
    logger.start()

    # Coordinate transformation extrinsic matrices from config
    ext_cfg = config.get("extrinsics", {})
    R_ext = np.array(ext_cfg.get("R", [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float32)
    T_ext = np.array(ext_cfg.get("T", [0,0,0]), dtype=np.float32).reshape(3, 1)

    # UDP loopback for 3D plot (plot_3d.py)
    plot_cfg = config.get("communication", {})
    plot_udp_port = plot_cfg.get("plot_udp_port", 5006)
    plot_sock = stdlib_socket.socket(stdlib_socket.AF_INET, stdlib_socket.SOCK_DGRAM)

    # Load system performance constants from config
    sys_cfg = config.get("system", {})
    max_retries         = sys_cfg.get("camera_max_retries",       5)
    camera_retry_delay  = sys_cfg.get("camera_retry_delay_s",     2.0)
    robot_track_subsamp = sys_cfg.get("robot_tracking_subsample", 15)
    depth_color_subsamp = sys_cfg.get("depth_color_subsample",    6)
    show_hsv_mask       = sys_cfg.get("show_hsv_mask",            False)
    show_robot_mask     = False   # toggle with 'r' key
    show_fit_plot       = True    # toggle with 'f' key — on by default

    # Robot position offset correction (marker ≠ robot reference center)
    tracker_cfg   = config.get("robot_tracker", {})
    robot_offset_x_cm  = float(tracker_cfg.get("offset_x_cm",     0.0))
    robot_offset_y_cm  = float(tracker_cfg.get("offset_y_cm",     0.0))
    robot_max_depth_m  = float(tracker_cfg.get("max_cam_depth_m", 2.0))

    # Systematic position calibration (measured 2026-06-04 with ball_accuracy_test.py)
    # true = (measured - offset) / scale  applied to pos_world (meters)
    cal_cfg     = config.get("calibration", {})
    cal_x_scale = float(cal_cfg.get("x_scale",    1.0))
    cal_y_off   = float(cal_cfg.get("y_offset_m", 0.0))
    cal_y_scale = float(cal_cfg.get("y_scale",    1.0))
    cal_y_z_coup = float(cal_cfg.get("y_z_coupling", 0.0))
    cal_z_off    = float(cal_cfg.get("z_offset_m", 0.0))
    cal_z_scale  = float(cal_cfg.get("z_scale",    1.0))

    # Pipeline Setup
    pipeline = None
    device = None
    q_rgb = None
    q_depth = None

    for attempt in range(max_retries):
        try:
            pipeline, q_rgb, q_depth = create_camera_pipeline(config)
            pipeline.start()
            device = pipeline.getDefaultDevice()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(camera_retry_delay)
            else:
                print(f"[FATAL] Camera connection failed after {max_retries} attempts: {e}")
                comms.stop()
                return
    
    # Retrieve Intrinsic Parameters dynamically from device
    calib = device.readCalibration()
    w = config.get("camera", {}).get("resolution_w", 640)
    h = config.get("camera", {}).get("resolution_h", 360)
    camera_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h))
    dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

    fps_last_time = time.perf_counter()
    fps_counter = 0
    fps = 0.0

    missing_frames = 0
    max_missing_frames = sys_cfg.get("max_missing_frames", 15)

    # Set Zero Calibration System
    is_calibrated = False       # เริ่มต้นในโหมด Calibration
    calibration_time = 0        # เวลาที่กด SET ZERO (รอ 10 วิก่อนส่ง)
    warmup_notified = False     # แจ้งเตือนครบ 10 วิแล้วหรือยัง
    pos_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)        # Ball origin offset
    robot_pos_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Robot marker origin offset
    robot_pos_zero_pending = False  # True = waiting for first marker reading after SET ZERO
    latest_raw_pos = None       # เก็บตำแหน่ง 3D ล่าสุดสำหรับ Set Zero
    latest_pos_camera = None    # เก็บตำแหน่ง camera-space ล่าสุด (raw) สำหรับ height calibration fallback
    pos_camera_filtered = None  # เก็บตำแหน่งฟิลเตอร์ 3D ในระบบกล้องสำหรับคำนวณความสูงอัตโนมัติ
    
    # Transmission limit settings
    release_time = None
    has_sent_this_cycle = False   # One-shot flag: True = already sent prediction this throw
    prediction_history = []       # Rolling buffer of recent predictions for stability check
    PRED_STABLE_N   = 3           # consecutive frames needed for stable classification
    PRED_STABLE_CM  = 3.0         # max std-dev (cm) — within this = converged
    PRED_DEADLINE_S = 0.30        # send immediately if t_land_from_now drops below this
    consecutive_missing = 0       # frames ball has been undetected during flight
    LOSS_SEND_TRIGGER = 3         # tracking-loss fallback: send after this many missed frames

    # Per-throw timing for ClipRecorder
    first_pred_ms  = None   # ms from release to first valid prediction
    stable_pred_ms = None   # ms from release to first sent prediction

    # Axis-lock flags for testing (toggle with 'x' / 'y' keys)
    # When True, that axis is forced to 0 before sending to robot
    freeze_x = False
    freeze_y = False

    # ClipRecorder — auto-records a clip for every throw
    recorder = ClipRecorder(config)

    # DepthBackground — กรองพื้นหลังด้วย depth (กด 'b' เพื่อวัด)
    bg_depth    = DepthBackground(config)
    depth_fg_mask = None   # อัปเดตทุก frame เมื่อ bg_depth.is_ready

    # AdaptiveHSV — ปรับ HSV threshold ตามสีลูกบอลจริง (กด 'c' เพื่อ calibrate)
    _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    hsv_cal   = HSVCalibrator(config)

    # Depth fallback: reuse last valid stereo depth for up to 3 consecutive
    # invalid frames during flight (stereo returns 0 on fast motion / textureless ball)
    last_valid_depth_mm = 0
    depth_fallback_count = 0
    latest_mask = None          # เก็บ HSV mask ล่าสุดสำหรับ HSV Masking window

    # SYSTEM ENGINEERING OPTIMIZATIONS (Sub-sampling counters & caches)
    frame_counter = 0
    latest_robot_pos_cam = None
    latest_robot_pos     = None
    latest_robot_corners = None
    robot_pos_history    = deque(maxlen=10)   # rolling buffer ~5s at 2 Hz (subsamp=15, fps=30)
    latest_color_depth   = None
    latest_traj_plot     = None
    cached_workspace     = None   # C-06: Cached workspace boundary projection (recomputed on SET ZERO only)
    
    # Headless mode: GUI shows during calibration, then auto-closes after SET ZERO
    headless_config = config.get("system", {}).get("headless", False)
    headless = False  # Always start with GUI on (for SET ZERO)

    import msvcrt  # Always import for headless keyboard fallback

    # Disable GC during hot loop — periodic collection every 300 frames prevents memory buildup
    disable_gc()

    try:
        while True:
            frame_counter += 1
            # Periodic GC: collect every 300 frames (~10s at 30fps)
            # Prevents memory buildup without causing random pauses during critical frames
            if frame_counter % 300 == 0:
                gc.collect()
            profiler.start_frame()
            profiler.start_stage("Frame Capture")

            # Get synchronized RGB and Depth frames from our optimized helper
            in_rgb, in_depth = get_synced_frames(q_rgb, q_depth)

            if in_rgb is None or in_depth is None:
                # Keep OpenCV windows responsive even when frames are dropped
                # (prevents Windows "Not Responding" on the GUI windows)
                if not headless:
                    cv2.waitKey(1)
                time.sleep(0.005)
                continue

            frame_rgb = in_rgb.getCvFrame()
            frame_depth = in_depth.getCvFrame()
            
            # CRITICAL SAFETY: Use precise hardware device timestamps instead of host perf_counter
            # to eliminate host-side CPU/OS scheduling jitter and group delay from the parabolic fit.
            current_time = in_rgb.getTimestampDevice().total_seconds()
            profiler.end_stage("Frame Capture")

            # --- Warmup & CAMERA_READY Transmission ---
            warmup_ok = False
            if is_calibrated:
                warmup_ok = (time.time() - calibration_time) >= 10.0
                if warmup_ok:
                    if not warmup_notified:
                        warmup_notified = True
                        comms.send_camera_ready()
                        comms.pending_request_pos = False

            # --- MODULE B0: Depth Background — feed capturing buffer + compute fg mask ---
            if bg_depth.is_capturing:
                bg_depth.feed(frame_depth)
            if bg_depth.is_ready and frame_depth is not None:
                h_rgb = frame_rgb.shape[0]
                w_rgb = frame_rgb.shape[1]
                depth_fg_mask = bg_depth.foreground_mask(frame_depth, (h_rgb, w_rgb))
            else:
                depth_fg_mask = None

            # --- MODULE B1: Adaptive HSV Calibration ---
            if hsv_cal.is_sampling:
                _cal_done = hsv_cal.feed(frame_rgb)   # frame_rgb is BGR from OAK-D
                if _cal_done:
                    hsv_cal.apply_to_detector(ball_detector)
                    hsv_cal.save_to_config(_cfg_path)

            # --- MODULE B & C: HSV Segmentation & Centroid Detection ---
            profiler.start_stage("HSV Segmentation & Blob")
            # Enable ball detection during calibration (not is_calibrated) or after ready (is_calibrated and warmup_ok)
            if (not is_calibrated) or (is_calibrated and warmup_ok):
                mask = ball_detector.detect_red_ball(
                    frame_rgb,
                    use_motion=is_calibrated,
                    depth_fg_mask=depth_fg_mask,
                )
                ball_info = ball_detector.find_ball_centroid(mask)
                latest_mask = mask
            else:
                ball_info = None
            profiler.end_stage("HSV Segmentation & Blob")

            prediction = None
            pos_world = None
            raw_pos_world = None

            if ball_info is not None:
                missing_frames = 0
                consecutive_missing = 0
                cx, cy = ball_info["cx"], ball_info["cy"]

                # --- MODULE D: Depth Lookup & 3D Projection ---
                profiler.start_stage("Depth Lookup & 3D Proj")
                # Sample depth only from pixels within the color-segmented contour
                z_mm = BallDetector.contour_depth_sample(
                    frame_depth, ball_info["contour"], ball_info["bbox"]
                )

                # Front-surface bias correction:
                # Stereo depth measures the ball's nearest surface, not its centre.
                # Using the pinhole model  R = r_px * Z / f  (where r_px is the
                # projected pixel radius, Z is depth, f is focal length), we can
                # estimate the physical radius and shift the depth to the centre.
                # Clamped to 80 mm (8 cm max radius) to avoid over-correcting for
                # noise-inflated blob areas.
                if z_mm > 0:
                    fx = camera_matrix[0, 0]
                    r_px = float(np.sqrt(ball_info["area"] / np.pi))
                    r_correction_mm = min(r_px * z_mm / fx, 80.0)
                    z_mm += r_correction_mm

                    # Frame-to-frame depth outlier rejection:
                    # Caps the depth change per frame to a physically plausible limit.
                    # Before release (stationary / accuracy test): 150 mm/frame cap
                    #   — rejects stereo noise spikes (ball doesn't move >15 cm/frame)
                    # After release (ball in flight): 400 mm/frame cap
                    #   — allows fast balls up to ~12 m/s closing speed at 30 fps
                    if last_valid_depth_mm > 0:
                        max_depth_jump = 400.0 if release_detector.released else 150.0
                        if abs(z_mm - last_valid_depth_mm) > max_depth_jump:
                            z_mm = last_valid_depth_mm

                    last_valid_depth_mm = z_mm
                    depth_fallback_count = 0
                elif release_detector.released and last_valid_depth_mm > 0 and depth_fallback_count < 3:
                    # Stereo invalid this frame — reuse last known depth during flight
                    z_mm = last_valid_depth_mm
                    depth_fallback_count += 1

                ball_info["depth_m"] = z_mm / 1000.0

                if z_mm > 0:
                    # Undistort pixels before projection
                    pts_px = np.array([[[cx, cy]]], dtype=np.float32)
                    undistorted_pts = cv2.undistortPoints(pts_px, camera_matrix, dist_coeffs, P=camera_matrix)
                    ucx = undistorted_pts[0][0][0]
                    ucy = undistorted_pts[0][0][1]

                    # Project: Camera coordinates in meters
                    z_m = z_mm / 1000.0
                    fx = camera_matrix[0, 0]
                    fy = camera_matrix[1, 1]
                    cx0 = camera_matrix[0, 2]
                    cy0 = camera_matrix[1, 2]
                    x_c = (ucx - cx0) * z_m / fx
                    y_c = (ucy - cy0) * z_m / fy
                    pos_camera = np.array([x_c, y_c, z_m])
                    profiler.end_stage("Depth Lookup & 3D Proj")

                    # Calculate raw world position (with zero offset subtracted)
                    raw_pos_world = (R_ext @ pos_camera.reshape(3, 1) + T_ext).flatten() - pos_zero

                    # --- MODULE E: Median Filter ---
                    profiler.start_stage("Median Filtering")
                    pos_camera_filtered = median_filter.update(pos_camera)
                    profiler.end_stage("Median Filtering")

                    # Apply Extrinsics mapping: Camera -> World/Robot system
                    pos_world = (R_ext @ pos_camera_filtered.reshape(3, 1) + T_ext).flatten()

                    # เก็บตำแหน่งดิบสำหรับ Set Zero
                    latest_raw_pos = pos_world.copy()
                    latest_pos_camera = pos_camera.copy()

                    # Apply zero offset (Set Zero calibration)
                    pos_world = pos_world - pos_zero

                    # Apply systematic calibration corrections
                    pos_world[0] = pos_world[0] / cal_x_scale
                    # Y: offset + Z-dependent coupling (camera looks up at high Z → depth underestimate)
                    pos_world[1] = (pos_world[1] - cal_y_off - cal_y_z_coup * pos_world[2]) / cal_y_scale
                    pos_world[2] = (pos_world[2] - cal_z_off) / cal_z_scale

                    # Send 3D position to plot_3d.py via UDP loopback
                    x_cm = pos_world[0] * 100.0
                    y_cm = pos_world[1] * 100.0
                    z_cm = pos_world[2] * 100.0

                    # --- MODULE F: Release Detection ---
                    profiler.start_stage("Release Detection")
                    released = release_detector.update(pos_world, current_time, frame_bgr=frame_rgb, ball_info=ball_info)
                    profiler.end_stage("Release Detection")

                    # --- MODULE G: Projectile Predictor ---
                    # ไม่รัน predictor ก่อน SET ZERO — กันข้อมูล noise ตอน calibration
                    if not is_calibrated:
                        released = False
                    if released:
                        if release_time is None:
                            release_time = current_time
                            first_pred_ms  = None
                            stable_pred_ms = None

                        elapsed_since_release = current_time - release_time

                        profiler.start_stage("Curve Fitting & Pred")
                        predictor.add_point(pos_world, current_time)
                        prediction = predictor.predict_landing()
                        profiler.end_stage("Curve Fitting & Pred")

                        if prediction is not None:
                            prediction["elapsed_since_release"] = elapsed_since_release
                            prediction_history.append(prediction)

                            # Track: release → first valid prediction (ms)
                            if first_pred_ms is None:
                                first_pred_ms = elapsed_since_release * 1000.0

                            # C-13: Timeout recovery — if READY never came back, reset robot_ready
                            if not comms.robot_ready and comms.last_target_time > 0:
                                if (time.time() - comms.last_target_time) > 10.0:
                                    comms.robot_ready = True

                            if (is_calibrated and warmup_ok
                                    and not has_sent_this_cycle and comms.robot_ready):

                                px_cm = prediction["x"] * 100.0
                                py_cm = prediction["y"] * 100.0
                                should_send = False

                                # Stable: last PRED_STABLE_N predictions converge within PRED_STABLE_CM
                                if len(prediction_history) >= PRED_STABLE_N:
                                    recent = prediction_history[-PRED_STABLE_N:]
                                    std_x = float(np.std([p["x"] for p in recent])) * 100.0
                                    std_y = float(np.std([p["y"] for p in recent])) * 100.0
                                    if std_x <= PRED_STABLE_CM and std_y <= PRED_STABLE_CM:
                                        px_cm = float(np.mean([p["x"] for p in recent])) * 100.0
                                        py_cm = float(np.mean([p["y"] for p in recent])) * 100.0
                                        should_send = True

                                # Hard deadline: if ball is about to land, send best available now
                                if prediction["t_land_from_now"] <= PRED_DEADLINE_S:
                                    should_send = True

                                if should_send:
                                    profiler.start_stage("Transmission")
                                    tx_x = 0.0 if freeze_x else px_cm
                                    tx_y = 0.0 if freeze_y else py_cm
                                    comms.send_target(tx_x, tx_y)
                                    has_sent_this_cycle = True
                                    # Track: release → first sent prediction (ms)
                                    if stable_pred_ms is None:
                                        stable_pred_ms = elapsed_since_release * 1000.0
                                    freeze_tag = (f" [FREEZE X]" if freeze_x else "") + (f" [FREEZE Y]" if freeze_y else "")
                                    print(f"[TX] ✅ BALL_POS sent → X={tx_x:.1f}cm Y={tx_y:.1f}cm (pred X={px_cm:.1f} Y={py_cm:.1f}){freeze_tag} delay={elapsed_since_release:.3f}s")
                                    profiler.end_stage("Transmission")

                    # Send plot data (with or without prediction)
                    try:
                        if prediction is not None:
                            pred_x_cm = prediction["x"] * 100.0
                            pred_y_cm = prediction["y"] * 100.0
                            pred_z_cm = prediction["z"] * 100.0
                            plot_msg = f"{x_cm:.1f},{y_cm:.1f},{z_cm:.1f},{pred_x_cm:.1f},{pred_y_cm:.1f},{pred_z_cm:.1f}"
                        else:
                            plot_msg = f"{x_cm:.1f},{y_cm:.1f},{z_cm:.1f},None,None,None"
                        # Append robot position (world-relative, with offset)
                        if is_calibrated and latest_robot_pos is not None:
                            rx_w = latest_robot_pos - robot_pos_zero
                            rx_plot = float(rx_w[0] / cal_x_scale) * 100.0 + robot_offset_x_cm
                            ry_plot = float(rx_w[1] / cal_y_scale) * 100.0 + robot_offset_y_cm
                            plot_msg += f",{rx_plot:.1f},{ry_plot:.1f}"
                        else:
                            plot_msg += ",None,None"
                        plot_sock.sendto(plot_msg.encode(), ("127.0.0.1", plot_udp_port))
                    except Exception:
                        pass
                else:
                    profiler.end_stage("Depth Lookup & 3D Proj")
            else:
                # Ball not detected this frame — still send robot position to keep plot fresh
                consecutive_missing += 1
                if is_calibrated and latest_robot_pos is not None:
                    try:
                        rx_w = latest_robot_pos - robot_pos_zero
                        rx_plot = float(rx_w[0] / cal_x_scale) * 100.0 + robot_offset_x_cm
                        ry_plot = float(rx_w[1] / cal_y_scale) * 100.0 + robot_offset_y_cm
                        plot_sock.sendto(
                            f"None,None,None,None,None,None,{rx_plot:.1f},{ry_plot:.1f}".encode(),
                            ("127.0.0.1", plot_udp_port)
                        )
                    except Exception:
                        pass

            # ── Tracking-loss fallback: ลูกหายกลางอากาศก่อนส่ง prediction ──
            # ถ้าหายติดต่อกัน LOSS_SEND_TRIGGER frames ระหว่างบิน ส่ง best prediction ทันที
            if (consecutive_missing >= LOSS_SEND_TRIGGER
                    and release_time is not None
                    and not has_sent_this_cycle
                    and comms.robot_ready
                    and is_calibrated and warmup_ok
                    and len(prediction_history) > 0):
                n = min(len(prediction_history), PRED_STABLE_N)
                recent = prediction_history[-n:]
                px_cm = float(np.mean([p["x"] for p in recent])) * 100.0
                py_cm = float(np.mean([p["y"] for p in recent])) * 100.0
                tx_x = 0.0 if freeze_x else px_cm
                tx_y = 0.0 if freeze_y else py_cm
                comms.send_target(tx_x, tx_y)
                has_sent_this_cycle = True
                print(f"[TX] ✅ BALL_POS sent (tracking-loss) → X={px_cm:.1f}cm Y={py_cm:.1f}cm")

            # ── Auto-Reset: ส่งเสร็จ + หุ่นพร้อม → reset ทันที ──
            # เมื่อส่ง prediction ไปแล้ว (has_sent_this_cycle=True)
            # และ ESP32 ตอบ OKAY กลับมา (robot_ready=True)
            # → reset ทุกอย่างเพื่อพร้อมรับลูกถัดไป (ไม่ต้องรอลูกหาย)
            if has_sent_this_cycle and comms.robot_ready:
                release_detector.reset()
                predictor.reset()
                median_filter.reset()
                visualizer.reset_trail()
                release_time = None
                has_sent_this_cycle = False
                prediction_history.clear()
                missing_frames = 0
                consecutive_missing = 0
                prediction = None
                first_pred_ms  = None
                stable_pred_ms = None

            else:
                missing_frames += 1
            # Fallback: ถ้าลูกหายจากภาพนานเกินไป → reset อยู่ดี
            if missing_frames > max_missing_frames:
                release_detector.reset()
                predictor.reset()
                median_filter.reset()
                visualizer.reset_trail()
                release_time = None
                has_sent_this_cycle = False
                prediction_history.clear()
                consecutive_missing = 0
                last_valid_depth_mm = 0   # clear depth anchor — ball re-appears at a new position

            # --- MODULE H: Robot Position Tracking (DISABLED) ---
            # ปิดการ track green marker ชั่วคราว — robot pos ส่งเป็น 0,0 เสมอ
            robot_pos_cam = None
            robot_pos     = None
            robot_corners = None

            # --- MODULE H2: Respond to REQUEST_POS from ESP32 ---
            # ESP32 ส่ง "REQUEST_POS" เมื่อวิ่งกลับ Home
            # Vision ตอบกลับด้วยพิกัดหุ่นปัจจุบันจาก HSV Green tracker (relative to robot_pos_zero)
            if comms.pending_request_pos and is_calibrated and warmup_ok:
                # Force robot position to always 0,0 (always at home)
                print(f"[RobotPos] → rx=+0.0cm  ry=+0.0cm (forced)")
                comms.send_robot_pos(0.0, 0.0)

            # Respond to WAITING_FOR_CAMERA from ESP32
            if comms.pending_wait_camera:
                if is_calibrated and warmup_ok:
                    comms.send_camera_ready()
                comms.pending_wait_camera = False
            fps_counter += 1
            if current_time - fps_last_time >= 1.0:
                fps = fps_counter / (current_time - fps_last_time)
                fps_counter = 0
                fps_last_time = current_time

            profiler.end_frame()

            # --- MODULE J: Visualizer Overlays ---
            profiler.start_stage("GUI Visualizer")

            if not headless:
                # Project parabolic curve in 3D to 2D camera pixel coordinates
                projected_curve = None
                if prediction is not None and is_calibrated:
                    projected_curve = project_parabolic_curve(
                        prediction, pos_zero, R_ext, T_ext, camera_matrix
                    )

                # Project workspace safety circle at catch height (z_catch)
                # C-06: Use cached projection (computed once on SET ZERO)
                projected_workspace = cached_workspace

                # Use relative robot position for display if calibrated
                robot_pos_display = (robot_pos - robot_pos_zero) if (is_calibrated and robot_pos is not None) else robot_pos

                annotated_rgb = visualizer.draw_all(
                    frame_rgb,
                    ball_info,
                    predictor.buffer,
                    prediction,
                    robot_pos_display,
                    robot_corners,
                    fps,
                    profiler.get_latest(),
                    comms.last_error,
                    release_detector.released,
                    projected_curve=projected_curve,
                    projected_workspace=projected_workspace
                )
                # SYSTEM OPTIMIZATION: Sub-sample CPU-heavy depth colorization
                if frame_counter % depth_color_subsamp == 0 or latest_color_depth is None:
                    color_depth = visualizer.colorize_depth(frame_depth, motion_mask=ball_detector.motion_mask)
                    latest_color_depth = color_depth
                else:
                    color_depth = latest_color_depth

                visualizer.show_frames(annotated_rgb, color_depth)

                # Axis-lock overlay (top-center) — visible when any axis is frozen
                if freeze_x or freeze_y:
                    lock_parts = (["X=0"] if freeze_x else []) + (["Y=0"] if freeze_y else [])
                    lock_text = "TEST LOCK: " + "  ".join(lock_parts)
                    cv2.putText(annotated_rgb, lock_text,
                                (annotated_rgb.shape[1] // 2 - 95, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)

                # Background depth status overlay (bottom-left, above HUD)
                _bh = annotated_rgb.shape[0]
                if bg_depth.is_capturing:
                    pct = int(bg_depth.progress * 100)
                    bar_w = int(bg_depth.progress * 160)
                    cv2.rectangle(annotated_rgb, (10, _bh - 160), (170, _bh - 148), (60, 60, 60), -1)
                    cv2.rectangle(annotated_rgb, (10, _bh - 160), (10 + bar_w, _bh - 148), (0, 220, 255), -1)
                    cv2.putText(annotated_rgb, f"BG capture {pct}%  (hold still)",
                                (10, _bh - 163),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
                elif bg_depth.is_ready:
                    cv2.putText(annotated_rgb, "BG depth: READY  [b]=recapture",
                                (10, _bh - 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 80), 1, cv2.LINE_AA)
                else:
                    cv2.putText(annotated_rgb, "BG depth: --  press [b] to capture",
                                (10, _bh - 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

                # Adaptive HSV calibration ROI overlay
                hsv_cal.draw_overlay(annotated_rgb)

                # Feed annotated frame to clip recorder (only after calibration)
                if is_calibrated:
                    recorder.feed_frame(
                        frame_bgr=annotated_rgb,
                        ball_detected=(ball_info is not None),
                        prediction=prediction,
                        released=release_detector.released,
                        elapsed_since_release=(current_time - release_time) if release_time else 0.0,
                        first_pred_ms=first_pred_ms,
                        stable_pred_ms=stable_pred_ms,
                    )

                # HSV Masking window (for presentation — toggle with 'm')
                if show_hsv_mask and latest_mask is not None:
                    visualizer.show_hsv_mask_window(
                        frame_rgb, latest_mask,
                        motion_mask=ball_detector.motion_mask,
                        ball_info=ball_info
                    )

                # Trajectory Fit Plot window (toggle with 'f')
                if show_fit_plot and len(predictor.buffer) >= 2:
                    visualizer.show_fit_plot_window(predictor.buffer, prediction)

                # Calibration mode overlay
                if not is_calibrated:
                    cv2.putText(annotated_rgb, "CALIBRATION MODE", (10, h - 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    cv2.putText(annotated_rgb, "Place ball at Robot Center (0,0)", (10, h - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(annotated_rgb, "Press 'z' to SET ZERO", (10, h - 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("MCE14-Vission-16 (RGB Feed)", annotated_rgb)

            # Record log row (always, even in headless)
            logger.log(
                timestamp=current_time,
                is_calibrated=is_calibrated,
                raw_pos=raw_pos_world,
                filt_pos=pos_world,
                is_released=release_detector.released,
                prediction=prediction
            )
            # Periodic flush: write buffered CSV data to disk every 90 frames (~3s)
            if frame_counter % 90 == 0:
                logger.flush()
            profiler.end_stage("GUI Visualizer")

            # Check for keys
            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Headless: read key from terminal (non-blocking)
                if msvcrt.kbhit():
                    key = ord(msvcrt.getch().lower())
                    if key == ord('q'):
                        break
                else:
                    key = 255

            if key == ord('m'):
                show_hsv_mask = not show_hsv_mask
                if not show_hsv_mask:
                    cv2.destroyWindow("MCE14 — HSV Masking")

            if key == ord('r'):
                show_robot_mask = not show_robot_mask
                if not show_robot_mask:
                    cv2.destroyWindow("Robot Green Mask")

            if key == ord('f'):
                show_fit_plot = not show_fit_plot
                if not show_fit_plot:
                    cv2.destroyWindow("Trajectory Fit")

            if key == ord('b'):
                bg_depth.start_capture()

            if key == ord('c'):
                hsv_cal.start()

            if key == ord('x'):
                freeze_x = not freeze_x
                state = "LOCKED=0" if freeze_x else "FREE"
                print(f"[TestMode] X axis → {state}  (Y={'LOCKED=0' if freeze_y else 'FREE'})")

            if key == ord('y'):
                freeze_y = not freeze_y
                state = "LOCKED=0" if freeze_y else "FREE"
                print(f"[TestMode] Y axis → {state}  (X={'LOCKED=0' if freeze_x else 'FREE'})")

            if key == ord('z'):
                # Set Zero: ตั้งตำแหน่งปัจจุบันเป็นจุด origin (0,0,0) โดยใช้ลูกบอล
                if latest_raw_pos is not None:
                    # AUTOMATIC HEIGHT CALIBRATION:
                    # The vertical height of the camera above the floor is exactly the camera-space y coordinate
                    # of the ball when it is placed on the table/floor surface (since the camera Y axis points down).
                    if pos_camera_filtered is not None:
                        measured_height = float(pos_camera_filtered[1])
                        T_ext[2, 0] = measured_height
                        pos_world_calibrated = (R_ext @ pos_camera_filtered.reshape(3, 1) + T_ext).flatten()
                        pos_zero = pos_world_calibrated.copy()
                        try:
                            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                            if not os.path.exists(config_path):
                                config_path = "config.yaml"
                            with open(config_path, "r", encoding="utf-8") as f:
                                config_data = yaml.safe_load(f)
                            config_data["extrinsics"]["T"][2] = float(measured_height)
                            with open(config_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(config_data, f, default_flow_style=False)
                        except Exception:
                            pass
                    else:
                        if latest_pos_camera is not None:
                            measured_height = float(latest_pos_camera[1])
                            T_ext[2, 0] = measured_height
                            pos_world_calibrated = (R_ext @ latest_pos_camera.reshape(3, 1) + T_ext).flatten()
                            pos_zero = pos_world_calibrated.copy()
                        else:
                            pos_zero = latest_raw_pos.copy()

                    # robot_pos_zero will be set from first green marker reading after SET ZERO
                    robot_pos_zero_pending = True
                    robot_pos_zero = pos_zero.copy()  # fallback if marker never detected

                    is_calibrated = True
                    calibration_time = time.time()
                    warmup_notified = False

                    # Reset tracking states
                    release_detector.reset()
                    predictor.reset()
                    median_filter.reset()
                    visualizer.reset_trail()
                    release_time = None
                    has_sent_this_cycle = False
                    prediction_history.clear()
                    robot_pos_history.clear()
                    consecutive_missing = 0
                    # C-06: Pre-compute workspace boundary once (never changes after SET ZERO)
                    cached_workspace = project_workspace_boundary(
                        pos_zero, predictor.z_catch, predictor.workspace_radius_m,
                        R_ext, T_ext, camera_matrix
                    )

                    # ── Auto-switch to headless after SET ZERO ──
                    if headless_config and not headless:
                        headless = True
                        cv2.destroyAllWindows()
                        competition_mode()

    except KeyboardInterrupt:
        pass
    finally:
        cleanup_performance()   # Restore timer resolution + re-enable GC
        recorder.stop()
        logger.stop()
        comms.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
