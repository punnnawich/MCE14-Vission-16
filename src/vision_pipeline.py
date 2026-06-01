import cv2
import numpy as np
import yaml
import time
import os
import socket as stdlib_socket
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
            print(f"[Recovery] Closing stale BOOTED device: {dev_info.name}")
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

    # Map string preset to depthai PresetMode
    preset_map = {
        "ACCURACY": dai.node.StereoDepth.PresetMode.ACCURACY,
        "DEFAULT": dai.node.StereoDepth.PresetMode.DEFAULT,
        "DENSITY": dai.node.StereoDepth.PresetMode.DENSITY,
        "FACE": dai.node.StereoDepth.PresetMode.FACE,
        "FAST_ACCURACY": dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
        "FAST_DENSITY": dai.node.StereoDepth.PresetMode.FAST_DENSITY,
        "HIGH_DETAIL": dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
        "ROBOTICS": dai.node.StereoDepth.PresetMode.ROBOTICS,
    }
    preset_mode = preset_map.get(preset_str.upper(), dai.node.StereoDepth.PresetMode.DEFAULT)

    # 1. RGB Camera (unified Camera node replaces ColorCamera)
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((width, height), type=dai.ImgFrame.Type.BGR888i, fps=fps)

    # 2. Stereo Depth Node (autoCreateCameras handles mono cameras)
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
        print(f"[Camera] INFO: subpixel=true → median filter auto-disabled "
              f"(hardware limit: 1024 < 32×disparity). Set median_filter: 0 in config to suppress this message.")
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

    # 3. If queues were empty, block-read to wait for new frames
    if last_rgb is None:
        try:
            last_rgb = q_rgb.get()
        except Exception:
            return None, None
            
    if last_depth is None:
        try:
            last_depth = q_depth.get()
        except Exception:
            return None, None

    # 4. Align frames by matching their sequence numbers (max 10 search attempts)
    for _ in range(10):
        seq_rgb = last_rgb.getSequenceNum()
        seq_depth = last_depth.getSequenceNum()
        
        if seq_rgb == seq_depth:
            return last_rgb, last_depth
        elif seq_rgb < seq_depth:
            # RGB is older, get a newer RGB frame
            next_rgb = q_rgb.tryGet()
            if next_rgb is None:
                try:
                    next_rgb = q_rgb.get()
                except Exception:
                    return None, None
            last_rgb = next_rgb
        else:
            # Depth is older, get a newer Depth frame
            next_depth = q_depth.tryGet()
            if next_depth is None:
                try:
                    next_depth = q_depth.get()
                except Exception:
                    return None, None
            last_depth = next_depth

    return last_rgb, last_depth

def project_parabolic_curve(prediction, pos_zero, R_ext, T_ext, camera_matrix):
    """
    Evaluates the fitted 3D parabola from the current time to the impact time,
    transforms it to camera space, and projects it into 2D camera pixel coordinates.
    """
    if prediction is None:
        return None
        
    coeff_x = np.array(prediction.get("coeff_x"))
    coeff_y = np.array(prediction.get("coeff_y"))
    coeff_z = np.array(prediction.get("coeff_z"))
    t_start = prediction.get("t_start")
    t_land = prediction.get("t_land")
    
    # Generate time steps from start to landing time
    t_steps = np.linspace(t_start, t_land, 30)
    curve_pixels = []
    
    # R_ext transpose for inverse transform (Camera <- World)
    R_inv = R_ext.T
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]
    
    for t in t_steps:
        # 1. Evaluate 3D position in normalized predictor world space
        x_p = np.polyval(coeff_x, t)
        y_p = np.polyval(coeff_y, t)
        z_p = np.polyval(coeff_z, t)
        
        # Apply air drag correction proportionally from start to landing
        x0 = coeff_x[1]
        y0 = coeff_y[1]
        if t_land > t_start:
            progress = (t - t_start) / (t_land - t_start)
            drag_factor = 1.0 - (1.0 - 0.92) * progress
        else:
            drag_factor = 1.0
            
        x_world = x0 + (x_p - x0) * drag_factor + pos_zero[0]
        y_world = y0 + (y_p - y0) * drag_factor + pos_zero[1]
        z_world = z_p + pos_zero[2]
        
        # 2. Transform World to Camera space: P_camera = R_inv @ (P_world - T_ext)
        pos_w = np.array([x_world, y_world, z_world])
        pos_c = R_inv @ (pos_w.reshape(3, 1) - T_ext)
        xc, yc, zc = pos_c.flatten()
        
        # 3. Project to pixel space
        if zc > 0.1: # Protect against behind camera
            u = int(xc * fx / zc + cx0)
            v = int(yc * fy / zc + cy0)
            if -500 <= u <= 2000 and -500 <= v <= 2000:
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
    vel_t = rel_cfg.get("vel_threshold", 1.5)
    disp_t = rel_cfg.get("displacement_threshold_m", 0.15)
    release_detector = ReleaseDetector(vel_threshold=vel_t, displacement_threshold=disp_t)
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

    # Pipeline Setup
    print("[Main] Connecting to OAK-D Lite camera...")
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
                print(f"[Main] Connection attempt {attempt+1} failed ({e}). Retrying in {camera_retry_delay:.0f}s...")
                time.sleep(camera_retry_delay)
            else:
                print(f"\n[FATAL] Could not connect to DepthAI OAK-D camera after {max_retries} attempts: {e}")
                print("Please ensure your OAK-D Lite is plugged in and try again.")
                comms.stop()
                return

    print("[Main] Camera connected successfully. Setting up calibration details...")
    
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
    pos_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Offset จุด origin
    latest_raw_pos = None       # เก็บตำแหน่ง 3D ล่าสุดสำหรับ Set Zero
    pos_camera_filtered = None  # เก็บตำแหน่งฟิลเตอร์ 3D ในระบบกล้องสำหรับคำนวณความสูงอัตโนมัติ
    
    # Transmission limit settings
    release_time = None
    max_transmission_delay_s = config.get("release", {}).get("max_transmission_delay_s", 0.3)
    
    # SYSTEM ENGINEERING OPTIMIZATIONS (Sub-sampling counters & caches)
    frame_counter = 0
    latest_robot_pos     = None
    latest_robot_corners = None
    latest_color_depth   = None
    latest_traj_plot     = None
    
    print("[Main] System initialized. Entering real-time vision loop.")
    print("[Main] Press 'z' to SET ZERO (place ball at robot center first)")
    print("[Main] Press 'q' to quit.")

    try:
        while True:
            frame_counter += 1
            profiler.start_frame()
            profiler.start_stage("Frame Capture")

            # Get synchronized RGB and Depth frames from our optimized helper
            in_rgb, in_depth = get_synced_frames(q_rgb, q_depth)

            if in_rgb is None or in_depth is None:
                print("[Main] Warning: Could not retrieve synced frame pair. Skipping frame...")
                time.sleep(0.01)
                continue

            frame_rgb = in_rgb.getCvFrame()
            frame_depth = in_depth.getCvFrame()
            
            # CRITICAL SAFETY: Use precise hardware device timestamps instead of host perf_counter
            # to eliminate host-side CPU/OS scheduling jitter and group delay from the parabolic fit.
            current_time = in_rgb.getTimestampDevice().total_seconds()
            profiler.end_stage("Frame Capture")

            # --- MODULE B & C: HSV Segmentation & Centroid Detection ---
            profiler.start_stage("HSV Segmentation & Blob")
            mask = ball_detector.detect_red_ball(frame_rgb, use_motion=is_calibrated)
            ball_info = ball_detector.find_ball_centroid(mask)
            profiler.end_stage("HSV Segmentation & Blob")

            prediction = None
            pos_world = None
            raw_pos_world = None

            if ball_info is not None:
                missing_frames = 0
                cx, cy = ball_info["cx"], ball_info["cy"]

                # --- MODULE D: Depth Lookup & 3D Projection ---
                profiler.start_stage("Depth Lookup & 3D Proj")
                # Adaptive depth sampling: larger ROI for distant balls, closest-half median
                z_mm = BallDetector.adaptive_depth_sample(
                    frame_depth, cx, cy, ball_info["area"], h, w
                )

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

                    # Apply zero offset (Set Zero calibration)
                    pos_world = pos_world - pos_zero

                    # Send 3D position to plot_3d.py via UDP loopback
                    x_cm = pos_world[0] * 100.0
                    y_cm = pos_world[1] * 100.0
                    z_cm = pos_world[2] * 100.0

                    # --- MODULE F: Release Detection ---
                    profiler.start_stage("Release Detection")
                    released = release_detector.update(pos_world, current_time)
                    profiler.end_stage("Release Detection")

                    # --- MODULE G: Projectile Predictor ---
                    if released:
                        if release_time is None:
                            release_time = current_time
                        
                        elapsed_since_release = current_time - release_time
                        
                        profiler.start_stage("Curve Fitting & Pred")
                        predictor.add_point(pos_world, current_time)
                        prediction = predictor.predict_landing()
                        profiler.end_stage("Curve Fitting & Pred")

                        if prediction is not None:
                            prediction["elapsed_since_release"] = elapsed_since_release
                            
                            # Only transmit target if:
                            #  - within max_transmission_delay_s window (0.3s)
                            #  - ESP32 has sent OKAY (robot_ready == True)
                            if (elapsed_since_release <= max_transmission_delay_s
                                    and comms.robot_ready):
                                profiler.start_stage("Transmission")
                                px_cm = prediction["x"] * 100.0
                                py_cm = prediction["y"] * 100.0
                                comms.send_target(px_cm, py_cm)
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
                        plot_sock.sendto(plot_msg.encode(), ("127.0.0.1", plot_udp_port))
                    except Exception:
                        pass
                else:
                    profiler.end_stage("Depth Lookup & 3D Proj")
            else:
                missing_frames += 1
                # If ball is missing for a while, reset tracking states
                if missing_frames > max_missing_frames:
                    release_detector.reset()
                    predictor.reset()
                    median_filter.reset()
                    visualizer.reset_trail()
                    release_time = None

            # --- MODULE H: Robot Position Tracking (HSV Gold) ---
            # Sub-sample: รัน 1 ครั้งทุก robot_track_subsamp frames (~4 FPS)
            # หยุดทำงานขณะลูกบินเพื่อรักษา CPU
            if not release_detector.released and (frame_counter % robot_track_subsamp == 0):
                profiler.start_stage("Robot Tracking")
                robot_pos, robot_corners = robot_tracker.track(
                    frame_rgb,
                    frame_depth,
                    camera_matrix,
                    dist_coeffs
                )
                # Map robot position to world coordinates if detected
                if robot_pos is not None and robot_pos[2] > 0.0:
                    robot_pos = (R_ext @ robot_pos.reshape(3, 1) + T_ext).flatten()
                profiler.end_stage("Robot Tracking")
                latest_robot_pos     = robot_pos
                latest_robot_corners = robot_corners
            else:
                robot_pos     = latest_robot_pos
                robot_corners = latest_robot_corners

            # --- MODULE H2: Respond to REQUEST_POS from ESP32 ---
            # ESP32 ส่ง "REQUEST_POS" เมื่อวิ่งกลับ Home (S_BACK)
            # Vision ตอบกลับด้วยพิกัดหุ่นปัจจุบันจาก HSV Gold tracker
            if comms.pending_request_pos:
                if robot_pos is not None:
                    rx_world = (robot_pos - pos_zero) if is_calibrated else robot_pos
                    rx_cm = float(rx_world[0]) * 100.0
                    ry_cm = float(rx_world[1]) * 100.0
                    comms.send_robot_pos(rx_cm, ry_cm)
                else:
                    comms.send_robot_pos(0.0, 0.0)
                    print("[Main] WARNING: REQUEST_POS received but robot not visible — sending (0,0)")
            fps_counter += 1
            if current_time - fps_last_time >= 1.0:
                fps = fps_counter / (current_time - fps_last_time)
                fps_counter = 0
                fps_last_time = current_time

            profiler.end_frame()

            # --- MODULE J: Visualizer Overlays ---
            profiler.start_stage("GUI Visualizer")
            
            # Project parabolic curve in 3D to 2D camera pixel coordinates
            projected_curve = None
            if prediction is not None and is_calibrated:
                projected_curve = project_parabolic_curve(
                    prediction, pos_zero, R_ext, T_ext, camera_matrix
                )

            # Project workspace safety circle at catch height (z_catch)
            projected_workspace = None
            if is_calibrated:
                projected_workspace = project_workspace_boundary(
                    pos_zero, predictor.z_catch, predictor.workspace_radius_m,
                    R_ext, T_ext, camera_matrix
                )

            annotated_rgb = visualizer.draw_all(
                frame_rgb,
                ball_info,
                predictor.buffer,
                prediction,
                robot_pos,
                robot_corners,
                fps,
                profiler.get_latest(),
                comms.last_error,
                release_detector.released,
                projected_curve=projected_curve,
                projected_workspace=projected_workspace
            )
            # SYSTEM OPTIMIZATION: Sub-sample CPU-heavy depth colorization to 10 FPS (every 6 frames at 60 FPS)
            if frame_counter % depth_color_subsamp == 0 or latest_color_depth is None:
                color_depth = visualizer.colorize_depth(frame_depth, motion_mask=ball_detector.motion_mask)
                latest_color_depth = color_depth
            else:
                color_depth = latest_color_depth
                
            visualizer.show_frames(annotated_rgb, color_depth)
            # Debug: Show HSV mask for tuning detection parameters
            cv2.imshow("MCE14-Vission-16 (HSV Mask)", mask)

            # Calibration mode overlay
            if not is_calibrated:
                cv2.putText(annotated_rgb, "CALIBRATION MODE", (10, h - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.putText(annotated_rgb, "Place ball at Robot Center (0,0)", (10, h - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated_rgb, "Press 'z' to SET ZERO", (10, h - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("MCE14-Vission-16 (RGB Feed)", annotated_rgb)
            
            # Record log row
            logger.log(
                timestamp=current_time,
                is_calibrated=is_calibrated,
                raw_pos=raw_pos_world,
                filt_pos=pos_world,
                is_released=release_detector.released,
                prediction=prediction
            )
            profiler.end_stage("GUI Visualizer")

            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                # Set Zero: ตั้งตำแหน่งปัจจุบันเป็นจุด origin (0,0,0)
                if latest_raw_pos is not None:
                    # AUTOMATIC HEIGHT CALIBRATION:
                    # The vertical height of the camera above the floor is exactly the camera-space y coordinate
                    # of the ball when it is placed on the table/floor surface (since the camera Y axis points down).
                    if pos_camera_filtered is not None:
                        measured_height = float(pos_camera_filtered[1])
                        print(f"\n[Auto-Height Calibration] Auto-detecting camera height...")
                        print(f"  Old configured height: {T_ext[2, 0]*100:.1f} cm")
                        T_ext[2, 0] = measured_height
                        print(f"  New physical camera height calibrated: {T_ext[2, 0]*100:.1f} cm")
                        
                        # Re-evaluate absolute coordinates with the newly calibrated height
                        pos_world_calibrated = (R_ext @ pos_camera_filtered.reshape(3, 1) + T_ext).flatten()
                        pos_zero = pos_world_calibrated.copy()
                        
                        # Persist calibrated height back to config.yaml
                        try:
                            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                            if not os.path.exists(config_path):
                                config_path = "config.yaml"
                            
                            with open(config_path, "r", encoding="utf-8") as f:
                                config_data = yaml.safe_load(f)
                            
                            config_data["extrinsics"]["T"][2] = float(measured_height)
                            
                            with open(config_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(config_data, f, default_flow_style=False)
                            print(f"  [Persisted] Saved new height {measured_height*100:.1f}cm back to config.yaml!")
                        except Exception as e:
                            print(f"  [Warning] Could not persist new height: {e}")
                    else:
                        pos_zero = latest_raw_pos.copy()
                        
                    is_calibrated = True
                    # z_catch stays at 0.25m (25cm catch height from config)
                    print(f"\n{'='*50}")
                    print(f"  [SET ZERO] Zero calibration successful!")
                    print(f"  Origin offset: X={pos_zero[0]*100:.1f}cm, Y={pos_zero[1]*100:.1f}cm, Z={pos_zero[2]*100:.1f}cm")
                    print(f"  z_catch: {predictor.z_catch*100:.0f}cm (catch height)")
                    print(f"{'='*50}\n")
                    # Reset tracking states
                    release_detector.reset()
                    predictor.reset()
                    median_filter.reset()
                    visualizer.reset_trail()
                    release_time = None
                else:
                    print("[SET ZERO FAILED] Could not set zero - place ball at robot center (0,0) first.")

    except KeyboardInterrupt:
        print("[Main] Terminated by user.")
    finally:
        print("[Main] Stopping communication and releasing resources...")
        logger.stop()
        comms.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
