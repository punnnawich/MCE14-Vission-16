import cv2
import numpy as np
import yaml
import time
import os
import depthai as dai

from ball_detector import BallDetector
from median_filter import MedianFilter3D
from release_detector import ReleaseDetector
from projectile_predictor import ProjectilePredictor
from robot_tracker import RobotTracker
from robot_comms import RobotComms
from debug_visualizer import DebugVisualizer
from latency_profiler import LatencyProfiler

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
    Creates and configures the DepthAI camera pipeline.
    """
    pipeline = dai.Pipeline()
    
    camera_cfg = config.get("camera", {})
    width = camera_cfg.get("resolution_w", 640)
    height = camera_cfg.get("resolution_h", 360)
    fps = camera_cfg.get("fps", 30)

    # 1. BGR Color Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(fps)
    
    # Crop or scale color preview
    cam_rgb.setPreviewSize(width, height)

    # 2. Mono Cameras (Left & Right) for Depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    
    # 3. Stereo Depth Node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setExtendedDisparity(camera_cfg.get("extended_disparity", True))
    stereo.setLeftRightCheck(camera_cfg.get("left_right_check", True))
    
    # Enable spatial and temporal filtering to clean up raw depth
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    
    # Align depth map to RGB camera perspective
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Link mono cameras to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # 4. XLink Outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline

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
    release_detector = ReleaseDetector(vel_threshold=config.get("release", {}).get("vel_threshold", 0.5))
    predictor = ProjectilePredictor(config)
    robot_tracker = RobotTracker(config)
    
    comms = RobotComms(config)
    comms.start()  # Starts UDP transmitter and heartbeat background thread
    
    profiler = LatencyProfiler()
    visualizer = DebugVisualizer(config)

    # Coordinate transformation extrinsic matrices from config
    ext_cfg = config.get("extrinsics", {})
    R_ext = np.array(ext_cfg.get("R", [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float32)
    T_ext = np.array(ext_cfg.get("T", [0,0,0]), dtype=np.float32).reshape(3, 1)

    # Pipeline Setup
    pipeline = create_camera_pipeline(config)

    print("[Main] Connecting to OAK-D Lite camera...")
    try:
        device = dai.Device(pipeline)
    except Exception as e:
        print(f"\n[FATAL] Could not connect to DepthAI OAK-D camera: {e}")
        print("Please ensure your OAK-D Lite is plugged in via a USB3 port and try again.")
        comms.stop()
        return

    print("[Main] Camera connected successfully. Setting up calibration details...")
    
    # Retrieve Intrinsic Parameters dynamically from device
    calib = device.readCalibration()
    w = config.get("camera", {}).get("resolution_w", 640)
    h = config.get("camera", {}).get("resolution_h", 360)
    camera_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h))
    dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB))

    # Input Queues
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    fps_last_time = time.perf_counter()
    fps_counter = 0
    fps = 0.0

    missing_frames = 0
    max_missing_frames = 15
    
    print("[Main] System initialized. Entering real-time vision loop. Press 'q' to quit.")

    try:
        while True:
            profiler.start_frame()
            profiler.start_stage("Frame Capture")

            # Blocking read for RGB and non-blocking for depth
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            if in_rgb is None or in_depth is None:
                continue

            frame_rgb = in_rgb.getCvFrame()
            frame_depth = in_depth.getCvFrame()
            
            current_time = time.perf_counter()
            profiler.end_stage("Frame Capture")

            # --- MODULE B & C: HSV Segmentation & Centroid Detection ---
            profiler.start_stage("HSV Segmentation & Blob")
            mask = ball_detector.detect_red_ball(frame_rgb)
            ball_info = ball_detector.find_ball_centroid(mask)
            profiler.end_stage("HSV Segmentation & Blob")

            prediction = None
            pos_world = None

            if ball_info is not None:
                missing_frames = 0
                cx, cy = ball_info["cx"], ball_info["cy"]

                # --- MODULE D: Depth Lookup & 3D Projection ---
                profiler.start_stage("Depth Lookup & 3D Proj")
                # Sample ROI to handle depth noise holes around the center pixel
                roi_size = 5
                y_start = max(0, cy - roi_size // 2)
                y_end = min(h - 1, cy + roi_size // 2 + 1)
                x_start = max(0, cx - roi_size // 2)
                x_end = min(w - 1, cx + roi_size // 2 + 1)
                depth_roi = frame_depth[y_start:y_end, x_start:x_end]
                valid_depths = depth_roi[depth_roi > 0]

                if len(valid_depths) > 0:
                    z_mm = np.median(valid_depths)
                else:
                    z_mm = float(frame_depth[cy, cx])

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

                    # --- MODULE E: Median Filter ---
                    profiler.start_stage("Median Filtering")
                    pos_camera_filtered = median_filter.update(pos_camera)
                    profiler.end_stage("Median Filtering")

                    # Apply Extrinsics mapping: Camera -> World/Robot system
                    pos_world = (R_ext @ pos_camera_filtered.reshape(3, 1) + T_ext).flatten()

                    # --- MODULE F: Release Detection ---
                    profiler.start_stage("Release Detection")
                    released = release_detector.update(pos_world, current_time)
                    profiler.end_stage("Release Detection")

                    # --- MODULE G: Projectile Predictor ---
                    if released:
                        profiler.start_stage("Curve Fitting & Pred")
                        predictor.add_point(pos_world, current_time)
                        prediction = predictor.predict_landing()
                        profiler.end_stage("Curve Fitting & Pred")

                        # Send target coordinates to ESP32 (convert meters to centimeters)
                        if prediction is not None:
                            profiler.start_stage("Transmission")
                            px_cm = prediction["x"] * 100.0
                            py_cm = prediction["y"] * 100.0
                            comms.send_target(px_cm, py_cm)
                            profiler.end_stage("Transmission")
                else:
                    profiler.end_stage("Depth Lookup & 3D Proj")
            else:
                missing_frames += 1
                # If ball is missing for a while, reset tracking states
                if missing_frames > max_missing_frames:
                    release_detector.reset()
                    predictor.reset()
                    median_filter.reset()

            # --- MODULE H: Robot Position Tracking ---
            profiler.start_stage("Robot Tracking")
            robot_pos, robot_corners = robot_tracker.track(
                frame_rgb,
                frame_depth,
                camera_matrix,
                dist_coeffs
            )
            
            # Map robot position to world coordinates if detected
            if robot_pos is not None and robot_pos[2] > 0.0:  # Valid depth
                robot_pos = (R_ext @ robot_pos.reshape(3, 1) + T_ext).flatten()
            profiler.end_stage("Robot Tracking")

            # FPS calculation
            fps_counter += 1
            if current_time - fps_last_time >= 1.0:
                fps = fps_counter / (current_time - fps_last_time)
                fps_counter = 0
                fps_last_time = current_time

            profiler.end_frame()

            # --- MODULE J: Visualizer Overlays ---
            profiler.start_stage("GUI Visualizer")
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
                release_detector.released
            )
            color_depth = visualizer.colorize_depth(frame_depth)
            visualizer.show_frames(annotated_rgb, color_depth)
            profiler.end_stage("GUI Visualizer")

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[Main] Terminated by user.")
    finally:
        print("[Main] Stopping communication and releasing resources...")
        comms.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
