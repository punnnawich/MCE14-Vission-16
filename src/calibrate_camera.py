"""
MCE14 Vission 16 — Camera Calibration Tool
============================================
เครื่องมือ calibrate กล้อง OAK-D Lite

ฟีเจอร์:
  1. แสดง Factory Calibration จากกล้อง (intrinsics)
  2. Calibrate ด้วย Checkerboard pattern (ถ้าต้องการ)
  3. ทดสอบ undistortion แบบ live
  4. ทดสอบ 3D projection + coordinate transform

วิธีใช้:
  python calibrate_camera.py

คีย์ลัด:
  c = Capture checkerboard frame (ถ่ายภาพ chessboard)
  u = Toggle undistortion view
  d = แสดง depth + 3D coordinates ที่เมาส์ชี้
  s = Save calibration & exit
  q = Quit
"""

import cv2
import numpy as np
import yaml
import os
import json
import depthai as dai

# ============================
# Configuration
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
CALIB_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "calibration_data.json")

# Checkerboard settings (inner corners)
CHECKERBOARD = (9, 6)   # จำนวนมุมด้านใน (columns, rows)
SQUARE_SIZE_MM = 25.0    # ขนาดช่องตาราง (mm)

# ============================
# Load config
# ============================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

camera_cfg = config.get("camera", {})
WIDTH = camera_cfg.get("resolution_w", 640)
HEIGHT = camera_cfg.get("resolution_h", 360)
FPS = camera_cfg.get("fps", 30)

ext_cfg = config.get("extrinsics", {})
R_ext = np.array(ext_cfg.get("R", [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float64)
T_ext = np.array(ext_cfg.get("T", [0,0,0]), dtype=np.float64).reshape(3, 1)

# ============================
# Create Pipeline
# ============================
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.Camera)
cam_rgb.build(dai.CameraBoardSocket.CAM_A)
rgb_out = cam_rgb.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.BGR888i, fps=FPS)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.build(autoCreateCameras=True, presetMode=dai.node.StereoDepth.PresetMode.ACCURACY)
stereo.setExtendedDisparity(False)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(WIDTH - (WIDTH % 16), HEIGHT)

q_rgb = rgb_out.createOutputQueue()
q_depth = stereo.depth.createOutputQueue()

# Prepare checkerboard 3D object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# Storage for calibration frames
obj_points = []  # 3D points in world
img_points = []  # 2D points in image
captured_count = 0

# State
show_undistort = False
show_depth_info = True
mouse_x, mouse_y = WIDTH // 2, HEIGHT // 2

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# ============================
# Main
# ============================
print("=" * 60)
print("  MCE14 Vission 16 — Camera Calibration Tool")
print("=" * 60)

with pipeline:
    device = pipeline.getDefaultDevice()
    
    # Read factory calibration
    calib = device.readCalibration()
    factory_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT))
    factory_dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
    
    fx, fy = factory_matrix[0, 0], factory_matrix[1, 1]
    cx, cy = factory_matrix[0, 2], factory_matrix[1, 2]
    
    print(f"\n  [Factory Calibration]")
    print(f"  Resolution: {WIDTH} x {HEIGHT}")
    print(f"  Focal Length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal Point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Distortion: {factory_dist[:5]}")
    print(f"\n  [Extrinsics from config.yaml]")
    print(f"  R = {R_ext.tolist()}")
    print(f"  T = {T_ext.flatten().tolist()}")
    print(f"\n  [คีย์ลัด]")
    print(f"  c = Capture checkerboard frame")
    print(f"  u = Toggle undistortion")
    print(f"  d = Toggle depth info at cursor")
    print(f"  s = Save calibration & exit")
    print(f"  q = Quit")
    print("=" * 60)
    
    # Use factory calibration as current
    camera_matrix = factory_matrix.copy()
    dist_coeffs = factory_dist.copy()
    
    # Undistortion maps
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (WIDTH, HEIGHT), 1, (WIDTH, HEIGHT)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (WIDTH, HEIGHT), cv2.CV_32FC1
    )
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    pipeline.start()
    
    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()
        
        if in_rgb is None:
            continue
            
        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame() if in_depth is not None else None
        
        # Apply undistortion if toggled
        if show_undistort:
            display = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            cv2.putText(display, "UNDISTORTED", (WIDTH - 160, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            display = frame.copy()
        
        # Detect checkerboard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        
        if found:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, found)
            cv2.putText(display, f"CHECKERBOARD DETECTED - Press 'c' to capture",
                        (10, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show depth & 3D info at cursor position
        if show_depth_info and depth_frame is not None:
            # Clamp mouse to valid range
            mx = max(0, min(mouse_x, depth_frame.shape[1] - 1))
            my = max(0, min(mouse_y, depth_frame.shape[0] - 1))
            
            z_mm = float(depth_frame[my, mx])
            
            if z_mm > 0:
                z_m = z_mm / 1000.0
                # Camera frame 3D
                x_cam = (mx - cx) * z_m / fx
                y_cam = (my - cy) * z_m / fy
                pos_cam = np.array([x_cam, y_cam, z_m])
                
                # World frame 3D
                pos_world = (R_ext @ pos_cam.reshape(3, 1) + T_ext).flatten()
                
                # Draw crosshair
                cv2.drawMarker(display, (mx, my), (0, 255, 255), cv2.MARKER_CROSSHAIR, 20, 1)
                
                # Info box
                info_y = 30
                cv2.putText(display, f"Pixel: ({mx}, {my})", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                info_y += 18
                cv2.putText(display, f"Depth: {z_m:.3f} m", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                info_y += 18
                cv2.putText(display, f"Camera: [{x_cam:.3f}, {y_cam:.3f}, {z_m:.3f}] m", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                info_y += 18
                cv2.putText(display, f"World:  X={pos_world[0]*100:.1f}cm Y={pos_world[1]*100:.1f}cm Z={pos_world[2]*100:.1f}cm", 
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                info_y += 18
                cv2.putText(display, f"  (X=left/right, Y=forward, Z=height)", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            else:
                cv2.drawMarker(display, (mx, my), (0, 0, 255), cv2.MARKER_CROSSHAIR, 20, 1)
                cv2.putText(display, "No depth", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        
        # Status bar
        status = f"Captured: {captured_count} | Undistort: {'ON' if show_undistort else 'OFF'} | Depth: {'ON' if show_depth_info else 'OFF'}"
        cv2.putText(display, status, (10, HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Calibration", display)
        
        # Depth heatmap
        if depth_frame is not None:
            depth_vis = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c') and found:
            obj_points.append(objp.copy())
            img_points.append(corners_refined)
            captured_count += 1
            print(f"  [Capture {captured_count}] Checkerboard frame captured!")
            if captured_count >= 10:
                print(f"  [INFO] คุณมี {captured_count} ภาพแล้ว — กด 's' เพื่อ calibrate & save")
        
        elif key == ord('u'):
            show_undistort = not show_undistort
            print(f"  [Undistort] {'ON' if show_undistort else 'OFF'}")
        
        elif key == ord('d'):
            show_depth_info = not show_depth_info
            print(f"  [Depth Info] {'ON' if show_depth_info else 'OFF'}")
        
        elif key == ord('s'):
            if captured_count >= 3:
                print(f"\n  [Calibrating] กำลัง calibrate จาก {captured_count} ภาพ...")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points, img_points, gray.shape[::-1], None, None
                )
                
                # Calculate reprojection error
                total_error = 0
                for i in range(len(obj_points)):
                    projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
                    total_error += error
                mean_error = total_error / len(obj_points)
                
                print(f"\n  [Results]")
                print(f"  Reprojection Error: {mean_error:.4f} px")
                print(f"  New fx={mtx[0,0]:.2f}, fy={mtx[1,1]:.2f}")
                print(f"  New cx={mtx[0,2]:.2f}, cy={mtx[1,2]:.2f}")
                print(f"  Distortion: {dist[0][:5]}")
                
                # Compare with factory
                print(f"\n  [Comparison with Factory]")
                print(f"  Δfx={mtx[0,0]-fx:.2f}, Δfy={mtx[1,1]-fy:.2f}")
                print(f"  Δcx={mtx[0,2]-cx:.2f}, Δcy={mtx[1,2]-cy:.2f}")
                
                # Save
                calib_data = {
                    "factory": {
                        "camera_matrix": factory_matrix.tolist(),
                        "dist_coeffs": factory_dist.tolist()
                    },
                    "custom": {
                        "camera_matrix": mtx.tolist(),
                        "dist_coeffs": dist.tolist(),
                        "reprojection_error": mean_error,
                        "num_frames": captured_count
                    },
                    "checkerboard": {
                        "pattern": list(CHECKERBOARD),
                        "square_size_mm": SQUARE_SIZE_MM
                    }
                }
                
                with open(CALIB_OUTPUT_PATH, "w") as f:
                    json.dump(calib_data, f, indent=2)
                print(f"\n  [Saved] {CALIB_OUTPUT_PATH}")
                
            else:
                # Save factory calibration only
                calib_data = {
                    "factory": {
                        "camera_matrix": factory_matrix.tolist(),
                        "dist_coeffs": factory_dist.tolist()
                    }
                }
                with open(CALIB_OUTPUT_PATH, "w") as f:
                    json.dump(calib_data, f, indent=2)
                print(f"\n  [Saved] Factory calibration → {CALIB_OUTPUT_PATH}")
                print(f"  (ถ่าย checkerboard อย่างน้อย 3 ภาพเพื่อ custom calibration)")
            
            break
    
    pipeline.stop()

cv2.destroyAllWindows()
print("\n  Calibration tool closed.")
