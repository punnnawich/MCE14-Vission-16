import cv2
import numpy as np
import yaml
import os
import depthai as dai

def load_config(config_path="../src/config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "src", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), config_path

def save_config(config, config_path):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def main():
    print("=== MCE14 Vission 16 Extrinsic Calibration ===")
    print("This tool will compute the extrinsic camera transformation matrix (R, T)")
    print("using a calibration checkerboard placed flat on the floor (Z = 0).\n")

    # Load configuration
    try:
        config, config_path = load_config()
    except Exception as e:
        print(f"[Error] Could not load config: {e}")
        return

    # Configuration for checkerboard
    rows = int(input("Enter checkerboard internal corner rows (default 6): ") or "6")
    cols = int(input("Enter checkerboard internal corner columns (default 9): ") or "9")
    square_size = float(input("Enter checkerboard square size in meters (default 0.025): ") or "0.025")

    # Define 3D corner coordinates on the floor plane (Z = 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    # Create DepthAI Pipeline
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    w = config.get("camera", {}).get("resolution_w", 640)
    h = config.get("camera", {}).get("resolution_h", 360)
    cam_rgb.setPreviewSize(w, h)
    cam_rgb.setInterleaved(False)
    
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam_rgb.preview.link(xout.input)

    print("[Calib] Connecting to OAK-D Lite...")
    try:
        with dai.Device(pipeline) as device:
            print("[Calib] Camera connected. Fetching intrinsics...")
            calib = device.readCalibration()
            camera_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h))
            dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB))

            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            print("\nInstructions:")
            print("1. Place the checkerboard flat on the floor in the camera view.")
            print("2. The board center will represent the origin of the X, Y coordinate axes on the floor.")
            print("3. Press 'c' to capture calibration when checkerboard corners are highlighted.")
            print("4. Press 'q' to quit without saving.\n")

            while True:
                in_rgb = q_rgb.get()
                if in_rgb is None:
                    continue

                frame = in_rgb.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

                display_frame = frame.copy()
                if ret:
                    # Draw corners
                    cv2.drawChessboardCorners(display_frame, (cols, rows), corners, ret)
                    cv2.putText(display_frame, "Checkerboard Found! Press 'c' to Calibrate.", 
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Searching for Checkerboard...", 
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Calibration Stream", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[Calib] Calibration cancelled.")
                    break
                elif key == ord('c') and ret:
                    print("[Calib] Performing SolvePnP Extrinsic Calibration...")
                    
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # Solve Pose Estimation
                    # Find transformation from object points (world) to image points (camera)
                    success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

                    if success:
                        # Convert rotation vector to 3x3 matrix
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # Invert transformation to get Camera relative to World (floor) coordinates
                        # pos_cam = R @ pos_world + T -> pos_world = R_inv @ (pos_cam - T)
                        # Let's save R and T directly. In vision_pipeline: pos_world = R_ext @ pos_cam + T_ext.
                        # Wait!
                        # solvePnP returns rotation and translation that takes World coordinate points to Camera coordinate points:
                        # point_camera = R_val @ point_world + T_val.
                        # But we want to convert Camera points to World points:
                        # point_world = R_val^T @ (point_camera - T_val) = R_val^T @ point_camera - R_val^T @ T_val
                        # So:
                        # R_ext = R_val^T
                        # T_ext = -R_val^T @ T_val
                        
                        R_ext = R.T
                        T_ext = -R.T @ tvec
                        
                        # The height of the camera above the floor is the Z coordinate of the camera center in world coordinates,
                        # which is T_ext[2].
                        z_floor_computed = -T_ext[2][0]  # Or height in camera frame

                        print("\n=== Extrinsic Calibration Results ===")
                        print(f"Computed Camera Height: {z_floor_computed:.3f} meters")
                        print("Rotation Matrix (R_ext):")
                        print(R_ext)
                        print("Translation Vector (T_ext):")
                        print(T_ext.flatten())

                        # Save back to config
                        config["extrinsics"] = {
                            "R": R_ext.tolist(),
                            "T": T_ext.flatten().tolist()
                        }
                        
                        # Set Z_floor of the floor in camera frame.
                        # Wait, what coordinate system does predictor use?
                        # The predictor fits trajectory coordinates *in the world frame*, or camera frame?
                        # In vision_pipeline, we converted pos_camera to pos_world:
                        # pos_world = R_ext @ pos_camera + T_ext.
                        # In world coordinates, Z is the vertical height above the floor plane, because we defined checkerboard object points with Z=0 on the floor!
                        # So in the world frame, the floor is at Z = 0.0 meters.
                        # Therefore, we can set predictor's z_floor to 0.0 directly, and the curve fitting will solve for when Z_world(t) = 0.0!
                        # This is extremely clean and mathematically elegant!
                        config["predictor"]["z_floor"] = 0.0

                        save_config(config, config_path)
                        print(f"\n[Success] Calibration saved to config file: {config_path}")
                        
                        # Show confirmation image
                        cv2.imshow("Calibration Stream", display_frame)
                        cv2.waitKey(2000)
                        break
                    else:
                        print("[Error] SolvePnP failed to converge.")

    except Exception as e:
        print(f"\n[FATAL] Calibration failed: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
