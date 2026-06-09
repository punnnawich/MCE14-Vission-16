import cv2
import numpy as np
from performance import gpu_cvt_color, gpu_in_range, gpu_morphology, to_cpu

class RobotTracker:
    def __init__(self, config):
        """
        Initialize RobotTracker for green-colored marker tracking using HSV binary thresholding.
        Replaces the computationally heavy ArUco markers with highly optimized HSV color segmentation.
        """
        tracker_cfg = config.get("robot_tracker", {})
        self.lower_green = np.array(tracker_cfg.get("lower_green", [35, 60, 60]))
        self.upper_green = np.array(tracker_cfg.get("upper_green", [85, 255, 255]))
        self.min_area = tracker_cfg.get("min_area", 100)
        self.max_area = tracker_cfg.get("max_area", 50000)
        self.kernel = np.ones((5, 5), np.uint8)

    def track(self, frame_bgr, depth_frame=None, camera_matrix=None, dist_coeffs=None,
              debug_window=False):
        """
        Thresholds the frame for gold color, detects the marker contour,
        and extracts the 3D position in the camera frame.

        Returns:
            np.array([x, y, z]) in meters, and corner points in pixels, or (None, None).
        """
        # GPU-accelerated HSV conversion + thresholding + morphology
        frame_hsv = gpu_cvt_color(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = gpu_in_range(frame_hsv, self.lower_green, self.upper_green)
        mask = gpu_morphology(mask, self.kernel, erode_iter=1, dilate_iter=2)
        mask = to_cpu(mask)

        if debug_window:
            # 3-panel: original | green mask | overlay with contours
            h, w = frame_bgr.shape[:2]
            panel_mask = np.zeros((h, w, 3), dtype=np.uint8)
            panel_mask[mask > 0] = (0, 255, 80)
            overlay = frame_bgr.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 180, 0]) * 0.6).astype(np.uint8)
            cv2.putText(panel_mask, f"green px={cv2.countNonZero(mask)}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(panel_mask, f"H:{self.lower_green[0]}-{self.upper_green[0]}"
                        f" S:{self.lower_green[1]}+ V:{self.lower_green[2]}+", (6, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
            cv2.imshow("Robot Green Mask", np.hstack([frame_bgr, panel_mask, overlay]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                if area > best_area:
                    best_area = area
                    best_cnt = cnt
                    
        if best_cnt is None:
            return None, None
            
        # Calculate pixel centroid (cx, cy)
        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return None, None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate bounding box to simulate corner points for visualizer
        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(best_cnt)
        marker_corners = np.array([
            [x_bb, y_bb],
            [x_bb + w_bb, y_bb],
            [x_bb + w_bb, y_bb + h_bb],
            [x_bb, y_bb + h_bb]
        ], dtype=np.float32)
        
        # Default Z calculation: lookup depth map
        z_m = 0.0
        if depth_frame is not None:
            # Sample depth ROI around centroid to avoid noise holes
            h, w = depth_frame.shape
            half = 2
            y_start = max(0, cy - half)
            y_end = min(h, cy + half + 1)
            x_start = max(0, cx - half)
            x_end = min(w, cx + half + 1)
            
            depth_roi = depth_frame[y_start:y_end, x_start:x_end]
            valid = depth_roi[depth_roi > 0]
            
            if len(valid) > 0:
                z_mm = np.median(valid)
            else:
                z_mm = depth_frame[cy, cx]
                
            if z_mm > 0:
                z_m = z_mm / 1000.0
                
        # 3D projection
        if z_m > 0.0 and camera_matrix is not None:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx0 = camera_matrix[0, 2]
            cy0 = camera_matrix[1, 2]
            
            # Undistort centroid pixel if distortion coefficients are provided
            if dist_coeffs is not None:
                pts_px = np.array([[[cx, cy]]], dtype=np.float32)
                undistorted_pts = cv2.undistortPoints(pts_px, camera_matrix, dist_coeffs, P=camera_matrix)
                ucx = undistorted_pts[0][0][0]
                ucy = undistorted_pts[0][0][1]
            else:
                ucx, ucy = cx, cy
                
            x = (ucx - cx0) * z_m / fx
            y = (ucy - cy0) * z_m / fy
            return np.array([x, y, z_m]), marker_corners
            
        return None, None
