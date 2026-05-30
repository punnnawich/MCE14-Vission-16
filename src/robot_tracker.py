import cv2
import numpy as np

class RobotTracker:
    def __init__(self, config):
        """
        Initialize RobotTracker with configuration.
        """
        tracker_cfg = config.get("robot_tracker", {})
        dict_name = tracker_cfg.get("aruco_dict", "DICT_4X4_50")
        self.marker_id = tracker_cfg.get("marker_id", 0)
        self.marker_size = tracker_cfg.get("marker_size_m", 0.1)  # meters

        # Get the ArUco dictionary ID from cv2.aruco
        try:
            dict_id = getattr(cv2.aruco, dict_name)
        except AttributeError:
            dict_id = cv2.aruco.DICT_4X4_50

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.params = cv2.aruco.DetectorParameters()

        # Try to use ArucoDetector class if available (OpenCV 4.7+)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        else:
            self.detector = None

        # 3D corners of the ArUco marker in its own coordinate frame (centered at zero)
        half_size = self.marker_size / 2.0
        self.obj_points = np.array([
            [-half_size,  half_size, 0.0],
            [ half_size,  half_size, 0.0],
            [ half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0]
        ], dtype=np.float32)

    def track(self, frame_bgr, depth_frame=None, camera_matrix=None, dist_coeffs=None):
        """
        Detects the ArUco marker and estimates its 3D position in the camera frame.
        Returns:
            np.array([x, y, z]) in meters, and corner points in pixels, or (None, None).
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB if len(frame_bgr.shape) == 2 else cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame_gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame_gray, self.aruco_dict, parameters=self.params)

        if ids is None:
            return None, None

        # Search for our specific marker ID
        target_idx = -1
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id == self.marker_id:
                target_idx = idx
                break

        if target_idx == -1:
            return None, None

        marker_corners = corners[target_idx][0]  # Array of 4 corners (x, y)

        # Method 1: SolvePnP if camera parameters are provided
        if camera_matrix is not None and dist_coeffs is not None:
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                marker_corners.astype(np.float32),
                camera_matrix,
                dist_coeffs
            )
            if success:
                # tvec contains [x, y, z] in camera coordinate frame (in meters)
                return tvec.flatten(), marker_corners

        # Method 2: Fallback to depth lookup at center of marker
        if depth_frame is not None and camera_matrix is not None:
            cx = int(np.mean(marker_corners[:, 0]))
            cy = int(np.mean(marker_corners[:, 1]))
            
            # Ensure indices are within frame boundaries
            h, w = depth_frame.shape
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            z_mm = depth_frame[cy, cx]
            if z_mm > 0:
                z = z_mm / 1000.0  # mm to meters
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                cx0 = camera_matrix[0, 2]
                cy0 = camera_matrix[1, 2]
                
                x = (cx - cx0) * z / fx
                y = (cy - cy0) * z / fy
                return np.array([x, y, z]), marker_corners

        # Fallback to pixel centroid only if depth is not available
        cx = float(np.mean(marker_corners[:, 0]))
        cy = float(np.mean(marker_corners[:, 1]))
        return np.array([cx, cy, 0.0]), marker_corners
