import cv2
import numpy as np

class DebugVisualizer:
    def __init__(self, config):
        """
        Initialize the debug visualizer with window dimensions and settings.
        """
        camera_cfg = config.get("camera", {})
        self.w = camera_cfg.get("resolution_w", 640)
        self.h = camera_cfg.get("resolution_h", 360)

    def draw_all(self, frame_bgr, ball_info, trajectory, prediction, robot_pos, robot_corners, fps, profiler_data, comms_error, release_detected):
        """
        Annotates the RGB frame with tracking, prediction, robot pose, and telemetry info.
        """
        annotated = frame_bgr.copy()

        # 1. Draw Robot Marker / Position
        if robot_corners is not None:
            pts = robot_corners.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
            for pt in pts:
                cv2.circle(annotated, tuple(pt), 4, (0, 0, 255), -1)
            
            if robot_pos is not None:
                rx, ry, rz = robot_pos
                cv2.putText(annotated, f"Robot: [{rx:.2f}, {ry:.2f}, {rz:.2f}]m", 
                            (int(pts[0][0]), int(pts[0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 2. Draw Red Ball Centroid & Contour
        if ball_info is not None:
            cx, cy = ball_info["cx"], ball_info["cy"]
            # Draw crosshair
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            
            # Bounding box
            if "bbox" in ball_info:
                xb, yb, wb, hb = ball_info["bbox"]
                cv2.rectangle(annotated, (xb, yb), (xb + wb, yb + hb), (0, 0, 255), 2)
            
            # Contour
            if "contour" in ball_info:
                cv2.drawContours(annotated, [ball_info["contour"]], -1, (0, 255, 0), 1)

        # 3. Draw Trajectory Points
        if len(trajectory) > 0:
            # We can project the 3D points back to the image plane or just draw them
            # Let's draw lines between consecutive ball locations in the pixel space
            # (In this simple version, we'll draw dots if we store their pixel centroids, 
            # but since trajectory stores (X, Y, Z, t) in meters, we can just overlay them if we project them back,
            # or we can collect pixel centroids on the fly and draw them).
            pass

        # 4. Draw Predicted Landing Location on Screen
        if prediction is not None:
            px, py, pz_floor = prediction["x"], prediction["y"], prediction["z"]
            t_impact = prediction["t_land_from_now"]
            
            # Text overlay for predicted coordinates
            cv2.putText(annotated, f"PREDICTED LANDING: X={px*100:.1f}cm, Y={py*100:.1f}cm", 
                        (20, self.h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated, f"Time to Impact: {t_impact:.3f}s", 
                        (20, self.h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 5. Draw HUD Status Info (FPS, Latency, Network, Release Status)
        hud_bg = np.zeros((130, 220, 3), dtype=np.uint8)
        # Background transparency overlay
        annotated[10:140, 10:230] = cv2.addWeighted(annotated[10:140, 10:230], 0.4, hud_bg, 0.6, 0.0)

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Latencies
        pipeline_lat = profiler_data.get("Total Pipeline", 0.0)
        cv2.putText(annotated, f"Latency: {pipeline_lat:.1f} ms", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Release state
        release_text = "RELEASED" if release_detected else "WAITING"
        release_color = (0, 255, 0) if release_detected else (0, 165, 255)
        cv2.putText(annotated, f"Status: {release_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, release_color, 1)

        # Network error
        net_text = "Connected" if comms_error is None else f"Error: {comms_error}"
        net_color = (0, 255, 0) if comms_error is None else (0, 0, 255)
        cv2.putText(annotated, f"Network: {net_text[:18]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, net_color, 1)

        # Extracted metrics if available
        if ball_info is not None:
            cv2.putText(annotated, f"Ball Centroid: ({cx}, {cy})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if "depth_m" in ball_info:
                cv2.putText(annotated, f"Depth: {ball_info['depth_m']:.2f} m", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return annotated

    def colorize_depth(self, depth_frame):
        """
        Normalizes and applies a color map to the raw 16-bit depth frame (mm) for visualization.
        """
        if depth_frame is None:
            return None
        
        # Raw depth is mm (uint16). Convert to 0-255 scale for visualization
        # Standard range is roughly 400mm (0.4m) to 4000mm (4.0m)
        depth_vis = np.clip(depth_frame, 400, 4000)
        depth_vis = ((depth_vis - 400) / (4000 - 400) * 255).astype(np.uint8)
        
        # Invert colors so closer objects are brighter/different
        depth_vis = 255 - depth_vis
        
        # Apply colormap
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return depth_color

    def show_frames(self, rgb_annotated, depth_colorized):
        """
        Draws windows for BGR and Depth outputs.
        """
        if rgb_annotated is not None:
            cv2.imshow("MCE14-Vission-16 (RGB Feed)", rgb_annotated)
        if depth_colorized is not None:
            cv2.imshow("MCE14-Vission-16 (Depth Map)", depth_colorized)
