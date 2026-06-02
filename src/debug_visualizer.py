import cv2
import numpy as np
from collections import deque

class DebugVisualizer:
    def __init__(self, config):
        """
        Initialize the debug visualizer with window dimensions and settings.
        """
        camera_cfg = config.get("camera", {})
        self.w = camera_cfg.get("resolution_w", 640)
        self.h = camera_cfg.get("resolution_h", 360)

        # Pixel trail for trajectory overlay on camera feed
        self.pixel_trail = deque(maxlen=50)

        # Trajectory plot canvas dimensions
        self.plot_w = 700
        self.plot_h = 400

    def draw_all(self, frame_bgr, ball_info, trajectory, prediction, robot_pos, robot_corners, fps, profiler_data, comms_error, release_detected, projected_curve=None, projected_workspace=None):
        """
        Annotates the RGB frame with tracking, prediction, robot pose, telemetry info, projected 3D curve, and safety workspace circle.
        """
        # C-07: Annotate directly on the frame (no copy needed — frame is
        # overwritten by the next camera frame in the next loop iteration)
        annotated = frame_bgr

        # 1. Draw Workspace Boundary Cylinder (Base Z=0, Catch Z=z_catch)
        if projected_workspace is not None:
            # Normal: Cyan/Green, Warning (clamped): red
            ws_color = (0, 255, 0)
            if prediction is not None and prediction.get("is_clamped", False):
                ws_color = (0, 0, 255)
                
            base_pts = projected_workspace.get("base", [])
            catch_pts = projected_workspace.get("catch", [])
            pillars = projected_workspace.get("pillars", [])
            
            # Draw Base Circle (Z=0, physically on the table/floor)
            if len(base_pts) > 1:
                pts_base = np.array(base_pts, dtype=np.int32)
                cv2.polylines(annotated, [pts_base], True, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(annotated, "ROBOT BASE (0,0)", (pts_base[0][0], pts_base[0][1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

            # Draw Catching Circle (Z=z_catch, floating in space)
            if len(catch_pts) > 1:
                pts_catch = np.array(catch_pts, dtype=np.int32)
                cv2.polylines(annotated, [pts_catch], True, ws_color, 2, cv2.LINE_AA)
                z_height_cm = int(prediction.get("z", 0.25) * 100) if prediction is not None else 25
                cv2.putText(annotated, f"CATCH PLANE ({z_height_cm}cm)", (pts_catch[0][0], pts_catch[0][1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ws_color, 1, cv2.LINE_AA)
                
            # Draw Vertical Pillars connecting base to catch plane
            for pt_b, pt_c in pillars:
                cv2.line(annotated, pt_b, pt_c, (120, 120, 120), 1, cv2.LINE_AA)

        # 2. Draw Robot Marker / Position
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

        # 3. Draw Red Ball Centroid & Contour
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

            # Store pixel centroid for trajectory trail
            self.pixel_trail.append((cx, cy))

        # 4. Draw Trajectory Trail (pixel-space)
        if len(self.pixel_trail) > 1:
            trail_pts = list(self.pixel_trail)
            n = len(trail_pts)
            for i in range(1, n):
                # Fade older points: newer = brighter, older = dimmer
                alpha = int(255 * (i / n))
                color = (alpha, 255 - alpha, 0)  # Green → Cyan gradient
                thickness = max(1, int(3 * (i / n)))
                cv2.line(annotated, trail_pts[i - 1], trail_pts[i], color, thickness)
            # Draw dot at current position
            cv2.circle(annotated, trail_pts[-1], 5, (0, 255, 255), -1)

        # 5. Draw Predicted Landing Location on Screen
        if prediction is not None:
            px, py, pz_catch = prediction["x"], prediction["y"], prediction["z"]
            t_impact = prediction["t_land_from_now"]
            is_clamped = prediction.get("is_clamped", False)
            
            # Text overlay for predicted coordinates
            color = (0, 0, 255) if is_clamped else (255, 255, 0)
            cv2.putText(annotated, f"PREDICTED LANDING: X={px*100:.1f}cm, Y={py*100:.1f}cm", 
                        (20, self.h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, f"Time to Impact: {t_impact:.3f}s", 
                        (20, self.h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display transmission status
            elapsed = prediction.get("elapsed_since_release", 0.0)
            if elapsed <= 0.3:
                tx_status = f"TX: ACTIVE ({elapsed:.2f}s)"
                tx_color = (0, 255, 0)  # Green
            else:
                tx_status = f"TX: LOCKED (LATE - {elapsed:.2f}s)"
                tx_color = (0, 165, 255)  # Orange
            cv2.putText(annotated, tx_status, (20, self.h - 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tx_color, 2)
            
            if is_clamped:
                cv2.putText(annotated, "WARNING: OUT OF WORKSPACE (CLAMPED TO 50CM)", 
                            (20, self.h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw the 3D projected parabolic curve on the RGB feed
            if projected_curve is not None and len(projected_curve) > 1:
                pts = np.array(projected_curve, dtype=np.int32)
                # Draw the smooth curve in Cyan/Yellow with anti-aliasing
                cv2.polylines(annotated, [pts], False, (255, 255, 0), 2, cv2.LINE_AA)
                # Draw a prominent crosshair at the projected impact target position (the last point)
                cv2.drawMarker(annotated, tuple(pts[-1]), color, cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.circle(annotated, tuple(pts[-1]), 8, color, 2)

        # 6. Draw HUD Status Info (FPS, Latency, Network, Release Status)
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

    def colorize_depth(self, depth_frame, motion_mask=None):
        """
        Normalizes and applies a color map to the raw 16-bit depth frame (mm) for visualization.
        Only shows depth for pixels with active motion, rendering static areas black.
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
        
        # Mask out pixels without motion if a motion mask is provided
        if motion_mask is not None:
            # Resize motion mask to match depth image dimensions if they differ
            if motion_mask.shape[:2] != depth_color.shape[:2]:
                motion_mask_resized = cv2.resize(motion_mask, (depth_color.shape[1], depth_color.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                motion_mask_resized = motion_mask
            
            # Static pixels are 0 in the motion mask. Set them to black [0, 0, 0] in depth_color.
            depth_color[motion_mask_resized == 0] = 0
            
        return depth_color

    def show_frames(self, rgb_annotated, depth_colorized):
        """
        Draws windows for BGR and Depth outputs.
        """
        if rgb_annotated is not None:
            cv2.imshow("MCE14-Vission-16 (RGB Feed)", rgb_annotated)
        if depth_colorized is not None:
            cv2.imshow("MCE14-Vission-16 (Depth Map)", depth_colorized)

    def reset_trail(self):
        """Clear pixel trail when tracking is lost."""
        self.pixel_trail.clear()

