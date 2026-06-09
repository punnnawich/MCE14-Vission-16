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

            # Draw projected trajectory curve + landing marker on RGB feed
            if projected_curve is not None and len(projected_curve) > 1:
                pts = np.array(projected_curve, dtype=np.int32)

                # Glow effect: thick dim line underneath + bright thin line on top
                cv2.polylines(annotated, [pts], False, (40, 120, 255), 5, cv2.LINE_AA)
                cv2.polylines(annotated, [pts], False, (0, 220, 255), 2, cv2.LINE_AA)

                # Landing point marker
                impact = tuple(pts[-1])
                lnd_col = (0, 60, 255) if is_clamped else (0, 255, 100)
                cv2.circle(annotated,    impact, 14, lnd_col, 2, cv2.LINE_AA)
                cv2.circle(annotated,    impact,  4, lnd_col, -1, cv2.LINE_AA)
                cv2.drawMarker(annotated, impact, (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)

                # Coord label next to landing point
                px_cm = prediction["x"] * 100
                py_cm = prediction["y"] * 100
                lbl = f"({px_cm:+.0f},{py_cm:+.0f})cm"
                lx = min(impact[0] + 10, self.w - 140)
                ly = max(impact[1] - 10, 14)
                cv2.putText(annotated, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, lnd_col, 1, cv2.LINE_AA)

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

        # 7. Real-time Calculation Panel (right side)
        if trajectory is not None and len(trajectory) > 0:
            self._draw_calc_panel(annotated, trajectory, prediction)

        return annotated

    def _draw_calc_panel(self, frame, trajectory, prediction):
        """
        Semi-transparent right-side panel แสดงค่าคำนวณแบบ real-time:
          • ตำแหน่ง 3D ปัจจุบันของลูก (cm, relative to pos_zero)
          • ความเร็ว Vx / Vy / Vz (m/s)
          • จำนวนจุดใน buffer
          • ผล prediction: t_land, X, Y, raw values
        """
        PW = 210   # panel width
        PAD = 8
        LINE_H = 17
        x0 = frame.shape[1] - PW - 4
        y0 = 10

        # ── gather latest ball pos from buffer ──────────────────────────────
        pts = list(trajectory)
        last = pts[-1]
        bx_m, by_m, bz_m = last[0], last[1], last[2]

        # ── count lines needed ──────────────────────────────────────────────
        n_lines = 14 if prediction is not None else 7
        ph = PAD * 2 + n_lines * LINE_H + 6
        y1 = y0 + ph

        # clip if panel goes off-screen
        y1 = min(y1, frame.shape[0] - 4)

        # ── dark semi-transparent background ────────────────────────────────
        roi = frame[y0:y1, x0:x0 + PW]
        bg  = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.35, bg, 0.65, 0, roi)
        frame[y0:y1, x0:x0 + PW] = roi

        # ── helper: draw one text line ───────────────────────────────────────
        tx = x0 + PAD
        ty = [y0 + PAD + LINE_H]

        def line(text, color=(210, 210, 210), bold=False):
            cv2.putText(frame, text, (tx, ty[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 2 if bold else 1, cv2.LINE_AA)
            ty[0] += LINE_H

        def divider(label=""):
            color = (120, 200, 255)
            cv2.putText(frame, f"-- {label} --" if label else "─" * 22,
                        (tx, ty[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, color, 1, cv2.LINE_AA)
            ty[0] += LINE_H

        # ── BALL STATE ───────────────────────────────────────────────────────
        n_buf = len(pts)
        divider("BALL STATE")
        line(f"Buf: {n_buf} pts", (180, 180, 180))
        line(f"X: {bx_m*100:+.1f} cm", (100, 220, 100))
        line(f"Y: {by_m*100:+.1f} cm", (100, 220, 100))
        line(f"Z: {bz_m*100:.1f} cm",  (100, 220, 100))

        # ── VELOCITY (from prediction coeffs if available) ───────────────────
        if prediction is not None:
            vx = prediction.get("vx", 0.0)
            vy = prediction.get("vy", 0.0)
            vz = prediction.get("vz", 0.0)
            divider("VELOCITY")
            line(f"Vx: {vx:+.2f} m/s", (100, 180, 255))
            line(f"Vy: {vy:+.2f} m/s", (100, 180, 255))
            line(f"Vz: {vz:+.2f} m/s", (100, 180, 255))

            # ── PREDICTION ───────────────────────────────────────────────────
            px_cm  = prediction["x"]  * 100
            py_cm  = prediction["y"]  * 100
            rx_cm  = prediction.get("raw_x", prediction["x"]) * 100
            ry_cm  = prediction.get("raw_y", prediction["y"]) * 100
            t_left = prediction.get("t_land_from_now", 0.0)
            n_pred = prediction.get("n_pts", n_buf)
            clamped = prediction.get("is_clamped", False)

            pred_col = (0, 80, 255) if clamped else (80, 255, 80)
            divider("PREDICTION")
            line(f"n: {n_pred} pts", (180, 180, 180))
            line(f"t_land: {t_left:.3f}s", (220, 220, 80))
            line(f"→ X: {px_cm:+.1f} cm", pred_col, bold=True)
            line(f"→ Y: {py_cm:+.1f} cm", pred_col, bold=True)
            if clamped:
                line(f"raw {rx_cm:+.1f},{ry_cm:+.1f}", (80, 80, 255))
            if clamped:
                line("CLAMPED", (0, 60, 255), bold=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Fit Plot — X(t), Y(t), Z(t) with fitted curves, rendered via OpenCV
    # ─────────────────────────────────────────────────────────────────────────

    # Canvas size
    _FP_W = 660
    _FP_H = 420
    _FP_BG  = (18, 18, 32)       # dark background
    _FP_GRID = (40, 40, 55)

    def draw_fit_plot(self, trajectory, prediction) -> np.ndarray:
        """
        แสดง 3 subplot: X(t), Y(t), Z(t)
          • จุดสีเทา  = ค่าวัดจริงใน buffer
          • เส้นสี   = fitted curve  (X=quadratic cyan, Y=linear orange, Z=parabolic magenta)
          • ◆ เขียว  = ค่าที่ทำนายที่ t_land
          • เส้นขาว  = t_latest (เวลาปัจจุบัน)
        """
        W, H = self._FP_W, self._FP_H
        canvas = np.full((H, W, 3), self._FP_BG, dtype=np.uint8)

        if trajectory is None or len(trajectory) < 2:
            cv2.putText(canvas, "Waiting for throw ...",
                        (W // 2 - 100, H // 2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(canvas, "X / Y / Z fit will appear here during flight",
                        (W // 2 - 190, H // 2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 90), 1, cv2.LINE_AA)
            return canvas

        pts     = np.array(trajectory)          # (N, 4) — x,y,z,t
        xs_m    = pts[:, 0]
        ys_m    = pts[:, 1]
        zs_m    = pts[:, 2]
        ts_raw  = pts[:, 3]
        t0      = ts_raw[0]
        ts      = ts_raw - t0                   # normalised, starts at 0
        t_latest = ts[-1]

        coeff_x = coeff_y = coeff_z = None
        t_land = t_latest + 0.5                 # fallback
        if prediction is not None:
            t_land   = prediction.get("t_land",   t_latest + 0.5)
            cx = prediction.get("coeff_x")
            cy = prediction.get("coeff_y")
            cz = prediction.get("coeff_z")
            if cx: coeff_x = np.array(cx)
            if cy: coeff_y = np.array(cy)
            if cz: coeff_z = np.array(cz)

        # Layout: 3 rows
        MARGIN_L = 58
        MARGIN_R = 18
        MARGIN_T = 14
        ROW_H    = (H - MARGIN_T - 10) // 3   # height per subplot
        rows = [
            (xs_m * 100, coeff_x, (0, 220, 220),  "X (cm)",  "Linear"),
            (ys_m * 100, coeff_y, (80, 160, 255),  "Y (cm)",  "Theil-Sen"),
            (zs_m * 100, coeff_z, (200,  80, 230), "Z (cm)",  "Parabola"),
        ]

        t_dense = np.linspace(0, t_land + 0.05, 120)

        for row_i, (vals, coeff, col, ylabel, fit_name) in enumerate(rows):
            # ── subplot bounding box ─────────────────────────────────────────
            y_top = MARGIN_T + row_i * ROW_H
            y_bot = y_top + ROW_H - 6
            x_l   = MARGIN_L
            x_r   = W - MARGIN_R

            px_w = x_r - x_l
            px_h = y_bot - y_top

            # ── data range ───────────────────────────────────────────────────
            all_vals = list(vals)
            if coeff is not None:
                fit_vals = np.polyval(coeff, t_dense)
                all_vals += list(fit_vals * 100) if ylabel[0] != "Z" else list(fit_vals)
                # coeff for Z are in metres; vals already in cm
                fit_vals_cm = fit_vals * 100 if len(coeff) == 2 else fit_vals * 100
                # All coeffs operate in metres; multiply by 100 for cm display
                fit_vals_cm = np.polyval(coeff, t_dense) * 100

            v_min = min(all_vals)
            v_max = max(all_vals)
            v_pad = max((v_max - v_min) * 0.18, 2.0)
            v_min -= v_pad
            v_max += v_pad

            t_max_disp = max(t_land + 0.05, t_latest + 0.1)

            def to_px(t_val, v_val):
                px = x_l + int((t_val / max(t_max_disp, 1e-9)) * px_w)
                py = y_bot - int(((v_val - v_min) / max(v_max - v_min, 1e-9)) * px_h)
                return np.clip(px, x_l - 1, x_r + 1), np.clip(py, y_top - 1, y_bot + 1)

            # ── border + subtle grid ─────────────────────────────────────────
            cv2.rectangle(canvas, (x_l, y_top), (x_r, y_bot), self._FP_GRID, 1)
            for g in [0.25, 0.5, 0.75]:
                gx = x_l + int(g * px_w)
                cv2.line(canvas, (gx, y_top), (gx, y_bot), self._FP_GRID, 1)
                gy = y_top + int(g * px_h)
                cv2.line(canvas, (x_l, gy), (x_r, gy), self._FP_GRID, 1)

            # ── zero line (v=0) ───────────────────────────────────────────────
            if v_min < 0 < v_max:
                _, py0 = to_px(0, 0)
                cv2.line(canvas, (x_l, py0), (x_r, py0), (55, 55, 70), 1)

            # ── t_latest vertical line ────────────────────────────────────────
            px_tl, _ = to_px(t_latest, v_min)
            cv2.line(canvas, (px_tl, y_top), (px_tl, y_bot), (200, 200, 200), 1)

            # ── fitted curve ─────────────────────────────────────────────────
            if coeff is not None:
                curve_pts = []
                for t_v in t_dense:
                    v_fit = float(np.polyval(coeff, t_v)) * 100
                    px_, py_ = to_px(t_v, v_fit)
                    curve_pts.append((px_, py_))
                for k in range(1, len(curve_pts)):
                    cv2.line(canvas, curve_pts[k - 1], curve_pts[k],
                             col, 2, cv2.LINE_AA)

                # ── predicted value at t_land ─────────────────────────────────
                v_pred_cm = float(np.polyval(coeff, t_land)) * 100
                px_p, py_p = to_px(t_land, v_pred_cm)
                cv2.drawMarker(canvas, (px_p, py_p), (80, 255, 80),
                               cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{v_pred_cm:+.1f}",
                            (max(x_l + 2, px_p - 22), max(y_top + 10, py_p - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 255, 80), 1, cv2.LINE_AA)

            # ── raw data points ───────────────────────────────────────────────
            for t_v, v_v in zip(ts, vals):
                px_, py_ = to_px(t_v, v_v)
                cv2.circle(canvas, (px_, py_), 3, (160, 160, 160), -1, cv2.LINE_AA)
            # Highlight latest point
            px_, py_ = to_px(t_latest, vals[-1])
            cv2.circle(canvas, (px_, py_), 5, (255, 80, 80), -1, cv2.LINE_AA)

            # ── Y-axis label + tick values ────────────────────────────────────
            cv2.putText(canvas, ylabel,
                        (4, (y_top + y_bot) // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            for tick_v in [v_min + v_pad, (v_min + v_max) / 2, v_max - v_pad]:
                _, py_t = to_px(0, tick_v)
                cv2.putText(canvas, f"{tick_v:.0f}",
                            (x_l - 54, py_t + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

            # ── fit name (top-right of subplot) ──────────────────────────────
            n_label = f"{fit_name}  n={len(vals)}"
            cv2.putText(canvas, n_label,
                        (x_r - len(n_label) * 5, y_top + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        # ── X-axis time ticks (bottom) ────────────────────────────────────────
        row_i = 2
        y_bot_last = MARGIN_T + (row_i + 1) * ROW_H - 6
        for t_v in np.arange(0.0, t_land + 0.1, 0.1):
            px_t = MARGIN_L + int((t_v / max(t_land + 0.05, 0.01)) * (W - MARGIN_L - MARGIN_R))
            cv2.putText(canvas, f"{t_v:.1f}",
                        (px_t - 7, y_bot_last + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1)

        cv2.putText(canvas, "t (s)", (W - MARGIN_R - 26, y_bot_last + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        # ── title ─────────────────────────────────────────────────────────────
        t_left_str = ""
        if prediction is not None:
            t_left_str = f"  |  t_land in {prediction.get('t_land_from_now', 0):.3f}s"
        cv2.putText(canvas, f"Trajectory Fit{t_left_str}",
                    (MARGIN_L, MARGIN_T - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    def show_fit_plot_window(self, trajectory, prediction):
        """Draw and imshow the fit plot window."""
        img = self.draw_fit_plot(trajectory, prediction)
        cv2.imshow("Trajectory Fit", img)

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

    def show_hsv_mask_window(self, frame_bgr, hsv_mask, motion_mask=None, ball_info=None):
        """
        3-panel HSV Masking window for presentation / debugging.
        Left  : Original RGB (dimmed background, red channel boosted in detected area)
        Middle: HSV mask (detected regions = bright orange-red, rest = black)
        Right : Overlay — original frame with detected region highlighted + contour

        Toggle with 'm' key during runtime.  Off by default in config.
        """
        h, w = frame_bgr.shape[:2]

        # ── Panel 1: Original RGB (slightly dimmed for contrast) ─────────────
        panel_rgb = (frame_bgr * 0.7).astype(np.uint8)
        # Boost detected pixels back to full brightness
        if hsv_mask is not None:
            panel_rgb[hsv_mask > 0] = frame_bgr[hsv_mask > 0]
        cv2.putText(panel_rgb, "RGB Feed", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Panel 2: HSV Mask (colorized) ────────────────────────────────────
        panel_mask = np.zeros((h, w, 3), dtype=np.uint8)
        if hsv_mask is not None:
            # Detected region: orange-red fill
            panel_mask[hsv_mask > 0] = (30, 80, 255)   # BGR orange-red
            # Contour outline in bright green
            if ball_info is not None and "contour" in ball_info:
                cv2.drawContours(panel_mask, [ball_info["contour"]], -1, (0, 255, 80), 2, cv2.LINE_AA)
        # Motion mask overlay: show in dim blue when available
        if motion_mask is not None:
            m_resized = cv2.resize(motion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            motion_only = (m_resized > 0) & (hsv_mask == 0 if hsv_mask is not None else True)
            panel_mask[motion_only] = (80, 30, 0)
        cv2.putText(panel_mask, "HSV Mask", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        if ball_info is not None:
            area = ball_info.get("area", 0)
            cx_b, cy_b = ball_info["cx"], ball_info["cy"]
            cv2.putText(panel_mask, f"area={int(area)}", (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Panel 3: Overlay ──────────────────────────────────────────────────
        panel_overlay = frame_bgr.copy()
        if hsv_mask is not None and np.any(hsv_mask > 0):
            # Semi-transparent orange tint over detected region
            tint = np.zeros_like(frame_bgr)
            tint[hsv_mask > 0] = (0, 80, 255)
            cv2.addWeighted(tint, 0.45, panel_overlay, 1.0, 0, panel_overlay)
        if ball_info is not None:
            # Contour + bounding box + crosshair
            if "contour" in ball_info:
                cv2.drawContours(panel_overlay, [ball_info["contour"]], -1, (0, 255, 80), 2, cv2.LINE_AA)
            if "bbox" in ball_info:
                xb, yb, wb_, hb = ball_info["bbox"]
                cv2.rectangle(panel_overlay, (xb, yb), (xb + wb_, yb + hb), (0, 200, 255), 1)
            cx_b, cy_b = ball_info["cx"], ball_info["cy"]
            cv2.drawMarker(panel_overlay, (cx_b, cy_b), (0, 255, 255),
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            depth = ball_info.get("depth_m", 0.0)
            cv2.putText(panel_overlay, f"{depth:.2f}m", (cx_b + 12, cy_b - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_overlay, "Detection", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Combine and display ───────────────────────────────────────────────
        combined = np.hstack([panel_rgb, panel_mask, panel_overlay])
        cv2.imshow("MCE14 — HSV Masking", combined)

    def reset_trail(self):
        """Clear pixel trail when tracking is lost."""
        self.pixel_trail.clear()

