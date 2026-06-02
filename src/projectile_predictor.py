import numpy as np
from collections import deque

class ProjectilePredictor:
    def __init__(self, config):
        """
        Initialize predictor with configuration parameters.
        z_catch: ความสูงที่หุ่นยนต์รับลูกบอล (เช่น 0.25m = 25cm)
        """
        pred_cfg = config.get("predictor", {})
        self.min_points = pred_cfg.get("min_points", 4)
        self.z_catch = pred_cfg.get("z_catch", 0.25)
        self.drag_correction = pred_cfg.get("drag_correction", 0.92)
        self.workspace_radius_m = pred_cfg.get("workspace_radius_m", 0.5)
        
        # Buffer to store trajectory points: (X, Y, Z, timestamp)
        self.buffer = deque(maxlen=20)

    def add_point(self, pos_3d, timestamp):
        """
        Add a 3D point (meters) with its timestamp (seconds) to the buffer.
        """
        if pos_3d is not None:
            self.buffer.append((*pos_3d, timestamp))

    def predict_landing(self):
        """
        Fit curves to current trajectory and predict where the ball will be
        at z_catch height (25cm) while descending.
        
        Logic:
          1. Fit Z(t) = a*t² + b*t + c (parabolic curve)
          2. Check a < 0 (gravity pulling down)
          3. Check ball is past peak (descending): dZ/dt at latest time < 0
          4. Solve Z(t) = z_catch for future t
          5. Predict X(t), Y(t) at that time
        
        Returns:
            dict with predicted x, y, z coordinates and time-to-impact,
            or None if prediction is not possible.
        """
        if len(self.buffer) < self.min_points:
            return None  # Insufficient data points

        pts = np.array(self.buffer)
        xs, ys, zs, ts = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

        # CRITICAL SAFETY: Ensure points span a sufficient time window
        # 66ms ≈ 2 frame intervals at 30 FPS — minimum for stable quadratic fit
        if (ts[-1] - ts[0]) < 0.066:
            return None

        # Normalize time starting from 0 to prevent numerical instability
        t0 = ts[0]
        ts_norm = ts - t0

        # Fit vertical motion Z(t) = a*t² + b*t + c
        try:
            coeff_z = np.polyfit(ts_norm, zs, 2)
            a, b, c = coeff_z
        except np.linalg.LinAlgError:
            return None

        if a >= 0:
            return None  # Ball is not accelerating downwards (invalid flight curve)

        # Velocity at latest time: dZ/dt = 2a*t + b (for debug output)
        t_latest = ts_norm[-1]
        vz_latest = 2 * a * t_latest + b

        # Fit horizontal motions X(t) and Y(t) linearly
        try:
            coeff_x = np.polyfit(ts_norm, xs, 1)
            coeff_y = np.polyfit(ts_norm, ys, 1)
        except np.linalg.LinAlgError:
            return None

        # Solve for t_land where Z(t) = z_catch
        # a*t² + b*t + (c - z_catch) = 0
        discriminant = b**2 - 4 * a * (c - self.z_catch)
        if discriminant < 0:
            return None  # No real intersection with the catch plane

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Always pick the LATER root = the descent crossing through z_catch
        # (the earlier root is the ascent crossing, which we don't want)
        t_land = max(t1, t2)

        # Must be in the future
        if t_land <= t_latest:
            return None

        # Predict horizontal coordinates at t_land
        x_raw = np.polyval(coeff_x, t_land)
        y_raw = np.polyval(coeff_y, t_land)

        # Apply air drag correction to scale coordinates towards release position
        x0 = coeff_x[1]
        y0 = coeff_y[1]
        x_land = x0 + (x_raw - x0) * self.drag_correction
        y_land = y0 + (y_raw - y0) * self.drag_correction

        # Save drag-corrected coordinates before clamping
        x_unclamped = x_land
        y_unclamped = y_land

        # Clamp horizontal coordinates to the workspace circle radius
        dist = np.sqrt(x_land**2 + y_land**2)
        is_clamped = False
        if dist > self.workspace_radius_m:
            x_land = (x_land / dist) * self.workspace_radius_m
            y_land = (y_land / dist) * self.workspace_radius_m
            is_clamped = True

        return {
            "x": x_land,
            "y": y_land,
            "z": self.z_catch,
            "t_land_from_now": t_land - t_latest,
            "raw_x": x_raw,
            "raw_y": y_raw,
            "unclamped_x": x_unclamped,
            "unclamped_y": y_unclamped,
            "is_clamped": is_clamped,
            "vz": vz_latest,   # Current vertical velocity (for debugging)
            "coeff_x": coeff_x.tolist() if isinstance(coeff_x, np.ndarray) else coeff_x,
            "coeff_y": coeff_y.tolist() if isinstance(coeff_y, np.ndarray) else coeff_y,
            "coeff_z": coeff_z.tolist() if isinstance(coeff_z, np.ndarray) else coeff_z,
            "t_start": float(ts_norm[0]),
            "t_land": float(t_land)
        }

    def reset(self):
        """
        Clear prediction buffers.
        """
        self.buffer.clear()
