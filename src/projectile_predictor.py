import numpy as np
from collections import deque

class ProjectilePredictor:
    def __init__(self, config):
        """
        Initialize predictor with configuration parameters.
        """
        pred_cfg = config.get("predictor", {})
        self.min_points = pred_cfg.get("min_points", 4)
        self.z_floor = pred_cfg.get("z_floor", 0.0)
        self.drag_correction = pred_cfg.get("drag_correction", 0.92)
        
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
        Fit curves to current trajectory and predict the landing point on the floor.
        Returns:
            dict containing predicted x, y, z coordinates and time-to-impact (seconds),
            or None if prediction is not possible.
        """
        if len(self.buffer) < self.min_points:
            return None  # Insufficient data points

        pts = np.array(self.buffer)
        xs, ys, zs, ts = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

        # Normalize time starting from 0 to prevent numerical instability
        t0 = ts[0]
        ts_norm = ts - t0

        # Fit vertical motion Z(t) = a*t^2 + b*t + c
        # (Must yield a downward acceleration, i.e., a < 0)
        try:
            coeff_z = np.polyfit(ts_norm, zs, 2)
            a, b, c = coeff_z
        except np.linalg.LinAlgError:
            return None

        if a >= 0:
            return None  # Ball is not accelerating downwards (invalid flight curve)

        # Fit horizontal motions X(t) and Y(t) linearly
        try:
            coeff_x = np.polyfit(ts_norm, xs, 1)
            coeff_y = np.polyfit(ts_norm, ys, 1)
        except np.linalg.LinAlgError:
            return None

        # Solve for t_land where Z(t) = z_floor
        # a*t^2 + b*t + (c - z_floor) = 0
        discriminant = b**2 - 4 * a * (c - self.z_floor)
        if discriminant < 0:
            return None  # No real intersection with the floor plane

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Select the valid future landing time
        t_land = None
        for t in [t1, t2]:
            if t > ts_norm[-1]:
                if t_land is None or t < t_land:
                    t_land = t

        if t_land is None:
            return None

        # Predict horizontal coordinates at t_land
        x_raw = np.polyval(coeff_x, t_land)
        y_raw = np.polyval(coeff_y, t_land)

        # Apply air drag correction to scale coordinates towards release position
        # For a standard trajectory, the drag pulls the ball closer to origin
        # Release coordinates (X0, Y0)
        x0 = coeff_x[1]
        y0 = coeff_y[1]
        x_land = x0 + (x_raw - x0) * self.drag_correction
        y_land = y0 + (y_raw - y0) * self.drag_correction

        return {
            "x": x_land,
            "y": y_land,
            "z": self.z_floor,
            "t_land_from_now": t_land - ts_norm[-1],
            "raw_x": x_raw,
            "raw_y": y_raw
        }

    def reset(self):
        """
        Clear prediction buffers.
        """
        self.buffer.clear()
