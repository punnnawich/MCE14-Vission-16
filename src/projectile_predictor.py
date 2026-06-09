import numpy as np
from collections import deque


class ProjectilePredictor:
    def __init__(self, config):
        pred_cfg = config.get("predictor", {})
        self.min_points = pred_cfg.get("min_points", 4)
        self.z_catch = pred_cfg.get("z_catch", 0.25)
        self.drag_correction = pred_cfg.get("drag_correction", 0.92)
        self.workspace_radius_m = pred_cfg.get("workspace_radius_m", 0.5)

        # Buffer: (X, Y, Z, timestamp)
        self.buffer = deque(maxlen=20)

    # ── RANSAC linear regression (Irondron Algorithm approach) ───────────────
    # Robust to stereo depth noise on Y axis.  Uses all-pairs slope enumeration
    # (Theil-Sen median estimator) which is equivalent to RANSAC on a clean
    # dataset and avoids random sampling instability for small buffers.
    @staticmethod
    def _ransac_linear(ts, ys, inlier_thresh=0.05):
        """
        Theil-Sen robust linear fit: median of all pairwise slopes.
        inlier_thresh: residual threshold in metres (matches Irondron's 0.05 m).
        Returns [slope, intercept] consistent with np.polyfit order.
        """
        n = len(ts)
        if n < 2:
            return np.polyfit(ts, ys, 1)

        # All pairwise slopes
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                dt = ts[j] - ts[i]
                if abs(dt) > 1e-9:
                    slopes.append((ys[j] - ys[i]) / dt)
        if not slopes:
            return np.polyfit(ts, ys, 1)

        slope = float(np.median(slopes))
        intercept = float(np.median(ys - slope * ts))

        # Refine: re-fit on inliers only (mirrors RANSAC refinement step)
        residuals = np.abs(ys - (slope * ts + intercept))
        inliers = residuals < inlier_thresh
        if inliers.sum() >= 2:
            slope = float(np.mean((ys[inliers] - intercept) / ts[inliers])
                          if False else slope)  # keep median slope, refine intercept
            intercept = float(np.median(ys[inliers] - slope * ts[inliers]))

        return np.array([slope, intercept])

    def add_point(self, pos_3d, timestamp):
        if pos_3d is not None:
            self.buffer.append((*pos_3d, timestamp))

    def predict_landing(self):
        """
        Landing prediction following the Irondron Algorithm approach:

        X axis  — quadratic fit (linear velocity + air-drag deceleration term).
        Y axis  — RANSAC / Theil-Sen robust linear fit (handles stereo depth
                  noise without needing a separate y_moving gating condition).
        Z axis  — standard parabolic fit under gravity.

        Landing time: solve Z(t) = z_catch (= 0.25 m, 25 cm from floor).
        X, Y landing positions: evaluate their fitted equations at t_land.

        Returns dict with predicted x, y, z and diagnostics, or None.
        """
        if len(self.buffer) < self.min_points:
            return None

        pts = np.array(self.buffer)
        xs, ys, zs, ts = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

        if (ts[-1] - ts[0]) < 0.066:
            return None

        t0 = ts[0]
        ts_norm = ts - t0
        t_latest = ts_norm[-1]

        # Exponential time-weighting (τ = 0.15 s, recent frames dominate)
        weights = np.exp((ts_norm - t_latest) / 0.15)

        # ── Z: parabolic height fit ──────────────────────────────────────────
        try:
            coeff_z = np.polyfit(ts_norm, zs, 2, w=weights)
            a, b, c = coeff_z
        except np.linalg.LinAlgError:
            return None

        if not (-12.0 <= a <= -1.5):
            return None

        z_rms = float(np.sqrt(np.mean((zs - np.polyval(coeff_z, ts_norm)) ** 2)))
        if z_rms > 0.08:
            return None

        vz_latest = 2 * a * t_latest + b

        # ── X: linear fit (Irondron: lstsq degree-1, constant velocity) ────────
        # X(t) = vx*t + x0  — no drag term, matches Irondron fit_parabolic_curve
        try:
            coeff_x = np.polyfit(ts_norm, xs, 1, w=weights)
        except np.linalg.LinAlgError:
            return None

        vx_latest = float(coeff_x[0])   # constant velocity (slope)

        # ── Y: RANSAC / Theil-Sen robust linear fit (Irondron: fit_curve_yaxis) ─
        coeff_y = self._ransac_linear(ts_norm, ys)
        vy = coeff_y[0]

        # Horizontal speed sanity check
        h_speed = np.sqrt(vx_latest ** 2 + vy ** 2)
        if not (0.3 <= h_speed <= 8.0):
            return None

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]

        # Velocity–displacement consistency
        if abs(dx) > 0.015 and vx_latest * dx <= 0:
            return None
        if abs(dy) > 0.015 and vy * dy <= 0:
            return None

        # ── Solve Z(t) = z_catch (Irondron: fine_time_to_ground) ────────────
        discriminant = b ** 2 - 4 * a * (c - self.z_catch)
        if discriminant < 0:
            return None

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t_land = max(t1, t2)  # descent crossing

        if t_land <= t_latest:
            return None
        if t_land - t_latest > 3.0:
            return None

        # ── X landing (Irondron: px = Eqx[0]*t + Eqx[1], direct extrapolation) ─
        x_land = float(np.polyval(coeff_x, t_land))
        x_raw  = x_land   # no separate raw vs corrected for linear model

        # ── Y landing (Irondron: py = Eqy[0]*t + Eqy[1], direct extrapolation) ─
        y_land = float(np.polyval(coeff_y, t_land))
        y_raw  = y_land

        # ── Landing direction sanity check ───────────────────────────────────
        if abs(dx) > 0.015 and (x_land - xs[0]) * dx <= 0:
            return None
        if abs(dy) > 0.015 and (y_land - ys[0]) * dy <= 0:
            return None

        # ── Clamp to workspace radius ────────────────────────────────────────
        x_unclamped = x_land
        y_unclamped = y_land
        dist = np.sqrt(x_land ** 2 + y_land ** 2)
        is_clamped = False
        if dist > self.workspace_radius_m:
            x_land = (x_land / dist) * self.workspace_radius_m
            y_land = (y_land / dist) * self.workspace_radius_m
            is_clamped = True

        return {
            "x":               x_land,
            "y":               y_land,
            "z":               self.z_catch,
            "t_land_from_now": t_land - t_latest,
            "raw_x":           x_raw,
            "raw_y":           y_raw,
            "unclamped_x":     x_unclamped,
            "unclamped_y":     y_unclamped,
            "is_clamped":      is_clamped,
            "vx":              vx_latest,
            "vy":              vy,
            "vz":              vz_latest,
            "n_pts":           len(self.buffer),
            "coeff_x": coeff_x.tolist() if isinstance(coeff_x, np.ndarray) else coeff_x,
            "coeff_y": coeff_y.tolist() if isinstance(coeff_y, np.ndarray) else coeff_y,
            "coeff_z": coeff_z.tolist() if isinstance(coeff_z, np.ndarray) else coeff_z,
            "t_start":  float(ts_norm[0]),
            "t_land":   float(t_land),
            "t_latest": float(t_latest),
        }

    def reset(self):
        self.buffer.clear()
