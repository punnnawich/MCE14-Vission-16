import cv2
import numpy as np

class ReleaseDetector:
    def __init__(self, vel_threshold=1.2, displacement_threshold=0.10, skin_cfg=None):
        """
        Hybrid release detection:
          1. Skin-based (FAST ~33ms): ตรวจว่ามือหายจากรอบลูกบอล + ลูกเริ่มเคลื่อนที่
          2. Velocity+Displacement (FALLBACK ~100ms): dual-threshold แบบเดิม

        vel_threshold: Minimum velocity in meters/second for fallback release.
        displacement_threshold: Minimum distance in meters for fallback release.
        skin_cfg: dict with skin HSV thresholds for hand detection.
        """
        self.prev_pos = None
        self.prev_time = None
        self.start_pos = None
        self.released = False
        self.vel_threshold = vel_threshold
        self.displacement_threshold = displacement_threshold

        # ── Skin-based release detection config ──
        if skin_cfg is None:
            skin_cfg = {}
        self.skin_lower = np.array(skin_cfg.get("skin_lower", [5, 40, 80]))
        self.skin_upper = np.array(skin_cfg.get("skin_upper", [25, 255, 255]))
        # Minimum ratio of skin pixels in annular region to consider "hand present"
        self.skin_ratio_threshold = skin_cfg.get("skin_ratio_threshold", 0.08)
        # Minimum velocity for skin-based release (lower than velocity fallback)
        self.skin_vel_threshold = skin_cfg.get("skin_vel_threshold", 0.5)

        # ── Internal state ──
        self._current_velocity = 0.0
        self.release_method = None   # "skin" or "velocity" — for debugging

    def _check_skin_near_ball(self, frame_bgr, cx, cy, ball_radius):
        """
        Check if skin-colored pixels exist in an annular (donut) region around the ball.

        Annular region:
          inner radius = ball edge (skip the red pixels of the ball itself)
          outer radius = 2.5× ball radius (where the hand/fingers would be)

        Returns: ratio of skin pixels to total pixels in the annular region (0.0 – 1.0).
                 High ratio = hand is present, Low ratio = hand is gone.
        """
        h, w = frame_bgr.shape[:2]

        # Define annular region radii
        inner_r = max(int(ball_radius), 5)
        outer_r = max(int(ball_radius * 2.5), inner_r + 15)

        # Crop a square ROI around the ball (for performance — avoid processing full frame)
        y1 = max(0, cy - outer_r)
        y2 = min(h, cy + outer_r)
        x1 = max(0, cx - outer_r)
        x2 = min(w, cx + outer_r)

        if (y2 - y1) < 10 or (x2 - x1) < 10:
            return 1.0  # ROI too small → assume hand is present (safe default)

        roi = frame_bgr[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create annular (donut) mask in ROI coordinates
        roi_h, roi_w = roi.shape[:2]
        center_x = cx - x1
        center_y = cy - y1

        Y, X = np.ogrid[:roi_h, :roi_w]
        dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
        annular_mask = (dist_sq >= inner_r ** 2) & (dist_sq <= outer_r ** 2)

        total_annulus = np.sum(annular_mask)
        if total_annulus == 0:
            return 1.0

        # Threshold for skin color within annular region
        skin_mask = cv2.inRange(roi_hsv, self.skin_lower, self.skin_upper)
        skin_in_annulus = np.sum((skin_mask > 0) & annular_mask)

        return skin_in_annulus / total_annulus

    def update(self, pos_3d, timestamp, frame_bgr=None, ball_info=None):
        """
        Hybrid release detection (OR logic — whichever triggers first wins):

          Method 1 — Skin (FAST):
            ✅ skin_ratio < threshold  (hand gone from around ball)
            ✅ velocity > 0.5 m/s      (ball is actually moving, not sitting on table)
            → Triggers in ~33ms (1 frame)

          Method 2 — Velocity+Displacement (FALLBACK):
            ✅ velocity > 1.2 m/s
            ✅ displacement > 10 cm
            → Triggers in ~100ms (3-5 frames)

        pos_3d: np.array([x, y, z]) in meters.
        timestamp: float in seconds.
        frame_bgr: BGR frame (optional, for skin detection).
        ball_info: dict with cx, cy, area keys (optional).
        """
        if self.released:
            return True

        if self.start_pos is None:
            self.start_pos = pos_3d.copy()

        if self.prev_pos is None:
            self.prev_pos = pos_3d.copy()
            self.prev_time = timestamp
            return False

        dt = timestamp - self.prev_time
        if dt <= 0:
            return self.released

        # Calculate 3D velocity and displacement
        velocity = np.linalg.norm(pos_3d - self.prev_pos) / dt
        displacement = np.linalg.norm(pos_3d - self.start_pos)
        self._current_velocity = velocity

        self.prev_pos = pos_3d.copy()
        self.prev_time = timestamp

        # ── Method 1: Skin-based release (FAST ~33ms) ──────────
        # Check if hand has left the ball area while ball is moving
        # C-04: Skip expensive skin detection when velocity is low — release can't trigger anyway
        if frame_bgr is not None and ball_info is not None and velocity > self.skin_vel_threshold:
            cx, cy = ball_info["cx"], ball_info["cy"]
            ball_radius = np.sqrt(ball_info["area"] / np.pi)

            skin_ratio = self._check_skin_near_ball(frame_bgr, cx, cy, ball_radius)

            # Release if: no hand detected AND ball has some velocity
            if skin_ratio < self.skin_ratio_threshold:
                self.released = True
                self.release_method = "skin"
                return True

        # ── Method 2: Velocity + Displacement (FALLBACK ~100ms) ──
        # Traditional dual-threshold: prevents depth noise false triggers
        if velocity > self.vel_threshold and displacement > self.displacement_threshold:
            self.released = True
            self.release_method = "velocity"

        return self.released

    def reset(self):
        """
        Reset release detection status.
        """
        self.released = False
        self.prev_pos = None
        self.prev_time = None
        self.start_pos = None
        self._current_velocity = 0.0
        self.release_method = None
