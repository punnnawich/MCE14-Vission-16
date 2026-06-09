import cv2
import numpy as np
import os
from ultralytics import YOLO


class BallDetector:
    def __init__(self, config):
        """
        YOLOv8n-based ball detector (COCO sports ball, class 32).
        Red colour is confirmed with HSV thresholds after YOLO detection.
        HSVCalibrator can still update lower1/upper1/lower2/upper2 live.
        """
        # HSV thresholds — used for red-colour confirmation inside YOLO bbox
        # (also updated live by HSVCalibrator when user presses 'c')
        hsv_cfg = config.get("hsv", {})
        self.lower1 = np.array(hsv_cfg.get("lower_red_1", [0,  100, 80]), dtype=np.uint8)
        self.upper1 = np.array(hsv_cfg.get("upper_red_1", [10, 255, 255]), dtype=np.uint8)
        self.lower2 = np.array(hsv_cfg.get("lower_red_2", [170, 100, 80]), dtype=np.uint8)
        self.upper2 = np.array(hsv_cfg.get("upper_red_2", [180, 255, 255]), dtype=np.uint8)

        blob_cfg = config.get("blob", {})
        self.min_area        = blob_cfg.get("min_area",         200)
        self.max_area        = blob_cfg.get("max_area",       50000)
        self.min_circularity = blob_cfg.get("min_circularity",  0.45)

        # YOLOv8n — downloads ~6 MB weight file on first run, then cached
        # COCO class 32 = "sports ball"
        _model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(_model_dir, exist_ok=True)
        _model_path = os.path.join(_model_dir, "yolov8n.pt")
        self.model = YOLO(_model_path)   # auto-downloads if not present
        self.model.fuse()                # merge BN layers for faster CPU inference

        # Minimum red-pixel ratio inside YOLO bbox to accept as red ball
        self._red_ratio_min = 0.12

        # Keep motion_mask attribute for visualizer.colorize_depth() compatibility
        self.motion_mask = None
        self.has_motion  = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_red_ball(self, frame_bgr, use_motion=True, depth_fg_mask=None):
        """
        Run YOLOv8n on frame, keep only 'sports ball' detections (class 32),
        confirm red colour via HSV, return a binary mask.

        The mask is an ellipse drawn at each accepted YOLO bounding box so that
        find_ball_centroid() and contour_depth_sample() work unchanged downstream.
        """
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        results = self.model(
            frame_bgr,
            classes=[32],     # sports ball only
            conf=0.25,
            verbose=False,
            imgsz=640,
        )

        if results and len(results[0].boxes) > 0:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w, x2);  y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Red-colour confirmation — require ≥12% red pixels inside bbox
                roi = hsv[y1:y2, x1:x2]
                red1 = cv2.inRange(roi, self.lower1, self.upper1)
                red2 = cv2.inRange(roi, self.lower2, self.upper2)
                red_pixels = cv2.countNonZero(cv2.bitwise_or(red1, red2))
                bbox_area  = (x2 - x1) * (y2 - y1)
                if red_pixels < bbox_area * self._red_ratio_min:
                    continue

                # Fill an ellipse that matches the YOLO bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                rx = max(1, (x2 - x1) // 2)
                ry = max(1, (y2 - y1) // 2)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        if depth_fg_mask is not None:
            mask = cv2.bitwise_and(mask, depth_fg_mask)

        return mask

    def find_ball_centroid(self, mask):
        """
        Find the best ball contour in the binary mask.
        Unchanged from the original HSV-based implementation.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
            if float(w_bb) / max(h_bb, 1) < 0.3 or float(w_bb) / max(h_bb, 1) > 3.0:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if best is None or area > best["area"]:
                best = {
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "contour": cnt,
                    "bbox": (x_bb, y_bb, w_bb, h_bb),
                }

        return best

    # Pre-allocated erosion kernel (shared across calls)
    _ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @staticmethod
    def contour_depth_sample(depth_frame, contour, bbox):
        """
        Sample depth from pixels inside the ball contour with IQR outlier rejection.
        Unchanged from the original implementation.
        Returns depth in mm (float), or 0 if insufficient valid pixels.
        """
        x_bb, y_bb, w_bb, h_bb = bbox
        if w_bb <= 0 or h_bb <= 0:
            return 0.0

        depth_roi = depth_frame[y_bb:y_bb + h_bb, x_bb:x_bb + w_bb]

        contour_local = (
            contour.reshape(-1, 2) - np.array([x_bb, y_bb])
        ).reshape(-1, 1, 2).astype(np.int32)
        mask_full = np.zeros((h_bb, w_bb), dtype=np.uint8)
        cv2.drawContours(mask_full, [contour_local], -1, 255, cv2.FILLED)

        mask_inner = cv2.erode(mask_full, BallDetector._ERODE_KERNEL)
        valid = depth_roi[mask_inner > 0]
        valid = valid[valid > 0]

        if len(valid) < 5:
            valid = depth_roi[mask_full > 0]
            valid = valid[valid > 0]

        if len(valid) == 0:
            cy_l, cx_l = h_bb // 2, w_bb // 2
            patch = depth_roi[
                max(0, cy_l - 2): min(h_bb, cy_l + 3),
                max(0, cx_l - 2): min(w_bb, cx_l + 3),
            ]
            valid = patch[patch > 0]
            if len(valid) == 0:
                return 0.0

        if len(valid) >= 6:
            q1, q3 = float(np.percentile(valid, 25)), float(np.percentile(valid, 75))
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                inliers = valid[(valid >= lo) & (valid <= hi)]
                if len(inliers) >= 3:
                    valid = inliers

        return float(np.median(valid))
