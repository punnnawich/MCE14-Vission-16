import cv2
import numpy as np

class BallDetector:
    def __init__(self, config):
        """
        Initialize ball detector with parameters from configuration dict.
        """
        hsv_cfg = config.get("hsv", {})
        self.lower1 = np.array(hsv_cfg.get("lower_red_1", [0, 100, 80]))
        self.upper1 = np.array(hsv_cfg.get("upper_red_1", [10, 255, 255]))
        self.lower2 = np.array(hsv_cfg.get("lower_red_2", [170, 100, 80]))
        self.upper2 = np.array(hsv_cfg.get("upper_red_2", [180, 255, 255]))

        blob_cfg = config.get("blob", {})
        self.min_area = blob_cfg.get("min_area", 200)
        self.max_area = blob_cfg.get("max_area", 50000)
        self.min_circularity = blob_cfg.get("min_circularity", 0.7)

    def detect_red_ball(self, frame_bgr):
        """
        Thresholds the RGB/BGR frame to isolate red color.
        """
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Red hue range 1: H = 0–10
        mask1 = cv2.inRange(frame_hsv, self.lower1, self.upper1)
        # Red hue range 2: H = 170–180 (wrap-around)
        mask2 = cv2.inRange(frame_hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def find_ball_centroid(self, mask):
        """
        Finds the centroid of the best matching ball contour.
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

            # Bounding box aspect ratio (should be close to square for a ball)
            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
            aspect_ratio = float(w_bb) / h_bb
            if not (0.5 <= aspect_ratio <= 1.5):
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
                    "bbox": (x_bb, y_bb, w_bb, h_bb)
                }

        return best
