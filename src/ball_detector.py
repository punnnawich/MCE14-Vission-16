import cv2
import numpy as np
from performance import gpu_cvt_color, gpu_in_range, gpu_morphology, gpu_bitwise_or, gpu_bitwise_and, to_cpu, to_gpu, is_gpu_available

class BallDetector:
    def __init__(self, config):
        """
        Initialize ball detector with parameters from configuration dict.
        Includes motion-based detection for improved accuracy and performance.
        GPU-accelerated via OpenCL UMat when available.
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

        # Motion detection (background subtractor)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,          # จำนวนเฟรมเรียนรู้ฉากหลัง
            varThreshold=40,      # Sensitivity ของ motion
            detectShadows=False   # ปิดการตรวจจับเงา (เร็วขึ้น)
        )
        self.motion_mask = None
        self.has_motion = False

        # Morphological kernels (pre-allocated)
        self._kernel_hsv = np.ones((5, 5), np.uint8)
        self._kernel_motion = np.ones((7, 7), np.uint8)

    def detect_red_ball(self, frame_bgr, use_motion=True):
        """
        Thresholds the BGR frame to isolate red color.
        Uses GPU (OpenCL UMat) for HSV conversion, inRange, and morphology.
        Combines HSV mask with motion mask when motion is available.
        Returns the combined mask (always as numpy array for contour detection).
        """
        # GPU-accelerated: BGR → HSV conversion
        frame_hsv = gpu_cvt_color(frame_bgr, cv2.COLOR_BGR2HSV)

        # GPU-accelerated: dual-range red thresholding
        # Red hue range 1: H = 0–10
        mask1 = gpu_in_range(frame_hsv, self.lower1, self.upper1)
        # Red hue range 2: H = 170–180 (wrap-around)
        mask2 = gpu_in_range(frame_hsv, self.lower2, self.upper2)
        hsv_mask = gpu_bitwise_or(mask1, mask2)

        # GPU-accelerated: morphological cleanup
        hsv_mask = gpu_morphology(hsv_mask, self._kernel_hsv, erode_iter=1, dilate_iter=2)

        # Convert to CPU for contour detection (findContours requires numpy array)
        hsv_mask = to_cpu(hsv_mask)

        # C-05: Sub-sample MOG2 motion detection every 2nd frame to save CPU (~2-4ms/frame)
        # Motion mask from previous frame is reused on skipped frames
        self._motion_counter = getattr(self, '_motion_counter', 0) + 1
        if self._motion_counter % 2 == 0 or self.motion_mask is None:
            raw_motion = self.bg_subtractor.apply(frame_bgr, learningRate=0.005)
            # Clean up motion mask (GPU-accelerated dilate)
            if is_gpu_available():
                motion_gpu = to_gpu(raw_motion)
                self.motion_mask = to_cpu(cv2.dilate(motion_gpu, self._kernel_motion, iterations=2))
            else:
                self.motion_mask = cv2.dilate(raw_motion, self._kernel_motion, iterations=2)
            self.has_motion = cv2.countNonZero(self.motion_mask) > 100

        if not use_motion:
            return hsv_mask

        # Combine: prioritize motion areas
        # When motion exists, use intersection of HSV + dilated motion mask
        # This eliminates static red objects (tape, signs, etc.)
        if self.has_motion:
            # Dilate motion mask more aggressively to cover the ball fully
            if is_gpu_available():
                motion_dilated = to_cpu(cv2.dilate(to_gpu(self.motion_mask), self._kernel_motion, iterations=3))
            else:
                motion_dilated = cv2.dilate(self.motion_mask, self._kernel_motion, iterations=3)
            combined_mask = cv2.bitwise_and(hsv_mask, motion_dilated)

            # If combined mask is empty (ball might have just stopped),
            # fall back to pure HSV mask
            if cv2.countNonZero(combined_mask) < 10:
                return hsv_mask
            return combined_mask

        return hsv_mask

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

    @staticmethod
    def adaptive_depth_sample(depth_frame, cx, cy, ball_area, frame_h, frame_w):
        """
        Adaptive depth ROI sampling.
        - Larger ROI when ball is small (far away) for more stable readings
        - Uses closest-half median to reject background pixels
        
        Returns: depth in mm (float), or 0 if invalid
        """
        # Adaptive ROI: smaller ball → bigger ROI (more pixels to sample from)
        if ball_area < 200:
            roi_size = 11     # Very small ball = far away
        elif ball_area < 800:
            roi_size = 9
        elif ball_area < 2000:
            roi_size = 7
        else:
            roi_size = 5      # Big ball = close

        half = roi_size // 2
        y_start = max(0, cy - half)
        y_end = min(frame_h, cy + half + 1)
        x_start = max(0, cx - half)
        x_end = min(frame_w, cx + half + 1)

        depth_roi = depth_frame[y_start:y_end, x_start:x_end]
        valid = depth_roi[depth_roi > 0]

        if len(valid) == 0:
            # Fallback: try larger region
            half2 = half * 2
            y_s = max(0, cy - half2)
            y_e = min(frame_h, cy + half2 + 1)
            x_s = max(0, cx - half2)
            x_e = min(frame_w, cx + half2 + 1)
            roi2 = depth_frame[y_s:y_e, x_s:x_e]
            valid = roi2[roi2 > 0]
            if len(valid) == 0:
                return 0.0

        # Use closest-half median: sort depths, take median of closer half
        # This rejects background pixels that leak into the ROI
        sorted_depths = np.sort(valid)
        half_idx = max(1, len(sorted_depths) // 2)
        return float(np.median(sorted_depths[:half_idx]))
