"""
MCE14 Vission-16 — Adaptive HSV Calibrator
===========================================
วัดสี HSV จากลูกบอลจริงแล้วคำนวณ threshold อัตโนมัติ

วิธีใช้:
  1. กด 'c' — วงกลมเป้าจะขึ้นกลางจอ
  2. นำลูกบอลเข้าไปใน ROI แล้วถือนิ่งๆ
  3. รอ progress bar เต็ม (~1 วินาที)
  4. threshold ใหม่ถูก apply ทันที + บันทึกลง config.yaml

รองรับ Red hue wrap-around (H ≈ 0–10 และ H ≈ 170–180):
  • วิเคราะห์ distribution ของ H แล้วแยก range ให้อัตโนมัติ
  • S/V ใช้ mean – 2σ เป็น lower bound (รองรับ highlight)
"""

import cv2
import numpy as np
import yaml
import os


class HSVCalibrator:
    STATE_IDLE     = "IDLE"
    STATE_SAMPLING = "SAMPLING"
    STATE_DONE     = "DONE"

    # Colour (BGR) for UI
    _COL_ROI    = (0, 255, 120)
    _COL_ACTIVE = (0, 200, 255)
    _COL_DONE   = (80, 255, 80)
    _COL_TXT    = (220, 220, 220)

    def __init__(self, config, n_frames: int = 30, roi_radius: int = 60):
        """
        n_frames   : จำนวน frame ที่ sample (30 = ~1s ที่ 30fps)
        roi_radius : รัศมี ROI วงกลม (px) — ขยายได้ถ้าลูกใหญ่/ใกล้
        """
        cam_cfg = config.get("camera", {})
        self.W = cam_cfg.get("resolution_w", 640)
        self.H = cam_cfg.get("resolution_h", 360)

        self.n_frames  = n_frames
        self.roi_r     = roi_radius
        self.roi_cx    = self.W // 2
        self.roi_cy    = self.H // 2

        # H margin ≥ 12 (อย่างน้อย) กันแสงกระเพื่อม
        self.h_margin_min = 12
        # S/V lower ไม่ต่ำกว่า 40 (กันพื้นขาว/เทา)
        self.sv_lower_min = 40

        self.state    = self.STATE_IDLE
        self._buffer  = []           # list of np.ndarray (1-D H/S/V arrays)
        self._result  = None         # (lower1, upper1, lower2, upper2) or None
        self._prev_result = None     # เก็บผลก่อนหน้าสำหรับ display

        # Pre-build ROI circle mask (same size every frame)
        self._roi_mask = self._build_roi_mask()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """เริ่ม sampling ใหม่ (เรียกเมื่อกด 'c')"""
        self._buffer.clear()
        self._result = None
        self.state   = self.STATE_SAMPLING
        print(f"[HSVCal] Sampling started  ROI centre=({self.roi_cx},{self.roi_cy}) r={self.roi_r}px")

    def feed(self, frame_bgr: np.ndarray) -> bool:
        """
        เรียกทุก frame ขณะ SAMPLING.
        คืน True เมื่อ sampling เสร็จและผลพร้อมใช้
        """
        if self.state != self.STATE_SAMPLING:
            return False

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # ดึงเฉพาะ pixel ใน ROI วงกลม
        roi_pixels = hsv[self._roi_mask > 0]   # shape (N, 3)

        if len(roi_pixels) > 0:
            # กรองเอา pixel ที่ S > 50, V > 50 เท่านั้น (ตัด highlight/เงา)
            sv_ok = (roi_pixels[:, 1] > 50) & (roi_pixels[:, 2] > 50)
            valid = roi_pixels[sv_ok]
            if len(valid) >= 10:
                self._buffer.append(valid)

        if len(self._buffer) >= self.n_frames:
            self._compute_result()
            self.state = self.STATE_DONE
            return True
        return False

    def apply_to_detector(self, ball_detector) -> bool:
        """
        อัปเดต HSV thresholds ใน BallDetector instance.
        คืน True ถ้ามีผลพร้อม
        """
        r = self._result
        if r is None:
            return False
        l1, u1, l2, u2 = r
        ball_detector.lower1 = np.array(l1, dtype=np.uint8)
        ball_detector.upper1 = np.array(u1, dtype=np.uint8)
        ball_detector.lower2 = np.array(l2, dtype=np.uint8)
        ball_detector.upper2 = np.array(u2, dtype=np.uint8)
        self._prev_result = r
        return True

    def save_to_config(self, config_path: str) -> bool:
        """บันทึก threshold ใหม่ลง config.yaml"""
        r = self._result
        if r is None:
            return False
        l1, u1, l2, u2 = r
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            data.setdefault("hsv", {})
            data["hsv"]["lower_red_1"] = [int(v) for v in l1]
            data["hsv"]["upper_red_1"] = [int(v) for v in u1]
            data["hsv"]["lower_red_2"] = [int(v) for v in l2]
            data["hsv"]["upper_red_2"] = [int(v) for v in u2]
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            print(f"[HSVCal] Saved to {os.path.basename(config_path)}")
            print(f"         Red1 H:{l1[0]}-{u1[0]}  S:{l1[1]}-255  V:{l1[2]}-255")
            print(f"         Red2 H:{l2[0]}-{u2[0]}  S:{l2[1]}-255  V:{l2[2]}-255")
            return True
        except Exception as e:
            print(f"[HSVCal] Save failed: {e}")
            return False

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        วาด ROI circle + progress bar + ผลลัพธ์บน frame
        """
        if self.state == self.STATE_IDLE:
            # แสดง ROI 희ม (hint) และคำแนะนำ
            cv2.circle(frame, (self.roi_cx, self.roi_cy), self.roi_r,
                       (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press [c] to calibrate HSV",
                        (self.roi_cx - 115, self.roi_cy + self.roi_r + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 100, 100), 1, cv2.LINE_AA)

        elif self.state == self.STATE_SAMPLING:
            progress = len(self._buffer) / self.n_frames
            pct = int(progress * 100)

            # Pulsing ROI circle (active)
            cv2.circle(frame, (self.roi_cx, self.roi_cy), self.roi_r,
                       self._COL_ACTIVE, 2, cv2.LINE_AA)
            cv2.putText(frame, "Place ball in circle  hold still",
                        (self.roi_cx - 140, self.roi_cy + self.roi_r + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self._COL_ACTIVE, 1, cv2.LINE_AA)

            # Progress bar bellow ROI
            bx = self.roi_cx - 80
            by = self.roi_cy + self.roi_r + 28
            cv2.rectangle(frame, (bx, by), (bx + 160, by + 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(160 * progress), by + 10),
                          self._COL_ACTIVE, -1)
            cv2.putText(frame, f"{pct}%", (bx + 68, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._COL_TXT, 1, cv2.LINE_AA)

        elif self.state == self.STATE_DONE and self._prev_result is not None:
            l1, u1, l2, u2 = self._prev_result
            # Filled ROI circle (success)
            overlay = frame.copy()
            cv2.circle(overlay, (self.roi_cx, self.roi_cy), self.roi_r,
                       self._COL_DONE, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "HSV calibrated!",
                        (self.roi_cx - 75, self.roi_cy + self.roi_r + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, self._COL_DONE, 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"H1:{l1[0]}-{u1[0]}  H2:{l2[0]}-{u2[0]}  "
                        f"S>{l1[1]}  V>{l1[2]}",
                        (self.roi_cx - 150, self.roi_cy + self.roi_r + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self._COL_DONE, 1, cv2.LINE_AA)

        return frame

    # Properties for pipeline checks
    @property
    def is_sampling(self) -> bool:
        return self.state == self.STATE_SAMPLING

    @property
    def is_done(self) -> bool:
        return self.state == self.STATE_DONE

    @property
    def progress(self) -> float:
        return len(self._buffer) / self.n_frames if self.is_sampling else 0.0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(mask, (self.roi_cx, self.roi_cy), self.roi_r, 255, -1)
        return mask

    def _compute_result(self):
        """
        Compute HSV thresholds from collected samples.
        Handles red hue wrap-around automatically.
        """
        all_pixels = np.vstack(self._buffer)   # (N_total, 3) — H,S,V
        H = all_pixels[:, 0].astype(np.float32)
        S = all_pixels[:, 1].astype(np.float32)
        V = all_pixels[:, 2].astype(np.float32)

        # ── Hue wrap-around handling ─────────────────────────────────────
        # Red straddles H=0: low-end (0-10) and high-end (170-180).
        # Unwrap: shift H > 90 to negative → -10 to +10 range
        H_unwrapped = np.where(H > 90, H - 180.0, H)

        h_mean = float(np.mean(H_unwrapped))
        h_std  = float(np.std(H_unwrapped))
        h_margin = max(self.h_margin_min, int(np.ceil(2.0 * h_std)))

        # S / V bounds
        s_mean = float(np.mean(S))
        s_std  = float(np.std(S))
        v_mean = float(np.mean(V))
        v_std  = float(np.std(V))

        s_lo = max(self.sv_lower_min, int(s_mean - 2.0 * s_std))
        v_lo = max(self.sv_lower_min, int(v_mean - 2.0 * v_std))

        # ── Build dual-range thresholds ──────────────────────────────────
        h_lo_raw = h_mean - h_margin
        h_hi_raw = h_mean + h_margin

        # Range 1: low-hue side (wraps from negative → 0)
        h1_lo = max(0,   int(h_lo_raw) if h_lo_raw >= 0 else 0)
        h1_hi = min(179, int(h_hi_raw) if h_hi_raw <= 10 else 10)

        # Range 2: high-hue side (wraps to 180)
        # unwrapped h_mean ≈ -5 → real H ≈ 175; add 180 to put back in [160-180] space
        h2_lo = max(0,   int(h_lo_raw + 180))
        h2_hi = min(179, int(h_hi_raw + 180))

        # Clamp ranges sensibly
        h1_lo = max(0,   h1_lo)
        h1_hi = min(15,  h1_hi)
        h2_lo = max(160, h2_lo)
        h2_hi = min(179, h2_hi)

        lower1 = [h1_lo, s_lo, v_lo]
        upper1 = [h1_hi, 255, 255]
        lower2 = [h2_lo, s_lo, v_lo]
        upper2 = [h2_hi, 255, 255]

        self._result = (lower1, upper1, lower2, upper2)

        print(f"[HSVCal] ✓  h_mean={h_mean:.1f}° ±{h_margin}°  "
              f"S≥{s_lo}  V≥{v_lo}  n_pixels={len(all_pixels)}")
        print(f"         Range1 H:{h1_lo}-{h1_hi}   Range2 H:{h2_lo}-{h2_hi}")
