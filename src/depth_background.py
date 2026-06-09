"""
MCE14 Vission-16 — Depth Background Model
==========================================
วัด depth พื้นหลัง (median ของ N frames) ตอนเริ่ม
แล้วสร้าง foreground mask ให้ ball_detector ใช้:

    foreground = bg_depth - current_depth > fg_thresh_mm
                 AND current_depth > 0

ทำให้ลูกบอลที่อยู่ใกล้กว่าพื้นหลัง (เช่น กำแพง, พื้น, ป้าย)
ถูกตรวจจับได้แม้พื้นหลังจะมีสีใกล้เคียงกัน

วิธีใช้:
    bg = DepthBackground(config)
    # กด 'b' เพื่อเริ่มวัด
    bg.start_capture()
    # ทุก frame ใน main loop
    bg.feed(depth_frame)           # เติม buffer ขณะ capturing
    mask = bg.foreground_mask(depth_frame, (H, W))  # None = ยังไม่พร้อม
"""

import cv2
import numpy as np


class DepthBackground:
    STATE_IDLE      = "IDLE"
    STATE_CAPTURING = "CAPTURING"
    STATE_READY     = "READY"

    def __init__(self, config):
        bg_cfg = config.get("background_depth", {})
        self.n_frames    = int(bg_cfg.get("n_frames",      20))
        self.fg_thresh   = int(bg_cfg.get("fg_thresh_mm", 150))   # mm
        # dilation to cover ball edge (fg pixels slightly larger than diff)
        self._dilate_px  = int(bg_cfg.get("dilate_px",    12))

        self.state       = self.STATE_IDLE
        self._buffer     = []
        self.bg_frame    = None          # uint16, same shape as depth camera
        self._kernel     = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._dilate_px * 2 + 1,) * 2
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start_capture(self):
        """เริ่มเก็บ background frames ใหม่ (เรียกเมื่อกด 'b')"""
        self._buffer.clear()
        self.bg_frame = None
        self.state    = self.STATE_CAPTURING
        print(f"[BG] Capturing background depth ({self.n_frames} frames)…")

    def feed(self, depth_frame: np.ndarray):
        """เรียกทุก frame ขณะ state == CAPTURING"""
        if self.state != self.STATE_CAPTURING or depth_frame is None:
            return
        self._buffer.append(depth_frame.astype(np.float32))
        if len(self._buffer) >= self.n_frames:
            stack = np.stack(self._buffer, axis=0)
            self.bg_frame = np.median(stack, axis=0).astype(np.uint16)
            self._buffer.clear()
            self.state = self.STATE_READY
            valid = int(np.sum(self.bg_frame > 0))
            total = self.bg_frame.size
            print(f"[BG] Background captured  "
                  f"valid={valid}/{total} ({100*valid/total:.1f}%)  "
                  f"thresh={self.fg_thresh}mm")

    def foreground_mask(self, depth_frame: np.ndarray,
                        out_shape_hw: tuple | None = None) -> np.ndarray | None:
        """
        คืน uint8 mask (255 = foreground, 0 = background) หรือ None ถ้ายังไม่พร้อม.
        out_shape_hw: (H, W) สำหรับ resize ให้ตรงกับ RGB frame ถ้าต่างกัน
        """
        if self.state != self.STATE_READY or self.bg_frame is None:
            return None
        if depth_frame is None:
            return None

        bg   = self.bg_frame.astype(np.int32)
        curr = depth_frame.astype(np.int32)

        # foreground: current closer than background by > threshold
        diff = bg - curr
        fg = np.zeros(curr.shape, dtype=np.uint8)
        fg[(diff > self.fg_thresh) & (curr > 0)] = 255

        # pixels where background has no valid reading → pass through (don't block)
        fg[bg <= 0] = 255

        # Dilate foreground slightly to cover ball edges
        if self._dilate_px > 0:
            fg = cv2.dilate(fg, self._kernel, iterations=1)

        # Resize to RGB frame resolution if needed
        if out_shape_hw is not None:
            fh, fw = out_shape_hw
            if fg.shape != (fh, fw):
                fg = cv2.resize(fg, (fw, fh), interpolation=cv2.INTER_NEAREST)

        return fg

    # ── Properties for UI ─────────────────────────────────────────────────────

    @property
    def is_capturing(self) -> bool:
        return self.state == self.STATE_CAPTURING

    @property
    def is_ready(self) -> bool:
        return self.state == self.STATE_READY

    @property
    def progress(self) -> float:
        """0.0 – 1.0 ระหว่าง capturing"""
        if self.state == self.STATE_CAPTURING:
            return len(self._buffer) / self.n_frames
        return 1.0 if self.state == self.STATE_READY else 0.0

    def debug_colormap(self, depth_frame: np.ndarray,
                       out_shape_hw: tuple | None = None) -> np.ndarray | None:
        """
        คืน BGR image แสดง depth difference map สำหรับ debug
        เขียว = foreground (ลูก), แดง = background filtered
        """
        if self.bg_frame is None or depth_frame is None:
            return None
        bg   = self.bg_frame.astype(np.int32)
        curr = depth_frame.astype(np.int32)
        diff = np.clip(bg - curr, 0, 1000).astype(np.uint8)
        diff_norm = (diff * 255 // 1000).astype(np.uint8)
        colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        if out_shape_hw is not None:
            fh, fw = out_shape_hw
            if colored.shape[:2] != (fh, fw):
                colored = cv2.resize(colored, (fw, fh))
        return colored
