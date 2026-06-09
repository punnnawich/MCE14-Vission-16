"""
MCE14 Vission-16 — Clip Recorder
=================================
อัดคลิปอัตโนมัติทุกครั้งที่ detect การปล่อยลูก
แต่ละคลิปแสดง:
  • วิถีจริง (pixel trail — green/yellow)
  • วิถีทำนาย (yellow curve ไปถึงจุดตก)
  • จุดตกที่ทำนาย (X marker + พิกัด)
  • ระยะเวลาคำนวณ: release → first prediction (ms)

บันทึกลง  logs/clips/throw_YYYYMMDD_HHMMSS_NNN.mp4
"""

import cv2
import numpy as np
import os
import time
from collections import deque
from datetime import datetime


class ClipRecorder:
    # ─────────────────────────────────────────────────────────────────────────
    # Colours (BGR)
    COL_PANEL_BG  = (20,  20,  20)
    COL_CYAN      = (255, 220,  50)
    COL_GREEN     = (80,  255,  80)
    COL_ORANGE    = (30,  165, 255)
    COL_RED       = (50,   50, 240)
    COL_WHITE     = (220, 220, 220)
    COL_DIM       = (120, 120, 120)
    COL_YELLOW    = (0,   230, 255)
    COL_REC       = (0,    0,  255)
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, config, output_dir=None):
        cam_cfg = config.get("camera", {})
        self.W   = cam_cfg.get("resolution_w", 640)
        self.H   = cam_cfg.get("resolution_h", 360)
        self.FPS = float(cam_cfg.get("fps", 30))

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'clips')
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Pre-throw ring buffer (2 s)
        self._pre_buf: deque = deque(maxlen=int(self.FPS * 2))

        # Recording state
        self._recording    = False
        self._writer       = None
        self._clip_idx     = 0
        self._clip_path    = ""

        # Per-throw stats (reset at each new release)
        self._first_pred_ms: float | None = None
        self._stable_ms: float | None = None
        self._final_pred: dict | None = None
        self._missing_frames = 0

        # Stop conditions
        self.MAX_RECORD_S    = 4.5   # max clip length after release
        self.STOP_MISSING    = 25    # stop if ball missing this many frames (≈0.8s)

        print(f"[ClipRecorder] Output dir: {self.output_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def feed_frame(self,
                   frame_bgr: np.ndarray,
                   ball_detected: bool,
                   prediction: dict | None,
                   released: bool,
                   elapsed_since_release: float,
                   first_pred_ms: float | None,
                   stable_pred_ms: float | None = None):
        """
        Call every frame from vision_pipeline.py.

        frame_bgr            : annotated RGB frame (already has trail + curve overlay)
        ball_detected        : True if ball found this frame
        prediction           : latest prediction dict or None
        released             : True while in released state (ball in flight)
        elapsed_since_release: seconds since release was first detected (0 if not released)
        first_pred_ms        : ms from release to first valid prediction (None = not yet)
        stable_pred_ms       : ms from release to first stable/sent prediction (None = not yet)
        """
        # Stamp the frame with stats overlay
        frame_out = self._draw_stats_panel(
            frame_bgr.copy(), prediction, released,
            elapsed_since_release, first_pred_ms, stable_pred_ms
        )

        # Always push to pre-throw buffer
        self._pre_buf.append(frame_out)

        # ── Auto-start on first release event ─────────────────────────────
        if released and not self._recording:
            self._start_clip()

        # ── While recording ────────────────────────────────────────────────
        if self._recording:
            if self._writer is not None:
                self._writer.write(frame_out)

            # Track stats
            if not ball_detected:
                self._missing_frames += 1
            else:
                self._missing_frames = 0

            if prediction is not None:
                self._final_pred = prediction
            if first_pred_ms is not None and self._first_pred_ms is None:
                self._first_pred_ms = first_pred_ms
            if stable_pred_ms is not None and self._stable_ms is None:
                self._stable_ms = stable_pred_ms

            # ── Stop conditions ────────────────────────────────────────────
            timeout = elapsed_since_release > self.MAX_RECORD_S
            lost    = (self._missing_frames > self.STOP_MISSING
                       and elapsed_since_release > 0.5)

            if timeout or lost:
                self._stop_clip()

    def stop(self):
        """Call in finally block of vision_pipeline."""
        if self._recording:
            self._stop_clip()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _start_clip(self):
        self._recording      = True
        self._missing_frames = 0
        self._first_pred_ms  = None
        self._stable_ms      = None
        self._final_pred     = None
        self._clip_idx      += 1

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clip_path = os.path.join(
            self.output_dir, f"throw_{ts}_{self._clip_idx:03d}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self._clip_path, fourcc, self.FPS, (self.W, self.H)
        )

        # Write pre-throw buffer first
        for f in self._pre_buf:
            self._writer.write(f)

        print(f"[ClipRecorder] ▶  recording → {os.path.basename(self._clip_path)}")

    def _stop_clip(self):
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        fp = self._first_pred_ms
        sp = self._stable_ms
        pred = self._final_pred

        parts = [f"[ClipRecorder] ■  saved: {os.path.basename(self._clip_path)}"]
        if fp is not None:
            parts.append(f"  1st_pred={fp:.0f}ms")
        if sp is not None:
            parts.append(f"  stable={sp:.0f}ms")
        if pred is not None:
            parts.append(
                f"  final → X={pred['x']*100:+.1f}cm Y={pred['y']*100:+.1f}cm"
            )
        print("".join(parts))

    def _draw_stats_panel(self, frame: np.ndarray,
                          prediction, released,
                          elapsed_s, first_pred_ms, stable_pred_ms) -> np.ndarray:
        """
        Adds a semi-transparent stats panel (bottom-right corner) and a REC dot.
        """
        PW, PH = 242, 102          # panel width / height
        px0 = self.W - PW - 4      # panel left-x
        py0 = self.H - PH - 4      # panel top-y

        # Semi-transparent dark background
        roi = frame[py0:py0 + PH, px0:px0 + PW]
        bg  = np.full_like(roi, 25)
        cv2.addWeighted(bg, 0.62, roi, 0.38, 0, roi)
        frame[py0:py0 + PH, px0:px0 + PW] = roi

        def txt(text, row, col=None):
            col = col or self.COL_WHITE
            cv2.putText(frame, text,
                        (px0 + 6, py0 + 16 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # Title
        txt("── PRED TIMING ──", 0, self.COL_CYAN)

        # Release elapsed
        if released:
            txt(f"Release: {elapsed_s*1000:>6.0f} ms", 1, self.COL_GREEN)
        else:
            txt("Release: waiting",             1, self.COL_DIM)

        # First prediction latency
        if first_pred_ms is not None:
            col = self.COL_GREEN if first_pred_ms < 200 else self.COL_ORANGE
            txt(f"1st pred: {first_pred_ms:>5.0f} ms", 2, col)
        else:
            txt("1st pred: ---",                2, self.COL_DIM)

        # Stable / sent prediction latency
        if stable_pred_ms is not None:
            col = self.COL_GREEN if stable_pred_ms < 300 else self.COL_ORANGE
            txt(f"Sent at:  {stable_pred_ms:>5.0f} ms", 3, col)
        else:
            txt("Sent at:  ---",                3, self.COL_DIM)

        # Final prediction coordinates
        if prediction is not None:
            px_cm = prediction["x"] * 100.0
            py_cm = prediction["y"] * 100.0
            t_rem = prediction["t_land_from_now"]
            txt(f"X={px_cm:+.1f} cm  Y={py_cm:+.1f} cm", 4, self.COL_YELLOW)
            txt(f"t_land = {t_rem:.3f} s",               5, (150, 200, 255))
        else:
            txt("Pred: --",                     4, self.COL_DIM)

        # ── REC indicator (top-left) ──────────────────────────────────────
        if self._recording:
            cv2.circle(frame, (14, 14), 8, self.COL_REC, -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (28, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COL_REC, 1, cv2.LINE_AA)

        return frame
