"""
hud.py
------
Person B | HUD (Heads-Up Display) Layer

All drawing logic lives here — circles, labels, count panel, FPS counter.
Keep this module separate so it's easy to restyle without touching pipeline logic.
"""

import cv2
import numpy as np
import time
from collections import deque


# ── Palette ──────────────────────────────────────────────────────────────────
BGR = {
    "red":    (0,   30,  220),
    "green":  (0,   200, 60),
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "yellow": (0,   220, 220),
    "gray":   (160, 160, 160),
    "panel":  (20,  20,  20),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


# ── FPS Counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Smooth FPS using a rolling window of timestamps.
    Call .tick() once per processed frame; call .fps() to read the value.
    """

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── HUD Renderer ──────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Draws all on-screen elements onto a frame in-place.

    Parameters
    ----------
    circle_thickness : int  – outline thickness for ball circles
    label_font_scale : float
    """

    def __init__(self, circle_thickness: int = 2, label_font_scale: float = 0.55):
        self.circle_thickness = circle_thickness
        self.label_font_scale = label_font_scale
        self.fps_counter      = FPSCounter()

    # ------------------------------------------------------------------
    # Ball annotations
    # ------------------------------------------------------------------

    def draw_tracked_balls(
        self,
        frame: np.ndarray,
        tracked: dict,         # output from CentroidTracker.update()
        radii: dict | None = None,  # optional {obj_id: radius_px}
    ) -> None:
        """
        Draw a circle + label for every tracked object.

        tracked : {obj_id: {"centroid": (x,y), "color": "red"|"green"}}
        radii   : optional radius per ID; falls back to a default of 20 px.
        """
        default_radius = 20

        for obj_id, info in tracked.items():
            cx, cy = info["centroid"]
            color  = info["color"]
            bgr    = BGR.get(color, BGR["white"])
            radius = (radii or {}).get(obj_id, default_radius)

            # Outer circle
            cv2.circle(frame, (cx, cy), radius, bgr, self.circle_thickness)
            # Inner cross-hair dot
            cv2.circle(frame, (cx, cy), 3, bgr, -1)

            # Label  "Target: Red  #3"
            label   = f"Target: {color.capitalize()}  #{obj_id}"
            (tw, th), _ = cv2.getTextSize(label, FONT, self.label_font_scale, 1)

            # Background pill for readability
            pad = 4
            cv2.rectangle(
                frame,
                (cx - pad, cy - radius - th - pad * 2),
                (cx + tw + pad, cy - radius + pad),
                BGR["panel"], -1
            )
            cv2.putText(
                frame, label,
                (cx, cy - radius - pad),
                FONT, self.label_font_scale, bgr, 1, cv2.LINE_AA
            )

    # ------------------------------------------------------------------
    # Info panel  (top-left corner)
    # ------------------------------------------------------------------

    def draw_info_panel(
        self,
        frame: np.ndarray,
        counts: dict,           # {"red": N, "green": M}
        fps: float,
        extra_lines: list[str] | None = None,
    ) -> None:
        """
        Draws a semi-transparent panel in the top-left corner showing:
          • Red  : N
          • Green: M
          • FPS  : XX.X
          • Any extra_lines you pass in
        """
        h, w = frame.shape[:2]
        panel_w = 220
        lines   = [
            f"  Red   : {counts.get('red',  0):>3}",
            f"  Green : {counts.get('green',0):>3}",
            f"  FPS   : {fps:>5.1f}",
        ]
        if extra_lines:
            lines += [f"  {l}" for l in extra_lines]

        line_h  = 26
        panel_h = len(lines) * line_h + 20

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), BGR["panel"], -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Title bar
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 32), (40, 40, 40), -1)
        cv2.putText(frame, "BALL TRACKER v1.0", (16, 26),
                    FONT_BOLD, 0.5, BGR["yellow"], 1, cv2.LINE_AA)

        # Data rows
        for i, line in enumerate(lines):
            y  = 32 + line_h * (i + 1) - 4
            # Colour-code the counts
            text_color = BGR["white"]
            if "Red" in line:
                text_color = BGR["red"]
            elif "Green" in line:
                text_color = BGR["green"]
            elif "FPS" in line:
                fps_val = fps
                text_color = BGR["green"] if fps_val >= 20 else (
                    BGR["yellow"] if fps_val >= 10 else (0, 0, 255)
                )
            cv2.putText(frame, line, (16, y),
                        FONT, 0.52, text_color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Tick helper  (call once per rendered frame)
    # ------------------------------------------------------------------

    def tick_and_draw(
        self,
        frame:   np.ndarray,
        tracked: dict,
        counts:  dict,
        radii:   dict | None = None,
        extra_lines: list[str] | None = None,
    ) -> float:
        """
        Convenience wrapper:
          1. Ticks FPS counter
          2. Draws all ball annotations
          3. Draws info panel
        Returns current FPS value.
        """
        self.fps_counter.tick()
        fps = self.fps_counter.fps()

        self.draw_tracked_balls(frame, tracked, radii)
        self.draw_info_panel(frame, counts, fps, extra_lines)

        return fps
