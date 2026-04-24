"""
hsv_control_panel.py
---------------------
Person B | Trackbar Control Panel

Creates a cv2 window with sliders so Person A can tune HSV ranges
while the program is running — no restarts needed.

Usage
-----
    panel = HSVControlPanel()
    panel.create()

    # Inside the main loop:
    hsv_ranges = panel.get_ranges()
    # returns:
    # {
    #   "red":   {"lower1": array, "upper1": array,
    #             "lower2": array, "upper2": array},   # red wraps hue
    #   "yellow": {"lower":  array, "upper":  array},
    # }
"""

import cv2
import numpy as np


WINDOW_NAME = "HSV Control Panel"


class HSVControlPanel:
    """
    Provides two trackbar groups:
      RED  : H-Lo1, H-Hi1, H-Lo2, H-Hi2  (red wraps 0..10 and 160..180)
             S-Lo, S-Hi, V-Lo, V-Hi
    YELLOW: H-Lo, H-Hi, S-Lo, S-Hi, V-Lo, V-Hi

    All trackbars live in a single named window.
    """

    # ── Default HSV ranges (good indoor starting points) ──────────────
    DEFAULTS = {
        # Red hue wraps: 0-10 and 160-180
        "R_H_Lo1": 0,   "R_H_Hi1": 10,
        "R_H_Lo2": 160, "R_H_Hi2": 180,
        "R_S_Lo":  120, "R_S_Hi":  255,
        "R_V_Lo":  70,  "R_V_Hi":  255,
        # Yellow hue: ~20-35 (typical indoor starting point)
        "Y_H_Lo":  20,  "Y_H_Hi":  35,
        "Y_S_Lo":  80,  "Y_S_Hi":  255,
        "Y_V_Lo":  80,  "Y_V_Hi":  255,
    }

    def __init__(self, window_name: str = WINDOW_NAME):
        self.win = window_name

    # ------------------------------------------------------------------

    def create(self) -> None:
        """Create the trackbar window (call once before the main loop)."""
        # A small black canvas — just enough for OpenCV to attach trackbars
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 400, 560)

        d = self.DEFAULTS

        # ── RED group ─────────────────────────────────────────────────
        cv2.createTrackbar("RED  H-Lo1 (0–10)",   self.win, d["R_H_Lo1"], 180, self._noop)
        cv2.createTrackbar("RED  H-Hi1 (0–10)",   self.win, d["R_H_Hi1"], 180, self._noop)
        cv2.createTrackbar("RED  H-Lo2 (160–180)",self.win, d["R_H_Lo2"], 180, self._noop)
        cv2.createTrackbar("RED  H-Hi2 (160–180)",self.win, d["R_H_Hi2"], 180, self._noop)
        cv2.createTrackbar("RED  S-Lo",            self.win, d["R_S_Lo"],  255, self._noop)
        cv2.createTrackbar("RED  S-Hi",            self.win, d["R_S_Hi"],  255, self._noop)
        cv2.createTrackbar("RED  V-Lo",            self.win, d["R_V_Lo"],  255, self._noop)
        cv2.createTrackbar("RED  V-Hi",            self.win, d["R_V_Hi"],  255, self._noop)

        # ── YELLOW group ──────────────────────────────────────────────
        cv2.createTrackbar("YLW  H-Lo",            self.win, d["Y_H_Lo"],  180, self._noop)
        cv2.createTrackbar("YLW  H-Hi",            self.win, d["Y_H_Hi"],  180, self._noop)
        cv2.createTrackbar("YLW  S-Lo",            self.win, d["Y_S_Lo"],  255, self._noop)
        cv2.createTrackbar("YLW  S-Hi",            self.win, d["Y_S_Hi"],  255, self._noop)
        cv2.createTrackbar("YLW  V-Lo",            self.win, d["Y_V_Lo"],  255, self._noop)
        cv2.createTrackbar("YLW  V-Hi",            self.win, d["Y_V_Hi"],  255, self._noop)

    # ------------------------------------------------------------------

    @staticmethod
    def _noop(_) -> None:
        """OpenCV requires a callback; this one does nothing."""
        pass

    # ------------------------------------------------------------------

    def _tb(self, name: str) -> int:
        """Read a single trackbar value."""
        return cv2.getTrackbarPos(name, self.win)

    def get_ranges(self) -> dict:
        """
        Read all trackbars and return structured HSV range dicts ready
        for use in Person A's get_mask() function.

        Returns
        -------
        {
          "red": {
              "lower1": np.array([H, S, V]),
              "upper1": np.array([H, S, V]),
              "lower2": np.array([H, S, V]),
              "upper2": np.array([H, S, V]),
          },
                    "yellow": {
              "lower": np.array([H, S, V]),
              "upper": np.array([H, S, V]),
          }
        }
        """
        r_s_lo = self._tb("RED  S-Lo")
        r_s_hi = self._tb("RED  S-Hi")
        r_v_lo = self._tb("RED  V-Lo")
        r_v_hi = self._tb("RED  V-Hi")

        y_s_lo = self._tb("YLW  S-Lo")
        y_s_hi = self._tb("YLW  S-Hi")
        y_v_lo = self._tb("YLW  V-Lo")
        y_v_hi = self._tb("YLW  V-Hi")

        return {
            "red": {
                "lower1": np.array([self._tb("RED  H-Lo1 (0–10)"),   r_s_lo, r_v_lo]),
                "upper1": np.array([self._tb("RED  H-Hi1 (0–10)"),   r_s_hi, r_v_hi]),
                "lower2": np.array([self._tb("RED  H-Lo2 (160–180)"),r_s_lo, r_v_lo]),
                "upper2": np.array([self._tb("RED  H-Hi2 (160–180)"),r_s_hi, r_v_hi]),
            },
            "yellow": {
                "lower": np.array([self._tb("YLW  H-Lo"), y_s_lo, y_v_lo]),
                "upper": np.array([self._tb("YLW  H-Hi"), y_s_hi, y_v_hi]),
            },
        }

    def print_current_values(self) -> None:
        """Utility: print ranges to console (handy for finalising values)."""
        r = self.get_ranges()
        print("\n── Current HSV Ranges ────────────────────────")
        print(f"  RED  lower1 = {r['red']['lower1']}  upper1 = {r['red']['upper1']}")
        print(f"  RED  lower2 = {r['red']['lower2']}  upper2 = {r['red']['upper2']}")
        print(f"  YLW  lower  = {r['yellow']['lower']}  upper  = {r['yellow']['upper']}")
        print("──────────────────────────────────────────────\n")
