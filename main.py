"""
main.py
-------
Person B | Main Processing Pipeline
Project: #29 – Red/Green Ball Detector

This is the "main loop" that ties together:
  • ThreadedCamera     — lag-free frame capture
  • Person A's module  — get_mask(), detect_balls()
  • CentroidTracker    — persistent IDs across frames
  • HUDRenderer        — on-screen annotations & FPS
  • HSVControlPanel    — live-tuning trackbars

Controls
--------
  Q / ESC  → quit
  P        → print current HSV values to console
  S        → save a screenshot to disk
  D        → toggle debug view (show raw masks)

Hand-off contract (what Person A must provide)
-----------------------------------------------
  from ball_detector import get_mask, detect_balls

  get_mask(frame, color, hsv_ranges) -> binary mask (np.ndarray)
  detect_balls(mask)                 -> list of {"centroid": (x,y), "radius": int}
"""

import cv2
import sys
import time
import datetime
import numpy as np

# ── Person B modules ──────────────────────────────────────────────────────────
from threaded_camera   import ThreadedCamera
from centroid_tracker  import CentroidTracker
from hud               import HUDRenderer
from hsv_control_panel import HSVControlPanel

# ── Person A module (stub provided below if not yet delivered) ────────────────
try:
    from ball_detector import get_mask, detect_balls
    PERSON_A_READY = True
except ImportError:
    PERSON_A_READY = False
    print("[WARN] ball_detector.py not found — using built-in stub.")


# ── Stub (so Person B can run & test the pipeline independently) ──────────────

def _stub_get_mask(frame: np.ndarray, color: str, hsv_ranges: dict) -> np.ndarray:
    """
    Minimal HSV mask.  Replace with Person A's version when delivered.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        r = hsv_ranges["red"]
        m1 = cv2.inRange(hsv, r["lower1"], r["upper1"])
        m2 = cv2.inRange(hsv, r["lower2"], r["upper2"])
        mask = cv2.bitwise_or(m1, m2)
    else:
        g = hsv_ranges["green"]
        mask = cv2.inRange(hsv, g["lower"], g["upper"])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _stub_detect_balls(mask: np.ndarray) -> list[dict]:
    """
    Minimal blob detector.  Replace with Person A's version when delivered.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:          # noise filter — tune for your ball size
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        results.append({"centroid": (int(cx), int(cy)), "radius": int(radius)})
    return results


if not PERSON_A_READY:
    get_mask     = _stub_get_mask
    detect_balls = _stub_detect_balls


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(camera_src: int | str = 0) -> None:
    """
    Entry point.  camera_src can be:
      0, 1, 2 …  → webcam index
      "video.mp4" → recorded file (great for offline testing)
    """

    # ── Initialise subsystems ─────────────────────────────────────────
    cam     = ThreadedCamera(src=camera_src, resolution=(1280, 720)).start()
    tracker = CentroidTracker(max_disappeared=12, max_distance=70)
    hud     = HUDRenderer(circle_thickness=2, label_font_scale=0.55)
    panel   = HSVControlPanel()
    panel.create()

    # Wait for camera to warm up
    time.sleep(0.3)

    debug_mode   = False
    COLORS       = ["red", "green"]
    OUTPUT_WIN   = "Ball Tracker — Press Q to quit"

    cv2.namedWindow(OUTPUT_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OUTPUT_WIN, 1280, 720)

    print("\n[INFO] Pipeline running.")
    print("       Q/ESC = quit | P = print HSV | S = screenshot | D = debug masks\n")

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        ret, frame = cam.read()

        if not ret or frame is None:
            print("[WARN] No frame received — waiting…")
            time.sleep(0.05)
            continue

        # Read current HSV ranges from trackbars
        hsv_ranges = panel.get_ranges()

        # ── Detection (Person A contract) ─────────────────────────────
        detections: list[dict] = []
        radii_map:  dict[int, int] = {}          # will be filled after tracking
        masks: dict[str, np.ndarray] = {}

        for color in COLORS:
            mask  = get_mask(frame, color, hsv_ranges)
            balls = detect_balls(mask)
            masks[color] = mask

            for b in balls:
                detections.append({
                    "centroid": b["centroid"],
                    "color":    color,
                })

        # ── Tracking (Person B) ───────────────────────────────────────
        tracked = tracker.update(detections)

        # Build radii_map: match tracked IDs back to detected radii
        for oid, info in tracked.items():
            cx, cy = info["centroid"]
            color  = info["color"]
            # find nearest detection of same colour for its radius
            best_r = 20
            best_d = float("inf")
            mask   = get_mask(frame, color, hsv_ranges)
            balls  = detect_balls(mask)
            for b in balls:
                bx, by = b["centroid"]
                d = ((cx - bx)**2 + (cy - by)**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_r = b.get("radius", 20)
            radii_map[oid] = best_r

        counts = tracker.count_by_color()

        # ── HUD ───────────────────────────────────────────────────────
        display = frame.copy()
        fps     = hud.tick_and_draw(
            display, tracked, counts, radii_map,
            extra_lines=["P=HSV  S=screenshot", "D=debug masks"]
        )

        # ── Debug view: show masks side-by-side ───────────────────────
        if debug_mode:
            h, w = frame.shape[:2]
            debug_h   = h // 3
            debug_w   = w // 3
            combined  = np.zeros((debug_h, debug_w * 2, 3), dtype=np.uint8)

            r_viz = cv2.resize(masks["red"],   (debug_w, debug_h))
            g_viz = cv2.resize(masks["green"], (debug_w, debug_h))

            combined[:, :debug_w]         = cv2.cvtColor(r_viz, cv2.COLOR_GRAY2BGR)
            combined[:, debug_w:debug_w*2] = cv2.cvtColor(g_viz, cv2.COLOR_GRAY2BGR)

            cv2.putText(combined, "RED mask",   (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,30,220), 1)
            cv2.putText(combined, "GREEN mask", (debug_w+10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,60), 1)

            # Overlay debug strip at bottom of display
            display[h - debug_h:, :debug_w * 2] = combined

        cv2.imshow(OUTPUT_WIN, display)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):          # Q or ESC
            print("[INFO] Quit requested.")
            break

        elif key == ord('p'):              # Print HSV values
            panel.print_current_values()

        elif key == ord('s'):              # Screenshot
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.png"
            cv2.imwrite(path, display)
            print(f"[INFO] Screenshot saved: {path}")

        elif key == ord('d'):              # Toggle debug masks
            debug_mode = not debug_mode
            print(f"[INFO] Debug mode: {'ON' if debug_mode else 'OFF'}")

    # ── Cleanup ───────────────────────────────────────────────────────
    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # Pass a filename string to test with a recorded video, e.g.:
    #   python main.py test_video.mp4
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        src = sys.argv[1]
    run(camera_src=src)
