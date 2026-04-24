"""
ball_detector.py
----------------
Person A | Vision Engineering Layer
Project: #29 – Red/Yellow Ball Detector

Covers:
  1. HSV color space optimisation (with dual-range red handling)
  2. Gaussian blur + morphological noise reduction
  3. Contour filtering by area, circularity, and minEnclosingCircle
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, blur_ksize: int = 11) -> np.ndarray:
    """
    Convert to HSV after applying a Gaussian blur.

    Gaussian blur smooths out high-frequency noise (dust, specular highlights)
    so the HSV threshold does not pick up tiny false positives.

    Parameters
    ----------
    frame      : raw BGR frame from the camera
    blur_ksize : kernel size for GaussianBlur (must be odd).
                 Larger = more smoothing but softer edges.
                 11 is a good default for 720p.

    Returns
    -------
    HSV image (same shape as frame)
    """
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), sigmaX=0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MASKING  (HSV thresholding + morphological cleanup)
# ─────────────────────────────────────────────────────────────────────────────

def get_mask(frame: np.ndarray, color: str, hsv_ranges: dict) -> np.ndarray:
    """
    Build a clean binary mask for the requested color.

    Why HSV instead of RGB?
    -----------------------
    RGB values change dramatically with lighting. A red ball in shadow
    looks almost black in RGB. In HSV, the Hue channel stays stable;
    only Value (brightness) shifts, which is easy to handle with a wide
    V range.

    Red wraps around 0/180:
    -----------------------
    OpenCV HSV hue goes 0-180. Red sits at both ends (0-10 AND 160-180),
    so we threshold twice and OR the results.

    Morphological pipeline:
    -----------------------
      OPEN  (erode then dilate)  kills small noise blobs
      CLOSE (dilate then erode)  fills holes inside the ball mask
      DILATE x1                  slightly expands edges to catch ball border

    Parameters
    ----------
    frame      : raw BGR frame
    color      : "red" or "yellow"
    hsv_ranges : dict from HSVControlPanel.get_ranges()

    Returns
    -------
    Binary mask uint8 — 255 where the color is present, 0 elsewhere.
    """
    hsv = preprocess(frame)

    # ── Threshold ─────────────────────────────────────────────────────
    if color == "red":
        r    = hsv_ranges["red"]
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, r["lower1"], r["upper1"]),  # hue  0-10
            cv2.inRange(hsv, r["lower2"], r["upper2"]),  # hue 160-180
        )
    elif color == "yellow":
        y    = hsv_ranges["yellow"]
        mask = cv2.inRange(hsv, y["lower"], y["upper"])
    else:
        raise ValueError(f"Unsupported color: {color!r}. Use 'red' or 'yellow'.")

    # ── Morphological cleanup ──────────────────────────────────────────
    # Elliptical kernel — better than square for round objects
    k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_small,  iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_medium, iterations=2)
    mask = cv2.dilate(mask, k_small, iterations=1)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONTOUR DETECTION & SHAPE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

# Tune these for your specific balls and camera distance
MIN_AREA        = 500     # px2 — ignore anything smaller (noise)
MAX_AREA        = 80_000  # px2 — ignore anything larger (merged blobs / reflections)
MIN_CIRCULARITY = 0.60    # 0.0 = any shape, 1.0 = perfect circle
                          # 0.60 handles slightly squashed/occluded balls


def _circularity(contour) -> float:
    """
    Computes  4*pi*Area / Perimeter^2
    Perfect circle -> 1.0
    Anything non-circular -> approaches 0
    """
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def detect_balls(mask: np.ndarray) -> list:
    """
    Find individual balls in a binary mask.

    Filtering pipeline
    ------------------
    1. Find all external contours
    2. Reject contours whose area is outside [MIN_AREA, MAX_AREA]
    3. Reject contours whose circularity < MIN_CIRCULARITY
    4. For survivors, compute minEnclosingCircle -> centroid + radius

    Parameters
    ----------
    mask : binary mask from get_mask()

    Returns
    -------
    list of {"centroid": (x: int, y: int), "radius": int, "area": float}
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ── Area filter ────────────────────────────────────────────────
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        # ── Circularity filter ─────────────────────────────────────────
        if _circularity(cnt) < MIN_CIRCULARITY:
            continue

        # ── Bounding circle ────────────────────────────────────────────
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        results.append({
            "centroid": (int(cx), int(cy)),
            "radius":   int(radius),
            "area":     round(area, 1),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALIBRATION HELPER  (run standalone to find your HSV values)
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(camera_src: int = 0) -> None:
    """
    Standalone calibration tool.

    Opens a trackbar window showing original + mask side by side.
    Adjust sliders until only your ball is white in the mask,
    then press P to print the values.

    Usage:  python ball_detector.py
            python ball_detector.py 1   (for /dev/video1)
    """
    cap = cv2.VideoCapture(camera_src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    win = "Calibration  |  ESC = quit  |  P = print values"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1000, 420)

    cv2.createTrackbar("H Lo",  win,   0, 180, lambda _: None)
    cv2.createTrackbar("H Hi",  win,  30, 180, lambda _: None)
    cv2.createTrackbar("S Lo",  win,  80, 255, lambda _: None)
    cv2.createTrackbar("S Hi",  win, 255, 255, lambda _: None)
    cv2.createTrackbar("V Lo",  win,  50, 255, lambda _: None)
    cv2.createTrackbar("V Hi",  win, 255, 255, lambda _: None)
    cv2.createTrackbar("Blur",  win,  11,  31, lambda _: None)

    def tb(name): return cv2.getTrackbarPos(name, win)

    print("\n[Calibration] Adjust sliders until only your ball is white in the mask.")
    print("              P = print values,  ESC = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        blur_k = tb("Blur") | 1   # ensure odd
        lower  = np.array([tb("H Lo"), tb("S Lo"), tb("V Lo")])
        upper  = np.array([tb("H Hi"), tb("S Hi"), tb("V Hi")])

        blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = cv2.inRange(hsv, lower, upper)

        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display  = np.hstack([
            cv2.resize(frame,    (480, 360)),
            cv2.resize(mask_bgr, (480, 360)),
        ])
        cv2.putText(display, "Original",   (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Mask",       (490, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('p'):
            print(f"  lower = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
            print(f"  upper = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
            print(f"  blur kernel size = {blur_k}\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_calibration(src)
