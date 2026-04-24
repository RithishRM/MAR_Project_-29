"""
centroid_tracker.py
-------------------
Person B | Object Tracking & Stability Layer

Assigns persistent IDs to detected blobs (balls).
If a ball at (100,100) moves to (105,102), it remains the same ID.
Handles disappearances gracefully with a "deregister after N frames" policy.
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """
    Tracks objects across frames using Euclidean distance between centroids.

    Parameters
    ----------
    max_disappeared : int
        Number of consecutive frames an object can be missing before
        it is deregistered. Helps survive brief occlusions / flicker.
    max_distance : float
        Maximum pixel distance to consider two centroids the same object.
    """

    def __init__(self, max_disappeared: int = 10, max_distance: float = 60.0):
        self.next_object_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()   # id -> centroid (x, y)
        self.colors:  OrderedDict[int, str]        = OrderedDict()   # id -> "red" | "yellow"
        self.disappeared: OrderedDict[int, int]    = OrderedDict()   # id -> frame count missing

        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: np.ndarray, color: str) -> None:
        """Add a brand-new object with a fresh ID."""
        self.objects[self.next_object_id]     = centroid
        self.colors[self.next_object_id]      = color
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        """Remove an object that has been missing too long."""
        del self.objects[object_id]
        del self.colors[object_id]
        del self.disappeared[object_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[dict]) -> OrderedDict:
        """
        Feed in the latest detections from Person A and receive an up-to-date
        registry of tracked objects.

        Parameters
        ----------
        detections : list of dict
            Each dict must contain:
                "centroid" : (x, y)  – centre pixel of the detected blob
                "color"    : str     – "red" or "yellow"

        Returns
        -------
        OrderedDict  {object_id: {"centroid": (x, y), "color": str}}
        """

        # ── Case 1: No detections this frame ──────────────────────────
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._build_output()

        # ── Extract incoming centroids ────────────────────────────────
        input_centroids = np.array([d["centroid"] for d in detections], dtype="float")
        input_colors    = [d["color"] for d in detections]

        # ── Case 2: Nothing tracked yet → register everything ─────────
        if len(self.objects) == 0:
            for centroid, color in zip(input_centroids, input_colors):
                self._register(centroid, color)
            return self._build_output()

        # ── Case 3: Match existing objects to new detections ──────────
        object_ids       = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute pairwise Euclidean distances  (rows=existing, cols=new)
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

        # Greedy matching: sort by smallest distance first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            self.objects[oid]     = input_centroids[col]
            self.colors[oid]      = input_colors[col]   # update color in case mask changes
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Rows with no match → increment disappeared counter
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # New detections with no match → register
        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col], input_colors[col])

        return self._build_output()

    # ------------------------------------------------------------------

    def _build_output(self) -> OrderedDict:
        result = OrderedDict()
        for oid in self.objects:
            result[oid] = {
                "centroid": tuple(int(v) for v in self.objects[oid]),
                "color":    self.colors[oid],
            }
        return result

    def count_by_color(self) -> dict:
        """Returns {"red": N, "yellow": M} for the current frame."""
        counts = {"red": 0, "yellow": 0}
        for oid in self.colors:
            c = self.colors[oid]
            if c in counts:
                counts[c] += 1
        return counts
