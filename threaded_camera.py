"""
threaded_camera.py
------------------
Person B | Camera Threading Layer

cv2.VideoCapture.read() is blocking — it waits for the next frame.
This class runs the capture in a background thread so the main
processing loop always has the *latest* frame ready instantly.

Usage
-----
    cam = ThreadedCamera(src=0).start()
    frame = cam.read()
    ...
    cam.stop()
"""

import threading
import cv2


class ThreadedCamera:
    """
    Wraps cv2.VideoCapture in a daemon thread.

    The background thread continuously grabs frames and stores only
    the most recent one.  The main thread calls .read() to get it
    without ever blocking on I/O.
    """

    def __init__(self, src: int | str = 0, resolution: tuple[int, int] | None = None):
        """
        Parameters
        ----------
        src        : Camera index (int) or video file path (str).
        resolution : Optional (width, height) to set on the capture device.
        """
        self.cap = cv2.VideoCapture(src)

        if resolution is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Minimise internal buffer so we always get the *latest* frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret   = False
        self.frame = None
        self._lock   = threading.Lock()
        self._stopped = threading.Event()

    # ------------------------------------------------------------------

    def start(self) -> "ThreadedCamera":
        """Spawn the background reader thread and return self (for chaining)."""
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        """Internal: runs in background thread, grabs frames continuously."""
        while not self._stopped.is_set():
            ret, frame = self.cap.read()
            with self._lock:
                self.ret   = ret
                self.frame = frame

    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, any]:
        """
        Returns (ret, frame) — identical API to cap.read().
        Never blocks; returns the most recently captured frame.
        """
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self) -> None:
        """Signal the thread to stop and release the capture device."""
        self._stopped.set()
        self._thread.join(timeout=2.0)
        self.cap.release()

    # ------------------------------------------------------------------
    # Context-manager support  (with ThreadedCamera(0) as cam: ...)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def get_resolution(self) -> tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
