"""`video_file` capture: deterministic frame sequence decoded from a video
file, no camera involved (REQ-13, REQ-15).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.capture.resolution import apply_resolution_cap
from poker_vision.capture.timestamps import replay_timestamp
from poker_vision.config import Resolution

# Used only if the container doesn't report a usable FPS; timestamps are
# still fully deterministic (derived from frame_index), just at an
# arbitrary, documented cadence.
_FALLBACK_FPS = 30.0


class VideoFileCapture(Capture):
    """Yields one `Frame` per decoded video frame, in decode order.

    Repeated decodes of the same file are deterministic (standard OpenCV
    guarantee for a static, non-live source), which is what REQ-15's
    "identical input -> identical sequence" relies on.
    """

    def __init__(
        self,
        path: str | Path,
        resolution_cap: Resolution,
        source_id: str | None = None,
    ) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"video file not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise ValueError(f"failed to open video file: {self._path}")
        reported_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = reported_fps if reported_fps and reported_fps > 0 else _FALLBACK_FPS
        self._resolution_cap = resolution_cap
        self._index = 0
        self.source_id = source_id or f"video_file:{self._path}"

    def __next__(self) -> Frame:
        ok, image = self._cap.read()
        if not ok:
            raise StopIteration
        image = apply_resolution_cap(image, self._resolution_cap)
        frame = Frame(
            image=image,
            timestamp=replay_timestamp(self._index, self._fps),
            frame_index=self._index,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
