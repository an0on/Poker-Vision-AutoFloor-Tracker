"""`continuity` capture: live frames from the iPhone Continuity Camera via
AVFoundation, selected by device index (REQ-13, REQ-16).

A missing/unopenable camera is a hard error (`RuntimeError`), never a
silent fallback to another source. `capture_factory` is injectable so tests
can exercise this class's error handling and frame plumbing without any
Continuity hardware or macOS-specific backend present.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

import cv2

from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.capture.resolution import apply_resolution_cap
from poker_vision.config import Resolution


class VideoCaptureLike(Protocol):
    """The subset of `cv2.VideoCapture`'s interface this module relies on."""

    def isOpened(self) -> bool: ...  # noqa: N802 (matches cv2's API)

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


def _default_capture_factory(device_index: int) -> VideoCaptureLike:
    return cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)


class ContinuityCapture(Capture):
    """Yields live frames from the camera at `device_index` until closed."""

    def __init__(
        self,
        device_index: int,
        resolution_cap: Resolution,
        source_id: str | None = None,
        capture_factory: Callable[[int], VideoCaptureLike] = _default_capture_factory,
    ) -> None:
        self._cap = capture_factory(device_index)
        if not self._cap.isOpened():
            # Construction never completes, so the caller never gets an
            # object to call close() on — release here or the native
            # AVFoundation handle leaks and can block a later retry.
            self._cap.release()
            raise RuntimeError(
                f"continuity camera not available at device index {device_index} "
                "(no fallback to another source, see REQ-16)"
            )
        self._resolution_cap = resolution_cap
        self._index = 0
        self.source_id = source_id or f"continuity:{device_index}"

    def __next__(self) -> Frame:
        ok, image = self._cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame from continuity camera ({self.source_id})")
        image = apply_resolution_cap(image, self._resolution_cap)
        frame = Frame(
            image=image,
            timestamp=datetime.now(UTC),
            frame_index=self._index,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()
