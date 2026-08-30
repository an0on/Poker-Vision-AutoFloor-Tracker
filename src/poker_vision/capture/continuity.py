"""`continuity` capture: live frames from the iPhone Continuity Camera via
AVFoundation, selected by device index (REQ-13, REQ-16).

A missing/unopenable camera is a hard error (`RuntimeError`), never a
silent fallback to another source. `capture_factory` is injectable so tests
can exercise this class's error handling and frame plumbing without any
Continuity hardware or macOS-specific backend present.

REQ-44's backpressure/pacing requirement applies only to this source (not
`video_file`/`image_dir`, which stay fully synchronous and deterministic):
a background thread reads continuously from the camera and publishes each
frame into `_LatestFrameBuffer`, a thread-safe single-slot, latest-wins
buffer carrying a monotonically increasing version counter -- the same
pattern REQ-46's `LatestFrameHub` will later use for the debug MJPEG
stream, applied here to the capture side instead. `__next__()` blocks on
that buffer with a short timeout (`threading.Condition.wait_for`, never a
busy-spin or a bare sleep loop) instead of calling `read()` directly, so:

- a frame loop slower than the camera never builds a backlog -- it always
  gets the freshest frame, with older ones silently dropped by the
  background thread simply overwriting the slot before they're ever read;
- a frame loop faster than the camera never receives the same frame
  twice -- `get_latest()` only returns once a version newer than the last
  one it delivered exists, blocking (with periodic timeout wakeups, not
  forever) until then.

`frame_index` is assigned in the background thread, once per frame
actually read from the camera -- not once per frame delivered to the
loop. A frame loop slower than the camera therefore sees gaps in
`frame_index` across dropped frames, which is exactly what REQ-24's
`HysteresisFilter` already expects and accounts for (see its own
docstring: a `frame_index` jump between two calls means that many frames
elapsed with no data at all).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

import cv2

from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.capture.resolution import apply_resolution_cap
from poker_vision.config import Resolution

# How long `__next__()` blocks per wait_for() call before checking again --
# a "kurzer blockierender Wait mit Timeout", not a value the caller ever
# observes directly (a timeout just means "no new frame yet, keep
# waiting"), so it only trades off shutdown responsiveness against wakeup
# frequency.
_DEFAULT_WAIT_TIMEOUT_SECONDS = 0.5


class VideoCaptureLike(Protocol):
    """The subset of `cv2.VideoCapture`'s interface this module relies on."""

    def isOpened(self) -> bool: ...  # noqa: N802 (matches cv2's API)

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


def _default_capture_factory(device_index: int) -> VideoCaptureLike:
    return cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)


class _LatestFrameBuffer:
    """Thread-safe single-slot, latest-wins buffer with a version counter.

    One producer (the background reader thread) calls `publish()`/
    `publish_error()`; one consumer (`__next__()`, called from the
    pipeline's own frame-loop thread) calls `get_latest()`. `publish()`
    always overwrites the slot -- there is never a backlog to drain.
    `get_latest()` only ever returns a given version once.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._version = 0
        self._delivered_version = 0
        self._error: BaseException | None = None
        self._closed = False

    def publish(self, frame: Frame) -> None:
        with self._condition:
            if self._closed:
                return
            self._frame = frame
            self._version += 1
            self._condition.notify_all()

    def publish_error(self, error: BaseException) -> None:
        with self._condition:
            if self._closed:
                return
            self._error = error
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def get_latest(self, timeout: float) -> Frame | None:
        """Block up to `timeout` seconds for a frame newer than the last one
        delivered. Returns `None` on a plain timeout -- the caller is
        expected to call again, not treat that as exhaustion or failure.

        Raises whatever the background thread reported via
        `publish_error()` (a real camera read failure -- REQ-16's "kein
        Fallback") -- but only once every already-published, not-yet-
        delivered frame has been returned first: a slow consumer that
        catches up right as the camera fails must still see the last real
        frame before the error, not have it swallowed. Raises
        `RuntimeError` once the buffer has been closed and every
        already-published frame has been delivered.
        """
        with self._condition:
            got_something = self._condition.wait_for(
                lambda: self._version != self._delivered_version
                or self._error is not None
                or self._closed,
                timeout=timeout,
            )
            if not got_something:
                return None
            if self._version != self._delivered_version:
                self._delivered_version = self._version
                return self._frame
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            # Only reachable via `_closed` becoming true with no new frame
            # and no error pending -- the capture was closed while a
            # `__next__()` call was still waiting.
            raise RuntimeError("continuity capture closed while waiting for a frame")


class ContinuityCapture(Capture):
    """Yields live frames from the camera at `device_index` until closed."""

    def __init__(
        self,
        device_index: int,
        resolution_cap: Resolution,
        source_id: str | None = None,
        capture_factory: Callable[[int], VideoCaptureLike] = _default_capture_factory,
        wait_timeout: float = _DEFAULT_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        self._cap = capture_factory(device_index)
        if not self._cap.isOpened():
            # Construction never completes, so the caller never gets an
            # object to call close() on -- release here or the native
            # AVFoundation handle leaks and can block a later retry.
            self._cap.release()
            raise RuntimeError(
                f"continuity camera not available at device index {device_index} "
                "(no fallback to another source, see REQ-16)"
            )
        self._resolution_cap = resolution_cap
        self.source_id = source_id or f"continuity:{device_index}"
        self._wait_timeout = wait_timeout

        self._buffer = _LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name=f"continuity-reader-{self.source_id}",
            daemon=True,
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        # Runs entirely on the background thread: reads as fast as the
        # camera/backend allows and always overwrites the buffer's single
        # slot -- the frame loop (a separate thread, via __next__()) only
        # ever sees the latest one, never a backlog. `frame_index` is
        # assigned here, once per frame actually read (see module
        # docstring) -- not once per frame the loop ends up consuming.
        index = 0
        try:
            while not self._stop_event.is_set():
                ok, image = self._cap.read()
                if self._stop_event.is_set():
                    # close() ran while this read() was in flight -- don't
                    # publish (the buffer is closing/closed already) and
                    # don't treat a stop-induced read failure as a real
                    # camera error.
                    return
                if not ok:
                    self._buffer.publish_error(
                        RuntimeError(
                            f"failed to read frame from continuity camera ({self.source_id})"
                        )
                    )
                    return
                image = apply_resolution_cap(image, self._resolution_cap)
                frame = Frame(
                    image=image,
                    timestamp=datetime.now(UTC),
                    frame_index=index,
                    source_id=self.source_id,
                )
                index += 1
                self._buffer.publish(frame)
        except Exception as exc:  # noqa: BLE001 -- surfaced to __next__(), not swallowed
            if not self._stop_event.is_set():
                self._buffer.publish_error(exc)

    def __next__(self) -> Frame:
        while True:
            frame = self._buffer.get_latest(self._wait_timeout)
            if frame is not None:
                return frame

    def close(self) -> None:
        # Signal the reader thread to stop and unblock any waiting
        # __next__()/get_latest() call, then release() the underlying
        # capture *before* joining: release() unblocks a background
        # read() that's currently in flight (the stop_event alone can't --
        # nothing else interrupts a blocking read()), and the reader
        # thread's own stop_event check right after read() returning
        # keeps that release-induced failure from being misreported as a
        # real camera error.
        self._stop_event.set()
        self._buffer.close()
        self._cap.release()
        self._reader_thread.join(timeout=5.0)
