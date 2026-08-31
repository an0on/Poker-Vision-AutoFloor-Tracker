"""`LatestFrameHub`: the thread-safe single-slot bridge between the pipeline
loop's own thread and the debug MJPEG server's per-client streams (REQ-46).

Per CLAUDE.md's "capture <-> debug: FrameHub" section: `publish(frame,
context_snapshot)` is called by the loop (`runner/loop.py`), once per
successfully processed frame -- never for a frame the core chain
discarded (see that module's error policy). `get_latest()` is called by
`debug.mjpeg.MjpegDebugServer`, once per connected client. Deliberately
carries no rendering or FastAPI knowledge of its own: `publish()` only
ever swaps two references and bumps a version counter under a
briefly-held lock, so it can never be slow ("kein Lock über
Rendering-Dauer") and never blocks the loop on how long overlay
rendering or JPEG encoding takes. That work happens entirely in
`debug.mjpeg`'s `_stream()`, on demand, only for a connected client --
this module has no `render_overlay` import at all, so "ohne verbundenen
Client findet kein Rendering statt" (REQ-46) holds by construction, not
by a runtime check.

This is the same single-slot, latest-wins, versioned-buffer pattern
`capture.continuity`'s `_LatestFrameBuffer` uses for the opposite
direction (camera thread -> loop thread) -- see that module's docstring,
which names this class as the one it followed. The one structural
difference: `_LatestFrameBuffer` has exactly one consumer (`__next__()`),
so it can track a single shared "delivered version" internally.
`LatestFrameHub` can have an arbitrary number of independently-polling
MJPEG clients, so each caller of `get_latest()` tracks and passes its own
`since_version` (the version it last received) instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from poker_vision.assignment.models import FrameAssignments
from poker_vision.capture.frame import Frame
from poker_vision.state.snapshot import StateSnapshot
from poker_vision.tracking.models import TrackedFrame


@dataclass(slots=True, frozen=True)
class DebugSnapshot:
    """Everything `debug.overlay.render_overlay` needs besides the raw frame
    and calibration, bundled so a single `LatestFrameHub` slot carries one
    frame's tracks/assignments/state consistently -- the loop fills all
    three from the same `FrameContext` it just finished committing, so
    they're always in sync with each other and with the published frame,
    never mixed with an earlier or later frame's data.
    """

    tracked_frame: TrackedFrame
    frame_assignments: FrameAssignments
    state_snapshot: StateSnapshot


class LatestFrameHub:
    """Thread-safe single-slot, latest-wins hub with a version counter.

    `publish()` always overwrites the slot -- there is never a queued
    backlog to drain, so a debug client that's behind always renders the
    newest frame available, never catches up frame-by-frame through stale
    ones. `get_latest()` blocks up to `timeout` seconds for a version
    strictly newer than `since_version` (a short blocking wait, not a
    busy-spin or a bare sleep loop -- the same `threading.Condition.
    wait_for` pattern `capture.continuity` uses), and returns `None` on a
    plain timeout; the caller is expected to call again, not treat that as
    exhaustion or failure.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._snapshot: DebugSnapshot | None = None
        self._version = 0

    def publish(self, frame: Frame, snapshot: DebugSnapshot) -> None:
        with self._condition:
            self._frame = frame
            self._snapshot = snapshot
            self._version += 1
            self._condition.notify_all()

    def get_latest(
        self, since_version: int = 0, timeout: float | None = None
    ) -> tuple[Frame, DebugSnapshot, int] | None:
        with self._condition:
            got_newer = self._condition.wait_for(
                lambda: self._version > since_version, timeout=timeout
            )
            if not got_newer:
                return None
            # `got_newer` is only true once `publish()` has run at least
            # once (it's what first makes `_version > since_version >= 0`
            # possible), so both are always set here.
            assert self._frame is not None
            assert self._snapshot is not None
            return self._frame, self._snapshot, self._version
