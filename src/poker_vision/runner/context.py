"""`FrameContext`: the runner's own per-frame bookkeeping (REQ-44).

Created internally by `runner/loop.py`, once per frame, and progressively
filled in as each stage of the core chain (detection -> tracking ->
assignment -> state) succeeds. Mirrors the CLAUDE.md architecture note:
"FrameContext ... wird vom Loop pro Frame intern erzeugt und nach jeder
Stufe fortgeschrieben: frame_id, timestamp, Raw-Frame, Detections, Tracks,
Zonen-Zuordnung, State-Snapshot, Stufen-Fehlerliste."

A plain runtime dataclass, not a `StrictModel` (REQ-4): like `capture.
frame.Frame`, it carries a raw `numpy.ndarray` image buffer and holds
other stages' own schema objects by reference -- it is the loop's internal
bookkeeping, not wire/on-disk schema data in its own right. No pipeline
stage imports or receives this type (see `runner/loop.py`'s module
docstring): the loop calls every stage with that stage's own existing,
typed signature and records the result here itself, preserving the
`runner -> Stufen`-only dependency direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from poker_vision.assignment.models import FrameAssignments
from poker_vision.capture.frame import Frame
from poker_vision.detection.models import FrameDetections
from poker_vision.state.events import Event
from poker_vision.state.snapshot import StateSnapshot
from poker_vision.tracking.models import TrackedFrame


@dataclass(slots=True)
class FrameContext:
    """One frame's progress through the core chain, plus its outcome.

    `detections`/`tracks`/`assignments` are `None` until the stage that
    produces them has actually run. On a frame the core chain ends up
    discarding, every field through the stage that raised is still filled
    in (useful for diagnosing *where* it failed); the stage that raised,
    and every stage after it, leave their fields at the default. This is
    purely observability -- the loop's actual "no partial update"
    guarantee (REQ-44) is that `tracker`/`hysteresis`/`state_machine`
    themselves are never `commit()`-ed for such a frame, which these
    field values have no bearing on either way. `events` is empty both
    before the state stage runs and when it produces no events for this
    frame -- the two are not distinguished here (the caller already knows
    which, from whether `errors` is empty). `errors` holds one entry per
    core-chain exception that discarded this frame (in practice at most
    one, since the loop stops the chain at the first failure, but the
    field stays a list rather than an `Exception | None` so it reads the
    same as every other "the stage hasn't run" default -- empty).
    """

    frame_id: int
    timestamp: datetime
    raw_frame: Frame
    detections: FrameDetections | None = None
    tracks: TrackedFrame | None = None
    assignments: FrameAssignments | None = None
    events: list[Event] = field(default_factory=list)
    state_snapshot: StateSnapshot | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Whether the full core chain (detection -> ... -> state) ran to completion."""
        return not self.errors
