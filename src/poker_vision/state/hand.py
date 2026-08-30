"""Hand-lifecycle state machine (REQ-32).

`HandTracker` turns each frame's `FrameAssignments` (REQ-26/27, already
restricted to hysteresis-confirmed tracks -- REQ-24/25) into
`hand_started`/`hand_ended` events: a hand begins the moment the board goes
from stably empty to non-empty (at least one `card` track in `board_zone`),
and ends the moment it goes back to stably empty (AGENTS.md's "Board leer
-> nicht leer -> leer"). Rank/count of the cards don't matter here -- only
whether the board is empty or not; turning a count into a street is
`StreetTracker`'s job (REQ-31), not this one's.

`StreetTracker` reacts to the exact same board-empty <-> non-empty signal
for its own `hand_id` numbering (see `street.py`'s docstring). Both
trackers share the one piece of detection logic behind that signal --
counting `card` tracks in `board_zone` -- via `count_board_cards`
(`board.py`), so it is computed once and can't drift between the two. Each
tracker still owns its own private `hand_id` counter, though: this module
is the canonical source of hand boundaries per REQ-32.
`PipelineStateMachine` (`machine.py`) wires that up as the literal, shared
value on `StreetChangedEvent`, overwriting `StreetTracker`'s own private
counter with this tracker's `hand_id` for REQ-33, on the strength of the
fact that both trackers apply the identical rule to the same input stream
and so already agree numerically on a single-table replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from poker_vision.assignment.models import FrameAssignments
from poker_vision.state.board import count_board_cards
from poker_vision.state.events import EVENT_SCHEMA_VERSION, HandEndedEvent, HandStartedEvent

HandLifecycleEvent = HandStartedEvent | HandEndedEvent


@dataclass(frozen=True, slots=True)
class HandUpdate:
    """Pure result of `HandTracker.compute_update()`.

    `transition` is `"started"`/`"ended"` when this frame crosses a hand
    boundary, `None` otherwise. REQ-44's core-chain commit policy splits
    what used to be one mutating `update()` into `compute_update()` (pure)
    and `commit()`; `update()` is kept as the two called back-to-back for
    standalone callers, `PipelineStateMachine` uses the two steps
    separately (see `runner/loop.py`).
    """

    board_active: bool
    hand_id: int | None
    next_hand_id: int
    transition: Literal["started", "ended"] | None
    frame_index: int


class HandTracker:
    """Debounced-by-construction: only reacts to already-stable tracks.

    Like `StreetTracker`, there is no seat universe to validate against --
    `board_zone` is the one global zone this tracker inspects.

    `hand_id` starts at 1 and is assigned the moment a hand starts (the
    empty -> non-empty transition); the same id is carried on that hand's
    `hand_ended` event, and only bumps for the *next* hand's
    `hand_started` (AC-20's "zweite Hand erhält hand_id + 1"). Before the
    first hand ever starts, no id has been assigned yet.

    `sequence` numbers emitted events with a private, monotonically
    increasing counter starting at 0, independent of the other trackers'
    -- `PipelineStateMachine` (`machine.py`) composes sibling event sources
    under one shared, globally monotonic counter for REQ-33 (see
    `occupancy.py`).
    """

    def __init__(self) -> None:
        self._board_active = False
        self._hand_id: int | None = None
        self._next_hand_id = 1
        self._sequence = 0

    def compute_update(self, frame_assignments: FrameAssignments) -> HandUpdate:
        """Pure computation of this frame's hand-lifecycle transition.

        Never mutates `self` -- see module docstring.
        """
        count = count_board_cards(frame_assignments)
        frame_index = frame_assignments.frame_index

        if count > 0 and not self._board_active:
            return HandUpdate(
                board_active=True,
                hand_id=self._next_hand_id,
                next_hand_id=self._next_hand_id + 1,
                transition="started",
                frame_index=frame_index,
            )

        if count == 0 and self._board_active:
            assert self._hand_id is not None  # set when board_active was set True
            return HandUpdate(
                board_active=False,
                hand_id=self._hand_id,
                next_hand_id=self._next_hand_id,
                transition="ended",
                frame_index=frame_index,
            )

        return HandUpdate(
            board_active=self._board_active,
            hand_id=self._hand_id,
            next_hand_id=self._next_hand_id,
            transition=None,
            frame_index=frame_index,
        )

    def commit(self, update: HandUpdate, timestamp: datetime) -> list[HandLifecycleEvent]:
        """Apply a previously computed `HandUpdate` to `self`."""
        self._board_active = update.board_active
        self._hand_id = update.hand_id
        self._next_hand_id = update.next_hand_id
        if update.transition is None:
            return []
        assert update.hand_id is not None  # set on every transition
        event_cls = HandStartedEvent if update.transition == "started" else HandEndedEvent
        event: HandLifecycleEvent = event_cls(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=self._next_sequence(),
            timestamp=timestamp,
            frame_index=update.frame_index,
            hand_id=update.hand_id,
        )
        return [event]

    def update(self, frame_assignments: FrameAssignments) -> list[HandLifecycleEvent]:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(frame_assignments), datetime.now(UTC))

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> tuple[int | None, bool]:
        """`(hand_id, hand_active)` for `StateSnapshot` (REQ-33).

        `hand_id` is the current hand's id while active, the just-ended
        hand's id right after `hand_ended`, or `None` if no hand has ever
        started yet.
        """
        return self._hand_id, self._board_active
