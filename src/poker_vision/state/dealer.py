"""Dealer-seat state machine (REQ-30).

`DealerSeatTracker` turns each frame's `FrameAssignments` (REQ-26/27,
already restricted to hysteresis-confirmed tracks -- REQ-24/25) into
`dealer_moved` events: the dealer seat is whichever seat the current
`dealer_button` track resolved to -- either a direct `player_area` hit or
`apply_dealer_nearest_seat_fallback`'s `dealer_area` resolution (REQ-27,
AC-16) -- and a `dealer_moved(from, to)` event fires only when that seat
actually changes from the last known one (AC-18).

The button disappearing from the frame -- not detected at all, or present
but left seat-less (`zone=DEALER_AREA, seat_id=None`, beyond REQ-27's
threshold) -- does not change the dealer seat: the last known seat is
carried forward silently, per REQ-30's "Verschwinden des Buttons ändert
den Dealer-Seat NICHT" (AC-18).

REQ-44's core-chain commit policy splits what used to be one mutating
`update()` into `compute_update()` (pure) and `commit()` (applies a
previously computed `DealerUpdate`, assigning `sequence`/`timestamp` at
that point). `update()` is kept as the two called back-to-back for
standalone callers; `PipelineStateMachine` uses the two steps separately
so this tracker's mutation can be deferred until the whole core chain has
succeeded for the frame (see `runner/loop.py`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import EVENT_SCHEMA_VERSION, DealerMovedEvent

_SEAT_RESOLVED_ZONE_KINDS = frozenset({ZoneKind.PLAYER_AREA, ZoneKind.DEALER_AREA})


@dataclass(frozen=True, slots=True)
class DealerUpdate:
    """Pure result of `DealerSeatTracker.compute_update()`.

    `dealer_seat` is the would-be new value (unchanged from the current
    one when nothing resolved, or resolved but not a change). `moved` is
    `(from_seat, to_seat)` when this frame is an actual seat change worth
    a `dealer_moved` event, `None` otherwise (no seat-resolved button, no
    change, or the first-ever resolution establishing a starting position).
    """

    dealer_seat: str | None
    moved: tuple[str, str] | None
    frame_index: int


class DealerSeatTracker:
    """Debounced-by-construction: only reacts to already-stable tracks.

    `seat_ids` is the full table seat universe (from the calibration used
    to produce the `FrameAssignments` this tracker is fed) -- used only to
    validate that an assignment's `seat_id` is a real seat, the same
    defensive check `SeatOccupancyTracker` makes (REQ-29). Unlike
    occupancy, there is no per-seat `False` starting state here: the
    dealer seat starts genuinely unknown (`None`), because unlike "seat
    empty", "no dealer seat" is not a real state the physical button is
    ever in -- it is only this tracker not having caught up to the
    button's already-existing position yet. So the *first* seat a button
    resolves to establishes that starting position silently, with no
    event; only a seat resolved *after* that counts as an actual
    "Seat-Wechsel" and emits `dealer_moved(from, to)` (AC-18 -- the
    "Seat 1 -> Seat 2" fixture emits exactly one event, not one per
    seat ever observed).

    `sequence` numbers emitted events with a private, monotonically
    increasing counter starting at 0, independent of
    `SeatOccupancyTracker`'s -- `PipelineStateMachine` (`machine.py`) is
    what composes sibling event sources under one shared, globally
    monotonic counter for REQ-33 (see `occupancy.py`).
    """

    def __init__(self, seat_ids: Iterable[str]) -> None:
        self._seat_ids = frozenset(seat_ids)
        self._dealer_seat: str | None = None
        self._sequence = 0

    def compute_update(self, frame_assignments: FrameAssignments) -> DealerUpdate:
        """Pure computation of this frame's dealer-seat resolution.

        Never mutates `self` -- see module docstring.
        """
        seat_id = self._resolve_candidate_seat(frame_assignments)
        frame_index = frame_assignments.frame_index
        if seat_id is None or seat_id == self._dealer_seat:
            # No seat-resolved button this frame, or the same seat as
            # before -- carries forward unchanged, no event (AC-18).
            return DealerUpdate(dealer_seat=self._dealer_seat, moved=None, frame_index=frame_index)

        previous_seat = self._dealer_seat
        if previous_seat is None:
            # First resolution ever: establishes the starting position,
            # not a "Seat-Wechsel" -- no event (AC-18).
            return DealerUpdate(dealer_seat=seat_id, moved=None, frame_index=frame_index)

        return DealerUpdate(
            dealer_seat=seat_id, moved=(previous_seat, seat_id), frame_index=frame_index
        )

    def commit(self, update: DealerUpdate, timestamp: datetime) -> list[DealerMovedEvent]:
        """Apply a previously computed `DealerUpdate` to `self`."""
        self._dealer_seat = update.dealer_seat
        if update.moved is None:
            return []
        previous_seat, seat_id = update.moved
        event = DealerMovedEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=self._next_sequence(),
            timestamp=timestamp,
            frame_index=update.frame_index,
            from_seat=previous_seat,
            to_seat=seat_id,
        )
        return [event]

    def update(self, frame_assignments: FrameAssignments) -> list[DealerMovedEvent]:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(frame_assignments), datetime.now(UTC))

    def _resolve_candidate_seat(self, frame_assignments: FrameAssignments) -> str | None:
        """Which seat the current dealer_button resolves to this frame -- read-only.

        Returns `None` when no seat-resolved button is present (carry
        forward, no error). Raises before touching `self._dealer_seat`, so
        this doubles as the validation half of `validate()`: either the
        frame resolves cleanly (or not at all) and the caller gets a clean
        result, or nothing about this tracker's state has changed yet.
        """
        candidates = [
            assignment
            for assignment in frame_assignments.assignments
            if assignment.object_class is DetectionClass.DEALER_BUTTON
            and assignment.zone in _SEAT_RESOLVED_ZONE_KINDS
            and assignment.seat_id is not None
        ]
        if len(candidates) > 1:
            raise ValueError(
                "FrameAssignments contains multiple seat-resolved dealer_button "
                f"assignments in one frame ({[a.seat_id for a in candidates]}) -- only "
                "one physical dealer button is expected at a time"
            )
        if not candidates:
            return None

        seat_id = candidates[0].seat_id
        assert seat_id is not None  # filtered above
        if seat_id not in self._seat_ids:
            raise ValueError(
                f"FrameAssignments references seat '{seat_id}', which is not in this "
                "tracker's seat universe -- constructed from a different calibration?"
            )
        return seat_id

    def validate(self, frame_assignments: FrameAssignments) -> None:
        """Raise if `frame_assignments` violates this tracker's invariants -- no mutation.

        Lets `PipelineStateMachine` (`machine.py`) check every tracker's
        invariants for a frame before mutating any of them, so a frame one
        sibling tracker rejects can't leave this one's state half-applied.
        """
        self._resolve_candidate_seat(frame_assignments)

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> str | None:
        """Current dealer seat, or `None` if no button has resolved yet.

        Feeds `StateSnapshot.dealer_seat` (REQ-33).
        """
        return self._dealer_seat
