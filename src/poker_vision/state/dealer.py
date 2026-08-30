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
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import EVENT_SCHEMA_VERSION, DealerMovedEvent

_SEAT_RESOLVED_ZONE_KINDS = frozenset({ZoneKind.PLAYER_AREA, ZoneKind.DEALER_AREA})


class DealerSeatTracker:
    """Debounced-by-construction: only reacts to already-stable tracks.

    `seat_ids` is the full table seat universe (from the calibration used
    to produce the `FrameAssignments` this tracker is fed) -- used only to
    validate that an assignment's `seat_id` is a real seat, the same
    defensive check `SeatOccupancyTracker` makes (REQ-29). Unlike
    occupancy, there is no per-seat `False` starting state here: the
    dealer seat starts genuinely unknown (`None`), since no button has
    been observed yet, and the first seat a button resolves to is itself
    a "Seat-Wechsel" away from that unknown state --
    `dealer_moved(None, seat)`.

    `sequence` numbers emitted events with a private, monotonically
    increasing counter starting at 0, independent of
    `SeatOccupancyTracker`'s -- composing sibling event sources under one
    shared, globally monotonic counter is later work's job, not this
    one's (see `occupancy.py`).
    """

    def __init__(self, seat_ids: Iterable[str]) -> None:
        self._seat_ids = frozenset(seat_ids)
        self._dealer_seat: str | None = None
        self._sequence = 0

    def update(self, frame_assignments: FrameAssignments) -> list[DealerMovedEvent]:
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
            # No seat-resolved button this frame -- absent entirely, or
            # present but still seat-less (REQ-27's threshold missed).
            # Either way the last known dealer seat carries forward
            # unchanged, no event (AC-18).
            return []

        seat_id = candidates[0].seat_id
        assert seat_id is not None  # filtered above
        if seat_id not in self._seat_ids:
            raise ValueError(
                f"FrameAssignments references seat '{seat_id}', which is not in this "
                "tracker's seat universe -- constructed from a different calibration?"
            )
        if seat_id == self._dealer_seat:
            return []

        event = DealerMovedEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=self._next_sequence(),
            timestamp=datetime.now(UTC),
            frame_index=frame_assignments.frame_index,
            from_seat=self._dealer_seat,
            to_seat=seat_id,
        )
        self._dealer_seat = seat_id
        return [event]

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> str | None:
        """Current dealer seat, or `None` if no button has resolved yet.

        Feeds `StateSnapshot.dealer_seat` (REQ-33).
        """
        return self._dealer_seat
