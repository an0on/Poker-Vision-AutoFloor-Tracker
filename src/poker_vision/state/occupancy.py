"""Seat occupancy state machine (REQ-29).

`SeatOccupancyTracker` turns each frame's `FrameAssignments` (REQ-26/27,
already restricted to hysteresis-confirmed tracks -- REQ-24/25) into
`seat_occupied`/`seat_vacated` events: a seat counts as occupied exactly
when at least one `chip` track lands in that seat's `chip_zone`. A `chip`
that only reached the seat's `player_area` (REQ-26's fallback candidate)
does not count -- AC-15 draws that line, not this module.

Events are emitted only on an actual state change, never once per frame:
`update()` diffs this frame's occupied set against the last one and stays
silent for every seat whose occupancy didn't change.

REQ-44's core-chain commit policy splits what used to be one mutating
`update()` into `compute_update()` (pure) and `commit()` (applies a
previously computed `OccupancyUpdate`, assigning `sequence`/`timestamp` at
that point). `update()` is kept as the two called back-to-back for
standalone callers (this module's own tests, and `PipelineStateMachine`'s
own `update()`); `PipelineStateMachine.compute_update()`/`commit()` use
the two steps separately so this tracker's mutation can be deferred until
the whole core chain has succeeded for the frame (see `runner/loop.py`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import EVENT_SCHEMA_VERSION, SeatOccupiedEvent, SeatVacatedEvent

SeatOccupancyEvent = SeatOccupiedEvent | SeatVacatedEvent


@dataclass(frozen=True, slots=True)
class OccupancyUpdate:
    """Pure result of `SeatOccupancyTracker.compute_update()`.

    `changes` is the ordered list of `(seat_id, is_occupied)` transitions
    this frame -- `commit()` turns each into its event, assigning
    `sequence`/`timestamp` only at that point.
    """

    occupied: dict[str, bool]
    changes: list[tuple[str, bool]]
    frame_index: int


class SeatOccupancyTracker:
    """Debounced-by-construction: only reacts to already-stable tracks.

    `seat_ids` is the full table seat universe (from the calibration used
    to produce the `FrameAssignments` this tracker is fed), so every seat
    has a known `False` starting state even before it appears in any
    assignment -- required to detect the first `seat_occupied` transition
    for a seat that never showed a chip before.

    `sequence` numbers emitted events with a private, monotonically
    increasing counter starting at 0 -- correct for this tracker in
    isolation, but not "global monoton über ALLE Event-Quellen" once
    `DealerSeatTracker`/`StreetTracker`/`HandTracker` run alongside it.
    `PipelineStateMachine` (`machine.py`) is what composes all four under
    one shared, globally monotonic counter for REQ-33.
    """

    def __init__(self, seat_ids: Iterable[str]) -> None:
        self._occupied: dict[str, bool] = {seat_id: False for seat_id in seat_ids}
        self._sequence = 0

    def compute_update(self, frame_assignments: FrameAssignments) -> OccupancyUpdate:
        """Pure computation of this frame's occupancy transitions.

        Never mutates `self` -- see module docstring.
        """
        occupied_now = self._resolve_occupied_seats(frame_assignments)

        occupied = dict(self._occupied)
        changes: list[tuple[str, bool]] = []
        for seat_id, was_occupied in self._occupied.items():
            is_occupied = seat_id in occupied_now
            if is_occupied == was_occupied:
                continue
            occupied[seat_id] = is_occupied
            changes.append((seat_id, is_occupied))
        return OccupancyUpdate(
            occupied=occupied, changes=changes, frame_index=frame_assignments.frame_index
        )

    def commit(self, update: OccupancyUpdate, timestamp: datetime) -> list[SeatOccupancyEvent]:
        """Apply a previously computed `OccupancyUpdate` to `self`."""
        self._occupied = update.occupied
        events: list[SeatOccupancyEvent] = []
        for seat_id, is_occupied in update.changes:
            event_cls = SeatOccupiedEvent if is_occupied else SeatVacatedEvent
            events.append(
                event_cls(
                    schema_version=EVENT_SCHEMA_VERSION,
                    sequence=self._next_sequence(),
                    timestamp=timestamp,
                    frame_index=update.frame_index,
                    seat=seat_id,
                )
            )
        return events

    def update(self, frame_assignments: FrameAssignments) -> list[SeatOccupancyEvent]:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(frame_assignments), datetime.now(UTC))

    def _resolve_occupied_seats(self, frame_assignments: FrameAssignments) -> set[str]:
        """Which seats have a chip in their `chip_zone` this frame -- read-only.

        Raises before touching `self._occupied`, so this doubles as the
        validation half of `validate()`: either every referenced seat is
        known and the caller gets a clean result, or nothing about this
        tracker's state has changed yet.
        """
        occupied_now: set[str] = set()
        for assignment in frame_assignments.assignments:
            if assignment.zone is not ZoneKind.CHIP_ZONE:
                continue
            # `ZoneAssignment` doesn't itself tie `zone` to `object_class` --
            # only `assign_zones`'s own dispatch (REQ-26) ever produces a
            # `CHIP_ZONE` assignment for a chip track today. Checking the
            # class explicitly keeps REQ-29's "chip-Track" condition true by
            # construction here too, not just by relying on that invariant
            # holding upstream.
            if assignment.object_class is not DetectionClass.CHIP:
                continue
            seat_id = assignment.seat_id
            if seat_id not in self._occupied:
                raise ValueError(
                    f"FrameAssignments references seat '{seat_id}', which is not in this "
                    "tracker's seat universe -- constructed from a different calibration?"
                )
            occupied_now.add(seat_id)
        return occupied_now

    def validate(self, frame_assignments: FrameAssignments) -> None:
        """Raise if `frame_assignments` references an unknown seat -- no mutation.

        Lets `PipelineStateMachine` (`machine.py`) check every tracker's
        invariants for a frame before mutating any of them, so a frame one
        sibling tracker rejects can't leave this one's state half-applied.
        """
        self._resolve_occupied_seats(frame_assignments)

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> dict[str, bool]:
        """Current occupied/vacated state per seat (for `StateSnapshot`, REQ-33)."""
        return dict(self._occupied)
