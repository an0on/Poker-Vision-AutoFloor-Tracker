"""Street state machine (REQ-31).

`StreetTracker` turns each frame's `FrameAssignments` (REQ-26/27, already
restricted to hysteresis-confirmed tracks -- REQ-24/25) into
`street_changed` events: the street is derived purely from how many `card`
tracks currently land in the table's single `board_zone` -- 3 -> flop,
4 -> turn, 5 -> river. Rank/Farbe are out of scope (MVP); only the count
matters.

Counts of 1 or 2 (a flop briefly occluded down to fewer visible cards) and
counts above 5 (a misdetection) never map to a street -- they are logged as
a warning and otherwise ignored, per REQ-31 ("erzeugen kein Event, nur
Warnlog").

Within a hand, only monotonically increasing transitions are accepted
(flop -> turn -> river, or a transition that skips a step forward): a count
that maps to a street at or behind the currently committed one is treated
as a still-frame or a misdetection, not a "real" street change, and is
silently ignored (AC-19's "4 -> 3 innerhalb einer Hand erzeugt kein
Event", and the "3 -> 2 -> 3" flicker fixture, which must emit exactly one
`flop` event rather than one on each `3`). The monotonic gate resets only
once the board is observed stably empty (`count == 0`) -- "Rücksprung erst
bei leerem Board" -- never on a mere dip to an in-between count.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import EVENT_SCHEMA_VERSION, Street, StreetChangedEvent

logger = logging.getLogger(__name__)

_COUNT_TO_STREET: dict[int, Street] = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}

# Ordinal rank used to decide "monotonically increasing" -- `None` (no
# street reached yet this hand) ranks below `flop`, which ranks below
# `turn`, which ranks below `river`.
_STREET_RANK: dict[Street | None, int] = {
    None: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
}


class StreetTracker:
    """Debounced-by-construction: only reacts to already-stable tracks.

    Unlike `SeatOccupancyTracker`/`DealerSeatTracker`, there is no seat
    universe to validate against -- `board_zone` is the one global zone a
    `card` track can land in, so this tracker only ever counts, never
    resolves a seat.

    `sequence` numbers emitted events with a private, monotonically
    increasing counter starting at 0, independent of the other trackers'
    -- composing sibling event sources under one shared, globally monotonic
    counter is later work's job, not this one's (see `occupancy.py`).

    `hand_id` is REQ-33's mandatory field on `StreetChangedEvent`, but
    REQ-31 has no concept of a "hand" of its own -- that is REQ-32's
    `hand_started`/`hand_ended` tracker. Until that tracker exists and a
    composing state machine can hand this tracker the real, shared
    `hand_id`, this tracker maintains its own private counter that bumps
    every time the board goes from a committed street back to stably
    empty -- the same board-empty signal REQ-32 will need to detect a hand
    boundary in the first place. That makes the numbering here consistent
    with what REQ-32 will produce for a single-table replay, but it is not
    the same counter object; reconciling the two remains later work.
    """

    def __init__(self) -> None:
        self._current_street: Street | None = None
        self._hand_id = 1
        self._sequence = 0

    def update(self, frame_assignments: FrameAssignments) -> list[StreetChangedEvent]:
        count = sum(
            1
            for assignment in frame_assignments.assignments
            if assignment.zone is ZoneKind.BOARD_ZONE
            and assignment.object_class is DetectionClass.CARD
        )

        if count == 0:
            if self._current_street is not None:
                # Stable empty board -- reset the monotonic gate for the
                # next hand (AC-19's "Rücksprung erst bei leerem Board").
                self._current_street = None
                self._hand_id += 1
            return []

        street = _COUNT_TO_STREET.get(count)
        if street is None:
            logger.warning(
                "board_zone has %d stable card track(s), which maps to no street "
                "(expected 3, 4, or 5) -- ignoring this frame",
                count,
            )
            return []

        if _STREET_RANK[street] <= _STREET_RANK[self._current_street]:
            # Not a forward transition within this hand -- a flicker back
            # to an already-passed or same street, ignored silently.
            return []

        self._current_street = street
        event = StreetChangedEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=self._next_sequence(),
            timestamp=datetime.now(UTC),
            frame_index=frame_assignments.frame_index,
            hand_id=self._hand_id,
            street=street,
        )
        return [event]

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> Street | None:
        """Current street, or `None` before the flop (for `StateSnapshot`, REQ-33)."""
        return self._current_street
