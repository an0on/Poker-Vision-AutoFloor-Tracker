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
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments
from poker_vision.state.board import count_board_cards
from poker_vision.state.events import EVENT_SCHEMA_VERSION, Street, StreetChangedEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreetUpdate:
    """Pure result of `StreetTracker.compute_update()`.

    `changed_street` is the new street when this frame is an actual
    forward transition worth a `street_changed` event, `None` otherwise.
    REQ-44's core-chain commit policy splits what used to be one mutating
    `update()` into `compute_update()` (pure) and `commit()`; `update()`
    is kept as the two called back-to-back for standalone callers,
    `PipelineStateMachine` uses the two steps separately (see
    `runner/loop.py`).
    """

    current_street: Street | None
    board_active: bool
    hand_id: int
    changed_street: Street | None
    frame_index: int

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
    -- `PipelineStateMachine` (`machine.py`) composes sibling event sources
    under one shared, globally monotonic counter for REQ-33 (see
    `occupancy.py`).

    `hand_id` is REQ-33's mandatory field on `StreetChangedEvent`, but
    REQ-31 has no concept of a "hand" of its own -- that is REQ-32's
    `HandTracker` (`hand.py`), defined purely by the board going empty <->
    non-empty (AGENTS.md), independent of whether a valid street was ever
    reached in between. This tracker still maintains its own private
    counter that bumps on every non-empty -> empty transition of the board
    -- the same transition `HandTracker` reacts to, via the shared
    `count_board_cards` (`board.py`) -- so it advances even for a "hand"
    that only ever showed 1, 2, or >5 stable cards and never reached a
    valid street. `PipelineStateMachine` overwrites this tracker's
    `hand_id` on every emitted `StreetChangedEvent` with `HandTracker`'s
    real, canonical one instead -- the two stay numerically in sync on a
    single-table replay regardless, since both apply that identical rule
    to the same input stream, but REQ-33's composition is what actually
    makes `HandTracker` the one source of truth rather than relying on
    that emergent agreement.
    """

    def __init__(self) -> None:
        self._current_street: Street | None = None
        self._board_active = False
        self._hand_id = 1
        self._sequence = 0

    def compute_update(self, frame_assignments: FrameAssignments) -> StreetUpdate:
        """Pure computation of this frame's street transition.

        Never mutates `self` -- see module docstring.
        """
        count = count_board_cards(frame_assignments)
        frame_index = frame_assignments.frame_index

        board_active = self._board_active
        current_street = self._current_street
        hand_id = self._hand_id

        if count > 0 and not board_active:
            board_active = True
        elif count == 0 and board_active:
            # Stable empty board -- hand boundary regardless of whether a
            # valid street was ever reached this hand (AGENTS.md's "Board
            # leer <-> nicht-leer"). Reset the monotonic gate and advance
            # the hand id for the next hand.
            board_active = False
            current_street = None
            hand_id += 1

        no_change = StreetUpdate(
            current_street=current_street,
            board_active=board_active,
            hand_id=hand_id,
            changed_street=None,
            frame_index=frame_index,
        )

        if count == 0:
            return no_change

        street = _COUNT_TO_STREET.get(count)
        if street is None:
            logger.warning(
                "board_zone has %d stable card track(s), which maps to no street "
                "(expected 3, 4, or 5) -- ignoring this frame",
                count,
            )
            return no_change

        if _STREET_RANK[street] <= _STREET_RANK[current_street]:
            # Not a forward transition within this hand -- a flicker back
            # to an already-passed or same street, ignored silently.
            return no_change

        return StreetUpdate(
            current_street=street,
            board_active=board_active,
            hand_id=hand_id,
            changed_street=street,
            frame_index=frame_index,
        )

    def commit(self, update: StreetUpdate, timestamp: datetime) -> list[StreetChangedEvent]:
        """Apply a previously computed `StreetUpdate` to `self`."""
        self._current_street = update.current_street
        self._board_active = update.board_active
        self._hand_id = update.hand_id
        if update.changed_street is None:
            return []
        event = StreetChangedEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=self._next_sequence(),
            timestamp=timestamp,
            frame_index=update.frame_index,
            hand_id=update.hand_id,
            street=update.changed_street,
        )
        return [event]

    def update(self, frame_assignments: FrameAssignments) -> list[StreetChangedEvent]:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(frame_assignments), datetime.now(UTC))

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> Street | None:
        """Current street, or `None` before the flop (for `StateSnapshot`, REQ-33)."""
        return self._current_street
