"""Composing state machine: one global sequence, one queryable snapshot (REQ-33).

`SeatOccupancyTracker` (REQ-29), `DealerSeatTracker` (REQ-30), `StreetTracker`
(REQ-31), and `HandTracker` (REQ-32) each turn the same per-frame
`FrameAssignments` into their own slice of typed events, but each keeps its
own private, per-tracker `sequence` counter -- fine in isolation, but not
"global monoton über ALLE Event-Quellen" once all four run together.
`PipelineStateMachine` is that composition: it drives all four trackers over
one frame, discards their private `sequence` values, and reassigns every
event a slot from one shared, monotonically increasing counter -- in a fixed
per-frame order (occupancy, dealer, hand, street) so that within a single
frame the ordering is deterministic and reproducible, not an artifact of
dict/list iteration.

It also resolves the one piece of state the individual trackers explicitly
left undone: `StreetTracker` keeps its own private `hand_id` counter (see
`street.py`'s docstring) that stays numerically in sync with `HandTracker`'s
canonical one only because both apply the identical empty/non-empty rule to
the same input stream. Rather than rely on that emergent agreement,
`PipelineStateMachine` overwrites every `StreetChangedEvent.hand_id` with
`HandTracker`'s actual `hand_id` for that frame, so `HandTracker` (REQ-32) is
the one real source of hand identity, per-tracker or not.

`SeatOccupancyTracker`/`DealerSeatTracker` each raise `ValueError` on an
invalid frame (an unknown seat, or -- dealer only -- multiple seat-resolved
dealer_button tracks at once), and each does so before touching its own
state, so a single tracker's `update()` is already all-or-nothing. Calling
four trackers one after another is not, though: if `_occupancy.update()`
mutates state and returns events before `_dealer.update()` raises on the
same frame, that exception discards the already-computed occupancy events
while `snapshot()` keeps reflecting the mutation that produced them --
silently losing a legitimate transition the caller never sees. `update()`
therefore validates the frame against both trackers' invariants up front,
via their `validate()` methods, before calling either tracker's `update()`
-- so a frame that will be rejected is rejected before anything mutates.

`update()` and `snapshot()` are guarded by the same lock. The pipeline
calls `update()` from its own frame loop, while the `websocket` export
adapter (REQ-35) calls `snapshot()` from a separate ASGI server
thread/loop -- without the lock, a `/status` request or a fresh WebSocket
connection could read the four trackers mid-`update()`, after some but not
all of them have mutated for the current frame, and see a state that never
existed at any single point in time (e.g. this frame's new dealer seat
paired with the previous frame's `sequence`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments
from poker_vision.state.dealer import DealerSeatTracker
from poker_vision.state.events import Event
from poker_vision.state.hand import HandTracker
from poker_vision.state.occupancy import SeatOccupancyTracker
from poker_vision.state.snapshot import STATE_SNAPSHOT_SCHEMA_VERSION, SeatOccupancy, StateSnapshot
from poker_vision.state.street import StreetTracker


class PipelineStateMachine:
    """Runs all four trackers over a frame under one global event sequence.

    `seat_ids` is the table's full seat universe, forwarded to the two
    trackers that validate assignments against it (`SeatOccupancyTracker`,
    `DealerSeatTracker`) -- the same seat universe used to produce the
    `FrameAssignments` this machine is fed.
    """

    def __init__(self, seat_ids: Iterable[str]) -> None:
        seat_ids = list(seat_ids)
        self._occupancy = SeatOccupancyTracker(seat_ids)
        self._dealer = DealerSeatTracker(seat_ids)
        self._hand = HandTracker()
        self._street = StreetTracker()

        self._sequence = 0
        self._frame_index = 0
        self._timestamp = datetime.now(UTC)
        self._lock = threading.Lock()

    def update(self, frame_assignments: FrameAssignments) -> list[Event]:
        """Advance every tracker by one frame, returning its events in sequence order.

        All events produced from this single call share one `timestamp` --
        the moment this frame was processed -- rather than each tracker's own
        `datetime.now(UTC)` call, which would otherwise introduce spurious
        sub-millisecond skew between events that all describe the same frame.
        """
        with self._lock:
            self._occupancy.validate(frame_assignments)
            self._dealer.validate(frame_assignments)

            timestamp = datetime.now(UTC)

            occupancy_events = self._occupancy.update(frame_assignments)
            dealer_events = self._dealer.update(frame_assignments)
            hand_events = self._hand.update(frame_assignments)
            street_events = self._street.update(frame_assignments)

            if street_events:
                hand_id, _ = self._hand.snapshot()
                # A street_changed event only fires while the board is non-empty
                # (count > 0), and HandTracker's update() above already ran for
                # this same frame -- so its board-active flag, and thus hand_id,
                # is guaranteed to already reflect this frame's transition.
                assert hand_id is not None, (
                    "StreetTracker emitted a street_changed event while HandTracker "
                    "reports no active hand for the same frame -- the two trackers "
                    "have diverged on the board empty/non-empty signal"
                )
                street_events = [
                    event.model_copy(update={"hand_id": hand_id}) for event in street_events
                ]

            ordered: list[Event] = [
                *occupancy_events,
                *dealer_events,
                *hand_events,
                *street_events,
            ]
            events = [
                event.model_copy(
                    update={"sequence": self._next_sequence(), "timestamp": timestamp}
                )
                for event in ordered
            ]

            self._frame_index = frame_assignments.frame_index
            self._timestamp = timestamp
            return events

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def snapshot(self) -> StateSnapshot:
        """Full pipeline state as of the last processed frame, queryable at any time.

        Before the first `update()` call, this reflects the machine's initial
        state: frame 0, no seats occupied, no dealer seat, no hand in
        progress.
        """
        with self._lock:
            hand_id, hand_active = self._hand.snapshot()
            return StateSnapshot(
                schema_version=STATE_SNAPSHOT_SCHEMA_VERSION,
                sequence=self._sequence,
                timestamp=self._timestamp,
                frame_index=self._frame_index,
                seats=[
                    SeatOccupancy(seat=seat_id, occupied=occupied)
                    for seat_id, occupied in self._occupancy.snapshot().items()
                ],
                dealer_seat=self._dealer.snapshot(),
                hand_id=hand_id,
                street=self._street.snapshot(),
                hand_active=hand_active,
            )
