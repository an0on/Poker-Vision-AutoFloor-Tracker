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

REQ-44's core-chain commit policy splits what used to be one mutating
`update()` into `compute_update()` (pure -- validates, then computes every
sub-tracker's pure update without mutating any of them or `self`) and
`commit()` (applies all four sub-trackers' updates plus this machine's own
`sequence`/`frame_index`/`timestamp`, still under `self._lock`). `update()`
is kept as the two called back-to-back, holding the lock across both, so
every other caller (this module's own tests included) sees the exact same
mutate-immediately, atomic-per-call behavior as before; only the runner's
frame loop needs the two steps split apart, to defer this stage's
mutation until the whole core chain (tracking -> assignment -> state) has
succeeded for the frame (see `runner/loop.py`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_vision.assignment.models import FrameAssignments
from poker_vision.state.dealer import DealerSeatTracker, DealerUpdate
from poker_vision.state.events import Event
from poker_vision.state.hand import HandTracker, HandUpdate
from poker_vision.state.occupancy import OccupancyUpdate, SeatOccupancyTracker
from poker_vision.state.snapshot import STATE_SNAPSHOT_SCHEMA_VERSION, SeatOccupancy, StateSnapshot
from poker_vision.state.street import StreetTracker, StreetUpdate


@dataclass(frozen=True, slots=True)
class StateUpdate:
    """Pure result of `PipelineStateMachine.compute_update()`.

    Bundles all four sub-trackers' own pure updates so `commit()` can
    apply them in the fixed per-frame order (occupancy, dealer, hand,
    street) without recomputing anything.
    """

    occupancy_update: OccupancyUpdate
    dealer_update: DealerUpdate
    hand_update: HandUpdate
    street_update: StreetUpdate
    frame_index: int


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

    def compute_update(self, frame_assignments: FrameAssignments) -> StateUpdate:
        """Pure computation of this frame's update across all four trackers.

        Never mutates `self` or any of the four sub-trackers -- see module
        docstring. `validate()` runs first so a frame that will be
        rejected is rejected before anything is even computed, matching
        `update()`'s existing all-or-nothing guarantee for occupancy/
        dealer's invariants.

        Not lock-guarded: unlike `commit()`, nothing here mutates shared
        state, so there is nothing for a concurrent `snapshot()` to
        observe half-done.
        """
        self._occupancy.validate(frame_assignments)
        self._dealer.validate(frame_assignments)
        return StateUpdate(
            occupancy_update=self._occupancy.compute_update(frame_assignments),
            dealer_update=self._dealer.compute_update(frame_assignments),
            hand_update=self._hand.compute_update(frame_assignments),
            street_update=self._street.compute_update(frame_assignments),
            frame_index=frame_assignments.frame_index,
        )

    def commit(self, update: StateUpdate) -> list[Event]:
        """Apply a previously computed `StateUpdate` to all four trackers.

        All events produced from this single call share one `timestamp` --
        the moment this update was committed -- rather than each tracker's
        own `datetime.now(UTC)` call, which would otherwise introduce
        spurious sub-millisecond skew between events that all describe the
        same frame.
        """
        with self._lock:
            timestamp = datetime.now(UTC)

            occupancy_events = self._occupancy.commit(update.occupancy_update, timestamp)
            dealer_events = self._dealer.commit(update.dealer_update, timestamp)
            hand_events = self._hand.commit(update.hand_update, timestamp)
            street_events = self._street.commit(update.street_update, timestamp)

            if street_events:
                hand_id, _ = self._hand.snapshot()
                # A street_changed event only fires while the board is non-empty
                # (count > 0), and HandTracker's commit() above already ran for
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

            self._frame_index = update.frame_index
            self._timestamp = timestamp
            return events

    def update(self, frame_assignments: FrameAssignments) -> list[Event]:
        """Compute and immediately commit this frame's update (see module docstring)."""
        return self.commit(self.compute_update(frame_assignments))

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
