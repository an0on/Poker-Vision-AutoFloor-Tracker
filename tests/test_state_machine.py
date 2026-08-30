"""REQ-33: global event sequence + queryable state snapshot (AC-21)."""

from __future__ import annotations

import threading

import pytest

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import (
    DealerMovedEvent,
    HandStartedEvent,
    SeatOccupiedEvent,
    StreetChangedEvent,
)
from poker_vision.state.machine import PipelineStateMachine


def _assignment(
    track_id: int, object_class: DetectionClass, zone: ZoneKind, seat_id: str | None
) -> ZoneAssignment:
    return ZoneAssignment(
        schema_version="1.0",
        track_id=track_id,
        object_class=object_class,
        zone=zone,
        seat_id=seat_id,
    )


def _frame(*assignments: ZoneAssignment, frame_index: int) -> FrameAssignments:
    return FrameAssignments(
        schema_version="1.0", frame_index=frame_index, assignments=list(assignments)
    )


def _chip(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CHIP, ZoneKind.CHIP_ZONE, seat_id)


def _dealer_button(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.DEALER_BUTTON, ZoneKind.PLAYER_AREA, seat_id)


def _cards(count: int, *, frame_index: int) -> FrameAssignments:
    cards = [
        _assignment(track_id, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None)
        for track_id in range(1, count + 1)
    ]
    return _frame(*cards, frame_index=frame_index)


# --- AC-21: sequence is globally monotonic across all four event sources ---


def test_sequence_is_globally_monotonic_across_all_sources():
    machine = PipelineStateMachine(["seat_1", "seat_2"])

    # frame 0: seat_1 gets a chip, dealer button lands at seat_1, board
    # shows a flop -- one frame that fires occupancy, dealer, hand, and
    # street events all at once.
    events = machine.update(
        _frame(
            _chip(1, "seat_1"),
            _dealer_button(2, "seat_1"),
            _assignment(3, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(4, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(5, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            frame_index=0,
        )
    )

    assert [type(e) for e in events] == [
        SeatOccupiedEvent,
        HandStartedEvent,
        StreetChangedEvent,
    ]
    assert [e.sequence for e in events] == [0, 1, 2]

    # dealer button hasn't moved yet -- first resolution establishes the
    # starting position, no dealer_moved event (AC-18). Move it next frame.
    more_events = machine.update(
        _frame(
            _chip(1, "seat_1"),
            _dealer_button(2, "seat_2"),
            _assignment(3, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(4, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(5, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(6, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            frame_index=1,
        )
    )

    assert [type(e) for e in more_events] == [DealerMovedEvent, StreetChangedEvent]
    assert [e.sequence for e in more_events] == [3, 4]


def test_sequence_continues_across_frames_with_no_events():
    machine = PipelineStateMachine(["seat_1"])

    first = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    quiet = machine.update(_frame(_chip(1, "seat_1"), frame_index=1))
    vacated = machine.update(_frame(frame_index=2))

    assert first[0].sequence == 0
    assert quiet == []
    # frame 1 issued no sequence numbers, so the next real event picks up
    # right where frame 0 left off.
    assert vacated[0].sequence == 1


# --- events within one frame appear in a fixed, deterministic order --------


def test_within_frame_order_is_occupancy_dealer_hand_street():
    machine = PipelineStateMachine(["seat_1"])

    events = machine.update(
        _frame(
            _chip(1, "seat_1"),
            _dealer_button(2, "seat_1"),
            _assignment(3, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(4, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(5, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            frame_index=0,
        )
    )

    assert [type(e) for e in events] == [SeatOccupiedEvent, HandStartedEvent, StreetChangedEvent]


# --- all events from one update() call share one timestamp -----------------


def test_events_from_one_frame_share_one_timestamp():
    machine = PipelineStateMachine(["seat_1"])

    events = machine.update(
        _frame(
            _chip(1, "seat_1"),
            _assignment(2, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(3, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(4, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            frame_index=0,
        )
    )

    timestamps = {e.timestamp for e in events}
    assert len(timestamps) == 1


# --- street_changed's hand_id is HandTracker's canonical value -------------


def test_street_changed_hand_id_matches_hand_tracker_canonical_id():
    machine = PipelineStateMachine([])

    machine.update(_cards(0, frame_index=0))
    started = machine.update(_cards(3, frame_index=1))
    machine.update(_cards(0, frame_index=2))
    second_started = machine.update(_cards(3, frame_index=3))

    hand_started_first = next(e for e in started if isinstance(e, HandStartedEvent))
    street_first = next(e for e in started if isinstance(e, StreetChangedEvent))
    assert street_first.hand_id == hand_started_first.hand_id == 1

    hand_started_second = next(e for e in second_started if isinstance(e, HandStartedEvent))
    street_second = next(e for e in second_started if isinstance(e, StreetChangedEvent))
    assert street_second.hand_id == hand_started_second.hand_id == 2


# --- snapshot: queryable at any time ----------------------------------------


def test_snapshot_before_any_update_reflects_empty_initial_state():
    machine = PipelineStateMachine(["seat_1", "seat_2"])

    snapshot = machine.snapshot()

    assert snapshot.sequence == 0
    assert snapshot.frame_index == 0
    assert {s.seat: s.occupied for s in snapshot.seats} == {"seat_1": False, "seat_2": False}
    assert snapshot.dealer_seat is None
    assert snapshot.hand_id is None
    assert snapshot.street is None
    assert snapshot.hand_active is False


def test_snapshot_reflects_full_state_after_updates():
    machine = PipelineStateMachine(["seat_1", "seat_2"])

    machine.update(
        _frame(
            _chip(1, "seat_1"),
            _dealer_button(2, "seat_1"),
            _assignment(3, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(4, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            _assignment(5, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None),
            frame_index=0,
        )
    )

    snapshot = machine.snapshot()

    assert snapshot.frame_index == 0
    assert {s.seat: s.occupied for s in snapshot.seats} == {"seat_1": True, "seat_2": False}
    assert snapshot.dealer_seat == "seat_1"
    assert snapshot.hand_id == 1
    assert snapshot.hand_active is True
    assert snapshot.street.value == "flop"
    # sequence 0, 1, 2 were consumed by the three events above.
    assert snapshot.sequence == 3


# --- a frame rejected by one tracker mutates none of them (REQ-33) ---------


def test_invalid_frame_leaves_all_trackers_unmutated():
    machine = PipelineStateMachine(["seat_1", "seat_2"])

    # seat_1 gets a chip (would be a valid occupancy transition on its
    # own), but the same frame also has two seat-resolved dealer buttons --
    # DealerSeatTracker rejects that outright. The whole frame must be
    # rejected before occupancy ever mutates, not just before dealer does.
    with pytest.raises(ValueError, match="multiple"):
        machine.update(
            _frame(
                _chip(1, "seat_1"),
                _dealer_button(2, "seat_1"),
                _dealer_button(3, "seat_2"),
                frame_index=0,
            )
        )

    snapshot = machine.snapshot()
    assert {s.seat: s.occupied for s in snapshot.seats} == {"seat_1": False, "seat_2": False}
    assert snapshot.dealer_seat is None
    assert snapshot.sequence == 0

    # Retrying with a corrected frame still emits the occupancy event --
    # it was never silently consumed by the rejected attempt.
    events = machine.update(_frame(_chip(1, "seat_1"), frame_index=1))
    assert [type(e) for e in events] == [SeatOccupiedEvent]
    assert events[0].sequence == 0


def test_snapshot_after_hand_ends_keeps_last_hand_id_but_marks_inactive():
    machine = PipelineStateMachine([])

    machine.update(_cards(3, frame_index=0))
    machine.update(_cards(0, frame_index=1))

    snapshot = machine.snapshot()
    assert snapshot.hand_id == 1
    assert snapshot.hand_active is False
    assert snapshot.street is None


# --- concurrent access (REQ-35): update() and snapshot() are mutually exclusive


def test_snapshot_blocks_until_a_concurrent_update_finishes():
    # The websocket export adapter (REQ-35) calls snapshot() from an ASGI
    # server thread while the pipeline's own thread may be mid-update() --
    # without the lock, snapshot() could observe some trackers already
    # mutated for the new frame and others still on the old one.
    machine = PipelineStateMachine(["seat_1"])
    entered_update = threading.Event()
    release_update = threading.Event()

    # `commit()` is the phase that actually mutates state under `self._lock`
    # (REQ-44: `compute_update()` is pure and unlocked, since there's
    # nothing for a concurrent snapshot() to observe half-done there) --
    # blocking there is what demonstrates snapshot() waits for the lock.
    original_dealer_commit = machine._dealer.commit

    def blocking_dealer_commit(update, timestamp):
        entered_update.set()
        assert release_update.wait(timeout=5), "test setup: release was never signaled"
        return original_dealer_commit(update, timestamp)

    machine._dealer.commit = blocking_dealer_commit

    update_thread = threading.Thread(
        target=machine.update, args=(_frame(_chip(1, "seat_1"), frame_index=0),)
    )
    update_thread.start()
    assert entered_update.wait(timeout=5), "update() never reached the dealer tracker"

    snapshots: list = []
    snapshot_thread = threading.Thread(target=lambda: snapshots.append(machine.snapshot()))
    snapshot_thread.start()

    # update() is still holding the lock inside the blocked dealer tracker --
    # snapshot() must not have returned yet.
    snapshot_thread.join(timeout=0.2)
    assert snapshot_thread.is_alive()
    assert snapshots == []

    release_update.set()
    update_thread.join(timeout=5)
    snapshot_thread.join(timeout=5)

    assert len(snapshots) == 1
    assert snapshots[0].sequence == 1
    assert snapshots[0].seats[0].occupied is True
