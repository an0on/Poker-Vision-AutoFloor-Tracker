"""REQ-32: hand lifecycle from board empty <-> non-empty (AC-20)."""

from __future__ import annotations

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import HandEndedEvent, HandStartedEvent
from poker_vision.state.hand import HandTracker
from poker_vision.state.street import StreetTracker


def _card(track_id: int) -> ZoneAssignment:
    return ZoneAssignment(
        schema_version="1.0",
        track_id=track_id,
        object_class=DetectionClass.CARD,
        zone=ZoneKind.BOARD_ZONE,
        seat_id=None,
    )


def _frame(count: int, *, frame_index: int) -> FrameAssignments:
    return FrameAssignments(
        schema_version="1.0",
        frame_index=frame_index,
        assignments=[_card(track_id) for track_id in range(1, count + 1)],
    )


# --- AC-20: empty -> non-empty starts a hand --------------------------------


def test_empty_to_non_empty_emits_hand_started():
    tracker = HandTracker()

    events = tracker.update(_frame(3, frame_index=0))

    assert len(events) == 1
    assert isinstance(events[0], HandStartedEvent)
    assert events[0].hand_id == 1


def test_starting_empty_emits_nothing():
    tracker = HandTracker()
    assert tracker.update(_frame(0, frame_index=0)) == []


def test_non_empty_to_non_empty_emits_nothing():
    tracker = HandTracker()

    tracker.update(_frame(3, frame_index=0))
    events = tracker.update(_frame(4, frame_index=1))

    assert events == []


# --- AC-20: non-empty -> stably empty ends the hand -------------------------


def test_non_empty_to_empty_emits_hand_ended_with_same_hand_id():
    tracker = HandTracker()

    started = tracker.update(_frame(3, frame_index=0))
    ended = tracker.update(_frame(0, frame_index=1))

    assert len(ended) == 1
    assert isinstance(ended[0], HandEndedEvent)
    assert ended[0].hand_id == started[0].hand_id == 1


def test_empty_to_empty_emits_nothing():
    tracker = HandTracker()

    tracker.update(_frame(3, frame_index=0))
    tracker.update(_frame(0, frame_index=1))
    events = tracker.update(_frame(0, frame_index=2))

    assert events == []


# --- AC-20: "leer -> 3 -> 5 -> leer" fixture --------------------------------


def test_empty_three_five_empty_fixture_emits_started_then_ended():
    tracker = HandTracker()

    empty_events = tracker.update(_frame(0, frame_index=0))
    started_events = tracker.update(_frame(3, frame_index=1))
    mid_events = tracker.update(_frame(5, frame_index=2))
    ended_events = tracker.update(_frame(0, frame_index=3))

    assert empty_events == []
    assert [type(e) for e in started_events] == [HandStartedEvent]
    assert mid_events == []
    assert [type(e) for e in ended_events] == [HandEndedEvent]
    assert started_events[0].hand_id == ended_events[0].hand_id == 1


def test_second_hand_gets_hand_id_plus_one():
    tracker = HandTracker()

    tracker.update(_frame(0, frame_index=0))
    tracker.update(_frame(3, frame_index=1))
    tracker.update(_frame(5, frame_index=2))
    tracker.update(_frame(0, frame_index=3))

    second_started = tracker.update(_frame(3, frame_index=4))
    assert second_started[0].hand_id == 2


# --- hand_id stays in sync with StreetTracker's own counter -----------------


def test_hand_id_stays_in_sync_with_street_tracker():
    hand_tracker = HandTracker()
    street_tracker = StreetTracker()

    frames = [
        _frame(0, frame_index=0),
        _frame(3, frame_index=1),
        _frame(5, frame_index=2),
        _frame(0, frame_index=3),
        _frame(3, frame_index=4),
    ]

    hand_events = [event for frame in frames for event in hand_tracker.update(frame)]
    street_events = [event for frame in frames for event in street_tracker.update(frame)]

    assert [e.hand_id for e in hand_events] == [1, 1, 2]
    assert [e.hand_id for e in street_events] == [1, 1, 2]


# --- 1 or 2 stable cards still count as "hand active" -----------------------


def test_hand_starts_even_on_a_count_that_maps_to_no_street():
    tracker = HandTracker()

    events = tracker.update(_frame(1, frame_index=0))

    assert len(events) == 1
    assert isinstance(events[0], HandStartedEvent)


# --- events carry frame_index and a monotonic per-tracker sequence ---------


def test_events_carry_frame_index_and_monotonic_sequence():
    tracker = HandTracker()

    started = tracker.update(_frame(3, frame_index=7))
    ended = tracker.update(_frame(0, frame_index=8))

    assert started[0].frame_index == 7
    assert ended[0].frame_index == 8
    assert ended[0].sequence == started[0].sequence + 1


# --- snapshot ----------------------------------------------------------------


def test_snapshot_starts_at_none_and_inactive():
    tracker = HandTracker()
    assert tracker.snapshot() == (None, False)


def test_snapshot_reflects_active_hand():
    tracker = HandTracker()
    tracker.update(_frame(3, frame_index=0))
    assert tracker.snapshot() == (1, True)


def test_snapshot_reflects_ended_hand():
    tracker = HandTracker()
    tracker.update(_frame(3, frame_index=0))
    tracker.update(_frame(0, frame_index=1))
    assert tracker.snapshot() == (1, False)
