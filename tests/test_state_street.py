"""REQ-31: street from stable card-track count in board_zone (AC-19)."""

from __future__ import annotations

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import Street, StreetChangedEvent
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


# --- AC-19: 3 -> 4 -> 5 emits flop, turn, river in order --------------------


def test_three_to_four_to_five_emits_flop_turn_river_in_order():
    tracker = StreetTracker()

    flop_events = tracker.update(_frame(3, frame_index=0))
    turn_events = tracker.update(_frame(4, frame_index=1))
    river_events = tracker.update(_frame(5, frame_index=2))

    assert [e.street for e in flop_events] == [Street.FLOP]
    assert [e.street for e in turn_events] == [Street.TURN]
    assert [e.street for e in river_events] == [Street.RIVER]
    assert all(isinstance(e, StreetChangedEvent) for e in flop_events + turn_events + river_events)


# --- AC-19: 3 -> 2 -> 3 flicker emits exactly one flop event ---------------


def test_flicker_three_two_three_emits_exactly_one_flop_event():
    tracker = StreetTracker()

    first = tracker.update(_frame(3, frame_index=0))
    flicker = tracker.update(_frame(2, frame_index=1))
    back = tracker.update(_frame(3, frame_index=2))

    assert len(first) == 1
    assert first[0].street == Street.FLOP
    assert flicker == []
    assert back == []


# --- AC-19: 4 -> 3 within a hand emits no event -----------------------------


def test_four_then_three_within_a_hand_emits_no_event_on_the_drop():
    tracker = StreetTracker()

    turn_events = tracker.update(_frame(4, frame_index=0))
    drop_events = tracker.update(_frame(3, frame_index=1))

    assert len(turn_events) == 1
    assert turn_events[0].street == Street.TURN
    assert drop_events == []


# --- 1, 2, and >5 cards produce no event (warning only) ---------------------


def test_one_card_emits_no_event():
    tracker = StreetTracker()
    assert tracker.update(_frame(1, frame_index=0)) == []


def test_two_cards_emits_no_event():
    tracker = StreetTracker()
    assert tracker.update(_frame(2, frame_index=0)) == []


def test_more_than_five_cards_emits_no_event():
    tracker = StreetTracker()
    assert tracker.update(_frame(6, frame_index=0)) == []


# --- reset only on a stably empty board -------------------------------------


def test_empty_board_resets_and_next_hand_reaches_flop_again():
    tracker = StreetTracker()

    tracker.update(_frame(3, frame_index=0))
    tracker.update(_frame(5, frame_index=1))
    empty_events = tracker.update(_frame(0, frame_index=2))

    assert empty_events == []
    assert tracker.snapshot() is None

    second_hand_flop = tracker.update(_frame(3, frame_index=3))
    assert len(second_hand_flop) == 1
    assert second_hand_flop[0].street == Street.FLOP
    assert second_hand_flop[0].hand_id == 2


def test_dip_to_in_between_count_does_not_reset_the_gate():
    tracker = StreetTracker()

    tracker.update(_frame(3, frame_index=0))
    tracker.update(_frame(2, frame_index=1))  # dip, not empty -- no reset
    same_street_again = tracker.update(_frame(3, frame_index=2))

    assert same_street_again == []
    assert tracker.snapshot() == Street.FLOP


def test_hand_id_bumps_even_if_no_valid_street_was_ever_reached():
    # Hand boundaries are board empty <-> non-empty (AGENTS.md), not
    # "a valid street was reached" -- a hand that only ever showed a
    # misdetected count still counts as a hand for numbering purposes.
    tracker = StreetTracker()

    tracker.update(_frame(1, frame_index=0))
    tracker.update(_frame(0, frame_index=1))
    flop_events = tracker.update(_frame(3, frame_index=2))

    assert len(flop_events) == 1
    assert flop_events[0].hand_id == 2


# --- events carry frame_index and a monotonic per-tracker sequence ---------


def test_events_carry_frame_index_and_monotonic_sequence():
    tracker = StreetTracker()

    flop_events = tracker.update(_frame(3, frame_index=7))
    turn_events = tracker.update(_frame(4, frame_index=8))

    assert flop_events[0].frame_index == 7
    assert turn_events[0].frame_index == 8
    assert turn_events[0].sequence == flop_events[0].sequence + 1


def test_hand_id_bumps_across_a_hand_that_never_reached_a_valid_street():
    tracker = StreetTracker()

    # First hand: board goes non-empty but only ever shows a misdetected
    # count (>5), never a valid street, then goes empty again.
    tracker.update(_frame(6, frame_index=0))
    tracker.update(_frame(0, frame_index=1))

    # Second hand reaches flop -- must carry hand_id 2, not 1.
    second_hand_flop = tracker.update(_frame(3, frame_index=2))
    assert second_hand_flop[0].hand_id == 2


def test_snapshot_starts_at_none():
    tracker = StreetTracker()
    assert tracker.snapshot() is None
