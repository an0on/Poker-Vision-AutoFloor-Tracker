"""REQ-30: dealer seat from dealer_button assignment, events only on change (AC-18)."""

from __future__ import annotations

import pytest

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.dealer import DealerSeatTracker
from poker_vision.state.events import DealerMovedEvent


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


def _player_area_button(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.DEALER_BUTTON, ZoneKind.PLAYER_AREA, seat_id)


def _dealer_area_button(track_id: int, seat_id: str | None) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.DEALER_BUTTON, ZoneKind.DEALER_AREA, seat_id)


# --- AC-18: button seat 1 -> seat 2, then disappears -----------------------


def test_button_seat1_to_seat2_fixture_matches_ac18():
    tracker = DealerSeatTracker(["seat_1", "seat_2"])

    # button sits at seat 1 for a few frames -- the first observation only
    # establishes the starting position (no prior seat to have "changed"
    # from), repeats emit nothing either.
    first = tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    steady = tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=1))
    assert first == []
    assert steady == []

    # button moves to seat 2 -- exactly one dealer_moved(1, 2)
    moved = tracker.update(_frame(_player_area_button(1, "seat_2"), frame_index=2))
    assert len(moved) == 1
    assert isinstance(moved[0], DealerMovedEvent)
    assert moved[0].from_seat == "seat_1"
    assert moved[0].to_seat == "seat_2"

    # button disappears entirely -- no event, dealer seat stays seat 2
    gone = tracker.update(_frame(frame_index=3))
    assert gone == []
    assert tracker.snapshot() == "seat_2"


# --- first resolution establishes state silently, no event -----------------


def test_first_button_resolution_emits_no_event():
    tracker = DealerSeatTracker(["seat_1"])
    events = tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    assert events == []
    assert tracker.snapshot() == "seat_1"


def test_second_seat_after_first_resolution_emits_dealer_moved():
    tracker = DealerSeatTracker(["seat_1", "seat_2"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    events = tracker.update(_frame(_player_area_button(1, "seat_2"), frame_index=1))
    assert len(events) == 1
    assert events[0].from_seat == "seat_1"
    assert events[0].to_seat == "seat_2"
    assert events[0].frame_index == 1


# --- no event without an actual state change --------------------------------


def test_repeated_same_seat_emits_no_further_event():
    tracker = DealerSeatTracker(["seat_1"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    events = tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=1))
    assert events == []


def test_repeated_empty_frame_emits_no_event():
    tracker = DealerSeatTracker(["seat_1"])
    events = tracker.update(_frame(frame_index=0))
    assert events == []


# --- button disappearing / left seat-less does not change the seat ---------


def test_button_missing_from_frame_keeps_last_known_seat():
    tracker = DealerSeatTracker(["seat_1"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    events = tracker.update(_frame(frame_index=1))
    assert events == []
    assert tracker.snapshot() == "seat_1"


def test_dealer_area_hit_beyond_threshold_without_seat_id_keeps_last_known_seat():
    # REQ-27/AC-16: beyond the nearest-seat threshold, the fallback leaves
    # the button as a seat-less dealer_area entry -- "unassigned", not a
    # dealer_moved trigger.
    tracker = DealerSeatTracker(["seat_1"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    events = tracker.update(_frame(_dealer_area_button(1, None), frame_index=1))
    assert events == []
    assert tracker.snapshot() == "seat_1"


def test_dealer_area_fallback_resolution_counts_as_a_seat_change():
    # apply_dealer_nearest_seat_fallback resolves a seat-less dealer_area
    # hit to a seat within threshold -- that resolution is a real seat too.
    tracker = DealerSeatTracker(["seat_1", "seat_2"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    events = tracker.update(_frame(_dealer_area_button(1, "seat_2"), frame_index=1))
    assert len(events) == 1
    assert events[0].from_seat == "seat_1"
    assert events[0].to_seat == "seat_2"


# --- sequence numbering (REQ-33) --------------------------------------------


def test_sequence_is_monotonic_across_updates():
    tracker = DealerSeatTracker(["seat_1", "seat_2"])
    tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=0))
    first = tracker.update(_frame(_player_area_button(1, "seat_2"), frame_index=1))
    second = tracker.update(_frame(_player_area_button(1, "seat_1"), frame_index=2))
    assert first[0].sequence == 0
    assert second[0].sequence == 1


# --- dealer seat requires an actual dealer_button track, not just the label -


def test_non_dealer_button_assignment_does_not_count():
    tracker = DealerSeatTracker(["seat_1"])
    bogus = _assignment(1, DetectionClass.CHIP, ZoneKind.PLAYER_AREA, "seat_1")
    events = tracker.update(_frame(bogus, frame_index=0))
    assert events == []
    assert tracker.snapshot() is None


# --- unknown seat is a hard error, not a silent no-op -----------------------


def test_unknown_seat_id_in_assignment_raises():
    tracker = DealerSeatTracker(["seat_1"])
    with pytest.raises(ValueError, match="seat_9"):
        tracker.update(_frame(_player_area_button(1, "seat_9"), frame_index=0))


def test_multiple_seat_resolved_dealer_buttons_in_one_frame_raises():
    tracker = DealerSeatTracker(["seat_1", "seat_2"])
    with pytest.raises(ValueError, match="multiple"):
        tracker.update(
            _frame(
                _player_area_button(1, "seat_1"),
                _player_area_button(2, "seat_2"),
                frame_index=0,
            )
        )


# --- snapshot ----------------------------------------------------------------


def test_snapshot_starts_none():
    tracker = DealerSeatTracker(["seat_1"])
    assert tracker.snapshot() is None


def test_snapshot_reflects_current_dealer_seat():
    tracker = DealerSeatTracker(["seat_1", "seat_2"])
    tracker.update(_frame(_player_area_button(1, "seat_2"), frame_index=0))
    assert tracker.snapshot() == "seat_2"
