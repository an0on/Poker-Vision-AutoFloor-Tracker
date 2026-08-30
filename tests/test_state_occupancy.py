"""REQ-29: seat occupancy from chip-zone presence, events only on change (AC-17)."""

from __future__ import annotations

import pytest

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.state.events import SeatOccupiedEvent, SeatVacatedEvent
from poker_vision.state.occupancy import SeatOccupancyTracker


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


def _chip_zone(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CHIP, ZoneKind.CHIP_ZONE, seat_id)


def _player_area(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CHIP, ZoneKind.PLAYER_AREA, seat_id)


# --- AC-17: chip in, chip out --------------------------------------------


def test_chip_entering_chip_zone_emits_seat_occupied():
    tracker = SeatOccupancyTracker(["seat_3"])
    events = tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=5))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SeatOccupiedEvent)
    assert event.seat == "seat_3"
    assert event.frame_index == 5


def test_chip_leaving_chip_zone_emits_seat_vacated():
    tracker = SeatOccupancyTracker(["seat_3"])
    tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=5))
    events = tracker.update(_frame(frame_index=6))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SeatVacatedEvent)
    assert event.seat == "seat_3"
    assert event.frame_index == 6


def test_full_chip_in_out_sequence_matches_ac17():
    tracker = SeatOccupancyTracker(["seat_3"])
    assert tracker.update(_frame(frame_index=0)) == []
    occupied = tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=1))
    vacated = tracker.update(_frame(frame_index=2))

    event_types = [type(e).__name__ for e in occupied + vacated]
    assert event_types == ["SeatOccupiedEvent", "SeatVacatedEvent"]
    assert [e.frame_index for e in occupied + vacated] == [1, 2]


# --- no event without an actual state change ------------------------------


def test_repeated_occupied_frame_emits_no_further_event():
    tracker = SeatOccupancyTracker(["seat_3"])
    tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=1))
    events = tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=2))
    assert events == []


def test_repeated_empty_frame_emits_no_event():
    tracker = SeatOccupancyTracker(["seat_3"])
    events = tracker.update(_frame(frame_index=1))
    assert events == []


def test_different_chip_track_id_in_same_chip_zone_still_counts_as_occupied():
    # REQ-29 only cares that >=1 chip track is present, not which track.
    tracker = SeatOccupancyTracker(["seat_3"])
    tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=1))
    events = tracker.update(_frame(_chip_zone(2, "seat_3"), frame_index=2))
    assert events == []


# --- AC-15: player_area alone does not count as occupancy -----------------


def test_chip_in_player_area_only_does_not_occupy_seat():
    tracker = SeatOccupancyTracker(["seat_3"])
    events = tracker.update(_frame(_player_area(1, "seat_3"), frame_index=1))
    assert events == []
    assert tracker.snapshot() == {"seat_3": False}


def test_chip_zone_to_player_area_only_emits_vacated():
    tracker = SeatOccupancyTracker(["seat_3"])
    tracker.update(_frame(_chip_zone(1, "seat_3"), frame_index=1))
    events = tracker.update(_frame(_player_area(1, "seat_3"), frame_index=2))
    assert len(events) == 1
    assert isinstance(events[0], SeatVacatedEvent)


# --- multiple seats tracked independently ----------------------------------


def test_seats_tracked_independently():
    tracker = SeatOccupancyTracker(["seat_1", "seat_2"])
    events = tracker.update(_frame(_chip_zone(1, "seat_1"), frame_index=1))
    assert len(events) == 1
    assert events[0].seat == "seat_1"
    assert tracker.snapshot() == {"seat_1": True, "seat_2": False}


def test_two_seats_changing_in_same_frame_both_emit():
    tracker = SeatOccupancyTracker(["seat_1", "seat_2"])
    events = tracker.update(
        _frame(_chip_zone(1, "seat_1"), _chip_zone(2, "seat_2"), frame_index=1)
    )
    assert {e.seat for e in events} == {"seat_1", "seat_2"}
    assert all(isinstance(e, SeatOccupiedEvent) for e in events)


# --- sequence numbering (REQ-33) -------------------------------------------


def test_sequence_is_monotonic_across_updates():
    tracker = SeatOccupancyTracker(["seat_1"])
    first = tracker.update(_frame(_chip_zone(1, "seat_1"), frame_index=1))
    second = tracker.update(_frame(frame_index=2))
    assert first[0].sequence == 0
    assert second[0].sequence == 1


# --- occupancy requires an actual chip track, not just the zone label ------


def test_non_chip_assignment_labeled_chip_zone_does_not_count():
    # ZoneAssignment doesn't itself tie zone to object_class, so a
    # schema-valid but pipeline-impossible CHIP_ZONE hit from a different
    # class must not count as occupancy (REQ-29 requires a chip track).
    tracker = SeatOccupancyTracker(["seat_3"])
    bogus = _assignment(1, DetectionClass.DEALER_BUTTON, ZoneKind.CHIP_ZONE, "seat_3")
    events = tracker.update(_frame(bogus, frame_index=1))
    assert events == []
    assert tracker.snapshot() == {"seat_3": False}


# --- unknown seat is a hard error, not a silent no-op ----------------------


def test_unknown_seat_id_in_assignment_raises():
    tracker = SeatOccupancyTracker(["seat_1"])
    with pytest.raises(ValueError, match="seat_9"):
        tracker.update(_frame(_chip_zone(1, "seat_9"), frame_index=1))


# --- snapshot ---------------------------------------------------------------


def test_snapshot_reflects_current_state():
    tracker = SeatOccupancyTracker(["seat_1", "seat_2"])
    tracker.update(_frame(_chip_zone(1, "seat_1"), frame_index=1))
    assert tracker.snapshot() == {"seat_1": True, "seat_2": False}
