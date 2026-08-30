"""REQ-26/REQ-28: point-in-polygon zone assignment for stable tracks (AC-15)."""

from __future__ import annotations

import logging

import pytest

from poker_vision.assignment import zone_assignment as zone_assignment_module
from poker_vision.assignment.models import ZoneAssignment, ZoneKind
from poker_vision.assignment.zone_assignment import apply_dealer_nearest_seat_fallback, assign_zones
from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Resolution
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking.models import TrackedFrame, TrackedObject

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


def _seat(seat_id: str, player_area: TablePolygon, chip_zone: TablePolygon) -> CalibrationSeat:
    return CalibrationSeat(
        seat_id=seat_id, zones=SeatZones(player_area=player_area, chip_zone=chip_zone)
    )


def _calibration(
    seats: list[CalibrationSeat], board_zone: TablePolygon, dealer_area: TablePolygon
) -> CalibrationRuntime:
    return CalibrationRuntime(
        schema_version="1.0",
        table_id="test_table",
        based_on="test",
        inference_resolution=Resolution(width=1920, height=1080),
        camera=CameraIntrinsics(fx=1400.0, fy=1400.0, cx=960.0, cy=540.0),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=TableDimensions(width=1200.0, height=900.0, unit=TableUnit.MM),
        seats=seats,
        zones=GlobalZones(board_zone=board_zone, dealer_area=dealer_area),
    )


# seat_3: player_area (0,0)-(100,100), chip_zone (10,10)-(50,50) -- AC-15's
# "Chip in chip_zone Seat 3" case. seat_1 is a second, disjoint seat so
# multi-seat scenarios (e.g. "no zone matches at all") have something to not
# match against.
SEAT_3 = _seat(
    "seat_3",
    _polygon((0, 0), (100, 0), (100, 100), (0, 100)),
    _polygon((10, 10), (50, 10), (50, 50), (10, 50)),
)
SEAT_1 = _seat(
    "seat_1",
    _polygon((200, 0), (300, 0), (300, 100), (200, 100)),
    _polygon((210, 10), (250, 10), (250, 50), (210, 50)),
)
BOARD_ZONE = _polygon((400, 400), (600, 400), (600, 500), (400, 500))
DEALER_AREA = _polygon((700, 700), (750, 700), (750, 750), (700, 750))

CALIBRATION = _calibration([SEAT_3, SEAT_1], BOARD_ZONE, DEALER_AREA)


def _track(
    track_id: int, object_class: DetectionClass, x: float, y: float, confidence: float = 0.9
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        confidence=confidence,
        center=TablePoint(x=x, y=y),
    )


def _frame(*tracks: TrackedObject, frame_index: int = 0) -> TrackedFrame:
    return TrackedFrame(schema_version="1.0", frame_index=frame_index, tracks=list(tracks))


def _only(frame_assignments) -> ZoneAssignment:
    assert len(frame_assignments.assignments) == 1
    return frame_assignments.assignments[0]


# --- AC-15 ---------------------------------------------------------------


def test_chip_in_chip_zone_assigns_that_seat():
    result = assign_zones(_frame(_track(1, DetectionClass.CHIP, 30, 30)), CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.CHIP_ZONE
    assert assignment.seat_id == "seat_3"


def test_chip_in_player_area_outside_chip_zone_is_player_area_not_chip_zone():
    # (70, 70) is in seat_3's player_area but outside its chip_zone
    # ([10,50]x[10,50]) -- REQ-26 still reports the player_area match (so
    # `state`/REQ-29 has something to reason about), but AC-15 requires
    # that this is distinguishable from a chip_zone hit ("keine Occupancy").
    result = assign_zones(_frame(_track(1, DetectionClass.CHIP, 70, 70)), CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.PLAYER_AREA
    assert assignment.seat_id == "seat_3"


def test_chip_outside_every_zone_is_unassigned():
    result = assign_zones(_frame(_track(1, DetectionClass.CHIP, 150, 150)), CALIBRATION)
    assert result.assignments == []


def test_card_in_board_zone_assigns_board_zone():
    result = assign_zones(_frame(_track(1, DetectionClass.CARD, 500, 450)), CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.BOARD_ZONE
    assert assignment.seat_id is None


def test_card_outside_board_zone_is_unassigned():
    result = assign_zones(_frame(_track(1, DetectionClass.CARD, 30, 30)), CALIBRATION)
    assert result.assignments == []


# --- dealer_button (REQ-26) ------------------------------------------------


def test_dealer_button_in_seat_player_area_assigns_that_seat():
    result = assign_zones(_frame(_track(1, DetectionClass.DEALER_BUTTON, 70, 70)), CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.PLAYER_AREA
    assert assignment.seat_id == "seat_3"


def test_dealer_button_in_dealer_area_outside_any_seat_assigns_dealer_area():
    # dealer_area (REQ-7) belongs to no seat, so this still carries no
    # seat_id -- but REQ-26 requires reporting the dealer_area hit in its
    # own right, distinct from matching nothing at all. Resolving it to a
    # seat is REQ-27's (not yet implemented) job, not this stage's.
    result = assign_zones(_frame(_track(1, DetectionClass.DEALER_BUTTON, 720, 720)), CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.DEALER_AREA
    assert assignment.seat_id is None


def test_dealer_button_outside_every_zone_is_unassigned():
    result = assign_zones(_frame(_track(1, DetectionClass.DEALER_BUTTON, 150, 150)), CALIBRATION)
    assert result.assignments == []


# --- dealer_button nearest-seat fallback (REQ-27, AC-16) -------------------

# (130, 50) is in neither seat's player_area nor dealer_area (so REQ-26
# alone leaves the track absent from `assignments`). seat_3's player_area
# centroid is (50, 50) (distance 80), seat_1's is (250, 50) (distance 120)
# -- seat_3 must win as the smaller euclidean distance (AC-16).
_OUTSIDE_ALL_ZONES_NEAR_SEAT_3 = (130, 50)
_DISTANCE_TO_SEAT_3 = 80.0
_DISTANCE_TO_SEAT_1 = 120.0


def test_fallback_assigns_nearest_seat_when_button_matched_no_zone_and_under_threshold():
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, *_OUTSIDE_ALL_ZONES_NEAR_SEAT_3))
    assignments = assign_zones(frame, CALIBRATION)
    assert assignments.assignments == []  # REQ-26 alone: unassigned

    result = apply_dealer_nearest_seat_fallback(
        frame, assignments, CALIBRATION, max_distance=_DISTANCE_TO_SEAT_3 + 1
    )
    assignment = _only(result)
    assert assignment.zone == ZoneKind.DEALER_AREA
    assert assignment.seat_id == "seat_3"


def test_fallback_leaves_button_unassigned_when_matched_no_zone_and_over_threshold():
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, *_OUTSIDE_ALL_ZONES_NEAR_SEAT_3))
    assignments = assign_zones(frame, CALIBRATION)

    result = apply_dealer_nearest_seat_fallback(
        frame, assignments, CALIBRATION, max_distance=_DISTANCE_TO_SEAT_3 - 1
    )
    assert result.assignments == []


def test_fallback_picks_seat_with_smaller_euclidean_distance():
    # AC-16's "Testbild mit zwei Kandidaten-Seats" -- both seats are within
    # threshold, so this isolates the tie-break: the nearer one must win.
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, *_OUTSIDE_ALL_ZONES_NEAR_SEAT_3))
    assignments = assign_zones(frame, CALIBRATION)

    result = apply_dealer_nearest_seat_fallback(
        frame, assignments, CALIBRATION, max_distance=_DISTANCE_TO_SEAT_1 + 1
    )
    assignment = _only(result)
    assert assignment.seat_id == "seat_3"


def test_fallback_assigns_nearest_seat_to_seatless_dealer_area_hit_when_under_threshold():
    # (720, 720) is inside dealer_area but no seat's player_area (REQ-26's
    # own test above) -- distance to seat_1's centroid (250, 50) is ~818.24.
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, 720, 720))
    assignments = assign_zones(frame, CALIBRATION)
    original = _only(assignments)
    assert original.zone == ZoneKind.DEALER_AREA
    assert original.seat_id is None

    result = apply_dealer_nearest_seat_fallback(frame, assignments, CALIBRATION, max_distance=850.0)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.DEALER_AREA
    assert assignment.seat_id == "seat_1"


def test_fallback_leaves_seatless_dealer_area_hit_unchanged_when_over_threshold():
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, 720, 720))
    assignments = assign_zones(frame, CALIBRATION)

    result = apply_dealer_nearest_seat_fallback(frame, assignments, CALIBRATION, max_distance=100.0)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.DEALER_AREA
    assert assignment.seat_id is None


def test_fallback_passes_through_dealer_button_already_assigned_a_seat():
    frame = _frame(_track(1, DetectionClass.DEALER_BUTTON, 70, 70))
    assignments = assign_zones(frame, CALIBRATION)
    original = _only(assignments)
    assert original.zone == ZoneKind.PLAYER_AREA
    assert original.seat_id == "seat_3"

    result = apply_dealer_nearest_seat_fallback(frame, assignments, CALIBRATION, max_distance=0.001)
    assert result.assignments == [original]


def test_fallback_ignores_non_dealer_button_tracks():
    frame = _frame(
        _track(1, DetectionClass.CHIP, 150, 150),  # unassigned by REQ-26, not a dealer_button
        _track(2, DetectionClass.DEALER_BUTTON, 70, 70),
    )
    assignments = assign_zones(frame, CALIBRATION)
    assert len(assignments.assignments) == 1  # only track 2

    result = apply_dealer_nearest_seat_fallback(
        frame, assignments, CALIBRATION, max_distance=1000.0
    )
    assert {a.track_id for a in result.assignments} == {2}


# --- multiple tracks per frame ---------------------------------------------


def test_multiple_tracks_assigned_independently():
    frame = _frame(
        _track(1, DetectionClass.CHIP, 30, 30),
        _track(2, DetectionClass.CARD, 500, 450),
        _track(3, DetectionClass.CHIP, 150, 150),  # unassigned
    )
    result = assign_zones(frame, CALIBRATION)
    by_track = {a.track_id: a for a in result.assignments}
    assert set(by_track) == {1, 2}
    assert by_track[1].seat_id == "seat_3"
    assert by_track[2].zone == ZoneKind.BOARD_ZONE


# --- REQ-28: at most one zone, deterministic tie-break on multi-seat hit ---

# seat_a and seat_b's player_areas deliberately overlap (REQ-11 only forbids
# chip_zone/chip_zone and board_zone/chip_zone overlap, not player_area/
# player_area) so a point in the overlap has two seat candidates.
SEAT_A = _seat(
    "seat_a",
    _polygon((0, 0), (60, 0), (60, 60), (0, 60)),
    _polygon((0, 0), (10, 0), (10, 10), (0, 10)),
)
SEAT_B = _seat(
    "seat_b",
    _polygon((35, 0), (100, 0), (100, 60), (35, 60)),
    _polygon((90, 0), (100, 0), (100, 10), (90, 10)),
)
OVERLAP_CALIBRATION = _calibration([SEAT_A, SEAT_B], BOARD_ZONE, DEALER_AREA)


def test_multi_seat_hit_picks_nearest_centroid_and_logs_warning(caplog):
    # (50, 30) is in both player_areas but neither chip_zone. seat_a's
    # player_area centroid is (30, 30) (distance 20), seat_b's is (67.5, 30)
    # (distance 17.5) -- seat_b must win.
    with caplog.at_level(logging.WARNING):
        result = assign_zones(_frame(_track(1, DetectionClass.CHIP, 50, 30)), OVERLAP_CALIBRATION)
    assignment = _only(result)
    assert assignment.zone == ZoneKind.PLAYER_AREA
    assert assignment.seat_id == "seat_b"
    assert any("seat_a" in message and "seat_b" in message for message in caplog.messages)


def test_zone_assignment_rejects_seat_id_on_global_zone():
    with pytest.raises(Exception):
        ZoneAssignment(
            schema_version="1.0",
            track_id=1,
            object_class=DetectionClass.CARD,
            zone=ZoneKind.BOARD_ZONE,
            seat_id="seat_3",
        )


# --- _centroid: area-weighted, not a vertex average (REQ-28) --------------


def test_centroid_of_rectangle_matches_vertex_average():
    # For a rectangle, the area-weighted centroid and the plain vertex
    # average coincide -- this is the case the old (wrong) implementation
    # got right, so the fix must not have broken it.
    rectangle = _polygon((0, 0), (10, 0), (10, 4), (0, 4))
    centroid = zone_assignment_module._centroid(rectangle)
    assert centroid.x == pytest.approx(5.0)
    assert centroid.y == pytest.approx(2.0)


def test_centroid_uses_area_weighting_not_vertex_average():
    # Triangle (0,0)-(0,12)-(4,0) with an extra vertex (2,6) added exactly
    # on the hypotenuse (collinear, no effect on the actual shape). Its true
    # (area-weighted) centroid is the average of the *triangle's own three*
    # vertices, (4/3, 4) -- but naively averaging all four listed points
    # gives (1.5, 4.5) instead. A centroid that shifts just from adding a
    # collinear vertex to an unchanged shape could flip REQ-28's nearest-seat
    # tie-break for reasons that have nothing to do with the polygon's
    # actual geometry.
    triangle_with_collinear_vertex = _polygon((0, 0), (0, 12), (2, 6), (4, 0))
    centroid = zone_assignment_module._centroid(triangle_with_collinear_vertex)
    assert centroid.x == pytest.approx(4 / 3)
    assert centroid.y == pytest.approx(4.0)


def test_zone_assignment_rejects_missing_seat_id_on_seat_zone():
    with pytest.raises(Exception):
        ZoneAssignment(
            schema_version="1.0",
            track_id=1,
            object_class=DetectionClass.CHIP,
            zone=ZoneKind.CHIP_ZONE,
            seat_id=None,
        )
