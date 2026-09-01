"""REQ-37: debug overlay rendering (AC-24's "Zonen, Track-IDs, Gummiband, State")."""

from __future__ import annotations

import numpy as np

from poker_vision.assignment.models import (
    ASSIGNMENT_SCHEMA_VERSION,
    FrameAssignments,
    ZoneAssignment,
    ZoneKind,
)
from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Resolution
from poker_vision.debug.overlay import render_overlay
from poker_vision.detection.models import DetectionClass
from poker_vision.state.snapshot import SeatOccupancy, StateSnapshot
from poker_vision.tracking.models import TrackedFrame, TrackedObject

# Identity homography + zero distortion: a table point (x, y) projects back
# to pixel point (x, y) exactly (see detection/geometry.py -- undistort then
# redistort round-trips through the same camera matrix when distortion is
# zero). This lets tests assert on specific pixel coordinates without
# involving real camera calibration math, the same trick test_assignment.py
# uses for its own calibration fixture.
_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

_IMAGE_SIZE = 400


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


def _seat(seat_id: str, player_area: TablePolygon, chip_zone: TablePolygon) -> CalibrationSeat:
    return CalibrationSeat(
        seat_id=seat_id, zones=SeatZones(player_area=player_area, chip_zone=chip_zone)
    )


# Kept below y=100 so nothing here collides with the state-text panel drawn
# across the top of the frame (see draw_state's opaque background, which is
# deliberately topmost and thus paints over anything underneath it).
SEAT_1 = _seat(
    "seat_1",
    _polygon((0, 150), (100, 150), (100, 250), (0, 250)),
    _polygon((10, 160), (50, 160), (50, 200), (10, 200)),
)
BOARD_ZONE = _polygon((150, 300), (250, 300), (250, 350), (150, 350))
DEALER_AREA = _polygon((300, 300), (350, 300), (350, 350), (300, 350))

CALIBRATION = CalibrationRuntime(
    schema_version="1.1",
    table_id="test_table",
    based_on="test",
    inference_resolution=Resolution(width=1920, height=1080),
    camera=CameraIntrinsics(fx=1400.0, fy=1400.0, cx=960.0, cy=540.0),
    distortion=DistortionCoefficients(),
    homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
    table=TableDimensions(width=1200.0, height=900.0, unit=TableUnit.MM),
    seats=[SEAT_1],
    zones=GlobalZones(board_zone=BOARD_ZONE, dealer_area=DEALER_AREA),
    card_dealer_seat_id="seat_1",
)


def _blank_image() -> np.ndarray:
    return np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)


def _track(track_id: int, object_class: DetectionClass, x: float, y: float) -> TrackedObject:
    return TrackedObject(
        track_id=track_id, object_class=object_class, confidence=0.9, center=TablePoint(x=x, y=y)
    )


def _tracked_frame(*tracks: TrackedObject) -> TrackedFrame:
    return TrackedFrame(schema_version="1.0", frame_index=0, tracks=list(tracks))


def _assignment(
    track_id: int, object_class: DetectionClass, zone: ZoneKind, seat_id: str | None
) -> ZoneAssignment:
    return ZoneAssignment(
        schema_version=ASSIGNMENT_SCHEMA_VERSION,
        track_id=track_id,
        object_class=object_class,
        zone=zone,
        seat_id=seat_id,
    )


def _frame_assignments(*assignments: ZoneAssignment) -> FrameAssignments:
    return FrameAssignments(
        schema_version=ASSIGNMENT_SCHEMA_VERSION, frame_index=0, assignments=list(assignments)
    )


def _snapshot(**overrides) -> StateSnapshot:
    defaults = dict(
        schema_version="1.0",
        sequence=0,
        timestamp="2026-01-01T00:00:00Z",
        frame_index=0,
        seats=[SeatOccupancy(seat="seat_1", occupied=True)],
        dealer_seat="seat_1",
        hand_id=1,
        street=None,
        hand_active=True,
    )
    defaults.update(overrides)
    return StateSnapshot(**defaults)


# --- render_overlay: does not mutate the input frame ------------------------


def test_render_overlay_does_not_mutate_input_frame():
    image = _blank_image()
    original = image.copy()
    render_overlay(image, CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot())
    assert np.array_equal(image, original)


def test_render_overlay_returns_a_different_array_same_shape():
    image = _blank_image()
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot()
    )
    assert annotated is not image
    assert annotated.shape == image.shape


# --- AC-24: zones from calibration -------------------------------------------


def test_zones_are_drawn_onto_the_frame():
    image = _blank_image()
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot(seats=[])
    )
    # seat_1's player_area boundary runs along x=0..100 at y=150 -- with
    # identity homography/zero distortion this is exactly pixel (50, 150).
    assert annotated[150, 50].any()
    # board_zone boundary at (150..250, 300).
    assert annotated[300, 200].any()


def test_no_zones_drawn_when_frame_is_otherwise_blank_stays_blank_elsewhere():
    image = _blank_image()
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot(seats=[])
    )
    # A point far from every zone, track or the state-text band stays untouched.
    assert not annotated[399, 399].any()


# --- AC-24: stable tracks with ID/class --------------------------------------


def test_track_marker_is_drawn_at_projected_position():
    image = _blank_image()
    track = _track(7, DetectionClass.CHIP, 30, 180)
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(track), _frame_assignments(), _snapshot(seats=[])
    )
    assert annotated[180, 30].any()


# --- AC-24: rubber-band track -> seat line -----------------------------------


def test_rubber_band_line_drawn_for_seat_assigned_track():
    image = _blank_image()
    # Chip sits in seat_1's chip_zone, near corner (10, 160).
    track = _track(1, DetectionClass.CHIP, 12, 162)
    assignment = _assignment(1, DetectionClass.CHIP, ZoneKind.CHIP_ZONE, "seat_1")
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(track), _frame_assignments(assignment), _snapshot()
    )
    # seat_1's player_area centroid is (50, 200) (rectangle (0,150)-(100,250)).
    # The line from (12, 162) to (50, 200) passes through its own midpoint.
    midpoint = (31, 181)
    assert annotated[midpoint[1], midpoint[0]].any()


def test_no_rubber_band_line_for_unassigned_track():
    image_with_track_only = render_overlay(
        _blank_image(),
        CALIBRATION,
        _tracked_frame(_track(1, DetectionClass.CARD, 200, 175)),
        _frame_assignments(),  # a board_zone card assignment never carries a seat_id
        _snapshot(seats=[]),
    )
    image_without_track = render_overlay(
        _blank_image(), CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot(seats=[])
    )
    # Region strictly between the card and any seat centroid: identical with
    # or without the track, since a seat-less assignment draws no line there.
    region_with = image_with_track_only[350:360, 350:360]
    region_without = image_without_track[350:360, 350:360]
    assert np.array_equal(region_with, region_without)


# --- AC-24: current occupancy/dealer/street state ----------------------------


def test_state_text_block_is_drawn_at_top_left():
    image = _blank_image()
    annotated = render_overlay(
        image, CALIBRATION, _tracked_frame(), _frame_assignments(), _snapshot()
    )
    # The state panel's background is deliberately black (same as the blank
    # canvas) so it blends in when nothing is drawn under it; its white text
    # glyphs are what actually needs to show up somewhere in that band.
    assert annotated[0:100, :].any()


def test_state_text_reflects_snapshot_fields():
    from poker_vision.debug.overlay import _state_lines

    lines = _state_lines(_snapshot())
    joined = " ".join(lines)
    assert "hand 1" in joined
    assert "active" in joined
    assert "seat_1" in joined
    assert "occupied" in joined


def test_state_text_handles_no_active_hand():
    from poker_vision.debug.overlay import _state_lines

    lines = _state_lines(
        _snapshot(hand_id=None, hand_active=False, dealer_seat=None, street=None)
    )
    joined = " ".join(lines)
    assert "no hand" in joined
    assert "inactive" in joined
