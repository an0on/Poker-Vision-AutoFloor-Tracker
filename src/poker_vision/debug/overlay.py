"""Debug overlay rendering (REQ-37).

Draws, onto a copy of the raw captured frame: every seat/global zone from
the calibration, every stable track with its ID and class, a Phase-0-style
rubber-band line from each assigned track to its seat (with the distance,
in table units, as text -- REQ-0.7's "Distanz als Text" continued here),
and the current occupancy/dealer/street state from `PipelineStateMachine.
snapshot()`.

All zone/track geometry lives in table-plane coordinates (REQ-5); this is
one of only two places in the project that projects it back into pixel
space (the other is `detection/geometry.py`'s own pixel -> table
direction), via `HomographyMatrix.inverse` plus the calibration's camera/
distortion parameters -- see `apply_inverse_homography_to_point`'s
docstring, which names this exact use case.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from poker_vision.assignment.models import FrameAssignments
from poker_vision.calibration.geometry import TablePoint, TablePolygon, polygon_centroid
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat
from poker_vision.detection.geometry import apply_inverse_homography_to_point
from poker_vision.state.snapshot import StateSnapshot
from poker_vision.tracking.models import TrackedFrame

_COLOR_PLAYER_AREA = (200, 120, 0)  # BGR: blue
_COLOR_CHIP_ZONE = (0, 180, 0)  # green
_COLOR_BOARD_ZONE = (0, 200, 200)  # yellow
_COLOR_DEALER_AREA = (200, 0, 200)  # magenta
_COLOR_TRACK = (255, 255, 255)  # white
_COLOR_RUBBER_BAND = (0, 255, 255)  # yellow
_COLOR_STATE_TEXT = (255, 255, 255)  # white
_COLOR_STATE_BACKGROUND = (0, 0, 0)  # black

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LINE = cv2.LINE_AA
_TRACK_MARKER_RADIUS = 6
_STATE_LINE_HEIGHT = 18
_STATE_MARGIN = 8


def _to_pixel(point: TablePoint, calibration: CalibrationRuntime) -> tuple[int, int]:
    """Table-plane point -> raw (distorted) pixel coordinates, rounded for drawing."""
    pixel = apply_inverse_homography_to_point(
        point, calibration.homography, calibration.camera, calibration.distortion
    )
    return round(pixel.x), round(pixel.y)


def _polygon_to_pixels(
    polygon: TablePolygon, calibration: CalibrationRuntime
) -> np.ndarray:
    return np.array(
        [_to_pixel(point, calibration) for point in polygon.points], dtype=np.int32
    )


def _draw_zone(
    image: np.ndarray,
    polygon: TablePolygon,
    calibration: CalibrationRuntime,
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = _polygon_to_pixels(polygon, calibration)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2, lineType=_LINE)
    anchor = points[0]
    cv2.putText(
        image, label, (int(anchor[0]), int(anchor[1]) - 4), _FONT, _FONT_SCALE, color,
        _FONT_THICKNESS, _LINE,
    )


def draw_zones(image: np.ndarray, calibration: CalibrationRuntime) -> None:
    """Draw every seat's `player_area`/`chip_zone` plus the global `board_zone`/`dealer_area`."""
    for seat in calibration.seats:
        _draw_zone(
            image, seat.zones.player_area, calibration, _COLOR_PLAYER_AREA,
            f"{seat.seat_id}:player_area",
        )
        _draw_zone(
            image, seat.zones.chip_zone, calibration, _COLOR_CHIP_ZONE, f"{seat.seat_id}:chip_zone"
        )
    _draw_zone(image, calibration.zones.board_zone, calibration, _COLOR_BOARD_ZONE, "board_zone")
    _draw_zone(image, calibration.zones.dealer_area, calibration, _COLOR_DEALER_AREA, "dealer_area")


def draw_tracks(
    image: np.ndarray, tracked_frame: TrackedFrame, calibration: CalibrationRuntime
) -> None:
    """Draw every stable track as a marker labeled with its ID and class."""
    for track in tracked_frame.tracks:
        x, y = _to_pixel(track.center, calibration)
        cv2.circle(image, (x, y), _TRACK_MARKER_RADIUS, _COLOR_TRACK, thickness=-1, lineType=_LINE)
        cv2.putText(
            image, f"#{track.track_id} {track.object_class.value}", (x + 8, y - 8), _FONT,
            _FONT_SCALE, _COLOR_TRACK, _FONT_THICKNESS, _LINE,
        )


def _seat_by_id(seats: list[CalibrationSeat], seat_id: str) -> CalibrationSeat:
    return next(seat for seat in seats if seat.seat_id == seat_id)


def draw_assignments(
    image: np.ndarray,
    tracked_frame: TrackedFrame,
    frame_assignments: FrameAssignments,
    calibration: CalibrationRuntime,
) -> None:
    """Draw a rubber-band line from each seat-assigned track to that seat.

    REQ-37, continuing Phase 0's REQ-0.7 rubber-band-plus-distance style.

    Only assignments carrying a `seat_id` draw a line (`chip_zone`/
    `player_area` for `chip`/`dealer_button`, and a `dealer_area` hit
    resolved by REQ-27's nearest-seat fallback) -- a `board_zone` card
    assignment never carries one (see `ZoneAssignment`), and is drawn as a
    track marker only.
    """
    tracks_by_id = {track.track_id: track for track in tracked_frame.tracks}
    for assignment in frame_assignments.assignments:
        if assignment.seat_id is None:
            continue
        track = tracks_by_id.get(assignment.track_id)
        if track is None:
            continue
        seat = _seat_by_id(calibration.seats, assignment.seat_id)
        anchor = polygon_centroid(seat.zones.player_area)
        start = _to_pixel(track.center, calibration)
        end = _to_pixel(anchor, calibration)
        cv2.line(image, start, end, _COLOR_RUBBER_BAND, thickness=2, lineType=_LINE)
        distance = math.hypot(track.center.x - anchor.x, track.center.y - anchor.y)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            image, f"{distance:.1f}", midpoint, _FONT, _FONT_SCALE, _COLOR_RUBBER_BAND,
            _FONT_THICKNESS, _LINE,
        )


def _state_lines(snapshot: StateSnapshot) -> list[str]:
    hand = f"hand {snapshot.hand_id}" if snapshot.hand_id is not None else "no hand"
    hand_status = "active" if snapshot.hand_active else "inactive"
    street = snapshot.street.value if snapshot.street is not None else "-"
    dealer = snapshot.dealer_seat if snapshot.dealer_seat is not None else "-"
    seats = ", ".join(
        f"{seat.seat}={'occupied' if seat.occupied else 'empty'}" for seat in snapshot.seats
    )
    return [
        f"frame {snapshot.frame_index}  seq {snapshot.sequence}",
        f"{hand} ({hand_status})  street {street}",
        f"dealer {dealer}",
        f"seats: {seats}" if seats else "seats: -",
    ]


def draw_state(image: np.ndarray, snapshot: StateSnapshot) -> None:
    """Draw occupancy/dealer/street state text from `PipelineStateMachine.snapshot()`."""
    lines = _state_lines(snapshot)
    height = _STATE_MARGIN * 2 + _STATE_LINE_HEIGHT * len(lines)
    width = image.shape[1]
    overlay = image[0:height, 0:width]
    cv2.rectangle(overlay, (0, 0), (width, height), _COLOR_STATE_BACKGROUND, thickness=-1)
    for index, line in enumerate(lines):
        y = _STATE_MARGIN + _STATE_LINE_HEIGHT * index + _STATE_LINE_HEIGHT // 2
        cv2.putText(
            image, line, (_STATE_MARGIN, y), _FONT, _FONT_SCALE, _COLOR_STATE_TEXT,
            _FONT_THICKNESS, _LINE,
        )


def render_overlay(
    frame_image: np.ndarray,
    calibration: CalibrationRuntime,
    tracked_frame: TrackedFrame,
    frame_assignments: FrameAssignments,
    snapshot: StateSnapshot,
) -> np.ndarray:
    """Render the full debug overlay onto a copy of `frame_image` (REQ-37, AC-24).

    Draw order: zones (background), rubber-band lines, track markers (on
    top of both), then the state text block (topmost, most important).
    Never mutates `frame_image` itself -- the caller (`MjpegDebugServer.
    _stream`, rendering on demand per connected client -- REQ-46) still
    owns that buffer for its own purposes (e.g. other export adapters
    reading the same `Frame`).
    """
    annotated = frame_image.copy()
    draw_zones(annotated, calibration)
    draw_assignments(annotated, tracked_frame, frame_assignments, calibration)
    draw_tracks(annotated, tracked_frame, calibration)
    draw_state(annotated, snapshot)
    return annotated
