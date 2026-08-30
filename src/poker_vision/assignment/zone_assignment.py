"""Point-in-polygon zone assignment for stable tracks (REQ-26, REQ-28).

`assign_zones` is the one entry point: it takes a `HysteresisFilter.
update()` result (REQ-25's "nur bestätigte Tracks") and the calibration's
zone polygons, and maps each track to at most one zone via
`calibration.topology.point_in_polygon` — no sampling, no distance
thresholds, purely "is this point inside that polygon" (nearest-seat
fallback for an unmatched `dealer_button` is REQ-27, a separate, later
stage on top of this one's output).

Per REQ-26, which zones a class is tested against differs:

- `chip`: a seat's `chip_zone` first: since `chip_zone` is validated (REQ-11)
  to lie entirely within its own seat's `player_area`, a `chip_zone` hit
  always implies that seat's `player_area` too, so it is reported as the
  more specific `chip_zone` match rather than treating the two as separate
  candidates. Only when no seat's `chip_zone` contains the point does a
  seat's `player_area` become a candidate — this is what lets `state`
  (REQ-29) tell "in the zone that counts for occupancy" apart from
  "on the table in front of a seat, but not close enough" (AC-15).
- `card`: only the single global `board_zone`.
- `dealer_button`: a seat's `player_area` first (so a button sitting in
  front of a seat resolves straight to that seat), falling back to the
  global `dealer_area` when no seat's `player_area` contains it.

REQ-28 ("höchstens einer Zone") only has teeth at the per-class-of-zone
level above: a point can be inside more than one seat's zone only if seats'
zones happen to overlap (typically an authoring mistake, since REQ-11 only
forbids `chip_zone`-`chip_zone` and `board_zone`-`chip_zone` overlap, not
`player_area`-`player_area`). When that happens, the seat whose zone
centroid is nearest the track's position wins, and a warning is logged —
never a silent multi-assignment.
"""

from __future__ import annotations

import logging
import math

from poker_vision.assignment.models import (
    ASSIGNMENT_SCHEMA_VERSION,
    FrameAssignments,
    ZoneAssignment,
    ZoneKind,
)
from poker_vision.calibration.geometry import TablePoint, TablePolygon, polygon_signed_area
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.topology import point_in_polygon
from poker_vision.calibration.zones import CalibrationSeat
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking.models import TrackedFrame, TrackedObject

logger = logging.getLogger(__name__)


def _centroid(polygon: TablePolygon) -> TablePoint:
    """Area-weighted polygon centroid (REQ-28's "kleinste Zentroid-Distanz").

    Not the mean of `polygon.points`: a plain vertex average is only the
    true centroid for a regular polygon, and diverges from it for an
    irregular or concave one, or one with extra collinear vertices along a
    straight edge — any of which can flip which seat REQ-28's tie-break
    picks. `TablePolygon`'s own validator already rejects the zero-area
    case this formula would divide by (REQ-11).
    """
    points = polygon.points
    area = polygon_signed_area(points)
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        cross = a.x * b.y - b.x * a.y
        cx += (a.x + b.x) * cross
        cy += (a.y + b.y) * cross
    return TablePoint(x=cx / (6 * area), y=cy / (6 * area))


def _distance(a: TablePoint, b: TablePoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _matching_seats(
    seats: list[CalibrationSeat], center: TablePoint, zone: ZoneKind
) -> list[tuple[str, TablePolygon]]:
    polygon_of = (
        (lambda seat: seat.zones.chip_zone)
        if zone is ZoneKind.CHIP_ZONE
        else (lambda seat: seat.zones.player_area)
    )
    return [
        (seat.seat_id, polygon_of(seat))
        for seat in seats
        if point_in_polygon(polygon_of(seat), center)
    ]


def _resolve_seat(
    candidates: list[tuple[str, TablePolygon]],
    center: TablePoint,
    object_class: DetectionClass,
    track_id: int,
    zone: ZoneKind,
) -> str:
    """REQ-28: at most one zone per track.

    A multi-seat hit picks the candidate whose zone centroid is nearest the
    track's position and logs a warning — this can only happen when two
    seats' zones overlap.
    """
    if len(candidates) == 1:
        return candidates[0][0]
    ranked = sorted(candidates, key=lambda candidate: _distance(center, _centroid(candidate[1])))
    logger.warning(
        "track %d (%s) matched %d seats' %s (%s); assigning nearest seat '%s' by centroid distance",
        track_id,
        object_class.value,
        len(candidates),
        zone.value,
        ", ".join(seat_id for seat_id, _ in candidates),
        ranked[0][0],
    )
    return ranked[0][0]


def _assign_chip(track: TrackedObject, calibration: CalibrationRuntime) -> ZoneAssignment | None:
    chip_zone_hits = _matching_seats(calibration.seats, track.center, ZoneKind.CHIP_ZONE)
    if chip_zone_hits:
        zone = ZoneKind.CHIP_ZONE
    else:
        chip_zone_hits = _matching_seats(calibration.seats, track.center, ZoneKind.PLAYER_AREA)
        zone = ZoneKind.PLAYER_AREA
    if not chip_zone_hits:
        return None
    seat_id = _resolve_seat(chip_zone_hits, track.center, track.object_class, track.track_id, zone)
    return ZoneAssignment(
        schema_version=ASSIGNMENT_SCHEMA_VERSION,
        track_id=track.track_id,
        object_class=track.object_class,
        zone=zone,
        seat_id=seat_id,
    )


def _assign_card(track: TrackedObject, calibration: CalibrationRuntime) -> ZoneAssignment | None:
    if not point_in_polygon(calibration.zones.board_zone, track.center):
        return None
    return ZoneAssignment(
        schema_version=ASSIGNMENT_SCHEMA_VERSION,
        track_id=track.track_id,
        object_class=track.object_class,
        zone=ZoneKind.BOARD_ZONE,
        seat_id=None,
    )


def _assign_dealer_button(
    track: TrackedObject, calibration: CalibrationRuntime
) -> ZoneAssignment | None:
    player_area_hits = _matching_seats(calibration.seats, track.center, ZoneKind.PLAYER_AREA)
    if player_area_hits:
        seat_id = _resolve_seat(
            player_area_hits, track.center, track.object_class, track.track_id, ZoneKind.PLAYER_AREA
        )
        return ZoneAssignment(
            schema_version=ASSIGNMENT_SCHEMA_VERSION,
            track_id=track.track_id,
            object_class=track.object_class,
            zone=ZoneKind.PLAYER_AREA,
            seat_id=seat_id,
        )
    if not point_in_polygon(calibration.zones.dealer_area, track.center):
        return None
    return ZoneAssignment(
        schema_version=ASSIGNMENT_SCHEMA_VERSION,
        track_id=track.track_id,
        object_class=track.object_class,
        zone=ZoneKind.DEALER_AREA,
        seat_id=None,
    )


_ASSIGN_BY_CLASS = {
    DetectionClass.CHIP: _assign_chip,
    DetectionClass.CARD: _assign_card,
    DetectionClass.DEALER_BUTTON: _assign_dealer_button,
}


def assign_zones(tracked_frame: TrackedFrame, calibration: CalibrationRuntime) -> FrameAssignments:
    """Map every stable track in `tracked_frame` to at most one zone (REQ-26).

    A track that lies in none of its class's candidate zones is simply
    absent from the result — there is no "unassigned" entry to carry, since
    absence from `assignments` already communicates that both to `state`
    (REQ-29 ff.) and to REQ-27's nearest-seat fallback, which only applies
    to a `dealer_button` track missing from this output.
    """
    assignments = [
        assignment
        for track in tracked_frame.tracks
        if (assignment := _ASSIGN_BY_CLASS[track.object_class](track, calibration)) is not None
    ]
    return FrameAssignments(
        schema_version=ASSIGNMENT_SCHEMA_VERSION,
        frame_index=tracked_frame.frame_index,
        assignments=assignments,
    )
