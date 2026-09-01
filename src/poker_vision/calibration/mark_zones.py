"""Pure geometry for `calib mark-zones` (REQ-10a): turn an operator's mouse
clicks on a reference photo into a `CalibrationAuthoring`.

Deliberately has no OpenCV window / mouse-callback code -- that lives in
`tools/mark_zones_cli.py` and just collects raw click coordinates before
calling into this module. Everything here is plain functions over tuples,
so it's unit-testable without a display.

Design choice: the reference photo's own pixel grid *is* the table
coordinate system (1 pixel == 1 nominal table unit, y-down, origin at the
photo's top-left). Real-world table measurements were deliberately dropped
(see PRD.md REQ-7's note) -- only the *relative* geometry seat assignment
and zone containment (REQ-11) actually depend on, and this sidesteps ever
having to physically measure the table. `calib learn-table` (REQ-10b)
later maps a live photo of any physically-identical table into this same
coordinate system via feature-matching homography composition, not by
re-measuring anything either.

Because table coordinates equal photo pixel coordinates here, the
authoring's `homography` is an identity mapping -- see
`_identity_homography_from_outer_oval`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from poker_vision.calibration.authoring import (
    CALIBRATION_AUTHORING_SCHEMA_VERSION,
    CalibrationAuthoring,
)
from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import (
    HomographyCorrespondences,
    HomographyPointCorrespondence,
)
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Resolution

Point = tuple[float, float]

DEFAULT_CHIP_ZONE_SHRINK_FACTOR = 0.5
DEFAULT_ARC_SAMPLE_COUNT = 16
_MIN_SEATS = 3
# Numerically arbitrary (undistortion is an exact identity when distortion
# is all-zero, see `undistort.py`) -- kept only for a plausible-looking
# authoring file; real intrinsics were never measured for this rig.
_PLACEHOLDER_FOCAL_LENGTH = 1400.0


@dataclass(frozen=True, slots=True)
class ArcClick:
    """One end-cap of an oval: tangent point, arc center, tangent point.

    `start`/`end` are the two points where the oval's curve meets the
    straight run on either long side (what the operator clicks first and
    last for this end); `center` is the circle's center. The arc between
    `start` and `end` is taken the long way around -- away from the other
    end's center -- since that's the physical curve, not the short way
    that would cut through the table's interior.
    """

    start: Point
    center: Point
    end: Point


@dataclass(frozen=True, slots=True)
class MarkedZones:
    """Raw click output for one reference photo, before derivation.

    `seat_polygons` is keyed by whatever the UI used to identify a click
    session (e.g. click order) -- deliberately *not* assumed to already be
    in clockwise physical order; `number_seats_clockwise` derives that
    itself from each polygon's centroid, so a scrambled click order can't
    silently mis-number seats.
    """

    seat_polygons: dict[str, list[Point]]
    dealer_seat_key: str
    board_zone_points: list[Point]
    inner_oval: tuple[ArcClick, ArcClick]
    outer_oval: tuple[ArcClick, ArcClick]
    image_size: tuple[int, int]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polygon_centroid(points: list[Point]) -> Point:
    """Vertex-average centroid -- good enough for seat ordering/chip-zone
    shrink (unlike `geometry.polygon_centroid`'s area-weighted one, which
    `topology`'s REQ-11 checks need for exactness, ordering by angle only
    needs a point *roughly* in the middle).
    """
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _shrink_toward_centroid(points: list[Point], factor: float) -> list[Point]:
    cx, cy = _polygon_centroid(points)
    return [(cx + factor * (x - cx), cy + factor * (y - cy)) for x, y in points]


def _arc_polyline(arc: ArcClick, away_from: Point, n: int) -> list[Point]:
    """Sample `n + 1` points along `arc`'s curve from `start` to `end`, the
    long way around (whichever of the two possible directions ends up
    farther, at its midpoint, from `away_from` -- the other end's center).
    """
    cx, cy = arc.center
    radius = (_dist(arc.center, arc.start) + _dist(arc.center, arc.end)) / 2.0
    if radius <= 1e-9:
        raise ValueError("degenerate arc: center coincides with its start/end point")
    angle_start = math.atan2(arc.start[1] - cy, arc.start[0] - cx)
    angle_end = math.atan2(arc.end[1] - cy, arc.end[0] - cx)

    def sample(going_positive: bool) -> list[Point]:
        diff = (angle_end - angle_start) % (2 * math.pi)
        if not going_positive:
            diff -= 2 * math.pi
        return [
            (
                cx + radius * math.cos(angle_start + diff * i / n),
                cy + radius * math.sin(angle_start + diff * i / n),
            )
            for i in range(n + 1)
        ]

    candidate_a = sample(True)
    candidate_b = sample(False)
    mid_a = candidate_a[len(candidate_a) // 2]
    mid_b = candidate_b[len(candidate_b) // 2]
    return candidate_a if _dist(mid_a, away_from) >= _dist(mid_b, away_from) else candidate_b


def build_oval_polygon(
    end_a: ArcClick, end_b: ArcClick, arc_samples: int = DEFAULT_ARC_SAMPLE_COUNT
) -> list[Point]:
    """The full stadium/capsule polygon from two 3-point end-cap arcs (see
    the click scheme described in PRD.md's REQ-10a): each end's own curve,
    sampled the long way around, joined by straight runs to the nearer
    tangent point of the other end -- not assumed to be axis-aligned, so a
    slightly non-rectangular (perspective-skewed) click session still
    closes into a simple polygon.
    """
    arc_a = _arc_polyline(end_a, away_from=end_b.center, n=arc_samples)
    arc_b = _arc_polyline(end_b, away_from=end_a.center, n=arc_samples)
    if _dist(arc_a[-1], end_b.start) > _dist(arc_a[-1], end_b.end):
        arc_b = list(reversed(arc_b))
    return arc_a + arc_b


def number_seats_clockwise(
    seat_polygons: dict[str, list[Point]], dealer_seat_key: str
) -> dict[str, str]:
    """Map each click-session key to a stable `seat_N` id (REQ-7).

    Clockwise starting at the seat right after `dealer_seat_key`, per
    PRD.md's REQ-7/REQ-10a. Ordering is derived from each polygon's own
    centroid relative to the table's overall centroid -- not from
    `seat_polygons`' dict/click order -- so it's correct even if the
    operator didn't click the ten wedges in physical order.

    Coordinates are pixel-convention (y-down, see `PixelPoint`'s
    docstring): increasing `atan2(dy, dx)` there is what a viewer sees as
    clockwise, the opposite of the usual y-up math convention.

    Every wedge gets a `seat_N`, the dealer/Kartengeber wedge included --
    it's the last one numerically (going clockwise, it's the seat right
    *before* wrapping back to `seat_1`), which for a 10-wedge table lands
    it on `seat_10` exactly as PRD.md's REQ-7 describes for the "10th
    player deals and plays" edge case. Whether that seat is *currently*
    played by a person is tournament/game state, decided elsewhere --
    calibration always gives it a real `seat_id` and geometry either way,
    since `card_dealer_seat_id` (REQ-11) must reference an existing seat.
    """
    if dealer_seat_key not in seat_polygons:
        raise ValueError(f"dealer_seat_key '{dealer_seat_key}' is not one of the marked seats")
    if len(seat_polygons) < _MIN_SEATS:
        raise ValueError(f"need at least {_MIN_SEATS} seats to number, got {len(seat_polygons)}")

    centroids = {key: _polygon_centroid(points) for key, points in seat_polygons.items()}
    table_cx = sum(p[0] for p in centroids.values()) / len(centroids)
    table_cy = sum(p[1] for p in centroids.values()) / len(centroids)

    def angle(key: str) -> float:
        x, y = centroids[key]
        return math.atan2(y - table_cy, x - table_cx)

    dealer_angle = angle(dealer_seat_key)

    def angular_offset_from_dealer(key: str) -> float:
        return (angle(key) - dealer_angle) % (2 * math.pi)

    ordered_others = sorted(
        (key for key in seat_polygons if key != dealer_seat_key), key=angular_offset_from_dealer
    )
    result = {key: f"seat_{i + 1}" for i, key in enumerate(ordered_others)}
    result[dealer_seat_key] = f"seat_{len(ordered_others) + 1}"
    return result


def _to_table_polygon(points: list[Point]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in points])


def _identity_homography_from_outer_oval(
    outer_oval: tuple[ArcClick, ArcClick],
) -> HomographyCorrespondences:
    """4 correspondence points for `calib compile` to solve (REQ-9).

    Table coordinates equal photo pixel coordinates here (see module
    docstring), so `image_point == table_point` for every correspondence;
    the outer oval's 4 already-clicked tangent points are reused as those
    correspondences rather than inventing new ones, since they're already
    well-spread reference points around the table.
    """
    end_a, end_b = outer_oval
    points = [end_a.start, end_a.end, end_b.start, end_b.end]
    return HomographyCorrespondences(
        points=[
            HomographyPointCorrespondence(
                image_point={"x": x, "y": y}, table_point={"x": x, "y": y}
            )
            for x, y in points
        ]
    )


def build_authoring_from_marked_zones(
    marked: MarkedZones,
    table_id: str,
    *,
    chip_zone_shrink_factor: float = DEFAULT_CHIP_ZONE_SHRINK_FACTOR,
    arc_samples: int = DEFAULT_ARC_SAMPLE_COUNT,
) -> CalibrationAuthoring:
    """Assemble a full `CalibrationAuthoring` from one marking session.

    Raises `ValueError`/`pydantic.ValidationError` for anything REQ-11
    would reject anyway (degenerate polygons, chip_zone escaping its
    player_area, ...) -- there is no separate "skip validation" path here,
    same as every other `calib` entry point (see `cli.py`'s docstring).
    """
    seat_ids = number_seats_clockwise(marked.seat_polygons, marked.dealer_seat_key)
    seats = [
        CalibrationSeat(
            seat_id=seat_ids[key],
            zones=SeatZones(
                player_area=_to_table_polygon(points),
                chip_zone=_to_table_polygon(
                    _shrink_toward_centroid(points, chip_zone_shrink_factor)
                ),
            ),
        )
        for key, points in marked.seat_polygons.items()
    ]

    dealer_area_points = build_oval_polygon(*marked.inner_oval, arc_samples=arc_samples)
    board_zone = _to_table_polygon(marked.board_zone_points)
    dealer_area = _to_table_polygon(dealer_area_points)

    width, height = marked.image_size
    resolution = Resolution(width=width, height=height)

    return CalibrationAuthoring(
        schema_version=CALIBRATION_AUTHORING_SCHEMA_VERSION,
        table_id=table_id,
        inference_resolution=resolution,
        camera=CameraIntrinsics(
            fx=_PLACEHOLDER_FOCAL_LENGTH,
            fy=_PLACEHOLDER_FOCAL_LENGTH,
            cx=width / 2.0,
            cy=height / 2.0,
        ),
        distortion=DistortionCoefficients(),
        homography=_identity_homography_from_outer_oval(marked.outer_oval),
        table={"width": float(width), "height": float(height), "unit": TableUnit.MM},
        seats=seats,
        zones=GlobalZones(board_zone=board_zone, dealer_area=dealer_area),
        card_dealer_seat_id=seat_ids[marked.dealer_seat_key],
    )
