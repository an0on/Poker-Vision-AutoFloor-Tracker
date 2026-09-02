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
authoring's `homography` is an identity mapping, solved from the photo's
own 4 corners -- see `_identity_homography_from_image_corners`. `dealer_area`
(REQ-7's "Action Area", the inner-oval region) is likewise not clicked
separately: `infer_inner_boundary_polygon` derives it from the seat
polygons the operator already clicked, since two manual arc-center clicks
per oval (originally REQ-10a's design) turned out to be a fragile way to
mark a precise curve by hand in practice -- a wrongly-placed center click
produces a wildly wrong radius with no way to visually sanity-check it
before the fact, whereas both derivations here only ever use points
already validated as real seat corners.
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
_MIN_SEATS = 3
# Numerically arbitrary (undistortion is an exact identity when distortion
# is all-zero, see `undistort.py`) -- kept only for a plausible-looking
# authoring file; real intrinsics were never measured for this rig.
_PLACEHOLDER_FOCAL_LENGTH = 1400.0


@dataclass(frozen=True, slots=True)
class MarkedZones:
    """Raw click output for one reference photo, before derivation.

    `seat_polygons` is keyed by whatever the UI used to identify a click
    session (e.g. click order) -- deliberately *not* assumed to already be
    in clockwise physical order; `number_seats_clockwise` derives that
    itself from each polygon's centroid, so a scrambled click order can't
    silently mis-number seats. There is no separate inner/outer-oval click
    data: `build_authoring_from_marked_zones` derives both `dealer_area`
    and the homography from `seat_polygons`/`image_size` alone (see this
    module's docstring for why).
    """

    seat_polygons: dict[str, list[Point]]
    dealer_seat_key: str
    board_zone_points: list[Point]
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


def infer_inner_boundary_polygon(seat_polygons: dict[str, list[Point]]) -> list[Point]:
    """Derive `dealer_area` (REQ-7's "Action Area") from the already-clicked
    seat wedges, instead of a separate manual oval-click step.

    For each seat, its two corners closest to the table's overall centroid
    are the ones facing the board -- an adjacent seat's own closest pair
    meets that same boundary, since neighbouring wedges share it. Collecting
    every seat's closest pair and sorting all of them by angle around the
    table centroid (the same technique `number_seats_clockwise` uses to
    order seats) produces a simple, star-shaped polygon hugging the true
    inner boundary -- correct for any seat point count (>= 3, REQ-10a's
    per-seat minimum) or click winding, and immune to the "manually click a
    circle's center" failure mode that motivated dropping the oval-click
    steps in the first place (a slightly mis-clicked corner only nudges the
    boundary a little; a mis-clicked arc-center could blow the whole curve
    up to any radius).
    """
    centroids = [_polygon_centroid(points) for points in seat_polygons.values()]
    table_centroid = (
        sum(c[0] for c in centroids) / len(centroids),
        sum(c[1] for c in centroids) / len(centroids),
    )

    inner_points: list[Point] = []
    for points in seat_polygons.values():
        closest_two = sorted(points, key=lambda p: _dist(p, table_centroid))[:2]
        inner_points.extend(closest_two)

    def angle(point: Point) -> float:
        return math.atan2(point[1] - table_centroid[1], point[0] - table_centroid[0])

    return sorted(inner_points, key=angle)


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


def _identity_homography_from_image_corners(
    width: float, height: float
) -> HomographyCorrespondences:
    """Correspondence points for `calib compile` to solve (REQ-9).

    Table coordinates equal photo pixel coordinates here (see module
    docstring), so `image_point == table_point` for every correspondence --
    the photo's own 4 corners already satisfy `cv2.findHomography`'s
    minimum-4-points/non-degenerate requirement and solve to the exact
    identity matrix (zero residual, since every correspondence already
    satisfies image_point == table_point) regardless of which 4 points are
    used. There is nothing an operator could usefully click here; a
    previous design solved this from an operator-clicked outer oval
    instead, which existed for this purpose alone.
    """
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return HomographyCorrespondences(
        points=[
            HomographyPointCorrespondence(
                image_point={"x": x, "y": y}, table_point={"x": x, "y": y}
            )
            for x, y in corners
        ]
    )


def build_authoring_from_marked_zones(
    marked: MarkedZones,
    table_id: str,
    *,
    chip_zone_shrink_factor: float = DEFAULT_CHIP_ZONE_SHRINK_FACTOR,
) -> CalibrationAuthoring:
    """Assemble a full `CalibrationAuthoring` from one marking session.

    Raises `ValueError`/`pydantic.ValidationError` for anything REQ-11
    would reject anyway (degenerate polygons, chip_zone escaping its
    player_area, ...) -- there is no separate "skip validation" path here,
    same as every other `calib` entry point (see `cli.py`'s docstring).

    `chip_zone_shrink_factor` must be in `(0, 1]`: outside that range this
    isn't a "shrink" at all -- 0 collapses every chip_zone to a single
    point (rejected downstream anyway, as a degenerate `TablePolygon`), a
    negative factor reflects each polygon through its own centroid (for a
    convex seat wedge, that reflection can still land fully inside
    `player_area`, so REQ-11 alone wouldn't catch it -- a "valid-looking"
    chip_zone on the wrong side of the seat), and anything above 1 expands
    rather than shrinks, which REQ-11 does catch, but only as a confusing
    "chip_zone not contained in player_area" error instead of a clear one
    naming the actual mistake.
    """
    if not (0 < chip_zone_shrink_factor <= 1):
        raise ValueError(
            f"chip_zone_shrink_factor must be in (0, 1], got {chip_zone_shrink_factor}"
        )
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

    dealer_area_points = infer_inner_boundary_polygon(marked.seat_polygons)
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
        homography=_identity_homography_from_image_corners(float(width), float(height)),
        table={"width": float(width), "height": float(height), "unit": TableUnit.MM},
        seats=seats,
        zones=GlobalZones(board_zone=board_zone, dealer_area=dealer_area),
        card_dealer_seat_id=seat_ids[marked.dealer_seat_key],
    )
