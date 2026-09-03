"""Pure geometry for `calib mark-zones` (REQ-10a): turn an operator's mouse
clicks on a reference photo into a `CalibrationAuthoring`.

Deliberately has no OpenCV window / mouse-callback code -- that lives in
`mark_zones_interactive.py` and just collects raw click coordinates before
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
own 4 corners -- see `_identity_homography_from_image_corners`.

`dealer_area` (REQ-7's "Action Area", the inner-oval region) IS a separate
manual click step -- an earlier design tried to derive it from the already
-clicked seat corners instead (closest-to-centroid corner pairs), but that
measure isn't reliably "inner vs. outer" on an elongated oval table with
uneven per-seat click density: a genuine rail corner on a seat far off the
table's long axis can measure numerically closer to the table centroid than
a different seat's own true inner corner. So `dealer_area` is a freehand
polyline the operator traces along the true printed inner-oval curve
(clicking arbitrarily many points, the same style already used for a
seat's own outer rail curve) -- not the older "3-point arc" scheme
(start/center/end): a center click placed even slightly wrong there
produces an arbitrarily wrong radius with no way to sanity-check it before
saving, which is what motivated dropping arc-parameter ovals in the first
place. A freehand trace has no such single fragile point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import ValidationError

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

# A small, generic pixel margin, not derived from any on-table printed
# feature (a proposed "size it off the DOPO POKER lettering's height" rule
# was rejected in favor of this -- table branding/print size isn't a stable
# thing to key a default off of across different physical table designs,
# where a fixed small pixel default travels fine). Zero on the rail-facing
# side always, regardless of this value -- see `_derive_chip_zone`.
DEFAULT_CHIP_ZONE_INSET_PIXELS = 10.0

# An edge counts as "rail" (gets zero chip_zone margin) only if its outward
# normal points within ~60 degrees of straight away from the table center --
# i.e. it's part of the curved/straight outer boundary the operator traced
# along the physical rail, however many points that took. Every other edge
# (both side edges facing a neighbouring seat, and the inner edge facing the
# action zone) necessarily falls outside that cone and gets inset instead --
# one classification handles "no margin to rail" and "minimal margin to
# neighbours/action zone" identically, without needing separate seat-
# adjacency data to tell "next to a neighbour" apart from "facing the board".
_RAIL_NORMAL_MIN_DOT = 0.5  # cos(60 degrees)

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
    silently mis-number seats. `inner_oval_points` is the operator's own
    freehand trace of the action-area boundary and is used for
    `dealer_area` directly, unmodified (see this module's docstring for
    why it isn't derived from `seat_polygons` instead).
    """

    seat_polygons: dict[str, list[Point]]
    dealer_seat_key: str
    inner_oval_points: list[Point]
    board_zone_points: list[Point]
    image_size: tuple[int, int]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(vector: Point) -> Point:
    length = math.hypot(vector[0], vector[1])
    if length == 0:
        raise ValueError("cannot normalize a zero-length vector")
    return (vector[0] / length, vector[1] / length)


def _polygon_centroid(points: list[Point]) -> Point:
    """Vertex-average centroid -- good enough for seat ordering/chip-zone
    derivation (unlike `geometry.polygon_centroid`'s area-weighted one,
    which `topology`'s REQ-11 checks need for exactness, both only need a
    point *roughly* in the middle).
    """
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _edge_outward_unit_normal(p1: Point, p2: Point, interior_reference: Point) -> Point:
    """The unit normal of edge `p1`->`p2` that points away from
    `interior_reference` (a point known to be inside the polygon, e.g. its
    own centroid) -- i.e. the edge's true *outward* normal, regardless of
    the polygon's winding order.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError(f"degenerate zero-length polygon edge at {p1}")
    candidate = (-dy / length, dx / length)
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    to_midpoint = (midpoint[0] - interior_reference[0], midpoint[1] - interior_reference[1])
    if _dot(candidate, to_midpoint) < 0:
        candidate = (-candidate[0], -candidate[1])
    return candidate


def _line_intersection(point_a: Point, normal_a: Point, point_b: Point, normal_b: Point) -> Point:
    """Where the two lines (each given as a point on it + its normal) cross.

    Falls back to `point_a` for (near-)parallel lines -- two adjacent edges
    of a finely-traced curve are frequently near-collinear (consecutive
    click points a curve's own tangent barely turns between), and both
    lines coincide almost exactly in that case, so any point on either is
    an equally good answer; there is no meaningfully "more correct" one.
    """
    det = normal_a[0] * normal_b[1] - normal_a[1] * normal_b[0]
    if abs(det) < 1e-9:
        return point_a
    rhs_a = _dot(point_a, normal_a)
    rhs_b = _dot(point_b, normal_b)
    x = (rhs_a * normal_b[1] - rhs_b * normal_a[1]) / det
    y = (normal_a[0] * rhs_b - normal_b[0] * rhs_a) / det
    return (x, y)


_MITER_LIMIT_FACTOR = 4.0


def _offset_vertex(
    vertex: Point,
    point_a: Point,
    normal_a: Point,
    offset_a: float,
    point_b: Point,
    normal_b: Point,
    offset_b: float,
) -> Point:
    """One vertex's new position: the two adjacent (possibly offset) edge
    lines' intersection, unless that miter point lands unreasonably far
    away (see `_derive_chip_zone`'s docstring) -- then the bevel fallback,
    the average of each edge's own straight-perpendicular offset of
    `vertex`.
    """
    miter = _line_intersection(point_a, normal_a, point_b, normal_b)
    max_offset = max(offset_a, offset_b)
    if max_offset > 0 and _dist(vertex, miter) > _MITER_LIMIT_FACTOR * max_offset:
        bevel_a = (vertex[0] - normal_a[0] * offset_a, vertex[1] - normal_a[1] * offset_a)
        bevel_b = (vertex[0] - normal_b[0] * offset_b, vertex[1] - normal_b[1] * offset_b)
        return ((bevel_a[0] + bevel_b[0]) / 2, (bevel_a[1] + bevel_b[1]) / 2)
    return miter


def _derive_chip_zone(
    points: list[Point], table_centroid: Point, inset_pixels: float
) -> list[Point]:
    """One seat's `chip_zone`: `points` (its `player_area`) with every
    non-rail edge's line pulled inward by `inset_pixels` and the rail
    edge(s) left untouched (see module docstring / `_RAIL_NORMAL_MIN_DOT`),
    each vertex then recomputed as the intersection of its two adjacent
    (possibly shifted) edge lines -- the standard mitered polygon-offset
    construction.

    Not a sequential "clip the whole polygon against each shifted edge in
    turn": a seat's rail or inner boundary is often traced with many short
    points along the real curve, and repeatedly clipping the *entire*
    polygon against every one of those many near-parallel shifted lines
    compounds -- each further clip can cut back into what the previous
    clip already produced, eroding far more than `inset_pixels` in total
    (confirmed on the real reference table's click data: some seats lost
    over 80% of their area to a nominal 10px inset). Recomputing each
    vertex from only its own two adjacent edges has no such compounding:
    a vertex between two untouched rail edges lands back exactly on its
    original position, and a run of many consecutive inset edges traces a
    properly offset curve through their pairwise intersections instead.

    That per-vertex intersection ("miter join", in stroke-rendering terms)
    has its own known failure mode at a sharp or reflex corner: real click
    traces have those too (confirmed on the real reference table -- a
    seat's two adjacent edges meeting at a tight angle), and the two
    offset lines there can cross arbitrarily far from the original vertex,
    the same spike a sharp miter join produces in vector graphics. Past a
    `_MITER_LIMIT_FACTOR` multiple of the offset itself, `_offset_vertex`
    falls back to a "bevel" (the average of each edge's own independently
    offset point) instead of the raw line intersection -- bounded, and
    exactly what every vector-graphics stroke renderer does for the same
    reason.

    Operates on the polygon's own edges directly rather than moving
    individual vertices toward some reference point: an earlier design
    shrank every vertex toward the seat's centroid uniformly, which eats
    into the rail-facing side just as much as every other side (the exact
    opposite of what a poker player playing this table needs, since chips
    are stacked right up against the rail).
    """
    seat_centroid = _polygon_centroid(points)
    outward = _unit((seat_centroid[0] - table_centroid[0], seat_centroid[1] - table_centroid[1]))

    n = len(points)
    edge_lines: list[tuple[Point, Point, float]] = []
    for i in range(n):
        p1, p2 = points[i], points[(i + 1) % n]
        normal = _edge_outward_unit_normal(p1, p2, seat_centroid)
        offset = 0.0 if _dot(normal, outward) > _RAIL_NORMAL_MIN_DOT else inset_pixels
        line_point = (p1[0] - normal[0] * offset, p1[1] - normal[1] * offset)
        edge_lines.append((line_point, normal, offset))

    chip_zone = []
    for i in range(n):
        point_a, normal_a, offset_a = edge_lines[i - 1]
        point_b, normal_b, offset_b = edge_lines[i]
        chip_zone.append(
            _offset_vertex(points[i], point_a, normal_a, offset_a, point_b, normal_b, offset_b)
        )
    return chip_zone


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
    chip_zone_inset_pixels: float = DEFAULT_CHIP_ZONE_INSET_PIXELS,
) -> CalibrationAuthoring:
    """Assemble a full `CalibrationAuthoring` from one marking session.

    Raises `ValueError`/`pydantic.ValidationError` for anything REQ-11
    would reject anyway (degenerate polygons, chip_zone escaping its
    player_area, ...) -- there is no separate "skip validation" path here,
    same as every other `calib` entry point (see `cli.py`'s docstring).

    `chip_zone_inset_pixels` must be `>= 0` -- a margin can't be negative
    (that would mean expanding *past* the seat's own player_area, which
    REQ-11 rejects anyway, but with a confusing "not contained" error
    instead of one naming the actual mistake). `0` is a legitimate value
    (chip_zone == player_area on every non-rail edge too), not rejected.
    """
    if chip_zone_inset_pixels < 0:
        raise ValueError(f"chip_zone_inset_pixels must be >= 0, got {chip_zone_inset_pixels}")

    seat_centroids = [_polygon_centroid(points) for points in marked.seat_polygons.values()]
    table_centroid = (
        sum(c[0] for c in seat_centroids) / len(seat_centroids),
        sum(c[1] for c in seat_centroids) / len(seat_centroids),
    )

    seat_ids = number_seats_clockwise(marked.seat_polygons, marked.dealer_seat_key)
    seats = []
    for key, points in marked.seat_polygons.items():
        try:
            chip_zone_points = _derive_chip_zone(points, table_centroid, chip_zone_inset_pixels)
            chip_zone = _to_table_polygon(chip_zone_points)
        except (ValueError, ValidationError) as exc:
            # The likeliest real-world cause (confirmed against the actual
            # reference table's click data): two adjacent player_area
            # points a few pixels apart -- an accidental double-click while
            # tracing -- are close enough that even a small inset swings
            # the derived corner past nearby geometry. Re-tracing that
            # seat without the near-duplicate point is the fix, not a
            # smaller inset (a globally smaller inset for every other,
            # cleanly-traced seat isn't the actual problem).
            raise ValueError(
                f"seat '{key}' ({seat_ids[key]}): chip_zone_inset_pixels="
                f"{chip_zone_inset_pixels} produces an invalid chip_zone ({exc}). "
                "Likely cause: two of this seat's player_area points are only a "
                "few pixels apart (an accidental double-click) -- re-trace this "
                "seat without the near-duplicate point."
            ) from exc
        seats.append(
            CalibrationSeat(
                seat_id=seat_ids[key],
                zones=SeatZones(player_area=_to_table_polygon(points), chip_zone=chip_zone),
            )
        )

    board_zone = _to_table_polygon(marked.board_zone_points)
    dealer_area = _to_table_polygon(marked.inner_oval_points)

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
