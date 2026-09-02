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
from poker_vision.calibration.topology import polygon_contains, polygons_overlap
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Resolution

Point = tuple[float, float]

# Rough estimate of one felt-pattern diamond's width in the reference
# photo's own pixel grid -- table coordinates are photo pixels here (see
# module docstring), so this is a nominal per-photo default, not a
# universal constant; pass a measured value via `chip_zone_inset_pixels`
# once one is available for a given reference photo.
DEFAULT_CHIP_ZONE_INSET_PIXELS = 40.0
# See its one use in `_derive_chip_zone`: absorbs float64 rounding noise
# from chained polygon-clip line intersections, several orders of
# magnitude below anything visible at reference-photo pixel scale.
_CHAINED_CLIP_SAFETY_MARGIN = 0.05
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
    """Vertex-average centroid -- good enough for seat ordering (unlike
    `geometry.polygon_centroid`'s area-weighted one, which `topology`'s
    REQ-11 checks need for exactness, ordering by angle only needs a point
    *roughly* in the middle).
    """
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _halfplane_side(point: Point, line_point: Point, normal: Point) -> float:
    """Signed distance-like value: >0 on the side `normal` points into, 0 on the line."""
    return (point[0] - line_point[0]) * normal[0] + (point[1] - line_point[1]) * normal[1]


def _halfplane_intersection(p1: Point, p2: Point, line_point: Point, normal: Point) -> Point:
    """Where segment p1-p2 crosses the line through `line_point` normal to `normal`.

    Only ever called from `_clip_polygon_to_halfplane` on a pair known to
    cross (one point measures positive, the other negative under
    `_halfplane_side`), so the denominator below is never zero.
    """
    d1 = _halfplane_side(p1, line_point, normal)
    d2 = _halfplane_side(p2, line_point, normal)
    t = d1 / (d1 - d2)
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def _clip_polygon_to_halfplane(
    polygon: list[Point], line_point: Point, normal: Point
) -> list[Point]:
    """Sutherland-Hodgman clip of a simple polygon against one half-plane.

    Keeps only the region where `_halfplane_side(point, line_point, normal)
    >= 0`. The result is always a subset of `polygon`'s enclosed area (an
    intersection can only remove area, never add it) -- unlike moving
    individual vertices by a fixed offset, which for an irregular,
    non-convex-looking polygon can push a vertex outside the original
    shape entirely, this can't produce a point outside the input.
    """
    output: list[Point] = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        prev = polygon[i - 1]
        curr_in = _halfplane_side(curr, line_point, normal) >= -1e-9
        prev_in = _halfplane_side(prev, line_point, normal) >= -1e-9
        if curr_in:
            if not prev_in:
                output.append(_halfplane_intersection(prev, curr, line_point, normal))
            output.append(curr)
        elif prev_in:
            output.append(_halfplane_intersection(prev, curr, line_point, normal))
    return output


def _clip_away_from_point(
    polygon: list[Point], from_point: Point, inset_pixels: float
) -> list[Point]:
    """Clip `polygon` to remove the sliver within `inset_pixels` of `from_point`.

    The cut line is perpendicular to the direction from `from_point` to
    `polygon`'s own centroid, positioned `inset_pixels` further from
    `from_point` than `polygon`'s own closest point *projected onto that
    direction* (not that point's raw distance to `from_point` -- an
    off-axis corner's raw distance overstates how far along this
    particular line it sits, which would clip away that very point even
    at `inset_pixels=0` instead of the cut landing exactly on it).

    Shared by `_derive_chip_zone`'s two uses: clipping a seat's
    `player_area` away from the table centroid (the inner, `dealer_area`-
    facing edge) and away from each neighbouring seat's centroid (the two
    side edges) -- same operation, only which point it clips away from
    differs. A one-line clip can only ever remove area from `polygon`,
    never move a point outside it, unlike moving individual vertices by a
    fixed offset (tried first; real click sessions produce irregular
    polygons where no single "inward" direction is every vertex's own true
    local inward normal, and a vertex can end up pushed clean through a
    nearby edge of its own polygon).
    """
    polygon_centroid = _polygon_centroid(polygon)
    dx = polygon_centroid[0] - from_point[0]
    dy = polygon_centroid[1] - from_point[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return list(polygon)
    normal = (dx / length, dy / length)

    closest_projection = min(
        (p[0] - from_point[0]) * normal[0] + (p[1] - from_point[1]) * normal[1] for p in polygon
    )
    cut_distance = closest_projection + inset_pixels
    line_point = (
        from_point[0] + normal[0] * cut_distance,
        from_point[1] + normal[1] * cut_distance,
    )
    return _clip_polygon_to_halfplane(polygon, line_point, normal)


def _derive_chip_zone(
    player_area_points: list[Point],
    table_centroid: Point,
    neighbor_clips: list[tuple[Point, float]],
    inset_pixels: float,
) -> list[Point]:
    """The chip-detection zone within one seat's `player_area` (REQ-7).

    Keeps the outer, rail-facing boundary at its full extent -- players
    commonly stack chips right up against the rail, so shrinking that edge
    away (as a uniform shrink-toward-centroid, REQ-10a's original design,
    does) wastes real detection area for no benefit. Trims back only the
    inner edge (facing `dealer_area`, clipped away from `table_centroid` by
    `inset_pixels`) and the two side edges (facing this seat's immediate
    clockwise and counter-clockwise neighbours, one clip per `(centroid,
    inset)` pair in `neighbor_clips`, clipped away from each in turn --
    each with its own inset, not necessarily `inset_pixels`, see
    `_resolve_chip_zone_overlaps`).

    A real click session's player_area polygons are not a perfect,
    gapless tiling -- two independently hand-clicked adjacent wedges can
    end up touching or very slightly overlapping at their shared boundary
    -- so trimming only the inner edge is not enough on its own to
    guarantee `chip_zone`s of different seats don't overlap (REQ-11); the
    side clips are what actually make that guarantee hold on real data,
    not just tidiness.
    """
    zone = _clip_away_from_point(player_area_points, table_centroid, inset_pixels)
    for neighbor_centroid, neighbor_inset in neighbor_clips:
        # Not guarded against too few resulting points: an inset too large
        # for this wedge can legitimately clip it down to nothing usable,
        # same as every other REQ-11 violation -- surfaces downstream as
        # `TablePolygon`'s own "at least 3 points" error, not silently
        # patched over here (see `build_authoring_from_marked_zones`'s
        # docstring).
        if len(zone) < 3:
            return zone
        # A tiny extra margin on every neighbour clip: `zone` going in
        # already contains points computed as the *previous* clip's
        # line-line intersection, not original clicks (the very first
        # neighbour clip's input already went through the table-centroid
        # clip above). Clipping again can then legitimately need to
        # reproduce one of those same points near-exactly, and float64
        # rounding across two chained interpolations occasionally lands a
        # few 1e-10 units outside `player_area`'s own edge, past REQ-11's
        # containment check's tolerance (real case this margin fixes: REQ-
        # 11 rejected a point measured at 0.0 distance from the true
        # boundary by any real-world standard). `_CHAINED_CLIP_SAFETY_
        # MARGIN` is far larger than that noise and still visually
        # meaningless against reference-photo pixel coordinates.
        zone = _clip_away_from_point(
            zone, neighbor_centroid, neighbor_inset + _CHAINED_CLIP_SAFETY_MARGIN
        )
    return zone


_MAX_CHIP_ZONE_INSET_RETRIES = 10
_MAX_OVERLAP_ESCALATIONS = 6


def _safe_chip_zone(
    player_area_points: list[Point],
    table_centroid: Point,
    neighbor_clips: list[tuple[Point, float]],
    inset_pixels: float,
) -> list[Point]:
    """`_derive_chip_zone`, backing off to a smaller *inner-edge* inset if
    the requested one doesn't actually fit this specific wedge (REQ-11).

    A single straight cut line is exact for a convex wedge, but real click
    sessions occasionally produce a seat polygon that's slightly concave or
    has a near-self-touching vertex (an operator's dense curve-tracing
    meeting the next straight run at a shallow angle) -- for those, a large
    `inset_pixels` can legitimately clip past a local pinch in the shape
    and end up outside `player_area`, even though the exact same operation
    is fine for every other, better-behaved seat at the table. Halving
    `inset_pixels` up to `_MAX_CHIP_ZONE_INSET_RETRIES` times and re-
    validating with the same `polygon_contains` check REQ-11 itself uses
    finds a safe inset for that one seat automatically, rather than either
    failing the whole session over one awkward wedge or silently writing an
    invalid `chip_zone` that would fail REQ-11 anyway.

    `neighbor_clips` is passed straight through, unmodified, on every
    attempt: cross-seat overlap (the failure mode neighbour clips guard
    against) is resolved separately, per pair, by
    `_resolve_chip_zone_overlaps` -- only this seat's own containment
    against its own `player_area` is retried here.
    """
    candidate_inset = inset_pixels
    for _ in range(_MAX_CHIP_ZONE_INSET_RETRIES):
        zone_points = _derive_chip_zone(
            player_area_points, table_centroid, neighbor_clips, candidate_inset
        )
        try:
            player_area = TablePolygon(
                points=[TablePoint(x=x, y=y) for x, y in player_area_points]
            )
            zone = TablePolygon(points=[TablePoint(x=x, y=y) for x, y in zone_points])
        except ValidationError:
            candidate_inset /= 2.0
            continue
        if polygon_contains(player_area, zone):
            return zone_points
        candidate_inset /= 2.0
    # Every retry failed (pathological wedge shape): return the original,
    # unclipped `inset_pixels` result and let REQ-11's own validation in
    # `build_authoring_from_marked_zones` reject it with its usual clear
    # error, rather than silently falling back to something unrequested.
    return _derive_chip_zone(player_area_points, table_centroid, neighbor_clips, inset_pixels)


def _compute_all_chip_zones(
    seat_polygons: dict[str, list[Point]],
    table_centroid: Point,
    neighbor_keys: dict[str, list[str]],
    inset_pixels: float,
) -> dict[str, list[Point]]:
    """`_safe_chip_zone` for every seat, escalating specific neighbour pairs
    until none of the resulting `chip_zone`s overlap (REQ-11).

    A real click session's player_area polygons are not a perfect, gapless
    tiling: two independently hand-clicked adjacent wedges can end up
    genuinely overlapping each other by some real (if usually modest)
    amount at their shared boundary -- not a rendering/precision artifact,
    a fact about the two polygons -- which no fixed `inset_pixels` clip
    from each side is guaranteed to fully clear. Each still-overlapping
    pair's own two-sided clip inset doubles (independently of every other
    pair, and of the constant inner-edge inset against `dealer_area`) and
    every seat touching that pair is recomputed, up to
    `_MAX_OVERLAP_ESCALATIONS` rounds; a pair that still overlaps after
    that many doublings is left as `inset_pixels` and surfaces through
    REQ-11's own overlap error in `build_authoring_from_marked_zones`,
    same as any other genuinely-too-close pair of clicks.
    """
    pair_insets = {
        frozenset((key, neighbor)): inset_pixels
        for key, neighbors in neighbor_keys.items()
        for neighbor in neighbors
    }

    def compute_with(pair_insets: dict[frozenset[str], float]) -> dict[str, list[Point]]:
        return {
            key: _safe_chip_zone(
                points,
                table_centroid,
                [
                    (_polygon_centroid(seat_polygons[nk]), pair_insets[frozenset((key, nk))])
                    for nk in neighbor_keys[key]
                ],
                inset_pixels,
            )
            for key, points in seat_polygons.items()
        }

    zones = compute_with(pair_insets)
    for _ in range(_MAX_OVERLAP_ESCALATIONS):
        overlapping_pairs = _find_overlapping_pairs(zones, neighbor_keys)
        if not overlapping_pairs:
            return zones
        for pair in overlapping_pairs:
            pair_insets[pair] *= 2.0
        zones = compute_with(pair_insets)
    return zones


def _find_overlapping_pairs(
    zones: dict[str, list[Point]], neighbor_keys: dict[str, list[str]]
) -> set[frozenset[str]]:
    polygons: dict[str, TablePolygon | None] = {}
    for key, points in zones.items():
        try:
            polygons[key] = TablePolygon(points=[TablePoint(x=x, y=y) for x, y in points])
        except ValidationError:
            polygons[key] = None  # not this function's job to diagnose; just not "overlapping"

    overlapping: set[frozenset[str]] = set()
    for key, neighbors in neighbor_keys.items():
        this_polygon = polygons[key]
        if this_polygon is None:
            continue
        for neighbor in neighbors:
            neighbor_polygon = polygons[neighbor]
            if neighbor_polygon is not None and polygons_overlap(this_polygon, neighbor_polygon):
                overlapping.add(frozenset((key, neighbor)))
    return overlapping


def _table_centroid(seat_polygons: dict[str, list[Point]]) -> Point:
    centroids = [_polygon_centroid(points) for points in seat_polygons.values()]
    return (
        sum(c[0] for c in centroids) / len(centroids),
        sum(c[1] for c in centroids) / len(centroids),
    )


def infer_inner_boundary_polygon(seat_polygons: dict[str, list[Point]]) -> list[Point]:
    """Derive `dealer_area` (REQ-7's "Action Area") from the already-clicked
    seat wedges, instead of a separate manual oval-click step.

    For each seat, its single corner closest to the table's overall centroid
    is the one facing the board; collecting all ten and sorting them by
    angle around the table centroid (the same technique
    `number_seats_clockwise` uses to order seats) produces a simple,
    star-shaped polygon following the true inner boundary.

    One point per seat, not two: an earlier version took each seat's two
    *closest* points, on the reasoning that adjacent wedges share that
    boundary corner pair. That broke on a real elongated-oval table with
    seats clicked at very different point densities (a wedge whose outer
    rail curve was traced with a dozen points vs. a neighbour clicked as a
    plain quadrilateral): "distance to one shared centroid" is not
    monotonic with "how far towards the board" once a seat sits well off
    to one side of that centroid along the table's long axis -- a genuine
    outer, rail-side corner there can measure numerically closer to the
    centroid than the seat's own true inner corner on the *other* side,
    because the outer corner happens to sit more towards the middle of the
    long axis. Concretely: a seat's rank-1-closest point was reliably its
    true inner corner in that real data; its rank-2-closest point was not,
    for exactly this reason. Using only the reliable rank-1 point trades a
    little boundary resolution (one vertex per seat instead of two) for
    correctness independent of per-seat click density or table elongation.
    """
    table_centroid = _table_centroid(seat_polygons)

    inner_points = [
        min(points, key=lambda p: _dist(p, table_centroid)) for points in seat_polygons.values()
    ]

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
    chip_zone_inset_pixels: float = DEFAULT_CHIP_ZONE_INSET_PIXELS,
) -> CalibrationAuthoring:
    """Assemble a full `CalibrationAuthoring` from one marking session.

    Raises `ValueError`/`pydantic.ValidationError` for anything REQ-11
    would reject anyway (degenerate polygons, chip_zone escaping its
    player_area, ...) -- there is no separate "skip validation" path here,
    same as every other `calib` entry point (see `cli.py`'s docstring).

    `chip_zone_inset_pixels` must be >= 0 (see `_derive_chip_zone`); too
    large a value for a given wedge's size can still produce a degenerate
    or escaping chip_zone, which surfaces as the same REQ-11 error a
    hand-authored one would.
    """
    if chip_zone_inset_pixels < 0:
        raise ValueError(f"chip_zone_inset_pixels must be >= 0, got {chip_zone_inset_pixels}")
    table_centroid = _table_centroid(marked.seat_polygons)
    seat_ids = number_seats_clockwise(marked.seat_polygons, marked.dealer_seat_key)

    # Same clockwise order `seat_ids` numbers around -- each key's immediate
    # neighbours (by seat number, wrapping around) are the two seats whose
    # chip_zone a side clip actually needs to stay clear of (see
    # `_derive_chip_zone`'s docstring on why the inner-edge clip alone
    # isn't enough on real, not-perfectly-tiled click data).
    # `seat_ids[key]` is "seat_N" -- sort by the integer N, not the string
    # (lexicographic order would put "seat_10" before "seat_2").
    keys_by_seat_number = sorted(
        marked.seat_polygons, key=lambda key: int(seat_ids[key].removeprefix("seat_"))
    )
    key_index = {key: i for i, key in enumerate(keys_by_seat_number)}
    n = len(keys_by_seat_number)

    def neighbor_keys_for(key: str) -> list[str]:
        i = key_index[key]
        return list({keys_by_seat_number[(i - 1) % n], keys_by_seat_number[(i + 1) % n]})

    neighbor_keys = {key: neighbor_keys_for(key) for key in marked.seat_polygons}
    chip_zones = _compute_all_chip_zones(
        marked.seat_polygons, table_centroid, neighbor_keys, chip_zone_inset_pixels
    )
    seats = [
        CalibrationSeat(
            seat_id=seat_ids[key],
            zones=SeatZones(
                player_area=_to_table_polygon(points),
                chip_zone=_to_table_polygon(chip_zones[key]),
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
