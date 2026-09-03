"""`calib learn-table`: derive a full `CalibrationRuntime` for a new
physical table instance of the same design from one live photo (REQ-10b).

Ergänzt REQ-9/REQ-10/REQ-10a, ersetzt sie nicht: `calib mark-zones` is still
required once, against the one reference photo, to author the canonical
geometry. This module then repeats *only* the image-to-table homography for
every further physical table of the same design (felt colour may differ,
the printed geometry doesn't) -- no manual re-marking.

Pipeline (REQ-10b):

1. ORB feature matching between the reference photo and the live photo,
   restricted on the reference side to its visually design-stable center
   strip -- the printed card-area outline, inner-oval contour and any
   table branding (`_center_strip_mask`). Not the felt itself, which is the
   one thing that legitimately varies between two physical tables of the
   same design and would otherwise dominate (and mislead) matching.
2. RANSAC-homography over the matches, in *undistorted* pixel space (to
   compose validly with `reference.homography.forward`, itself only
   defined there -- see `homography.py`), giving one image-pixel ->
   table-plane matrix for the new photo. The zones themselves are already
   in table-plane coordinates and are carried over unchanged -- only the
   image -> table homography differs per physical table instance.
3. Too few matches, too low a RANSAC inlier ratio, or inliers spatially
   clustered in too small a region of the mask (`_check_inlier_spread`)
   raise `LearnTableError` instead of emitting an implausible calibration
   (AC-6b).

Camera intrinsics, distortion, `inference_resolution` and table dimensions
are carried over from `reference` unchanged (REQ-10b's stated assumption:
the same camera model/setup for every capture) -- checked against the
actual photos' aspect ratio rather than trusted blindly (a different pixel
*resolution* at the same aspect ratio, e.g. from a photo resized by an
export/sharing step, is resized back and accepted; a different aspect
ratio is not -- see `_normalize_grayscale`).

The live photo should show an empty table, same as the one `calib mark-
zones` was originally run against (REQ-10a): cards/chips lying across the
center strip reduce how much of that region actually matches between the
two photos and can push the RANSAC inlier ratio below `min_inlier_ratio`.
Verified against two real second physical tables (different felt colour,
one with a hand already in progress): one cleared the default thresholds
outright, the other needed `min_inlier_ratio`/`min_match_count` loosened
from their defaults, and the recovered geometry was accurate in both
cases -- an empty-table photo is expected to clear the defaults
comfortably.

Do not loosen `min_inlier_ratio`/`min_match_count` below their defaults as
a blanket policy, though: verified against a third real physical table
that additionally carries its own club branding (logos not present on the
reference table at all, not just a different felt colour -- a real
violation of REQ-10b's "same design" premise), the *default* thresholds
correctly rejected it (0.44 inlier ratio) -- exactly AC-6b's required
behavior. Forcing it through anyway with relaxed thresholds originally
produced a homography that fit the small immediately-matching region (the
shared "DOPO POKER" card outline) but extrapolated to a badly wrong result
everywhere else (seat zones landing entirely off the table): most of the
"matches" behind that lower ratio were themselves spurious, matching the
reference's plain text against the live photo's extra logos/objects, but
concentrated closely enough together to still pass a lenient inlier
*ratio*. `_check_inlier_spread` now independently catches this exact
case (its inliers covered only ~10% of the mask's extent on both axes,
against 60-95% for every one of the other three real photos verified as
actually correct) regardless of how `min_inlier_ratio`/`min_match_count`
are set -- but still only loosen those two per-call for a specific photo
you've separately confirmed (e.g. by rendering `debug.overlay.draw_zones`
over it) is actually a correct match, never as a new default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poker_vision.calibration.geometry import Matrix3x3, TablePoint, matrix3x3_multiply
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.undistort import distort_points, undistort_points

DEFAULT_MIN_MATCH_COUNT = 15
DEFAULT_MIN_INLIER_RATIO = 0.5
DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS = 5.0
# Minimum fraction of the center-strip mask's own bounding-box extent (each
# axis independently) the RANSAC inliers must span. A homography has 8
# degrees of freedom: a small, spatially clustered set of inliers (e.g. all
# sitting around one shared logo/text region) can still reach a high inlier
# *ratio* while leaving the fit almost entirely unconstrained everywhere
# else on the table, extrapolating to badly wrong seat zones there even
# though it fits well right where the matches are. Verified against a real
# false-positive case (a physical table with its own extra, reference-
# absent branding): its inliers spanned only ~10% of the mask extent on
# both axes, against 60-95% for every real match verified as actually
# correct (self-match and two other physical tables) -- 0.3 sits with
# ample margin on both sides of that gap.
DEFAULT_MIN_INLIER_SPREAD_RATIO = 0.3
# Fraction of the photo's own width -- a fixed, tight band width around
# `dealer_area`'s boundary curve regardless of how large that zone itself
# is, wide enough to absorb authoring imprecision (a hand-clicked polygon
# not landing exactly on the printed line) but not wide enough to reach
# content sitting well inside the oval (see `_center_strip_mask`).
DEFAULT_CENTER_STRIP_MARGIN_RATIO = 0.02

# Relative tolerance on width/height ratio when comparing a photo's aspect
# ratio against the reference calibration's `inference_resolution`. A photo
# at a different pixel resolution but the same framing (e.g. resized by an
# export/compression step) is still usable -- see `_normalize_grayscale`;
# a genuinely different framing/aspect ratio is not.
DEFAULT_ASPECT_RATIO_TOLERANCE = 0.01

# Lowe's ratio test threshold for filtering ORB/BFMatcher knn matches -- the
# standard value from Lowe's original SIFT paper, equally applicable to any
# descriptor distance (here Hamming, via `cv2.NORM_HAMMING`).
_LOWE_RATIO = 0.75
_ORB_FEATURE_COUNT = 4000


class LearnTableError(ValueError):
    """`calib learn-table` cannot produce a reliable calibration (REQ-10b,
    AC-6b): too few/unreliable feature matches, or a degenerate resulting
    homography. A `ValueError` subclass so every `calib` subcommand's
    existing `except (ValueError, ValidationError, OSError)` handling
    (`cli.py`) already covers it without a new branch.
    """


@dataclass(frozen=True)
class LearnTableConfig:
    """Tunable thresholds for `learn_table_calibration` (REQ-10b step 3).

    Validated at construction (not just against the built-in defaults)
    since the CLI passes user-supplied values straight through: an
    unvalidated `min_match_count` below 4 (`cv2.findHomography`'s DLT
    minimum) or a non-finite `min_inlier_ratio` (e.g. `nan`, which makes
    every `<` comparison against it false) would otherwise either crash
    deep inside OpenCV or silently defeat the very reliability checks
    REQ-10b step 3 requires.
    """

    min_match_count: int = DEFAULT_MIN_MATCH_COUNT
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO
    min_inlier_spread_ratio: float = DEFAULT_MIN_INLIER_SPREAD_RATIO
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS
    center_strip_margin_ratio: float = DEFAULT_CENTER_STRIP_MARGIN_RATIO
    aspect_ratio_tolerance: float = DEFAULT_ASPECT_RATIO_TOLERANCE

    def __post_init__(self) -> None:
        if self.min_match_count < 4:
            raise LearnTableError(
                f"min_match_count must be >= 4 (cv2.findHomography's minimum), "
                f"got {self.min_match_count}"
            )
        if not math.isfinite(self.min_inlier_ratio) or not (0.0 <= self.min_inlier_ratio <= 1.0):
            raise LearnTableError(
                f"min_inlier_ratio must be a finite value in [0, 1], got {self.min_inlier_ratio}"
            )
        spread_ratio = self.min_inlier_spread_ratio
        if not math.isfinite(spread_ratio) or not (0.0 <= spread_ratio <= 1.0):
            raise LearnTableError(
                f"min_inlier_spread_ratio must be a finite value in [0, 1], got {spread_ratio}"
            )
        if not math.isfinite(self.ransac_reproj_threshold) or self.ransac_reproj_threshold <= 0.0:
            raise LearnTableError(
                "ransac_reproj_threshold must be a finite positive value, got "
                f"{self.ransac_reproj_threshold}"
            )
        margin_ratio = self.center_strip_margin_ratio
        if not math.isfinite(margin_ratio) or margin_ratio < 0.0:
            raise LearnTableError(
                "center_strip_margin_ratio must be a finite non-negative value, got "
                f"{self.center_strip_margin_ratio}"
            )
        aspect_tolerance = self.aspect_ratio_tolerance
        if not math.isfinite(aspect_tolerance) or aspect_tolerance < 0.0:
            raise LearnTableError(
                "aspect_ratio_tolerance must be a finite non-negative value, got "
                f"{self.aspect_ratio_tolerance}"
            )


def _read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise LearnTableError(f"could not read image '{path}'")
    return image


def _normalize_grayscale(
    image: np.ndarray,
    expected_width: int,
    expected_height: int,
    aspect_ratio_tolerance: float,
    label: str,
) -> np.ndarray:
    """Check `image`'s aspect ratio against the reference calibration's
    `inference_resolution`, then resize it to that exact resolution if it
    doesn't already match.

    A photo doesn't have to come off the camera at the calibration's exact
    pixel resolution to be usable -- e.g. after being resized by an export/
    sharing step -- only at the *same framing* (aspect ratio). Distortion
    is `0.0` in every calibration this project has authored so far, which
    makes `undistort_points`/`distort_points` an identity operation
    regardless of the specific camera-intrinsics values used, so resizing
    the photo itself (not rescaling intrinsics) is the one thing that
    actually needs to happen to bring it into the one fixed pixel space
    `reference.homography` is defined against. A genuinely different
    aspect ratio (different framing/camera, not just a resize) still
    fails clearly rather than producing a silently distorted match.
    """
    height, width = image.shape[:2]
    image_ratio = width / height
    expected_ratio = expected_width / expected_height
    relative_difference = abs(image_ratio - expected_ratio) / expected_ratio
    if relative_difference > aspect_ratio_tolerance:
        raise LearnTableError(
            f"{label} aspect ratio {width}x{height} ({image_ratio:.4f}) does not match "
            f"the reference calibration's inference_resolution {expected_width}x"
            f"{expected_height} ({expected_ratio:.4f}) -- REQ-10b assumes the same "
            "camera framing for every capture (a different pixel resolution at the "
            "same aspect ratio, e.g. from export/compression, is fine)"
        )
    if (width, height) == (expected_width, expected_height):
        return image
    return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_AREA)


def _apply_homography_matrix(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    """Apply a 3x3 homography matrix to one point.

    Deliberately local to `calibration/`, not a call into `detection/
    geometry.py`'s equivalent: `detection/` already imports `calibration/`
    (e.g. `undistort.py`, `homography.py`), and that dependency only ever
    runs the one way (see `undistort.py`'s module docstring) -- importing
    back from here would reverse it.
    """
    points = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(points, np.array(matrix, dtype=np.float64))
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def _table_point_to_raw_reference_pixel(
    point: TablePoint, reference: CalibrationRuntime
) -> tuple[float, float]:
    """Table-plane point -> raw (distorted) reference-photo pixel: the exact
    inverse of how a detected pixel is turned into a table point elsewhere
    in the pipeline (undistort, then homography-forward) -- homography-
    inverse first here, then redistort.
    """
    undistorted_x, undistorted_y = _apply_homography_matrix(
        reference.homography.inverse, point.x, point.y
    )
    ((raw_x, raw_y),) = distort_points(
        [(undistorted_x, undistorted_y)], reference.camera, reference.distortion
    )
    return raw_x, raw_y


def _polygon_to_reference_pixels(
    polygon_points: list[TablePoint], reference: CalibrationRuntime
) -> np.ndarray:
    pixels = [_table_point_to_raw_reference_pixel(point, reference) for point in polygon_points]
    return np.array([[round(x), round(y)] for x, y in pixels], dtype=np.int32)


def _center_strip_mask(
    reference: CalibrationRuntime, image_width: int, image_height: int, margin_ratio: float
) -> np.ndarray:
    """Mask, in the reference photo's raw pixel space, of the table's
    visually design-stable "center strip" (REQ-10b step 1): `board_zone`'s
    filled area (the card-field outline and any text right next to it,
    small and always relevant) plus a band around `dealer_area`'s boundary
    *curve* only -- not its full filled interior.

    That distinction matters: a real second physical table (verified,
    not hypothetical) can carry its own extra branding -- e.g. club logos
    -- printed well inside the oval, away from its boundary, that simply
    doesn't exist on the reference table at all. Matching against the
    whole `dealer_area` interior pulled in that extra content and produced
    spurious matches (the reference's plain print resembling *something*
    in the unrelated logo, purely by descriptor coincidence) that fit a
    homography locally but extrapolated to a badly wrong result far from
    the match cluster. Restricting to the boundary curve mirrors the
    physical reality instead: the printed oval *outline* is part of the
    shared base design (REQ-10b's premise), the felt inside it is not.

    `margin_ratio` is a fraction of `image_width` (not of the zone's own,
    much larger, extent, as an earlier version had it) -- keeping the
    boundary band a fixed, tight width regardless of how large the zone
    itself is, wide enough to cover the authoring imprecision inherent in
    a hand-clicked polygon not landing exactly on the printed line, not
    wide enough to reach a logo sitting well inside the oval.
    """
    board_pixels = _polygon_to_reference_pixels(reference.zones.board_zone.points, reference)
    dealer_pixels = _polygon_to_reference_pixels(reference.zones.dealer_area.points, reference)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [board_pixels], 255)
    cv2.polylines(mask, [dealer_pixels], isClosed=True, color=255, thickness=1)

    margin_pixels = max(1, round(margin_ratio * image_width))
    kernel_size = 2 * margin_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel)

    if not mask.any():
        raise LearnTableError(
            "center-strip mask (board_zone + dealer_area boundary) is empty or falls "
            "entirely outside the reference photo"
        )
    return mask


def _mask_bbox_extent(mask: np.ndarray, reference: CalibrationRuntime) -> tuple[float, float]:
    """(width, height) of the bounding box of `mask`'s nonzero pixels, in
    *undistorted* pixel space.

    The reference scale `_check_inlier_spread` measures RANSAC inliers
    against -- computed once from the actual mask rather than re-deriving
    the zone geometry a second time, so it can never drift out of sync
    with what `_match_keypoints` actually restricted the search to.
    Undistorted, not raw, space: `_solve_live_to_reference_homography`'s
    inlier points are undistorted (`reference.homography.forward` is only
    defined there -- see `homography.py`), but `mask` itself is rasterized
    in raw reference-photo pixel coordinates, so comparing its extent
    directly against those points would silently mix the two coordinate
    systems for any calibration with nonzero lens distortion (every one
    authored so far happens to use zero distortion, which makes undistort
    an identity operation and hid this -- but the schema explicitly
    supports nonzero values). Undistorting just the raw bounding box's
    four corners, rather than every nonzero mask pixel, is an
    approximation (undistortion is nonlinear, so it isn't exactly the
    bounding box of the undistorted region) -- adequate here since this
    feeds a coarse anti-clustering heuristic, not a precision measurement.
    """
    ys, xs = np.nonzero(mask)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    undistorted_corners = undistort_points(corners, reference.camera, reference.distortion)
    undistorted_xs = [x for x, _ in undistorted_corners]
    undistorted_ys = [y for _, y in undistorted_corners]
    return (
        max(undistorted_xs) - min(undistorted_xs),
        max(undistorted_ys) - min(undistorted_ys),
    )


def _check_inlier_spread(
    reference_points: np.ndarray, mask_extent: tuple[float, float], min_spread_ratio: float
) -> None:
    """Reject a homography whose RANSAC inliers are spatially clustered in
    a small corner of the center-strip mask, even if their inlier *ratio*
    is high.

    A homography has 8 degrees of freedom: a cluster of correspondences
    concentrated in one small region (e.g. all sitting around one shared
    logo/text glyph) can still satisfy `cv2.findHomography`'s reprojection
    threshold there while leaving the fit almost entirely unconstrained
    everywhere else on the table -- extrapolating to a badly wrong result
    (verified against a real false-positive case: seat zones landing
    entirely off the table) despite a passing inlier ratio. Requiring the
    inliers to actually span a meaningful fraction of the mask's own
    extent, on both axes independently, catches this the ratio check
    alone cannot.
    """
    mask_width, mask_height = mask_extent
    x_spread = float(reference_points[:, 0].max() - reference_points[:, 0].min())
    y_spread = float(reference_points[:, 1].max() - reference_points[:, 1].min())
    x_ratio = x_spread / mask_width if mask_width > 0 else 0.0
    y_ratio = y_spread / mask_height if mask_height > 0 else 0.0
    if x_ratio < min_spread_ratio or y_ratio < min_spread_ratio:
        raise LearnTableError(
            "RANSAC inliers are too spatially clustered to trust (covering "
            f"{x_ratio:.2f}x{y_ratio:.2f} of the center-strip mask's extent, "
            f"required >= {min_spread_ratio:.2f} on both axes) -- a locally-fitting "
            "homography can still extrapolate to a badly wrong result elsewhere on "
            "the table"
        )


def _filter_reliable_matches(
    raw_matches: list[list[cv2.DMatch]], min_match_count: int
) -> list[cv2.DMatch]:
    """Lowe's ratio test, then deduplicate to the single best (lowest-
    distance) match per reference keypoint.

    `knnMatch(live, reference, k=2)` finds each live descriptor's own
    nearest reference descriptor independently, so a repeated pattern in
    the live photo (e.g. a symmetric card outline or a repeated logo) can
    legitimately match several different live keypoints to the *same*
    reference keypoint. Left alone, those duplicates would each count
    separately toward `min_match_count` and the RANSAC inlier ratio,
    letting a coherent-looking but wrong repeated-region match slip past
    both reliability checks -- deduplicating enforces a proper one-to-one
    correspondence before either check runs. Split out from
    `_match_keypoints` so this filtering logic is directly unit-testable
    against hand-built `cv2.DMatch` pairs, without needing real
    images/ORB detection.
    """
    best_match_by_reference_index: dict[int, cv2.DMatch] = {}
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance >= _LOWE_RATIO * second_best.distance:
            continue
        existing = best_match_by_reference_index.get(best.trainIdx)
        if existing is None or best.distance < existing.distance:
            best_match_by_reference_index[best.trainIdx] = best
    good_matches = list(best_match_by_reference_index.values())
    if len(good_matches) < min_match_count:
        raise LearnTableError(
            "too few reliable feature matches between reference and live photo "
            f"(need >= {min_match_count}, got {len(good_matches)})"
        )
    return good_matches


def _match_keypoints(
    reference_gray: np.ndarray,
    live_gray: np.ndarray,
    reference_mask: np.ndarray,
    config: LearnTableConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """ORB features + BFMatcher/Hamming + Lowe ratio test (REQ-10b step 1).

    `reference_mask` (not the live photo, whose table position/orientation
    in-frame is exactly what this function is solving for) restricts which
    reference keypoints are ever considered -- the actual mechanism behind
    "Suchraum primär im Mittelstreifen": a match can only ever pair a
    reference feature from that region with *some* point in the live
    photo, wherever it lands there.
    """
    detector = cv2.ORB_create(nfeatures=_ORB_FEATURE_COUNT)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_gray, reference_mask
    )
    live_keypoints, live_descriptors = detector.detectAndCompute(live_gray, None)
    if not reference_keypoints or not live_keypoints:
        raise LearnTableError(
            "no ORB features detected in the reference center strip or the live photo"
        )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(live_descriptors, reference_descriptors, k=2)
    good_matches = _filter_reliable_matches(raw_matches, config.min_match_count)
    live_points = np.array(
        [live_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float64
    )
    reference_points = np.array(
        [reference_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float64
    )
    return live_points, reference_points


def _solve_live_to_reference_homography(
    live_points_raw: np.ndarray,
    reference_points_raw: np.ndarray,
    reference: CalibrationRuntime,
    mask_extent: tuple[float, float],
    config: LearnTableConfig,
) -> np.ndarray:
    """RANSAC-homography over the matched keypoints, in undistorted pixel
    space (REQ-10b step 2): `reference.homography.forward` is itself only
    defined for undistorted reference-pixel coordinates (see
    `homography.py`), so this must operate in that same space to compose
    validly with it in `_compose_homography`. Both photos are assumed to
    share one camera/lens setup (REQ-10b), so the reference's own
    `camera`/`distortion` apply to the live photo's matched points too.
    """
    live_points = undistort_points(
        [(float(x), float(y)) for x, y in live_points_raw], reference.camera, reference.distortion
    )
    reference_points = undistort_points(
        [(float(x), float(y)) for x, y in reference_points_raw],
        reference.camera,
        reference.distortion,
    )
    src = np.array(live_points, dtype=np.float64)
    dst = np.array(reference_points, dtype=np.float64)
    homography, inlier_mask = cv2.findHomography(
        src, dst, cv2.RANSAC, config.ransac_reproj_threshold
    )
    if homography is None:
        raise LearnTableError(
            "could not solve a homography from the matched features (degenerate match set)"
        )
    total = len(src)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / total if total else 0.0
    if inlier_ratio < config.min_inlier_ratio:
        raise LearnTableError(
            f"too few reliable inlier matches after RANSAC ({inliers}/{total} = "
            f"{inlier_ratio:.2f} inlier ratio, required >= {config.min_inlier_ratio:.2f})"
        )
    inlier_reference_points = dst[inlier_mask.ravel().astype(bool)]
    _check_inlier_spread(inlier_reference_points, mask_extent, config.min_inlier_spread_ratio)
    return homography


def _compose_homography(
    reference_forward: Matrix3x3, live_to_reference: np.ndarray
) -> HomographyMatrix:
    live_to_reference_matrix = [[float(v) for v in row] for row in live_to_reference.tolist()]
    composed = matrix3x3_multiply(reference_forward, live_to_reference_matrix)
    forward = [[float(v) for v in row] for row in composed]
    try:
        inverse = np.linalg.inv(np.array(forward, dtype=np.float64)).tolist()
    except np.linalg.LinAlgError as exc:
        raise LearnTableError("composed live-photo homography is not invertible") from exc
    return HomographyMatrix(forward=forward, inverse=inverse)


def learn_table_calibration(
    reference: CalibrationRuntime,
    reference_image_path: str | Path,
    live_image_path: str | Path,
    based_on: str,
    table_id: str | None = None,
    config: LearnTableConfig | None = None,
) -> CalibrationRuntime:
    """Derive a full `CalibrationRuntime` for `live_image_path`, a new photo
    of the same physical table design as `reference` (REQ-10b), without
    repeating `calib mark-zones`'s manual authoring.

    `reference` must already be a compiled `CalibrationRuntime` (the output
    of `calib compile` on the REQ-10a-authored reference calibration), and
    `reference_image_path` the same photo that reference was authored
    against -- required to locate the center strip (`_center_strip_mask`)
    in the reference's own pixel space.

    Raises `LearnTableError` (a `ValueError`) if either image can't be
    read, its aspect ratio doesn't match `reference.inference_resolution`
    (a different pixel resolution at the *same* aspect ratio is resized
    and accepted -- see `_normalize_grayscale`), too few/unreliable
    feature matches are found, or the resulting homography is degenerate
    (REQ-10b step 3, AC-6b) -- never returns an implausible calibration
    silently.
    """
    config = config or LearnTableConfig()
    expected_width = reference.inference_resolution.width
    expected_height = reference.inference_resolution.height

    reference_gray = _normalize_grayscale(
        _read_grayscale(reference_image_path),
        expected_width,
        expected_height,
        config.aspect_ratio_tolerance,
        "reference image",
    )
    live_gray = _normalize_grayscale(
        _read_grayscale(live_image_path),
        expected_width,
        expected_height,
        config.aspect_ratio_tolerance,
        "live image",
    )

    reference_mask = _center_strip_mask(
        reference,
        reference_gray.shape[1],
        reference_gray.shape[0],
        config.center_strip_margin_ratio,
    )
    mask_extent = _mask_bbox_extent(reference_mask, reference)

    live_points_raw, reference_points_raw = _match_keypoints(
        reference_gray, live_gray, reference_mask, config
    )
    live_to_reference = _solve_live_to_reference_homography(
        live_points_raw, reference_points_raw, reference, mask_extent, config
    )
    homography = _compose_homography(reference.homography.forward, live_to_reference)

    return CalibrationRuntime(
        schema_version=reference.schema_version,
        table_id=table_id if table_id is not None else reference.table_id,
        based_on=based_on,
        inference_resolution=reference.inference_resolution,
        camera=reference.camera,
        distortion=reference.distortion,
        homography=homography,
        table=reference.table,
        seats=reference.seats,
        zones=reference.zones,
        card_dealer_seat_id=reference.card_dealer_seat_id,
    )
