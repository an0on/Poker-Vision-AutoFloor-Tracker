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
   table branding (`_center_strip_bbox`). Not the felt itself, which is the
   one thing that legitimately varies between two physical tables of the
   same design and would otherwise dominate (and mislead) matching.
2. RANSAC-homography over the matches, in *undistorted* pixel space (to
   compose validly with `reference.homography.forward`, itself only
   defined there -- see `homography.py`), giving one image-pixel ->
   table-plane matrix for the new photo. The zones themselves are already
   in table-plane coordinates and are carried over unchanged -- only the
   image -> table homography differs per physical table instance.
3. Too few matches, or too low a RANSAC inlier ratio, raise
   `LearnTableError` instead of emitting an implausible calibration
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
Verified against a real second physical table (different felt colour, a
hand already in progress) -- the geometry recovered was accurate, but only
after loosening `min_inlier_ratio`/`min_match_count` from their defaults;
an empty-table photo is expected to clear the defaults comfortably.
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
DEFAULT_CENTER_STRIP_MARGIN_RATIO = 0.15

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


def _center_strip_bbox(
    reference: CalibrationRuntime, image_width: int, image_height: int, margin_ratio: float
) -> tuple[int, int, int, int]:
    """Bounding box, in the reference photo's raw pixel space, of
    `board_zone` and `dealer_area` (REQ-10b step 1): the printed card
    outline and inner-oval/branding region that stays visually consistent
    between two physical tables of the same design even when the felt
    colour differs.

    `_table_point_to_raw_reference_pixel` maps each zone's table-plane
    points back through the reference's already-solved homography into
    raw (distorted) reference pixel space -- the exact inverse of how a
    raw detection is turned into a table point, so both directions of the
    reference calibration agree by construction. Expanded by
    `margin_ratio` (of each side's own extent), since these zones were
    authored a little inside the actual printed lines, not exactly on
    them.
    """
    zone_points: list[TablePoint] = list(reference.zones.board_zone.points) + list(
        reference.zones.dealer_area.points
    )
    pixels = [_table_point_to_raw_reference_pixel(point, reference) for point in zone_points]
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    margin_x = (max_x - min_x) * margin_ratio
    margin_y = (max_y - min_y) * margin_ratio
    x0 = max(0, int(min_x - margin_x))
    y0 = max(0, int(min_y - margin_y))
    x1 = min(image_width, int(max_x + margin_x) + 1)
    y1 = min(image_height, int(max_y + margin_y) + 1)
    if x1 <= x0 or y1 <= y0:
        raise LearnTableError(
            "center-strip bounding box (board_zone + dealer_area) is degenerate "
            "or falls outside the reference photo"
        )
    return x0, y0, x1, y1


def _center_strip_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    return mask


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
    against -- required to locate the center strip (`_center_strip_bbox`)
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

    bbox = _center_strip_bbox(
        reference,
        reference_gray.shape[1],
        reference_gray.shape[0],
        config.center_strip_margin_ratio,
    )
    reference_mask = _center_strip_mask(reference_gray.shape, bbox)

    live_points_raw, reference_points_raw = _match_keypoints(
        reference_gray, live_gray, reference_mask, config
    )
    live_to_reference = _solve_live_to_reference_homography(
        live_points_raw, reference_points_raw, reference, config
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
