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
actual photos' pixel dimensions rather than trusted blindly, since a
resolution mismatch would silently invalidate that assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poker_vision.calibration.geometry import Matrix3x3, TablePoint, matrix3x3_multiply
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.undistort import undistort_points
from poker_vision.detection.geometry import apply_inverse_homography_to_point

DEFAULT_MIN_MATCH_COUNT = 15
DEFAULT_MIN_INLIER_RATIO = 0.5
DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS = 5.0
DEFAULT_CENTER_STRIP_MARGIN_RATIO = 0.15

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
    """Tunable thresholds for `learn_table_calibration` (REQ-10b step 3)."""

    min_match_count: int = DEFAULT_MIN_MATCH_COUNT
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS
    center_strip_margin_ratio: float = DEFAULT_CENTER_STRIP_MARGIN_RATIO


def _read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise LearnTableError(f"could not read image '{path}'")
    return image


def _check_resolution(
    image: np.ndarray, expected_width: int, expected_height: int, label: str
) -> None:
    height, width = image.shape[:2]
    if (width, height) != (expected_width, expected_height):
        raise LearnTableError(
            f"{label} resolution {width}x{height} does not match the reference "
            f"calibration's inference_resolution {expected_width}x{expected_height} "
            "(REQ-10b assumes the same camera setup for every capture)"
        )


def _center_strip_bbox(
    reference: CalibrationRuntime, image_width: int, image_height: int, margin_ratio: float
) -> tuple[int, int, int, int]:
    """Bounding box, in the reference photo's raw pixel space, of
    `board_zone` and `dealer_area` (REQ-10b step 1): the printed card
    outline and inner-oval/branding region that stays visually consistent
    between two physical tables of the same design even when the felt
    colour differs.

    `apply_inverse_homography_to_point` maps each zone's table-plane
    points back through the reference's already-solved homography into
    raw (distorted) reference pixel space -- the exact inverse of how
    `detection/geometry.py` turns a raw detection into a table point, so
    both directions of the reference calibration agree by construction.
    Expanded by `margin_ratio` (of each side's own extent), since these
    zones were authored a little inside the actual printed lines, not
    exactly on them.
    """
    zone_points: list[TablePoint] = list(reference.zones.board_zone.points) + list(
        reference.zones.dealer_area.points
    )
    pixels = [
        apply_inverse_homography_to_point(
            point, reference.homography, reference.camera, reference.distortion
        )
        for point in zone_points
    ]
    xs = [p.x for p in pixels]
    ys = [p.y for p in pixels]
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
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < _LOWE_RATIO * second_best.distance:
            good_matches.append(best)
    if len(good_matches) < config.min_match_count:
        raise LearnTableError(
            "too few reliable feature matches between reference and live photo "
            f"(need >= {config.min_match_count}, got {len(good_matches)})"
        )
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
    read, its resolution doesn't match `reference.inference_resolution`,
    too few/unreliable feature matches are found, or the resulting
    homography is degenerate (REQ-10b step 3, AC-6b) -- never returns an
    implausible calibration silently.
    """
    config = config or LearnTableConfig()
    expected_width = reference.inference_resolution.width
    expected_height = reference.inference_resolution.height

    reference_gray = _read_grayscale(reference_image_path)
    _check_resolution(reference_gray, expected_width, expected_height, "reference image")
    live_gray = _read_grayscale(live_image_path)
    _check_resolution(live_gray, expected_width, expected_height, "live image")

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
