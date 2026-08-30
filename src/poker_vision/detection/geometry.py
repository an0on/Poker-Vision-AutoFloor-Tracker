"""Pixel-space box math and the pixel -> table-plane transform (REQ-17).

`box_center` is the exact bounding-box-centre method verified in Phase 0
(`phase0_poc.py::Detection.center`): `((x1+x2)/2, (y1+y2)/2)`, nothing more.
Every detector implementation that derives a centre from a pixel box must
route through it so all detectors agree on the same method.

`apply_homography_to_point`/`transform_box_to_table` are the pixel -> table
transform itself, applied here in the detection stage so nothing behind it
ever sees pixel coordinates (REQ-5). `HomographyMatrix.forward` is defined
for *undistorted* pixel coordinates (see calibration/homography.py), so both
functions undistort via the calibration's camera/distortion parameters
before applying it -- REQ-8 owns image-level undistortion and the CLI that
bundles it with the homography into one authoring stage, but a raw
detector's per-point pixel coordinates still need this correction wherever
the homography is applied, so it happens here too.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import Matrix3x3, PixelPoint, TablePoint
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.detection.models import TableBoundingBox

PixelBox = tuple[float, float, float, float]
"""A pixel-space bounding box as (x1, y1, x2, y2), matching Phase 0's convention."""

# A homogeneous w this close to zero (relative to the matrix's own scale --
# see _apply_matrix's normalisation) maps to (or past) the horizon -- not a
# point a top-down table camera can ever actually observe -- so treat it as
# an invalid transform rather than emit +/-inf table coordinates.
_W_EPSILON = 1e-9

# Points sampled per edge (corners included) when bounding a transformed
# box. Undistortion is nonlinear, so a box's straight edges become curves;
# more samples track that curve more tightly at a small, fixed extra cost.
_BOX_EDGE_SAMPLES = 8


def box_center(box: PixelBox) -> PixelPoint:
    """Exact bounding-box centre in pixel space (Phase 0's REQ-0.4 method)."""
    x1, y1, x2, y2 = box
    return PixelPoint(x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0)


def apply_homography_to_point(
    point: PixelPoint,
    homography: HomographyMatrix,
    camera: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> TablePoint:
    """Undistort one pixel-space point, then map it onto the table plane."""
    ((ux, uy),) = _undistort_points([(point.x, point.y)], camera, distortion)
    x, y = _apply_matrix(homography.forward, ux, uy)
    return TablePoint(x=x, y=y)


def transform_box_to_table(
    box: PixelBox,
    homography: HomographyMatrix,
    camera: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> TableBoundingBox:
    """Undistort a pixel-space box's edges, then map them onto the table plane.

    A homography can rotate/skew, and undistortion is nonlinear, so neither
    step generally keeps the box's sides straight or axis-aligned; the
    result is the bounding box of many sampled perimeter points (not just
    the four corners, which a curved edge can bulge past), never a literal
    corner-to-corner mapping.
    """
    perimeter = _sample_box_perimeter(box)
    undistorted = _undistort_points(perimeter, camera, distortion)
    transformed = [_apply_matrix(homography.forward, x, y) for x, y in undistorted]
    xs = [x for x, _ in transformed]
    ys = [y for _, y in transformed]
    return TableBoundingBox(
        min=TablePoint(x=min(xs), y=min(ys)),
        max=TablePoint(x=max(xs), y=max(ys)),
    )


def _sample_box_perimeter(
    box: PixelBox, samples_per_edge: int = _BOX_EDGE_SAMPLES
) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    steps = [i / (samples_per_edge - 1) for i in range(samples_per_edge)]
    points: list[tuple[float, float]] = []
    for t in steps:
        points.append((x1 + t * (x2 - x1), y1))  # top edge
        points.append((x1 + t * (x2 - x1), y2))  # bottom edge
        points.append((x1, y1 + t * (y2 - y1)))  # left edge
        points.append((x2, y1 + t * (y2 - y1)))  # right edge
    return points


def _camera_matrix(camera: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _distortion_vector(distortion: DistortionCoefficients) -> np.ndarray:
    return np.array(
        [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3],
        dtype=np.float64,
    )


def _undistort_points(
    points: list[tuple[float, float]],
    camera: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> list[tuple[float, float]]:
    """Undo lens distortion, returning points back in pixel units (not normalized)."""
    src = np.array([[list(p)] for p in points], dtype=np.float64)
    camera_matrix = _camera_matrix(camera)
    # P=camera_matrix re-projects into the same pixel units the homography
    # was solved against, instead of cv2's default normalized coordinates.
    undistorted = cv2.undistortPoints(
        src, camera_matrix, _distortion_vector(distortion), P=camera_matrix
    )
    return [(float(p[0][0]), float(p[0][1])) for p in undistorted]


def _apply_matrix(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    # A homography is only defined up to scale (HomographyMatrix's own
    # invertibility check accepts any uniformly-rescaled forward/inverse
    # pair), so normalising by the Frobenius norm first makes the horizon
    # threshold below independent of that arbitrary scale; tx/tw and ty/tw
    # are unaffected since numerator and denominator scale together.
    norm = math.sqrt(sum(entry * entry for row in matrix for entry in row))
    m = [[entry / norm for entry in row] for row in matrix]
    tx = m[0][0] * x + m[0][1] * y + m[0][2]
    ty = m[1][0] * x + m[1][1] * y + m[1][2]
    tw = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(tw) < _W_EPSILON:
        raise ValueError(
            f"homography maps pixel point ({x}, {y}) to the horizon (w={tw}); "
            "not a valid table-plane point"
        )
    return tx / tw, ty / tw
