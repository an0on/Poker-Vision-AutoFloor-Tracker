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

import cv2
import numpy as np

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import Matrix3x3, PixelPoint, TablePoint
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.detection.models import TableBoundingBox

PixelBox = tuple[float, float, float, float]
"""A pixel-space bounding box as (x1, y1, x2, y2), matching Phase 0's convention."""

# A homogeneous w this close to zero maps to (or past) the horizon -- not a
# point a top-down table camera can ever actually observe -- so treat it as
# an invalid transform rather than emit +/-inf table coordinates.
_W_EPSILON = 1e-9


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
    """Undistort a pixel-space box's four corners, then map them onto the table plane.

    A homography can rotate/skew, so the four transformed corners are not
    generally axis-aligned in table space; the result is their bounding
    box, not a literal corner-to-corner mapping.
    """
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    undistorted = _undistort_points(corners, camera, distortion)
    transformed = [_apply_matrix(homography.forward, x, y) for x, y in undistorted]
    xs = [x for x, _ in transformed]
    ys = [y for _, y in transformed]
    return TableBoundingBox(
        min=TablePoint(x=min(xs), y=min(ys)),
        max=TablePoint(x=max(xs), y=max(ys)),
    )


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
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    tw = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(tw) < _W_EPSILON:
        raise ValueError(
            f"homography maps pixel point ({x}, {y}) to the horizon (w={tw}); "
            "not a valid table-plane point"
        )
    return tx / tw, ty / tw
