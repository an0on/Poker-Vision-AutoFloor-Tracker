"""Lens undistortion helpers shared by `detection/geometry.py` (REQ-17) and
`calibration/compile.py` (REQ-9).

Lives in `calibration/` (not `detection/`) so `calib compile` can undistort
the raw `image_point`s an operator picked (see `homography.py`'s docstring)
before solving the homography from them, without `calibration/` importing
`detection/` (the dependency only ever runs the other way -- `detection/`
already imports `calibration/`).
"""

from __future__ import annotations

import cv2
import numpy as np

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients


def camera_matrix(camera: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distortion_vector(distortion: DistortionCoefficients) -> np.ndarray:
    return np.array(
        [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3],
        dtype=np.float64,
    )


def undistort_points(
    points: list[tuple[float, float]],
    camera: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> list[tuple[float, float]]:
    """Undo lens distortion, returning points back in pixel units (not normalized)."""
    src = np.array([[list(p)] for p in points], dtype=np.float64)
    matrix = camera_matrix(camera)
    # P=camera_matrix re-projects into the same pixel units the homography
    # is solved against, instead of cv2's default normalized coordinates.
    undistorted = cv2.undistortPoints(src, matrix, distortion_vector(distortion), P=matrix)
    return [(float(p[0][0]), float(p[0][1])) for p in undistorted]


def distort_points(
    points: list[tuple[float, float]],
    camera: CameraIntrinsics,
    distortion: DistortionCoefficients,
) -> list[tuple[float, float]]:
    """Apply lens distortion forward: the exact inverse of `undistort_points`.

    cv2 has no direct "distort points" call. The standard workaround: treat
    each undistorted pixel as a normalized point at z=1 and reproject it
    with `cv2.projectPoints`, which applies the distortion model forward
    (a closed-form polynomial, not an iterative solve) before re-applying
    the camera matrix -- unlike `cv2.undistortPoints`, which iterates to
    invert that same polynomial.
    """
    matrix = camera_matrix(camera)
    object_points = np.array(
        [[[(x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, 1.0]] for x, y in points],
        dtype=np.float64,
    )
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, matrix, distortion_vector(distortion)
    )
    return [(float(p[0][0]), float(p[0][1])) for p in projected]
