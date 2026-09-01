"""`calib compile`: `CalibrationAuthoring` -> `CalibrationRuntime` (REQ-9).

Solves the authoring schema's raw homography point correspondences into a
precomputed forward/inverse matrix pair, so pipeline startup (`runner.
lifecycle.run_command`/`validate_command`) never re-solves geometry on
load -- it only loads and validates the already-compiled `CalibrationRuntime`
via `calibration.runtime.load_calibration_runtime`.

Deterministic (AC-6): `_solve_homography` and `cv2.findHomography` (method 0,
i.e. no RANSAC/LMedS randomness) are pure functions of their input, and
`CalibrationRuntime`'s field order is fixed by its class definition -- the
same authoring file compiled twice produces byte-identical JSON.
"""

from __future__ import annotations

import cv2
import numpy as np

from poker_vision.calibration.authoring import CalibrationAuthoring
from poker_vision.calibration.homography import HomographyCorrespondences, HomographyMatrix
from poker_vision.calibration.runtime import CALIBRATION_RUNTIME_SCHEMA_VERSION, CalibrationRuntime
from poker_vision.calibration.undistort import undistort_points
from poker_vision.config import Resolution


def _solve_homography(authoring: CalibrationAuthoring) -> HomographyMatrix:
    """Undistort each authored `image_point`, then solve pixel -> table.

    `image_point`s are picked by an operator directly on the raw (as
    captured, still-distorted) frame -- see `homography.py`'s docstring --
    while `HomographyMatrix.forward` is defined for *undistorted* pixel
    coordinates (`detection/geometry.py` undistorts every detected point
    the same way before applying it, frame by frame; this does the
    equivalent correction once, here, for the fixed calibration reference
    points).
    """
    correspondences: HomographyCorrespondences = authoring.homography
    raw_image_points = [(p.image_point.x, p.image_point.y) for p in correspondences.points]
    table_points = [(p.table_point.x, p.table_point.y) for p in correspondences.points]
    undistorted_image_points = undistort_points(
        raw_image_points, authoring.camera, authoring.distortion
    )

    src = np.array(undistorted_image_points, dtype=np.float64)
    dst = np.array(table_points, dtype=np.float64)
    # method=0: regular least-squares (normalized DLT) over every point --
    # deterministic, unlike RANSAC/LMedS, and appropriate here since
    # authoring correspondences are hand-picked, not noisy sensor data.
    forward, _ = cv2.findHomography(src, dst, method=0)
    if forward is None:
        raise ValueError(
            "could not solve a homography from the authored point correspondences "
            "(are they collinear or otherwise degenerate?)"
        )
    inverse = np.linalg.inv(forward)
    return HomographyMatrix(forward=forward.tolist(), inverse=inverse.tolist())


def compile_calibration(authoring: CalibrationAuthoring, based_on: str) -> CalibrationRuntime:
    """Compile a `CalibrationAuthoring` into a `CalibrationRuntime` (REQ-9).

    `based_on` is the identifier/path of the source authoring file, carried
    through to `CalibrationRuntime.based_on` verbatim -- it is the caller's
    job (the CLI) to supply the same value for the same input so output
    stays deterministic (AC-6).
    """
    return CalibrationRuntime(
        schema_version=CALIBRATION_RUNTIME_SCHEMA_VERSION,
        table_id=authoring.table_id,
        based_on=based_on,
        inference_resolution=Resolution(
            width=authoring.inference_resolution.width,
            height=authoring.inference_resolution.height,
        ),
        camera=authoring.camera,
        distortion=authoring.distortion,
        homography=_solve_homography(authoring),
        table=authoring.table,
        seats=authoring.seats,
        zones=authoring.zones,
    )
