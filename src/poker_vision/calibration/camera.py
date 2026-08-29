"""Camera intrinsics and lens distortion (REQ-4, REQ-7).

Follows OpenCV's pinhole camera convention so values can be passed straight
into `cv2.undistort` / `cv2.initUndistortRectifyMap` without reshaping.
"""

from __future__ import annotations

from pydantic import Field

from poker_vision.schema_base import StrictModel


class CameraIntrinsics(StrictModel):
    """Pinhole camera matrix parameters, in pixels."""

    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float


class DistortionCoefficients(StrictModel):
    """OpenCV plumb-bob distortion model coefficients (k1, k2, p1, p2, k3)."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
