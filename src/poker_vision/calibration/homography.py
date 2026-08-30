"""Pixel-to-table homography (REQ-4, REQ-7, REQ-9).

Authoring stores the raw point correspondences an operator picked (image
pixel <-> table-plane pairs); `calib compile` (REQ-9) solves those into the
precomputed forward/inverse matrices that `CalibrationRuntime` carries, so
runtime code never re-solves a homography on load.
"""

from __future__ import annotations

import math

from pydantic import Field, model_validator

from poker_vision.calibration.geometry import (
    Matrix3x3,
    PixelPoint,
    TablePoint,
    matrix3x3_multiply,
)
from poker_vision.schema_base import StrictModel

# Tolerance for forward @ inverse == identity (REQ-11); generous relative to
# double-precision round-trip error, tight enough to catch a wrong/stale
# inverse.
_IDENTITY_EPSILON = 1e-6


class HomographyPointCorrespondence(StrictModel):
    image_point: PixelPoint
    table_point: TablePoint


class HomographyCorrespondences(StrictModel):
    """Point pairs used to solve for the pixel -> table-plane homography.

    At least 4 pairs are required (minimum for `cv2.findHomography`/DLT).
    """

    points: list[HomographyPointCorrespondence] = Field(min_length=4)


class HomographyMatrix(StrictModel):
    """Precomputed homography, row-major 3x3.

    `forward` maps undistorted pixel coordinates to table-plane coordinates;
    `inverse` is its matrix inverse (table -> pixel), used e.g. by the debug
    overlay (REQ-37) to draw table-plane geometry back onto the frame.
    """

    forward: Matrix3x3
    inverse: Matrix3x3

    @model_validator(mode="after")
    def _check_invertible(self) -> HomographyMatrix:
        # REQ-11: forward @ inverse must be the identity matrix. This is
        # deliberately not a separate `det(forward) != 0` check: a
        # homography is only defined up to scale, and a 3x3 determinant
        # scales with the cube of that factor, so an absolute-value
        # threshold on it would reject validly-scaled (if unusually small)
        # matrices. The round-trip check below doesn't have that problem
        # (rescaling forward and inverse by reciprocal factors leaves their
        # product unchanged) and is strictly stronger anyway: if `forward`
        # were truly singular, no `inverse` could satisfy this equation
        # (det(forward) * det(inverse) would have to equal det(identity) =
        # 1, which is impossible when det(forward) = 0), so this alone
        # already proves `forward` is invertible.
        product = matrix3x3_multiply(self.forward, self.inverse)
        for row in range(3):
            for col in range(3):
                entry = product[row][col]
                # Individually-finite entries can still multiply/sum into an
                # overflowed inf or (inf + -inf =) nan product; a bare
                # `abs(nan - expected) > epsilon` is always False (NaN
                # comparisons never succeed), which would silently let a
                # garbage product through, so check finiteness explicitly.
                if not math.isfinite(entry):
                    raise ValueError(
                        "homography is not invertible: forward @ inverse contains a "
                        "non-finite value (overflow)"
                    )
                expected = 1.0 if row == col else 0.0
                if abs(entry - expected) > _IDENTITY_EPSILON:
                    raise ValueError(
                        "homography is not invertible: forward @ inverse does not "
                        "equal the identity matrix"
                    )
        return self
