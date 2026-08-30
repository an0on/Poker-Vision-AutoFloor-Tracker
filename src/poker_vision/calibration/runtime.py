"""Calibration runtime schema (REQ-4, REQ-6, REQ-7, REQ-9).

The compiled output of `calib compile`: the same zones and table geometry
as the `CalibrationAuthoring` it was built from, plus precomputed matrices
(homography forward/inverse) so pipeline code never re-solves geometry on
load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.zones import CalibrationGeometryModel
from poker_vision.config import Resolution

CALIBRATION_RUNTIME_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class CalibrationRuntime(CalibrationGeometryModel):
    schema_version: Literal["1.0"]
    table_id: str = Field(min_length=1)
    based_on: str = Field(
        min_length=1, description="Identifier/path of the source CalibrationAuthoring"
    )
    # REQ-14: carried over from CalibrationAuthoring unchanged by `calib
    # compile` — the inference resolution this calibration's pixel-space
    # geometry was solved against, expected to match `Config.source.
    # resolution_cap` at runtime.
    inference_resolution: Resolution
    camera: CameraIntrinsics
    distortion: DistortionCoefficients
    homography: HomographyMatrix
    table: TableDimensions


def load_calibration_runtime(path: str | Path) -> CalibrationRuntime:
    """Load and validate a CalibrationRuntime from a JSON file. Raises on any schema violation."""
    calibration_path = Path(path)
    try:
        raw = json.loads(calibration_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{calibration_path}: not valid JSON ({exc})") from exc
    try:
        return CalibrationRuntime.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{calibration_path}: invalid calibration ({exc})") from exc
