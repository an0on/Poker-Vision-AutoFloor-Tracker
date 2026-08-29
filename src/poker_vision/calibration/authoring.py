"""Calibration authoring schema (REQ-4, REQ-6, REQ-7).

The one schema an operator edits by hand (or via the calibration CLI,
REQ-10): camera intrinsics/distortion, the raw homography point
correspondences, table dimensions, and the seat/board/dealer zones in
table-plane coordinates. `calib compile` (REQ-9) turns this into a
`CalibrationRuntime`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import ImageDimensions, TableDimensions
from poker_vision.calibration.homography import HomographyCorrespondences
from poker_vision.calibration.zones import GlobalZones, SeatListModel

CALIBRATION_AUTHORING_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class CalibrationAuthoring(SeatListModel):
    schema_version: Literal["1.0"]
    table_id: str = Field(min_length=1)
    image: ImageDimensions
    camera: CameraIntrinsics
    distortion: DistortionCoefficients
    homography: HomographyCorrespondences
    table: TableDimensions
    zones: GlobalZones


def load_calibration_authoring(path: str | Path) -> CalibrationAuthoring:
    """Load and validate a CalibrationAuthoring from a JSON file. Raises on any schema violation."""
    calibration_path = Path(path)
    try:
        raw = json.loads(calibration_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{calibration_path}: not valid JSON ({exc})") from exc
    try:
        return CalibrationAuthoring.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{calibration_path}: invalid calibration ({exc})") from exc
