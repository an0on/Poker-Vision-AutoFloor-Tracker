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
from poker_vision.calibration.geometry import TableDimensions
from poker_vision.calibration.homography import HomographyCorrespondences
from poker_vision.calibration.zones import CalibrationGeometryModel
from poker_vision.config import Resolution

CALIBRATION_AUTHORING_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class CalibrationAuthoring(CalibrationGeometryModel):
    schema_version: Literal["1.0"]
    table_id: str = Field(min_length=1)
    # REQ-14: the inference resolution the pixel-space homography points
    # below were picked against. Must match `Config.source.resolution_cap`
    # once `calib compile`/pipeline wiring cross-checks the two — a
    # calibration authored against a different resolution than what
    # `capture` actually delivers produces silently wrong table coordinates.
    inference_resolution: Resolution
    camera: CameraIntrinsics
    distortion: DistortionCoefficients
    homography: HomographyCorrespondences
    table: TableDimensions


def load_calibration_authoring(path: str | Path) -> CalibrationAuthoring:
    """Load and validate a CalibrationAuthoring from a JSON file. Raises
    `ValueError` on any problem -- a missing/unreadable file included, the
    same way `calibration.runtime.load_calibration_runtime` does, so
    `calib validate`/`calib compile` (REQ-9, REQ-10) can treat every
    "invalid authoring" case identically without needing to know which
    failed underneath.
    """
    calibration_path = Path(path)
    try:
        raw = json.loads(calibration_path.read_text())
    except OSError as exc:
        raise ValueError(f"{calibration_path}: could not be read ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{calibration_path}: not valid JSON ({exc})") from exc
    try:
        return CalibrationAuthoring.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{calibration_path}: invalid calibration ({exc})") from exc


def write_calibration_authoring(authoring: CalibrationAuthoring, path: str | Path) -> None:
    """Serialize a `CalibrationAuthoring` to JSON (REQ-10's create/edit CLI)."""
    Path(path).write_text(authoring.model_dump_json(indent=2) + "\n")
