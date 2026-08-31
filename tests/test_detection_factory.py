"""REQ-45: `detection.factory.create_detector` -- config-driven mock-mode selection."""

from __future__ import annotations

import json

import pytest

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import (
    ArucoDetectionConfig,
    Config,
    PathsConfig,
    PerturbationConfig,
    Resolution,
    SourceConfig,
    SourceType,
)
from poker_vision.detection.factory import create_detector
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.mock_aruco import MockArucoDetector
from poker_vision.detection.mock_perturbation import PerturbedDetector
from poker_vision.detection.models import DetectionClass

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


def _calibration() -> CalibrationRuntime:
    seat = CalibrationSeat(
        seat_id="seat_1",
        zones=SeatZones(
            player_area=_polygon((0, 0), (50, 0), (50, 50), (0, 50)),
            chip_zone=_polygon((10, 10), (30, 10), (30, 30), (10, 30)),
        ),
    )
    return CalibrationRuntime(
        schema_version="1.0",
        table_id="t",
        based_on="test",
        inference_resolution=Resolution(width=100, height=100),
        camera=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=50.0, cy=50.0),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM),
        seats=[seat],
        zones=GlobalZones(
            board_zone=_polygon((60, 60), (90, 60), (90, 90), (60, 90)),
            dealer_area=_polygon((0, 60), (20, 60), (20, 80), (0, 80)),
        ),
    )


def _config(paths: PathsConfig, **overrides: object) -> Config:
    return Config(
        schema_version="1.0",
        device="cpu",
        source=SourceConfig(type=SourceType.IMAGE_DIR, path="images"),
        paths=paths,
        **overrides,
    )


def _paths(**kwargs: object) -> PathsConfig:
    return PathsConfig(
        calibration_authoring="calib_authoring.json",
        calibration_runtime="calib_runtime.json",
        jsonl_export_dir="events",
        **kwargs,
    )


def test_create_detector_selects_mock_script_mode(tmp_path):
    script = tmp_path / "script.jsonl"
    script.write_text(json.dumps({"frame_index": 0, "detections": []}) + "\n")
    config = _config(_paths(mock_script=script))

    detector = create_detector(config, _calibration())

    assert isinstance(detector, MockDetector)


def test_create_detector_selects_aruco_mode():
    config = _config(
        _paths(),
        aruco=ArucoDetectionConfig(marker_class_map={0: DetectionClass.CHIP}),
    )

    detector = create_detector(config, _calibration())

    assert isinstance(detector, MockArucoDetector)


def test_create_detector_wraps_with_perturbation_when_configured(tmp_path):
    script = tmp_path / "script.jsonl"
    script.write_text(json.dumps({"frame_index": 0, "detections": []}) + "\n")
    config = _config(_paths(mock_script=script), perturbation=PerturbationConfig(seed=1))

    detector = create_detector(config, _calibration())

    assert isinstance(detector, PerturbedDetector)


def test_create_detector_rejects_no_mode_selected():
    config = _config(_paths())

    with pytest.raises(ValueError, match="exactly one of"):
        create_detector(config, _calibration())


def test_create_detector_rejects_ambiguous_mode_selection(tmp_path):
    script = tmp_path / "script.jsonl"
    script.write_text(json.dumps({"frame_index": 0, "detections": []}) + "\n")
    config = _config(
        _paths(mock_script=script),
        aruco=ArucoDetectionConfig(marker_class_map={0: DetectionClass.CHIP}),
    )

    with pytest.raises(ValueError, match="exactly one of"):
        create_detector(config, _calibration())
