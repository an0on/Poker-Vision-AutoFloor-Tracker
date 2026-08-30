"""REQ-17: Detector interface, pixel -> table transform, AC-10."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from poker_vision.calibration.geometry import PixelPoint, TablePoint
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.geometry import (
    apply_homography_to_point,
    box_center,
    transform_box_to_table,
)
from poker_vision.detection.models import DetectionClass

VALID_SEATS: list[dict] = [
    {
        "seat_id": "seat_1",
        "zones": {
            "player_area": {
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 100, "y": 0},
                    {"x": 100, "y": 100},
                    {"x": 0, "y": 100},
                ]
            },
            "chip_zone": {
                "points": [
                    {"x": 10, "y": 10},
                    {"x": 50, "y": 10},
                    {"x": 50, "y": 50},
                    {"x": 10, "y": 50},
                ]
            },
        },
    }
]

VALID_ZONES: dict = {
    "board_zone": {
        "points": [
            {"x": 400, "y": 400},
            {"x": 600, "y": 400},
            {"x": 600, "y": 500},
            {"x": 400, "y": 500},
        ]
    },
    "dealer_area": {
        "points": [
            {"x": 700, "y": 700},
            {"x": 750, "y": 700},
            {"x": 750, "y": 750},
            {"x": 700, "y": 750},
        ]
    },
}

# Scale x by 2, y by 3, then translate by (10, 20): table = (2x + 10, 3y + 20).
# Deliberately not the identity matrix, so a test can't pass just because the
# transform happens to be a no-op.
SCALE_TRANSLATE_FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
SCALE_TRANSLATE_INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]

# Rotate 90 degrees about the origin: (x, y) -> (-y, x). Used to check that
# the box transform takes the bounding box of all four transformed corners,
# not a literal corner-to-corner remap (a homography need not stay axis-aligned).
ROTATE_90_FORWARD = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
ROTATE_90_INVERSE = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def _runtime(forward: list[list[float]], inverse: list[list[float]]) -> CalibrationRuntime:
    payload = {
        "schema_version": "1.0",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        "inference_resolution": {"width": 1920, "height": 1080},
        "camera": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "homography": {"forward": forward, "inverse": inverse},
        "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(frame_index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


class _StubDetector(Detector):
    """Minimal `Detector` used only to exercise the base class's transform."""

    def __init__(self, calibration: CalibrationRuntime, raw: list[RawDetection]) -> None:
        super().__init__(calibration)
        self._raw = raw

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        return self._raw


# --- geometry.box_center: Phase 0's verified method -------------------------


def test_box_center_matches_phase0_method():
    assert box_center((0.0, 0.0, 10.0, 20.0)) == PixelPoint(x=5.0, y=10.0)


def test_box_center_matches_phase0_reference_values():
    # Phase 0's own reported centre for the dealer-button placeholder
    # (docs/phase0 run on Test1.jpeg), reconstructed from its box.
    center = box_center((2094.0, 2814.0, 2414.54, 3015.22))
    assert center.x == pytest.approx(2254.27, abs=1.0)
    assert center.y == pytest.approx(2914.61, abs=1.0)


# --- geometry.apply_homography_to_point / transform_box_to_table -----------


def test_apply_homography_to_point_scale_translate():
    homography = HomographyMatrix(forward=SCALE_TRANSLATE_FORWARD, inverse=SCALE_TRANSLATE_INVERSE)
    result = apply_homography_to_point(PixelPoint(x=10.0, y=10.0), homography)
    assert result == TablePoint(x=30.0, y=50.0)


def test_transform_box_to_table_scale_translate():
    homography = HomographyMatrix(forward=SCALE_TRANSLATE_FORWARD, inverse=SCALE_TRANSLATE_INVERSE)
    box = transform_box_to_table((0.0, 0.0, 10.0, 10.0), homography)
    assert box.min == TablePoint(x=10.0, y=20.0)
    assert box.max == TablePoint(x=30.0, y=50.0)


def test_transform_box_to_table_handles_rotation():
    # A homography need not preserve axis alignment; the result must be the
    # bounding box of all four transformed corners, not a corner remap.
    homography = HomographyMatrix(forward=ROTATE_90_FORWARD, inverse=ROTATE_90_INVERSE)
    box = transform_box_to_table((0.0, 0.0, 10.0, 20.0), homography)
    assert box.min == TablePoint(x=-20.0, y=0.0)
    assert box.max == TablePoint(x=0.0, y=10.0)


def test_apply_homography_to_point_rejects_horizon_point():
    # A real, invertible homography whose last row is [1, 0, 0]: at (0, 0)
    # the homogeneous w = 1*0 + 0*0 + 0 = 0, i.e. this specific point (not
    # the whole matrix) maps to the horizon.
    forward = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    inverse = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]
    homography = HomographyMatrix(forward=forward, inverse=inverse)
    with pytest.raises(ValueError, match="horizon"):
        apply_homography_to_point(PixelPoint(x=0.0, y=0.0), homography)


# --- Detector: interface + AC-10 transform boundary -------------------------


def test_detector_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Detector(_runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE))  # type: ignore[abstract]


def test_detect_transforms_center_and_box_into_table_coordinates():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    raw = RawDetection(
        object_class=DetectionClass.CHIP,
        confidence=0.9,
        center=PixelPoint(x=10.0, y=10.0),
        box=(0.0, 0.0, 10.0, 10.0),
    )
    detector = _StubDetector(calibration, [raw])

    result = detector.detect(_frame(frame_index=7))

    assert result.schema_version == "1.0"
    assert result.frame_index == 7
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.object_class is DetectionClass.CHIP
    assert detection.confidence == 0.9
    # AC-10: the output centre is the *transformed* table point, never the
    # raw pixel value the stub handed in.
    assert detection.center == TablePoint(x=30.0, y=50.0)
    assert detection.center != TablePoint(x=raw.center.x, y=raw.center.y)
    assert detection.box is not None
    assert detection.box.min == TablePoint(x=10.0, y=20.0)
    assert detection.box.max == TablePoint(x=30.0, y=50.0)


def test_detect_without_raw_box_leaves_detection_box_none():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    raw = RawDetection(
        object_class=DetectionClass.DEALER_BUTTON,
        confidence=0.5,
        center=PixelPoint(x=0.0, y=0.0),
    )
    detector = _StubDetector(calibration, [raw])

    result = detector.detect(_frame())

    assert result.detections[0].box is None


def test_detect_empty_raw_detections_yields_empty_frame():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    detector = _StubDetector(calibration, [])

    result = detector.detect(_frame(frame_index=3))

    assert result.frame_index == 3
    assert result.detections == []
