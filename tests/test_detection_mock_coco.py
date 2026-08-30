"""REQ-20: `mock` detector, Modus C (pretrained COCO model), AC-11 slice."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import CocoDetectionConfig, DeviceType
from poker_vision.detection.mock_coco import CocoMockDetector
from poker_vision.detection.models import DetectionClass

# Identity homography + zero distortion, so table coordinates equal pixel
# coordinates exactly -- AC-11's "Toleranz 1 px im Pixelraum" then applies
# directly to `detection.center` without an extra conversion step.
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ZERO_DISTORTION = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}

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
            {"x": 4000, "y": 4000},
            {"x": 6000, "y": 4000},
            {"x": 6000, "y": 5000},
            {"x": 4000, "y": 5000},
        ]
    },
    "dealer_area": {
        "points": [
            {"x": 7000, "y": 7000},
            {"x": 7500, "y": 7000},
            {"x": 7500, "y": 7500},
            {"x": 7000, "y": 7500},
        ]
    },
}

PHASE0_IMAGE = Path(__file__).parent.parent / "docs" / "phase0" / "Test1.jpeg"

# Recorded phase0_poc.py output for docs/phase0/Test1.jpeg at conf=0.25,
# device=mps (docs/phase0/README.md). AC-11 requires Modus C to reproduce
# these centres within 1 px.
PHASE0_MOUSE_CENTER = (2254.27, 2914.61)
PHASE0_CELL_PHONE_CENTER = (987.17, 3027.89)


def _runtime(width: int, height: int) -> CalibrationRuntime:
    payload = {
        "schema_version": "1.0",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        "inference_resolution": {"width": width, "height": height},
        "camera": {"fx": 1000.0, "fy": 1000.0, "cx": width / 2, "cy": height / 2},
        "distortion": ZERO_DISTORTION,
        "homography": {"forward": IDENTITY, "inverse": IDENTITY},
        "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(image: np.ndarray, frame_index: int = 0) -> Frame:
    return Frame(
        image=image,
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


def _config(class_map: dict[str, str]) -> CocoDetectionConfig:
    return CocoDetectionConfig(class_map=class_map)


# --- config validation --------------------------------------------------------


def test_empty_class_map_is_rejected():
    with pytest.raises(ValidationError, match="class_map"):
        CocoDetectionConfig(class_map={})


def test_unknown_object_class_in_class_map_is_rejected():
    with pytest.raises(ValidationError):
        CocoDetectionConfig(class_map={"mouse": "not_a_real_class"})


def test_default_model_path_is_yolov8n():
    config = CocoDetectionConfig(class_map={"mouse": "dealer_button"})
    assert config.model_path == Path("yolov8n.pt")


# --- detection behaviour --------------------------------------------------------


def test_blank_image_yields_no_detections():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    detector = CocoMockDetector(
        _runtime(640, 480),
        _config({"mouse": "dealer_button", "cell phone": "chip"}),
        device=DeviceType.CPU,
        confidence_threshold=0.25,
    )

    result = detector.detect(_frame(image))

    assert result.detections == []


def test_frame_index_is_preserved():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    detector = CocoMockDetector(
        _runtime(640, 480),
        _config({"mouse": "dealer_button"}),
        device=DeviceType.CPU,
        confidence_threshold=0.25,
    )

    result = detector.detect(_frame(image, frame_index=7))

    assert result.frame_index == 7


# --- AC-11: reproduces phase0_poc.py's centres on the phase0 test image ------


@pytest.mark.skipif(not PHASE0_IMAGE.exists(), reason="phase0 fixture image not present")
def test_coco_mode_reproduces_phase0_centers_and_classes():
    image = cv2.imread(str(PHASE0_IMAGE))
    assert image is not None
    height, width = image.shape[:2]
    detector = CocoMockDetector(
        _runtime(width, height),
        _config({"mouse": "dealer_button", "cell phone": "chip"}),
        device=DeviceType.CPU,
        confidence_threshold=0.25,
    )

    result = detector.detect(_frame(image))

    by_class = {d.object_class: d for d in result.detections}
    assert DetectionClass.DEALER_BUTTON in by_class
    assert DetectionClass.CHIP in by_class

    dealer_button = by_class[DetectionClass.DEALER_BUTTON]
    assert dealer_button.center.x == pytest.approx(PHASE0_MOUSE_CENTER[0], abs=1.0)
    assert dealer_button.center.y == pytest.approx(PHASE0_MOUSE_CENTER[1], abs=1.0)

    chip = by_class[DetectionClass.CHIP]
    assert chip.center.x == pytest.approx(PHASE0_CELL_PHONE_CENTER[0], abs=1.0)
    assert chip.center.y == pytest.approx(PHASE0_CELL_PHONE_CENTER[1], abs=1.0)
