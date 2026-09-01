"""REQ-21: `mock` detector perturbation wrapper."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import PerturbationConfig
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.mock_perturbation import PerturbedDetector
from poker_vision.detection.models import DetectionClass

# Identity-ish homography with real distortion, so the pixel -> table
# transform is actually exercised (mirrors test_detection_mock_aruco.py).
FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]
CAMERA = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0}
DISTORTION = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}

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

IMAGE_SIZE = 1920, 1080  # width, height
TABLE_WIDTH = 1200.0
TABLE_HEIGHT = 900.0


def _runtime() -> CalibrationRuntime:
    payload = {
        "schema_version": "1.0",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        "inference_resolution": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "camera": CAMERA,
        "distortion": DISTORTION,
        "homography": {"forward": FORWARD, "inverse": INVERSE},
        "table": {"width": TABLE_WIDTH, "height": TABLE_HEIGHT, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
        "card_dealer_seat_id": "seat_1",
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(frame_index: int = 0) -> Frame:
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 255, dtype=np.uint8)
    return Frame(
        image=image, timestamp=datetime.now(UTC), frame_index=frame_index, source_id="test"
    )


class _FixedDetector(Detector):
    """Test double: always returns the same fixed pixel-space detections."""

    def __init__(self, calibration: CalibrationRuntime, raw_detections: list[RawDetection]) -> None:
        super().__init__(calibration)
        self._raw_detections = raw_detections

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        return list(self._raw_detections)


CHIP_AND_CARD = [
    RawDetection(
        object_class=DetectionClass.CHIP, confidence=0.9, center=PixelPoint(x=500.0, y=500.0)
    ),
    RawDetection(
        object_class=DetectionClass.CARD, confidence=0.8, center=PixelPoint(x=700.0, y=400.0)
    ),
]


def _config(**overrides: object) -> PerturbationConfig:
    return PerturbationConfig(seed=1, **overrides)


# --- passthrough (no perturbation configured) --------------------------------


def test_no_perturbation_passes_through_unchanged():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(calibration, inner, _config())

    inner_result = inner.detect(_frame())
    wrapped_result = wrapped.detect(_frame())

    assert len(wrapped_result.detections) == len(inner_result.detections)
    for inner_detection, wrapped_detection in zip(
        inner_result.detections, wrapped_result.detections, strict=True
    ):
        assert wrapped_detection.object_class is inner_detection.object_class
        assert wrapped_detection.confidence == pytest.approx(inner_detection.confidence)
        assert wrapped_detection.center.x == pytest.approx(inner_detection.center.x, abs=1e-6)
        assert wrapped_detection.center.y == pytest.approx(inner_detection.center.y, abs=1e-6)


def test_frame_index_is_preserved():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(calibration, inner, _config())

    result = wrapped.detect(_frame(frame_index=7))

    assert result.frame_index == 7


# --- frame dropout -------------------------------------------------------------


def test_dropout_probability_one_always_suppresses_frame():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(calibration, inner, _config(dropout_probability=1.0))

    result = wrapped.detect(_frame())

    assert result.detections == []


def test_dropout_probability_zero_never_suppresses_frame():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(calibration, inner, _config(dropout_probability=0.0))

    result = wrapped.detect(_frame())

    assert len(result.detections) == len(CHIP_AND_CARD)


# --- position jitter -------------------------------------------------------------


def test_jitter_moves_detection_away_from_original_position():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    unperturbed_center = inner.detect(_frame()).detections[0].center

    wrapped = PerturbedDetector(calibration, inner, _config(position_jitter_std=50.0))
    jittered_center = wrapped.detect(_frame()).detections[0].center

    assert (
        abs(jittered_center.x - unperturbed_center.x) > 1e-3
        or abs(jittered_center.y - unperturbed_center.y) > 1e-3
    )


def test_jitter_std_zero_leaves_position_unchanged():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    unperturbed_center = inner.detect(_frame()).detections[0].center

    wrapped = PerturbedDetector(calibration, inner, _config(position_jitter_std=0.0))
    result_center = wrapped.detect(_frame()).detections[0].center

    assert result_center.x == pytest.approx(unperturbed_center.x, abs=1e-6)
    assert result_center.y == pytest.approx(unperturbed_center.y, abs=1e-6)


def test_same_seed_reproduces_identical_jitter():
    calibration = _runtime()
    config = _config(position_jitter_std=25.0)

    result_a = PerturbedDetector(
        calibration, _FixedDetector(calibration, CHIP_AND_CARD), config
    ).detect(_frame())
    result_b = PerturbedDetector(
        calibration, _FixedDetector(calibration, CHIP_AND_CARD), config
    ).detect(_frame())

    for detection_a, detection_b in zip(result_a.detections, result_b.detections, strict=True):
        assert detection_a.center.x == pytest.approx(detection_b.center.x, abs=1e-9)
        assert detection_a.center.y == pytest.approx(detection_b.center.y, abs=1e-9)


def test_different_seed_gives_different_jitter():
    calibration = _runtime()

    result_a = PerturbedDetector(
        calibration,
        _FixedDetector(calibration, CHIP_AND_CARD),
        PerturbationConfig(seed=1, position_jitter_std=25.0),
    ).detect(_frame())
    result_b = PerturbedDetector(
        calibration,
        _FixedDetector(calibration, CHIP_AND_CARD),
        PerturbationConfig(seed=2, position_jitter_std=25.0),
    ).detect(_frame())

    first_a = result_a.detections[0].center
    first_b = result_b.detections[0].center
    assert first_a.x != pytest.approx(first_b.x, abs=1e-9) or first_a.y != pytest.approx(
        first_b.y, abs=1e-9
    )


# --- ghost detections -------------------------------------------------------------


def test_ghost_probability_one_adds_extra_detection():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(
        calibration,
        inner,
        _config(
            ghost_probability=1.0,
            ghost_classes=[DetectionClass.DEALER_BUTTON],
            ghost_confidence=0.42,
        ),
    )

    result = wrapped.detect(_frame())

    assert len(result.detections) == len(CHIP_AND_CARD) + 1
    ghosts = [d for d in result.detections if d.object_class is DetectionClass.DEALER_BUTTON]
    assert len(ghosts) == 1
    assert ghosts[0].confidence == pytest.approx(0.42)


def test_ghost_position_is_within_table_bounds():
    calibration = _runtime()
    inner = _FixedDetector(calibration, [])
    wrapped = PerturbedDetector(
        calibration,
        inner,
        _config(ghost_probability=1.0, ghost_classes=[DetectionClass.CHIP]),
    )

    result = wrapped.detect(_frame())

    ghost = result.detections[0]
    tolerance = 1e-6
    assert -tolerance <= ghost.center.x <= TABLE_WIDTH + tolerance
    assert -tolerance <= ghost.center.y <= TABLE_HEIGHT + tolerance


def test_ghost_probability_zero_never_adds_detection():
    calibration = _runtime()
    inner = _FixedDetector(calibration, CHIP_AND_CARD)
    wrapped = PerturbedDetector(calibration, inner, _config(ghost_probability=0.0))

    result = wrapped.detect(_frame())

    assert len(result.detections) == len(CHIP_AND_CARD)


# --- config validation -------------------------------------------------------------


def test_default_config_only_requires_seed():
    config = PerturbationConfig(seed=42)

    assert config.position_jitter_std == 0.0
    assert config.dropout_probability == 0.0
    assert config.ghost_probability == 0.0


def test_positive_ghost_probability_with_empty_ghost_classes_is_rejected():
    with pytest.raises(ValidationError, match="ghost_classes"):
        PerturbationConfig(seed=1, ghost_probability=0.5, ghost_classes=[])


def test_empty_ghost_classes_allowed_when_ghost_probability_is_zero():
    config = PerturbationConfig(seed=1, ghost_probability=0.0, ghost_classes=[])
    assert config.ghost_classes == []
