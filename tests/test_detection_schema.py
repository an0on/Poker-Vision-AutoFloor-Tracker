import json

import pytest
from pydantic import ValidationError

from poker_vision.detection.models import Detection, FrameDetections

VALID_DETECTION: dict = {
    "object_class": "chip",
    "confidence": 0.87,
    "center": {"x": 12.5, "y": 30.0},
    "box": {"min": {"x": 10.0, "y": 27.0}, "max": {"x": 15.0, "y": 33.0}},
}

VALID_FRAME: dict = {
    "schema_version": "1.0",
    "frame_index": 0,
    "detections": [VALID_DETECTION],
}


def _payload(base: dict, **overrides: object) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        merged[key] = value
    return merged


def test_valid_frame_detections_loads():
    frame = FrameDetections.model_validate(VALID_FRAME)
    assert frame.frame_index == 0
    assert frame.detections[0].object_class.value == "chip"
    assert frame.detections[0].box is not None


def test_detection_without_box_is_valid():
    payload = _payload(VALID_DETECTION)
    del payload["box"]
    detection = Detection.model_validate(payload)
    assert detection.box is None


# AC-3: wrong schema_version fails
def test_frame_detections_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        FrameDetections.model_validate(_payload(VALID_FRAME, schema_version="2.0"))


def test_frame_detections_missing_schema_version_rejected():
    payload = json.loads(json.dumps(VALID_FRAME))
    del payload["schema_version"]
    with pytest.raises(ValidationError):
        FrameDetections.model_validate(payload)


# AC-3: unknown top-level field fails
def test_frame_detections_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        FrameDetections.model_validate(_payload(VALID_FRAME, unexpected_field="nope"))


def test_detection_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Detection.model_validate(_payload(VALID_DETECTION, typo_field=1))


def test_detection_unknown_class_rejected():
    with pytest.raises(ValidationError):
        Detection.model_validate(_payload(VALID_DETECTION, object_class="hand"))


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_detection_confidence_out_of_range_rejected(confidence):
    with pytest.raises(ValidationError):
        Detection.model_validate(_payload(VALID_DETECTION, confidence=confidence))


def test_detection_box_min_after_max_rejected():
    payload = _payload(VALID_DETECTION)
    payload["box"] = {"min": {"x": 20.0, "y": 27.0}, "max": {"x": 15.0, "y": 33.0}}
    with pytest.raises(ValidationError):
        Detection.model_validate(payload)


def test_frame_detections_negative_frame_index_rejected():
    with pytest.raises(ValidationError):
        FrameDetections.model_validate(_payload(VALID_FRAME, frame_index=-1))
