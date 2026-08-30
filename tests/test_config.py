import json

import pytest
from pydantic import ValidationError

from poker_vision.config import Config, load_config

VALID_CONFIG: dict = {
    "schema_version": "1.0",
    "device": "cpu",
    "source": {
        "type": "image_dir",
        "path": "data/raw/images",
    },
    "paths": {
        "calibration_authoring": "calibration/instance.json",
        "calibration_runtime": "calibration/runtime.json",
        "jsonl_export_dir": "data/events",
    },
}


def _config(**overrides: dict) -> dict:
    merged = json.loads(json.dumps(VALID_CONFIG))
    for key, value in overrides.items():
        merged[key] = value
    return merged


def test_valid_config_loads_with_defaults():
    config = Config.model_validate(VALID_CONFIG)
    assert config.schema_version == "1.0"
    assert config.device.value == "cpu"
    assert config.thresholds.detection_confidence == 0.25
    assert config.hysteresis.n_on == 3
    assert config.ports.websocket == 8765
    assert config.source.resolution_cap.width == 1920
    assert config.source.resolution_cap.height == 1080


# AC-3: wrong schema_version fails
def test_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_config(schema_version="2.0"))


def test_missing_schema_version_rejected():
    payload = json.loads(json.dumps(VALID_CONFIG))
    del payload["schema_version"]
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


# AC-3: unknown top-level field fails
def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_config(unexpected_field="nope"))


# AC-3: unknown nested field fails
def test_unknown_nested_field_rejected():
    payload = _config()
    payload["thresholds"] = {"detection_confidence": 0.5, "typo_field": 1}
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


# AC-2 / REQ-3: device 'cuda' rejected with a clear message
def test_device_cuda_rejected_with_clear_message():
    with pytest.raises(ValidationError, match="reserved and not supported in v0.1"):
        Config.model_validate(_config(device="cuda"))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_device_cpu_and_mps_accepted(device):
    config = Config.model_validate(_config(device=device))
    assert config.device.value == device


def test_unknown_device_value_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_config(device="tpu"))


# AC-13 / REQ-22: detector defaults to mock, the only detector v0.1 ships.
def test_detector_defaults_to_mock():
    config = Config.model_validate(VALID_CONFIG)
    assert config.detector.value == "mock"


def test_detector_mock_accepted():
    config = Config.model_validate(_config(detector="mock"))
    assert config.detector.value == "mock"


# AC-13 (REQ-22): selecting `detector: yolo` fails at config load with a
# message pointing at v0.2, mirroring device 'cuda' (REQ-3/AC-2).
def test_detector_yolo_rejected_with_v02_hint():
    with pytest.raises(ValidationError, match="not available in v0.1") as exc_info:
        Config.model_validate(_config(detector="yolo"))
    assert "v0.2" in str(exc_info.value)


def test_unknown_detector_value_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_config(detector="ssd"))


def test_continuity_source_requires_device_index():
    payload = _config(source={"type": "continuity"})
    with pytest.raises(ValidationError, match="device_index is required"):
        Config.model_validate(payload)


def test_continuity_source_with_device_index_accepted():
    payload = _config(source={"type": "continuity", "device_index": 0})
    config = Config.model_validate(payload)
    assert config.source.device_index == 0


@pytest.mark.parametrize("source_type", ["video_file", "image_dir"])
def test_file_backed_source_requires_path(source_type):
    payload = _config(source={"type": source_type})
    with pytest.raises(ValidationError, match="path is required"):
        Config.model_validate(payload)


def test_ports_must_be_distinct():
    payload = _config(ports={"websocket": 9000, "rest": 9000, "mjpeg": 9001})
    with pytest.raises(ValidationError, match="must be distinct"):
        Config.model_validate(payload)


def test_hysteresis_per_class_override():
    payload = _config(
        hysteresis={"n_on": 3, "n_off": 3, "per_class": {"chip": {"n_on": 5, "n_off": 2}}}
    )
    config = Config.model_validate(payload)
    assert config.hysteresis.per_class["chip"].n_on == 5
    assert config.hysteresis.per_class["chip"].n_off == 2


# Codex finding: a typo'd or unsupported per_class key (e.g. "chips") must
# fail config validation, not silently never match any track's class and
# fall back to the global thresholds.
def test_hysteresis_per_class_unknown_key_rejected():
    payload = _config(
        hysteresis={"n_on": 3, "n_off": 3, "per_class": {"chips": {"n_on": 5}}}
    )
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


def test_load_config_from_json_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(VALID_CONFIG))
    config = load_config(config_path)
    assert config.schema_version == "1.0"


def test_load_config_invalid_json_raises(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(config_path)


def test_load_config_schema_violation_raises(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(device="cuda")))
    with pytest.raises(ValueError, match="invalid config"):
        load_config(config_path)
