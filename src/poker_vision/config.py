"""Central runtime configuration (REQ-2).

Every module reads its settings from a `Config` instance instead of
environment variables or hard-coded constants. `Config` and all its nested
models are Pydantic v2 models with `schema_version` and reject unknown
fields (REQ-4).
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from poker_vision.schema_base import StrictModel

CONFIG_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class DeviceType(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    # Reserved for the later Windows/TD phase. Rejected in v0.1 by
    # Config._reject_cuda_device below — this is the only place in `src/`
    # where the string "cuda" may legitimately appear (REQ-3, AC-2).
    CUDA = "cuda"


class SourceType(StrEnum):
    CONTINUITY = "continuity"
    VIDEO_FILE = "video_file"
    IMAGE_DIR = "image_dir"


class Resolution(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SourceConfig(StrictModel):
    type: SourceType
    device_index: int | None = Field(default=None, ge=0)
    path: Path | None = None
    resolution_cap: Resolution = Field(default_factory=lambda: Resolution(width=1920, height=1080))

    @model_validator(mode="after")
    def _check_required_field_for_type(self) -> SourceConfig:
        if self.type is SourceType.CONTINUITY and self.device_index is None:
            raise ValueError("source.device_index is required when source.type is 'continuity'")
        if self.type is not SourceType.CONTINUITY and self.path is None:
            raise ValueError(f"source.path is required when source.type is '{self.type.value}'")
        return self


class HysteresisOverride(StrictModel):
    n_on: int | None = Field(default=None, ge=1)
    n_off: int | None = Field(default=None, ge=1)


class HysteresisConfig(StrictModel):
    n_on: int = Field(default=3, ge=1)
    n_off: int = Field(default=3, ge=1)
    per_class: dict[str, HysteresisOverride] = Field(default_factory=dict)


class ThresholdsConfig(StrictModel):
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    tracking_max_distance: float = Field(default=0.05, gt=0.0)
    dealer_nearest_seat_max_distance: float = Field(default=0.1, gt=0.0)


class PortsConfig(StrictModel):
    websocket: int = Field(default=8765, ge=1, le=65535)
    rest: int = Field(default=8000, ge=1, le=65535)
    mjpeg: int = Field(default=8001, ge=1, le=65535)

    @model_validator(mode="after")
    def _check_ports_distinct(self) -> PortsConfig:
        values = [self.websocket, self.rest, self.mjpeg]
        if len(set(values)) != len(values):
            raise ValueError("ports.websocket, ports.rest and ports.mjpeg must be distinct")
        return self


class PathsConfig(StrictModel):
    calibration_authoring: Path
    calibration_runtime: Path
    jsonl_export_dir: Path
    mock_script: Path | None = None


class Config(StrictModel):
    schema_version: Literal["1.0"]
    device: DeviceType
    source: SourceConfig
    paths: PathsConfig
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    ports: PortsConfig = Field(default_factory=PortsConfig)

    @field_validator("device")
    @classmethod
    def _reject_cuda_device(cls, value: DeviceType) -> DeviceType:
        if value is DeviceType.CUDA:
            raise ValueError(
                "device 'cuda' is reserved and not supported in v0.1; use 'cpu' or 'mps'"
            )
        return value


def load_config(path: str | Path) -> Config:
    """Load and validate a Config from a JSON file. Raises on any schema violation."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path}: not valid JSON ({exc})") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{config_path}: invalid config ({exc})") from exc
