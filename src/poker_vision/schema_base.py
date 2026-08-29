"""Shared base for every Pydantic v2 schema in the project (REQ-4).

All data structures — config, calibration, detections, events, state
snapshot — build on `StrictModel` so unknown fields are hard errors
everywhere, not just at the top level of one schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model shared by all schemas: unknown fields are hard errors."""

    model_config = ConfigDict(extra="forbid")
