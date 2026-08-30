"""Zone assignment output schema (REQ-4, REQ-26).

`assign_zones` (see `zone_assignment.py`) turns each stable `TrackedObject`
into at most one `ZoneAssignment`: which zone it landed in, and — for
zones that belong to a specific seat rather than the whole table —
which seat. Purely geometric (REQ-26's own scope): no occupancy/street/
dealer-seat decision lives here, that is `state`'s job (REQ-29 ff.).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from poker_vision.detection.models import DetectionClass
from poker_vision.schema_base import StrictModel

ASSIGNMENT_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class ZoneKind(StrEnum):
    CHIP_ZONE = "chip_zone"
    PLAYER_AREA = "player_area"
    BOARD_ZONE = "board_zone"
    DEALER_AREA = "dealer_area"


# Zone kinds that belong to one specific seat rather than the whole table;
# a `ZoneAssignment` for one of these always carries `seat_id`, the global
# ones (`board_zone`, `dealer_area`) never do (see `_check_seat_id_matches_zone`).
_SEAT_ZONE_KINDS = frozenset({ZoneKind.CHIP_ZONE, ZoneKind.PLAYER_AREA})


class ZoneAssignment(StrictModel):
    schema_version: Literal["1.0"]
    track_id: int = Field(ge=1)
    object_class: DetectionClass
    zone: ZoneKind
    seat_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_seat_id_matches_zone(self) -> ZoneAssignment:
        if self.zone in _SEAT_ZONE_KINDS and self.seat_id is None:
            raise ValueError(f"zone '{self.zone.value}' requires a seat_id")
        if self.zone not in _SEAT_ZONE_KINDS and self.seat_id is not None:
            raise ValueError(f"zone '{self.zone.value}' must not carry a seat_id")
        return self


class FrameAssignments(StrictModel):
    schema_version: Literal["1.0"]
    frame_index: int = Field(ge=0)
    assignments: list[ZoneAssignment] = Field(default_factory=list)
