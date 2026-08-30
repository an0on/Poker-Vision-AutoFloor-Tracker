"""Shared board-card counting primitive for REQ-31/REQ-32.

`StreetTracker` (REQ-31) and `HandTracker` (REQ-32) both derive their state
purely from how many `card` tracks currently land in the table's single
`board_zone` -- `StreetTracker` maps the count to a street, `HandTracker`
only cares whether it is zero or not. That count is the one piece of
detection logic they actually share ("dieselbe Board-Übergangserkennung"),
so it lives here once rather than being computed twice and risking drift
between the two trackers.
"""

from __future__ import annotations

from poker_vision.assignment.models import FrameAssignments, ZoneKind
from poker_vision.detection.models import DetectionClass


def count_board_cards(frame_assignments: FrameAssignments) -> int:
    """Number of stable `card` tracks currently assigned to `board_zone`."""
    return sum(
        1
        for assignment in frame_assignments.assignments
        if assignment.zone is ZoneKind.BOARD_ZONE
        and assignment.object_class is DetectionClass.CARD
    )
