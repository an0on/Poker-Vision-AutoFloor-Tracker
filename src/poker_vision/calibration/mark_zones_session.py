"""Click-collection state machine for `calib mark-zones` (REQ-10a).

Deliberately has no OpenCV/display dependency: `ClickSession` only reacts to
abstract `add_point`/`undo`/`finish_polygon` calls and exposes its current
step + collected state for a caller to render. This is what makes it
unit-testable without a display -- the actual interactive tool
(`mark_zones_interactive.py`) is a thin cv2 mouse-callback wrapper around
it, and stays untested the way every other display-driving code in this
project does (there is no headless CI display, REQ-39/41).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto

from poker_vision.calibration.mark_zones import ArcClick, MarkedZones, Point

_SEAT_COUNT = 10
_OVAL_CLICKS_PER_END = 3
_OVAL_CLICKS_TOTAL = _OVAL_CLICKS_PER_END * 2
_BOARD_ZONE_CLICKS = 4
_MIN_POLYGON_POINTS = 3


class Step(StrEnum):
    SEATS = auto()
    PICK_DEALER = auto()
    INNER_OVAL = auto()
    OUTER_OVAL = auto()
    BOARD_ZONE = auto()
    DONE = auto()


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon, good enough for picking a seat by click.

    Not `topology.point_in_polygon` (REQ-11's exact triangulation-based
    one): that operates on validated `TablePolygon`s, and a seat clicked
    together here hasn't been validated yet -- this only has to pick the
    right seat for a click roughly inside it, not certify geometry.
    """
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


@dataclass
class ClickSession:
    """Collects one `calib mark-zones` session's worth of clicks.

    `image_size` is fixed at construction (needed for the eventual
    `MarkedZones.image_size`); everything else accumulates as
    `add_point`/`finish_polygon`/`pick_dealer_at` are called, in the fixed
    step order `Step` lists. Raises `ValueError` for any call invalid in
    the current step (e.g. `pick_dealer_at` before all 10 seats exist) --
    callers (the interactive tool) are expected to only offer the actions
    valid for `self.step`.
    """

    image_size: tuple[int, int]
    step: Step = Step.SEATS
    seats: dict[str, list[Point]] = field(default_factory=dict)
    dealer_seat_key: str | None = None
    board_zone_points: list[Point] = field(default_factory=list)
    inner_oval_points: list[Point] = field(default_factory=list)
    outer_oval_points: list[Point] = field(default_factory=list)
    _current_polygon: list[Point] = field(default_factory=list)
    _next_seat_number: int = 1

    @property
    def current_polygon(self) -> list[Point]:
        """The in-progress seat polygon's points so far (SEATS step only; empty otherwise)."""
        return list(self._current_polygon)

    def add_point(self, point: Point) -> None:
        """Add one clicked point to whatever the current step is collecting."""
        if self.step is Step.SEATS:
            self._current_polygon.append(point)
        elif self.step is Step.INNER_OVAL:
            self.inner_oval_points.append(point)
            self._advance_oval_if_complete(Step.INNER_OVAL, Step.OUTER_OVAL)
        elif self.step is Step.OUTER_OVAL:
            self.outer_oval_points.append(point)
            self._advance_oval_if_complete(Step.OUTER_OVAL, Step.BOARD_ZONE)
        elif self.step is Step.BOARD_ZONE:
            self.board_zone_points.append(point)
            if len(self.board_zone_points) == _BOARD_ZONE_CLICKS:
                self.step = Step.DONE
        else:
            raise ValueError(f"add_point is not valid in step {self.step}")

    def _advance_oval_if_complete(self, current: Step, next_step: Step) -> None:
        points = self.inner_oval_points if current is Step.INNER_OVAL else self.outer_oval_points
        if len(points) == _OVAL_CLICKS_TOTAL:
            self.step = next_step

    def undo(self) -> None:
        """Remove the most recently added point, anywhere it landed.

        INNER_OVAL's 6th point, OUTER_OVAL's 6th, and BOARD_ZONE's 4th each
        auto-advance the step the instant they're added (`add_point`), so a
        mistaken final click there leaves the *new* step's own buffer empty
        -- undo has to walk back across that boundary and pop the point
        that actually caused the transition, not silently no-op. Repeated
        calls chain naturally back through INNER_OVAL -> OUTER_OVAL ->
        BOARD_ZONE -> DONE this way.

        Does *not* reopen an already-committed seat (SEATS -> PICK_DEALER,
        `finish_polygon`): unlike the above, that transition is always an
        explicit, deliberate action (Enter/Space), never a last-click
        surprise, so there is no accidental point to undo back to -- the
        operator had every chance to fix the polygon before confirming it.
        """
        if self.step is Step.SEATS and self._current_polygon:
            self._current_polygon.pop()
        elif self.step is Step.INNER_OVAL and self.inner_oval_points:
            self.inner_oval_points.pop()
        elif self.step is Step.OUTER_OVAL:
            if self.outer_oval_points:
                self.outer_oval_points.pop()
            elif self.inner_oval_points:
                self.step = Step.INNER_OVAL
                self.inner_oval_points.pop()
        elif self.step is Step.BOARD_ZONE:
            if self.board_zone_points:
                self.board_zone_points.pop()
            elif self.outer_oval_points:
                self.step = Step.OUTER_OVAL
                self.outer_oval_points.pop()
        elif self.step is Step.DONE and self.board_zone_points:
            self.step = Step.BOARD_ZONE
            self.board_zone_points.pop()

    def finish_polygon(self) -> None:
        """Commit the in-progress seat polygon and move on (SEATS step only).

        Auto-advances to `PICK_DEALER` once the 10th seat is committed.
        """
        if self.step is not Step.SEATS:
            raise ValueError(f"finish_polygon is only valid in step SEATS, not {self.step}")
        if len(self._current_polygon) < _MIN_POLYGON_POINTS:
            raise ValueError(
                f"a seat polygon needs at least {_MIN_POLYGON_POINTS} points, "
                f"got {len(self._current_polygon)}"
            )
        key = f"click_{self._next_seat_number}"
        self.seats[key] = list(self._current_polygon)
        self._next_seat_number += 1
        self._current_polygon = []
        if len(self.seats) == _SEAT_COUNT:
            self.step = Step.PICK_DEALER

    def pick_dealer_at(self, point: Point) -> None:
        """Mark whichever already-committed seat contains `point` as the dealer seat."""
        if self.step is not Step.PICK_DEALER:
            raise ValueError(f"pick_dealer_at is only valid in step PICK_DEALER, not {self.step}")
        for key, polygon in self.seats.items():
            if _point_in_polygon(point, polygon):
                self.dealer_seat_key = key
                self.step = Step.INNER_OVAL
                return
        raise ValueError(f"{point} is not inside any marked seat")

    def _oval_arc_clicks(self, points: list[Point]) -> tuple[ArcClick, ArcClick]:
        (a_start, a_center, a_end, b_start, b_center, b_end) = points
        return (
            ArcClick(start=a_start, center=a_center, end=a_end),
            ArcClick(start=b_start, center=b_center, end=b_end),
        )

    def build(self) -> MarkedZones:
        """Assemble the completed session into a `MarkedZones` (REQ-10a).

        Raises `ValueError` if called before step `DONE`.
        """
        if self.step is not Step.DONE:
            raise ValueError(f"session is not complete yet (currently in step {self.step})")
        assert self.dealer_seat_key is not None  # guaranteed by PICK_DEALER having completed
        return MarkedZones(
            seat_polygons=dict(self.seats),
            dealer_seat_key=self.dealer_seat_key,
            board_zone_points=list(self.board_zone_points),
            inner_oval=self._oval_arc_clicks(self.inner_oval_points),
            outer_oval=self._oval_arc_clicks(self.outer_oval_points),
            image_size=self.image_size,
        )
