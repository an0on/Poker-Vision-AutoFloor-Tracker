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

from poker_vision.calibration.mark_zones import MarkedZones, Point

_SEAT_COUNT = 10
_BOARD_ZONE_CLICKS = 4
_MIN_POLYGON_POINTS = 3


class Step(StrEnum):
    SEATS = auto()
    PICK_DEALER = auto()
    INNER_OVAL = auto()
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
    `add_point`/`finish_polygon`/`pick_dealer_at`/`finish_inner_oval` are
    called, in the fixed step order `Step` lists. Raises `ValueError` for
    any call invalid in the current step (e.g. `pick_dealer_at` before all
    10 seats exist) -- callers (the interactive tool) are expected to only
    offer the actions valid for `self.step`.
    """

    image_size: tuple[int, int]
    step: Step = Step.SEATS
    seats: dict[str, list[Point]] = field(default_factory=dict)
    dealer_seat_key: str | None = None
    inner_oval_points: list[Point] = field(default_factory=list)
    board_zone_points: list[Point] = field(default_factory=list)
    _current_polygon: list[Point] = field(default_factory=list)
    _next_seat_number: int = 1

    @property
    def current_polygon(self) -> list[Point]:
        """The in-progress polygon's points so far: the current seat (SEATS
        step) or the inner-oval trace (INNER_OVAL step); empty otherwise.
        """
        if self.step is Step.INNER_OVAL:
            return list(self.inner_oval_points)
        return list(self._current_polygon)

    def add_point(self, point: Point) -> None:
        """Add one clicked point to whatever the current step is collecting."""
        if self.step is Step.SEATS:
            self._current_polygon.append(point)
        elif self.step is Step.INNER_OVAL:
            self.inner_oval_points.append(point)
        elif self.step is Step.BOARD_ZONE:
            self.board_zone_points.append(point)
            if len(self.board_zone_points) == _BOARD_ZONE_CLICKS:
                self.step = Step.DONE
        else:
            raise ValueError(f"add_point is not valid in step {self.step}")

    def undo(self) -> None:
        """Remove the most recently added point, anywhere it landed.

        BOARD_ZONE's 4th point auto-advances the step to DONE the instant
        it's added (`add_point`), so a mistaken final click there leaves
        DONE's own state with nothing to pop -- undo has to walk back
        across that boundary and pop the point that actually caused the
        transition, not silently no-op.

        Does *not* reopen an already-committed seat (SEATS -> PICK_DEALER,
        `finish_polygon`), the dealer pick itself (PICK_DEALER ->
        INNER_OVAL, `pick_dealer_at`), or a finished inner-oval trace
        (INNER_OVAL -> BOARD_ZONE, `finish_inner_oval`): unlike
        BOARD_ZONE's last click, every one of those transitions is always
        an explicit, deliberate action (a click on a specific seat, or an
        Enter/Space keypress), never a last-click surprise, so there is no
        accidental point to undo back to -- the operator had every chance
        to fix their choice before confirming it.
        """
        if self.step is Step.SEATS and self._current_polygon:
            self._current_polygon.pop()
        elif self.step is Step.INNER_OVAL and self.inner_oval_points:
            self.inner_oval_points.pop()
        elif self.step is Step.BOARD_ZONE and self.board_zone_points:
            self.board_zone_points.pop()
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

    def seat_at(self, point: Point) -> str | None:
        """The key of whichever marked seat contains `point`, or `None`."""
        for key, polygon in self.seats.items():
            if _point_in_polygon(point, polygon):
                return key
        return None

    def pick_dealer_at(self, point: Point) -> None:
        """Mark whichever already-committed seat contains `point` as the dealer seat."""
        if self.step is not Step.PICK_DEALER:
            raise ValueError(f"pick_dealer_at is only valid in step PICK_DEALER, not {self.step}")
        key = self.seat_at(point)
        if key is None:
            raise ValueError(f"{point} is not inside any marked seat")
        self.dealer_seat_key = key
        self.step = Step.INNER_OVAL

    def finish_inner_oval(self) -> None:
        """Commit the freehand inner-oval trace and move on (INNER_OVAL step only)."""
        if self.step is not Step.INNER_OVAL:
            raise ValueError(f"finish_inner_oval is only valid in step INNER_OVAL, not {self.step}")
        if len(self.inner_oval_points) < _MIN_POLYGON_POINTS:
            raise ValueError(
                f"the inner oval trace needs at least {_MIN_POLYGON_POINTS} points, "
                f"got {len(self.inner_oval_points)}"
            )
        self.step = Step.BOARD_ZONE

    def reopen_seat(self, seat_key: str) -> None:
        """Move an already-committed seat's points back into the
        in-progress buffer for re-clicking (DONE step only) -- the
        recovery path for a save-time REQ-11 rejection of that seat's
        geometry (see `mark_zones.py`'s error messages, which name the
        seat to re-trace). Without this, that "re-trace this seat"
        guidance couldn't actually be acted on short of aborting and
        re-clicking all 10 seats from scratch.

        Also discards the dealer pick, inner-oval trace and board_zone
        points: `pick_dealer_at` needs an intact set of seats to search
        (the seat being edited is briefly gone from `self.seats`), and if
        the reopened seat *was* the dealer seat, `dealer_seat_key` would
        otherwise point at a seat that no longer exists -- simpler and
        safer to have every downstream step redone than to work out case
        by case which of them are still consistent.
        """
        if self.step is not Step.DONE:
            raise ValueError(f"reopen_seat is only valid in step DONE, not {self.step}")
        if seat_key not in self.seats:
            raise ValueError(f"'{seat_key}' is not a marked seat")
        self._current_polygon = self.seats.pop(seat_key)
        self.dealer_seat_key = None
        self.inner_oval_points = []
        self.board_zone_points = []
        self.step = Step.SEATS

    def reopen_inner_oval(self) -> None:
        """Return to INNER_OVAL for a fresh trace (DONE step only), the
        `reopen_seat` recovery path's counterpart for a save-time REQ-11
        rejection of `dealer_area` itself rather than a seat. Also
        discards board_zone's points, which were only ever valid relative
        to the trace being replaced.
        """
        if self.step is not Step.DONE:
            raise ValueError(f"reopen_inner_oval is only valid in step DONE, not {self.step}")
        self.inner_oval_points = []
        self.board_zone_points = []
        self.step = Step.INNER_OVAL

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
            inner_oval_points=list(self.inner_oval_points),
            board_zone_points=list(self.board_zone_points),
            image_size=self.image_size,
        )
