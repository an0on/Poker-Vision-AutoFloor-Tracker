"""Interactive click-based tool for `calib mark-zones` (REQ-10a).

Thin cv2 window/mouse-callback wrapper around `ClickSession` (the actual
step machine -- see that module's docstring for why this split exists).
Untested here: there's no headless-CI display to open a real cv2 window
against (REQ-39/41), the same reason every other window/camera-driving
file in this project (`capture/continuity.py`, `debug/mjpeg.py`'s own
server loop) has no direct test coverage either -- the logic that
*matters* to get right lives in `mark_zones.py`/`mark_zones_session.py`,
both fully unit-tested.

Controls: left-click adds a point; Enter/Space finishes the current seat
polygon (SEATS step); Backspace/'u' undoes the last point; 's' saves once
the session reaches DONE; Esc aborts without writing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from poker_vision.calibration.authoring import write_calibration_authoring
from poker_vision.calibration.mark_zones import (
    DEFAULT_CHIP_ZONE_SHRINK_FACTOR,
    Point,
    build_authoring_from_marked_zones,
)
from poker_vision.calibration.mark_zones_session import ClickSession, Step

_WINDOW_NAME = "calib mark-zones"

_COLOR_SEAT_DONE = (200, 120, 0)
_COLOR_SEAT_CURRENT = (0, 255, 255)
_COLOR_DEALER = (0, 0, 255)
_COLOR_OVAL_DONE = (120, 120, 0)
_COLOR_OVAL_CURRENT = (0, 200, 200)
_COLOR_BOARD = (255, 0, 255)
_COLOR_TEXT = (255, 255, 255)
_COLOR_TEXT_BG = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INSTRUCTION_BAR_HEIGHT = 30

_KEY_ESC = 27
_KEY_ENTER = 13
_KEY_SPACE = 32
_KEY_BACKSPACE = 8

# Reference photos come straight off a phone (e.g. 4032x3024) -- far bigger
# than the window fits on a typical screen at 1:1. Scaled down for display
# only; every click is converted back to full-resolution image coordinates
# ourselves (see `run_interactive_mark_zones`) rather than trusting a
# resizable window's backend to report already-unscaled coordinates, which
# isn't consistent across OpenCV's GUI backends/platforms.
_MAX_DISPLAY_DIMENSION = 1400

_STEP_INSTRUCTIONS: dict[Step, str] = {
    Step.SEATS: "Click a seat's player_area corners, Enter/Space to finish it (need 10 seats)",
    Step.PICK_DEALER: "Click inside the seat that is the fixed card-dealer (Kartengeber) position",
    Step.INNER_OVAL: "Click inner oval: end A start/center/end, end B start/center/end (6 pts)",
    Step.OUTER_OVAL: "Click outer oval: end A start/center/end, end B start/center/end (6 pts)",
    Step.BOARD_ZONE: "Click board_zone's 4 corners",
    Step.DONE: "Done -- 's' to save, Esc to discard",
}


def _draw_polyline(
    image: np.ndarray, points: list[Point], color: tuple[int, int, int], closed: bool
) -> None:
    for x, y in points:
        cv2.circle(image, (round(x), round(y)), 4, color, -1)
    if len(points) >= 2:
        array = np.array([(round(x), round(y)) for x, y in points], dtype=np.int32)
        cv2.polylines(
            image, [array], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA
        )


def _render(base_image: np.ndarray, session: ClickSession) -> np.ndarray:
    image = base_image.copy()
    for key, points in session.seats.items():
        color = _COLOR_DEALER if key == session.dealer_seat_key else _COLOR_SEAT_DONE
        _draw_polyline(image, points, color, closed=True)
    if session.step is Step.SEATS:
        _draw_polyline(image, session.current_polygon, _COLOR_SEAT_CURRENT, closed=False)

    step_order = list(Step)
    current_index = step_order.index(session.step)
    if session.step is Step.INNER_OVAL:
        _draw_polyline(image, session.inner_oval_points, _COLOR_OVAL_CURRENT, closed=False)
    elif current_index > step_order.index(Step.INNER_OVAL):
        _draw_polyline(image, session.inner_oval_points, _COLOR_OVAL_DONE, closed=False)
    if session.step is Step.OUTER_OVAL:
        _draw_polyline(image, session.outer_oval_points, _COLOR_OVAL_CURRENT, closed=False)
    elif current_index > step_order.index(Step.OUTER_OVAL):
        _draw_polyline(image, session.outer_oval_points, _COLOR_OVAL_DONE, closed=False)

    if session.step is Step.BOARD_ZONE:
        _draw_polyline(image, session.board_zone_points, _COLOR_BOARD, closed=False)
    elif session.step is Step.DONE:
        _draw_polyline(image, session.board_zone_points, _COLOR_BOARD, closed=True)

    instructions = _STEP_INSTRUCTIONS[session.step]
    if session.step is Step.SEATS:
        instructions += f" ({len(session.seats)}/10 done)"
    cv2.rectangle(image, (0, 0), (image.shape[1], _INSTRUCTION_BAR_HEIGHT), _COLOR_TEXT_BG, -1)
    cv2.putText(image, instructions, (8, 20), _FONT, 0.6, _COLOR_TEXT, 1, cv2.LINE_AA)
    return image


def run_interactive_mark_zones(
    image_path: Path,
    out_path: Path,
    table_id: str,
    chip_zone_shrink_factor: float = DEFAULT_CHIP_ZONE_SHRINK_FACTOR,
) -> int:
    """Open `image_path` in a click-to-mark window and write the resulting
    `CalibrationAuthoring` to `out_path` on save (REQ-10a). Returns an exit
    code, matching every other `calib` subcommand's contract.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"mark-zones: could not read image '{image_path}'", file=sys.stderr)
        return 1
    height, width = image.shape[:2]
    session = ClickSession(image_size=(width, height))
    display_scale = min(1.0, _MAX_DISPLAY_DIMENSION / max(width, height))
    display_size = (round(width * display_scale), round(height * display_scale))

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = (x / display_scale, y / display_scale)
        try:
            if session.step is Step.PICK_DEALER:
                session.pick_dealer_at(point)
            elif session.step is not Step.DONE:
                session.add_point(point)
        except ValueError as exc:
            print(f"mark-zones: {exc}", file=sys.stderr)

    # AUTOSIZE (not NORMAL): the window can't be resized by the user/window
    # manager, so `display_size` -- and therefore `display_scale` -- stays
    # exactly what `on_mouse` above assumes for the whole session.
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_WINDOW_NAME, on_mouse)

    try:
        while True:
            frame = _render(image, session)
            if display_scale < 1.0:
                frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
            cv2.imshow(_WINDOW_NAME, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == _KEY_ESC:
                print("mark-zones: aborted, nothing written", file=sys.stderr)
                return 1
            if key in (_KEY_ENTER, _KEY_SPACE) and session.step is Step.SEATS:
                try:
                    session.finish_polygon()
                except ValueError as exc:
                    print(f"mark-zones: {exc}", file=sys.stderr)
            if key in (_KEY_BACKSPACE, ord("u")):
                session.undo()
            if key == ord("s") and session.step is Step.DONE:
                break
    finally:
        cv2.destroyWindow(_WINDOW_NAME)

    try:
        marked = session.build()
        authoring = build_authoring_from_marked_zones(
            marked, table_id=table_id, chip_zone_shrink_factor=chip_zone_shrink_factor
        )
    except (ValueError, ValidationError) as exc:
        print(f"mark-zones: {exc}", file=sys.stderr)
        return 1
    write_calibration_authoring(authoring, out_path)
    print(f"mark-zones: wrote '{out_path}'")
    return 0
