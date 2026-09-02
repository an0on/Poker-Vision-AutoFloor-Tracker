"""Interactive click-based tool for `calib mark-zones` (REQ-10a).

Thin cv2 window/mouse-callback wrapper around `ClickSession` (the actual
step machine -- see that module's docstring for why this split exists).
Untested here: there's no headless-CI display to open a real cv2 window
against (REQ-39/41), the same reason every other window/camera-driving
file in this project (`capture/continuity.py`, `debug/mjpeg.py`'s own
server loop) has no direct test coverage either -- the logic that
*matters* to get right lives in `mark_zones.py`/`mark_zones_session.py`,
both fully unit-tested.

Controls: left-click adds a point; Enter/Space finishes the current
freehand trace (SEATS step: the current seat polygon; INNER_OVAL step: the
`dealer_area` boundary); Backspace/'u' undoes the last point; 's' saves
once the session reaches DONE; Esc aborts without writing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from poker_vision.calibration.authoring import CalibrationAuthoring, write_calibration_authoring
from poker_vision.calibration.compile import compile_calibration
from poker_vision.calibration.mark_zones import (
    DEFAULT_CHIP_ZONE_INSET_PIXELS,
    Point,
    build_authoring_from_marked_zones,
)
from poker_vision.calibration.mark_zones_session import ClickSession, Step
from poker_vision.debug.overlay import draw_zones

_WINDOW_NAME = "calib mark-zones"

_COLOR_SEAT_DONE = (200, 120, 0)
_COLOR_SEAT_CURRENT = (0, 255, 255)
_COLOR_DEALER = (0, 0, 255)
_COLOR_INNER_OVAL_CURRENT = (0, 200, 200)
_COLOR_INNER_OVAL_DONE = (120, 120, 0)
_COLOR_BOARD = (255, 0, 255)
_COLOR_TEXT = (255, 255, 255)

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
    Step.INNER_OVAL: "Trace dealer_area's inner boundary, Enter/Space when done",
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


def _render_content(base_image: np.ndarray, session: ClickSession) -> np.ndarray:
    """The clickable photo content, at full resolution -- no instruction bar.

    Kept separate from the bar (see `_compose_display_frame`) rather than
    drawing the bar directly onto this frame and resizing the combination
    for display: that would permanently occlude whatever photo content
    happens to fall in the top `_INSTRUCTION_BAR_HEIGHT` pixels, on every
    single frame, for as long as the tool runs. Codex review flagged (as a
    coordinate-mapping bug) what's actually this occlusion risk -- the
    click math itself round-trips exactly either way, verified separately
    -- but drawing the bar outside the content entirely removes the
    occlusion too, which is worth doing regardless.
    """
    image = base_image.copy()
    for key, points in session.seats.items():
        color = _COLOR_DEALER if key == session.dealer_seat_key else _COLOR_SEAT_DONE
        _draw_polyline(image, points, color, closed=True)
    if session.step is Step.SEATS:
        _draw_polyline(image, session.current_polygon, _COLOR_SEAT_CURRENT, closed=False)

    step_order = list(Step)
    if session.step is Step.INNER_OVAL:
        _draw_polyline(image, session.current_polygon, _COLOR_INNER_OVAL_CURRENT, closed=False)
    elif step_order.index(session.step) > step_order.index(Step.INNER_OVAL):
        _draw_polyline(image, session.inner_oval_points, _COLOR_INNER_OVAL_DONE, closed=True)

    if session.step is Step.BOARD_ZONE:
        _draw_polyline(image, session.board_zone_points, _COLOR_BOARD, closed=False)
    elif session.step is Step.DONE:
        _draw_polyline(image, session.board_zone_points, _COLOR_BOARD, closed=True)
    return image


def _compose_display_frame(content: np.ndarray, session: ClickSession) -> np.ndarray:
    """Stack a fixed-height instruction bar above the (already display-sized)
    `content` frame -- a separate strip, not drawn over the content and
    scaled together with it, so nothing photographed near the top edge is
    ever hidden (see `_render_content`'s docstring). `content`'s own pixel
    (0, 0) therefore lands at bar height in the final window image; callers
    converting a click back to full-resolution coordinates must subtract
    `_INSTRUCTION_BAR_HEIGHT` first (see `run_interactive_mark_zones`).
    """
    bar = np.zeros((_INSTRUCTION_BAR_HEIGHT, content.shape[1], 3), dtype=np.uint8)
    instructions = _STEP_INSTRUCTIONS[session.step]
    if session.step is Step.SEATS:
        instructions += f" ({len(session.seats)}/10 done)"
    cv2.putText(bar, instructions, (8, 20), _FONT, 0.6, _COLOR_TEXT, 1, cv2.LINE_AA)
    return np.vstack([bar, content])


def run_interactive_mark_zones(
    image_path: Path,
    out_path: Path,
    table_id: str,
    chip_zone_inset_pixels: float = DEFAULT_CHIP_ZONE_INSET_PIXELS,
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
        # The instruction bar sits above the content (see
        # `_compose_display_frame`), so content pixel (0, 0) is at window
        # y = _INSTRUCTION_BAR_HEIGHT, not 0 -- a click inside the bar
        # itself (negative content_y) isn't a content click at all.
        content_y = y - _INSTRUCTION_BAR_HEIGHT
        if content_y < 0:
            return
        point = (x / display_scale, content_y / display_scale)
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
            content = _render_content(image, session)
            if display_scale < 1.0:
                content = cv2.resize(content, display_size, interpolation=cv2.INTER_AREA)
            cv2.imshow(_WINDOW_NAME, _compose_display_frame(content, session))
            key = cv2.waitKey(20) & 0xFF
            if key == _KEY_ESC:
                print("mark-zones: aborted, nothing written", file=sys.stderr)
                return 1
            if key in (_KEY_ENTER, _KEY_SPACE):
                try:
                    if session.step is Step.SEATS:
                        session.finish_polygon()
                    elif session.step is Step.INNER_OVAL:
                        session.finish_inner_oval()
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
            marked, table_id=table_id, chip_zone_inset_pixels=chip_zone_inset_pixels
        )
    except (ValueError, ValidationError) as exc:
        print(f"mark-zones: {exc}", file=sys.stderr)
        return 1
    write_calibration_authoring(authoring, out_path)
    print(f"mark-zones: wrote '{out_path}'")
    _show_and_save_result_preview(image, authoring, out_path)
    return 0


def _show_and_save_result_preview(
    image: np.ndarray, authoring: CalibrationAuthoring, out_path: Path
) -> None:
    """Render every final zone onto the original photo and show + save it.

    Best-effort, start to finish: `out_path` already holds the real
    calibration by the time this runs (see caller), so nothing in here --
    compiling, drawing, encoding the PNG, or the GUI display -- may be
    allowed to turn an already-successful `mark-zones` run into a crash.
    Every failure is reported and swallowed instead.
    """
    try:
        _render_and_display_result_preview(image, authoring, out_path)
    except (ValueError, OSError, cv2.error) as exc:
        print(f"mark-zones: could not show/save result preview: {exc}", file=sys.stderr)


def _render_and_display_result_preview(
    image: np.ndarray, authoring: CalibrationAuthoring, out_path: Path
) -> None:
    runtime = compile_calibration(authoring, based_on=str(out_path))
    preview = image.copy()
    draw_zones(preview, runtime)

    preview_path = out_path.with_name(f"{out_path.stem}_preview.png")
    if cv2.imwrite(str(preview_path), preview):
        print(f"mark-zones: wrote result preview '{preview_path}'")
    else:
        # cv2.imwrite fails by returning False, not raising (e.g. disk full,
        # unwritable path) -- report it rather than silently claiming success.
        print(f"mark-zones: could not write result preview '{preview_path}'", file=sys.stderr)

    display_scale = min(1.0, _MAX_DISPLAY_DIMENSION / max(preview.shape[1], preview.shape[0]))
    display = preview
    if display_scale < 1.0:
        display_size = (
            round(preview.shape[1] * display_scale),
            round(preview.shape[0] * display_scale),
        )
        display = cv2.resize(preview, display_size, interpolation=cv2.INTER_AREA)
    cv2.imshow(_WINDOW_NAME, display)
    print("mark-zones: showing result preview -- press any key to close")
    cv2.waitKey(0)
    cv2.destroyWindow(_WINDOW_NAME)
