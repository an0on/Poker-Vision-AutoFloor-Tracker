"""REQ-10a: `ClickSession`'s step machine, independent of any display."""

from __future__ import annotations

import pytest

from poker_vision.calibration.mark_zones_session import ClickSession, Step


def _click_square(session: ClickSession, cx: float, cy: float, half: float = 5.0) -> None:
    for point in [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]:
        session.add_point(point)
    session.finish_polygon()


def _fresh_session_with_all_seats() -> ClickSession:
    session = ClickSession(image_size=(1000, 1000))
    # 10 seats spread around a circle, far enough apart to click-pick unambiguously.
    import math

    for i in range(10):
        angle = 2 * math.pi * i / 10
        _click_square(session, 400 * math.cos(angle), 400 * math.sin(angle), half=20)
    return session


# --- SEATS step ---------------------------------------------------------------


def test_new_session_starts_in_seats_step():
    session = ClickSession(image_size=(1920, 1080))
    assert session.step is Step.SEATS
    assert session.seats == {}


def test_finish_polygon_with_too_few_points_rejected():
    session = ClickSession(image_size=(1000, 1000))
    session.add_point((0, 0))
    session.add_point((10, 0))
    with pytest.raises(ValueError, match="at least"):
        session.finish_polygon()


def test_undo_removes_last_point_of_in_progress_polygon():
    session = ClickSession(image_size=(1000, 1000))
    session.add_point((0, 0))
    session.add_point((10, 0))
    session.undo()
    session.add_point((10, 10))
    session.add_point((0, 10))
    session.finish_polygon()
    assert list(session.seats.values())[0] == [(0, 0), (10, 10), (0, 10)]


def test_undo_on_empty_buffer_is_a_harmless_no_op():
    session = ClickSession(image_size=(1000, 1000))
    session.undo()  # must not raise
    assert session.step is Step.SEATS


def test_tenth_seat_auto_advances_to_pick_dealer():
    session = _fresh_session_with_all_seats()
    assert session.step is Step.PICK_DEALER
    assert len(session.seats) == 10


def test_finish_polygon_wrong_step_rejected():
    session = _fresh_session_with_all_seats()
    with pytest.raises(ValueError, match="only valid in step SEATS"):
        session.finish_polygon()


# --- PICK_DEALER step ----------------------------------------------------------


def test_pick_dealer_at_point_inside_a_seat_selects_it():
    session = _fresh_session_with_all_seats()
    first_key, first_polygon = next(iter(session.seats.items()))
    cx = sum(p[0] for p in first_polygon) / 4
    cy = sum(p[1] for p in first_polygon) / 4
    session.pick_dealer_at((cx, cy))
    assert session.dealer_seat_key == first_key
    assert session.step is Step.INNER_OVAL


def test_pick_dealer_at_point_outside_every_seat_rejected():
    session = _fresh_session_with_all_seats()
    with pytest.raises(ValueError, match="not inside any marked seat"):
        session.pick_dealer_at((0, 0))  # center of the ring, inside no wedge
    assert session.step is Step.PICK_DEALER  # unchanged on failure


def test_pick_dealer_at_wrong_step_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="only valid in step PICK_DEALER"):
        session.pick_dealer_at((0, 0))


# --- INNER_OVAL step -------------------------------------------------------------


def _session_at_inner_oval() -> ClickSession:
    session = _fresh_session_with_all_seats()
    first_key, first_polygon = next(iter(session.seats.items()))
    cx = sum(p[0] for p in first_polygon) / 4
    cy = sum(p[1] for p in first_polygon) / 4
    session.pick_dealer_at((cx, cy))
    return session


def test_inner_oval_points_accumulate_without_auto_advancing():
    session = _session_at_inner_oval()
    for point in [(-5, -5), (5, -5), (5, 5), (-5, 5), (0, 6)]:
        session.add_point(point)
    assert session.step is Step.INNER_OVAL
    assert session.inner_oval_points == [(-5, -5), (5, -5), (5, 5), (-5, 5), (0, 6)]


def test_finish_inner_oval_with_too_few_points_rejected():
    session = _session_at_inner_oval()
    session.add_point((0, 0))
    session.add_point((1, 0))
    with pytest.raises(ValueError, match="at least"):
        session.finish_inner_oval()


def test_finish_inner_oval_advances_to_board_zone():
    session = _session_at_inner_oval()
    for point in [(-5, -5), (5, -5), (5, 5)]:
        session.add_point(point)
    session.finish_inner_oval()
    assert session.step is Step.BOARD_ZONE


def test_finish_inner_oval_wrong_step_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="only valid in step INNER_OVAL"):
        session.finish_inner_oval()


def test_undo_during_inner_oval_step_removes_last_point():
    session = _session_at_inner_oval()
    session.add_point((-5, -5))
    session.add_point((5, -5))
    session.undo()
    assert session.inner_oval_points == [(-5, -5)]


# --- BOARD_ZONE step ------------------------------------------------------------


def _session_at_board_zone() -> ClickSession:
    session = _session_at_inner_oval()
    for point in [(-5, -5), (5, -5), (5, 5)]:
        session.add_point(point)
    session.finish_inner_oval()
    return session


def test_board_zone_advances_after_exactly_four_points():
    session = _session_at_board_zone()
    points = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    for point in points[:3]:
        session.add_point(point)
    assert session.step is Step.BOARD_ZONE
    session.add_point(points[3])
    assert session.step is Step.DONE
    assert session.board_zone_points == points


def test_undo_during_board_zone_step_removes_last_point():
    session = _session_at_board_zone()
    session.add_point((-10, -10))
    session.add_point((10, -10))
    session.undo()
    assert session.board_zone_points == [(-10, -10)]


# Codex review finding (P2): BOARD_ZONE's 4th point auto-advances the step
# to DONE immediately -- a mistaken final click there must still be
# undoable, not silently no-op against the new, empty step.


def test_undo_after_board_zone_auto_advance_reopens_it():
    session = _session_at_board_zone()
    board_points = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    for point in board_points:
        session.add_point(point)
    assert session.step is Step.DONE

    session.undo()
    assert session.step is Step.BOARD_ZONE
    assert session.board_zone_points == board_points[:3]


def test_full_session_reaches_done_and_builds():
    session = _session_at_board_zone()
    assert session.step is Step.BOARD_ZONE
    for point in [(-10, -10), (10, -10), (10, 10), (-10, 10)]:
        session.add_point(point)
    assert session.step is Step.DONE

    marked = session.build()
    assert marked.image_size == (1000, 1000)
    assert len(marked.seat_polygons) == 10
    assert marked.dealer_seat_key in marked.seat_polygons
    assert marked.inner_oval_points == [(-5, -5), (5, -5), (5, 5)]
    assert len(marked.board_zone_points) == 4


def test_build_before_done_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="not complete yet"):
        session.build()


# --- DONE step: reopening after a failed save -----------------------------------
#
# Codex review finding (P2): a save-time REQ-11 rejection of a seat or the
# inner-oval trace (see mark_zones.py's error messages, which name the
# seat to re-trace) previously had no way to actually act on that guidance
# short of aborting and re-clicking the entire 10-seat session from
# scratch -- undo() only ever reaches back into BOARD_ZONE from DONE.


def _session_at_done() -> ClickSession:
    session = _session_at_board_zone()
    for point in [(-10, -10), (10, -10), (10, 10), (-10, 10)]:
        session.add_point(point)
    assert session.step is Step.DONE
    return session


def test_seat_at_point_inside_a_seat_returns_its_key():
    session = _fresh_session_with_all_seats()
    first_key, first_polygon = next(iter(session.seats.items()))
    cx = sum(p[0] for p in first_polygon) / 4
    cy = sum(p[1] for p in first_polygon) / 4
    assert session.seat_at((cx, cy)) == first_key


def test_seat_at_point_outside_every_seat_returns_none():
    session = _fresh_session_with_all_seats()
    assert session.seat_at((0, 0)) is None  # center of the ring, inside no wedge


def test_reopen_seat_moves_its_points_into_the_current_polygon():
    session = _session_at_done()
    key, points = next(iter(session.seats.items()))
    session.reopen_seat(key)
    assert session.step is Step.SEATS
    assert session.current_polygon == points
    assert key not in session.seats
    assert len(session.seats) == 9


def test_reopen_seat_discards_dealer_pick_oval_and_board_zone():
    session = _session_at_done()
    key = next(iter(session.seats))
    session.reopen_seat(key)
    assert session.dealer_seat_key is None
    assert session.inner_oval_points == []
    assert session.board_zone_points == []


def test_reopen_seat_of_unknown_key_rejected():
    session = _session_at_done()
    with pytest.raises(ValueError, match="not a marked seat"):
        session.reopen_seat("does_not_exist")


def test_reopen_seat_wrong_step_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="only valid in step DONE"):
        session.reopen_seat("click_1")


def test_reopen_seat_can_be_re_finished_and_reach_done_again():
    session = _session_at_done()
    key, points = next(iter(session.seats.items()))
    session.reopen_seat(key)
    session.finish_polygon()  # re-click nothing changed, just re-commit as-is
    assert session.step is Step.PICK_DEALER
    assert len(session.seats) == 10
    assert points in session.seats.values()


def test_reopen_inner_oval_returns_to_inner_oval_step():
    session = _session_at_done()
    session.reopen_inner_oval()
    assert session.step is Step.INNER_OVAL
    assert session.inner_oval_points == []


def test_reopen_inner_oval_discards_board_zone_but_keeps_dealer_pick():
    session = _session_at_done()
    dealer_seat_key = session.dealer_seat_key
    session.reopen_inner_oval()
    assert session.board_zone_points == []
    assert session.dealer_seat_key == dealer_seat_key


def test_reopen_inner_oval_wrong_step_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="only valid in step DONE"):
        session.reopen_inner_oval()
