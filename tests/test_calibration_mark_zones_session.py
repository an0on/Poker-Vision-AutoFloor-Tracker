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
    assert session.step is Step.BOARD_ZONE


def test_pick_dealer_at_point_outside_every_seat_rejected():
    session = _fresh_session_with_all_seats()
    with pytest.raises(ValueError, match="not inside any marked seat"):
        session.pick_dealer_at((0, 0))  # center of the ring, inside no wedge
    assert session.step is Step.PICK_DEALER  # unchanged on failure


def test_pick_dealer_at_wrong_step_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="only valid in step PICK_DEALER"):
        session.pick_dealer_at((0, 0))


# --- BOARD_ZONE step ------------------------------------------------------------


def _session_at_board_zone() -> ClickSession:
    session = _fresh_session_with_all_seats()
    first_key, first_polygon = next(iter(session.seats.items()))
    cx = sum(p[0] for p in first_polygon) / 4
    cy = sum(p[1] for p in first_polygon) / 4
    session.pick_dealer_at((cx, cy))
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
    assert len(marked.board_zone_points) == 4


def test_build_before_done_rejected():
    session = ClickSession(image_size=(1000, 1000))
    with pytest.raises(ValueError, match="not complete yet"):
        session.build()
