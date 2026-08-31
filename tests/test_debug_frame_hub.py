"""REQ-46: `LatestFrameHub` -- the thread-safe single-slot, latest-wins
bridge between the pipeline loop's publish and the debug server's
per-client, on-demand reads.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import numpy as np
import pytest

from poker_vision.assignment.models import FrameAssignments
from poker_vision.capture.frame import Frame
from poker_vision.debug.frame_hub import DebugSnapshot, LatestFrameHub
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.models import TrackedFrame


def _frame(frame_index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


def _snapshot(frame_index: int = 0) -> DebugSnapshot:
    return DebugSnapshot(
        tracked_frame=TrackedFrame(schema_version="1.0", frame_index=frame_index, tracks=[]),
        frame_assignments=FrameAssignments(
            schema_version="1.0", frame_index=frame_index, assignments=[]
        ),
        state_snapshot=PipelineStateMachine(["seat_1"]).snapshot(),
    )


# --- publish()/get_latest() basics -------------------------------------------


def test_get_latest_returns_none_with_nothing_ever_published():
    hub = LatestFrameHub()
    assert hub.get_latest(since_version=0, timeout=0.05) is None


def test_get_latest_returns_the_published_frame_and_snapshot():
    hub = LatestFrameHub()
    frame = _frame()
    snapshot = _snapshot()
    hub.publish(frame, snapshot)

    result = hub.get_latest(since_version=0, timeout=0.05)

    assert result is not None
    got_frame, got_snapshot, version = result
    assert got_frame is frame
    assert got_snapshot is snapshot
    assert version == 1


def test_publish_always_overwrites_the_single_slot():
    # "kein Queue-Backlog" -- a consumer that only reads once sees the
    # latest publish, never the first of several skipped in between.
    hub = LatestFrameHub()
    first, second, third = _frame(0), _frame(1), _frame(2)
    hub.publish(first, _snapshot(0))
    hub.publish(second, _snapshot(1))
    hub.publish(third, _snapshot(2))

    got_frame, _snap, version = hub.get_latest(since_version=0, timeout=0.05)

    assert got_frame is third
    assert version == 3


def test_get_latest_never_redelivers_the_same_version_to_the_same_caller():
    hub = LatestFrameHub()
    hub.publish(_frame(), _snapshot())
    _frame_a, _snap_a, version = hub.get_latest(since_version=0, timeout=0.05)

    # Same caller, now caught up (since_version == current version) -- no
    # newer frame has been published, so this must time out, not repeat
    # the frame it already delivered.
    assert hub.get_latest(since_version=version, timeout=0.05) is None


def test_independent_consumers_track_their_own_version():
    # Unlike capture.continuity's single-consumer buffer, an arbitrary
    # number of MJPEG clients each read this same hub independently.
    hub = LatestFrameHub()
    hub.publish(_frame(0), _snapshot(0))
    slow_client_version = 0  # never caught up to frame 0 yet

    hub.publish(_frame(1), _snapshot(1))
    fast_client_result = hub.get_latest(since_version=1, timeout=0.05)
    slow_client_result = hub.get_latest(since_version=slow_client_version, timeout=0.05)

    assert fast_client_result is not None
    assert fast_client_result[0].frame_index == 1
    assert slow_client_result is not None
    assert slow_client_result[0].frame_index == 1  # latest-wins, not frame 0


# --- thread safety: concurrent publish/get, no corruption or deadlock -------


def test_concurrent_publish_and_get_never_corrupts_or_deadlocks():
    hub = LatestFrameHub()
    hub.publish(_frame(0), _snapshot(0))  # seed so early get_latest calls have something
    stop = threading.Event()
    errors: list[BaseException] = []

    def publisher() -> None:
        index = 1
        while not stop.is_set():
            try:
                hub.publish(_frame(index), _snapshot(index))
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the test thread
                errors.append(exc)
                return
            index += 1

    def consumer() -> None:
        since_version = 0
        for _ in range(200):
            result = hub.get_latest(since_version=since_version, timeout=1.0)
            if result is None:
                continue
            frame, snapshot, version = result
            try:
                # A torn/corrupted read would show up here as a
                # frame/snapshot pair that don't agree with each other.
                assert frame.frame_index == snapshot.tracked_frame.frame_index
                assert frame.frame_index == snapshot.frame_assignments.frame_index
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            since_version = version

    publisher_thread = threading.Thread(target=publisher)
    consumer_threads = [threading.Thread(target=consumer) for _ in range(4)]

    publisher_thread.start()
    for thread in consumer_threads:
        thread.start()
    for thread in consumer_threads:
        thread.join(timeout=10.0)
    stop.set()
    publisher_thread.join(timeout=5.0)

    assert not any(thread.is_alive() for thread in consumer_threads), "consumer thread hung"
    assert not publisher_thread.is_alive(), "publisher thread hung"
    assert errors == []


# --- REQ-46: publish() never renders anything --------------------------------


def test_publish_does_not_import_or_invoke_overlay_rendering():
    # LatestFrameHub has no knowledge of debug.overlay.render_overlay at
    # all -- overlay rendering only ever happens in MjpegDebugServer.
    # _stream(), on demand per connected client (see tests/test_debug_
    # mjpeg.py). Asserted here structurally rather than by mocking, since
    # this module doesn't import render_overlay in the first place.
    import poker_vision.debug.frame_hub as frame_hub_module

    assert "render_overlay" not in dir(frame_hub_module)
    assert not hasattr(LatestFrameHub, "render") and not hasattr(LatestFrameHub, "_render")


@pytest.mark.parametrize("call_count", [1, 5])
def test_publish_is_synchronous_and_does_not_block_on_a_consumer(call_count):
    # publish() must return promptly regardless of whether a consumer is
    # currently blocked inside get_latest() -- it only ever swaps
    # references and bumps the version counter under a briefly-held lock.
    hub = LatestFrameHub()
    blocked = threading.Event()

    def blocked_consumer() -> None:
        blocked.set()
        hub.get_latest(since_version=10_000, timeout=2.0)  # never satisfied -> blocks

    consumer_thread = threading.Thread(target=blocked_consumer, daemon=True)
    consumer_thread.start()
    blocked.wait(timeout=1.0)

    for i in range(call_count):
        hub.publish(_frame(i), _snapshot(i))  # must not hang even with a blocked waiter
