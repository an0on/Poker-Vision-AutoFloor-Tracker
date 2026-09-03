"""REQ-40: replay fixtures with a documented expected event sequence.

`test_runner_loop.py` (REQ-44) already proves the core chain runs headlessly
end to end; this module is the "Testfixtures" REQ-40 itself asks for -- one
committed, deterministic replay set (a `mock` Modus-A script plus a matching
`image_dir`) that drives the *whole* pipeline (`capture` -> `detection` ->
`tracking` -> `assignment` -> `state`) through every event type AC-17
through AC-20 care about, in a single continuous session, plus the fault
cases REQ-40 explicitly names:

- Dealer-Wechsel: REQ-27's nearest-seat/player_area resolution as a single
  `dealer_button` track drifts, frame by frame, from seat_1's `player_area`
  into seat_2's -- never losing its track_id, so this is one continuous
  `DealerSeatTracker` transition (AC-18), not two independent sightings.
  The button's first-ever resolution (seat_1) only establishes a starting
  position and fires no event; only the later change to seat_2 does.
- Occupancy + Dropout: a `chip` in seat_1's `chip_zone` is dropped for two
  frames (below `n_off`, AC-12's "kein seat_vacated"), reappears, then
  dropped for exactly `n_off` frames (AC-12's "genau eines"). A second,
  short-lived "ghost" chip in seat_2's `chip_zone` never reaches `n_on` and
  so never produces a `seat_occupied` at all.
- Flop -> Turn -> River + a 3 -> 2 -> 3 flicker: three board cards reach
  hysteresis together (flop), one is then dropped for exactly `n_off`
  frames and re-detected (a genuine, hysteresis-mediated dip to a stable
  count of 2, not merely a same-frame dip in raw detections) -- AC-19
  requires this fires the `flop` event exactly once, not once per `3`.
  A fourth and fifth card follow for turn/river.
- Hand-Ende + zweite Hand: the board goes stably empty (`hand_ended`), then
  a short second hand starts and ends, its `hand_id` one more than the
  first's (AC-20).

A separate, smaller test covers the third named fault case (Jitter) via
`PerturbedDetector` (REQ-21) -- the real, seeded perturbation mechanism,
not hand-simulated noise -- confirming positional jitter well under the
tracker's matching threshold never breaks presence/occupancy.

Every event's expected `(frame_index, event_type, ...)` is asserted
against the actual JSONL export, in order, so this doubles as AC-25's "die
genannten Fixtures ... existieren mit hinterlegter Soll-Event-Sequenz" for
occupancy/dealer/street/hand-lifecycle combined in one replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.capture.image_dir import ImageDirCapture
from poker_vision.config import HysteresisConfig, PerturbationConfig, Resolution
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.mock_perturbation import PerturbedDetector
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.manager import ExportManager
from poker_vision.runner.loop import FrameLoop, LoopExitReason
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_RESOLUTION = Resolution(width=100, height=100)
_TABLE = TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM)
_N_ON = 3
_N_OFF = 3
_MAX_DISTANCE = 5.0
_FRAME_COUNT = 81


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


_SEAT_1 = CalibrationSeat(
    seat_id="seat_1",
    zones=SeatZones(
        player_area=_polygon((0, 0), (50, 0), (50, 50), (0, 50)),
        chip_zone=_polygon((10, 10), (30, 10), (30, 30), (10, 30)),
    ),
)
_SEAT_2 = CalibrationSeat(
    seat_id="seat_2",
    zones=SeatZones(
        player_area=_polygon((50, 0), (100, 0), (100, 50), (50, 50)),
        chip_zone=_polygon((60, 10), (90, 10), (90, 30), (60, 30)),
    ),
)
_BOARD_ZONE = _polygon((60, 60), (90, 60), (90, 90), (60, 90))
_DEALER_AREA = _polygon((0, 60), (20, 60), (20, 80), (0, 80))

# Board-card slots, pairwise >= 8 table units apart -- comfortably above
# `_MAX_DISTANCE` so each slot's detections only ever match its own track.
_CARD_1 = (63.0, 65.0)
_CARD_2 = (71.0, 65.0)
_CARD_3 = (79.0, 65.0)
_CARD_4 = (65.0, 75.0)
_CARD_5 = (75.0, 75.0)


def _calibration() -> CalibrationRuntime:
    return CalibrationRuntime(
        schema_version="1.1",
        table_id="test_table",
        based_on="test",
        inference_resolution=_RESOLUTION,
        camera=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=50.0, cy=50.0),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=_TABLE,
        seats=[_SEAT_1, _SEAT_2],
        zones=GlobalZones(board_zone=_BOARD_ZONE, dealer_area=_DEALER_AREA),
        card_dealer_seat_id="seat_1",
    )


def _detection(object_class: str, x: float, y: float) -> dict:
    return {
        "coordinate_space": "table",
        "object_class": object_class,
        "confidence": 0.9,
        "center": {"x": x, "y": y},
    }


def _build_script_lines() -> list[dict]:
    """The one continuous replay session this whole module exercises.

    A `dict[frame_index, list[detection]]` built up by named per-track
    helpers, then flattened into the one-line-per-frame script format
    `MockDetector` (REQ-18) expects. See the module docstring for the
    frame-by-frame reasoning behind each block.
    """
    by_frame: dict[int, list[dict]] = {}

    def add(frame_index: int, object_class: str, x: float, y: float) -> None:
        by_frame.setdefault(frame_index, []).append(_detection(object_class, x, y))

    # --- Dealer button: seat_1 -> seat_2, one continuous track (AC-18) ---
    # Confirms at (25, 25) after n_on=3 frames, then drifts right in
    # steps of 4 table units (< _MAX_DISTANCE) so the tracker never loses
    # it -- crossing from seat_1's into seat_2's player_area at frame 9.
    dealer_xs = [25, 25, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77]
    for frame_index, x in enumerate(dealer_xs):
        add(frame_index, "dealer_button", float(x), 25.0)

    # --- Occupancy + dropout fault case (AC-17, AC-12) ---
    for frame_index in (20, 21, 22):  # confirms at 22 -> seat_occupied
        add(frame_index, "chip", 20.0, 20.0)
    # frames 23-24 missing: below n_off=3, must NOT vacate
    add(25, "chip", 20.0, 20.0)  # reappears, resets the miss count
    # frames 26-28 missing: reaches n_off=3 at 28 -> seat_vacated

    # --- Ghost chip: never reaches n_on, never occupies seat_2 ---
    for frame_index in (35, 36):
        add(frame_index, "chip", 70.0, 20.0)

    # --- Board: flop, a genuine 3 -> 2 -> 3 flicker, turn, river ---
    for frame_index in range(50, 55):  # 50-52 confirm flop; 53-54 steady
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
    # card 3 alongside 1/2 through frame 54, then dropped 55-57 (n_off=3)
    # and re-detected from 58 onward, re-confirming at 60 (n_on=3) -- a
    # real hysteresis-mediated 3->2->3, not merely a same-frame dip.
    for frame_index in range(50, 55):
        add(frame_index, "card", *_CARD_3)
    for frame_index in range(58, 67):
        add(frame_index, "card", *_CARD_3)
    # cards 1/2 stay put the whole time so the count is driven only by
    # card 3's absence/return.
    for frame_index in range(55, 67):
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
    for frame_index in range(61, 67):  # confirms at 63 -> turn, then stays put
        add(frame_index, "card", *_CARD_4)
    for frame_index in range(64, 67):  # confirms at 66 -> river
        add(frame_index, "card", *_CARD_5)
    # frames 67-68: everything above still present via last-known state
    # (miss count 1, 2); frame 69 is the third consecutive miss for all
    # five at once -> board drops to stably empty -> hand_ended.

    # --- Second hand: same flop slots, hand_id must be +1 (AC-20) ---
    for frame_index in range(75, 78):  # confirms at 77 -> hand_started/flop
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
        add(frame_index, "card", *_CARD_3)
    # frames 78-79 missing (miss 1, 2); frame 80 is the third -> hand_ended

    return [
        {"frame_index": frame_index, "detections": detections}
        for frame_index, detections in sorted(by_frame.items())
    ]


def _write_script(path: Path, lines: list[dict]) -> Path:
    script_path = path / "script.jsonl"
    with script_path.open("w") as handle:
        for line in lines:
            handle.write(json.dumps(line))
            handle.write("\n")
    return script_path


def _make_image_dir(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(count):
        image = np.full((_RESOLUTION.height, _RESOLUTION.width, 3), i % 256, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"), image)
    return image_dir


def _run_replay(tmp_path: Path, detector_factory) -> list[dict]:
    """Runs the full pipeline over `_FRAME_COUNT` frames and returns the exported events."""
    calibration = _calibration()
    detector = detector_factory(calibration)
    tracker = NearestMatchTracker(max_distance=_MAX_DISTANCE, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=_N_ON, n_off=_N_OFF))
    state_machine = PipelineStateMachine(["seat_1", "seat_2"])
    jsonl_exporter = JsonlEventExporter(tmp_path / "exports")
    export_manager = ExportManager([jsonl_exporter])
    capture = ImageDirCapture(_make_image_dir(tmp_path, _FRAME_COUNT), _RESOLUTION)

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_MAX_DISTANCE,
        state_machine=state_machine,
        export_manager=export_manager,
    )

    reason = loop.run()
    jsonl_exporter.close()

    assert reason == LoopExitReason.EOF
    lines = jsonl_exporter.path.read_text().splitlines()
    return [json.loads(line) for line in lines]


# --- the replay set itself: one continuous session, the full event sequence ---


def test_replay_produces_the_documented_event_sequence(tmp_path):
    script_path = _write_script(tmp_path, _build_script_lines())
    events = _run_replay(tmp_path, lambda calibration: MockDetector(calibration, script_path))

    expected = [
        # Frame 2 confirms the button at seat_1, but a first-ever
        # resolution only establishes the starting position -- no event
        # (AC-18); only the frame-9 seat change to seat_2 fires one.
        (9, "dealer_moved", {"from_seat": "seat_1", "to_seat": "seat_2"}),
        (22, "seat_occupied", {"seat": "seat_1"}),
        (28, "seat_vacated", {"seat": "seat_1"}),
        (52, "hand_started", {"hand_id": 1}),
        (52, "street_changed", {"hand_id": 1, "street": "flop"}),
        (63, "street_changed", {"hand_id": 1, "street": "turn"}),
        (66, "street_changed", {"hand_id": 1, "street": "river"}),
        (69, "hand_ended", {"hand_id": 1}),
        (77, "hand_started", {"hand_id": 2}),
        (77, "street_changed", {"hand_id": 2, "street": "flop"}),
        (80, "hand_ended", {"hand_id": 2}),
    ]

    actual = [(event["frame_index"], event["event_type"]) for event in events]
    assert actual == [(frame_index, event_type) for frame_index, event_type, _ in expected]

    for event, (_, _, fields) in zip(events, expected, strict=True):
        for key, value in fields.items():
            assert event[key] == value, f"{event['event_type']}@{event['frame_index']}: {key}"

    # Sequence numbers are globally monotonic across the whole replay (REQ-33).
    assert [event["sequence"] for event in events] == list(range(len(events)))

    # The ghost chip in seat_2 (frames 35-36, below n_on) never occupies it,
    # and seat_2 never appears in any occupancy event across the whole replay.
    seat_events = [
        event for event in events if event["event_type"] in ("seat_occupied", "seat_vacated")
    ]
    assert all(event["seat"] == "seat_1" for event in seat_events)


# --- Jitter fault case: REQ-21's real PerturbedDetector, not hand-simulated noise ---


def test_replay_occupancy_survives_position_jitter(tmp_path):
    """A chip's detected position jitters every frame but never crosses a zone
    boundary or the tracker's matching threshold -- occupancy must be exactly
    as clean as the noise-free case (REQ-40's "Jitter" fault case, REQ-21's
    seeded `PerturbedDetector`)."""
    lines = [
        {
            "frame_index": frame_index,
            "detections": [_detection("chip", 20.0, 20.0)],
        }
        for frame_index in range(_N_ON + 2)
    ]
    script_path = _write_script(tmp_path, lines)

    def build(calibration: CalibrationRuntime) -> PerturbedDetector:
        inner = MockDetector(calibration, script_path)
        config = PerturbationConfig(seed=1234, position_jitter_std=0.5)
        return PerturbedDetector(calibration, inner, config)

    calibration = _calibration()
    detector = build(calibration)
    tracker = NearestMatchTracker(max_distance=_MAX_DISTANCE, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=_N_ON, n_off=_N_OFF))
    state_machine = PipelineStateMachine(["seat_1", "seat_2"])
    jsonl_exporter = JsonlEventExporter(tmp_path / "exports")
    export_manager = ExportManager([jsonl_exporter])
    capture = ImageDirCapture(_make_image_dir(tmp_path, len(lines)), _RESOLUTION)

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_MAX_DISTANCE,
        state_machine=state_machine,
        export_manager=export_manager,
    )
    reason = loop.run()
    jsonl_exporter.close()

    assert reason == LoopExitReason.EOF
    events = [json.loads(line) for line in jsonl_exporter.path.read_text().splitlines()]

    assert len(events) == 1
    assert events[0]["event_type"] == "seat_occupied"
    assert events[0]["seat"] == "seat_1"
    assert events[0]["frame_index"] == _N_ON - 1

    snapshot = state_machine.snapshot()
    assert any(s.seat == "seat_1" and s.occupied for s in snapshot.seats)
