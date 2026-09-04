"""REQ-40: replay fixtures with a documented expected event sequence.

`test_runner_loop.py` (REQ-44) already proves the core chain runs headlessly
end to end; this module is the "Testfixtures" REQ-40 itself asks for -- one
committed, deterministic replay set (`test-fixtures/replay/script.jsonl`, a
`mock` Modus-A script, plus the matching `test-fixtures/replay/images/`
directory -- see `test-fixtures/replay/scripts/generate_replay_fixture.py`
for how they were produced) that drives the *whole* pipeline (`capture` ->
`detection` -> `tracking` -> `assignment` -> `state`) through every event
type AC-17 through AC-20 care about, in a single continuous session, plus
the fault cases REQ-40 explicitly names:

- Dealer-Wechsel: REQ-27's nearest-seat/player_area resolution as a single
  `dealer_button` track drifts, frame by frame, from seat_1's `player_area`
  into seat_2's -- never losing its track_id, so this is one continuous
  `DealerSeatTracker` transition (AC-18), not two independent sightings.
  The button's first-ever resolution (seat_1) only establishes a starting
  position and fires no event; only the later change to seat_2 does. The
  button then vanishes from the script entirely (no further detections) --
  AC-18 requires this to leave the dealer seat exactly as it was, so the
  final `StateSnapshot` (not just the exported events) is asserted to
  still report `seat_2`.
- Occupancy + Dropout: a `chip` in seat_1's `chip_zone` is dropped for two
  frames (below `n_off`, AC-12's "kein seat_vacated"), reappears, then
  dropped for exactly `n_off` frames (AC-12's "genau eines"). A second,
  short-lived "ghost" chip in seat_2's `chip_zone` never reaches `n_on` and
  so never produces a `seat_occupied` at all.
- Flop -> Turn -> River + genuine 3 -> 2 -> 3 and 4 -> 3 -> 4 flickers:
  three board cards reach hysteresis together (flop); one is then dropped
  for exactly `n_off` frames and re-detected (a real, hysteresis-mediated
  dip to a stable count of 2, not merely a same-frame dip in raw
  detections) -- AC-19 requires this fires `flop` exactly once, not once
  per `3`. A fourth card confirms (turn), is itself dropped and
  re-detected the same way (a real dip to 3 within the hand -- AC-19's
  "4 -> 3 innerhalb einer Hand erzeugt kein Event"), then a fifth card
  confirms (river).
- Hand-Ende + zweite Hand: the board goes stably empty (`hand_ended`), then
  a short second hand starts and ends, its `hand_id` one more than the
  first's (AC-20).

A second test replays the exact same committed fixture through REQ-21's
`PerturbedDetector` -- the real, seeded perturbation mechanism -- with
small positional jitter and no dropout/ghost, asserting the identical
event sequence still holds. That covers REQ-40's third named fault case
(Jitter) without a second fixture: dropout and the flop flicker are
already genuine hysteresis-driven fault cases in the base script above.

Every event's expected `(frame_index, event_type, ...)` is asserted
against the actual JSONL export, in order, so this doubles as AC-25's "die
genannten Fixtures ... existieren mit hinterlegter Soll-Event-Sequenz" for
occupancy/dealer/street/hand-lifecycle combined in one replay.
"""

from __future__ import annotations

import json
from pathlib import Path

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.capture.image_dir import ImageDirCapture
from poker_vision.config import HysteresisConfig, PerturbationConfig, Resolution
from poker_vision.detection.base import Detector
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.mock_perturbation import PerturbedDetector
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.manager import ExportManager
from poker_vision.runner.loop import FrameLoop, LoopExitReason
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

_FIXTURE_DIR = Path(__file__).parent.parent / "test-fixtures" / "replay"
_IMAGES_DIR = _FIXTURE_DIR / "images"
_SCRIPT_PATH = _FIXTURE_DIR / "script.jsonl"

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_RESOLUTION = Resolution(width=100, height=100)
_TABLE = TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM)
_N_ON = 3
_N_OFF = 3
_MAX_DISTANCE = 5.0


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


def _run_replay(
    tmp_path: Path, calibration: CalibrationRuntime, detector: Detector
) -> tuple[list[dict], PipelineStateMachine]:
    """Runs the full pipeline over the committed fixture.

    Returns the exported events and the state machine itself (not just its
    events) -- AC-18's "Verschwinden des Buttons ändert den Dealer-Seat
    nicht" is a claim about the *final snapshot*, not about the absence of
    a `dealer_moved` event, so a caller needs the machine to check it.
    """
    tracker = NearestMatchTracker(max_distance=_MAX_DISTANCE, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=_N_ON, n_off=_N_OFF))
    state_machine = PipelineStateMachine(["seat_1", "seat_2"])
    jsonl_exporter = JsonlEventExporter(tmp_path / "exports")
    export_manager = ExportManager([jsonl_exporter])
    capture = ImageDirCapture(_IMAGES_DIR, _RESOLUTION)

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
    return [json.loads(line) for line in lines], state_machine


_EXPECTED_EVENTS = [
    # Frame 2 confirms the dealer button at seat_1, but a first-ever
    # resolution only establishes the starting position -- no event
    # (AC-18); only the frame-9 seat change to seat_2 fires one.
    (9, "dealer_moved", {"from_seat": "seat_1", "to_seat": "seat_2"}),
    (22, "seat_occupied", {"seat": "seat_1"}),
    (28, "seat_vacated", {"seat": "seat_1"}),
    (52, "hand_started", {"hand_id": 1}),
    (52, "street_changed", {"hand_id": 1, "street": "flop"}),
    # frame 57: card 3 removed (n_off), count dips 3 -> 2, no event.
    # frame 60: card 3 reconfirmed, count back to 3, no event (flop already
    # current -- this is the "3 -> 2 -> 3" flicker, exactly one flop event).
    (63, "street_changed", {"hand_id": 1, "street": "turn"}),
    # frame 66: card 4 removed (n_off), count dips 4 -> 3, no event -- this
    # is AC-19's "4 -> 3 innerhalb einer Hand erzeugt kein Event".
    # frame 69: card 4 reconfirmed, count back to 4, no event (turn already
    # current).
    (72, "street_changed", {"hand_id": 1, "street": "river"}),
    (75, "hand_ended", {"hand_id": 1}),
    (82, "hand_started", {"hand_id": 2}),
    (82, "street_changed", {"hand_id": 2, "street": "flop"}),
    (85, "hand_ended", {"hand_id": 2}),
]


def _assert_matches_expected_sequence(events: list[dict]) -> None:
    actual = [(event["frame_index"], event["event_type"]) for event in events]
    assert actual == [(frame_index, event_type) for frame_index, event_type, _ in _EXPECTED_EVENTS]

    for event, (_, _, fields) in zip(events, _EXPECTED_EVENTS, strict=True):
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


# --- the replay set itself: one continuous session, the full event sequence ---


def test_replay_produces_the_documented_event_sequence(tmp_path):
    calibration = _calibration()
    detector = MockDetector(calibration, _SCRIPT_PATH)
    events, state_machine = _run_replay(tmp_path, calibration, detector)
    _assert_matches_expected_sequence(events)

    # AC-18: the button disappearing (last seen at frame 15, never again)
    # must not change the dealer seat -- the final snapshot still reports
    # seat_2, silently, with no corresponding event.
    assert state_machine.snapshot().dealer_seat == "seat_2"


# --- Jitter fault case: REQ-21's real PerturbedDetector, not hand-simulated noise ---


def test_replay_survives_position_jitter(tmp_path):
    """The same committed replay, but every detection's table position gets
    small Gaussian noise (REQ-21's seeded `PerturbedDetector`, well under the
    tracker's matching threshold and every zone's margin to its boundary) --
    the documented event sequence must be unaffected (REQ-40's "Jitter"
    fault case)."""
    calibration = _calibration()
    inner = MockDetector(calibration, _SCRIPT_PATH)
    config = PerturbationConfig(seed=1234, position_jitter_std=0.5)
    detector = PerturbedDetector(calibration, inner, config)

    events, state_machine = _run_replay(tmp_path, calibration, detector)
    _assert_matches_expected_sequence(events)
    assert state_machine.snapshot().dealer_seat == "seat_2"
