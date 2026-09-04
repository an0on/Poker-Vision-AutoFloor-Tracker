"""Generates the committed REQ-40 replay fixture (`../script.jsonl`, `../images/`).

Run from anywhere with `uv run python
test-fixtures/replay/scripts/generate_replay_fixture.py`. Deterministic and
idempotent -- re-running overwrites the same output, matching this
project's convention for fixture-generator scripts (see
`test-fixtures/arbitrary/scripts/`). The generated files are committed;
this script is the reproducible recipe for them, not something tests
import or run themselves.

The frame images' pixel content is never inspected by the `mock` detector's
Modus A (REQ-18) -- it reads detections purely from `script.jsonl`, keyed
by frame index -- so each frame is a small solid-color placeholder,
distinguishable only for a human skimming the directory listing.

See `tests/test_replay_fixtures.py` for what this fixture actually proves:
the exact frame-by-frame reasoning behind every block below lives there,
next to the assertions it drives.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_OUTPUT_DIR = Path(__file__).resolve().parent.parent
_IMAGES_DIR = _OUTPUT_DIR / "images"
_SCRIPT_PATH = _OUTPUT_DIR / "script.jsonl"

_RESOLUTION = (100, 100)  # (width, height), matches tests/test_replay_fixtures.py's table
_FRAME_COUNT = 81

_CARD_1 = (63.0, 65.0)
_CARD_2 = (71.0, 65.0)
_CARD_3 = (79.0, 65.0)
_CARD_4 = (65.0, 75.0)
_CARD_5 = (75.0, 75.0)


def _detection(object_class: str, x: float, y: float) -> dict:
    return {
        "coordinate_space": "table",
        "object_class": object_class,
        "confidence": 0.9,
        "center": {"x": x, "y": y},
    }


def _build_script_lines() -> list[dict]:
    by_frame: dict[int, list[dict]] = {}

    def add(frame_index: int, object_class: str, x: float, y: float) -> None:
        by_frame.setdefault(frame_index, []).append(_detection(object_class, x, y))

    # --- Dealer button: seat_1 -> seat_2, one continuous track (AC-18) ---
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
    for frame_index in range(50, 55):
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
    for frame_index in range(50, 55):
        add(frame_index, "card", *_CARD_3)
    for frame_index in range(58, 67):
        add(frame_index, "card", *_CARD_3)
    for frame_index in range(55, 67):
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
    for frame_index in range(61, 67):
        add(frame_index, "card", *_CARD_4)
    for frame_index in range(64, 67):
        add(frame_index, "card", *_CARD_5)
    # frames 67-68: everything above still present via last-known state
    # (miss count 1, 2); frame 69 is the third consecutive miss for all
    # five at once -> board drops to stably empty -> hand_ended.

    # --- Second hand: same flop slots, hand_id must be +1 (AC-20) ---
    for frame_index in range(75, 78):
        add(frame_index, "card", *_CARD_1)
        add(frame_index, "card", *_CARD_2)
        add(frame_index, "card", *_CARD_3)
    # frames 78-79 missing (miss 1, 2); frame 80 is the third -> hand_ended

    return [
        {"frame_index": frame_index, "detections": detections}
        for frame_index, detections in sorted(by_frame.items())
    ]


def main() -> None:
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with _SCRIPT_PATH.open("w") as handle:
        for line in _build_script_lines():
            handle.write(json.dumps(line))
            handle.write("\n")

    width, height = _RESOLUTION
    for i in range(_FRAME_COUNT):
        image = np.full((height, width, 3), i % 256, dtype=np.uint8)
        cv2.imwrite(str(_IMAGES_DIR / f"frame_{i:04d}.png"), image)

    print(f"wrote {_SCRIPT_PATH} and {_FRAME_COUNT} frames to {_IMAGES_DIR}")


if __name__ == "__main__":
    main()
