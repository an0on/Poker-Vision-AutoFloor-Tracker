"""Shared marker definitions for the `dopo_poker_table` live-testing setup
(Modus B / REQ-19), used against the real table's calibration
(`calibration/authoring/dopo_poker_table.json`).

Not part of `poker_vision`'s runtime package: a standalone authoring-time
utility, imported by `generate_markers.py` and `generate_marker_print_
sheet.py` so both always render the exact same dictionary/ID/label set,
and mirrored by `configs/dopo_poker_table_images.json` and
`configs/dopo_poker_table_livefeed.json`'s `aruco.marker_class_map` -- keep
all three in sync if this changes.
"""

from __future__ import annotations

# Must match the configs' `aruco.dictionary`.
DICTIONARY_NAME = "DICT_4X4_50"

# One marker ID per seat (place at most one physical chip-marker per seat
# for a full 10-handed test), one for the dealer button, and five distinct
# board-card markers (so a flop/turn/river count is testable with real
# hardware). Class mapping mirrors `configs/dopo_poker_table_*.json`'s
# `aruco.marker_class_map`.
MARKER_LABELS: dict[int, str] = {
    0: "chip_seat_1",
    1: "chip_seat_2",
    2: "chip_seat_3",
    3: "chip_seat_4",
    4: "chip_seat_5",
    5: "chip_seat_6",
    6: "chip_seat_7",
    7: "chip_seat_8",
    8: "chip_seat_9",
    9: "chip_seat_10",
    10: "dealer_button",
    20: "card_1",
    21: "card_2",
    22: "card_3",
    23: "card_4",
    24: "card_5",
}
