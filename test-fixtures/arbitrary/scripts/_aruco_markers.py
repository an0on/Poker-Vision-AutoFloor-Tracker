"""Shared marker definitions for the REQ-19 ArUco wiring test scripts.

Not a standalone script -- imported by `generate_aruco_test_markers.py` and
`generate_marker_print_sheet.py` so both always render the exact same
dictionary/ID/label set, never two scripts drifting out of sync.
"""

from __future__ import annotations

# Must match ../configs/test_arbitrary.json's `aruco.dictionary`.
DICTIONARY_NAME = "DICT_4X4_50"

# Mirrors ../configs/test_arbitrary.json's `aruco.marker_class_map`, plus a
# human-readable label per ID for filenames / print-sheet captions.
MARKER_LABELS: dict[int, str] = {
    0: "chip_seat_1",
    1: "chip_seat_2",
    2: "chip_seat_3",
    3: "chip_seat_4",
    4: "chip_seat_5",
    5: "chip_all_in",
    10: "dealer_button",
    20: "card_1",
    21: "card_2",
    22: "card_3",
    23: "card_4",
    24: "card_5",
}
