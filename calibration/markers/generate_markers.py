"""Generate one printable ArUco marker PNG per ID in `_aruco_markers.py`'s
`MARKER_LABELS`, for the real `dopo_poker_table` live-testing setup (Modus
B / REQ-19). Standalone dev utility, not part of the `poker_vision` package.

Print each PNG, stick it on the matching physical object (a chip stack, the
dealer button, a board-card slot) and place them on the table before
running the pipeline against `configs/dopo_poker_table_images.json` or
`configs/dopo_poker_table_livefeed.json` -- see
`calibration/markers/README.md` for the full walkthrough. Prefer
`generate_marker_print_sheet.py` if you'd rather print one A4 sheet and cut
it apart than print 16 separate files.

Usage:
    uv run python calibration/markers/generate_markers.py

Output: one PNG per marker under `--out-dir` (default
`data/raw/markers/dopo_poker_table`, gitignored like the rest of `data/`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from _aruco_markers import DICTIONARY_NAME, MARKER_LABELS

DEFAULT_OUT_DIR = Path("data/raw/markers/dopo_poker_table")
DEFAULT_MARKER_SIZE_PX = 600  # print at a size legible to the camera at table distance


def generate_markers(out_dir: Path, marker_size_px: int) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    for marker_id, label in MARKER_LABELS.items():
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
        out_path = out_dir / f"aruco_{marker_id:02d}_{label}.png"
        cv2.imwrite(str(out_path), image)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--marker-size-px", type=int, default=DEFAULT_MARKER_SIZE_PX)
    args = parser.parse_args()
    generate_markers(args.out_dir, args.marker_size_px)


if __name__ == "__main__":
    main()
