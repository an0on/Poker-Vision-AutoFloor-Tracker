"""Generate printable ArUco marker PNGs for the REQ-19 wiring test.

Standalone dev utility, not part of the `poker_vision` package: renders one
PNG per marker ID in `../configs/test_arbitrary.json`'s
`aruco.marker_class_map`, using the same dictionary (DICT_4X4_50). Print
each PNG, stick it on the matching physical object, and place them in front
of the camera before running the `image_dir` pipeline against
`../configs/test_arbitrary.json` -- see that file and
`../calibration/runtime/test_arbitrary_v1.json` for the synthetic test
geometry these markers are meant to exercise.

Usage:
    uv run python test-fixtures/arbitrary/scripts/generate_aruco_test_markers.py

Output: one PNG per marker under `--out-dir` (default
`data/raw/markers/test_arbitrary`, gitignored like the rest of `data/`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from _aruco_markers import DICTIONARY_NAME, MARKER_LABELS

DEFAULT_OUT_DIR = Path("data/raw/markers/test_arbitrary")
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
