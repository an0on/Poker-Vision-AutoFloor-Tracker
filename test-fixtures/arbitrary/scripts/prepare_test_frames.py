"""Prepare arbitrary phone photos for the REQ-19 ArUco wiring test.

`Detector._check_resolution` (src/poker_vision/detection/base.py) rejects
any frame whose pixel size doesn't *exactly* match
`../calibration/runtime/test_arbitrary_v1.json`'s `inference_resolution`
(1920x1080) -- and `image_dir`'s resolution cap only ever shrinks,
preserving aspect ratio, never pads/crops/upscales. A phone photo in
portrait orientation, at a small export size, or at the iPhone's native
4:3 capture ratio will therefore never land on exactly 1920x1080 no matter
what `resolution_cap` says, and every frame gets dropped with a
"resolution does not match" error.

This script is the fix: point it at a folder of raw, arbitrary-sized/
oriented photos, and it writes 1920x1080 landscape versions -- portrait
input rotated 90 degrees, then centre-cropped to 16:9 and resized -- into
`../configs/test_arbitrary.json`'s `source.path`.

Usage:
    uv run python test-fixtures/arbitrary/scripts/prepare_test_frames.py

Default input:  data/raw/images/test_arbitrary/_source/  (drop raw photos here)
Default output: data/raw/images/test_arbitrary/           (pipeline reads this)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_IN_DIR = Path("data/raw/images/test_arbitrary/_source")
DEFAULT_OUT_DIR = Path("data/raw/images/test_arbitrary")
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080

# Below this on its shorter side, upscaling to 1920x1080 blurs a printed
# ArUco marker past the point the detector can find it -- worth a warning
# rather than a silent low-quality frame.
_LOW_RES_WARNING_THRESHOLD = 1080


def _rotate_to_landscape(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _center_crop_to_aspect(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    height, width = image.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = width / height
    if current_ratio > target_ratio:
        # Wider than target: crop the sides.
        new_width = round(height * target_ratio)
        x0 = (width - new_width) // 2
        return image[:, x0 : x0 + new_width]
    # Taller than target: crop top/bottom.
    new_height = round(width / target_ratio)
    y0 = (height - new_height) // 2
    return image[y0 : y0 + new_height, :]


def prepare_frame(image: np.ndarray) -> np.ndarray:
    landscape = _rotate_to_landscape(image)
    cropped = _center_crop_to_aspect(landscape, TARGET_WIDTH, TARGET_HEIGHT)
    shorter_side = min(cropped.shape[:2])
    interpolation = cv2.INTER_AREA if shorter_side >= TARGET_HEIGHT else cv2.INTER_CUBIC
    return cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=interpolation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.in_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in args.in_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not sources:
        print(f"no images found in {args.in_dir} -- drop raw photos there first")
        return

    for path in sources:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skipped {path}: unreadable")
            continue
        shorter_side = min(image.shape[:2])
        if shorter_side < _LOW_RES_WARNING_THRESHOLD:
            print(
                f"warning: {path.name} is {image.shape[1]}x{image.shape[0]} -- "
                f"upscaling to {TARGET_WIDTH}x{TARGET_HEIGHT} will blur markers; "
                "re-export at full/actual size if detection misses them"
            )
        prepared = prepare_frame(image)
        out_path = args.out_dir / path.name
        cv2.imwrite(str(out_path), prepared)
        print(f"wrote {out_path} ({TARGET_WIDTH}x{TARGET_HEIGHT})")


if __name__ == "__main__":
    main()
