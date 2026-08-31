"""Compose all REQ-19 ArUco test markers onto one printable A4 sheet.

`generate_aruco_test_markers.py` writes one PNG per marker, which
is fine for digital use but awkward to print one at a time. This script
lays the same 12 markers (shared definitions in `_aruco_markers.py`, so the
two scripts can't drift apart) into a 3x4 grid on a single A4 page at
300 DPI, each with its label printed above it and a dashed cut line around
the label+marker pair, so printing, cutting along the dashed lines, and
sticking each piece onto its object is a single pass.

Usage:
    uv run python test-fixtures/arbitrary/scripts/generate_marker_print_sheet.py

Output: data/raw/images/aruco_markers_a4.png (gitignored like the rest of
`data/`), print at "actual size" / 100% -- the PNG carries 300 DPI metadata
so a naive "fit to page" print would only be a minor deviation, but actual
size keeps marker dimensions predictable for the camera-distance framing
in zone_reference_4032x3024.png.
"""

from __future__ import annotations

import cv2
from _aruco_markers import DICTIONARY_NAME, MARKER_LABELS
from PIL import Image, ImageDraw, ImageFont

OUT_PATH = "data/raw/images/aruco_markers_a4.png"

DPI = 300
PAGE_W, PAGE_H = round(8.27 * DPI), round(11.69 * DPI)  # A4 portrait
MARGIN = 100
TITLE_H = 150
COLS, ROWS = 3, 4
GAP = 50
MARKER_SIZE = 600
LABEL_H = 70
CELL_PAD = 20  # inset between the cut line and its label+marker content

_BACKGROUND = (255, 255, 255)
_INK = (30, 30, 30)
_CUT_LINE = (120, 120, 120)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_rect(
    draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: tuple, width: int
) -> None:
    dash, gap = 14, 8
    for ax, ay, bx, by in ((x1, y1, x2, y1), (x1, y2, x2, y2)):
        x = ax
        while x < bx:
            draw.line([(x, ay), (min(x + dash, bx), by)], fill=color, width=width)
            x += dash + gap
    for ax, ay, bx, by in ((x1, y1, x1, y2), (x2, y1, x2, y2)):
        y = ay
        while y < by:
            draw.line([(ax, y), (bx, min(y + dash, by))], fill=color, width=width)
            y += dash + gap


def _marker_image(marker_id: int) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    array = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
    return Image.fromarray(array).convert("L")


def main() -> None:
    page = Image.new("RGB", (PAGE_W, PAGE_H), _BACKGROUND)
    draw = ImageDraw.Draw(page)

    title_font = _font(48)
    label_font = _font(34)

    draw.text(
        (MARGIN, MARGIN // 2),
        "ArUco-Testmarker (REQ-19) -- entlang der gestrichelten Linie ausschneiden",
        font=title_font,
        fill=_INK,
    )

    usable_w = PAGE_W - 2 * MARGIN
    usable_h = PAGE_H - 2 * MARGIN - TITLE_H
    cell_w = (usable_w - (COLS - 1) * GAP) // COLS
    cell_h = (usable_h - (ROWS - 1) * GAP) // ROWS

    grid_top = MARGIN + TITLE_H

    for index, (marker_id, label) in enumerate(MARKER_LABELS.items()):
        row, col = divmod(index, COLS)
        cell_x = MARGIN + col * (cell_w + GAP)
        cell_y = grid_top + row * (cell_h + GAP)

        _dashed_rect(
            draw, cell_x, cell_y, cell_x + cell_w, cell_y + cell_h, _CUT_LINE, width=2
        )

        caption = f"ID {marker_id:02d} - {label}"
        text_bbox = draw.textbbox((0, 0), caption, font=label_font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text(
            (cell_x + (cell_w - text_w) // 2, cell_y + CELL_PAD),
            caption,
            font=label_font,
            fill=_INK,
        )

        marker_img = _marker_image(marker_id)
        marker_x = cell_x + (cell_w - MARKER_SIZE) // 2
        marker_y = cell_y + LABEL_H
        page.paste(marker_img, (marker_x, marker_y))

    page.save(OUT_PATH, dpi=(DPI, DPI))
    print(f"wrote {OUT_PATH} ({PAGE_W}x{PAGE_H} @ {DPI} DPI, A4)")


if __name__ == "__main__":
    main()
