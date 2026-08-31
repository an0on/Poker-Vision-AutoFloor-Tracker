"""Generate a full-resolution placement guide for the REQ-19 ArUco wiring test.

A blank 4032x3024 image (the iPhone's native landscape 4:3 capture size)
with the same zones as the chat sketch drawn on it -- but at *raw camera*
scale, not the 1920x1080 the pipeline actually processes. That distinction
matters: `prepare_test_frames.py` centre-crops every raw photo down
to 16:9 before resizing to 1920x1080, so a raw 4:3 photo has margin strips
top and bottom that get thrown away. This guide draws that crop band
explicitly (dashed line + note) so placing a marker "in the zone" means the
same thing on this printout as it will after processing -- not just a naive
4032x3024 stretch of the 1920x1080 zone coordinates, which would put
everything in the wrong place relative to what survives the crop.

View this full-screen or print it at actual size, lay it next to (or under)
the camera, and align markers against it before shooting.

Usage (from repo root):
    uv run python test-fixtures/arbitrary/scripts/generate_zone_reference_image.py
"""

from __future__ import annotations

import cv2
import numpy as np

from poker_vision.calibration.geometry import TablePoint
from poker_vision.calibration.runtime import CalibrationRuntime, load_calibration_runtime

CALIBRATION_PATH = "test-fixtures/arbitrary/calibration/runtime/test_arbitrary_v1.json"
OUT_PATH = "data/raw/images/zone_reference_4032x3024.png"

RAW_WIDTH, RAW_HEIGHT = 4032, 3024  # iPhone native landscape (4:3) capture

_BACKGROUND = (235, 235, 235)  # light gray, BGR
_CROP_BAND_COLOR = (90, 90, 90)
_PLAYER_AREA_COLOR = (170, 170, 170)
_CHIP_ZONE_COLOR = (0, 130, 190)  # amber-ish, BGR
_BOARD_ZONE_COLOR = (190, 90, 0)  # blue-ish, BGR
_DEALER_AREA_COLOR = (150, 90, 190)  # purple-ish, BGR


def _crop_band(raw_w: int, raw_h: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Same centre-crop-to-target-aspect geometry as `prepare_test_frames._center_crop_to_aspect`.

    Returns (x_offset, y_offset, crop_width, crop_height) in raw pixels.
    """
    target_ratio = target_w / target_h
    current_ratio = raw_w / raw_h
    if current_ratio > target_ratio:
        crop_w = round(raw_h * target_ratio)
        return (raw_w - crop_w) // 2, 0, crop_w, raw_h
    crop_h = round(raw_w / target_ratio)
    return 0, (raw_h - crop_h) // 2, raw_w, crop_h


def _draw_dashed_rect(
    canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int
) -> None:
    dash, gap = 30, 20
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x1, y2), (x2, y2))):
        x = ax
        while x < bx:
            cv2.line(canvas, (x, ay), (min(x + dash, bx), by), color, thickness, cv2.LINE_AA)
            x += dash + gap
    for (ax, ay), (bx, by) in (((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))):
        y = ay
        while y < by:
            cv2.line(canvas, (ax, y), (bx, min(y + dash, by)), color, thickness, cv2.LINE_AA)
            y += dash + gap


def main() -> None:
    calibration: CalibrationRuntime = load_calibration_runtime(CALIBRATION_PATH)
    proc_w = calibration.inference_resolution.width
    proc_h = calibration.inference_resolution.height

    x_off, y_off, crop_w, crop_h = _crop_band(RAW_WIDTH, RAW_HEIGHT, proc_w, proc_h)
    scale = crop_w / proc_w  # == crop_h / proc_h

    def to_raw(point: TablePoint) -> tuple[int, int]:
        return round(x_off + point.x * scale), round(y_off + point.y * scale)

    canvas = np.full((RAW_HEIGHT, RAW_WIDTH, 3), _BACKGROUND, dtype=np.uint8)

    _draw_dashed_rect(
        canvas, x_off, y_off, x_off + crop_w, y_off + crop_h, _CROP_BAND_COLOR, thickness=5
    )
    cv2.putText(
        canvas,
        "wird nach Verarbeitung 1920x1080 (alles ausserhalb wird zugeschnitten)",
        (x_off + 20, y_off - 30 if y_off > 60 else y_off + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        _CROP_BAND_COLOR,
        3,
        cv2.LINE_AA,
    )

    def draw_zone(
        points: list[TablePoint], color: tuple, label: str, fill: bool, thickness: int = 6
    ) -> None:
        pts_raw = np.array([[to_raw(p) for p in points]], dtype=np.int32)
        if fill:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, pts_raw, color)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        cv2.polylines(canvas, pts_raw, isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if label:
            cx = int(pts_raw[0][:, 0].mean())
            cy = int(pts_raw[0][:, 1].mean())
            font_scale = 2.6
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)
            cv2.putText(
                canvas,
                label,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                5,
                cv2.LINE_AA,
            )

    for seat in calibration.seats:
        draw_zone(seat.zones.player_area.points, _PLAYER_AREA_COLOR, "", fill=False, thickness=3)
    for seat in calibration.seats:
        label = seat.seat_id.replace("seat_", "S")
        draw_zone(seat.zones.chip_zone.points, _CHIP_ZONE_COLOR, label, fill=True)

    draw_zone(calibration.zones.board_zone.points, _BOARD_ZONE_COLOR, "board", fill=True)
    draw_zone(calibration.zones.dealer_area.points, _DEALER_AREA_COLOR, "btn", fill=True)

    legend_y = RAW_HEIGHT - 60
    legend_items = [
        (_CHIP_ZONE_COLOR, "Chip-Zone (je Sitz)"),
        (_BOARD_ZONE_COLOR, "Board-Zone"),
        (_DEALER_AREA_COLOR, "Dealer-Bereich"),
        (_PLAYER_AREA_COLOR, "Spielerbereich (Rahmen)"),
    ]
    x = 40
    for color, text in legend_items:
        cv2.rectangle(canvas, (x, legend_y), (x + 50, legend_y + 50), color, -1)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cv2.putText(
            canvas, text, (x + 65, legend_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 40), 3, cv2.LINE_AA
        )
        x += 65 + tw + 60

    cv2.imwrite(OUT_PATH, canvas)
    print(f"wrote {OUT_PATH} ({RAW_WIDTH}x{RAW_HEIGHT})")


if __name__ == "__main__":
    main()
