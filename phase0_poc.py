#!/usr/bin/env python3
"""Phase 0 sandbox proof of concept for Poker Vision (PRD REQ-0.1 ... REQ-0.8).

Deliberately a single standalone script: no package structure, no config file,
no JSON parsing, no calibration, no homography, no tracking, no state machine
(REQ-0.1). Everything below happens in raw pixel space.

Pipeline:
  1. Run a pretrained COCO model (default YOLOv8n) on one static image
     (REQ-0.2, REQ-0.3). Device is "cpu" or "mps" only - never CUDA.
  2. Compute the exact bounding box centre of every relevant detection
     (REQ-0.4).
  3. Nearest-neighbour link: for the dealer-button placeholder, pick the
     closest detection of the *other* class (REQ-0.6) and compute the
     euclidean distance between the two centres (REQ-0.5).
  4. Write an annotated image with the "rubber band" line, both centre
     markers and the distance label (REQ-0.7).
  5. Abort loudly, without writing any output file, when the two expected
     classes are not found (REQ-0.8).

Placeholder mapping (REQ-0.2), overridable via CLI:
  COCO "mouse"      -> dealer button
  COCO "cell phone" -> chip stack

Example:
  python phase0_poc.py desk.jpg
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_DEALER_CLASS = "mouse"
DEFAULT_CHIP_CLASS = "cell phone"
DEFAULT_CONF = 0.25

# BGR colours for the overlay.
COLOR_DEALER = (0, 165, 255)
COLOR_CHIP = (0, 200, 0)
COLOR_OTHER = (150, 150, 150)
COLOR_BAND = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)

EXIT_ABORT = 2


@dataclass(frozen=True)
class Detection:
    """One COCO detection in pixel space."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        """Exact bounding box centre in pixels (REQ-0.4)."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def abort(message: str) -> None:
    """Stop with a clear message and without producing an output file (REQ-0.8)."""
    sys.stdout.flush()  # keep the abort message ordered after the report when piped
    print(f"ABBRUCH: {message}", file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance (Pythagoras) between two points (REQ-0.5)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_neighbour(
    anchor: Detection, candidates: list[Detection]
) -> tuple[Detection, float]:
    """Return the candidate closest to `anchor` plus its distance (REQ-0.6).

    Formulated as a generic nearest-neighbour search on purpose: with two
    objects the result is trivial, but the logic must not hard-code the pairing.
    """
    ranked = sorted(candidates, key=lambda d: euclidean(anchor.center, d.center))
    best = ranked[0]
    return best, euclidean(anchor.center, best.center)


def resolve_device(requested: str) -> str:
    """Resolve the inference device. CUDA is rejected by design (REQ-0.3)."""
    normalised = requested.strip().lower()
    if normalised.startswith("cuda") or normalised.startswith("gpu"):
        abort("Device 'cuda' ist in diesem Projekt nicht zulaessig. Erlaubt: cpu, mps.")
    if normalised in ("cpu", "mps"):
        return normalised
    if normalised != "auto":
        abort(f"Unbekanntes Device '{requested}'. Erlaubt: auto, cpu, mps.")

    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def detect(
    model_path: str, image_path: Path, device: str, conf: float
) -> list[Detection]:
    """Run the pretrained COCO model and return all detections above `conf`."""
    try:
        from ultralytics import YOLO
    except ImportError:
        abort(
            "Paket 'ultralytics' fehlt. Installation z. B.:\n"
            "  uv pip install ultralytics opencv-python"
        )

    model = YOLO(model_path)
    results = model.predict(
        source=str(image_path), device=device, conf=conf, verbose=False
    )
    if not results:
        abort("Das Modell hat kein Ergebnis fuer dieses Bild geliefert.")

    names = model.names
    detections: list[Detection] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                label=names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=(x1, y1, x2, y2),
            )
        )
    return detections


def put_label(
    image, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float
) -> None:
    """Draw text on a filled box so it stays readable on any background."""
    import cv2

    thickness = max(1, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - height - baseline - 4),
        (x + width + 4, y + baseline),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def draw_marker(
    image, center: tuple[float, float], color: tuple[int, int, int], radius: int
) -> None:
    """Draw a filled dot plus crosshair at the bounding box centre."""
    import cv2

    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(image, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius * 2, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(image, (cx - radius * 3, cy), (cx + radius * 3, cy), color, 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - radius * 3), (cx, cy + radius * 3), color, 1, cv2.LINE_AA)


def render(
    image_path: Path,
    dealer: Detection,
    chip: Detection,
    distance: float,
    other_candidates: list[Detection],
    output_path: Path,
) -> None:
    """Draw the rubber band, both centres and the distance, then save (REQ-0.7)."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        abort(f"Bild konnte nicht gelesen werden: {image_path}")

    height, width = image.shape[:2]
    scale = max(0.5, min(width, height) / 900.0)
    line_thickness = max(2, int(round(3 * scale)))
    radius = max(3, int(round(5 * scale)))

    # Rejected nearest-neighbour candidates stay visible but muted, so the
    # choice made in REQ-0.6 can be checked by eye.
    for candidate in other_candidates:
        x1, y1, x2, y2 = (int(round(v)) for v in candidate.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_OTHER, 1)
        draw_marker(image, candidate.center, COLOR_OTHER, max(2, radius // 2))
        put_label(
            image,
            f"{candidate.label} (verworfen)",
            (x1, max(20, y1 - 6)),
            COLOR_OTHER,
            0.5 * scale,
        )

    for detection, color in ((dealer, COLOR_DEALER), (chip, COLOR_CHIP)):
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        put_label(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            color,
            0.6 * scale,
        )

    # The rubber band itself.
    p1 = (int(round(dealer.center[0])), int(round(dealer.center[1])))
    p2 = (int(round(chip.center[0])), int(round(chip.center[1])))
    cv2.line(image, p1, p2, COLOR_BAND, line_thickness, cv2.LINE_AA)
    draw_marker(image, dealer.center, COLOR_DEALER, radius)
    draw_marker(image, chip.center, COLOR_CHIP, radius)

    midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    put_label(
        image,
        f"d = {distance:.1f} px",
        (midpoint[0] + 10, midpoint[1] - 10),
        (40, 40, 40),
        0.8 * scale,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        abort(f"Ergebnisbild konnte nicht geschrieben werden: {output_path}")
    print(f"Ergebnisbild gespeichert: {output_path}")


def show(image_path: Path) -> None:
    """Best-effort preview; silently skipped on headless setups."""
    import cv2

    try:
        image = cv2.imread(str(image_path))
        cv2.imshow("Phase 0 - Gummiband", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        print(f"Hinweis: Anzeige nicht moeglich ({exc}).", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 sandbox PoC: nearest-neighbour rubber band between "
        "two COCO placeholder objects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Input image (static photo).")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output image path."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained COCO model.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu or mps (cuda is rejected).",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold."
    )
    parser.add_argument(
        "--dealer-class",
        default=DEFAULT_DEALER_CLASS,
        help="COCO class used as dealer-button placeholder.",
    )
    parser.add_argument(
        "--chip-class",
        default=DEFAULT_CHIP_CLASS,
        help="COCO class used as chip-stack placeholder.",
    )
    parser.add_argument("--show", action="store_true", help="Show the result window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.image.is_file():
        abort(f"Eingabebild nicht gefunden: {args.image}")
    if args.dealer_class == args.chip_class:
        abort("--dealer-class und --chip-class muessen unterschiedlich sein.")

    device = resolve_device(args.device)
    print(f"Bild:   {args.image}")
    print(f"Modell: {args.model}   Device: {device}   conf>={args.conf}")

    detections = detect(args.model, args.image, device, args.conf)
    dealers = [d for d in detections if d.label == args.dealer_class]
    chips = [d for d in detections if d.label == args.chip_class]

    print(f"Detections gesamt: {len(detections)}")
    for d in detections:
        print(f"  - {d.label:<14} conf={d.confidence:.3f} box={d.box}")

    # REQ-0.8: no silent fallback if the expected classes are missing.
    if not dealers:
        abort(
            f"Klasse '{args.dealer_class}' (Dealer Button) nicht gefunden. "
            f"Gefunden: {sorted({d.label for d in detections}) or 'nichts'}. "
            "Kein Ergebnisbild geschrieben."
        )
    if not chips:
        abort(
            f"Klasse '{args.chip_class}' (Chip-Haufen) nicht gefunden. "
            f"Gefunden: {sorted({d.label for d in detections}) or 'nichts'}. "
            "Kein Ergebnisbild geschrieben."
        )
    if len(dealers) > 1:
        abort(
            f"{len(dealers)}x Klasse '{args.dealer_class}' gefunden, erwartet genau 1 "
            "(der Dealer Button ist der Anker der Nearest-Neighbour-Suche). "
            "Kein Ergebnisbild geschrieben."
        )

    dealer = dealers[0]
    chip, distance = nearest_neighbour(dealer, chips)
    rejected = [c for c in chips if c is not chip]

    print("")
    print(f"Dealer Button  ({dealer.label}):  Mittelpunkt = "
          f"({dealer.center[0]:.2f}, {dealer.center[1]:.2f})  box={dealer.box}")
    print(f"Chip-Haufen    ({chip.label}):  Mittelpunkt = "
          f"({chip.center[0]:.2f}, {chip.center[1]:.2f})  box={chip.box}")
    if rejected:
        print(f"Nearest Neighbour: {len(chips)} Kandidaten, "
              f"{len(rejected)} verworfen (groessere Distanz).")
    print(f"Euklidische Distanz: {distance:.2f} px")
    print("")

    output_path = args.output or args.image.with_name(f"{args.image.stem}_phase0.png")
    render(args.image, dealer, chip, distance, rejected, output_path)

    if args.show:
        show(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
