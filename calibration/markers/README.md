# Modus B (ArUco) live testing — `dopo_poker_table`

Walkthrough for testing the full pipeline (capture → calibration →
tracking → assignment → state → debug/export) against the real DOPO POKER
table, using printed ArUco markers as stand-ins for chips/cards/dealer
button (REQ-19's Modus B — no trained chip/card model exists yet, that's
v0.2+).

## 1. Print the markers

```bash
uv run python calibration/markers/generate_marker_print_sheet.py
```

Writes `data/raw/markers/dopo_poker_table_a4.png` — print at **actual
size / 100%** (not "fit to page"), cut along the dashed lines, and stick
each piece onto its object:

- `chip_seat_1` .. `chip_seat_10` → one per seat's chip stack
- `dealer_button` → the physical dealer button
- `card_1` .. `card_5` → up to five board-card slots (place 3 for a flop
  test, 4 for turn, 5 for river)

(`generate_markers.py` writes the same 16 markers as separate PNGs instead,
if you'd rather print them individually.)

## 2. Test with photos first (`image_dir`)

The safest first test: no live-resolution surprises, because a photo taken
with the same iPhone camera at its native resolution matches the reference
photo's pixel space (4032x3024) exactly, which is what
`calibration/authoring/dopo_poker_table.json`'s zones are authored in.

1. Take one or more photos of the table (markers placed as above) from the
   same fixed top-down mount the reference photo was taken from, and drop
   them into `data/raw/images/dopo_poker_table/` (gitignored, create it if
   missing).
2. Validate config + calibration first, no camera/loop involved:
   ```bash
   uv run poker-vision validate --config configs/dopo_poker_table_images.json
   ```
3. Run it:
   ```bash
   uv run poker-vision run --config configs/dopo_poker_table_images.json
   ```
4. Watch it live at `http://localhost:8001/mjpeg` (zones, tracks,
   seat/dealer assignment, current state overlaid) while it processes the
   image directory, or read `data/exports/dopo_poker_table_images/` for the
   JSONL event log afterwards.

## 3. Test with the live iPhone stream (`continuity`)

Two things need to be right before this works, neither of which this repo
can verify without your actual hardware in front of it:

**a) `device_index`** — `configs/dopo_poker_table_livefeed.json` defaults
to `0`; that may not be your iPhone. Find the right index by trying a few:

```bash
uv run python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    print(i, 'ok' if ok else 'no frame', frame.shape if ok else '')
    cap.release()
"
```

Pick the index whose frame shape looks like your iPhone's feed (not your
Mac's built-in webcam), and set `source.device_index` in the config to it.

**b) Resolution/aspect ratio** — Continuity Camera used as a video source
very likely does **not** stream at the reference photo's 4032x3024 (4:3);
webcam-style video is commonly 1920x1080 (16:9). `resolution_cap` only ever
*downscales*, never upscales, and doesn't change aspect ratio — if your
live stream's aspect ratio differs from 4:3, the committed calibration's
zones will not line up with the frame at all (chips will appear to occupy
the wrong seats, or none).

Check what the printed frame `shape` above actually is. If its
width:height ratio isn't ~4:3 (4032:3024), don't just edit
`resolution_cap` — the zone geometry itself was authored in 4:3 pixel
space, so the fix is to derive a *new*, correctly-scaled runtime
calibration for your camera's actual live resolution via `calib
learn-table` (REQ-10b), which resizes/re-solves for whatever resolution
you feed it as long as the aspect ratio matches:

```bash
# One still frame from the live feed, empty table, same framing as the
# reference photo:
uv run python -c "
import cv2
cap = cv2.VideoCapture(<YOUR_DEVICE_INDEX>, cv2.CAP_AVFOUNDATION)
ok, frame = cap.read()
cv2.imwrite('data/raw/images/dopo_poker_table_live_frame.jpg', frame)
cap.release()
"

uv run calib learn-table \
  --reference-runtime calibration/runtime/dopo_poker_table.json \
  --reference-image calibration/reference/dopo_poker_table_blue_empty.jpeg \
  --live-image data/raw/images/dopo_poker_table_live_frame.jpg \
  --out calibration/runtime/dopo_poker_table_livefeed.json
```

If that succeeds, point `configs/dopo_poker_table_livefeed.json`'s
`paths.calibration_runtime` at the new file instead, and set
`source.resolution_cap` to match your live stream's actual frame shape
(so no downscaling occurs and the pixel space stays exactly what
`learn-table` solved for). If your live feed's aspect ratio genuinely
differs from 4:3 (not just a different pixel count at the same ratio),
`learn-table` will reject it with a clear error rather than silently
producing a broken mapping — in that case the reference photo itself would
need to be reframed and re-marked to your Continuity Camera's actual
aspect ratio.

Once both are set:

```bash
uv run poker-vision validate --config configs/dopo_poker_table_livefeed.json
uv run poker-vision run --config configs/dopo_poker_table_livefeed.json
```

Same MJPEG debug view at `http://localhost:8001/mjpeg` to confirm the
overlay actually lines up with the physical table before trusting the
event stream.
