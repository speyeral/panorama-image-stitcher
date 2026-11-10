# Dataset (Person 3, S1)

Working-size PNGs after EXIF undo, long side about 800 pixels.

## JSON

Named lists of `{x, y}` with x = column, y = row, 0-based.

- Rectify: `{"corners": [tl, tr, br, bl]}` (clockwise on screen).
- Overlap pair: `{"src": [...], "dst": [...]}` with aligned indices (8–12 pairs).

## Contents

- `rectify/` — rectangle photographed at an angle (replace `PLACEHOLDER_rectify.png` with a real shot) plus `corners.json`.
- `overlap/` — two photos, same position, fixed exposure, 30–40% overlap. Replace placeholders; keep `pairs.json` aligned.
- `synthetic_known_h/` — generated pair with documented `H`.
- `calibration/` — output of `make_calibration_image` for a one-time viewer readout check.

## Capture notes

Shoot JPEG, apply EXIF orientation, then `resize` so the long side is 800, save PNG. Type coordinates from an external viewer (Preview) into JSON; do not click inside this repo.
