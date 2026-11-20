import json
import numpy as np
import io_utils

left = io_utils.open_image("data/hero/level_left.png")
right = io_utils.open_image("data/hero/level_right.png")
h, w = left.shape[:2]
print(f"Actual image size: width={w}, height={h}")

# Slide a vertical strip from the left edge of mario2 across mario1 to find exact dx
best_dx, best_ssd = 0, float("inf")
strip = right[:, :max(20, w // 10)].astype(np.float64)
sw = strip.shape[1]
for dx in range(0, w - sw):
    patch = left[:, dx:dx + sw].astype(np.float64)
    ssd = np.sum((patch - strip) ** 2)
    if ssd < best_ssd:
        best_ssd = ssd
        best_dx = dx

overlap_w = w - best_dx
print(f"Exact horizontal offset dx = {best_dx} px (overlap width = {overlap_w} px)")

pts = {
    "src": [
        {"x": best_dx, "y": 0},
        {"x": w - 1, "y": 0},
        {"x": w - 1, "y": h - 1},
        {"x": best_dx, "y": h - 1},
        {"x": best_dx + overlap_w // 2, "y": h // 2}
    ],
    "dst": [
        {"x": 0, "y": 0},
        {"x": overlap_w - 1, "y": 0},
        {"x": overlap_w - 1, "y": h - 1},
        {"x": 0, "y": h - 1},
        {"x": overlap_w // 2, "y": h // 2}
    ]
}

with open("points.json", "w") as f:
    json.dump(pts, f, indent=2)
print("Saved exact points.json!")