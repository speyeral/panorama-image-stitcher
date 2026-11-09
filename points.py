"""Point JSON, calibration image, stamping, and determinant checks (M7)."""

import json
import numpy as np
import linalg

STAMP_RGB = np.array([255, 0, 255], dtype=np.uint8)
STAMP_GRAY = np.uint8(255)


def load_points(path):
    """Load named lists of {x, y} into a dict of (N, 2) float arrays."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    named = {}
    for key, items in data.items():
        named[key] = np.array([[pt["x"], pt["y"]] for pt in items], dtype=float)
    return named


def save_points(path, named_pts):
    """Write arrays via .tolist() so JSON round-trips exactly for finite floats."""
    obj = {}
    for key, pts in named_pts.items():
        arr = np.array(pts, dtype=float)
        obj[key] = [{"x": row[0], "y": row[1]} for row in arr.tolist()]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)


def make_calibration_image(height=60, width=40):
    """Non-square image with markers at known integer (x, y). Index as img[y, x]."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    pts = np.array(
        [
            [2.0, 3.0],
            [width - 3, 4.0],
            [5.0, height - 4],
            [width - 4, height - 5],
        ],
        dtype=float,
    )
    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )
    for (x, y), color in zip(pts, colors):
        img[int(y), int(x)] = color
    return img, pts


def mark_points(img, pts):
    """Copy img and stamp each rounded (x, y) pixel. Not interactive."""
    out = np.array(img).copy()
    h, w = out.shape[0], out.shape[1]
    pts = np.array(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be (N, 2) as (x, y)")
    for x, y in pts:
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            raise ValueError("point ({}, {}) is out of bounds".format(x, y))
        if out.ndim == 2:
            out[yi, xi] = STAMP_GRAY
        else:
            out[yi, xi] = STAMP_RGB
    return out


def are_collinear(p, q, r, tol=1e-6):
    """Three points are collinear iff det of rows [x y 1] is (near) zero."""
    m = np.array(
        [
            [p[0], p[1], 1.0],
            [q[0], q[1], 1.0],
            [r[0], r[1], 1.0],
        ],
        dtype=float,
    )
    return abs(linalg.det(m)) <= tol


def corner_order_ok(pts, tol=1e-12, winding=None):
    """Reject mixed signs and zeros of consecutive 2x2 edge determinants.

    Points are (x, y) with y down. Clicking TL, TR, BR, BL is clockwise on
    screen and gives positive signs. By default either consistent winding is
    accepted. Pass winding="clockwise" (positive signs) or
    winding="counterclockwise" (negative signs) to require one direction.
    """
    if winding not in (None, "clockwise", "counterclockwise"):
        raise ValueError('winding must be None, "clockwise" or "counterclockwise"')
    pts = np.array(pts, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError("corner_order_ok expects four (x, y) points")
    signs = []
    for i in range(4):
        v = pts[(i + 1) % 4] - pts[i]
        w = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        det2 = v[0] * w[1] - v[1] * w[0]
        if abs(det2) <= tol:
            return False
        signs.append(1 if det2 > 0 else -1)
    if not all(s == signs[0] for s in signs):
        return False
    if winding == "clockwise":
        return signs[0] > 0
    if winding == "counterclockwise":
        return signs[0] < 0
    return True
