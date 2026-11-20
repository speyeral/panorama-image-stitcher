"""Point JSON, calibration image, stamping, and determinant checks (M7)."""

from collections.abc import Mapping, Sequence
import json
from os import PathLike
from typing import Final, Literal, TypeAlias, TypedDict
import numpy as np
from numpy.typing import ArrayLike, NDArray
import linalg


# --- Interfaces & Type Aliases ---


class PointDict(TypedDict):
    """JSON dictionary representation of a single 2D coordinate.

    Example:
        `{"x": 120.5, "y": 340.0}`
    """

    x: float
    """Horizontal pixel coordinate (`x`)."""

    y: float
    """Vertical pixel coordinate (`y`)."""


FilePath: TypeAlias = str | PathLike[str]
"""Filesystem path to a JSON or image file.

Examples:
    `"data/corners.json"` or `Path("data/corners.json")`
"""

ImageArray: TypeAlias = NDArray[np.uint8]
"""Standard 8-bit image array (`0–255`), shape `(H, W, 3)` for RGB or `(H, W)` for grayscale.

Examples:
    `np.zeros((60, 40, 3), dtype=np.uint8)` or `np.zeros((60, 40), dtype=np.uint8)`
"""

FloatArray: TypeAlias = NDArray[np.float64]
"""General 64-bit float array (e.g., 2D edge displacement vectors).

Example:
    `np.array([15.0, -2.5], dtype=np.float64)`
"""

Point2D: TypeAlias = Sequence[float] | NDArray[np.floating]
"""Single 2D coordinate `(x, y)` as a sequence or floating-point array, shape `(2,)`.

Examples:
    `(10.0, 20.0)`, `[10.0, 20.0]`, or `np.array([10.0, 20.0], dtype=np.float64)`
"""

PointList2D: TypeAlias = NDArray[np.float64]
"""List of N 2D `(x, y)` coordinates, shape `(N, 2)`.

Example:
    `np.array([[2.0, 3.0], [37.0, 4.0], [36.0, 55.0], [5.0, 56.0]], dtype=np.float64)`
"""

Matrix3x3: TypeAlias = NDArray[np.float64]
"""3x3 homogeneous coordinate matrix (e.g., for collinearity determinant checks), shape `(3, 3)`.

Example:
    `np.array([[2.0, 3.0, 1.0], [37.0, 4.0, 1.0], [5.0, 56.0, 1.0]], dtype=np.float64)`
"""

WindingDirection: TypeAlias = Literal["clockwise", "counterclockwise"] | None
"""Expected polygon winding order on screen (`y` pointing down), or `None` to allow either.

Allowed values:
    `"clockwise"`, `"counterclockwise"`, or `None`
"""


# --- Constants ---

STAMP_RGB: Final[NDArray[np.uint8]] = np.array([255, 0, 255], dtype=np.uint8)
"""Bright magenta `[255, 0, 255]` RGB pixel value stamped onto color images by `mark_points`."""

STAMP_GRAY: Final[np.uint8] = np.uint8(255)
"""Pure white `255` pixel intensity stamped onto 2D grayscale images by `mark_points`."""


def load_points(path: FilePath) -> dict[str, PointList2D]:
    """Load named lists of {x, y} into a dict of (N, 2) float arrays."""
    data: dict[str, list[PointDict]]
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    named: dict[str, PointList2D] = {}
    for key, items in data.items():
        named[key] = np.array([[pt["x"], pt["y"]] for pt in items], dtype=np.float64)
    return named


def save_points(path: FilePath, named_pts: Mapping[str, ArrayLike]) -> None:
    """Write arrays via .tolist() so JSON round-trips exactly for finite floats."""
    obj: dict[str, list[PointDict]] = {}
    for key, pts in named_pts.items():
        arr: PointList2D = np.array(pts, dtype=np.float64)
        rows: list[list[float]] = arr.tolist()
        obj[key] = [{"x": row[0], "y": row[1]} for row in rows]

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)


def make_calibration_image(height: int = 60, width: int = 40) -> tuple[ImageArray, PointList2D]:
    """Non-square image with markers at known integer (x, y). Index as img[y, x]."""
    img: ImageArray = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    pts: PointList2D = np.array(
        [
            [2.0, 3.0],
            [width - 3, 4.0],
            [5.0, height - 4],
            [width - 4, height - 5],
        ],
        dtype=np.float64,
    )
    colors: NDArray[np.uint8] = np.array(
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


def mark_points(img: ArrayLike, pts: ArrayLike) -> ImageArray:
    """Copy img and stamp each rounded (x, y) pixel. Not interactive."""
    out: ImageArray = np.array(img, dtype=np.uint8).copy()
    h: int = out.shape[0]
    w: int = out.shape[1]

    pts_arr: PointList2D = np.array(pts, dtype=np.float64)
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
        raise ValueError("pts must be (N, 2) as (x, y)")

    x: float
    y: float
    for x, y in pts_arr:
        xi: int = int(round(x))
        yi: int = int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            raise ValueError(f"point ({x}, {y}) is out of bounds")
        if out.ndim == 2:
            out[yi, xi] = STAMP_GRAY
        else:
            out[yi, xi] = STAMP_RGB
    return out


def are_collinear(p: Point2D, q: Point2D, r: Point2D, tol: float = 1e-6) -> bool:
    """Three points are collinear iff det of rows [x y 1] is (near) zero."""
    m: Matrix3x3 = np.array(
        [
            [p[0], p[1], 1.0],
            [q[0], q[1], 1.0],
            [r[0], r[1], 1.0],
        ],
        dtype=np.float64,
    )
    return abs(float(linalg.det(m))) <= tol


def corner_order_ok(
    pts: ArrayLike,
    tol: float = 1e-12,
    winding: WindingDirection = None,
) -> bool:
    """Reject mixed signs and zeros of consecutive 2x2 edge determinants.

    Points are (x, y) with y down. Clicking TL, TR, BR, BL is clockwise on
    screen and gives positive signs. By default either consistent winding is
    accepted. Pass winding="clockwise" (positive signs) or
    winding="counterclockwise" (negative signs) to require one direction.
    """
    if winding not in (None, "clockwise", "counterclockwise"):
        raise ValueError('winding must be None, "clockwise" or "counterclockwise"')

    pts_arr: PointList2D = np.array(pts, dtype=np.float64)
    if pts_arr.shape != (4, 2):
        raise ValueError("corner_order_ok expects four (x, y) points")

    signs: list[int] = []
    for i in range(4):
        v: FloatArray = pts_arr[(i + 1) % 4] - pts_arr[i]
        w: FloatArray = pts_arr[(i + 2) % 4] - pts_arr[(i + 1) % 4]
        det2: float = float(v[0] * w[1] - v[1] * w[0])
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