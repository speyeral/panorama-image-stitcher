"""Image ops as linear maps: grayscale (matrix-vector), flips (permutation), resize R I C^T."""

from typing import Final, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- Type Aliases ---

ImageArray: TypeAlias = NDArray[np.uint8]
"""Standard 8-bit image array (`0–255`), shape `(H, W, 3)` for RGB or `(H, W)` for grayscale.

Examples:
    `np.zeros((1080, 1920, 3), dtype=np.uint8)` or `np.zeros((480, 640), dtype=np.uint8)`
"""

FloatArray: TypeAlias = NDArray[np.float64]
"""64-bit float image or intermediate calculation array, shape `(H, W)` or `(H, W, C)`.

Example:
    `np.zeros((480, 640, 3), dtype=np.float64)`
"""

FloatMatrix: TypeAlias = NDArray[np.float64]
"""2D 64-bit float linear operator matrix (e.g., exchange/permutation or averaging matrix), shape `(M, N)`.

Example:
    `np.eye(3, dtype=np.float64)[::-1]`
"""

WeightVector: TypeAlias = NDArray[np.float64]
"""1D 3-element RGB-to-grayscale luminance weight vector `[w_R, w_G, w_B]`, shape `(3,)`.

Example:
    `np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)`
"""

# --- Constants ---

WEIGHTS: Final[WeightVector] = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
"""Rec. 601 luminance weights `[0.2989, 0.5870, 0.1140]` for converting RGB pixels to grayscale via `I @ WEIGHTS`."""


def _as_array(img: ArrayLike) -> ImageArray:
    return np.array(img, dtype=np.uint8)


def _exchange(n: int) -> FloatMatrix:
    """n x n permutation (exchange) matrix that reverses index order."""
    return np.eye(n, dtype=np.float64)[::-1]


def _apply_left_right(
    img: ImageArray,
    left: FloatMatrix | None = None,
    right: FloatMatrix | None = None,
) -> FloatArray:
    """Color-safe products: left @ I and/or I @ right, per channel if RGB."""
    arr: FloatArray = img.astype(np.float64)
    if arr.ndim == 2:
        if left is not None:
            arr = left @ arr
        if right is not None:
            arr = arr @ right
        return arr
    if arr.ndim == 3:
        out: FloatArray = np.empty_like(arr)
        for c in range(arr.shape[2]):
            plane: FloatMatrix = arr[:, :, c]
            if left is not None:
                plane = left @ plane
            if right is not None:
                plane = plane @ right
            out[:, :, c] = plane
        return out
    raise ValueError("expected HxW or HxWxC image")


def grayscale(img: ArrayLike) -> ImageArray:
    """Matrix-vector product with Rec.601-style weights; round to uint8."""
    arr: ImageArray = _as_array(img)
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("grayscale expects HxW or HxWx3")
    gray: FloatArray = np.rint(arr.astype(np.float64) @ WEIGHTS)
    return np.clip(gray, 0, 255).astype(np.uint8)


def flip_v(img: ArrayLike) -> ImageArray:
    """Vertical flip: left-multiply by the H x H exchange matrix."""
    arr: ImageArray = _as_array(img)
    p: FloatMatrix = _exchange(arr.shape[0])
    out: FloatArray = _apply_left_right(arr, left=p)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def flip_h(img: ArrayLike) -> ImageArray:
    """Horizontal flip: right-multiply by the W x W exchange matrix."""
    arr: ImageArray = _as_array(img)
    q: FloatMatrix = _exchange(arr.shape[1])
    out: FloatArray = _apply_left_right(arr, right=q)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def transpose_img(img: ArrayLike) -> ImageArray:
    """Swap rows and columns (permutation of axes). Color channel stays last."""
    arr: ImageArray = _as_array(img)
    if arr.ndim == 2:
        return arr.T.copy()
    if arr.ndim == 3:
        return np.transpose(arr, (1, 0, 2)).copy()
    raise ValueError("expected HxW or HxWxC image")


def rotate90(img: ArrayLike, k: int = 1) -> ImageArray:
    """k quarter-turns counter-clockwise, matching np.rot90 with y down / row 0 at top."""
    arr: ImageArray = _as_array(img)
    turns: int = int(k) % 4
    out: ImageArray = arr
    for _ in range(turns):
        # transpose then reverse rows = 90 degrees CCW
        out = flip_v(transpose_img(out))
    return out


def crop(img: ArrayLike, x0: int, y0: int, x1: int, y1: int) -> ImageArray:
    """Slice img[y0:y1, x0:x1] and copy so later stamps cannot mutate the source."""
    arr: ImageArray = _as_array(img)
    return arr[y0:y1, x0:x1].copy()


def _averaging_matrix(n_old: int, n_new: int) -> FloatMatrix:
    """Map length n_old to n_new by box averaging. Each row sums to 1."""
    mat: FloatMatrix = np.zeros((n_new, n_old), dtype=np.float64)
    for i in range(n_new):
        start: float = i * n_old / float(n_new)
        end: float = (i + 1) * n_old / float(n_new)
        last_j: int = 0
        for j in range(n_old):
            last_j = j
            overlap: float = min(end, j + 1) - max(start, j)
            if overlap > 0:
                mat[i, j] = overlap
        row_sum: float = float(mat[i].sum())
        if row_sum > 0:
            mat[i] /= row_sum
        else:
            mat[i, min(last_j, n_old - 1)] = 1.0
    return mat


def resize_matrices(old_h: int, old_w: int, new_h: int, new_w: int) -> tuple[FloatMatrix, FloatMatrix]:
    """R (new_h x old_h) and C (new_w x old_w) for I' = R I C^T. Rows sum to 1."""
    r: FloatMatrix = _averaging_matrix(old_h, new_h)
    c: FloatMatrix = _averaging_matrix(old_w, new_w)
    return r, c


def resize(img: ArrayLike, new_h: int, new_w: int) -> ImageArray:
    """Box-average resize as the matrix product R I C^T (Section 1.3)."""
    arr: ImageArray = _as_array(img)
    old_h: int = arr.shape[0]
    old_w: int = arr.shape[1]

    r: FloatMatrix
    c: FloatMatrix
    r, c = resize_matrices(old_h, old_w, new_h, new_w)
    c_t: FloatMatrix = c.T

    out: FloatArray
    if arr.ndim == 2:
        out = r @ arr.astype(np.float64) @ c_t
    else:
        out = np.empty((new_h, new_w, arr.shape[2]), dtype=np.float64)
        for ch in range(arr.shape[2]):
            out[:, :, ch] = r @ arr[:, :, ch].astype(np.float64) @ c_t
    return np.rint(out).clip(0, 255).astype(np.uint8)


def apply_exif_orientation(img: ArrayLike, orientation: int) -> ImageArray:
    """Undo EXIF tags 1-8 so the array is upright. Tags 5-8 swap height and width."""
    arr: ImageArray = _as_array(img)
    tag: int = int(orientation)
    if tag == 1:
        return arr.copy()
    if tag == 2:
        return flip_h(arr)
    if tag == 3:
        return rotate90(arr, k=2)
    if tag == 4:
        return flip_v(arr)
    if tag == 5:
        return transpose_img(arr)
    if tag == 6:
        return rotate90(arr, k=-1)
    if tag == 7:
        return transpose_img(rotate90(arr, k=2))
    if tag == 8:
        return rotate90(arr, k=1)
    raise ValueError("orientation must be an integer 1 through 8")