"""Distance-to-border weight maps, exposure gain matching, and feather blending."""

from collections.abc import Sequence
from typing import Any, Final, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- Type Aliases ---

AnyImageArray: TypeAlias = NDArray[Any]
"""Image array of arbitrary numeric dtype (`uint8`, `float32`, `float64`), shape `(H, W)` or `(H, W, C)`.

Examples:
    `np.zeros((1080, 1920, 3), dtype=np.uint8)` or `np.zeros((480, 640), dtype=np.float64)`
"""

FloatArray: TypeAlias = NDArray[np.float64]
"""64-bit float array for coordinate grids, weight maps, or blended pixel bands.

Example:
    `np.zeros((1080, 1920), dtype=np.float64)`
"""

FloatVector: TypeAlias = NDArray[np.float64]
"""1D 64-bit float array (e.g., per-image exposure gains or flattened overlap samples), shape `(N,)`.

Example:
    `np.array([1.0, 0.94, 1.03], dtype=np.float64)`
"""

FloatStack: TypeAlias = NDArray[np.float64]
"""3D stack of `N` weight maps for a horizontal tile band, shape `(N, band_height, width)`.

Example:
    `np.zeros((2, 256, 1920), dtype=np.float64)`
"""

BoolMask: TypeAlias = NDArray[np.bool_]
"""2D True/False pixel validity or coverage mask, shape `(H, W)`.

Example:
    `np.array([[True, False], [True, True]], dtype=np.bool_)`
"""

Int64Array: TypeAlias = NDArray[np.int64]
"""64-bit integer index array (e.g., image selection indices or pixel coordinates).

Example:
    `np.array([0, 0, 1, 1], dtype=np.int64)`
"""

ImageShape: TypeAlias = tuple[int, ...]
"""Image dimensions tuple: `(height, width)` or `(height, width, channels)`.

Examples:
    `(1080, 1920, 3)` for RGB or `(1080, 1920)` for grayscale
"""

# --- Constants ---

WEIGHT_EPS: Final[float] = 1e-9
"""Minimum sum of feather weights (`1e-9`) before treating a pixel as lying right on an image border."""

GAIN_EPS: Final[float] = 1e-12
"""Minimum denominator `<I_i, I_i>` (`1e-12`) allowed when solving for exposure gain `g_i`."""

TILE_PIXELS: Final[int] = 1_000_000
"""Process the output canvas in horizontal strips of ~1M pixels at a time to keep peak RAM bounded."""


def edge_weights(
    x: ArrayLike,
    y: ArrayLike,
    valid: ArrayLike,
    shape: ImageShape,
) -> FloatArray:
    """Compute distance-to-border feather weights for one warped image (M1).

    w = min distance from the source point (x, y) to the image border,
    clipped at 0. Each of the four distances is affine in the source
    coordinates, so w is piecewise affine, zero on the border, zero outside
    the valid mask, and positive inside.
    """
    # Ensure source coordinate grids and validity mask are standard NumPy arrays
    x_arr: FloatArray = np.asarray(x, dtype=np.float64)
    y_arr: FloatArray = np.asarray(y, dtype=np.float64)
    valid_mask: BoolMask = np.asarray(valid, dtype=np.bool_)

    # Extract original source image height (rows) and width (columns)
    h: int
    w: int
    h, w = shape[0], shape[1]

    # Compute distance to all 4 edges: left (x), top (y), right (w - 1 - x), bottom (h - 1 - y)
    d: FloatArray = np.minimum(
        np.minimum(x_arr, y_arr),
        np.minimum((w - 1) - x_arr, (h - 1) - y_arr),
    )

    # Clamp negative distances (outside image bounds) to 0.0
    wgt: FloatArray = np.clip(d, 0.0, None)

    # Force weights to 0.0 anywhere the pixel fell outside the valid warped mask
    wgt[~valid_mask] = 0.0
    return wgt


def _estimate_gains(
    imgs: Sequence[AnyImageArray],
    weights: Sequence[FloatArray],
    h: int,
    w: int,
) -> FloatVector:
    """Tiled pass computing g_i = <I_0, I_i> / <I_i, I_i> over pixels where
    both weights are positive (M5). Returns gains[0] = 1.
    """
    n: int = len(imgs)
    # Accumulators for dot products <I_0, I_i> (numerator) and <I_i, I_i> (denominator)
    num: FloatVector = np.zeros(n, dtype=np.float64)
    den: FloatVector = np.zeros(n, dtype=np.float64)

    # Process canvas in horizontal row tiles to avoid large temporary copies
    rows_per_tile: int = max(1, TILE_PIXELS // max(w, 1))
    for r0 in range(0, h, rows_per_tile):
        r1: int = min(h, r0 + rows_per_tile)

        # Find pixels in this band where anchor image 0 has strictly positive weight
        mask0: BoolMask = weights[0][r0:r1] > WEIGHT_EPS
        ref: FloatArray = imgs[0][r0:r1].astype(np.float64)

        # Compare each subsequent image i against anchor image 0 in their shared overlap
        for i in range(1, n):
            mask: BoolMask = mask0 & (weights[i][r0:r1] > WEIGHT_EPS)
            if not np.any(mask):
                continue
            # Flatten overlapping pixel values (across all color channels) into 1D vectors
            a: FloatVector = ref[mask].ravel()
            b: FloatVector = imgs[i][r0:r1].astype(np.float64)[mask].ravel()
            # Accumulate inner products for least-squares brightness scaling
            num[i] += float(np.dot(a, b))
            den[i] += float(np.dot(b, b))

    # Default gain is 1.0 (anchor image 0 stays unchanged)
    gains: FloatVector = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        if den[i] > GAIN_EPS:
            g: float = float(num[i] / den[i])
            # Only accept finite, non-negative brightness multipliers
            if np.isfinite(g) and g >= 0.0:
                gains[i] = g
    return gains


def _coverage(arr_i_band: AnyImageArray, valid_i_band: BoolMask | None) -> BoolMask:
    """Boolean coverage for one band: the valid mask if given, else a
    nonzero-pixel heuristic (invalid warps are exactly zero).
    """
    # Prefer the explicit boolean validity mask from warp_image when available
    if valid_i_band is not None:
        return valid_i_band
    # Fallback for RGB/multi-channel bands: pixel is covered if any channel is non-zero
    if arr_i_band.ndim > 2:
        return np.any(arr_i_band != 0, axis=tuple(range(2, arr_i_band.ndim)))
    # Fallback for 2D grayscale bands: pixel is covered if intensity is non-zero
    return arr_i_band != 0


def feather_blend(
    imgs: Sequence[ArrayLike],
    weights: Sequence[ArrayLike],
    exposure_gain: bool = False,
    valid: Sequence[ArrayLike] | None = None,
) -> AnyImageArray:
    """Blend warped images into one canvas (M2, M3, M4).

    imgs: list of same-shaped arrays (h, w) or (h, w, c); invalid regions
    are expected to be zero (as produced by warp_image).
    weights: list of same-shaped (h, w) maps from edge_weights.
    exposure_gain: if True, rescale images 1..N-1 onto image 0 first (M5),
    estimated in a separate tiled pass.
    valid: optional list of (h, w) boolean coverage masks (e.g. the masks
    returned by warp_image). When given, the all-zero-weight fallback picks
    the first *valid* image, so valid black pixels are not mistaken for
    uncovered areas. Without it, the fallback uses a nonzero-pixel
    heuristic.

    Output = sum_i alpha_i * I_i with alpha_i = w_i / sum_j w_j, a convex
    combination. Where every weight is ~0 (image borders), falls back to a
    hard cut. The canvas is processed in row tiles so temporary memory stays
    bounded, and the output is written directly in the result dtype.
    """
    # Convert input sequences to NumPy arrays and validate matching counts/shapes
    img_list: list[AnyImageArray] = [np.asarray(im) for im in imgs]
    weight_list: list[FloatArray] = [np.asarray(w, dtype=np.float64) for w in weights]
    if len(img_list) == 0 or len(img_list) != len(weight_list):
        raise ValueError("imgs and weights must be non-empty and equal length")

    h: int
    w: int
    h, w = img_list[0].shape[:2]
    for im, wt in zip(img_list, weight_list):
        if im.shape[:2] != (h, w) or wt.shape != (h, w):
            raise ValueError("all imgs and weights must share the canvas shape")

    # Validate optional boolean coverage masks if provided
    valids: list[BoolMask] | None = None
    if valid is not None:
        valids = [np.asarray(v, dtype=np.bool_) for v in valid]
        if len(valids) != len(img_list) or any(v.shape != (h, w) for v in valids):
            raise ValueError("valid masks must match imgs")

    n: int = len(img_list)
    channel_shape: tuple[int, ...] = img_list[0].shape[2:]
    dtype: np.dtype[Any] = img_list[0].dtype
    same_dtype: bool = all(im.dtype == dtype for im in img_list)
    gain_applied: bool = exposure_gain and n > 1

    # Choose output pixel dtype: preserve integer/float input dtype when safe, else promote to float64
    out_dtype: np.dtype[Any]
    if same_dtype and np.issubdtype(dtype, np.integer):
        out_dtype = dtype
    elif same_dtype and np.issubdtype(dtype, np.floating) and not gain_applied:
        # Without gain, a convex combination cannot leave the input range.
        # With gain, values can exceed it (e.g. float16 -> inf), so float64.
        out_dtype = dtype
    else:
        out_dtype = np.dtype(np.float64)

    # Optionally compute per-image brightness multipliers to match image 0's exposure
    gains: FloatVector = _estimate_gains(img_list, weight_list, h, w) if gain_applied else np.ones(n, dtype=np.float64)

    # Pre-allocate the final output canvas
    out: AnyImageArray = np.zeros((h, w) + channel_shape, dtype=out_dtype)

    # Blend the canvas in horizontal row strips ("tiles") to keep peak memory usage low
    rows_per_tile: int = max(1, TILE_PIXELS // max(w, 1))
    for r0 in range(0, h, rows_per_tile):
        r1: int = min(h, r0 + rows_per_tile)
        bh: int = r1 - r0
        bw: int = w

        # Stack weights for this band into shape (N, bh, bw) and normalize so sum_i alpha_i == 1
        Wb: FloatStack = np.clip(
            np.stack([wt[r0:r1] for wt in weight_list]),
            0.0,
            None,
        )
        total: FloatArray = Wb.sum(axis=0)
        safe: BoolMask = total > WEIGHT_EPS
        denom: FloatArray = np.where(safe, total, 1.0)
        alpha: FloatStack = Wb / denom[None, :, :]
        alpha[:, ~safe] = 0.0

        # Accumulate weighted pixel values: band = sum_i (gain_i * I_i * alpha_i)
        band: FloatArray = np.zeros((bh, bw) + channel_shape, dtype=np.float64)
        for i in range(n):
            ai: FloatArray = img_list[i][r0:r1].astype(np.float64)
            if gains[i] != 1.0:
                ai = ai * gains[i]
            if channel_shape:
                # Broadcast 2D alpha (bh, bw) across color channels (bh, bw, 1)
                band += ai * alpha[i][..., None]
            else:
                band += ai * alpha[i]

        # Border fallback: where all weights are ~0 (~safe), pick the first covering image (hard cut)
        if np.any(~safe):
            first: Int64Array = np.full((bh, bw), -1, dtype=np.int64)
            for i in range(n):
                cov: BoolMask = _coverage(
                    img_list[i][r0:r1],
                    None if valids is None else valids[i][r0:r1],
                )
                first[(first < 0) & cov & ~safe] = i
            ys: Int64Array
            xs: Int64Array
            ys, xs = np.nonzero(~safe)
            pick: Int64Array = first[ys, xs]
            ok: BoolMask = pick >= 0
            if np.any(ok):
                for i in np.unique(pick[ok]):
                    m: BoolMask = pick == i
                    src: FloatArray = img_list[i][r0:r1][ys[m], xs[m]].astype(np.float64)
                    if gains[i] != 1.0:
                        src = src * gains[i]
                    band[ys[m], xs[m]] = src

        # Round and clamp to valid range (e.g. 0..255 for uint8) if returning an integer image
        if np.issubdtype(out_dtype, np.integer):
            bounds: np.iinfo[Any] = np.iinfo(out_dtype)
            band = np.rint(np.clip(band, bounds.min, bounds.max))

        # Write the finished tile strip into the output canvas
        out[r0:r1] = band.astype(out_dtype, copy=False)

    return out