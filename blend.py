import numpy as np

WEIGHT_EPS = 1e-9
GAIN_EPS = 1e-12
TILE_PIXELS = 1_000_000


def edge_weights(x, y, valid, shape):
    """Feather weight for one warped image (M1).

    w = min distance from the source point (x, y) to the image border,
    clipped at 0. Each of the four distances is affine in the source
    coordinates, so w is piecewise affine, zero on the border, zero outside
    the valid mask, and positive inside.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    h, w = shape[0], shape[1]
    d = np.minimum(np.minimum(x, y), np.minimum((w - 1) - x, (h - 1) - y))
    wgt = np.clip(d, 0.0, None)
    wgt[~valid] = 0.0
    return wgt


def _estimate_gains(imgs, weights, h, w):
    """Tiled pass computing g_i = <I_0, I_i> / <I_i, I_i> over pixels where
    both weights are positive (M5). Returns gains[0] = 1."""
    n = len(imgs)
    num = np.zeros(n)
    den = np.zeros(n)
    rows_per_tile = max(1, TILE_PIXELS // max(w, 1))
    for r0 in range(0, h, rows_per_tile):
        r1 = min(h, r0 + rows_per_tile)
        mask0 = weights[0][r0:r1] > WEIGHT_EPS
        ref = imgs[0][r0:r1].astype(np.float64)
        for i in range(1, n):
            mask = mask0 & (weights[i][r0:r1] > WEIGHT_EPS)
            if not np.any(mask):
                continue
            a = ref[mask].ravel()
            b = imgs[i][r0:r1].astype(np.float64)[mask].ravel()
            num[i] += float(np.dot(a, b))
            den[i] += float(np.dot(b, b))
    gains = np.ones(n)
    for i in range(1, n):
        if den[i] > GAIN_EPS:
            g = num[i] / den[i]
            if np.isfinite(g) and g >= 0.0:
                gains[i] = g
    return gains


def _coverage(arr_i_band, valid_i_band):
    """Boolean coverage for one band: the valid mask if given, else a
    nonzero-pixel heuristic (invalid warps are exactly zero)."""
    if valid_i_band is not None:
        return valid_i_band
    if arr_i_band.ndim > 2:
        return np.any(arr_i_band != 0, axis=tuple(range(2, arr_i_band.ndim)))
    return arr_i_band != 0


def feather_blend(imgs, weights, exposure_gain=False, valid=None):
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
    imgs = [np.asarray(im) for im in imgs]
    weights = [np.asarray(w) for w in weights]
    if len(imgs) == 0 or len(imgs) != len(weights):
        raise ValueError("imgs and weights must be non-empty and equal length")

    h, w = imgs[0].shape[:2]
    for im, wt in zip(imgs, weights):
        if im.shape[:2] != (h, w) or wt.shape != (h, w):
            raise ValueError("all imgs and weights must share the canvas shape")

    valids = None
    if valid is not None:
        valids = [np.asarray(v, dtype=bool) for v in valid]
        if len(valids) != len(imgs) or any(v.shape != (h, w) for v in valids):
            raise ValueError("valid masks must match imgs")

    n = len(imgs)
    channel_shape = imgs[0].shape[2:]
    dtype = imgs[0].dtype
    same_dtype = all(im.dtype == dtype for im in imgs)
    gain_applied = exposure_gain and n > 1
    if same_dtype and np.issubdtype(dtype, np.integer):
        out_dtype = dtype
    elif same_dtype and np.issubdtype(dtype, np.floating) and not gain_applied:
        # Without gain, a convex combination cannot leave the input range.
        # With gain, values can exceed it (e.g. float16 -> inf), so float64.
        out_dtype = dtype
    else:
        out_dtype = np.float64

    gains = _estimate_gains(imgs, weights, h, w) if gain_applied else np.ones(n)

    out = np.zeros((h, w) + channel_shape, dtype=out_dtype)
    rows_per_tile = max(1, TILE_PIXELS // max(w, 1))
    for r0 in range(0, h, rows_per_tile):
        r1 = min(h, r0 + rows_per_tile)
        bh, bw = r1 - r0, w

        Wb = np.clip(np.stack([np.asarray(wt[r0:r1], dtype=np.float64)
                               for wt in weights]), 0.0, None)  # (n, bh, bw)
        total = Wb.sum(axis=0)
        safe = total > WEIGHT_EPS
        denom = np.where(safe, total, 1.0)
        alpha = Wb / denom[None, :, :]
        alpha[:, ~safe] = 0.0

        band = np.zeros((bh, bw) + channel_shape, dtype=np.float64)
        for i in range(n):
            ai = imgs[i][r0:r1].astype(np.float64)
            if gains[i] != 1.0:
                ai = ai * gains[i]
            if channel_shape:
                band += ai * alpha[i][..., None]
            else:
                band += ai * alpha[i]

        if np.any(~safe):
            first = np.full((bh, bw), -1, dtype=np.int64)
            for i in range(n):
                cov = _coverage(imgs[i][r0:r1],
                                None if valids is None else valids[i][r0:r1])
                first[(first < 0) & cov & ~safe] = i
            ys, xs = np.nonzero(~safe)
            pick = first[ys, xs]
            ok = pick >= 0
            if np.any(ok):
                for i in np.unique(pick[ok]):
                    m = pick == i
                    src = imgs[i][r0:r1][ys[m], xs[m]].astype(np.float64)
                    if gains[i] != 1.0:
                        src = src * gains[i]
                    band[ys[m], xs[m]] = src

        if np.issubdtype(out_dtype, np.integer):
            bounds = np.iinfo(out_dtype)
            band = np.rint(np.clip(band, bounds.min, bounds.max))
        out[r0:r1] = band.astype(out_dtype, copy=False)

    return out
