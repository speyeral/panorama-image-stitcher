import numpy as np

WEIGHT_EPS = 1e-9
GAIN_EPS = 1e-12


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


def _apply_gain(arr, W):
    """Scale each image (index >= 1) by the least-squares gain that maps it
    onto image 0 over the region where both weights are positive (M5:
    g = <I_0, I_i> / <I_i, I_i>)."""
    ref = arr[0]
    for i in range(1, arr.shape[0]):
        mask = (W[0] > WEIGHT_EPS) & (W[i] > WEIGHT_EPS)
        if not np.any(mask):
            continue
        a = ref[mask].ravel()
        b = arr[i][mask].ravel()
        denom = float(np.dot(b, b))
        if denom <= GAIN_EPS:
            continue
        g = float(np.dot(a, b)) / denom
        if np.isfinite(g) and g > 0.0:
            arr[i] = arr[i] * g
    return arr


def feather_blend(imgs, weights, exposure_gain=False):
    """Blend warped images into one canvas (M2, M3, M4).

    imgs: list of same-shaped arrays (h, w) or (h, w, c); invalid regions
    are expected to be zero (as produced by warp_image).
    weights: list of same-shaped (h, w) maps from edge_weights.
    exposure_gain: if True, rescale images 1..N-1 onto image 0 first (M5).

    Output = sum_i alpha_i * I_i with alpha_i = w_i / sum_j w_j, a convex
    combination. Where every weight is ~0 (image borders), falls back to a
    hard cut: the first image with a nonzero pixel at that location.
    """
    imgs = [np.asarray(im) for im in imgs]
    weights = [np.asarray(w, dtype=np.float64) for w in weights]
    if len(imgs) == 0 or len(imgs) != len(weights):
        raise ValueError("imgs and weights must be non-empty and equal length")

    h, w = imgs[0].shape[:2]
    for im, wt in zip(imgs, weights):
        if im.shape[:2] != (h, w) or wt.shape != (h, w):
            raise ValueError("all imgs and weights must share the canvas shape")

    arr = np.stack([im.astype(np.float64) for im in imgs])  # (N, h, w[, c])
    W = np.clip(np.stack(weights), 0.0, None)               # (N, h, w)
    n = arr.shape[0]
    extra = arr.ndim - 3

    if exposure_gain and n > 1:
        arr = _apply_gain(arr, W)

    total = W.sum(axis=0)
    safe = total > WEIGHT_EPS
    denom = np.where(safe, total, 1.0)
    alpha = W / denom[None, :, :]
    alpha[:, ~safe] = 0.0
    out = np.sum(arr * alpha.reshape((n, h, w) + (1,) * extra), axis=0)

    if np.any(~safe):
        # Hard-cut fallback: invalid warps are exactly zero, so the first
        # image with a nonzero pixel is the covering image.
        if extra:
            nz = np.any(arr != 0, axis=tuple(range(3, arr.ndim)))  # (N, h, w)
        else:
            nz = arr != 0
        first = np.full((h, w), -1, dtype=np.int64)
        for i in range(n):
            first[(first < 0) & nz[i]] = i
        ys, xs = np.nonzero(~safe)
        pick = first[ys, xs]
        ok = pick >= 0
        if np.any(ok):
            out[ys[ok], xs[ok]] = arr[pick[ok], ys[ok], xs[ok]]

    dtype = imgs[0].dtype
    if all(im.dtype == dtype for im in imgs) and np.issubdtype(dtype, np.integer):
        return np.rint(out).astype(dtype)
    return out
