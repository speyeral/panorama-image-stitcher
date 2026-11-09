import numpy as np
import linalg
from homography import HomographyError

W_EPS = 1e-12
MAX_CANVAS_SIDE = 20000
MAX_CANVAS_PIXELS = 100_000_000


def _corners(shape):
    """Image extent corners (x, y) for a shape (h, w[, c])."""
    h, w = shape[0], shape[1]
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])


def _map_points(H, pts):
    """Apply homography H to (N, 2) points; returns (N, 2) Euclidean + (N,) w."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out_h = pts_h @ H.T
    w = out_h[:, 2]
    w_safe = np.where(np.abs(w) > W_EPS, w, 1.0)
    return out_h[:, :2] / w_safe[:, None], w


def check_h(H, shape):
    """Raise HomographyError if H is degenerate or its horizon crosses the image.

    w is linear in (x, y), so its extremes over the image rectangle occur at the
    corners. Same-sign corner w values mean w never vanishes inside the image,
    so the perspective division in the inverse warp is safe (M8).
    """
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise HomographyError("H must be a finite 3x3 matrix")
    _, w = _map_points(H, _corners(shape))
    pos = np.any(w > W_EPS)
    neg = np.any(w < -W_EPS)
    if pos and neg:
        raise HomographyError("homography horizon crosses the image")
    if not pos and not neg:
        raise HomographyError("degenerate homography: w vanishes on all corners")


def canvas_bounds(H_list, shapes):
    """Union bounds of all warped images.

    Every H is validated with check_h first: an H that cannot be warped
    safely has no meaningful canvas. Returns ((canvas_h, canvas_w), offset)
    where offset is the (x, y) world coordinate of canvas pixel (0, 0), i.e.
    canvas = world - offset. The canvas size is capped by MAX_CANVAS_SIDE /
    MAX_CANVAS_PIXELS.
    """
    if len(H_list) != len(shapes):
        raise HomographyError("H_list and shapes must have the same length")

    mins = np.full(2, np.inf)
    maxs = np.full(2, -np.inf)
    for H, shape in zip(H_list, shapes):
        check_h(H, shape)
        pts, w = _map_points(np.asarray(H, dtype=float), _corners(shape))
        pts = np.nan_to_num(pts, nan=0.0, posinf=1e15, neginf=-1e15)
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    offset = mins.copy()
    canvas_w = max(1, int(np.ceil(maxs[0] - mins[0])))
    canvas_h = max(1, int(np.ceil(maxs[1] - mins[1])))

    over_side = max(canvas_w / MAX_CANVAS_SIDE, canvas_h / MAX_CANVAS_SIDE)
    over_area = (canvas_w * canvas_h) / MAX_CANVAS_PIXELS
    if over_side > 1.0 or over_area > 1.0:
        scale = 1.0 / max(over_side, np.sqrt(over_area))
        canvas_w = max(1, int(np.floor(canvas_w * scale)))
        canvas_h = max(1, int(np.floor(canvas_h * scale)))

    return (canvas_h, canvas_w), offset


def warp_image(img, H, canvas_shape, offset):
    """Inverse-warp img onto the shared canvas with nearest-neighbor sampling.

    Canvas pixel p corresponds to world point p + offset, which maps back to
    the source through H^-1 (M8: Hc^-1 = H^-1 T^-1). Returns
    (out, valid, src_x, src_y) where src_x/src_y are the source coordinates.
    """
    img = np.asarray(img)
    H = np.asarray(H, dtype=float)
    cH, cW = int(canvas_shape[0]), int(canvas_shape[1])
    ox, oy = float(offset[0]), float(offset[1])
    h_src, w_src = img.shape[0], img.shape[1]

    H_inv = linalg.inverse(H)

    gx, gy = np.meshgrid(np.arange(cW, dtype=float) + ox,
                         np.arange(cH, dtype=float) + oy)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1)
    sp = pts @ H_inv.T

    w = sp[:, 2]
    good_w = np.isfinite(w) & (np.abs(w) > W_EPS)
    w_safe = np.where(good_w, w, 1.0)
    sx = sp[:, 0] / w_safe
    sy = sp[:, 1] / w_safe
    sx = np.clip(np.nan_to_num(sx, nan=0.0, posinf=1e15, neginf=-1e15), -1e15, 1e15)
    sy = np.clip(np.nan_to_num(sy, nan=0.0, posinf=1e15, neginf=-1e15), -1e15, 1e15)

    nx = np.floor(sx + 0.5).astype(np.int64)
    ny = np.floor(sy + 0.5).astype(np.int64)
    valid = (good_w
             & (sx > -0.5) & (sx < w_src - 0.5)
             & (sy > -0.5) & (sy < h_src - 0.5))

    out = np.zeros((cH, cW) + img.shape[2:], dtype=img.dtype)
    if np.any(valid):
        out[valid.reshape(cH, cW)] = img[ny[valid], nx[valid]]

    src_x = sx.reshape(cH, cW)
    src_y = sy.reshape(cH, cW)
    return out, valid.reshape(cH, cW), src_x, src_y


# Identity Stubs
def edge_weights(x, y, valid, shape): return np.zeros_like(x)
def feather_blend(imgs, weights): return imgs[0]
def ransac_homography(src, dst): return np.eye(3), np.ones(len(src), dtype=bool)
