import numpy as np
import linalg
from homography import HomographyError

W_EPS = 1e-12
DET_EPS = 1e-12
MAX_CANVAS_SIDE = 20000
MAX_CANVAS_PIXELS = 100_000_000
TILE_PIXELS = 1_000_000


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
    """Raise HomographyError if H cannot be warped safely.

    Rejects non-finite/non-3x3 input, singular matrices (warp_image needs
    H^-1), and any corner whose projective denominator w is zero or near
    zero. w is linear in (x, y), so its extremes over the image rectangle
    occur at the corners: a zero or sign change among them means the
    horizon touches or crosses the image, making the perspective division
    unsafe (M8).
    """
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise HomographyError("H must be a finite 3x3 matrix")
    if abs(linalg.det(H)) <= DET_EPS:
        raise HomographyError("H is singular")
    _, w = _map_points(H, _corners(shape))
    if np.any(np.abs(w) <= W_EPS):
        raise HomographyError("homography horizon touches the image boundary")
    if np.any(w > 0) and np.any(w < 0):
        raise HomographyError("homography horizon crosses the image")


def canvas_bounds(H_list, shapes):
    """Union bounds of all warped images.

    Every H is validated with check_h first: an H that cannot be warped
    safely has no meaningful canvas. Returns ((canvas_h, canvas_w), offset)
    where offset is the (x, y) world coordinate of canvas pixel (0, 0), i.e.
    canvas = world - offset. Raises HomographyError if the union exceeds
    MAX_CANVAS_SIDE / MAX_CANVAS_PIXELS rather than returning cropped
    bounds.
    """
    if len(H_list) != len(shapes):
        raise HomographyError("H_list and shapes must have the same length")

    mins = np.full(2, np.inf)
    maxs = np.full(2, -np.inf)
    for H, shape in zip(H_list, shapes):
        check_h(H, shape)
        pts, _ = _map_points(np.asarray(H, dtype=float), _corners(shape))
        pts = np.nan_to_num(pts, nan=0.0, posinf=1e15, neginf=-1e15)
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    offset = mins.copy()
    canvas_w = max(1, int(np.ceil(maxs[0] - mins[0])))
    canvas_h = max(1, int(np.ceil(maxs[1] - mins[1])))

    if (canvas_w > MAX_CANVAS_SIDE or canvas_h > MAX_CANVAS_SIDE
            or canvas_w * canvas_h > MAX_CANVAS_PIXELS):
        raise HomographyError(
            f"canvas size {(canvas_h, canvas_w)} exceeds limit "
            f"({MAX_CANVAS_SIDE} per side, {MAX_CANVAS_PIXELS} pixels)")

    return (canvas_h, canvas_w), offset


def warp_image(img, H, canvas_shape, offset):
    """Inverse-warp img onto the shared canvas with nearest-neighbor sampling.

    Canvas pixel p corresponds to world point p + offset, which maps back to
    the source through H^-1 (M8: Hc^-1 = H^-1 T^-1). Returns
    (out, valid, src_x, src_y) where src_x/src_y are the source coordinates.
    The canvas is processed in row tiles so temporaries stay bounded.
    """
    img = np.asarray(img)
    H = np.asarray(H, dtype=float)
    cH, cW = int(canvas_shape[0]), int(canvas_shape[1])
    ox, oy = float(offset[0]), float(offset[1])
    h_src, w_src = img.shape[0], img.shape[1]

    try:
        H_inv = linalg.inverse(H)
    except Exception as e:
        raise HomographyError("H is singular") from e

    out = np.zeros((cH, cW) + img.shape[2:], dtype=img.dtype)
    valid = np.zeros((cH, cW), dtype=bool)
    src_x = np.zeros((cH, cW), dtype=float)
    src_y = np.zeros((cH, cW), dtype=float)

    rows_per_tile = max(1, TILE_PIXELS // max(cW, 1))
    for r0 in range(0, cH, rows_per_tile):
        r1 = min(cH, r0 + rows_per_tile)
        gx, gy = np.meshgrid(np.arange(cW, dtype=float) + ox,
                             np.arange(r0, r1, dtype=float) + oy)
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
        band_valid = (good_w
                      & (sx > -0.5) & (sx < w_src - 0.5)
                      & (sy > -0.5) & (sy < h_src - 0.5))

        band_h = r1 - r0
        valid[r0:r1] = band_valid.reshape(band_h, cW)
        src_x[r0:r1] = sx.reshape(band_h, cW)
        src_y[r0:r1] = sy.reshape(band_h, cW)
        if np.any(band_valid):
            out[r0:r1][band_valid.reshape(band_h, cW)] = img[ny[band_valid], nx[band_valid]]

    return out, valid, src_x, src_y


# Identity Stubs
def edge_weights(x, y, valid, shape): return np.zeros_like(x)
def feather_blend(imgs, weights): return imgs[0]
def ransac_homography(src, dst): return np.eye(3), np.ones(len(src), dtype=bool)
