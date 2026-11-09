"""Image ops as linear maps: grayscale (matrix-vector), flips (permutation), resize R I C^T."""

import numpy as np

WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=float)


def _as_array(img):
    return np.array(img)


def _exchange(n):
    """n x n permutation (exchange) matrix that reverses index order."""
    return np.eye(n)[::-1]


def _apply_left_right(img, left=None, right=None):
    """Color-safe products: left @ I and/or I @ right, per channel if RGB."""
    arr = img.astype(float)
    if arr.ndim == 2:
        if left is not None:
            arr = left @ arr
        if right is not None:
            arr = arr @ right
        return arr
    if arr.ndim == 3:
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            plane = arr[:, :, c]
            if left is not None:
                plane = left @ plane
            if right is not None:
                plane = plane @ right
            out[:, :, c] = plane
        return out
    raise ValueError("expected HxW or HxWxC image")


def grayscale(img):
    """Matrix-vector product with Rec.601-style weights; round to uint8."""
    arr = _as_array(img)
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("grayscale expects HxW or HxWx3")
    gray = np.rint(arr.astype(float) @ WEIGHTS)
    return np.clip(gray, 0, 255).astype(np.uint8)


def flip_v(img):
    """Vertical flip: left-multiply by the H x H exchange matrix."""
    arr = _as_array(img)
    p = _exchange(arr.shape[0])
    out = _apply_left_right(arr, left=p)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def flip_h(img):
    """Horizontal flip: right-multiply by the W x W exchange matrix."""
    arr = _as_array(img)
    q = _exchange(arr.shape[1])
    out = _apply_left_right(arr, right=q)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def transpose_img(img):
    """Swap rows and columns (permutation of axes). Color channel stays last."""
    arr = _as_array(img)
    if arr.ndim == 2:
        return arr.T.copy()
    if arr.ndim == 3:
        return np.transpose(arr, (1, 0, 2)).copy()
    raise ValueError("expected HxW or HxWxC image")


def rotate90(img, k=1):
    """k quarter-turns counter-clockwise, matching np.rot90 with y down / row 0 at top."""
    arr = _as_array(img)
    k = int(k) % 4
    out = arr
    for _ in range(k):
        # transpose then reverse rows = 90 degrees CCW
        out = flip_v(transpose_img(out))
    return out


def crop(img, x0, y0, x1, y1):
    """Slice img[y0:y1, x0:x1] and copy so later stamps cannot mutate the source."""
    arr = _as_array(img)
    return arr[y0:y1, x0:x1].copy()


def _averaging_matrix(n_old, n_new):
    """Map length n_old to n_new by box averaging. Each row sums to 1."""
    mat = np.zeros((n_new, n_old), dtype=float)
    for i in range(n_new):
        start = i * n_old / float(n_new)
        end = (i + 1) * n_old / float(n_new)
        for j in range(n_old):
            overlap = min(end, j + 1) - max(start, j)
            if overlap > 0:
                mat[i, j] = overlap
        row_sum = mat[i].sum()
        if row_sum > 0:
            mat[i] /= row_sum
        else:
            mat[i, min(j, n_old - 1)] = 1.0
    return mat


def resize_matrices(old_h, old_w, new_h, new_w):
    """R (new_h x old_h) and C (new_w x old_w) for I' = R I C^T. Rows sum to 1."""
    r = _averaging_matrix(old_h, new_h)
    c = _averaging_matrix(old_w, new_w)
    return r, c


def resize(img, new_h, new_w):
    """Box-average resize as the matrix product R I C^T (Section 1.3)."""
    arr = _as_array(img)
    old_h, old_w = arr.shape[0], arr.shape[1]
    r, c = resize_matrices(old_h, old_w, new_h, new_w)
    c_t = c.T
    if arr.ndim == 2:
        out = r @ arr.astype(float) @ c_t
    else:
        out = np.empty((new_h, new_w, arr.shape[2]), dtype=float)
        for ch in range(arr.shape[2]):
            out[:, :, ch] = r @ arr[:, :, ch].astype(float) @ c_t
    return np.rint(out).clip(0, 255).astype(np.uint8)


def apply_exif_orientation(img, orientation):
    """Undo EXIF tags 1-8 so the array is upright. Tags 5-8 swap height and width."""
    arr = _as_array(img)
    tag = int(orientation)
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
