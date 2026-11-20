from collections.abc import Sequence
from typing import Final, TypeAlias, TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
import linalg
from homography import HomographyError
from blend import edge_weights, feather_blend  # noqa: F401 (re-exported for convenience; linter ignore unused)

# --- Type Aliases & Generics (for code clarity and static type checking) ---

DTypeT = TypeVar("DTypeT", bound=np.generic)
"""Generic NumPy scalar type variable that preserves input image pixel dtype.

Examples:
    `np.uint8` (standard 8-bit image) or `np.float64` (HDR/normalized image)
"""

GenericImage: TypeAlias = NDArray[DTypeT]
"""2D grayscale `(H, W)` or 3D color `(H, W, C)` image array of generic dtype.

Examples:
    `np.zeros((1080, 1920, 3), dtype=np.uint8)` or `np.zeros((480, 640), dtype=np.float64)`
"""

BoolMask: TypeAlias = NDArray[np.bool_]
"""True/False pixel validity mask, shape `(H, W)`.

Example:
    `np.array([[True, False], [True, True]], dtype=np.bool_)`
"""

Int64Vector: TypeAlias = NDArray[np.int64]
"""1D array of integer pixel indices, shape `(N,)`.

Example:
    `np.array([0, 15, 42, 108], dtype=np.int64)`
"""

FloatArray: TypeAlias = NDArray[np.float64]
"""General 64-bit float array of arbitrary dimensions (commonly 2D coordinate grids `(H, W)`).

Example:
    `np.zeros((100, 100), dtype=np.float64)`
"""

FloatVector: TypeAlias = NDArray[np.float64]
"""1D 64-bit float array, shape `(N,)`.

Example:
    `np.array([1.0, 0.98, 1.02], dtype=np.float64)`
"""

FloatMatrix: TypeAlias = NDArray[np.float64]
"""2D 64-bit float matrix, shape `(M, N)`.

Example:
    `np.array([[0.0, 0.0, 1.0], [100.0, 50.0, 1.0]], dtype=np.float64)`
"""

PointList2D: TypeAlias = NDArray[np.float64]
"""List of N 2D coordinates `[[x1, y1], ...]`, shape `(N, 2)`.

Example:
    `np.array([[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]], dtype=np.float64)`
"""

HomographyMatrix: TypeAlias = NDArray[np.float64]
"""3x3 projective transformation matrix, shape `(3, 3)`.

Example:
    `np.eye(3, dtype=np.float64)`
"""

ImageShape: TypeAlias = tuple[int, ...]
"""Image dimensions tuple: `(height, width)` or `(height, width, channels)`.

Examples:
    `(1080, 1920, 3)` for RGB or `(1080, 1920)` for grayscale
"""

CanvasShape: TypeAlias = tuple[int, int]
"""Output canvas dimensions: `(canvas_height, canvas_width)`.

Example:
    `(1200, 2850)`
"""

Offset2D: TypeAlias = Sequence[float] | NDArray[np.floating]
"""2D translation `(offset_x, offset_y)` shifting world coordinates so top-left aligns with canvas `(0, 0)`.

Examples:
    `(-145.5, -20.0)` or `np.array([-145.5, -20.0], dtype=np.float64)`
"""


# --- Safety & Memory Constants ---

W_EPS: Final[float] = 1e-12
"""Minimum allowed `|w|` before perspective division (`x/w`, `y/w`) to prevent divide-by-zero."""

MAX_CANVAS_SIDE: Final[int] = 20000
"""Guardrail: maximum allowed width or height of the output canvas in pixels (`20,000 px`)."""

MAX_CANVAS_PIXELS: Final[int] = 100_000_000
"""Guardrail: maximum total output pixel count (`100 megapixels`) to prevent out-of-memory crashes."""

TILE_PIXELS: Final[int] = 1_000_000
"""Process the output canvas in horizontal strips of ~1M pixels at a time to keep peak RAM low."""


def _corners(shape: ImageShape) -> PointList2D:
    """Return the 4 outer border corners (x, y) of an image in clockwise order: TL, TR, BR, BL.

    Note: NumPy image shapes are (height, width[, channels]), so shape[0] is y-height and shape[1] is x-width.
    """
    h: int = shape[0]
    w: int = shape[1]
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)


def _map_points(H: HomographyMatrix, pts: PointList2D) -> tuple[PointList2D, FloatVector]:
    """Forward-project 2D points (x, y) through homography H and perform perspective division by w.

    Math: [x'w', y'w', w']^T = H @ [x, y, 1]^T  -->  (x', y') = (x'w' / w', y'w' / w').
    Because points are stored as rows (N, 2), we compute pts_h @ H.T (equivalent to (H @ pts_h.T).T).
    Returns both the 2D Euclidean coordinates and the raw denominators 'w' for horizon safety checks.
    """
    # Convert (N, 2) points into homogeneous (N, 3) points [x, y, 1]
    pts_h: FloatMatrix = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)

    # Apply 3x3 homography H to all N points simultaneously
    out_h: FloatMatrix = pts_h @ H.T

    # Extract the 3rd homogeneous coordinate w (the projective scaling factor)
    w: FloatVector = out_h[:, 2]

    # Guard against division by zero if w is near 0; replace unsafe w with 1.0 temporarily
    w_safe: FloatVector = np.where(np.abs(w) > W_EPS, w, 1.0)

    # Perspective division (homogeneous normalization): divide x'w' and y'w' by w
    return out_h[:, :2] / w_safe[:, None], w


def check_h(H: ArrayLike, shape: ImageShape) -> None:
    """Validate that homography H is invertible and safe to warp across the entire image rectangle.

    Checks three failure modes:
    1. Non-3x3 shape or NaN/Infinity entries.
    2. Singular matrix (H^-1 does not exist, which makes backward mapping impossible).
    3. Vanishing horizon crossing the image: w = h31*x + h32*y + h33 is linear in (x, y), so its
       min and max across the whole photo must occur at the 4 corners. If w == 0 at a corner or flips
       sign (+ to -) between corners, some pixels inside the photo project to infinity.
    """
    H_arr: HomographyMatrix = np.asarray(H, dtype=np.float64)
    if H_arr.shape != (3, 3) or not np.all(np.isfinite(H_arr)):
        raise HomographyError("H must be a finite 3x3 matrix")

    # Test invertibility directly by attempting to compute H^-1 (needed later by warp_image)
    H_inv: HomographyMatrix
    try:
        H_inv = linalg.inverse(H_arr)
    except Exception as e:
        raise HomographyError("H is singular") from e
    if not np.all(np.isfinite(H_inv)):
        raise HomographyError("H is numerically singular")

    # Project the 4 image corners forward and inspect their projective denominators w
    w: FloatVector
    _, w = _map_points(H_arr, _corners(shape))

    # If any corner has w == 0, the horizon line touches the image boundary
    if np.any(np.abs(w) <= W_EPS):
        raise HomographyError("homography horizon touches the image boundary")

    # If some corners have w > 0 and others have w < 0, the horizon line cuts right through the photo
    if np.any(w > 0) and np.any(w < 0):
        raise HomographyError("homography horizon crosses the image")


def canvas_bounds(
    H_list: Sequence[ArrayLike],
    shapes: Sequence[ImageShape],
) -> tuple[CanvasShape, FloatVector]:
    """Calculate the tightest rectangular canvas that fits all warped images without cropping.

    How it works:
    - Projects the 4 outer border corners of every image through its respective H.
    - Finds the global minimum and maximum (x, y) coordinates across all projected corners.
    - Because warping can push corners into negative coordinates (left of x=0 or above y=0),
      we save 'offset = (min_x, min_y)' so pixel (0, 0) on our canvas corresponds to (min_x, min_y).
    """
    if len(H_list) != len(shapes):
        raise HomographyError("H_list and shapes must have the same length")

    # Track global bounding box across all images: start at +inf for mins and -inf for maxs
    mins: FloatVector = np.full(2, np.inf, dtype=np.float64)
    maxs: FloatVector = np.full(2, -np.inf, dtype=np.float64)

    for H, shape in zip(H_list, shapes):
        # Validate H before trusting its projected corner positions
        check_h(H, shape)

        # Forward-map the 4 outer corners of this image into world/panorama coordinates
        pts: PointList2D
        pts, _ = _map_points(np.asarray(H, dtype=np.float64), _corners(shape))
        pts = np.nan_to_num(pts, nan=0.0, posinf=1e15, neginf=-1e15)

        # Expand the global bounding box to enclose these 4 warped corners
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    # 'offset' is the top-left world coordinate that maps to canvas pixel (0, 0)
    offset: FloatVector = mins.copy()

    # Total canvas width and height in integer pixels (rounded up with ceil so nothing clips)
    canvas_w: int = max(1, int(np.ceil(maxs[0] - mins[0])))
    canvas_h: int = max(1, int(np.ceil(maxs[1] - mins[1])))

    # Prevent extreme perspective warps from allocating gigabytes of RAM
    if (
        canvas_w > MAX_CANVAS_SIDE
        or canvas_h > MAX_CANVAS_SIDE
        or canvas_w * canvas_h > MAX_CANVAS_PIXELS
    ):
        raise HomographyError(
            f"canvas size {(canvas_h, canvas_w)} exceeds limit "
            f"({MAX_CANVAS_SIDE} per side, {MAX_CANVAS_PIXELS} pixels)"
        )

    return (canvas_h, canvas_w), offset


def warp_image(
    img: GenericImage[DTypeT],
    H: ArrayLike,
    canvas_shape: CanvasShape,
    offset: Offset2D,
) -> tuple[GenericImage[DTypeT], BoolMask, FloatArray, FloatArray]:
    """Inverse-warp (backward-map) a source image onto the destination canvas using Nearest-Neighbor.

    Why Inverse Mapping (H^-1)?
    If we pushed source pixels forward via H, stretching an image would leave empty black holes.
    Instead, we visit EVERY integer pixel (u, v) on the destination canvas, shift by 'offset' back
    into world coordinates, multiply by H^-1, and divide by w to look up where in 'img' it came from.

    Returns:
    - out: Warped image array of shape (cH, cW[, C])
    - valid: Boolean mask (cH, cW) indicating which canvas pixels landed inside 'img' bounds
    - src_x, src_y: Exact floating-point source coordinates (used later by blend.edge_weights)
    """
    img_arr: GenericImage[DTypeT] = np.asarray(img)
    H_arr: HomographyMatrix = np.asarray(H, dtype=np.float64)
    cH: int = int(canvas_shape[0])      # Canvas height (rows)
    cW: int = int(canvas_shape[1])      # Canvas width (columns)
    ox: float = float(offset[0])        # X shift from canvas (0, 0) to world space
    oy: float = float(offset[1])        # Y shift from canvas (0, 0) to world space
    h_src: int = img_arr.shape[0]       # Source image height
    w_src: int = img_arr.shape[1]       # Source image width

    # Compute the inverse homography H^-1 (maps destination canvas coords -> source image coords)
    H_inv: HomographyMatrix
    try:
        H_inv = linalg.inverse(H_arr)
    except Exception as e:
        raise HomographyError("H is singular") from e

    # Pre-allocate blank output buffers for the full canvas
    out: GenericImage[DTypeT] = np.zeros((cH, cW) + img_arr.shape[2:], dtype=img_arr.dtype)
    valid: BoolMask = np.zeros((cH, cW), dtype=np.bool_)
    src_x: FloatArray = np.zeros((cH, cW), dtype=np.float64)
    src_y: FloatArray = np.zeros((cH, cW), dtype=np.float64)

    # Process the canvas in horizontal row strips ("tiles") so large panoramas don't spike RAM
    rows_per_tile: int = max(1, TILE_PIXELS // max(cW, 1))
    for r0 in range(0, cH, rows_per_tile):
        r1: int = min(cH, r0 + rows_per_tile)

        # 1. Build a 2D coordinate grid of every destination pixel in this horizontal strip,
        #    adding (ox, oy) to convert canvas pixel indices into world coordinates
        gx: FloatArray
        gy: FloatArray
        gx, gy = np.meshgrid(
            np.arange(cW, dtype=np.float64) + ox,
            np.arange(r0, r1, dtype=np.float64) + oy,
        )

        # 2. Flatten the 2D grid into a list of homogeneous 3D row vectors [x_world, y_world, 1]
        pts: FloatMatrix = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, dtype=np.float64)], axis=1)

        # 3. Backward-map all destination points into source homogeneous space: sp = pts @ H_inv^T
        sp: FloatMatrix = pts @ H_inv.T

        # 4. Homogeneous normalization: divide by the 3rd coordinate w to get floating-point (sx, sy)
        w: FloatVector = sp[:, 2]
        good_w: BoolMask = np.isfinite(w) & (np.abs(w) > W_EPS)
        w_safe: FloatVector = np.where(good_w, w, 1.0)
        sx: FloatVector = sp[:, 0] / w_safe
        sy: FloatVector = sp[:, 1] / w_safe

        # Clamp extreme values to avoid integer overflow in the next rounding step
        sx = np.clip(np.nan_to_num(sx, nan=0.0, posinf=1e15, neginf=-1e15), -1e15, 1e15)
        sy = np.clip(np.nan_to_num(sy, nan=0.0, posinf=1e15, neginf=-1e15), -1e15, 1e15)

        # 5. Nearest-Neighbor sampling: round floating-point (sx, sy) to the closest integer pixel index
        nx: Int64Vector = np.floor(sx + 0.5).astype(np.int64)
        ny: Int64Vector = np.floor(sy + 0.5).astype(np.int64)

        # 6. Check which mapped coordinates actually fall inside the source image boundaries [0..w_src-1, 0..h_src-1]
        band_valid: BoolMask = (
            good_w
            & (sx > -0.5)
            & (sx < w_src - 0.5)
            & (sy > -0.5)
            & (sy < h_src - 0.5)
        )

        # 7. Reshape flat strip arrays back into 2D tile shape (band_h, cW) and copy valid pixels into 'out'
        band_h: int = r1 - r0
        valid[r0:r1] = band_valid.reshape(band_h, cW)
        src_x[r0:r1] = sx.reshape(band_h, cW)
        src_y[r0:r1] = sy.reshape(band_h, cW)
        if np.any(band_valid):
            out[r0:r1][band_valid.reshape(band_h, cW)] = img_arr[ny[band_valid], nx[band_valid]]

    return out, valid, src_x, src_y


# Placeholder stub for RANSAC outlier-rejection homography estimation (not yet implemented)
def ransac_homography(src: PointList2D, dst: PointList2D) -> tuple[HomographyMatrix, BoolMask]:
    return np.eye(3, dtype=np.float64), np.ones(len(src), dtype=np.bool_)