from typing import Final, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray
import linalg
import points

# --- Type Aliases (for code readability and static type checking) ---

FloatArray: TypeAlias = NDArray[np.float64]
"""General 64-bit float array of arbitrary shape.

Example:
    `np.zeros((3, 3), dtype=np.float64)`
"""

FloatVector: TypeAlias = NDArray[np.float64]
"""1D array of 64-bit floats, shape `(N,)`.

Example:
    `np.array([1.2, 0.0, 0.5, -0.1, 1.1, 0.0, 0.0, 0.0], dtype=np.float64)`
"""

FloatMatrix: TypeAlias = NDArray[np.float64]
"""2D matrix of 64-bit floats, shape `(M, N)`.

Example:
    `np.zeros((8, 8), dtype=np.float64)`
"""

PointList2D: TypeAlias = NDArray[np.float64]
"""List of N 2D `(x, y)` coordinates, shape `(N, 2)`.

Example:
    `np.array([[100.0, 150.0], [500.0, 140.0], [510.0, 420.0], [95.0, 430.0]], dtype=np.float64)`
"""

HomographyMatrix: TypeAlias = NDArray[np.float64]
"""3x3 projective transformation matrix, shape `(3, 3)`.

Example:
    `np.eye(3, dtype=np.float64)`
"""


# --- Numerical Tolerance Constants ---
# Computers store decimals with tiny rounding errors (e.g., 1e-16 instead of 0).
# These thresholds define when a value is "close enough to zero" to be treated as degenerate.

COLLINEAR_TOL: Final[float] = 1e-6
"""Minimum 3x3 determinant for 3 normalized points (`1e-6`).

If 3 points lie on the same straight line, the triangle between them has zero area,
making their 3x3 homogeneous determinant `0`.
"""

CORNER_ORDER_TOL: Final[float] = 1e-12
"""Minimum 2D cross-product magnitude (`1e-12`) when validating quadrilateral winding order.

Ensures the 4 corners trace a clean perimeter without crossing over into a self-intersecting bow-tie.
"""

MEAN_DIST_EPS: Final[float] = 1e-12
"""Minimum average Euclidean distance (`1e-12`) of points from their centroid.

Prevents division by zero during Hartley normalization if all clicked points sit on the exact same pixel.
"""

SINGULAR_TOL: Final[float] = 1e-10
"""Minimum `|det(A)|` (`1e-10`) for the 8x8 linear system matrix `A`.

If `|det(A)|` falls below this threshold, `A` is singular (non-invertible) and Gaussian elimination cannot solve it.
"""

H33_TOL: Final[float] = 1e-12
"""Minimum allowed magnitude (`1e-12`) for the bottom-right entry `H[2, 2]` before normalizing `H / H[2, 2]`."""

HORIZON_TOL: Final[float] = 1e-12
"""Minimum allowed magnitude (`1e-12`) for the projective denominator `w = h31*x + h32*y + h33`.

Since `(x', y') = (x'w' / w, y'w' / w)`, `w == 0` causes division by zero (projecting points to infinity).
"""


class HomographyError(Exception):
    """Custom exception raised whenever a homography cannot be safely computed or is invalid."""
    pass


def normalize_points(pts: ArrayLike) -> tuple[PointList2D, HomographyMatrix]:
    """Hartley Preconditioning: Shift points to (0, 0) center and scale average distance to sqrt(2).

    Why we do this: Raw pixel coordinates (like x=2000) become huge (4,000,000) when multiplied
    inside matrix A, causing floating-point rounding errors during Gaussian elimination.
    Shrinking all coordinates to around +/- 1.0 keeps the math numerically stable.
    """
    # Convert input list/array into a standard 64-bit float NumPy array
    pts_arr: PointList2D = np.asarray(pts, dtype=np.float64)

    # Validate shape is (N, 2) and contains no NaN or infinity values
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
        raise HomographyError("Points must be an (N, 2) array")
    if not np.all(np.isfinite(pts_arr)):
        raise HomographyError("Points must contain only finite numbers")

    # 1. Compute the centroid (center of mass): average x and average y of all points
    cx: float
    cy: float
    cx, cy = np.mean(pts_arr, axis=0)

    # 2. Shift all points so their center of mass sits right at the origin (0, 0)
    shifted: PointList2D = pts_arr - np.array([cx, cy], dtype=np.float64)

    # 3. Compute the Euclidean distance sqrt(dx^2 + dy^2) of each point from (0, 0), then average them
    mean_dist: float = float(np.mean(np.sqrt(np.sum(shifted**2, axis=1))))
    if mean_dist < MEAN_DIST_EPS:
        raise HomographyError("Points are coincident: mean distance from centroid is zero")

    # 4. Calculate uniform scale factor 's' so the new average distance from (0, 0) equals sqrt(2)
    s: float = float(np.sqrt(2.0)) / mean_dist

    # 5. Pack both the shift (-cx, -cy) and scaling (s) into a single 3x3 similarity matrix T
    T: HomographyMatrix = np.array(
        [
            [s, 0.0, -s * cx],
            [0.0, s, -s * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 6. Convert (N, 2) points to homogeneous (N, 3) points by appending a column of 1s
    pts_homo: FloatMatrix = np.hstack([pts_arr, np.ones((len(pts_arr), 1), dtype=np.float64)])

    # 7. Multiply T @ pts_homo.T to apply the shift and scale, transpose back, and drop the dummy 1
    norm_pts: PointList2D = (T @ pts_homo.T).T[:, :2]
    return norm_pts, T


def build_system(src_n: ArrayLike, dst_n: ArrayLike) -> tuple[FloatMatrix, FloatVector]:
    """Construct the linear system A * h = b from N matching point pairs.

    We fix h33 = 1, leaving 8 unknown entries in vector h = [h11, h12, h13, h21, h22, h23, h31, h32]^T.
    Each matching pair (x, y) -> (u, v) gives 2 linear equations:
        Row 2i:     x*h11 + y*h12 + 1*h13 + 0*h21 + 0*h22 + 0*h23 - u*x*h31 - u*y*h32 = u
        Row 2i + 1: 0*h11 + 0*h21 + 0*h13 + x*h21 + y*h22 + 1*h23 - v*x*h31 - v*y*h32 = v
    """
    src_arr: PointList2D = np.asarray(src_n, dtype=np.float64)
    dst_arr: PointList2D = np.asarray(dst_n, dtype=np.float64)
    N: int = len(src_arr)
    if len(dst_arr) != N:
        raise HomographyError("src and dst must have the same number of points")

    # Initialize empty coefficient matrix A (2N rows x 8 columns) and constant vector b (2N rows)
    A: FloatMatrix = np.zeros((2 * N, 8), dtype=np.float64)
    b: FloatVector = np.zeros(2 * N, dtype=np.float64)

    for i in range(N):
        x: float
        y: float
        u: float  # destination x'
        v: float  # destination y'
        x, y = src_arr[i]
        u, v = dst_arr[i]

        # First equation for point i (maps x coordinate)
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]
        b[2 * i] = u

        # Second equation for point i (maps y coordinate)
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]
        b[2 * i + 1] = v

    return A, b


def _check_horizon(H: ArrayLike, src: ArrayLike) -> None:
    """Ensure the perspective denominator w never hits zero or flips sign across the image.

    When we map a point [x, y, 1]^T through H, the 3rd coordinate is w = h31*x + h32*y + h33.
    If w == 0, dividing by w explodes to infinity (the "vanishing horizon" cuts through the photo).
    If w is positive for some corners and negative for others, the horizon crosses inside the quad.
    """
    H_arr: HomographyMatrix = np.asarray(H, dtype=np.float64)
    src_arr: PointList2D = np.asarray(src, dtype=np.float64)

    # Compute the 3rd homogeneous coordinate w for every source point
    w: FloatVector = H_arr[2, 0] * src_arr[:, 0] + H_arr[2, 1] * src_arr[:, 1] + H_arr[2, 2]

    if np.any(np.abs(w) <= HORIZON_TOL):
        raise HomographyError("homography horizon touches the image boundary: w is zero")
    if np.any(w > 0) and np.any(w < 0):
        raise HomographyError("homography horizon crosses the image: mixed-sign w")


def _check_collinearity(pts_n: PointList2D, name: str = "points") -> None:
    """Verify that no 3 of the 4 points lie on a single straight line.

    A 3x3 matrix of homogeneous 2D points [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]] has a
    determinant equal to twice the signed area of the triangle they form.
    If det == 0, the triangle has zero area -> the 3 points are collinear (which breaks homography).
    """
    # All 4 possible ways to choose 3 points out of 4
    combos: list[tuple[int, int, int]] = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for i, j, k in combos:
        m: HomographyMatrix = np.array(
            [
                [pts_n[i, 0], pts_n[i, 1], 1.0],
                [pts_n[j, 0], pts_n[j, 1], 1.0],
                [pts_n[k, 0], pts_n[k, 1], 1.0],
            ],
            dtype=np.float64,
        )
        if abs(linalg.det(m)) <= COLLINEAR_TOL:
            raise HomographyError(f"collinear: three {name} points are collinear")


def _check_corner_order(pts: ArrayLike, name: str = "points") -> None:
    """Verify the 4 corners trace a clean perimeter without crossing over into a 'bow-tie' shape."""
    if not points.corner_order_ok(pts, tol=CORNER_ORDER_TOL):
        raise HomographyError(f"order: invalid corner order or self-intersecting {name}")


def solve_homography(src: ArrayLike, dst: ArrayLike) -> HomographyMatrix:
    """Compute the exact 3x3 Homography matrix H from 4 matching point pairs (used for Rectification).

    Pipeline:
    Validate -> Normalize -> Geometry Checks -> Build 8x8 System -> Solve -> Denormalize -> Horizon Check.
    """
    src_arr: PointList2D = np.asarray(src, dtype=np.float64)
    dst_arr: PointList2D = np.asarray(dst, dtype=np.float64)

    # 1. Validate that we have exactly 4 (x, y) pairs and no NaN/Inf values
    if src_arr.shape != (4, 2) or dst_arr.shape != (4, 2):
        raise HomographyError("Exact homography requires (4, 2) arrays for src and dst")
    if not np.all(np.isfinite(src_arr)) or not np.all(np.isfinite(dst_arr)):
        raise HomographyError("Input points must contain only finite numbers")

    # 2. Hartley Normalization: center and scale both point sets for numerical stability
    src_norm: PointList2D
    T_src: HomographyMatrix
    src_norm, T_src = normalize_points(src_arr)

    dst_norm: PointList2D
    T_dst: HomographyMatrix
    dst_norm, T_dst = normalize_points(dst_arr)

    # 3. Check that no 3 corners lie in a straight line
    _check_collinearity(src_norm, name="source")
    _check_collinearity(dst_norm, name="destination")

    # 4. Check that the 4 corners form a proper quadrilateral (not a twisted bow-tie)
    _check_corner_order(src_arr, name="source")
    _check_corner_order(dst_arr, name="destination")

    # 5. Build the square 8x8 linear system A * h = b using the normalized points
    A: FloatMatrix
    b: FloatVector
    A, b = build_system(src_norm, dst_norm)

    # 6. Check if A is invertible (det != 0), then solve for the 8 unknowns in h8
    if abs(linalg.det(A)) <= SINGULAR_TOL:
        # Form augmented matrix [A | -b] to check its rank for a helpful error message
        M: FloatMatrix = np.hstack([A, -b.reshape(-1, 1)])
        rank_M: int = linalg.rank(M)
        raise HomographyError(
            f"degenerate system: system matrix A is singular (|det A| <= {SINGULAR_TOL}); rank M = {rank_M}"
        )

    h8: FloatVector = linalg.solve(A, b)
    # Append h33 = 1.0 to the 8 solved numbers and reshape into a 3x3 matrix H_norm
    H_norm: HomographyMatrix = np.append(h8, 1.0).reshape(3, 3)

    # 7. Denormalize: undo T_src and T_dst so H works on raw full-resolution pixel coordinates
    # Since p_dst_norm = H_norm @ p_src_norm  =>  T_dst @ p_dst = H_norm @ T_src @ p_src
    # Multiplying both sides on the left by T_dst^-1 gives: H = T_dst^-1 @ H_norm @ T_src
    T_dst_inv: HomographyMatrix = linalg.inverse(T_dst)
    H: HomographyMatrix = T_dst_inv @ H_norm @ T_src

    # Re-scale H so its bottom-right corner H[2, 2] is standardized back to 1.0
    if abs(float(H[2, 2])) <= H33_TOL:
        raise HomographyError(f"h33 near zero: H[2, 2] magnitude is below tolerance ({H33_TOL})")
    H = H / H[2, 2]

    # 8. Final safety check: make sure H doesn't send any source corner to infinity
    _check_horizon(H, src_arr)

    return H


def transfer_error(H: ArrayLike, src: ArrayLike, dst: ArrayLike) -> FloatVector:
    """Measure how accurate H is by projecting src points through H and measuring pixel distance to dst."""
    H_arr: HomographyMatrix = np.asarray(H, dtype=np.float64)
    src_arr: PointList2D = np.asarray(src, dtype=np.float64)
    dst_arr: PointList2D = np.asarray(dst, dtype=np.float64)
    if src_arr.ndim != 2 or src_arr.shape[1] != 2:
        raise HomographyError("src must be an (N, 2) array")
    if dst_arr.ndim != 2 or dst_arr.shape[1] != 2:
        raise HomographyError("dst must be an (N, 2) array")
    if len(src_arr) != len(dst_arr):
        raise HomographyError("src and dst must have the same length")

    # Convert src to homogeneous [x, y, 1] and multiply by H
    src_h: FloatMatrix = np.hstack([src_arr, np.ones((len(src_arr), 1), dtype=np.float64)])
    dst_proj: FloatMatrix = (H_arr @ src_h.T).T

    # Homogeneous normalization: divide projected x' and y' by the 3rd coordinate w'
    w: FloatMatrix = dst_proj[:, 2:3]
    w_safe: FloatMatrix = np.where(np.abs(w) > HORIZON_TOL, w, 1.0)
    dst_mapped: PointList2D = dst_proj[:, :2] / w_safe

    # Return Euclidean pixel distance sqrt((x_true - x_pred)^2 + (y_true - y_pred)^2) for each pair
    return np.sqrt(np.sum((dst_arr - dst_mapped) ** 2, axis=1))


def solve_homography_lsq(src: ArrayLike, dst: ArrayLike) -> HomographyMatrix:
    """Estimate best-fit 3x3 Homography from N >= 5 noisy point pairs (used for Panorama Stitching).

    With N >= 5 pairs, A has 2N >= 10 rows for only 8 unknowns (overdetermined system).
    Because human clicks have slight noise, no exact solution exists. Instead, we solve the
    Normal Equations: (A^T A) h = A^T b, which minimizes the squared error ||A h - b||^2.
    """
    src_arr: PointList2D = np.asarray(src, dtype=np.float64)
    dst_arr: PointList2D = np.asarray(dst, dtype=np.float64)

    # 1. Validate shapes and ensure we have at least 5 pairs
    if src_arr.ndim != 2 or src_arr.shape[1:] != (2,):
        raise HomographyError("src must be an (N, 2) array")
    if dst_arr.ndim != 2 or dst_arr.shape[1:] != (2,):
        raise HomographyError("dst must be an (N, 2) array")
    if len(src_arr) != len(dst_arr):
        raise HomographyError("src and dst must have the same number of points")
    if len(src_arr) < 5:
        raise HomographyError("least-squares homography requires at least five point pairs")
    if not np.all(np.isfinite(src_arr)) or not np.all(np.isfinite(dst_arr)):
        raise HomographyError("Input points must contain only finite numbers")

    # 2. Hartley Normalization on both point sets
    src_norm: PointList2D
    T_src: HomographyMatrix
    src_norm, T_src = normalize_points(src_arr)

    dst_norm: PointList2D
    T_dst: HomographyMatrix
    dst_norm, T_dst = normalize_points(dst_arr)

    # 3. Build the tall (2N x 8) matrix A and (2N,) vector b
    A: FloatMatrix
    b: FloatVector
    A, b = build_system(src_norm, dst_norm)

    # Verify A has 8 linearly independent columns (full column rank)
    if linalg.rank(A) < 8:
        raise HomographyError("rank-deficient point configuration; homography is not determined")

    # 4. Form the square (8 x 8) symmetric Normal Matrix (A^T @ A)
    normal_matrix: FloatMatrix = A.T @ A
    if linalg.rank(normal_matrix) < 8:
        raise HomographyError("rank-deficient normal equations; homography is not determined")

    # 5. Solve (A^T A) h = A^T b via Gaussian elimination
    h8: FloatVector = linalg.solve(normal_matrix, A.T @ b)
    H_norm: HomographyMatrix = np.append(h8, 1.0).reshape(3, 3)

    # 6. Denormalize back to raw pixel space and standardize H[2, 2] = 1
    H: HomographyMatrix = linalg.inverse(T_dst) @ H_norm @ T_src

    if abs(float(H[2, 2])) <= H33_TOL:
        raise HomographyError(f"h33 near zero: H[2, 2] magnitude is below tolerance ({H33_TOL})")
    H = H / H[2, 2]
    if not np.all(np.isfinite(H)):
        raise HomographyError("least-squares homography contains non-finite values")
    return H


def algebraic_residual(H: ArrayLike, src: ArrayLike, dst: ArrayLike) -> float:
    """Compute the algebraic error ||A h - b||^2 in normalized coordinates.

    Used to check how well a candidate homography H satisfies the linear system equations
    (0.0 means a perfect exact fit; small positive values reflect human clicking noise).
    """
    H_arr: HomographyMatrix = np.asarray(H, dtype=np.float64)
    src_arr: PointList2D = np.asarray(src, dtype=np.float64)
    dst_arr: PointList2D = np.asarray(dst, dtype=np.float64)

    # 1. Validate inputs
    if H_arr.shape != (3, 3) or not np.all(np.isfinite(H_arr)):
        raise HomographyError("H must be a finite (3, 3) array")
    if src_arr.ndim != 2 or src_arr.shape[1:] != (2,):
        raise HomographyError("src must be an (N, 2) array")
    if dst_arr.ndim != 2 or dst_arr.shape[1:] != (2,):
        raise HomographyError("dst must be an (N, 2) array")
    if len(src_arr) != len(dst_arr):
        raise HomographyError("src and dst must have the same number of points")
    if len(src_arr) < 1:
        raise HomographyError("at least one point pair is required")
    if not np.all(np.isfinite(src_arr)) or not np.all(np.isfinite(dst_arr)):
        raise HomographyError("Input points must contain only finite numbers")

    # 2. Normalize points to match the coordinate space where A and b are defined
    src_norm: PointList2D
    T_src: HomographyMatrix
    src_norm, T_src = normalize_points(src_arr)

    dst_norm: PointList2D
    T_dst: HomographyMatrix
    dst_norm, T_dst = normalize_points(dst_arr)

    # 3. Transform raw H into normalized space: H_norm = T_dst @ H @ T_src^-1
    H_norm: HomographyMatrix = T_dst @ H_arr @ linalg.inverse(T_src)
    if abs(float(H_norm[2, 2])) <= H33_TOL:
        raise HomographyError("normalized homography has h33 near zero")
    H_norm = H_norm / H_norm[2, 2]

    # 4. Build normalized system A, b and flatten H_norm's first 8 entries into vector h
    A: FloatMatrix
    b: FloatVector
    A, b = build_system(src_norm, dst_norm)
    h: FloatVector = np.array(
        [
            H_norm[0, 0], H_norm[0, 1], H_norm[0, 2],
            H_norm[1, 0], H_norm[1, 1], H_norm[1, 2],
            H_norm[2, 0], H_norm[2, 1],
        ],
        dtype=np.float64,
    )

    # 5. Compute residual vector r = A*h - b, then take dot product r . r = ||r||^2
    residual: FloatVector = A @ h - b
    return float(np.dot(residual, residual))