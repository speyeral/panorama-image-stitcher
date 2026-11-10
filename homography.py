import numpy as np
import linalg
import points

# Tolerance constants with mathematical rationale:
# COLLINEAR_TOL: Minimum absolute determinant of normalized coordinate triples [x y 1].
# Because points are normalized to mean distance sqrt(2), collinear triples have det near 0.
# A triangle with base and height 0.01 has normalized det ~ 1e-4. 1e-6 cleanly separates
# numerical noise on collinear lines from degenerate triples.
COLLINEAR_TOL = 1e-6

# CORNER_ORDER_TOL: Tolerance for 2x2 edge determinants (cross products) to detect
# collinearly flattened edges or degenerate order.
CORNER_ORDER_TOL = 1e-12

# MEAN_DIST_EPS: Tolerance below which points are considered coincident (zero spread).
MEAN_DIST_EPS = 1e-12

# SINGULAR_TOL: Tolerance for determinant/invertibility of system matrix A.
SINGULAR_TOL = 1e-10

# H33_TOL: Tolerance for non-vanishing homogeneous scaling factor h33.
H33_TOL = 1e-12

# HORIZON_TOL: Tolerance for denominator w = h31*x + h32*y + h33. If |w| <= HORIZON_TOL,
# the horizon touches the source point.
HORIZON_TOL = 1e-12


class HomographyError(Exception):
    """Raised when homography computation or validation fails."""
    pass


def normalize_points(pts):
    """Normalize 2D points to zero centroid and mean distance sqrt(2).

    Math anchors: M6, T-dim, T-rules.
    Transforms pts by 3x3 similarity T:
        T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    where (cx, cy) is the centroid and s = sqrt(2) / mean_dist.
    Raises HomographyError if all points are coincident (mean_dist < MEAN_DIST_EPS).
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HomographyError("Points must be an (N, 2) array")
    if not np.all(np.isfinite(pts)):
        raise HomographyError("Points must contain only finite numbers")

    cx, cy = np.mean(pts, axis=0)
    shifted = pts - [cx, cy]
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    if mean_dist < MEAN_DIST_EPS:
        raise HomographyError("Points are coincident: mean distance from centroid is zero")

    s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s * cx],
                  [0, s, -s * cy],
                  [0, 0, 1]], dtype=float)

    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    norm_pts = (T @ pts_homo.T).T[:, :2]
    return norm_pts, T


def build_system(src_n, dst_n):
    """Build the 2N x 8 linear system A h = b for homography unknowns with h33 = 1.

    Math anchors: M1, M3, M4, T-dim, T-sol, T-rn.
    For each correspondence (x, y) -> (x', y'):
        [x, y, 1, 0, 0, 0, -x*x', -y*x'] * h = x'
        [0, 0, 0, x, y, 1, -x*y', -y*y'] * h = y'
    Returns A of shape (2N, 8) and b of shape (2N,).
    """
    src_n = np.asarray(src_n, dtype=float)
    dst_n = np.asarray(dst_n, dtype=float)
    N = len(src_n)
    if len(dst_n) != N:
        raise HomographyError("src and dst must have the same number of points")

    A = np.zeros((2 * N, 8), dtype=float)
    b = np.zeros(2 * N, dtype=float)
    for i in range(N):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        b[2 * i] = u
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i + 1] = v
    return A, b


def _check_horizon(H, src):
    """Check that the projective denominator w = h31*x + h32*y + h33 has same sign for all src.

    Math anchors: M8, T-rules.
    If w crosses zero or touches zero across the source quad, the horizon intersects the image,
    making perspective division invalid.
    """
    H = np.asarray(H, dtype=float)
    src = np.asarray(src, dtype=float)
    w = H[2, 0] * src[:, 0] + H[2, 1] * src[:, 1] + H[2, 2]

    if np.any(np.abs(w) <= HORIZON_TOL):
        raise HomographyError("homography horizon touches the image boundary: w is zero")
    if np.any(w > 0) and np.any(w < 0):
        raise HomographyError("homography horizon crosses the image: mixed-sign w")


def _check_collinearity(pts_n, name="points"):
    """Check that no 3 points among 4 normalized points are collinear using 3x3 determinant.

    Math anchors: M7, T-equiv.
    """
    combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for i, j, k in combos:
        m = np.array([
            [pts_n[i, 0], pts_n[i, 1], 1.0],
            [pts_n[j, 0], pts_n[j, 1], 1.0],
            [pts_n[k, 0], pts_n[k, 1], 1.0]
        ], dtype=float)
        if abs(linalg.det(m)) <= COLLINEAR_TOL:
            raise HomographyError(f"collinear: three {name} points are collinear")


def _check_corner_order(pts, name="points"):
    """Check that the 4 points form a non-degenerate convex/properly wound quad.

    Math anchors: M7, T-equiv.
    Rejects self-intersecting quads (bow-ties) or swapped corners.
    """
    if not points.corner_order_ok(pts, tol=CORNER_ORDER_TOL):
        raise HomographyError(f"order: invalid corner order or self-intersecting {name}")


def solve_homography(src, dst):
    """Solve exact 4-point homography H mapping src to dst with H[2, 2] == 1.

    Math anchors: M1, M3, M4, M6, M7, M8; T-dim, T-sol, T-rn, T-equiv, T-inv, T-adj, T-rules.
    src and dst must be (4, 2) arrays of finite (x, y) coordinates.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    # 1. Validate input shapes and finiteness
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise HomographyError("Exact homography requires (4, 2) arrays for src and dst")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise HomographyError("Input points must contain only finite numbers")

    # 4. Normalize point sets
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)

    # 2. Check collinearity on normalized points
    _check_collinearity(src_norm, name="source")
    _check_collinearity(dst_norm, name="destination")

    # 3. Check corner order (reject bow-ties)
    _check_corner_order(src, name="source")
    _check_corner_order(dst, name="destination")

    # 5. Build system A h = b
    A, b = build_system(src_norm, dst_norm)

    # 6. Solve A h = b; guard singularity
    if abs(linalg.det(A)) <= SINGULAR_TOL:
        # Build 8x9 matrix M = [A | b] with b negated for diagnostics
        M = np.hstack([A, -b.reshape(-1, 1)])
        rank_M = linalg.rank(M)
        raise HomographyError(
            f"degenerate system: system matrix A is singular (|det A| <= {SINGULAR_TOL}); rank M = {rank_M}"
        )

    h8 = linalg.solve(A, b)
    H_norm = np.append(h8, 1.0).reshape(3, 3)

    # 7. Denormalize: H = T_dst^-1 @ H_norm @ T_src
    T_dst_inv = linalg.inverse(T_dst)
    H = T_dst_inv @ H_norm @ T_src

    if abs(H[2, 2]) <= H33_TOL:
        raise HomographyError(f"h33 near zero: H[2, 2] magnitude is below tolerance ({H33_TOL})")

    H = H / H[2, 2]

    # 8. Horizon check on original source points
    _check_horizon(H, src)

    return H


def transfer_error(H, src, dst):
    """Compute Euclidean pixel transfer error per correspondence pair.

    Math anchors: M8, T-rules.
    Maps src through H: [x', y', w']^T = H @ [x, y, 1]^T, then computes distance ||(x'/w', y'/w') - dst||.
    Returns array of shape (N,).
    """
    H = np.asarray(H, dtype=float)
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.ndim != 2 or src.shape[1] != 2:
        raise HomographyError("src must be an (N, 2) array")
    if dst.ndim != 2 or dst.shape[1] != 2:
        raise HomographyError("dst must be an (N, 2) array")
    if len(src) != len(dst):
        raise HomographyError("src and dst must have the same length")

    src_h = np.hstack([src, np.ones((len(src), 1))])
    dst_proj = (H @ src_h.T).T
    w = dst_proj[:, 2:3]
    w_safe = np.where(np.abs(w) > HORIZON_TOL, w, 1.0)
    dst_mapped = dst_proj[:, :2] / w_safe
    return np.sqrt(np.sum((dst - dst_mapped)**2, axis=1))


# Identity Stubs for interface freeze
def solve_homography_lsq(src, dst): return np.eye(3)
def algebraic_residual(H, src, dst): return 0.0
