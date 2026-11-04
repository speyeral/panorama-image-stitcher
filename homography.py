import numpy as np
import linalg

class HomographyError(Exception):
    pass

def normalize_points(pts):
    pts = np.asarray(pts, dtype=float)
    cx, cy = np.mean(pts, axis=0)
    shifted = pts - [cx, cy]
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    s = np.sqrt(2) / (mean_dist if mean_dist > 1e-12 else 1.0)
    
    T = np.array([[s, 0, -s * cx], 
                  [0, s, -s * cy], 
                  [0, 0, 1]], dtype=float)
    
    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    norm_pts = (T @ pts_homo.T).T[:, :2]
    return norm_pts, T

def build_system(src, dst):
    N = len(src)
    A = np.zeros((2 * N, 8), dtype=float)
    b = np.zeros(2 * N, dtype=float)
    for i in range(N):
        x, y = src[i]
        u, v = dst[i]
        A[2*i]   = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        b[2*i]   = u
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
        b[2*i+1] = v
    return A, b

def solve_homography(src, dst):
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) != 4 or len(dst) != 4:
        raise HomographyError("Exact homography requires 4 point pairs")
        
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)
    
    A, b = build_system(src_norm, dst_norm)
    h8 = linalg.solve(A, b)
    
    H_norm = np.append(h8, 1.0).reshape(3, 3)
    
    # Analytic inverse of triangular T_dst for the thin slice
    T_dst_inv = np.array([[1/T_dst[0,0], 0, -T_dst[0,2]/T_dst[0,0]],
                          [0, 1/T_dst[1,1], -T_dst[1,2]/T_dst[1,1]],
                          [0, 0, 1]])
                          
    H = T_dst_inv @ H_norm @ T_src
    if np.abs(H[2, 2]) < 1e-12:
        raise HomographyError("h33 is zero")
    return H / H[2, 2]

# Identity Stubs for interface freeze
def solve_homography_lsq(src, dst): return np.eye(3)
def algebraic_residual(H, src, dst): return 0.0
def transfer_error(H, src, dst): return np.zeros(len(src))
