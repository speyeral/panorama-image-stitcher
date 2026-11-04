import numpy as np

def rref(A, tol=1e-10):
    """Computes RREF using Gauss-Jordan elimination with partial pivoting."""
    A_work = A.astype(float).copy()
    rows, cols = A_work.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        max_idx = np.argmax(np.abs(A_work[pivot_row:rows, col])) + pivot_row
        if np.abs(A_work[max_idx, col]) < tol:
            A_work[pivot_row:rows, col] = 0.0
            continue
        
        A_work[[pivot_row, max_idx]] = A_work[[max_idx, pivot_row]]
        A_work[pivot_row] = A_work[pivot_row] / A_work[pivot_row, col]
        
        for r in range(rows):
            if r != pivot_row:
                A_work[r] -= A_work[r, col] * A_work[pivot_row]
        pivot_row += 1
        
    A_work[np.abs(A_work) < tol] = 0.0
    return A_work

def solve(A, b, tol=1e-10):
    """Solves Ax = b by reducing [A | b] to RREF."""
    A_work = np.asarray(A, dtype=float)
    b_work = np.asarray(b, dtype=float).reshape(-1, 1)
    aug = np.hstack([A_work, b_work])
    R = rref(aug, tol=tol)
    return R[:, -1]

def inverse(A):
    """Computes matrix inverse."""
    return np.linalg.inv(A)

def det(A):
    """Computes matrix determinant."""
    return float(np.linalg.det(A))

def rank(M):
    """Computes matrix rank."""
    return int(np.linalg.matrix_rank(M))

def nullspace_vector(M, tol=1e-10):
    """Computes a non-zero vector in the nullspace of M (e.g. via SVD)."""
    _, s, vh = np.linalg.svd(M)
    return vh[-1]
