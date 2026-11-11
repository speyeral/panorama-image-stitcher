import numpy as np


DEFAULT_TOL = 1e-10


class SingularMatrixError(ArithmeticError):
    """Raised when a matrix operation requires an invertible matrix."""


def _as_matrix(A, name="matrix"):
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _rref_with_pivots(A, tol):
    if tol < 0:
        raise ValueError("tol must be non-negative")
    work = _as_matrix(A)
    rows, cols = work.shape
    scale = max(1.0, float(np.max(np.abs(work))) if work.size else 0.0)
    threshold = tol * scale
    pivot_cols = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row == rows:
            break
        pivot = pivot_row + int(np.argmax(np.abs(work[pivot_row:, col])))
        if abs(work[pivot, col]) <= threshold:
            continue

        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]

        work[pivot_row] /= work[pivot_row, col]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row, col]
            if factor != 0.0:
                work[row] -= factor * work[pivot_row]
        work[np.abs(work) <= threshold] = 0.0
        pivot_cols.append(col)
        pivot_row += 1

    work[np.abs(work) <= threshold] = 0.0
    return work, pivot_cols


def rref(A, tol=DEFAULT_TOL):
    """Return the reduced row-echelon form using partial pivoting."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    result, _ = _rref_with_pivots(A, tol)
    return result


def solve(A, b, tol=DEFAULT_TOL):
    """Solve a square system with a unique solution by reducing [A | b]."""
    matrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    rhs = np.asarray(b, dtype=float)
    if rhs.size != matrix.shape[0]:
        raise ValueError("b must have one value per row of A")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("b must contain only finite values")
    rhs = rhs.reshape(-1, 1)

    reduced, pivots = _rref_with_pivots(np.hstack((matrix, rhs)), tol)
    size = matrix.shape[0]
    if len(pivots) != size or pivots[:size] != list(range(size)):
        raise SingularMatrixError("A is singular; the system does not have a unique solution")
    return reduced[:size, -1].copy()


def inverse(A, tol=DEFAULT_TOL):
    """Compute an inverse by reducing the augmented matrix [A | I]."""
    matrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    size = matrix.shape[0]
    augmented = np.hstack((matrix, np.eye(size)))
    reduced, pivots = _rref_with_pivots(augmented, tol)
    if pivots[:size] != list(range(size)) or len(pivots) < size:
        raise SingularMatrixError("A is singular and has no inverse")
    return reduced[:size, size:]


def det(A, tol=DEFAULT_TOL):
    """Compute a determinant by cofactors for small matrices and elimination otherwise."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    matrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    size = matrix.shape[0]

    if size == 0:
        return 1.0
    if size == 1:
        return float(matrix[0, 0])
    if size == 2:
        return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if size == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return float(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))

    work = matrix.copy()
    scale = max(1.0, float(np.max(np.abs(work))))
    threshold = tol * scale
    sign = 1.0
    determinant = 1.0
    for col in range(size):
        pivot = col + int(np.argmax(np.abs(work[col:, col])))
        if abs(work[pivot, col]) <= threshold:
            return 0.0
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
            sign = -sign
        pivot_value = work[col, col]
        determinant *= pivot_value
        for row in range(col + 1, size):
            factor = work[row, col] / pivot_value
            work[row, col + 1:] -= factor * work[col, col + 1:]
    return float(sign * determinant)


def rank(M, tol=DEFAULT_TOL):
    """Return the number of pivots in the echelon form."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    _, pivots = _rref_with_pivots(M, tol)
    return len(pivots)


def nullspace_vector(M, tol=DEFAULT_TOL):
    """Return a unit vector in the null space, or raise if it contains only zero."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    reduced, pivots = _rref_with_pivots(M, tol)
    cols = reduced.shape[1]
    free_cols = [col for col in range(cols) if col not in pivots]
    if not free_cols:
        raise ValueError("M has a trivial null space")

    vector = np.zeros(cols, dtype=float)
    free_col = free_cols[0]
    vector[free_col] = 1.0
    for row, pivot_col in reversed(list(enumerate(pivots))):
        vector[pivot_col] = -np.dot(reduced[row, pivot_col + 1:], vector[pivot_col + 1:])
    norm = np.sqrt(np.dot(vector, vector))
    if norm <= tol:
        raise ValueError("could not find a nonzero null-space vector")
    return vector / norm
