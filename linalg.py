from typing import Final, TypeAlias
import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- Type Aliases ---

FloatMatrix: TypeAlias = NDArray[np.float64]
"""2D matrix of 64-bit floats, shape `(M, N)`.

Example:
    `np.array([[2.0, 1.0], [5.0, 3.0]], dtype=np.float64)`
"""

FloatVector: TypeAlias = NDArray[np.float64]
"""1D vector of 64-bit floats, shape `(N,)`.

Example:
    `np.array([4.0, 11.0], dtype=np.float64)`
"""

# --- Constants ---

DEFAULT_TOL: Final[float] = 1e-10
"""Default relative pivot tolerance (`1e-10`) used to treat near-zero entries as zero during row reduction."""


class SingularMatrixError(ArithmeticError):
    """Raised when a matrix operation requires an invertible matrix."""


def _as_matrix(A: ArrayLike, name: str = "matrix") -> FloatMatrix:
    matrix: FloatMatrix = np.asarray(A, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _rref_with_pivots(A: ArrayLike, tol: float) -> tuple[FloatMatrix, list[int]]:
    if tol < 0:
        raise ValueError("tol must be non-negative")
    work: FloatMatrix = _as_matrix(A)
    rows: int
    cols: int
    rows, cols = work.shape
    scale: float = max(1.0, float(np.max(np.abs(work))) if work.size else 0.0)
    threshold: float = tol * scale
    pivot_cols: list[int] = []
    pivot_row: int = 0

    for col in range(cols):
        if pivot_row == rows:
            break
        pivot: int = pivot_row + int(np.argmax(np.abs(work[pivot_row:, col])))
        if abs(float(work[pivot, col])) <= threshold:
            continue

        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]

        work[pivot_row] /= work[pivot_row, col]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor: float = float(work[row, col])
            if factor != 0.0:
                work[row] -= factor * work[pivot_row]
        work[np.abs(work) <= threshold] = 0.0
        pivot_cols.append(col)
        pivot_row += 1

    work[np.abs(work) <= threshold] = 0.0
    return work, pivot_cols


def rref(A: ArrayLike, tol: float = DEFAULT_TOL) -> FloatMatrix:
    """Return the reduced row-echelon form using partial pivoting."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    result: FloatMatrix
    result, _ = _rref_with_pivots(A, tol)
    return result


def solve(A: ArrayLike, b: ArrayLike, tol: float = DEFAULT_TOL) -> FloatVector:
    """Solve a square system with a unique solution by reducing [A | b]."""
    matrix: FloatMatrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    rhs: FloatVector = np.asarray(b, dtype=np.float64)
    if rhs.size != matrix.shape[0]:
        raise ValueError("b must have one value per row of A")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("b must contain only finite values")
    rhs_col: FloatMatrix = rhs.reshape(-1, 1)

    reduced: FloatMatrix
    pivots: list[int]
    reduced, pivots = _rref_with_pivots(np.hstack((matrix, rhs_col)), tol)
    size: int = matrix.shape[0]
    if len(pivots) != size or pivots[:size] != list(range(size)):
        raise SingularMatrixError("A is singular; the system does not have a unique solution")
    return reduced[:size, -1].copy()


def inverse(A: ArrayLike, tol: float = DEFAULT_TOL) -> FloatMatrix:
    """Compute an inverse by reducing the augmented matrix [A | I]."""
    matrix: FloatMatrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    size: int = matrix.shape[0]
    augmented: FloatMatrix = np.hstack((matrix, np.eye(size, dtype=np.float64)))

    reduced: FloatMatrix
    pivots: list[int]
    reduced, pivots = _rref_with_pivots(augmented, tol)
    if pivots[:size] != list(range(size)) or len(pivots) < size:
        raise SingularMatrixError("A is singular and has no inverse")
    return reduced[:size, size:]


def det(A: ArrayLike, tol: float = DEFAULT_TOL) -> float:
    """Compute a determinant by cofactors for small matrices and elimination otherwise."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    matrix: FloatMatrix = _as_matrix(A, "A")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    size: int = matrix.shape[0]

    if size == 0:
        return 1.0
    if size == 1:
        return float(matrix[0, 0])
    if size == 2:
        return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if size == 3:
        a: float
        b: float
        c: float
        d: float
        e: float
        f: float
        g: float
        h: float
        i: float
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return float(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))

    work: FloatMatrix = matrix.copy()
    scale: float = max(1.0, float(np.max(np.abs(work))))
    threshold: float = tol * scale
    sign: float = 1.0
    determinant: float = 1.0
    for col in range(size):
        pivot: int = col + int(np.argmax(np.abs(work[col:, col])))
        if abs(float(work[pivot, col])) <= threshold:
            return 0.0
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
            sign = -sign
        pivot_value: float = float(work[col, col])
        determinant *= pivot_value
        for row in range(col + 1, size):
            factor: float = float(work[row, col]) / pivot_value
            work[row, col + 1:] -= factor * work[col, col + 1:]
    return float(sign * determinant)


def rank(M: ArrayLike, tol: float = DEFAULT_TOL) -> int:
    """Return the number of pivots in the echelon form."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    pivots: list[int]
    _, pivots = _rref_with_pivots(M, tol)
    return len(pivots)


def nullspace_vector(M: ArrayLike, tol: float = DEFAULT_TOL) -> FloatVector:
    """Return a unit vector in the null space, or raise if it contains only zero."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    reduced: FloatMatrix
    pivots: list[int]
    reduced, pivots = _rref_with_pivots(M, tol)
    cols: int = reduced.shape[1]
    free_cols: list[int] = [col for col in range(cols) if col not in pivots]
    if not free_cols:
        raise ValueError("M has a trivial null space")

    vector: FloatVector = np.zeros(cols, dtype=np.float64)
    free_col: int = free_cols[0]
    vector[free_col] = 1.0
    for row, pivot_col in reversed(list(enumerate(pivots))):
        vector[pivot_col] = -np.dot(reduced[row, pivot_col + 1:], vector[pivot_col + 1:])
    norm: float = float(np.sqrt(np.dot(vector, vector)))
    if norm <= tol:
        raise ValueError("could not find a nonzero null-space vector")
    return vector / norm