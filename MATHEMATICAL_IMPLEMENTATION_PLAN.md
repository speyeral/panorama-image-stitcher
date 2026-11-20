# Mathematical Implementation Plan: Panorama Image Stitcher & Rectifier

**Source documents:** `README.md` and `REFERENCES.md`.
**Reference text:** Kolman & Hill (2014), *Elementary Linear Algebra with Applications*, 9th ed.
**Explicitly excluded:** `match.py` (one-off check for a specific image; not part of the implementation).

---

## 1. Mathematical Background

This section explains the linear algebra concepts behind the implementation, each tied directly to a purpose in the program.

### 1.1 Vector Operations

**Concepts:** Linear combinations, span, and vector-space operations (Sections 4.2, 4.4, Kolman 2014).

A vector in ℝⁿ is an ordered n-tuple of real numbers. The **span** of a set of vectors is the set of all linear combinations of those vectors. Vector-space operations (addition and scalar multiplication) are closed in ℝⁿ: if **u**, **v** ∈ ℝⁿ and c ∈ ℝ, then **u** + **v** ∈ ℝⁿ and c**u** ∈ ℝⁿ. An image can be viewed as a vector: a grayscale image of size H×W is a vector in ℝ^(H·W); an RGB image is three such vectors (one per channel). Blending two images is a linear combination of these image-vectors.

**Purpose in program:**
- **Feather blending** (`blend.feather_blend`): The output pixel is a convex combination `Σᵢ αᵢ·Iᵢ(p)` of the warped image-vectors at pixel p, where weights αᵢ sum to 1. This is precisely a vector-space linear combination restricted to the convex hull.
- **Grayscale conversion** (`imgops.grayscale`): `gray = I · w` where `w = [0.2989, 0.5870, 0.1140]` is a linear combination of the three color channels at each pixel — a vector dot product (matrix-vector product with a 1×3 matrix).
- **Resize** (`imgops.resize`): `I′ = R·I·Cᵀ` where each output pixel is a weighted sum (linear combination) of input pixels in a local box neighborhood. Each row of R (and C) sums to 1, making it a convex combination per row/column.

**Theorem basis:** Vector-space closure under linear combinations (Sections 4.2, 4.4). REFERENCES §3 treats this as a definitional fact with no numbered theorem tag.

### 1.2 Homogeneous Coordinates

**Concept:** A point (x, y) in the image plane is represented as the homogeneous vector `[x, y, 1]ᵀ ∈ ℝ³`. More formally, it stands for the span `span{[x, y, 1]ᵀ}` in ℝ³ — all scalar multiples `c·[x, y, 1]ᵀ` (c ≠ 0) represent the same physical point. The Euclidean point is recovered by dividing by the third coordinate: `(x/w, y/w)`.

A homography `H` and any nonzero scalar multiple `cH` (`c ≠ 0`) induce the same map on the image plane, because `[c·x′, c·y′, c·w′]ᵀ` normalizes to the same `(x′/w′, y′/w′)`.

**Purpose in program:**
- All point mappings in `homography.py` and `warp.py` use homogeneous coordinates internally: points are appended with a `1` column (`np.hstack([pts, ones])`), multiplied by the 3×3 `H`, then divided by `w` (the third component) to return to Euclidean coordinates.
- This is what enables perspective division (`x′/w′`, `y′/w′`) and the `w` denominator that creates realistic vanishing lines.
- Gauge freedom: since `H` and `cH` are equivalent, we fix `h₃₃ = 1` to remove scale ambiguity, leaving exactly 8 unknowns.

**Theorem basis:** Homogeneous coordinates are a definitional construction (REFERENCES §2.2). The equivalence `H ~ cH` is the projective-scale invariance that motivates fixing `h₃₃ = 1` (T-dim: 3×3 matrices have dimension 9; removing scale leaves 8 unknowns).

### 1.3 Matrices and Transformations

#### Matrix Products (Section 1.3)

**Concept:** If A is m×n and B is n×p, the product AB is m×p with `(AB)ᵢⱼ = Σₖ Aᵢₖ·Bₖⱼ`. Matrix multiplication is associative but not commutative.

**Purpose in program:**
- Homography application: `p′ = H·p` where H is 3×3 and p is 3×1 → a 3×1 result.
- Denormalization: `H = T_dst⁻¹ · H̃ · T_src` (a product of three 3×3 matrices).
- Image resize: `I′ = R · I · Cᵀ` where R is new_h×old_h, I is old_h×old_w, Cᵀ is old_w×new_w.
- Grayscale: `gray = I · w` (H×W×3 image contracted with a 3-vector).

**Theorem basis:** Matrix product definition (Section 1.3). Associativity underlies the chain `T_dst⁻¹ · H̃ · T_src` being well-defined regardless of parenthesization.

#### Triangular Matrices (Section 1.5)

**Concept:** A matrix is upper triangular if all entries below the main diagonal are zero. The determinant of a triangular matrix is the product of its diagonal entries. A triangular matrix is nonsingular iff all diagonal entries are nonzero.

**Purpose in program:**
- The Hartley normalization matrix `T = [[s, 0, −s·cx], [0, s, −s·cy], [0, 0, 1]]` is upper triangular with diagonal entries `s, s, 1`. Since `s = √2/mean_dist > 0` (points are not coincident), all diagonal entries are nonzero, so `det T = s² ≠ 0` and `T` is invertible.
- Invertibility of T is required to denormalize: `H = T_dst⁻¹ · H̃ · T_src`.

**Theorem basis:** Triangular matrices and their determinants (Section 1.5). Nonsingularity of T follows from `det T = s² ≠ 0` (T-equiv: `det A ≠ 0 ⇔ A` nonsingular).


#### Permutation Matrices (used for image flips)

**Concept:** A permutation matrix has exactly one `1` in each row and column and `0`s elsewhere. Multiplying a matrix on the left by a permutation matrix permutes its rows; multiplying on the right permutes its columns. The exchange (reversal) matrix `Jₙ = Iₙ[::-1]` reverses index order.

**Purpose in program:**
- **Vertical flip** (`imgops.flip_v`): left-multiply by `J_H` (H×H exchange matrix) → `P @ arr` reverses row order.
- **Horizontal flip** (`imgops.flip_h`): right-multiply by `J_W` (W×W exchange matrix) → `arr @ Q` reverses column order.
- **EXIF orientation tags 1–8** (`imgops.apply_exif_orientation`): each tag is implemented as a composition of transposes, flips, and 90° rotations — all permutation operations on the pixel grid.
- **`rotate90`**: transpose then flip (two permutation matrices).

**Theorem basis:** Permutation matrices (Section 1.4/1.5; REFERENCES §2.1 lists "Permutation matrix — Image flips"). Each row operation / axis reversal is left- or right-multiplication by a permutation matrix.

#### Transpose and Symmetric Matrices

**Concept:** The transpose `Aᵀ` swaps rows and columns. A matrix is symmetric if `A = Aᵀ`. The structure tensor `M = Σ g·gᵀ` is symmetric by construction since `(g·gᵀ)ᵀ = g·gᵀ`.

**Purpose in program:**
- **Structure tensor** (REFERENCES §2.2): `M = Σ g·gᵀ` over a window, where `g` is the image gradient. Symmetry `Mᵀ = M` is used in corner detection (rank of M determines flat/edge/corner).
- **Normal equations**: `AᵀA` is symmetric (`(AᵀA)ᵀ = Aᵀ(Aᵀ)ᵀ = AᵀA`), which follows from transpose rules.
- **Resize**: `I′ = R·I·Cᵀ` explicitly uses the transpose `Cᵀ`.

**Theorem basis:** Transpose rules (Section 1.5): `(AB)ᵀ = BᵀAᵀ` (T-rules), `(Aᵀ)ᵀ = A`. Symmetric matrices (Section 1.5).

### 1.4 Linear Systems

#### Echelon Form, RREF, and Rank (Sections 2.1, 4.9)

**Concept:** A matrix is in row-echelon form if (1) all zero rows are at the bottom, (2) each leading entry of a row is to the right of the leading entry of the row above, (3) all entries below a leading entry are zero. Reduced row-echelon form (RREF) additionally requires each leading entry to be 1 and the only nonzero entry in its column. **RREF is unique** for a given matrix. The **rank** of a matrix is the number of nonzero rows (equivalently, the number of pivots) in its echelon form.

**Purpose in program:**
- `linalg.rref`: Computes RREF via Gauss-Jordan elimination with partial pivoting. Used as the primitive for `solve`, `inverse`, `rank`, and `nullspace_vector`.
- `linalg.rank`: Returns the number of pivots — used to check that the homography design matrix `A` has rank 8 (full column rank) before solving, and that `rank(AᵀA) = rank(A)`.
- Partial pivoting (choosing the column entry of largest magnitude) controls numerical roundoff, which matters because pixel coordinates can be large (thousands), producing entries of magnitude ~10⁶–10⁷.

**Theorem basis:** [T-rref]: RREF is unique; rank = number of nonzero rows of echelon form (Theorem 2.2, Sections 2.1, 4.9).


#### Consistent and Homogeneous Systems (Ax = b; Mh = 0)

**Concept:** A system `Ax = b` is **consistent** if it has at least one solution; inconsistent otherwise. A **homogeneous** system `Ax = 0` always has at least the trivial solution `x = 0`; it has nontrivial solutions iff the null space is nontrivial (i.e., rank < n).

**Purpose in program:**
- Homography estimation from 4 pairs: the system `Mh = 0` is 8×9 (8 equations, 9 unknowns). Since unknowns (9) exceed equations (8), a nonzero solution exists (T-sol). We then fix `h₃₃ = 1` to convert to `Ah = b` (8×8).
- Least-squares with N ≥ 5 pairs: `Ah = b` is overdetermined (2N > 8) and generally **inconsistent** because human-clicked points contain noise. We solve the normal equations instead (Section 2.4 below).
- Consistency check: `Ax = b` consistent ⇔ rank A = rank [A | b] (T-cons). For noisy data, `rank [A | b] = 9 > 8 = rank A`, confirming inconsistency and justifying least squares.

**Theorem basis:** [T-cons]: `Ax = b` consistent ⇔ b in column space ⇔ rank A = rank [A | b] (Theorem 4.18, Sections 4.4, 4.9). Homogeneous systems and nontrivial solutions (Section 1.1 / 4.7).

#### Null Space and Nullity (Section 4.7)

**Concept:** The null space (kernel) of an m×n matrix A is `{x : Ax = 0}`. The **nullity** is `dim(null space) = n − rank A` (rank-nullity theorem).

**Purpose in program:**
- `linalg.nullspace_vector`: Computes a unit vector in the null space. For the 8×9 homogeneous matrix `M = [A | -b]` (or the design matrix with h₃₃ free), the null space has dimension 1 when rank = 8, giving the homography coefficients up to scale.
- Homography uniqueness: H is unique up to scale exactly when `nullity(M) = 1` (i.e., `rank M = 8`). If nullity > 1, the point configuration is degenerate (e.g., collinear points) and H is not determined.

**Theorem basis:** [T-rn]: `rank A + nullity A = n` (Theorem 4.19, Sections 4.7, 4.9). Nullity 1 ⇒ unique solution up to scale (T-rn combined with T-equiv).

#### Determinants, Cofactor Expansion, and Adjoint Formulas (Sections 3.1, 3.3, 3.4)

**Concept:** The determinant of a square matrix is a scalar that determines invertibility: `det A ≠ 0 ⇔ A` is nonsingular. For 2×2: `det = ad − bc`. For 3×3: cofactor expansion along a row. The adjoint (adjugate) formula gives `A⁻¹ = adj(A) / det A` when `det A ≠ 0`. Cramer's rule solves `Ax = b` via ratios of determinants.

**Purpose in program:**
- `linalg.det`: Closed-form cofactor expansion for 1×1, 2×2 (`ad − bc`), 3×3 (`a(ei−fh) − b(di−fg) + c(dh−eg)`), and Gaussian-elimination product-of-pivots for larger matrices. Used to guard homography singularity (`|det A| ≤ SINGULAR_TOL`) and collinearity checks (`det` of 3×3 `[x y 1]` matrices).
- **Collinearity validation** (`homography._check_collinearity`, `points.are_collinear`): Three points are collinear iff `det([x y 1] rows) = 0`. We reject if `|det| ≤ COLLINEAR_TOL`.
- **Corner order validation** (`points.corner_order_ok`): Consecutive 2×2 edge determinants (`v × w = vₓ·wᵧ − vᵧ·wₓ`) must all share one sign; a zero determinant means a degenerate (collinear) edge.

**Theorem basis:** [T-det]: `det(AB) = det A · det B`, `det Aᵀ = det A`, row-operation effects (Section 3.2). [T-adj]: `det A ≠ 0 ⇔` nonsingular; adjoint formula (Theorem 3.12, Section 3.4). [T-cramer]: Cramer's rule (Theorem 3.13, Section 3.5).

### 1.5 Theorems Cited

| Tag | Theorem number (Kolman & Hill, 9th ed.) | Statement |
|-----|----------------------------------------|-----------|
| [T-rref] | Theorem 2.2 | RREF is unique; rank = nonzero rows of echelon form |
| [T-rules] | Theorems 1.4, 1.6, 1.8 | `(AB)⁻¹ = B⁻¹A⁻¹`, `(Aᵀ)⁻¹ = (A⁻¹)ᵀ`, `(AB)ᵀ = BᵀAᵀ` |
| [T-det] | Lemma 3.1; Theorems 3.1, 3.2, 3.5, 3.6 | `det(AB) = det A · det B`, `det Aᵀ = det A`, row-operation effects |
| [T-adj] | Theorem 3.12 | `det A ≠ 0 ⇔` nonsingular; adjoint formula `A⁻¹ = adj(A)/det A` |
| [T-cramer] | Theorem 3.13 | Cramer's rule for solving linear systems |
| [T-equiv] | Theorem 3.14 | Equivalent statements for n×n A: nonsingular ⇔ det ≠ 0 ⇔ rank n ⇔ row equivalent to I |
| [T-cons] | Theorem 4.18 | `Ax = b` consistent ⇔ b in col space ⇔ rank A = rank [A \| b] |
| [T-rn] | Theorem 4.19 | `rank A + nullity A = n`; null space |
| [T-count] | (to be filled; Section 1.1) | A system has 0, 1, or infinitely many solutions |
| [T-sol] | Theorem 2.4 | Nonzero solution when unknowns exceed equations (8×9 system) |
| [T-dim] | Example 5 (Section 4.6) | 3×3 matrices form a 9-dimensional space |

*(REFERENCES §6 notes that the exact theorem number for T-count and the triangular-determinant theorem remain to be filled in from the book.)*


## 2. Mathematical Formulation

This section defines all variables and symbols, then derives each formula step by step, connecting to the M-item labels from REFERENCES §4.

### 2.1 Variables and Symbols

| Symbol | Meaning |
|--------|---------|
| `(x, y)` | Source image pixel coordinates (column, row), 0-indexed |
| `(u, v)` = `(x′, y′)` | Destination image pixel coordinates |
| `H` | 3×3 homography matrix with entries `h₁₁..h₃₃` |
| `h` | 8-vector `[h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂]ᵀ` (with `h₃₃ = 1`) |
| `w′` | Third homogeneous coordinate: `w′ = h₃₁·x + h₃₂·y + h₃₃` |
| `T_src`, `T_dst` | Hartley normalization 3×3 similarity matrices for source/dest points |
| `H̃` | Homography in normalized coordinates |
| `A` | 2N×8 design matrix (each pair contributes 2 rows) |
| `b` | 2N-vector of destination coordinates |
| `M` | 2N×9 homogeneous matrix `[A \| -b]` (or full 9-unknown form) |
| `N` | Number of point correspondences |
| `s` | Hartley scale `s = √2 / mean_dist` |
| `(cx, cy)` | Centroid of a point set |
| `mean_dist` | Mean distance of points from their centroid |
| `I₀, I₁` | Warped image arrays on the shared canvas |
| `wᵢ(p)` | Feather weight of image i at canvas pixel p (distance to border) |
| `αᵢ(p)` | Normalized blend weight `wᵢ(p) / Σⱼ wⱼ(p)` |
| `gᵢ` | Exposure gain for image i (relative to reference image 0) |


### 2.2 Homography Degrees of Freedom [S1-1 M1]

**Statement:** A 3×3 homography matrix has 9 entries and lives in a 9-dimensional vector space. Since `H` and `cH` (`c ≠ 0`) define the same projective map, the matrix is defined only up to scale. Fixing `h₃₃ = 1` removes one degree of freedom, leaving **8 unknowns**.

**Justification:** 3×3 matrices form a 9-dimensional space [T-dim]. Projective scale invariance removes 1 dimension → 8 effective unknowns. Therefore we need 8 independent scalar equations to solve for `h` uniquely. Each point correspondence `(x, y) → (u, v)` provides exactly 2 equations (one for x, one for y), so 4 pairs give 8 equations for 8 unknowns — a determined system. With N ≥ 5 pairs we get 2N > 8 equations (overdetermined) and use least squares.

**Code tie-in:** `homography.solve_homography` requires exactly 4 pairs; `solve_homography_lsq` requires ≥ 5 pairs. The gauge `h₃₃ = 1` is enforced by appending `1.0` to the 8 solved coefficients: `H_norm = np.append(h8, 1.0).reshape(3, 3)`.


### 2.3 Building the Exact System [S0-1/S1-1 M2, M3]

**Derivation.** Start from the homogeneous mapping:

```
[x′]   [h₁₁ h₁₂ h₁₃] [x]     [x′]
[y′] = [h₂₁ h₂₂ h₂₃] [y]  =  [y′]
[w′ ]   [h₃₁ h₃₂  h₃₃] [1 ]     [w′ ]
```

where `x′/w′ = u`, `y′/w′ = v`, and with `h₃₃ = 1` we have `w′ = h₃₁·x + h₃₂·y + 1`.

From `x′ = u·w′` and `y′ = v·w′`:

```
x·h₁₁ + y·h₁₂ + h₁₃ = u·(h₃₁·x + h₃₂·y + 1)
x·h₂₁ + y·h₂₂ + h₂₃ = v·(h₃₁·x + h₃₂·y + 1)
```

Rearranging to isolate the 8 unknowns on the left:

```
x·h₁₁ + y·h₁₂ + h₁₃ − u·x·h₃₁ − u·y·h₃₂ = u
x·h₂₁ + y·h₂₂ + h₂₃ − v·x·h₃₁ − v·y·h₃₂ = v
```

Each correspondence `(x, y) → (u, v)` contributes exactly **two rows** to the design matrix `A` (2N×8) and two entries to `b` (2N×1):

```
A[2i]   = [x, y, 1, 0, 0, 0, −u·x, −u·y]      b[2i]   = u
A[2i+1] = [0, 0, 0, x, y, 1, −v·x, −v·y]      b[2i+1] = v
```

Stacking 4 pairs (N = 4) gives an 8×8 system `Ah = b`. Stacking the homogeneous form with `h₃₃` free gives an 8×9 system `Mh = 0`.

**Existence of a solution:** The homogeneous system `Mh = 0` has 8 equations and 9 unknowns. Since unknowns exceed equations, a nonzero solution exists [T-sol]. With `h₃₃ = 1` fixed and the 4 points in general position (no three collinear, proper corner order), `A` is invertible and the solution is unique.

**Code tie-in:** `homography.build_system(src_n, dst_n)` implements exactly these two rows per pair. `solve_homography` calls it with N = 4 and solves `linalg.solve(A, b)`; `solve_homography_lsq` calls it with N ≥ 5.


### 2.4 System Solvability [S1-1 M4]

**Statement:** The 8×8 matrix `A` is invertible exactly when:
1. `rank(A) = 8` (full column rank), and
2. the null vector of the homogeneous 8×9 system has `h₃₃ ≠ 0` (so the gauge `h₃₃ = 1` is valid).

**Justification:** By [T-equiv], `A` is nonsingular ⇔ `det A ≠ 0` ⇔ `rank A = 8` ⇔ `A` row-equivalent to `I`. If `rank(A) < 8`, the point configuration is degenerate (e.g., three collinear points, or all points on a line) and `A` is singular — no unique homography exists. The additional condition `h₃₃ ≠ 0` ensures the homogeneous solution can be scaled so that `h₃₃ = 1`; if `h₃₃ = 0` the homography is a pure affine/projective map at infinity and cannot be normalized this way.

**Code tie-in:** `solve_homography` checks `abs(linalg.det(A)) <= SINGULAR_TOL` and raises `HomographyError` with rank diagnostics if singular. `solve_homography_lsq` checks `linalg.rank(A) < 8` and `linalg.rank(A.T @ A) < 8` before solving. Both check `abs(H[2, 2]) <= H33_TOL` after denormalization.


### 2.5 Normalization [S0-1 M6]

**Why T is invertible [T-equiv]:** The Hartley normalization matrix

```
T = [[s, 0, −s·cx],
     [0, s, −s·cy],
     [0, 0, 1   ]]
```

is upper triangular with diagonal entries `s, s, 1`. Its determinant is `det T = s·s·1 = s²`. Since points are not coincident, `mean_dist > 0`, so `s = √2/mean_dist > 0`, hence `det T = s² ≠ 0`. By [T-equiv] (`det ≠ 0 ⇔` nonsingular), `T` is invertible, with `T⁻¹ = [[1/s, 0, cx], [0, 1/s, cy], [0, 0, 1]]`.

**Denormalization [T-rules]:** We compute the homography `H̃` in normalized coordinates (where points have centroid 0 and mean distance √2), then map back to original coordinates:

```
H = T_dst⁻¹ · H̃ · T_src
```

This follows from [T-rules]: `(AB)⁻¹ = B⁻¹A⁻¹`. If `p̃_dst = H̃ · p̃_src` and `p̃_src = T_src · p_src`, `p̃_dst = T_dst · p_dst`, then:

```
T_dst · p_dst = H̃ · T_src · p_src
p_dst = T_dst⁻¹ · H̃ · T_src · p_src
```

so `H = T_dst⁻¹ · H̃ · T_src`.

**Why normalize:** Raw pixel coordinates (e.g., x = 3000) produce products like `x·u ≈ 10⁷` in `A`, causing floating-point roundoff and rank-deficient matrices. Centering at the origin and scaling to mean distance √2 keeps entries O(1), making `A` well-conditioned.

**Code tie-in:** `homography.normalize_points` returns `(norm_pts, T)`. `solve_homography` and `solve_homography_lsq` both normalize, solve in normalized space, then denormalize via `H = linalg.inverse(T_dst) @ H_norm @ T_src`.


### 2.6 Least-Squares Overdetermined System [S2-1 M1–M5]

**Problem:** For N ≥ 5 pairs, `Ah = b` has 2N > 8 equations and is generally **inconsistent** because clicked points contain noise: `b` is not exactly in the column space of `A` [T-cons]. Indeed `rank [A | b] = 9 > 8 = rank A` for noisy data.

**Derivation of normal equations:** We seek `h` minimizing the squared algebraic residual `‖Ah − b‖² = (Ah − b)ᵀ(Ah − b)`. Expanding:

```
‖Ah − b‖² = hᵀAᵀAh − 2bᵀAh + bᵀb
```

Setting the gradient to zero (the normal equations):

```
AᵀA h = Aᵀ b
```

`AᵀA` is 8×8 and symmetric. It is nonsingular iff `rank(A) = 8`:

**Proof:** By [T-rules] and [T-rn], `rank(AᵀA) = rank(A)`. (Intuitively, `AᵀA` and `A` share the same null space: `AᵀAh = 0` implies `hᵀAᵀAh = ‖Ah‖² = 0`, so `Ah = 0`; hence `null(AᵀA) = null(A)`.) Therefore `rank(AᵀA) = rank(A) = 8`, so `AᵀA` is invertible [T-equiv], and `h = (AᵀA)⁻¹ Aᵀ b` is the unique least-squares minimizer.

**Code tie-in:** `solve_homography_lsq` builds `A, b`, checks `rank(A) >= 8`, forms `normal_matrix = A.T @ A`, checks `rank(normal_matrix) >= 8`, then `h8 = linalg.solve(normal_matrix, A.T @ b)`. `algebraic_residual` computes `‖Ah − b‖²` directly.


### 2.7 Inverse Warping [S1-2 M8, M9]

**Mechanics of w′ division:** Forward mapping (each source pixel → canvas) leaves holes because multiple source pixels can map to the same canvas pixel and some canvas pixels receive none. Instead we use **backward (inverse) mapping**: for every integer canvas pixel `(u, v)`, compute the world point `p = (u + ox, v + oy)`, then map back to the source via `H⁻¹`:

```
[sx]     [h′₁₁ h′₁₂ h′₁₃] [u + ox]
[sy ] =  [h′₂₁ h′₂₂ h′₂₃] [v + oy]
[w′ ]    [h′₃₁ h′₃₂ h′₃₃] [1     ]
source coords = (sx/w′, sy/w′)
```

where `H′ = H⁻¹`. We sample the source at nearest integer `(round(sx), round(sy))` if inside bounds; otherwise mark the canvas pixel invalid.

**Proof lines stay lines [T-rules]:** A line in homogeneous coordinates is `lᵀp = 0`. Under `p′ = Hp`, the line transforms as `l′ = H⁻ᵀl` (since `(H⁻ᵀl)ᵀ(Hp) = lᵀH⁻¹Hp = lᵀp = 0`). By [T-rules], `(H⁻¹)ᵀ = (Hᵀ)⁻¹`. Thus straight lines map to straight lines — essential for rectifying a rectangular object into a rectangle and for keeping board edges straight in panoramas.

**Horizon safety:** The denominator `w = h₃₁·x + h₃₂·y + h₃₃` is linear in `(x, y)`, so its extrema over a rectangular image occur at the four corners [M8]. If `w` is zero or changes sign among corners, the horizon (where `w = 0`) touches or crosses the image, making perspective division unsafe. `check_h` rejects such homographies.

**Code tie-in:** `warp.warp_image` computes `sp = pts @ H_inv.T` (equiv. `H⁻¹·p` row-wise), divides by `w`, clips source coords, and uses `floor(s + 0.5)` for nearest-neighbor sampling. `warp.check_h` verifies `H` is invertible (via `linalg.inverse`) and that `w` has one strict sign over all corners.


### 2.8 Determinant Validation [S1-3 M7]

**Collinearity via 3×3 determinant:** Three points `(x₁,y₁)`, `(x₂,y₂)`, `(x₃,y₃)` are collinear iff

```
det [x₁ y₁ 1] = 0
    [x₂ y₂ 1]
    [x₃ y₃ 1]
```

Geometrically this determinant is twice the signed area of the triangle; zero area means the points lie on a line. By [T-equiv], a zero determinant means the rows `[x y 1]` are linearly dependent, i.e., the points are collinear.

**Corner order:** For four points in order p₀,p₁,p₂,p₃, consecutive edge vectors `vᵢ = pᵢ₊₁ − pᵢ` and `wᵢ = pᵢ₊₂ − pᵢ₊₁` have 2×2 cross-product `det2 = vₓ·wᵧ − vᵧ·wₓ`. All four `det2` values must share one strict sign (all positive = clockwise, all negative = counterclockwise on screen with y down). A zero or mixed sign indicates a self-intersecting (bow-tie) or degenerate quad.

**Code tie-in:** `homography._check_collinearity` computes `linalg.det` of each 3×3 `[x y 1]` triple and rejects if `|det| <= COLLINEAR_TOL (1e-6)`. `points.corner_order_ok` computes the four 2×2 determinants and requires consistent sign (tolerance `CORNER_ORDER_TOL = 1e-12`).

### 2.9 Connection Between Math and CS

Matrix multiplication `p′ = H·p` directly implements a pixel transformation: each output pixel's source location is a linear combination of the input homogeneous coordinates, followed by perspective division by `w`. This is why:
- Translation is `h₁₃, h₂₃` (adds constants after the linear part).
- Rotation/scale/shear live in the upper-left 2×2 block.
- Perspective effects live in `h₃₁, h₃₂` (they make `w` vary across the image, producing vanishing points).

In code, this becomes NumPy broadcasting: `src_h @ H.T` applies `H` to N points simultaneously (N×3 times 3×3 → N×3), vectorized over the whole point set, and `pts @ H_inv.T` does the same for inverse warping across entire canvas tiles.


## 3. Implementation Plan

### 3.1 Data Representation

| Entity | Representation | Notes |
|--------|---------------|-------|
| Point `(x, y)` | `float64` scalar pair | Stored as rows of an `(N, 2)` array; column 0 = x, column 1 = y |
| Point set | `(N, 2)` `np.ndarray` of `float64` | All homography functions accept `(N, 2)` arrays |
| Homogeneous point | `(x, y, 1)` | Appended via `np.hstack([pts, ones((N,1))])` before multiplying by H |
| Homography `H` | `(3, 3)` `np.ndarray` of `float64` | Stored with `h₃₃ = 1` gauge (`H /= H[2, 2]` after computation) |
| Design matrix `A` | `(2N, 8)` `np.ndarray` of `float64` | 2 rows per correspondence |
| RHS vector `b` | `(2N,)` `np.ndarray` of `float64` | Destination coordinates interleaved u, v |
| Image | `uint8` H×W or H×W×3 `np.ndarray` | Only `io_utils` imports Pillow; everything else is pure NumPy |

### 3.2 Algorithmic Steps (Pipeline)

```
1. Point extraction
   ├─ Load JSON points → (N, 2) float arrays  (points.load_points)
   ├─ Validate corners: clockwise order, no three collinear  (points.corner_order_ok, homography._check_collinearity)
   └─ Validate pairs: matching counts, finite values

2. Normalization  (homography.normalize_points)
   ├─ centroid (cx, cy) = mean of points
   ├─ shift points to origin, compute mean_dist
   ├─ scale s = √2 / mean_dist
   ├─ build T = [[s,0,-s·cx],[0,s,-s·cy],[0,0,1]]
   └─ return normalized points and T

3. Solve for H
   ├─ If N == 4: exact solve  (homography.solve_homography)
   │   ├─ build_system → A (8×8), b (8×1)
   │   ├─ guard: |det A| > SINGULAR_TOL
   │   ├─ h8 = linalg.solve(A, b)     (Gauss-Jordan RREF)
   │   ├─ H_norm = reshape([h8, 1], (3,3))
   │   ├─ denormalize: H = T_dst⁻¹ · H_norm · T_src   ([T-rules])
   │   └─ normalize H /= H[2,2]; horizon check
   └─ If N >= 5: least squares  (homography.solve_homography_lsq)
       ├─ build_system → A (2N×8), b (2N×1)
       ├─ guard: rank(A) == 8
       ├─ normal equations: (AᵀA) h = Aᵀ b
       ├─ h8 = linalg.solve(AᵀA, Aᵀ b)
       ├─ denormalize + h₃₃ guard
       └─ return H

4. Canvas setup  (warp.canvas_bounds)
   ├─ Validate each H with check_h (invertible + horizon sign)
   ├─ Map 4 corners of each image through H
   ├─ Union min/max → canvas shape; offset = mins
   └─ Cap canvas size (MAX_CANVAS_SIDE, MAX_CANVAS_PIXELS)

5. Inverse warp  (warp.warp_image)
   ├─ H_inv = linalg.inverse(H)
   ├─ For each row tile: build canvas grid, sp = grid @ H_inv.T
   ├─ divide by w, nearest-neighbor floor(s + 0.5)
   ├─ mark valid where inside source bounds
   └─ return (warped image, valid mask, src_x, src_y)

6. Blending  (blend.edge_weights, blend.feather_blend)
   ├─ edge_weights: w = min distance to 4 borders, clipped ≥ 0, zero outside valid
   ├─ hard cut: binary choice by higher weight (demo.py)
   └─ feather_blend: αᵢ = wᵢ / Σw; output = Σ αᵢ·Iᵢ; fallback to first valid at zero-weight borders
       ├─ optional exposure gain gᵢ = <I₀,Iᵢ>/<Iᵢ,Iᵢ>  (_estimate_gains)
       └─ tiled processing for memory boundedness

7. Save  (io_utils.save_png)
```

### 3.3 Testing Plan

Validation is handled by `tests.py` (51 unit tests). Key mathematical coverage:

| Test class | Coverage | Math basis |
|------------|----------|------------|
| `TestLinalgAcceptance` (V0.1–V0.3) | RREF row-order independence; solve/inverse match NumPy; linear algebra identities | [T-rref], [T-equiv], [T-inv], [T-rules], [T-det], [T-adj], [T-cramer], [T-cons] |
| `TestHomographyExact` (V0.4–V0.6) | 100-trial random homography; large-coordinate identity; collinear/swapped/horizon failures | [T-dim], [T-sol], [T-rn], [T-equiv], determinant collinearity |
| `TestLeastSquaresHomography` (V1.1–V1.6) | Noise-free 8/12-point LSQ; normal-equation optimality vs 4-point; rank(AᵀA)=rank(A); collinear raises | [T-cons], [T-rules], [T-rn], [T-equiv], normal equations |
| `TestBlendV2` (V2.1–V2.7) | Feather weights, convex combination, hard-cut fallback, exposure gain, seam quality | Vector-space linear combinations (Sections 4.2, 4.4) |
| `TestNpLinalgBan` | np.linalg only in linalg.py and tests.py | Implementation constraint (hand-written linear algebra) |

**Validation approach:** Each theorem from Section 1.5 has at least one corresponding test. For example, V0.3 checks rank+nullity and Cramer's-rule-style identities; V0.1/V0.2 check matrix uniqueness (RREF uniqueness, solve/inverse correctness); V1.1–V1.6 check LSQ correctness and rank properties.


## 4. Code Implementation

Below are the key code blocks with comments connecting each to its mathematical operation, purpose, and expected interpretation.

### 4.1 `linalg.py` — Gauss-Jordan elimination core

**Purpose:** Hand-written linear algebra engine (no `np.linalg`) providing RREF, solve, inverse, determinant, rank, and null-space vector.

**Mathematical operation:** Gauss-Jordan elimination with partial pivoting on augmented matrices; produces RREF [T-rref]. All other operations are built on RREF.

**Expected result:** A numerically robust, tolerance-gated linear algebra primitive. `solve` returns the unique solution when `A` is invertible [T-equiv]; `inverse` returns `A⁻¹` via `[A \| I]` reduction; `det` returns the determinant via cofactors (≤3×3) or pivot product; `rank` returns pivot count; `nullspace_vector` returns a unit null-space vector [T-rn].

```python
def _rref_with_pivots(A, tol):
    """Reduced row-echelon form with partial pivoting [T-rref]."""
    work = _as_matrix(A)
    rows, cols = work.shape
    scale = max(1.0, float(np.max(np.abs(work))) if work.size else 0.0)
    threshold = tol * scale
    pivot_cols = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row == rows:
            break
        # Partial pivot: choose row with largest |entry| in this column
        pivot = pivot_row + int(np.argmax(np.abs(work[pivot_row:, col])))
        if abs(work[pivot, col]) <= threshold:
            continue  # column is numerically zero → skip (free variable)

        if pivot != pivot_row:
            work[[pivot_row, pivot]] = work[[pivot, pivot_row]]  # row swap

        work[pivot_row] /= work[pivot_row, col]   # normalize pivot to 1
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row, col]
            if factor != 0.0:
                work[row] -= factor * work[pivot_row]  # eliminate column
        work[np.abs(work) <= threshold] = 0.0  # snap near-zeros
        pivot_cols.append(col)
        pivot_row += 1

    work[np.abs(work) <= threshold] = 0.0
    return work, pivot_cols
```

**Interpretation:** Partial pivoting controls roundoff when pixel coordinates are large (~10³ → products ~10⁷). The tolerance `threshold = tol * scale` is relative to the matrix magnitude, so near-zero entries are snapped consistently. The number of `pivot_cols` is the `rank` [T-rref].


### 4.2 `linalg.py` — Solve and inverse

**Purpose:** Solve square systems `Ax = b` and compute `A⁻¹`.

**Mathematical operation:** Augment `[A \| b]` (or `[A \| I]`) and reduce to RREF; the solution/inverse is read from the last block. Unique solution exists iff `A` is nonsingular [T-equiv, T-inv].

**Expected result:** `x = A⁻¹b` for invertible `A`; raises `SingularMatrixError` otherwise.

```python
def solve(A, b, tol=DEFAULT_TOL):
    """Solve square system with unique solution via RREF of [A | b] [T-equiv, T-inv]."""
    matrix = _as_matrix(A, "A")
    rhs = np.asarray(b, dtype=float).reshape(-1, 1)
    reduced, pivots = _rref_with_pivots(np.hstack((matrix, rhs)), tol)
    size = matrix.shape[0]
    # Unique solution iff all n pivots are exactly columns 0..n-1
    if len(pivots) != size or pivots[:size] != list(range(size)):
        raise SingularMatrixError("A is singular; no unique solution")
    return reduced[:size, -1].copy()

def inverse(A, tol=DEFAULT_TOL):
    """Compute A⁻¹ by reducing [A | I] [T-inv, T-equiv]."""
    matrix = _as_matrix(A, "A")
    size = matrix.shape[0]
    augmented = np.hstack((matrix, np.eye(size)))
    reduced, pivots = _rref_with_pivots(augmented, tol)
    if pivots[:size] != list(range(size)) or len(pivots) < size:
        raise SingularMatrixError("A is singular and has no inverse")
    return reduced[:size, size:]
```

**Interpretation:** The pivot check `pivots[:size] == list(range(size))` guarantees every column is a pivot column, i.e., `rank(A) = n`, i.e., `A` is invertible [T-equiv]. This is the same guard used for the homography 8×8 system.

### 4.3 `linalg.py` — Determinant

**Purpose:** Compute determinants for singularity guards and collinearity tests.

**Mathematical operation:** Cofactor expansion for 1×1/2×2/3×3; Gaussian-elimination pivot product for larger [T-det, T-adj].

**Expected result:** `det(A)`; zero (within tolerance) indicates singularity.

```python
def det(A, tol=DEFAULT_TOL):
    """Determinant: closed-form cofactors for small matrices, elimination otherwise [T-det, T-adj]."""
    matrix = _as_matrix(A, "A")
    size = matrix.shape[0]
    if size == 1:
        return float(matrix[0, 0])
    if size == 2:
        return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if size == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return float(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
    # Larger: elimination, tracking sign from row swaps
    work = matrix.copy()
    sign = 1.0
    determinant = 1.0
    for col in range(size):
        pivot = col + int(np.argmax(np.abs(work[col:, col])))
        if abs(work[pivot, col]) <= threshold:
            return 0.0  # singular column
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
            sign = -sign  # row swap flips determinant sign [T-det]
        pivot_value = work[col, col]
        determinant *= pivot_value
        for row in range(col + 1, size):
            factor = work[row, col] / pivot_value
            work[row, col + 1:] -= factor * work[col, col + 1:]
    return float(sign * determinant)
```

**Interpretation:** The 3×3 closed form `a(ei−fh) − b(di−fg) + c(dh−eg)` is the cofactor expansion along the first row [T-adj]. Row swaps flip the sign [T-det]. A zero pivot below tolerance means the determinant is zero → matrix is singular [T-equiv].


### 4.4 `homography.py` — Hartley normalization

**Purpose:** Center points at the origin and scale to mean distance √2; build the similarity transform `T`.

**Mathematical operation:** `T = [[s,0,−s·cx],[0,s,−s·cy],[0,0,1]]` where `s = √2/mean_dist` [S0-1 M6]. `T` is upper triangular with `det T = s² ≠ 0` → invertible [T-equiv].

**Expected result:** Normalized points with centroid 0 and mean distance √2, plus the matrix `T` for later denormalization.

```python
def normalize_points(pts):
    """Hartley normalization: zero centroid, mean distance √2 [S0-1 M6]."""
    pts = np.asarray(pts, dtype=float)
    cx, cy = np.mean(pts, axis=0)               # centroid
    shifted = pts - [cx, cy]
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    if mean_dist < MEAN_DIST_EPS:
        raise HomographyError("Points are coincident")
    s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s * cx],
                  [0, s, -s * cy],
                  [0, 0, 1]], dtype=float)      # upper triangular, det = s² ≠ 0
    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    norm_pts = (T @ pts_homo.T).T[:, :2]        # p̃ = T · [x y 1]ᵀ
    return norm_pts, T
```

**Interpretation:** Centering removes large coordinate offsets; scaling to mean distance √2 keeps design-matrix entries O(1), preventing the `10⁷`-scale roundoff that would make `A` rank-deficient [T-equiv]. `T` is invertible because `s > 0` (points not coincident), so denormalization `H = T_dst⁻¹·H̃·T_src` is valid [T-rules].

### 4.5 `homography.py` — Build system

**Purpose:** Construct the 2N×8 system `Ah = b` from point correspondences.

**Mathematical operation:** Two rows per pair from cross-multiplied homogeneous equations [S0-1/S1-1 M2, M3]:

```
[x, y, 1, 0, 0, 0, −u·x, −u·y] · h = u
[0, 0, 0, x, y, 1, −v·x, −v·y] · h = v
```

**Expected result:** `A` of shape `(2N, 8)` and `b` of shape `(2N,)`; solving gives `h = [h₁₁..h₃₂]ᵀ` with `h₃₃ = 1`.

```python
def build_system(src_n, dst_n):
    """Build 2N×8 system Ah = b for homography unknowns (h₃₃ = 1) [S0-1/S1-1 M2, M3]."""
    N = len(src_n)
    A = np.zeros((2 * N, 8), dtype=float)
    b = np.zeros(2 * N, dtype=float)
    for i in range(N):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i]   = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        b[2 * i]   = u
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i + 1] = v
    return A, b
```

**Interpretation:** Each pair contributes 2 equations. With N = 4 we get 8×8 (determined); with N ≥ 5 we get 2N×8 (overdetermined → least squares). The `−u·x, −u·y` terms come from moving `w′`-dependent terms to the left side.


### 4.6 `homography.py` — Exact 4-point solver

**Purpose:** Compute `H` from exactly 4 point pairs.

**Mathematical operation:** Normalize → check collinearity (3×3 det = 0) [T-equiv] → check corner order (2×2 edge determinants) → build `A, b` → solve `linalg.solve(A, b)` → denormalize `H = T_dst⁻¹·H̃·T_src` [T-rules] → normalize `H /= H[2,2]` → horizon check [M8].

**Expected result:** A 3×3 `H` mapping `src` to `dst`, with `H[2,2] = 1`, valid over the source image (no horizon crossing).

```python
def solve_homography(src, dst):
    """Exact 4-point homography [S1-1 M1–M4]."""
    # 1. Validate shape and finiteness
    # 2. Normalize both sets [S0-1 M6]
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)
    # 3. Collinearity: det of each [x y 1] triple must be nonzero [T-equiv]
    _check_collinearity(src_norm); _check_collinearity(dst_norm)
    # 4. Corner order: consecutive 2×2 edge determinants share one sign
    _check_corner_order(src); _check_corner_order(dst)
    # 5. Build system
    A, b = build_system(src_norm, dst_norm)
    # 6. Guard: |det A| > SINGULAR_TOL  [T-equiv]
    if abs(linalg.det(A)) <= SINGULAR_TOL:
        raise HomographyError("degenerate system: A is singular")
    h8 = linalg.solve(A, b)
    H_norm = np.append(h8, 1.0).reshape(3, 3)
    # 7. Denormalize [T-rules]
    H = linalg.inverse(T_dst) @ H_norm @ T_src
    if abs(H[2, 2]) <= H33_TOL:
        raise HomographyError("h₃₃ near zero")
    H = H / H[2, 2]
    # 8. Horizon check [M8]
    _check_horizon(H, src)
    return H
```

**Interpretation:** Steps 3–4 use determinants to reject degenerate point configurations [T-equiv]. Step 6 uses `det A` as the invertibility guard [T-equiv, T-adj]. Step 7 applies [T-rules] `(AB)⁻¹ = B⁻¹A⁻¹` implicitly through the order `T_dst⁻¹ · H̃ · T_src`.

### 4.7 `homography.py` — Least-squares solver

**Purpose:** Estimate `H` from N ≥ 5 noisy point pairs.

**Mathematical operation:** Normalize → rank guard → form normal equations `(AᵀA)h = Aᵀb` → solve → denormalize [S2-1 M1–M5]. `AᵀA` is nonsingular iff `rank(A) = 8` [T-rules, T-rn, T-equiv].

**Expected result:** The least-squares homography minimizing `‖Ah − b‖²`.

```python
def solve_homography_lsq(src, dst):
    """Least-squares homography via normal equations [S2-1 M1–M5]."""
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)
    A, b = build_system(src_norm, dst_norm)
    if linalg.rank(A) < 8:
        raise HomographyError("rank-deficient: homography not determined")
    normal_matrix = A.T @ A                       # symmetric 8×8 [T-rules]
    if linalg.rank(normal_matrix) < 8:
        raise HomographyError("rank-deficient normal equations")
    h8 = linalg.solve(normal_matrix, A.T @ b)     # (AᵀA)⁻¹Aᵀb
    H_norm = np.append(h8, 1.0).reshape(3, 3)
    H = linalg.inverse(T_dst) @ H_norm @ T_src    # denormalize [T-rules]
    if abs(H[2, 2]) <= H33_TOL:
        raise HomographyError("h₃₃ near zero")
    H = H / H[2, 2]
    return H
```

**Interpretation:** `A.T @ A` is symmetric by `(AᵀA)ᵀ = AᵀA` [T-rules]. `rank(AᵀA) = rank(A)` [T-rn, T-rules], so the rank guard on `A` implies the normal matrix is invertible [T-equiv]. The solution `h = (AᵀA)⁻¹Aᵀb` is the unique least-squares minimizer of `‖Ah − b‖²` [T-cons justifies why exact solve is impossible for noisy data].


### 4.8 `homography.py` — Error metrics

**Purpose:** Quantify homography quality.

**Mathematical operation:**
- `transfer_error`: project `src` through `H`, divide by `w`, compute Euclidean distance to `dst` per pair [M8].
- `algebraic_residual`: compute `‖Ah − b‖²` in normalized coordinates [S2-1 M1–M5].

**Expected result:** Per-pair pixel errors (transfer error) and a scalar squared residual (algebraic residual).

```python
def transfer_error(H, src, dst):
    """Pixel transfer error per correspondence [M8]."""
    src_h = np.hstack([src, np.ones((len(src), 1))])
    dst_proj = (H @ src_h.T).T                  # [x′, y′, w′]ᵀ = H·[x, y, 1]ᵀ
    w = dst_proj[:, 2:3]
    w_safe = np.where(np.abs(w) > HORIZON_TOL, w, 1.0)
    dst_mapped = dst_proj[:, :2] / w_safe       # perspective division
    return np.sqrt(np.sum((dst - dst_mapped)**2, axis=1))
```

**Interpretation:** Dividing by `w′` converts homogeneous back to Euclidean [homogeneous coordinates]. The Euclidean distance in pixels is the direct perceptual measure of homography accuracy.

### 4.9 `warp.py` — Inverse warp

**Purpose:** Map every canvas pixel back to the source via `H⁻¹` (backward mapping).

**Mathematical operation:** For canvas pixel `(u, v)`, world point `p = (u+ox, v+oy)`; `sp = H⁻¹·p`; source = `(sx/w′, sy/w′)`; nearest-neighbor `floor(s + 0.5)` [S1-2 M8, M9]. Lines stay lines because `l′ = H⁻ᵀl` [T-rules].

**Expected result:** Warped image on canvas, boolean valid mask, and source coordinate maps.

```python
def warp_image(img, H, canvas_shape, offset):
    """Inverse-warp img onto canvas via H⁻¹ with nearest-neighbor sampling [S1-2 M8, M9]."""
    H_inv = linalg.inverse(H)                   # existence checked by check_h
    ...
    for r0 in range(0, cH, rows_per_tile):
        gx, gy = np.meshgrid(arange(cW) + ox, arange(r0, r1) + oy)
        pts = stack([gx, gy, ones])             # homogeneous canvas points
        sp = pts @ H_inv.T                       # sp = H⁻¹ · p  (row-wise)
        w = sp[:, 2]
        good_w = isfinite(w) & (abs(w) > W_EPS)
        sx = sp[:, 0] / w_safe                   # perspective division x′/w′
        sy = sp[:, 1] / w_safe                   # perspective division y′/w′
        nx = floor(sx + 0.5); ny = floor(sy + 0.5)   # nearest neighbor
        band_valid = good_w & (-0.5 < sx < w_src−0.5) & (-0.5 < sy < h_src−0.5)
        out[tile][valid] = img[ny[valid], nx[valid]]
    return out, valid, src_x, src_y
```

**Interpretation:** Backward mapping fills every canvas pixel (no holes) [README §4D]. The `w′` division creates perspective effects. Nearest-neighbor sampling via `floor(s + 0.5)` is the simplest reconstruction; valid mask tracks which canvas pixels actually source from inside the image.

### 4.10 `warp.py` — Canvas bounds and horizon check

**Purpose:** Compute the union canvas for multiple warped images and validate homographies.

**Mathematical operation:** Map the 4 corners of each image through `H`; take union min/max. `check_h` verifies invertibility via `linalg.inverse` and that `w = h₃₁x+h₃₂y+h₃₃` has one strict sign over corners (horizon does not cross) [M8, T-rules].

**Expected result:** `(canvas_shape, offset)` or a `HomographyError` if unsafe.

```python
def check_h(H, shape):
    """Reject H if singular or horizon touches/crosses the image [M8]."""
    H_inv = linalg.inverse(H)                   # invertibility test [T-equiv]
    _, w = _map_points(H, _corners(shape))      # w = h₃₁x + h₃₂y + h₃₃ at corners
    if any(abs(w) <= W_EPS):
        raise HomographyError("horizon touches boundary")
    if any(w > 0) and any(w < 0):
        raise HomographyError("horizon crosses image: mixed-sign w")
```

**Interpretation:** Because `w` is linear in `(x, y)`, its extrema on a rectangle occur at corners — so checking corners suffices [M8]. Mixed signs mean `w = 0` somewhere inside (horizon crosses), making perspective division undefined for some pixels.

