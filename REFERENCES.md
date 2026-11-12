# Paper Reference: Definitions, Theorems, and Mathematical Equivalences

Linear Algebra course project · Panorama stitching with homography matrices · Reference: Kolman & Hill (2014), *Elementary Linear Algebra with Applications*, 9th ed.

Scope: only the definitions and theorems the paper cites. No proofs are given here. Tasks, owners, schedule, interface, and verification live in the GitHub issues.

## 1. Theorem table

| Tag | Result used | Section | Book reference |
| --- | --- | --- | --- |
| T-count | A system has 0, 1, or infinitely many solutions | 1.1 | Thm __ (number to fill in) |
| T-sol | Nonzero solution when unknowns exceed equations (8×9 system) | 2.2 | Theorem 2.4 |
| T-rref | RREF is unique; rank = nonzero rows of echelon form | 2.1, 4.9 | Theorem 2.2 |
| T-inv | A nonsingular ⇔ row equivalent to I; computing A⁻¹ | 2.3 | Theorem 2.8, Corollary 2.2 |
| T-rules | (AB)⁻¹ = B⁻¹A⁻¹, (Aᵀ)⁻¹ = (A⁻¹)ᵀ, (AB)ᵀ = BᵀAᵀ | 1.4, 1.5 | Theorem 1.4, Theorem 1.6, Theorem 1.8 |
| T-det | det(AB) = det A · det B, det Aᵀ = det A, row-operation effects | 3.2 | Lemma 3.1, Theorem 3.1, Theorem 3.2, Theorem 3.5, Theorem 3.6 |
| T-adj | det A ≠ 0 ⇔ nonsingular; adjoint formula | 3.4 | Theorem 3.12 |
| T-cramer | Cramer’s rule | 3.5 | Theorem 3.13 |
| T-equiv | Equivalent statements for n×n A (chapter 3 summary, extended in Ch 4) | 3.6, 4.9 | Theorem 3.14 |
| T-cons | Ax = b consistent ⇔ b in the column space ⇔ rank A = rank [A b] | 4.4, 4.9 | Theorem 4.18 |
| T-rn | rank A + nullity A = n; null space | 4.7, 4.9 | Theorem 4.19 |
| T-dim | 3×3 matrices have dimension 9 | 4.6 | Example 5 |

Where a row groups several theorems, cite the group until the exact one-to-one assignment is checked in the book.

## 2. Definitions used in the paper

### 2.1 From the book

| Term | As used in the project | Section |
| --- | --- | --- |
| Matrix product | H applied to [x y 1]ᵀ; image resize I′ = R·I·Cᵀ | 1.3 |
| Transpose; symmetric matrix | Mᵀ = M for the structure tensor | 1.4, 1.5 |
| Triangular matrix | The normalization matrix T | 1.5 |
| Nonsingular matrix; inverse | A⁻¹ with A·A⁻¹ = I | 1.5, 2.3 |
| Permutation matrix | Image flips | section to confirm |
| Echelon form; RREF; rank | Rank = number of nonzero rows of the echelon form | 2.1, 4.9 |
| Elementary matrix | Each row operation is left multiplication by one | 2.3 |
| Consistent system; homogeneous system | Ax = b; Mh = 0 | 1.1, 4.7 |
| Null space; nullity | Solution set of Mh = 0 | 4.7 |
| Determinant; cofactor expansion; adjoint | 2×2 and 3×3 determinants; adjoint formula for the inverse | 3.1, 3.3, 3.4 |
| Vector-space operations | Sums and scalar multiples of images as vectors | 4.2 |
| Linear combination; span; column space | span{[x, y, 1]ᵀ}; b in the column space of A | 4.4 |
| Linear independence and dependence | Rows [x y 1] of three points | 4.5 |
| Basis; dimension | Dimension 9 for 3×3 matrices | 4.6 |

### 2.2 Project terms built from them

| Term | Definition |
| --- | --- |
| Homogeneous coordinates | The point (x, y) is written [x, y, 1]ᵀ and stands for span{[x, y, 1]ᵀ} in ℝ³; the output point is (x′/w, y′/w). H and cH (c ≠ 0) give the same map. |
| Homography H | A 3×3 matrix mapping one image plane onto another, defined up to scale; fixing h33 = 1 leaves 8 unknowns. |
| w | The third homogeneous coordinate, w = h31·x + h32·y + h33. |
| System forms | Mh = 0 (2N × 9), or Ah = b (2N × 8) once h33 = 1. |
| Normalization matrix T | Translates the centroid to the origin and scales; triangular with det T = s². |
| Algebraic residual | The minimized quantity ‖Ah − b‖², in normalized coordinates. |
| Transfer error | Pixel distance between H·(x, y) and the true (x′, y′). |
| Structure tensor | M = Σ g·gᵀ over a window, with g the image gradient. |
| Patch vector; SSD | An 11×11 patch as a vector in ℝ¹²¹; SSD = (p − q)·(p − q). |
| Feather weights | Weights from distance to the image border; output = Σ αᵢ·Iᵢ with αᵢ = wᵢ / Σ wⱼ. |

## 3. Mathematical equivalences

Each row pairs a project statement with the book result it rests on.

| Project statement | Book basis | Tag |
| --- | --- | --- |
| The 8×9 homogeneous system from 4 pairs has a nonzero solution | Unknowns exceed equations | T-sol |
| The solution set of Mh = 0 is the null space, of dimension 9 − rank M | rank + nullity = n | T-rn |
| The solution count is one point or infinitely many | 0, 1, or infinitely many solutions | T-count |
| H is unique up to scale exactly when rank M = 8 | Nullity 1 | T-rn |
| 3×3 matrices form a 9-dimensional space; removing scale leaves 8 unknowns | Dimension 9 | T-dim |
| The 8×8 matrix A is invertible ⇔ Ax = b has a unique solution for every b ⇔ A is row equivalent to I ⇔ det A ≠ 0 | Equivalent statements; inverse; determinant | T-equiv, T-inv, T-adj |
| A is invertible ⇔ rank M = 8 and the null vector has h33 ≠ 0 | The two rows above combined | T-rn, T-equiv |
| Gauss-Jordan on [A b] solves the system, independent of pivot order | RREF is unique | T-rref |
| Rank is the number of nonzero rows of the echelon form | RREF and rank | T-rref |
| Reducing [A I] to [I A⁻¹] gives the inverse | Computing A⁻¹ | T-inv |
| Each row operation is left multiplication by an elementary matrix | Elementary matrices | Section 2.3 |
| T is triangular with det T = s² ≠ 0, so it is invertible | Nonsingular ⇔ det ≠ 0 | T-equiv |
| H = T′⁻¹·H̃·T, and Hc = T·H gives Hc⁻¹ = H⁻¹·T⁻¹ | Product rule for inverses | T-rules |
| A line lᵀp = 0 maps to the line (H⁻ᵀl)ᵀp′ = 0 under p′ = Hp | Inverse and transpose rules | T-rules |
| Three points are collinear ⇔ the 3×3 determinant of the rows [x y 1] is zero ⇔ the rows are dependent | Equivalent statements; independence | T-equiv, Section 4.5 |
| Corner order is read from the signs of 2×2 determinants of consecutive edge vectors | Definition of the determinant | Section 3.1 |
| Ah = b is consistent ⇔ b is in the column space ⇔ rank A = rank [A b]; noisy data gives rank [A b] = 9 > 8 | Consistency criterion | T-cons |
| AᵀA is nonsingular ⇔ rank A = 8; rank(AᵀA) = rank(A) | Transpose rules; rank-nullity; equivalent statements | T-rules, T-rn, T-equiv |
| det(AB) = det A · det B, det Aᵀ = det A, a row swap flips the sign | Determinant properties | T-det |
| The adjoint formula gives the inverse of a 3×3 matrix | Adjoint formula | T-adj |
| Cramer’s rule solves small systems | Cramer’s rule | T-cramer |
| The structure tensor has rank 0 (flat), 1 (edge), or 2 (corner, gradients span ℝ²); det M ≠ 0 ⇔ M nonsingular | Span and rank; equivalent statements | Sections 4.4, 4.9; T-equiv |
| A finite difference is a fixed linear map, a matrix product | Matrix product | Section 1.3 |
| Resize is the matrix product R·I·Cᵀ | Matrix product | Section 1.3 |
| The blend output is a linear combination of the images | Vector-space operations; linear combinations | Sections 4.2, 4.4 (no numbered theorem) |

## 4. M-item index

The M-numbers restart inside each issue, so each is keyed by its GitHub issue ID. This is a lookup only: no statements or proofs.

| Issue | M-item | Topic | Book basis |
| --- | --- | --- | --- |
| S0-1 | M2 | System building: two rows per pair, Mh = 0 and Ah = b | No numbered theorem |
| S0-1, S1-1 | M6 | Normalization, H = T′⁻¹·H̃·T | T-equiv, T-rules |
| S1-1 | M1 | Homogeneous coordinates and degrees of freedom | Section 4.4, T-dim |
| S1-1 | M3 | Existence and uniqueness of H | T-sol, T-rn, T-count |
| S1-1 | M4 | When the 8×8 system is solvable | T-equiv, T-inv, T-adj |
| S1-2 | M8 | Inverse warp and the w guard | T-rules |
| S1-2 | M9 | Lines stay lines | T-rules |
| S1-3 | M7 | Validation by determinants (collinearity, corner order) | T-equiv, Sections 3.1, 4.5 |
| S1-3 | imgops note | Resize, flips, grayscale as matrix operations | Section 1.3 |
| S1-4 | M5 | Gauss-Jordan, RREF uniqueness, inverse by [A I] | T-rref, T-inv, Section 2.3 |
| S2-1 | M1 | Overdetermined system (N ≥ 5) | No numbered theorem |
| S2-1 | M2 | Why noisy data has no exact solution | T-cons |
| S2-1 | M3 | Normal equations | Outside our topics (calculus or Chapter 5) |
| S2-1 | M4 | When AᵀA is invertible | T-rules, T-rn, T-equiv |
| S2-1 | M5 | Conditioning | Outside our topics (singular values) |
| S2-2 | M1–M4 | Blend weights, convex combination, single-image regions, horizontal pans | Sections 4.2, 4.4 (no numbered theorem) |
| S2-2 | M5 | Exposure gain | Outside our topics (one-column normal equations) |
| S2-4 | V0.3 | Theorem identity checks | T-rn, T-det, T-adj, T-cramer, T-cons |
| S2-A | M6 | Algebraic versus pixel error | Definitions only (Section 2.2) |
| S3-1 | M4 | Why 4 pairs in a sample | T-sol, T-rn |
| S3-1 | M5 | Consensus | No numbered theorem |
| S3-1 | M6 | Iteration count | Probability; outside our topics |
| S3-2 | M1 | Gradients as linear maps | Section 1.3 |
| S3-2 | M2 | Structure tensor | T-rules, Sections 4.4, 4.9, T-equiv |
| S3-2 | M3 | Patch vectors and SSD | Definitions only (Section 2.2) |

## 5. Outside our topics (cited as such)

- Least squares (Chapter 5): the normal-equations step uses calculus or Chapter 5’s orthogonality.
- Eigenvalues (Chapter 7): avoided in the corner score by using det M and trace M.
- Singular values: the statement that forming AᵀA squares the condition number.
- Four points with no three collinear in either image give rank 8: stated as a standard fact.

## 6. To fill in from the book

- The theorem number for T-count (Section 1.1).
- The theorem number for the determinant of a triangular matrix (used for det T = s²; not in the table above).
- The section for permutation matrices.
- The one-to-one theorem assignment inside the grouped rows T-rules and T-det.

---

### Core Pipeline & Math Modules

- **[`linalg.py`]**: Custom linear algebra engine implemented from scratch without using `np.linalg`. Contains `rref`, `solve`, `inverse`, `det`, `rank`, and `nullspace_vector`.
- **[`homography.py`]**: Solves projective transformations between images.
  - `normalize_points`: Hartley centering & scaling to $\sqrt{2}$.
  - `solve_homography`: Exact 4-point homography solver.
  - `solve_homography_lsq`: Least-squares homography solver ($N \ge 5$) via normal equations $(A^T A) h = A^T b$ with rank guards.
  - `transfer_error` & `algebraic_residual`: Error metrics for correspondences.
- **[`warp.py`]**: Canvas setup and geometric mapping.
  - `check_h`: Validates homographies against horizon singularities and boundary touch.
  - `canvas_bounds`: Calculates bounding boxes for multiple transformed images.
  - `warp_image`: Backward / inverse mapping with nearest-neighbor sampling.
- **[`blend.py`]**: Panorama image blending.
  - `edge_weights`: Distance-to-border linear weights.
  - `feather_blend`: Seamless convex combination with optional exposure gain matching.

---

### I/O & Image Processing

- **[`io_utils.py`]**: The only file allowed to import Pillow. Handles opening images, saving PNGs, and reading EXIF orientation tags.
- **[`imgops.py`]**: Pure NumPy image operations treated as linear transformations: `grayscale` (matrix-vector product), `flip_h`/`flip_v` (permutation matrices), `resize` ($R \cdot I \cdot C^T$), and `apply_exif_orientation`.
- **[`points.py`]**: JSON loading/saving for $(x, y)$ coordinate points, calibration chart generation, pixel stamping (`mark_points`), collinearity detection (`are_collinear`), and corner winding checks (`corner_order_ok`).

---

### CLI & Testing

- **[`demo.py`]**: Primary user-facing command-line tool:
  - `rectify`: Squares off an angled photo of a rectangle.
  - `panorama`: Stitches two photos into a wide panorama (outputs hard-cut and blended).
  - `rectify_panorama`: Stitches two photos and rectifies the scene in one pass.
- **[`tests.py`]**: Full test runner (51 unit tests) covering mathematical proofs, contract checks, `np.linalg` ban verification, and acceptance criteria (V0.1–V0.9, V1.1–V1.6, V2.1–V2.7).
