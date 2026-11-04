## Rules 

1. **Only the Lead merges.** A PR is ready when its description lists the V-IDs it covers, those tests are in `tests.py`, `python tests.py` passes, and its signatures match the contract.
2. **PRs are due by 5PM** on the deadline day.
3. **Linear algebra only through `linalg.py`.** A test fails if np.linalg appears outside `linalg.py` and `tests.py`. After Person 4’s swap, `linalg.py` loses its exemption too.
4. **Merge order, Nov 9:** Person 4’s `linalg.py` core (only if V0.1 and V0.2 pass, else the wrapper stays), then Person 3, then Person 2, then your own `homography.py`. **Nov 11:** Person 4’s remaining `linalg.py`, re-run all tests, then your LSQ, then blend, then the panorama demo.
5. **Paper caveats to carry into every math note:** the normal equations come from calculus or Chapter 5 (outside Chapters 1–4); the claim that AᵀA squares the condition number needs singular values (outside our topics); four points in general position giving rank 8 is stated as a standard fact and verified numerically; the one-dimensional null space is generic.

## Schedule and labels

| Label | Date | What lands |
| --- | --- | --- |
| S0 | Wed Nov 4 | initial files, interface freeze, `linalg.py` wrapper |
| S1 | Mon Nov 9 | Exact solver, warp, I/O, points, imgops, `linalg.py` core, hero demo |
| S2 | Wed Nov 11 | LSQ, blend, rest of `linalg.py`, theorem tests, panorama demo, slides draft |
| S2 Analysis | Fri Nov 13 | Monte Carlo (V1.7, V1.8) |
| S3 buffer | Nov 12–15 | RANSAC, matching (droppable), delays |
| Final | Mon Nov 16 | Integration check, Reflection |
| Paper | Nov 10, 12, 14, 16 | See the paper issues below |

V-ID mapping: V0.x belongs to S0/S1, V1.x to S2, V2.x to S2 blend, V3.x to S3. Nov 9 and Nov 16 are Mondays: check the exam date against both.

## Interface contract

```
rref(M), solve(A, b), inverse(A), det(A), rank(M), nullspace_vector(M)   # linalg.py
normalize_points(pts) -> (pts_n, T)
solve_homography(src, dst) -> H              # N = 4, exact
solve_homography_lsq(src, dst) -> H          # N >= 5 (we use 8-12), normal equations
algebraic_residual(H, src, dst) -> float
transfer_error(H, src, dst) -> array (N,)    # pixel distance per pair
check_h(H, shape) -> None                    # or raises HomographyError
canvas_bounds(H_list, shapes) -> (shape, offset)
warp_image(img, H, canvas_shape, offset) -> (out, valid, x, y)
edge_weights(x, y, valid, shape) -> w
feather_blend(imgs, weights) -> out
ransac_homography(src, dst) -> (H, inlier_mask)
```

Points are (N, 2) float arrays of (x, y). H is 3×3 with h33 = 1. `warp_image` must return the source coordinates and the validity mask.

## Verification map

| V-ID | Check | Author | Date |
| --- | --- | --- | --- |
| V0.1, V0.2 | RREF uniqueness; solve and inverse versus np.linalg (tests only) | Person 4 | Nov 9 |
| V0.3 | Rank, nullity, consistency, determinant, adjoint, Cramer theorem checks | Person 4 | Nov 11 |
| V0.4–V0.6 | Known-H; identity and affine; invalid input | Lead | Nov 9 |
| V0.7 | Warp exactness, corners, canvas cap, no NaN | Person 2 | Nov 9 |
| V0.8, V0.9 | Points; imgops | Person 3 | Nov 9 |
| V0.10 | Hero demo on the real rectangle photo | Person 3 writes, Lead runs | Nov 9 |
| V1.1–V1.6 | LSQ correctness, optimality, rank guard | Person 4 | Nov 11 |
| V1.7, V1.8 | Monte Carlo with and without normalization | Person 4 | Nov 13 |
| V2.1–V2.7 | Blend | Person 2 | Nov 11 |
| V3.1–V3.3 | Gradients, Harris, patch matching | Person 3 | S3 |
| V3.4–V3.8 | RANSAC accuracy, seed, failure cases, iteration formula, real pair | Lead | S3 |
