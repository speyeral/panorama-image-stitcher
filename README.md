# Panorama Stitcher & Rectifier: User Guide & Math Primer

## 1. What Does This Software Do?

This software solves three primary geometric problems in computer vision:

1. **Rectification:** Takes a tilted photograph of a flat rectangular object (e.g., a whiteboard, document, poster, or screen) and transforms it into an upright, undistorted, frontal rectangle.
2. **Panorama Stitching:** Takes two overlapping photos taken from the same camera position and combines them into one wide, high-resolution panorama.
3. **Combined Rectified Panorama (`rectify_panorama`):** Stitches two photos of a wide rectangular object (e.g., a wide blackboard or painting) and squares it off into a clean, undistorted rectangular image in a single pass.

---

## 2. Interactive Mode with OpenCV (`run_demo_with_cv.py`)

Instead of manually typing pixel coordinates into JSON files, you can use `run_demo_with_cv.py` to interactively select and adjust points directly on your images.

It automatically generates the required JSON files inside the `inputs/` folder and invokes `demo.py` under the hood.

### Installation

Install OpenCV inside your virtual environment:

```bash
./venv/bin/pip install opencv-python
```

### Preparation

Place your input images inside the `inputs/` folder before running the interactive commands.

### Mode 1: Rectify a Single Image

Interactively select the four corners of a rectangular object.

```bash
./venv/bin/python run_demo_with_cv.py rectify inputs/<IMAGE> [ASPECT]
```

**Controls:**

- Drag the four labeled corners (`TL`, `TR`, `BR`, `BL`) onto the corresponding corners of the object.
- Press `ENTER` or `SPACE` to save the selected coordinates and execute rectification.
- Press `R` to reset the corner positions.
- Press `ESC` to quit.

**Output:**

- Automatically generates `inputs/<IMAGE>_corners.json`.
- Executes the `rectify` command in `demo.py`.

### Mode 2: Stitch Two Overlapping Photos (Standard Panorama)

Interactively select corresponding feature points between two overlapping images.

```bash
./venv/bin/python run_demo_with_cv.py panorama inputs/<IMAGE_1> inputs/<IMAGE_2>
```

**Controls:**

- Two windows open simultaneously, displaying the left and right images.
- Matching points are color-coded and numbered (`#1`, `#2`, `#3`, etc.).
- Drag matching numbers onto identical visual features in the overlap region.
- Press `S` to swap which window is LEFT (Anchor / src) and RIGHT (Warped / dst).
- Press `A` to add a linked point pair to both windows simultaneously, ensuring equal point counts.
- Press `D` to delete the last matching point pair.
- Press `ENTER` or `SPACE` to save the selected points and execute panorama stitching.

**Output:**

- Automatically generates `inputs/<IMAGE_1>_<IMAGE_2>_pairs.json`.
- Executes the `panorama` command in `demo.py`.

### Mode 3: Rectified Panorama (Combined Stitched & Squared Output)

Interactively select the matching points and rectangular corners required to stitch and rectify a wide object.

```bash
./venv/bin/python run_demo_with_cv.py rectify_panorama inputs/<IMAGE_1> inputs/<IMAGE_2> [corners_on=left|right] [ASPECT]
```

**Controls:**

1. First, the two-window interface opens for selecting matching feature points between the overlapping images.
2. After confirming the matching points, the four-corner picker opens on the selected image.
3. Select the four corners of the rectangular object in clockwise order (`TL`, `TR`, `BR`, `BL`).
4. Press `ENTER` or `SPACE` to save the coordinates and execute the combined operation.

The `corners_on` argument determines which image will be used for selecting the rectangular object's corners:

- `left`: Select corners on the left image (default).
- `right`: Select corners on the right image.

**Output:**

- Automatically generates the required pairs and corners JSON files inside `inputs/`.
- Executes the `rectify_panorama` command in `demo.py`.

---

## 3. Quickstart: Direct CLI Usage (`demo.py`)

If you already have pre-made JSON point files, you can bypass the interactive interface and execute `demo.py` directly.

```bash
# Make sure your virtual environment is active or use the venv binary
./venv/bin/python demo.py <command>
```

### Mode 1: Rectify a Single Image

Un-tilts an angled photo of a rectangle.

```bash
./venv/bin/python demo.py rectify <IMAGE> <CORNERS_JSON> [ASPECT]
```

- `<IMAGE>`: Path to your photo (e.g., `inputs/board.png`).
- `<CORNERS_JSON>`: JSON file containing the four corners of the board in clockwise order (`TL -> TR -> BR -> BL`).
- `[ASPECT]` _(Optional)_: Target width:height ratio (e.g., `16:9`, `4:3`, `2:1`, or `1.5`) to prevent perspective compression.
- **Outputs:** `rectify_before.png` and `rectify_after.png`.

---

### Mode 2: Stitch Two Overlapping Photos (Standard Panorama)

Stitches a left photo and a right photo together.

```bash
./venv/bin/python demo.py panorama <LEFT_IMAGE> <RIGHT_IMAGE> <PAIRS_JSON>
```

- `<LEFT_IMAGE>`: The anchor / reference photo.
- `<RIGHT_IMAGE>`: The overlapping photo to attach to the right.
- `<PAIRS_JSON>`: JSON file containing 4–12 matching feature points in the overlap area.
- **Outputs:** `panorama_hardcut.png` and `panorama_blended.png`.

---

### Mode 3: Rectified Panorama (Combined Stitched & Squared Output)

Stitches two photos of a wide board/screen and frames the result cleanly into an upright rectangle.

```bash
./venv/bin/python demo.py rectify_panorama <LEFT_IMAGE> <RIGHT_IMAGE> <PAIRS_JSON> <CORNERS_JSON> [corners_on=left|right] [ASPECT]
```

- `<LEFT_IMAGE>`, `<RIGHT_IMAGE>`: The two overlapping shots.
- `<PAIRS_JSON>`: Overlap matching points.
- `<CORNERS_JSON>`: The four corners of the rectangular object.
- `[corners_on]`: Specify `left` if corners were clicked on the left photo (default), or `right` if clicked on the right photo.
- `[ASPECT]`: Target ratio (e.g., `16:9`, `4:3`, `2:1`).
- **Outputs:** `rectified_panorama_hardcut.png` and `rectified_panorama_blended.png`.

---

## 4. How to Format the JSON Point Files

Coordinates are in pixels: `x` is the column distance from the left edge (0-indexed), and `y` is the row distance from the top edge (0-indexed).

### `corners.json`

Must contain exactly four points in **clockwise order**:

Top-Left → Top-Right → Bottom-Right → Bottom-Left.

```json
{
  "corners": [
    { "x": 145.0, "y": 155.0 },
    { "x": 583.0, "y": 251.0 },
    { "x": 653.0, "y": 629.0 },
    { "x": 301.0, "y": 915.0 }
  ]
}
```

### `pairs.json`

Contains corresponding points between the two photos inside their overlap region:

- `src`: Points on the left / anchor image.
- `dst`: Corresponding points on the right / warped image.

```json
{
  "src": [
    { "x": 583.0, "y": 251.0 },
    { "x": 653.0, "y": 629.0 },
    { "x": 465.0, "y": 240.0 },
    { "x": 551.0, "y": 292.0 },
    { "x": 430.0, "y": 289.0 },
    { "x": 441.0, "y": 348.0 }
  ],
  "dst": [
    { "x": 593.0, "y": 144.0 },
    { "x": 598.0, "y": 834.0 },
    { "x": 177.0, "y": 161.0 },
    { "x": 432.0, "y": 225.0 },
    { "x": 60.0, "y": 242.0 },
    { "x": 60.0, "y": 322.0 }
  ]
}
```

_Tip: Ensure point `i` in `"src"` corresponds to the exact same visual feature as point `i` in `"dst"`. Both arrays must contain the same number of points._

---

## 5. How It Works Mathematically (In Simple Terms)

### A. The Homography Matrix (H)

Whenever a flat plane is photographed from different angles, any point `(x, y)` in one image maps to `(x', y')` in another image via a projective transformation called a **Homography**, represented as a `3 × 3` matrix H:

\[
\begin{bmatrix}
x'\\
y'\\
w'
\end{bmatrix}
=
\begin{bmatrix}
h*{11} & h*{12} & h*{13}\\
h*{21} & h*{22} & h*{23}\\
h*{31} & h*{32} & 1
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
\]

The final pixel coordinates on the target image are obtained by dividing by `w'`:

\[
x*{\text{dest}} = \frac{x'}{w'}, \qquad
y*{\text{dest}} = \frac{y'}{w'}
\]

The denominator `w'` is what creates perspective: as an object gets further away or tilts, `w'` changes, creating realistic perspective vanishing lines.

### B. Solving for H: Exact 4-Point vs. Least-Squares

- Each matching pair gives two linear equations.
- The matrix H has eight unknown parameters (`h₃₃ = 1`).

- **With 4 pairs:** We have `4 × 2 = 8` equations for eight unknowns. We solve this exact square system via Gaussian elimination (`A h = b`).

- **With N ≥ 5 pairs:** Human clicking introduces slight noise. We have an overdetermined system (`2N > 8`). The software uses the **Normal Equations** of Least Squares:

\[
(A^T A)h = A^T b
\]

This turns `AᵀA` into a square `8 × 8` matrix and finds the single homography H that minimizes the total squared error:

\[
\|Ah-b\|^2
\]

across all clicked points.

### C. Hartley Preconditioning (Normalization)

If pixel coordinates (e.g., `x = 3000, y = 2000`) are multiplied directly, terms like `x · x'` become approximately `10⁷`, causing floating-point rounding errors and ill-conditioned matrices during Gaussian elimination.

Before solving, `normalize_points()` shifts the points so their center of mass (centroid) is at `(0, 0)` and scales them so their average distance from the origin is `√2`.

Once the normalized homography `H̃` is computed, it mathematically denormalizes the result back to full image scale:

\[
H = T*{\text{dst}}^{-1}\tilde{H}T*{\text{src}}
\]

### D. Matrix Composition for Single-Pass `rectify_panorama`

Rather than warping the right image twice (which compounds interpolation and rounding artifacts), the software multiplies the stitching homography and rectification homography together first.

The total transformation for the right image is:

\[
H*{\text{right,total}} =
H*{\text{rect}} \cdot H\_{\text{right}\rightarrow\text{left}}
\]

This combines the two geometric transformations into a single matrix.

Both images are then mapped directly onto the upright output rectangle in a single inverse warp.

This avoids an unnecessary intermediate warped image and reduces the accumulation of resampling artifacts.

### E. Inverse Warping (Backward Mapping)

Instead of taking each pixel in the source image and trying to project it onto the canvas (which leaves empty holes and gaps), the software uses **backward mapping**:

For every integer pixel `(u, v)` on the output canvas, it multiplies by `H⁻¹` to calculate where in the original photo that pixel came from, sampling the color cleanly.

---

## 6. Blended vs. Hard-Cut: What's the Difference?

When two overlapping photos are warped onto a shared canvas, there is an overlapping region where **both photos have color information**.

The software produces two outputs to compare:

| Feature                  | **Hard-Cut** (`*_hardcut.png`)                                                                                                          | **Feather-Blended** (`*_blended.png`)                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **How it works**         | Selects the pixel from whichever photo is farther from its own border (based on the distance weights).                                  | Uses a weighted average (convex combination) of both images across the overlap.                                                                                                    |
| **The Seam**             | **Visible line / edge:** Any difference in brightness, white balance, or slight vignette can create an abrupt step edge along the seam. | **Smooth transition:** The seam is smoothly crossfaded across the overlap.                                                                                                         |
| **Mathematical Formula** | Selects either `I_A(p)` or `I_B(p)` based on the larger distance weight.                                                                | \(\text{Output}(p) = \alpha_A(p) \cdot I_A(p) + \alpha_B(p) \cdot I_B(p)\), where \(\alpha_A + \alpha_B = 1\).                                                                     |
| **Weighting Method**     | Binary decision (`0` or `1`). The image with the greater distance-to-border weight is selected at each pixel.                           | **Distance-to-Border Weighting:** Pixels deep inside Photo A get weight near 1; as you approach Photo A's edge, its weight smoothly tapers to 0 while Photo B's weight rises to 1. |
| **Exposure Gain**        | None.                                                                                                                                   | Optional exposure matching: adjusts the overall gain `g` of Photo B so its brightness matches Photo A before blending.                                                             |
