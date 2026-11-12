# Panorama Stitcher & Rectifier: User Guide & Math Primer

Welcome to the **Panorama Image Stitcher & Rectifier**. This document explains **how to use this software**, **the mathematical principles behind it** in plain language, and **the difference between hard-cut and blended results**.

---

## 1. What Does This Software Do?

This software solves two primary geometric problems in computer vision:
1. **Rectification:** Takes a tilted photograph of a flat rectangular object (e.g. a whiteboard, document, poster, or screen) and transforms it into an upright, undistorted, frontal rectangle.
2. **Panorama Stitching:** Takes two overlapping photos taken from the same camera position and combines them into one wide, high-resolution panorama.
3. **Combined Rectified Panorama (`rectify_panorama`):** Stitches multiple photos of a wide rectangular object (e.g., a wide blackboard) and squares it off into a clean, undistorted rectangular image in a single pass.

---

## 2. Quickstart: How to Use the Software

All operations are executed via `demo.py`:

```bash
# Make sure your virtual environment is active or use the venv binary
./venv/bin/python demo.py <command>
```

### Mode 1: Rectify a Single Image
Un-tilts an angled photo of a rectangle.

```bash
./venv/bin/python demo.py rectify <IMAGE> <CORNERS_JSON> [ASPECT]
```
- `<IMAGE>`: Path to your photo (e.g., `board.png`).
- `<CORNERS_JSON>`: JSON file containing the 4 corners of the board in clockwise order (`TL -> TR -> BR -> BL`).
- `[ASPECT]` *(Optional)*: Target width:height ratio (e.g., `16:9`, `4:3`, `2:1`, or `1.5`) to prevent perspective compression.
- **Outputs:** `rectify_before.png` and `rectify_after.png`.

---

### Mode 2: Stitch Two Overlapping Photos (Standard Panorama)
Stitches a left photo and a right photo together.

```bash
./venv/bin/python demo.py panorama <LEFT_IMAGE> <RIGHT_IMAGE> <PAIRS_JSON>
```
- `<LEFT_IMAGE>`: The anchor / reference photo.
- `<RIGHT_IMAGE>`: The overlapping photo to attach to the right.
- `<PAIRS_JSON>`: JSON file containing 6–12 matching feature points in the overlap area.
- **Outputs:** `panorama_hardcut.png` and `panorama_blended.png`.

---

### Mode 3: Rectified Panorama (Combined Stitched & Squared Output)
Stitches two photos of a wide board/screen and frames the result cleanly into an upright rectangle.

```bash
./venv/bin/python demo.py rectify_panorama <LEFT_IMAGE> <RIGHT_IMAGE> <PAIRS_JSON> <CORNERS_JSON> [corners_on=left|right] [ASPECT]
```
- `<LEFT_IMAGE>`, `<RIGHT_IMAGE>`: The two overlapping shots.
- `<PAIRS_JSON>`: Overlap matching points.
- `<CORNERS_JSON>`: The 4 corners of the rectangular object.
- `[corners_on]`: Specify `left` if corners were clicked on the left photo (default), or `right` if clicked on the right photo.
- `[ASPECT]`: Target ratio (e.g. `16:9`, `4:3`, `2:1`).
- **Outputs:** `rectified_panorama_hardcut.png` and `rectified_panorama_blended.png`.

---

## 3. How to Format the JSON Point Files

Coordinates are in pixels: `x` is the column distance from the left edge (0-indexed), and `y` is the row distance from the top edge (0-indexed).

### `corners.json`
Must contain exactly 4 points in **clockwise order**: Top-Left $\to$ Top-Right $\to$ Bottom-Right $\to$ Bottom-Left.

```json
{
  "corners": [
    {"x": 145.0, "y": 155.0},
    {"x": 583.0, "y": 251.0},
    {"x": 653.0, "y": 629.0},
    {"x": 301.0, "y": 915.0}
  ]
}
```

### `pairs.json`
Contains corresponding points between the two photos inside their overlap region:

```json
{
  "src": [
    {"x": 583.0, "y": 251.0},
    {"x": 653.0, "y": 629.0},
    {"x": 465.0, "y": 240.0},
    {"x": 551.0, "y": 292.0},
    {"x": 430.0, "y": 289.0},
    {"x": 441.0, "y": 348.0}
  ],
  "dst": [
    {"x": 593.0, "y": 144.0},
    {"x": 598.0, "y": 834.0},
    {"x": 177.0, "y": 161.0},
    {"x": 432.0, "y": 225.0},
    {"x": 60.0, "y": 242.0},
    {"x": 60.0, "y": 322.0}
  ]
}
```
*Tip: Ensure point $i$ in `"src"` corresponds to the exact same visual feature as point $i$ in `"dst"`.*

---

## 4. How It Works Mathematically (In Simple Terms)

### A. The Homography Matrix ($H$)
Whenever a flat plane is photographed from different angles, any point $(x, y)$ in one image maps to $(x', y')$ in another image via a projective transformation called a **Homography**, represented as a $3 \times 3$ matrix $H$:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

The final pixel coordinates on the target image are obtained by dividing by $w'$:
$$x_{\text{dest}} = \frac{x'}{w'}, \qquad y_{\text{dest}} = \frac{y'}{w'}$$

The denominator $w'$ is what creates perspective: as an object gets further away or tilts, $w'$ changes, creating realistic perspective vanishing lines.

### B. Solving for $H$: Exact 4-Point vs. Least-Squares
- Each matching pair gives 2 linear equations.
- The matrix $H$ has 8 unknown parameters ($h_{33} = 1$).
- **With 4 pairs:** We have $4 \times 2 = 8$ equations for 8 unknowns. We solve this exact system via Gaussian elimination ($A h = b$).
- **With $N \ge 5$ pairs:** Human clicking introduces slight noise. We have an overdetermined system ($2N > 8$). The software uses the **Normal Equations** of Least Squares:
  $$(A^T A) h = A^T b$$
  This finds the single homography $H$ that minimizes the squared distance error across all clicked points.

### C. Hartley Preconditioning (Normalization)
If pixel coordinates (e.g. $x = 3000, y = 2000$) are multiplied directly, terms like $x \cdot x'$ become $10^7$, causing floating-point rounding errors and rank-deficient matrices.
Before solving, the software shifts the points so their center is at $(0, 0)$ and scales them so their average distance from the center is $\sqrt{2}$. Once $H$ is computed, it mathematically denormalizes the result back to full image scale.

### D. Inverse Warping (Backward Mapping)
Instead of taking each pixel in the source image and trying to project it onto the canvas (which leaves empty holes and gaps), the software uses **backward mapping**:
For every integer pixel $(u, v)$ on the output canvas, it multiplies by $H^{-1}$ to calculate where in the original photo that pixel came from, sampling the color cleanly.

---

## 5. Blended vs. Hard-Cut: What's the Difference?

When two overlapping photos are warped onto a shared canvas, there is an overlapping region where **both photos have color information**. The software produces two outputs to compare:

| Feature | **Hard-Cut** (`*_hardcut.png`) | **Feather-Blended** (`*_blended.png`) |
| :--- | :--- | :--- |
| **How it works** | Takes pixels from Image A where valid; if Image A is missing, it takes pixels from Image B. | Uses a weighted average (convex combination) of both images across the overlap. |
| **The Seam** | **Visible line / edge:** Any difference in brightness, white balance, or slight vignette creates an abrupt, noticeable step edge where Image A ends. | **Invisible transition:** The seam is completely smoothed out. |
| **Mathematical Formula** | $\text{Output}(p) = I_A(p)$ (if inside A)<br>$\text{Output}(p) = I_B(p)$ (if inside B) | $\text{Output}(p) = \alpha_A(p) \cdot I_A(p) + \alpha_B(p) \cdot I_B(p)$<br>where $\alpha_A + \alpha_B = 1$ |
| **Weighting Method** | Binary decision ($0$ or $1$). | **Distance-to-Border Weighting:** Pixels deep inside Photo A get weight near 1; as you approach Photo A's edge, its weight smoothly tapers to 0 while Photo B's weight rises to 1. |
| **Exposure Gain** | None. | Optional exposure matching: adjusts the overall gain $g$ of Photo B so its brightness matches Photo A before blending. |
