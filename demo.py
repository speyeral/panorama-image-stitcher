import sys
from typing import Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

# Custom project modules:
# - homography: solves 3x3 projective matrices (exact 4-point and least-squares N>=5)
# - warp: computes canvas bounds and inverse-warps pixels onto a target grid
# - io_utils: reads/writes PNGs and extracts camera orientation metadata (EXIF)
# - imgops: rotates/flips image arrays based on EXIF tags
# - points: loads JSON coordinate files and checks clockwise corner ordering
# - blend: computes distance-to-border weights and feather-blends overlapping images
import homography
import warp
import io_utils
import imgops
import points
import blend

# --- Type Aliases (for code clarity and static type checking) ---

ImageArray: TypeAlias = NDArray[np.uint8]
"""Standard 8-bit image (0-255), shape `(H, W, 3)` for RGB or `(H, W)` for grayscale.

Example:
    `np.zeros((1080, 1920, 3), dtype=np.uint8)`
"""

FloatArray: TypeAlias = NDArray[np.float64]
"""64-bit float array for coordinates, masks, and blending weights.

Example:
    `np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)`
"""

BoolMask: TypeAlias = NDArray[np.bool_]
"""True/False pixel validity mask, shape `(H, W)`.

Example:
    `np.array([[True, False], [True, True]], dtype=np.bool_)`
"""

Point2D: TypeAlias = NDArray[np.float64]
"""Single 2D point `[x, y]`, shape `(2,)`.

Example:
    `np.array([120.5, 340.0], dtype=np.float64)`
"""

PointList2D: TypeAlias = NDArray[np.float64]
"""List of N 2D points `[[x1, y1], ...]`, shape `(N, 2)`.

Example:
    `np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float64)`
"""

HomographyMatrix: TypeAlias = NDArray[np.float64]
"""3x3 projective transformation matrix, shape `(3, 3)`.

Example:
    `np.eye(3, dtype=np.float64)`
"""

AspectInput: TypeAlias = str | float | None
"""Target aspect ratio (width / height).

Examples:
    `"16:9"`, `"4/3"`, `1.777`, or `None`
"""

WindingSide: TypeAlias = Literal["left", "right"]
"""Specifies which image the reference corners belong to.

Allowed values:
    `"left"` or `"right"`
"""


def synthetic_demo() -> None:
    """Quick self-test using a synthetic square image when demo.py is run with no arguments."""
    # Create a blank 100x100 RGB black image (3 color channels)
    img: ImageArray = np.zeros((100, 100, 3), dtype=np.uint8)
    # Paint a reddish square in the center region (rows 20..79, cols 20..79)
    img[20:80, 20:80] = [200, 50, 50]
    
    # Convert RGB to grayscale using standard luminance weights (dot product across the 3 channels)
    weights: FloatArray = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    gray: ImageArray = np.dot(img, weights).astype(np.uint8)
    
    # Define 4 corners of the inner reddish box (src) and map them to the full 100x100 frame (dst)
    src: PointList2D = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float64)
    dst: PointList2D = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    
    # Solve for the exact 3x3 homography H that stretches the 60x60 box into 100x100
    H: HomographyMatrix = homography.solve_homography(src, dst)

    # Inverse-warp the grayscale image onto a 100x100 canvas starting at offset (0, 0)
    out: ImageArray
    valid: BoolMask
    x: FloatArray
    y: FloatArray
    out, valid, x, y = warp.warp_image(gray, H, (100, 100), (0, 0))
    print(f"H computed:\n{H}\nCenter pixel value: {out[50, 50]}")


def _parse_aspect(aspect_str: AspectInput) -> float | None:
    """Convert user-provided aspect ratio ('16:9', '4/3', or '1.777') into a decimal width/height ratio."""
    if not aspect_str:
        return None
    s: str = str(aspect_str).strip()
    # Handle colon notation like "16:9" -> 16.0 / 9.0
    if ":" in s:
        parts: list[str] = s.split(":")
        return float(parts[0]) / float(parts[1])
    # Handle slash notation like "4/3" -> 4.0 / 3.0
    if "/" in s:
        parts = s.split("/")
        return float(parts[0]) / float(parts[1])
    # Handle direct decimal input like "1.5"
    return float(s)


def _compute_target_dimensions(
    tl: Point2D,
    tr: Point2D,
    br: Point2D,
    bl: Point2D,
    aspect: AspectInput = None,
) -> tuple[int, int]:
    """Calculate the output (width, height) in pixels for a rectified rectangle.

    Why this matters: A tilted rectangle in a photo suffers from perspective foreshortening
    (edges angled away from the camera look shorter). If the user passes a known real-world
    aspect ratio (like 16:9 for a screen), we lock the height to the longest vertical edge
    and compute width = height * aspect so the output isn't squished.
    """
    # Compute Euclidean length sqrt(dx^2 + dy^2) of all four tilted sides in the image
    w_top: float = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
    w_bot: float = float(np.hypot(br[0] - bl[0], br[1] - bl[1]))
    h_left: float = float(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
    h_right: float = float(np.hypot(br[0] - tr[0], br[1] - tr[1]))

    width: int
    height: int

    if aspect is not None:
        aspect_val: float | None = _parse_aspect(aspect)
        if aspect_val is None:
            raise ValueError(f"Invalid aspect ratio provided: {aspect}")
        # Preserve maximum detail by choosing the taller of the left/right edges
        height = max(1, int(round(max(h_left, h_right))))
        # Scale width proportionally to match the true physical aspect ratio
        width = max(1, int(round(height * aspect_val)))
    else:
        # No aspect ratio given: fall back to the apparent top width and left height
        width = max(1, int(round(w_top)))
        height = max(1, int(round(h_left)))

    return width, height


def rectify(
    image_path: str,
    points_json: str,
    aspect: AspectInput = None,
    out_before: str = "rectify_before.png",
    out_after: str = "rectify_after.png",
) -> tuple[HomographyMatrix, ImageArray]:
    """Mode 1: Un-tilt a single angled photo of a rectangular object into a flat upright image."""
    # 1. Load the raw image and rotate it upright if phone/camera EXIF orientation tags exist
    img: ImageArray = io_utils.open_image(image_path)
    img = imgops.apply_exif_orientation(img, io_utils.read_exif_orientation(image_path))
    
    # 2. Load the 4 clicked corners from JSON and verify they wind clockwise (TL -> TR -> BR -> BL)
    loaded_pts: dict[str, PointList2D] = points.load_points(points_json)
    corners: PointList2D = loaded_pts["corners"]
    
    if not points.corner_order_ok(corners, winding="clockwise"):
        raise ValueError(
            "corners must be four distinct (x, y) points in the order "
            "top-left, top-right, bottom-right, bottom-left (clockwise on screen)"
        )
        
    tl: Point2D
    tr: Point2D
    br: Point2D
    bl: Point2D
    tl, tr, br, bl = corners
    
    # 3. Determine how wide and tall the straightened output rectangle should be
    width: int
    height: int
    width, height = _compute_target_dimensions(tl, tr, br, bl, aspect=aspect)
    
    # 4. Define the 4 destination corners of our clean upright rectangle (0-indexed pixel grid)
    dst: PointList2D = np.array(
        [
            [0.0, 0.0],                    # Top-Left
            [width - 1.0, 0.0],            # Top-Right
            [width - 1.0, height - 1.0],   # Bottom-Right
            [0.0, height - 1.0],           # Bottom-Left
        ],
        dtype=np.float64,
    )

    # 5. Solve the exact 8x8 linear system for the 3x3 homography H mapping tilted -> upright
    h_mat: HomographyMatrix = homography.solve_homography(corners, dst)
    
    # 6. Inverse-warp the tilted image onto the blank (height x width) destination grid
    # Note: warp_image takes canvas_shape as (rows, cols) = (height, width)
    out: ImageArray
    _valid: BoolMask
    _x: FloatArray
    _y: FloatArray
    out, _valid, _x, _y = warp.warp_image(img, h_mat, (height, width), (0, 0))
    
    # 7. Save both original and rectified images for comparison, and print H
    io_utils.save_png(out_before, img)
    io_utils.save_png(out_after, out)
    print(h_mat)
    return h_mat, out


def panorama(
    left_path: str,
    right_path: str,
    points_json: str,
    out_hardcut: str = "panorama_hardcut.png",
    out_blended: str = "panorama_blended.png",
    exposure_gain: bool = False,
) -> tuple[HomographyMatrix, ImageArray, ImageArray]:
    """Mode 2: Stitch two overlapping photos into a single wide panorama.

    Coordinate convention in pairs.json:
    - 'src' points live on the LEFT image (the stationary anchor frame).
    - 'dst' points live on the RIGHT image (the photo being warped to align with the left).
    """
    # 1. Load both images and apply EXIF orientation fixes
    img_left: ImageArray = io_utils.open_image(left_path)
    img_left = imgops.apply_exif_orientation(img_left, io_utils.read_exif_orientation(left_path))
    img_right: ImageArray = io_utils.open_image(right_path)
    img_right = imgops.apply_exif_orientation(img_right, io_utils.read_exif_orientation(right_path))

    # 2. Load corresponding feature pairs clicked inside the overlapping middle strip
    pts: dict[str, PointList2D] = points.load_points(points_json)
    src: PointList2D = pts["src"]  # Landmark coordinates on Left Image
    dst: PointList2D = pts["dst"]  # Matching landmark coordinates on Right Image

    # 3. Compute H_right mapping Right Image coordinates (dst) -> Left Image space (src).
    # Notice the argument order (dst, src): we want to transform points FROM right TO left!
    # - Use exact 4-point solver if N == 4
    # - Use Least-Squares Normal Equations (A^T A)h = A^T b if N >= 5 to average out click noise
    H_right: HomographyMatrix
    if len(src) == 4:
        H_right = homography.solve_homography(dst, src)
    else:
        H_right = homography.solve_homography_lsq(dst, src)

    # Left image is our stationary reference frame, so its transform is the 3x3 Identity matrix
    H_left: HomographyMatrix = np.eye(3, dtype=np.float64)

    # 4. Project the 4 outer border corners of both images to find the bounding box of the whole panorama.
    # 'offset' shifts negative coordinates (if Right Image warps above y=0 or left of x=0) onto (0, 0).
    canvas_shape: tuple[int, int]
    offset: tuple[float, float] | FloatArray
    canvas_shape, offset = warp.canvas_bounds([H_left, H_right], [img_left.shape, img_right.shape])

    # 5. Inverse-warp both images onto the shared panorama canvas.
    # valid_l / valid_r are boolean masks showing which canvas pixels actually landed inside each photo.
    warped_l: ImageArray
    valid_l: BoolMask
    x_l: FloatArray
    y_l: FloatArray
    warped_l, valid_l, x_l, y_l = warp.warp_image(img_left, H_left, canvas_shape, offset)

    warped_r: ImageArray
    valid_r: BoolMask
    x_r: FloatArray
    y_r: FloatArray
    warped_r, valid_r, x_r, y_r = warp.warp_image(img_right, H_right, canvas_shape, offset)

    # 6. Compute distance-to-border weight maps (1.0 deep inside a photo, tapering to 0.0 at its edge)
    weights_l: FloatArray = blend.edge_weights(x_l, y_l, valid_l, img_left.shape)
    weights_r: FloatArray = blend.edge_weights(x_r, y_r, valid_r, img_right.shape)

    # 7. Hard-cut composite (binary decision, no color mixing):
    # At every pixel, pick whichever photo is farther from its own border (higher edge weight).
    hard_cut: ImageArray = np.zeros_like(warped_l)
    take_l: BoolMask = valid_l & (~valid_r | (weights_l >= weights_r))
    take_r: BoolMask = valid_r & ~take_l
    hard_cut[take_r] = warped_r[take_r]
    hard_cut[take_l] = warped_l[take_l]

    # 8. Feather-blended composite (smooth crossfade):
    # Takes a weighted convex combination alpha_L * I_L + alpha_R * I_R across the overlap zone.
    blended: ImageArray = blend.feather_blend(
        [warped_l, warped_r],
        [weights_l, weights_r],
        exposure_gain=exposure_gain,
        valid=[valid_l, valid_r],
    )

    # 9. Save both outputs to disk and print H_right
    io_utils.save_png(out_hardcut, hard_cut)
    io_utils.save_png(out_blended, blended)
    print(H_right)
    return H_right, hard_cut, blended


def rectify_panorama(
    left_path: str,
    right_path: str,
    pairs_json: str,
    corners_json: str,
    corners_on: WindingSide = "left",
    aspect: AspectInput = None,
    out_hardcut: str = "rectified_panorama_hardcut.png",
    out_blended: str = "rectified_panorama_blended.png",
    exposure_gain: bool = False,
) -> tuple[HomographyMatrix, ImageArray, ImageArray]:
    """Mode 3: Stitch two photos AND rectify the combined scene into an upright rectangle in ONE warp.

    Why do both in one pass? By multiplying the two 3x3 homographies together:
        H_right_total = H_rect @ H_right_to_left
    we avoid warping the Right Image twice (which would double pixel-rounding blur/aliasing).
    """
    # 1. Load both photos and fix EXIF orientation
    img_left: ImageArray = io_utils.open_image(left_path)
    img_left = imgops.apply_exif_orientation(img_left, io_utils.read_exif_orientation(left_path))
    img_right: ImageArray = io_utils.open_image(right_path)
    img_right = imgops.apply_exif_orientation(img_right, io_utils.read_exif_orientation(right_path))

    # 2. Compute H_right_to_left from overlap pairs (maps Right Image coords -> Left Image space)
    pts: dict[str, PointList2D] = points.load_points(pairs_json)
    src: PointList2D = pts["src"]  # Left image overlap points
    dst: PointList2D = pts["dst"]  # Right image overlap points
    H_right_to_left: HomographyMatrix
    if len(src) == 4:
        H_right_to_left = homography.solve_homography(dst, src)
    else:
        H_right_to_left = homography.solve_homography_lsq(dst, src)

    # 3. Load the 4 outer corners of the board/painting and validate clockwise winding
    corners_data: PointList2D = points.load_points(corners_json)["corners"]
    if not points.corner_order_ok(corners_data, winding="clockwise"):
        raise ValueError(
            "corners must be four distinct (x, y) points in clockwise order: TL, TR, BR, BL"
        )

    # Ensure all 4 corners are expressed in Left Image coordinates:
    # If the user clicked the 4 corners on the Right Image instead, project them into Left space first
    # using H_right_to_left and homogeneous division by w.
    corners_left: PointList2D
    if corners_on.lower() == "right":
        c_h: FloatArray = np.column_stack((corners_data, np.ones(len(corners_data), dtype=np.float64)))
        mapped: FloatArray = (H_right_to_left @ c_h.T).T
        corners_left = mapped[:, :2] / mapped[:, 2:3]
    else:
        corners_left = corners_data

    # 4. Calculate target upright rectangle dimensions (width, height)
    tl: Point2D
    tr: Point2D
    br: Point2D
    bl: Point2D
    tl, tr, br, bl = corners_left
    
    width: int
    height: int
    width, height = _compute_target_dimensions(tl, tr, br, bl, aspect=aspect)
    
    dst_rect: PointList2D = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )

    # 5. Compute H_rect: maps Left Image coordinate space -> final upright rectangle
    H_rect: HomographyMatrix = homography.solve_homography(corners_left, dst_rect)

    # 6. Matrix Composition (chaining transformations via matrix multiplication):
    # - Left image only needs rectification: H_rect
    # - Right image first aligns to Left (H_right_to_left), then rectifies (H_rect): H_rect @ H_right_to_left
    H_left_total: HomographyMatrix = H_rect
    H_right_total: HomographyMatrix = H_rect @ H_right_to_left

    # Re-normalize both matrices so their bottom-right entry H[2, 2] == 1.0
    H_left_total = H_left_total / H_left_total[2, 2]
    H_right_total = H_right_total / H_right_total[2, 2]

    # 7. Canvas setup: unlike standard panorama, the output canvas is strictly the rectified (height, width) box
    canvas_shape: tuple[int, int] = (height, width)
    offset: FloatArray = np.array([0.0, 0.0], dtype=np.float64)

    # 8. Single-pass inverse warp for both images directly onto the upright rectangle
    warped_l: ImageArray
    valid_l: BoolMask
    x_l: FloatArray
    y_l: FloatArray
    warped_l, valid_l, x_l, y_l = warp.warp_image(img_left, H_left_total, canvas_shape, offset)

    warped_r: ImageArray
    valid_r: BoolMask
    x_r: FloatArray
    y_r: FloatArray
    warped_r, valid_r, x_r, y_r = warp.warp_image(img_right, H_right_total, canvas_shape, offset)

    # 9. Compute distance-to-border weights for blending
    weights_l: FloatArray = blend.edge_weights(x_l, y_l, valid_l, img_left.shape)
    weights_r: FloatArray = blend.edge_weights(x_r, y_r, valid_r, img_right.shape)

    # 10. Hard-cut composite (clean seam along the midpoint where weights_l == weights_r)
    hard_cut: ImageArray = np.zeros_like(warped_l)
    take_l: BoolMask = valid_l & (~valid_r | (weights_l >= weights_r))
    take_r: BoolMask = valid_r & ~take_l
    hard_cut[take_r] = warped_r[take_r]
    hard_cut[take_l] = warped_l[take_l]

    # 11. Feather-blended composite (smooth crossfade across the overlap)
    blended: ImageArray = blend.feather_blend(
        [warped_l, warped_r],
        [weights_l, weights_r],
        exposure_gain=exposure_gain,
        valid=[valid_l, valid_r],
    )

    # 12. Save outputs and print transformation matrices
    io_utils.save_png(out_hardcut, hard_cut)
    io_utils.save_png(out_blended, blended)
    print("Rectification H:\n", H_rect)
    print("Combined Right H:\n", H_right_total)
    return H_rect, hard_cut, blended


USAGE: str = (
    "usage:\n"
    "  python demo.py\n"
    "  python demo.py rectify IMAGE CORNERS_JSON [ASPECT]\n"
    "  python demo.py panorama LEFT_IMAGE RIGHT_IMAGE PAIRS_JSON\n"
    "  python demo.py rectify_panorama LEFT_IMAGE RIGHT_IMAGE PAIRS_JSON CORNERS_JSON [corners_on=left|right] [ASPECT]"
)


if __name__ == "__main__":
    # Parse command-line arguments passed after 'python demo.py'
    args: list[str] = sys.argv[1:]
    if not args:
        # No arguments provided -> run the built-in 100x100 synthetic test
        synthetic_demo()
    elif args[0] == "rectify" and len(args) in (3, 4):
        aspect_arg: str | None = args[3] if len(args) == 4 else None
        rectify(args[1], args[2], aspect=aspect_arg)
    elif args[0] == "panorama" and len(args) == 4:
        panorama(args[1], args[2], args[3])
    elif args[0] == "rectify_panorama" and len(args) in (5, 6, 7):
        # Optional arguments after CORNERS_JSON can be 'left'/'right' and/or an aspect ratio
        corners_on_arg: WindingSide = "left"
        aspect_pano_arg: str | None = None
        for extra in args[5:]:
            lower_extra: str = extra.lower()
            if lower_extra == "left":
                corners_on_arg = "left"
            elif lower_extra == "right":
                corners_on_arg = "right"
            else:
                aspect_pano_arg = extra
        rectify_panorama(
            args[1],
            args[2],
            args[3],
            args[4],
            corners_on=corners_on_arg,
            aspect=aspect_pano_arg,
        )
    else:
        # Invalid arguments -> print usage instructions and exit with error code 2
        print(USAGE)
        sys.exit(2)