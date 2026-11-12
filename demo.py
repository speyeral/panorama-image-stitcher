import sys
import numpy as np
import homography
import warp
import io_utils
import imgops
import points
import blend

def synthetic_demo():
    # Basic synthetic image generation (I/O inline script requirement)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [200, 50, 50]
    
    # Grayscale conversion via weight vector
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(img, weights).astype(np.uint8)
    
    src = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float)
    dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    
    H = homography.solve_homography(src, dst)
    out, valid, x, y = warp.warp_image(gray, H, (100, 100), (0, 0))
    print(f"H computed:\n{H}\nCenter pixel value: {out[50, 50]}")

def _parse_aspect(aspect_str):
    """Parse aspect ratio from string like '16:9', '4/3', or float '1.777'."""
    if not aspect_str:
        return None
    s = str(aspect_str).strip()
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) / float(parts[1])
    if "/" in s:
        parts = s.split("/")
        return float(parts[0]) / float(parts[1])
    return float(s)

def _compute_target_dimensions(tl, tr, br, bl, aspect=None):
    """Compute target width and height.

    If aspect (W / H) is provided, calculates height from max vertical edge
    and scales width = int(height * aspect) to prevent perspective compression.
    Otherwise defaults to image-plane edge lengths.
    """
    w_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    w_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    h_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    h_right = np.hypot(br[0] - tr[0], br[1] - tr[1])

    if aspect is not None:
        aspect_val = _parse_aspect(aspect)
        # Use the maximum apparent height to preserve full resolution
        height = max(1, int(round(max(h_left, h_right))))
        width = max(1, int(round(height * aspect_val)))
    else:
        width = max(1, int(round(w_top)))
        height = max(1, int(round(h_left)))

    return width, height

def rectify(image_path, points_json, aspect=None, out_before="rectify_before.png", out_after="rectify_after.png"):
    """Map four TL-TR-BR-BL corners to an axis-aligned rectangle at (0, 0) and print H."""
    img = io_utils.open_image(image_path)
    img = imgops.apply_exif_orientation(img, io_utils.read_exif_orientation(image_path))
    corners = points.load_points(points_json)["corners"]
    if not points.corner_order_ok(corners, winding="clockwise"):
        raise ValueError(
            "corners must be four distinct (x, y) points in the order "
            "top-left, top-right, bottom-right, bottom-left (clockwise on screen)"
        )
    tl, tr, br, bl = corners
    width, height = _compute_target_dimensions(tl, tr, br, bl, aspect=aspect)
    dst = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=float,
    )
    h_mat = homography.solve_homography(corners, dst)
    out, _valid, _x, _y = warp.warp_image(img, h_mat, (height, width), (0, 0))
    io_utils.save_png(out_before, img)
    io_utils.save_png(out_after, out)
    print(h_mat)
    return h_mat, out

def panorama(left_path, right_path, points_json,
             out_hardcut="panorama_hardcut.png",
             out_blended="panorama_blended.png",
             exposure_gain=False):
    """Stitch a two-photo pair into a panorama, saving hard-cut and blended results."""
    # 1. Load images and apply EXIF orientation
    img_left = io_utils.open_image(left_path)
    img_left = imgops.apply_exif_orientation(img_left, io_utils.read_exif_orientation(left_path))
    img_right = io_utils.open_image(right_path)
    img_right = imgops.apply_exif_orientation(img_right, io_utils.read_exif_orientation(right_path))

    # 2. Load point correspondences: src is on left image, dst is on right image
    pts = points.load_points(points_json)
    src = pts["src"]
    dst = pts["dst"]

    # 3. Fit H mapping right image coordinates (dst) into left image reference frame (src)
    if len(src) == 4:
        H_right = homography.solve_homography(dst, src)
    else:
        H_right = homography.solve_homography_lsq(dst, src)
    H_left = np.eye(3)

    # 4. Canvas bounds for both images
    canvas_shape, offset = warp.canvas_bounds([H_left, H_right], [img_left.shape, img_right.shape])

    # 5. Inverse-warp both images onto the shared canvas
    warped_l, valid_l, x_l, y_l = warp.warp_image(img_left, H_left, canvas_shape, offset)
    warped_r, valid_r, x_r, y_r = warp.warp_image(img_right, H_right, canvas_shape, offset)

    # 6. Compute weights
    weights_l = blend.edge_weights(x_l, y_l, valid_l, img_left.shape)
    weights_r = blend.edge_weights(x_r, y_r, valid_r, img_right.shape)

    # 7. Hard-cut composite: pick the photo with higher weight (clean seam between A and B)
    hard_cut = np.zeros_like(warped_l)
    take_l = valid_l & (~valid_r | (weights_l >= weights_r))
    take_r = valid_r & ~take_l
    hard_cut[take_r] = warped_r[take_r]
    hard_cut[take_l] = warped_l[take_l]

    # 8. Feather blend
    blended = blend.feather_blend(
        [warped_l, warped_r],
        [weights_l, weights_r],
        exposure_gain=exposure_gain,
        valid=[valid_l, valid_r]
    )

    # 8. Save outputs and print H
    io_utils.save_png(out_hardcut, hard_cut)
    io_utils.save_png(out_blended, blended)
    print(H_right)
    return H_right, hard_cut, blended


def rectify_panorama(left_path, right_path, pairs_json, corners_json,
                     corners_on="left",
                     aspect=None,
                     out_hardcut="rectified_panorama_hardcut.png",
                     out_blended="rectified_panorama_blended.png",
                     exposure_gain=False):
    """Stitch two photos and rectify the scene (e.g. whiteboard/screen) in one pass.

    corners_on specifies which image has the 4 clicked rectangular corners ('left' or 'right').
    aspect is optional target width:height (e.g. '16:9' or 1.777) to prevent foreshortening compression.
    """
    # 1. Load images and apply EXIF orientation
    img_left = io_utils.open_image(left_path)
    img_left = imgops.apply_exif_orientation(img_left, io_utils.read_exif_orientation(left_path))
    img_right = io_utils.open_image(right_path)
    img_right = imgops.apply_exif_orientation(img_right, io_utils.read_exif_orientation(right_path))

    # 2. Fit H_right_to_left from overlap pairs: src (left) and dst (right)
    pts = points.load_points(pairs_json)
    src = pts["src"]
    dst = pts["dst"]
    if len(src) == 4:
        H_right_to_left = homography.solve_homography(dst, src)
    else:
        H_right_to_left = homography.solve_homography_lsq(dst, src)

    # 3. Load 4 corners of the rectangular object
    corners_data = points.load_points(corners_json)["corners"]
    if not points.corner_order_ok(corners_data, winding="clockwise"):
        raise ValueError(
            "corners must be four distinct (x, y) points in clockwise order: TL, TR, BR, BL"
        )

    # If corners were clicked on the right photo, map them to left image frame
    if corners_on.lower() == "right":
        c_h = np.column_stack((corners_data, np.ones(len(corners_data))))
        mapped = (H_right_to_left @ c_h.T).T
        corners_left = mapped[:, :2] / mapped[:, 2:3]
    else:
        corners_left = corners_data

    # 4. Target dimensions from Euclidean distance or aspect ratio
    tl, tr, br, bl = corners_left
    width, height = _compute_target_dimensions(tl, tr, br, bl, aspect=aspect)
    dst_rect = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ], dtype=float)

    # 5. Compute rectification homography H_rect: maps left coords -> rectified upright coords
    H_rect = homography.solve_homography(corners_left, dst_rect)

    # 6. Composite homographies:
    H_left_total = H_rect
    H_right_total = H_rect @ H_right_to_left

    # Ensure homogeneous scaling:
    H_left_total = H_left_total / H_left_total[2, 2]
    H_right_total = H_right_total / H_right_total[2, 2]

    # 7. Canvas setup:
    # Frame strictly on the rectified target rectangle [0..width, 0..height] of Photo A.
    # Photo B is projected directly onto this rectified frame, yielding a clean rectangular
    # output free of perspective wings or sharp non-rectangular canvas borders.
    canvas_shape = (height, width)
    offset = np.array([0.0, 0.0])

    # 8. Inverse warp both images onto the clean rectified canvas
    warped_l, valid_l, x_l, y_l = warp.warp_image(img_left, H_left_total, canvas_shape, offset)
    warped_r, valid_r, x_r, y_r = warp.warp_image(img_right, H_right_total, canvas_shape, offset)

    # 9. Compute feather weights
    weights_l = blend.edge_weights(x_l, y_l, valid_l, img_left.shape)
    weights_r = blend.edge_weights(x_r, y_r, valid_r, img_right.shape)

    # 10. Hard-cut composite: assign pixels to the photo with higher edge weight (clean Voronoi seam)
    hard_cut = np.zeros_like(warped_l)
    take_l = valid_l & (~valid_r | (weights_l >= weights_r))
    take_r = valid_r & ~take_l
    hard_cut[take_r] = warped_r[take_r]
    hard_cut[take_l] = warped_l[take_l]

    # 11. Feather blend
    blended = blend.feather_blend(
        [warped_l, warped_r],
        [weights_l, weights_r],
        exposure_gain=exposure_gain,
        valid=[valid_l, valid_r]
    )

    # 11. Save outputs and print H_rect
    io_utils.save_png(out_hardcut, hard_cut)
    io_utils.save_png(out_blended, blended)
    print("Rectification H:\n", H_rect)
    print("Combined Right H:\n", H_right_total)
    return H_rect, hard_cut, blended


USAGE = (
    "usage:\n"
    "  python demo.py\n"
    "  python demo.py rectify IMAGE CORNERS_JSON [ASPECT]\n"
    "  python demo.py panorama LEFT_IMAGE RIGHT_IMAGE PAIRS_JSON\n"
    "  python demo.py rectify_panorama LEFT_IMAGE RIGHT_IMAGE PAIRS_JSON CORNERS_JSON [corners_on=left|right] [ASPECT]"
)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        synthetic_demo()
    elif args[0] == "rectify" and len(args) in (3, 4):
        aspect = args[3] if len(args) == 4 else None
        rectify(args[1], args[2], aspect=aspect)
    elif args[0] == "panorama" and len(args) == 4:
        panorama(args[1], args[2], args[3])
    elif args[0] == "rectify_panorama" and len(args) in (5, 6, 7):
        corners_on = "left"
        aspect = None
        for extra in args[5:]:
            if extra.lower() in ("left", "right"):
                corners_on = extra.lower()
            else:
                aspect = extra
        rectify_panorama(args[1], args[2], args[3], args[4], corners_on=corners_on, aspect=aspect)
    else:
        print(USAGE)
        sys.exit(2)
