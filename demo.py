import sys
import numpy as np
import homography
import warp
import io_utils
import imgops
import points

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

def rectify(image_path, points_json, out_before="rectify_before.png", out_after="rectify_after.png"):
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
    width = max(1, int(round(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))))
    height = max(1, int(round(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))))
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


USAGE = "usage: python demo.py [rectify IMAGE POINTS_JSON]"


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        synthetic_demo()
    elif args[0] == "rectify" and len(args) == 3:
        rectify(args[1], args[2])
    else:
        print(USAGE)
        sys.exit(2)