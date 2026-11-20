import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np


# Ensure inputs directory exists
INPUTS_DIR = Path("inputs")
INPUTS_DIR.mkdir(exist_ok=True)


def _save_json(filename: str, data: dict[str, Any]) -> Path:
    """Save point data into the inputs/ folder and return its path."""
    out_path = INPUTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {out_path}")
    return out_path


def _load_image_bgr(image_path: str) -> np.ndarray:
    """Load an image in BGR format for OpenCV display."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


# =====================================================================
# 1. RECTIFY INTERACTIVE PICKER (4 Draggable Corners: TL, TR, BR, BL)
# =====================================================================
def pick_corners_ui(image_path: str, json_filename: str = "picked_corners.json") -> Path:
    """Opens a window with 4 draggable corners (TL, TR, BR, BL). Press ENTER/SPACE to confirm."""
    img = _load_image_bgr(image_path)
    h, w = img.shape[:2]

    # Initialize 4 corners inset at 20% from the borders in clockwise order: TL, TR, BR, BL
    labels = ["TL", "TR", "BR", "BL"]
    corners = np.array(
        [
            [w * 0.2, h * 0.2],  # TL
            [w * 0.8, h * 0.2],  # TR
            [w * 0.8, h * 0.8],  # BR
            [w * 0.2, h * 0.8],  # BL
        ],
        dtype=np.float64,
    )

    state = {"dragging_idx": None}
    win_name = "Rectify Picker - Drag Corners (TL, TR, BR, BL)"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest corner within 25 pixels
            dists = np.linalg.norm(corners - np.array([x, y]), axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < 25.0:
                state["dragging_idx"] = min_idx
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging_idx"] is not None:
            # Clamp inside image bounds
            cx = float(np.clip(x, 0, w - 1))
            cy = float(np.clip(y, 0, h - 1))
            corners[state["dragging_idx"]] = [cx, cy]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging_idx"] = None

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = img.copy()

        # Draw semi-transparent polygon connecting TL -> TR -> BR -> BL
        pts_int = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw draggable handles and labels
        for i, (cx, cy) in enumerate(corners):
            center = (int(round(cx)), int(round(cy)))
            cv2.circle(canvas, center, 8, (0, 0, 255), -1)
            cv2.circle(canvas, center, 10, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                labels[i],
                (center[0] + 12, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Top instruction banner
        cv2.rectangle(canvas, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(
            canvas,
            "Drag TL, TR, BR, BL | [ENTER / SPACE] Confirm & Run | [R] Reset | [ESC] Quit",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # ENTER or SPACE
            break
        elif key in (ord("r"), ord("R")):
            corners[:] = [
                [w * 0.2, h * 0.2],
                [w * 0.8, h * 0.2],
                [w * 0.8, h * 0.8],
                [w * 0.2, h * 0.8],
            ]
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow(win_name)

    data = {"corners": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in corners]}
    return _save_json(json_filename, data)


# =====================================================================
# 2. PANORAMA INTERACTIVE PICKER (2 Windows, Linked Point Pairs)
# =====================================================================
def pick_pairs_ui(
    img1_path: str,
    img2_path: str,
    json_filename: str = "picked_pairs.json",
) -> tuple[str, str, Path]:
    """Opens two windows with linked draggable point pairs and Left/Right role toggling.

    Returns (left_image_path, right_image_path, saved_pairs_json_path).
    """
    img1 = _load_image_bgr(img1_path)
    img2 = _load_image_bgr(img2_path)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Start with 4 linked default pairs in the overlap area so counts are always equal
    pts1: list[list[float]] = [
        [w1 * 0.65, h1 * 0.25],
        [w1 * 0.85, h1 * 0.25],
        [w1 * 0.85, h1 * 0.75],
        [w1 * 0.65, h1 * 0.75],
    ]
    pts2: list[list[float]] = [
        [w2 * 0.15, h2 * 0.25],
        [w2 * 0.35, h2 * 0.25],
        [w2 * 0.35, h2 * 0.75],
        [w2 * 0.15, h2 * 0.75],
    ]

    state: dict[str, Any] = {
        "img1_is_left": True,  # Toggle with [S] key
        "drag_win": None,      # 1 or 2
        "drag_idx": None,      # index of point being dragged
    }

    win1 = "Window 1 (Image 1)"
    win2 = "Window 2 (Image 2)"

    def make_mouse_cb(win_id: int):
        pts_list = pts1 if win_id == 1 else pts2
        max_w = w1 if win_id == 1 else w2
        max_h = h1 if win_id == 1 else h2

        def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(pts_list) > 0:
                arr = np.array(pts_list)
                dists = np.linalg.norm(arr - np.array([x, y]), axis=1)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < 25.0:
                    state["drag_win"] = win_id
                    state["drag_idx"] = min_idx
            elif event == cv2.EVENT_MOUSEMOVE and state["drag_win"] == win_id:
                idx = state["drag_idx"]
                pts_list[idx] = [float(np.clip(x, 0, max_w - 1)), float(np.clip(y, 0, max_h - 1))]
            elif event == cv2.EVENT_LBUTTONUP:
                state["drag_win"] = None
                state["drag_idx"] = None

        return on_mouse

    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win1, make_mouse_cb(1))
    cv2.setMouseCallback(win2, make_mouse_cb(2))

    # Color palette to distinguish point pairs visually
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 100, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 255),
        (0, 165, 255),
    ]

    def render_window(base_img: np.ndarray, pts_list: list[list[float]], role_text: str) -> np.ndarray:
        canvas = base_img.copy()
        h, w = canvas.shape[:2]

        for i, (px, py) in enumerate(pts_list):
            color = colors[i % len(colors)]
            center = (int(round(px)), int(round(py)))
            cv2.circle(canvas, center, 8, color, -1)
            cv2.circle(canvas, center, 10, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                f"#{i + 1}",
                (center[0] + 12, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # Header banner with controls & Left/Right role indicator
        cv2.rectangle(canvas, (0, 0), (w, 65), (25, 25, 25), -1)
        role_color = (0, 255, 0) if "LEFT" in role_text else (0, 200, 255)
        cv2.putText(canvas, role_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, role_color, 2)
        cv2.putText(
            canvas,
            f"Pairs: {len(pts_list)} | [A] Add Pair | [D] Del Last | [S] Swap Left/Right | [ENTER] Confirm",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )
        return canvas

    while True:
        role1 = "ROLE: LEFT (Anchor / src)" if state["img1_is_left"] else "ROLE: RIGHT (Warped / dst)"
        role2 = "ROLE: RIGHT (Warped / dst)" if state["img1_is_left"] else "ROLE: LEFT (Anchor / src)"

        cv2.imshow(win1, render_window(img1, pts1, role1))
        cv2.imshow(win2, render_window(img2, pts2, role2))

        key = cv2.waitKey(20) & 0xFF

        if key in (ord("a"), ord("A")):
            # Add a linked point to BOTH windows simultaneously so counts stay identical
            offset = (len(pts1) % 5) * 25.0
            pts1.append([min(w1 - 20.0, w1 * 0.5 + offset), min(h1 - 20.0, h1 * 0.5 + offset)])
            pts2.append([min(w2 - 20.0, w2 * 0.5 + offset), min(h2 - 20.0, h2 * 0.5 + offset)])
        elif key in (ord("d"), ord("D")):
            # Delete last pair from BOTH windows (keep at least 4 points)
            if len(pts1) > 4:
                pts1.pop()
                pts2.pop()
            else:
                print("[Info] Minimum of 4 point pairs required.")
        elif key in (ord("s"), ord("S")):
            # Toggle which image is Left (anchor/src) vs Right (dst)
            state["img1_is_left"] = not state["img1_is_left"]
        elif key in (13, 32):  # ENTER or SPACE
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Assign Left -> src and Right -> dst to match demo.py's exact convention
    if state["img1_is_left"]:
        left_path, right_path = img1_path, img2_path
        src_pts, dst_pts = pts1, pts2
    else:
        left_path, right_path = img2_path, img1_path
        src_pts, dst_pts = pts2, pts1

    data = {
        "src": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in src_pts],
        "dst": [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in dst_pts],
    }
    saved_json = _save_json(json_filename, data)
    return left_path, right_path, saved_json


# =====================================================================
# 3. CLI ENTRYPOINT (Calls demo.py via subprocess)
# =====================================================================
USAGE = """
Interactive OpenCV Point Picker Wrapper for demo.py

Usage:
  python run_demo_with_cv.py rectify <IMAGE> [ASPECT]
  python run_demo_with_cv.py panorama <IMAGE_1> <IMAGE_2>
  python run_demo_with_cv.py rectify_panorama <IMAGE_1> <IMAGE_2> [corners_on=left|right] [ASPECT]

Notes:
  - Images should be inside the inputs/ folder (e.g., inputs/board.png).
  - Generated JSON files are saved in inputs/ and passed automatically to demo.py.
"""


def _run_demo_subprocess(cmd_args: list[str]) -> None:
    """Invoke demo.py using the exact same Python interpreter."""
    full_cmd = [sys.executable, "demo.py", *cmd_args]
    print(f"\n[Executing] {' '.join(full_cmd)}\n")
    subprocess.run(full_cmd, check=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(USAGE)
        sys.exit(1)

    mode = args[0]

    if mode == "rectify" and len(args) in (2, 3):
        img_path = args[1]
        stem = Path(img_path).stem
        corners_json = pick_corners_ui(img_path, json_filename=f"{stem}_corners.json")

        demo_args = ["rectify", img_path, str(corners_json)]
        if len(args) == 3:
            demo_args.append(args[2])  # Optional ASPECT
        _run_demo_subprocess(demo_args)

    elif mode == "panorama" and len(args) == 3:
        img1_path, img2_path = args[1], args[2]
        stem = f"{Path(img1_path).stem}_{Path(img2_path).stem}"
        left_path, right_path, pairs_json = pick_pairs_ui(
            img1_path, img2_path, json_filename=f"{stem}_pairs.json"
        )

        demo_args = ["panorama", left_path, right_path, str(pairs_json)]
        _run_demo_subprocess(demo_args)

    elif mode == "rectify_panorama" and len(args) in (3, 4, 5):
        img1_path, img2_path = args[1], args[2]
        corners_on = "left"
        aspect_arg = None

        for extra in args[3:]:
            if extra.lower() in ("left", "right"):
                corners_on = extra.lower()
            else:
                aspect_arg = extra

        stem = f"{Path(img1_path).stem}_{Path(img2_path).stem}"
        # Step 1: Pick linked overlap pairs and confirm which image is Left vs Right
        left_path, right_path, pairs_json = pick_pairs_ui(
            img1_path, img2_path, json_filename=f"{stem}_pairs.json"
        )

        # Step 2: Pick the 4 outer corners on whichever image was specified ('left' or 'right')
        corner_target_img = left_path if corners_on == "left" else right_path
        corners_json = pick_corners_ui(corner_target_img, json_filename=f"{stem}_corners.json")

        demo_args = [
            "rectify_panorama",
            left_path,
            right_path,
            str(pairs_json),
            str(corners_json),
            corners_on,
        ]
        if aspect_arg:
            demo_args.append(aspect_arg)
        _run_demo_subprocess(demo_args)

    else:
        print(USAGE)
        sys.exit(2)