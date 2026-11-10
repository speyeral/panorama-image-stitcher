"""File I/O only. Pillow opens, reads EXIF, saves PNG, and shows. Pixels become NumPy."""

from PIL import Image
import numpy as np

# EXIF Orientation tag id
_ORIENTATION_TAG = 274


def open_image(path):
    """Open with Pillow, then return a writable uint8 array (RGB or L)."""
    with Image.open(path) as pil_im:
        if pil_im.mode in ("L", "RGB"):
            converted = pil_im
        else:
            converted = pil_im.convert("RGB")
        return np.array(converted)


def read_exif_orientation(path):
    """Return EXIF Orientation in 1..8. Missing or invalid tags become 1."""
    with Image.open(path) as pil_im:
        exif = pil_im.getexif()
        value = exif.get(_ORIENTATION_TAG) if exif else None
    if value is None:
        return 1
    try:
        orientation = int(value)
    except (TypeError, ValueError):
        return 1
    if orientation < 1 or orientation > 8:
        return 1
    return orientation


def save_png(path, img):
    """Write a uint8 array as PNG. 2D is L, HxWx3 is RGB."""
    arr = np.array(img)
    if arr.dtype != np.uint8 or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3)):
        raise ValueError("save_png expects HxW or HxWx3 uint8")
    # Pillow infers mode L (2D) or RGB (HxWx3) from a uint8 array.
    Image.fromarray(arr).save(path, format="PNG")


def show(img):
    """Show with Pillow (Preview on macOS)."""
    arr = np.array(img)
    if arr.dtype != np.uint8 or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3)):
        raise ValueError("show expects HxW or HxWx3 uint8")
    Image.fromarray(arr).show()
