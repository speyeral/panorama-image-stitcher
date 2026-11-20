"""File I/O only. Pillow opens, reads EXIF, saves PNG, and shows. Pixels become NumPy."""

from os import PathLike
from typing import Any, Final, TypeAlias
from PIL import Image
from PIL.Image import Exif
import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- Type Aliases ---

FilePath: TypeAlias = str | PathLike[str]
"""Filesystem path to an image file.

Examples:
    `"photos/input.jpg"` or `Path("photos/input.jpg")`
"""

ImageArray: TypeAlias = NDArray[np.uint8]
"""Standard 8-bit image array (`0–255`), shape `(H, W, 3)` for RGB or `(H, W)` for grayscale (`L`).

Examples:
    `np.zeros((1080, 1920, 3), dtype=np.uint8)` or `np.zeros((480, 640), dtype=np.uint8)`
"""

# --- Constants ---

_ORIENTATION_TAG: Final[int] = 274
"""EXIF tag ID `274` (`0x0112`), which stores camera orientation codes (`1` through `8`)."""


def open_image(path: FilePath) -> ImageArray:
    """Open with Pillow, then return a writable uint8 array (RGB or L)."""
    with Image.open(path) as pil_im:
        converted: Image.Image
        if pil_im.mode in ("L", "RGB"):
            converted = pil_im
        else:
            converted = pil_im.convert("RGB")
        return np.array(converted, dtype=np.uint8)


def read_exif_orientation(path: FilePath) -> int:
    """Return EXIF Orientation in 1..8. Missing or invalid tags become 1."""
    value: Any
    with Image.open(path) as pil_im:
        exif: Exif | None = pil_im.getexif()
        value = exif.get(_ORIENTATION_TAG) if exif else None
    if value is None:
        return 1
    orientation: int
    try:
        orientation = int(value)
    except (TypeError, ValueError):
        return 1
    if orientation < 1 or orientation > 8:
        return 1
    return orientation


def save_png(path: FilePath, img: ArrayLike) -> None:
    """Write a uint8 array as PNG. 2D is L, HxWx3 is RGB."""
    arr: NDArray[Any] = np.array(img)
    if arr.dtype != np.uint8 or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3)):
        raise ValueError("save_png expects HxW or HxWx3 uint8")
    # Pillow infers mode L (2D) or RGB (HxWx3) from a uint8 array.
    Image.fromarray(arr).save(path, format="PNG")


def show(img: ArrayLike) -> None:
    """Show with Pillow (Preview on macOS)."""
    arr: NDArray[Any] = np.array(img)
    if arr.dtype != np.uint8 or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3)):
        raise ValueError("show expects HxW or HxWx3 uint8")
    Image.fromarray(arr).show()