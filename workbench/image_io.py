from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import WorkbenchError
from .hashing import array_fingerprint

PNG_SIGNATURE = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))


def validate_png_header(path: str | Path, width: int, height: int, channels: int = 3) -> dict[str, int]:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            prefix = handle.read(33)
    except FileNotFoundError as exc:
        raise WorkbenchError("E_CONFIG_INVALID", f"image not found: {path}", {"path": str(path)}) from exc
    except OSError as exc:
        raise WorkbenchError("E_CONFIG_INVALID", f"cannot read image: {path}: {exc}", {"path": str(path)}) from exc

    if len(prefix) < 26 or prefix[:8] != PNG_SIGNATURE:
        raise WorkbenchError("E_CONFIG_INVALID", "file does not have the PNG container signature", {"path": str(path)})
    chunk_length = struct.unpack(">I", prefix[8:12])[0]
    chunk_type = prefix[12:16]
    if chunk_type != b"IHDR" or chunk_length != 13:
        raise WorkbenchError("E_CONFIG_INVALID", "first PNG chunk must be a 13-byte IHDR chunk", {"path": str(path)})
    actual_w, actual_h = struct.unpack(">II", prefix[16:24])
    bit_depth = prefix[24]
    color_type = prefix[25]
    if (actual_w, actual_h) != (width, height):
        raise WorkbenchError(
            "E_CONFIG_INVALID",
            "PNG IHDR resolution does not match the fixed configured resolution",
            {"expected_width": width, "expected_height": height, "actual_width": actual_w, "actual_height": actual_h},
        )
    expected_color_type = {3: 2, 4: 6}.get(channels)
    if bit_depth != 8 or color_type != expected_color_type:
        raise WorkbenchError(
            "E_CONFIG_INVALID",
            "PNG must be 8-bit RGB/RGBA matching configured channels",
            {"bit_depth": bit_depth, "color_type": color_type, "channels": channels},
        )
    return {"width": actual_w, "height": actual_h, "bit_depth": bit_depth, "color_type": color_type}


def load_rgb_png(path: str | Path, width: int, height: int) -> tuple[np.ndarray, str]:
    validate_png_header(path, width, height, 3)
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.uint8)
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError("E_CONFIG_INVALID", f"Pillow could not decode PNG: {exc}", {"path": str(path)}) from exc
    if pixels.shape != (height, width, 3):
        raise WorkbenchError(
            "E_CONFIG_INVALID",
            "decoded PNG shape does not match fixed resolution",
            {"expected": [height, width, 3], "actual": list(pixels.shape)},
        )
    return pixels, array_fingerprint(pixels)


def write_png(path: str | Path, pixels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels.dtype != np.uint8:
        raise ValueError("write_png expects uint8 RGB pixels")
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", compress_level=6, optimize=False)
