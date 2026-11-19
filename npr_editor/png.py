"""Small deterministic RGBA PNG encoder (no external image dependency)."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

from .errors import require


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(image: np.ndarray) -> bytes:
    require(
        isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3,
        "E_RENDER_MISMATCH",
        "PNG input must be HxWx4 uint8",
    )
    require(
        image.shape[2] == 4,
        "E_RENDER_MISMATCH",
        "PNG encoder requires four channels",
        shape=image.shape,
    )
    height, width, _ = image.shape
    require(height > 0 and width > 0, "E_RENDER_MISMATCH", "cannot encode empty image")
    raw = b"".join(b"\x00" + image[row].tobytes(order="C") for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, level=9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: str | Path, image: np.ndarray) -> dict[str, str]:
    data = encode_png(image)
    target = Path(path)
    target.write_bytes(data)
    return image_hashes(image, data)


def image_hashes(image: np.ndarray, png_bytes: bytes | None = None) -> dict[str, str]:
    pixel_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes(order="C")).hexdigest()
    result = {"pixel_sha256": pixel_hash}
    if png_bytes is not None:
        result["png_sha256"] = hashlib.sha256(png_bytes).hexdigest()
    return result


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    require(
        left.shape == right.shape and left.ndim == 3 and left.shape[2] == 4,
        "E_RENDER_MISMATCH",
        "side-by-side images must have identical RGBA shapes",
    )
    height, width, _ = left.shape
    canvas = np.zeros((height, width * 2 + gap, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    canvas[:, :width] = left
    canvas[:, width : width + gap] = np.array([90, 90, 90, 255], dtype=np.uint8)
    canvas[:, width + gap :] = right
    return canvas
