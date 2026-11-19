from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, color_type: int, rows: bytes) -> None:
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows, 9)) + _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def premultiplied_bgra_to_rgba(pixels: np.ndarray) -> np.ndarray:
    alpha = pixels[:, :, 3].astype(np.uint32)
    rgba = np.zeros_like(pixels)
    rgba[:, :, 3] = pixels[:, :, 3]
    nonzero = alpha > 0
    for source, target in ((0, 2), (1, 1), (2, 0)):
        values = pixels[:, :, source].astype(np.uint32)
        straight = np.zeros_like(values)
        straight[nonzero] = (values[nonzero] * 255 + alpha[nonzero] // 2) // alpha[nonzero]
        rgba[:, :, target] = np.minimum(straight, 255).astype(np.uint8)
    return rgba


def write_bgra_png(path: Path, pixels: np.ndarray) -> None:
    rgba = premultiplied_bgra_to_rgba(pixels)
    rows = b"".join(b"\x00" + rgba[y].tobytes() for y in range(rgba.shape[0]))
    _write_png(path, rgba.shape[1], rgba.shape[0], 6, rows)


def write_coverage_png(path: Path, coverage: np.ndarray) -> None:
    rows = b"".join(b"\x00" + coverage[y].astype(np.uint8).tobytes() for y in range(coverage.shape[0]))
    _write_png(path, coverage.shape[1], coverage.shape[0], 0, rows)
