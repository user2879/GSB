"""Minimal deterministic PNG writer (RGB8, no external dependencies)."""
import struct
import zlib

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def write_png(path: str, rgb: np.ndarray) -> None:
    """rgb: (H, W, 3) uint8.  Deterministic bytes (fixed zlib level)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))
    data = (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, 9))
            + _chunk(b"IEND", b""))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    import os
    os.replace(tmp, path)
