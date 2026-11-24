from __future__ import annotations

"""Small dependency-free PNG codec for deterministic RGB sample frames.

Only non-interlaced, 8-bit grayscale/RGB/RGBA images are needed by the sample
generator and this component.  Supporting the five standard scanline filters is
enough to read ordinary PNG files as well.
"""

import struct
import zlib
from pathlib import Path

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png_rgb(path: str | Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("PNG image must have shape (H, W, 1|3|4)")
    channels = array.shape[2]
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    height, width = array.shape[:2]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    encoded = zlib.compress(raw, level=9)
    payload = (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", encoded)
        + _chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    offset = len(PNG_SIGNATURE)
    ihdr = None
    compressed = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])[0]
        if (zlib.crc32(kind + payload) & 0xFFFFFFFF) != crc:
            raise ValueError(f"PNG CRC mismatch in {kind!r}")
        offset += 12 + length
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if ihdr is None:
        raise ValueError("PNG has no IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if bit_depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError("only non-interlaced 8-bit PNG files are supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"unsupported PNG color type {color_type}")
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ValueError("invalid PNG IDAT stream") from exc
    stride = width * channels
    row_size = stride + 1
    if len(raw) != height * row_size:
        raise ValueError("unexpected PNG payload size")
    result = np.empty((height, width, channels), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=cursor).copy()
        cursor += stride
        if filter_type == 0:
            current = scan
        elif filter_type == 1:
            current = np.empty_like(scan)
            for index in range(stride):
                left = current[index - channels] if index >= channels else 0
                current[index] = (int(scan[index]) + int(left)) & 0xFF
        elif filter_type == 2:
            current = (scan.astype(np.uint16) + previous.astype(np.uint16)).astype(np.uint8)
        elif filter_type == 3:
            current = np.empty_like(scan)
            for index in range(stride):
                left = int(current[index - channels]) if index >= channels else 0
                above = int(previous[index])
                current[index] = (int(scan[index]) + ((left + above) // 2)) & 0xFF
        elif filter_type == 4:
            current = np.empty_like(scan)
            for index in range(stride):
                left = int(current[index - channels]) if index >= channels else 0
                above = int(previous[index])
                upper_left = int(previous[index - channels]) if index >= channels else 0
                current[index] = (int(scan[index]) + _paeth(left, above, upper_left)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter {filter_type}")
        result[row] = current.reshape(width, channels)
        previous = current
    if channels == 2:
        result = result[:, :, :1]
    return result
