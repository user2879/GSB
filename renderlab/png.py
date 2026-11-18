"""Minimal deterministic RGB PNG encoder/decoder (no Pillow dependency)."""

from __future__ import annotations

import struct
import zlib

import numpy as np


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_rgb_png(pixels: np.ndarray, *, compression_level: int = 9) -> bytes:
    """Encode an HxWx3 uint8 array as a deterministic RGB PNG."""

    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise ValueError("pixels must have dtype uint8")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    height, width, _ = array.shape
    if height <= 0 or width <= 0:
        raise ValueError("pixels must be non-empty")

    scanlines = bytearray()
    contiguous = np.ascontiguousarray(array)
    for row in contiguous:
        scanlines.append(0)  # PNG filter type 0 (None).
        scanlines.extend(row.tobytes(order="C"))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), level=compression_level)
    return _PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    pa = abs(estimate - left)
    pb = abs(estimate - above)
    pc = abs(estimate - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return above
    return upper_left


def decode_rgb_png(payload: bytes) -> np.ndarray:
    """Decode the 8-bit RGB PNG subset produced by this project.

    All five standard PNG row filters are supported so external comparison
    tools can also open the output.
    """

    if not payload.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    offset = len(_PNG_SIGNATURE)
    width = height = 0
    idat_parts: list[bytes] = []
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        body = payload[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"bad CRC for {kind!r}")
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if (bit_depth, color_type, comp, filt, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("only non-interlaced 8-bit RGB PNG is supported")
        elif kind == b"IDAT":
            idat_parts.append(body)
        elif kind == b"IEND":
            break

    raw = zlib.decompress(b"".join(idat_parts))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("unexpected PNG scanline length")
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    previous = bytearray(stride)
    position = 0
    for row_index in range(height):
        filter_type = raw[position]
        position += 1
        scanline = bytearray(raw[position : position + stride])
        position += stride
        for column in range(stride):
            left = scanline[column - 3] if column >= 3 else 0
            above = previous[column]
            upper_left = previous[column - 3] if column >= 3 else 0
            value = scanline[column]
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = (value + left) & 0xFF
            elif filter_type == 2:
                decoded = (value + above) & 0xFF
            elif filter_type == 3:
                decoded = (value + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                decoded = (value + _paeth(left, above, upper_left)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}")
            scanline[column] = decoded
        pixels[row_index] = np.frombuffer(bytes(scanline), dtype=np.uint8).reshape(width, 3)
        previous = scanline
    return pixels
