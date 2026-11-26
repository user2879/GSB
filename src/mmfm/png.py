"""Small deterministic PNG codec for RGB/RGBA 8-bit non-interlaced images.

It deliberately depends only on Python's standard library and NumPy so that
the renderer does not require Pillow or a system image library.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterator

import numpy as np

from .errors import E_PNG_CORRUPT, E_PNG_DIMENSION, E_PNG_MODE, E_PNG_SIGNATURE, fail

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def encode_png_rgb(pixels: np.ndarray) -> bytes:
    array = np.asarray(pixels)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        fail(E_PNG_MODE, "PNG writer requires uint8 RGB pixels", shape=list(array.shape), dtype=str(array.dtype))
    height, width, _ = array.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = bytearray()
    for row in array:
        rows.append(0)
        rows.extend(row.tobytes(order="C"))
    compressed = zlib.compress(bytes(rows), level=9)
    return b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", compressed),
            _chunk(b"IEND", b""),
        ]
    )


def _iter_chunks(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            fail(E_PNG_CORRUPT, "truncated PNG chunk header", offset=offset)
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            fail(E_PNG_CORRUPT, "truncated PNG chunk", chunk=kind.decode("latin-1", errors="replace"))
        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        actual_crc = zlib.crc32(kind)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            fail(
                E_PNG_CORRUPT,
                "PNG chunk CRC mismatch",
                chunk=kind.decode("latin-1", errors="replace"),
                expected=expected_crc,
                actual=actual_crc,
            )
        yield kind, payload
        offset = crc_end


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


def _unfilter(raw: bytes, height: int, width: int, channels: int) -> np.ndarray:
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        fail(
            E_PNG_CORRUPT,
            "unexpected PNG decompressed byte count",
            expected=expected,
            actual=len(raw),
        )
    output = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = np.frombuffer(raw[offset : offset + stride], dtype=np.uint8).copy()
        offset += stride
        reconstructed = np.zeros(stride, dtype=np.uint8)
        for x in range(stride):
            left = int(reconstructed[x - channels]) if x >= channels else 0
            up = int(previous[x])
            upper_left = int(previous[x - channels]) if x >= channels else 0
            value = int(scan[x])
            if filter_type == 0:
                result = value
            elif filter_type == 1:
                result = value + left
            elif filter_type == 2:
                result = value + up
            elif filter_type == 3:
                result = value + ((left + up) // 2)
            elif filter_type == 4:
                result = value + _paeth(left, up, upper_left)
            else:
                fail(E_PNG_CORRUPT, "unknown PNG scanline filter", filter_type=filter_type, row=y)
            reconstructed[x] = result & 0xFF
        output[y] = reconstructed
        previous = reconstructed
    return output.reshape((height, width, channels))


def decode_png(
    data: bytes,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_mode: str | None = None,
) -> np.ndarray:
    if not data.startswith(PNG_SIGNATURE):
        fail(E_PNG_SIGNATURE, "file does not begin with the PNG signature")
    chunks = list(_iter_chunks(data))
    if not chunks or chunks[0][0] != b"IHDR":
        fail(E_PNG_CORRUPT, "PNG must begin with IHDR")
    if chunks[-1][0] != b"IEND":
        fail(E_PNG_CORRUPT, "PNG must end with IEND")
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        fail(E_PNG_CORRUPT, "IHDR payload must contain 13 bytes", actual=len(ihdr))
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width <= 0 or height <= 0:
        fail(E_PNG_DIMENSION, "PNG dimensions must be positive", width=width, height=height)
    if expected_width is not None and width != expected_width:
        fail(
            E_PNG_DIMENSION,
            "PNG width differs from configured resolution",
            expected=expected_width,
            actual=width,
        )
    if expected_height is not None and height != expected_height:
        fail(
            E_PNG_DIMENSION,
            "PNG height differs from configured resolution",
            expected=expected_height,
            actual=height,
        )
    if bit_depth != 8:
        fail(E_PNG_MODE, "only 8-bit PNG samples are supported", bit_depth=bit_depth)
    if color_type not in {2, 6}:
        fail(E_PNG_MODE, "only true-colour RGB/RGBA PNGs are supported", color_type=color_type)
    channels = 3 if color_type == 2 else 4
    mode = "RGB" if channels == 3 else "RGBA"
    if expected_mode is not None and mode != expected_mode:
        fail(
            E_PNG_MODE,
            "PNG pixel mode differs from configured mode",
            expected=expected_mode,
            actual=mode,
        )
    if compression != 0 or filter_method != 0 or interlace != 0:
        fail(
            E_PNG_MODE,
            "PNG compression, filter, and interlace methods must be zero",
            compression=compression,
            filter_method=filter_method,
            interlace=interlace,
        )
    idat = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    if not idat:
        fail(E_PNG_CORRUPT, "PNG contains no IDAT data")
    try:
        raw = zlib.decompress(idat)
    except zlib.error as exc:
        fail(E_PNG_CORRUPT, "PNG IDAT stream cannot be decompressed", error=str(exc))
    return _unfilter(raw, height, width, channels)


def decode_png_rgb(
    data: bytes,
    *,
    width: int,
    height: int,
    mode: str = "RGB",
) -> np.ndarray:
    pixels = decode_png(data, expected_width=width, expected_height=height, expected_mode=mode)
    if mode == "RGBA":
        return pixels[:, :, :3].copy()
    return pixels
