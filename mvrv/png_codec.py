from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png_rgb(path: str | Path, rgb: np.ndarray) -> None:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    height, width, _ = array.shape
    rows = [b"".join((b"\x00", array[y].tobytes())) for y in range(height)]
    compressed = zlib.compress(b"".join(rows), 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


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


def read_png_rgb(path: str | Path, verify_crc: bool = True) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise PngError("E_PNG_SIGNATURE", "missing or corrupt PNG signature")

    offset = 8
    width = height = 0
    bit_depth = color_type = 0
    idat = bytearray()
    saw_ihdr = False

    while offset < len(data):
        if offset + 8 > len(data):
            raise PngError("E_PNG_CONTAINER", "truncated PNG chunk header")
        length, = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        body_start = offset + 8
        body_end = body_start + length
        crc_end = body_end + 4
        if crc_end > len(data):
            raise PngError("E_PNG_CONTAINER", "truncated PNG chunk")
        body = data[body_start:body_end]
        expected_crc, = struct.unpack(">I", data[body_end:crc_end])
        if verify_crc:
            actual_crc = zlib.crc32(chunk_type + body) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise PngError("E_PNG_CRC", f"CRC mismatch in {chunk_type!r}")
        if chunk_type == b"IHDR":
            if length != 13:
                raise PngError("E_PNG_HEADER", "IHDR has invalid length")
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", body)
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise PngError("E_PNG_UNSUPPORTED", "unsupported PNG compression/filter/interlace")
            saw_ihdr = True
        elif chunk_type == b"IDAT":
            idat.extend(body)
        elif chunk_type == b"IEND":
            break
        offset = crc_end

    if not saw_ihdr:
        raise PngError("E_PNG_HEADER", "missing IHDR")
    if bit_depth != 8 or color_type != 2:
        raise PngError("E_PNG_MODE", "only 8-bit RGB PNG is accepted")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise PngError("E_PNG_IDAT", f"invalid IDAT stream: {exc}") from exc

    channels = 3
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise PngError("E_PNG_IDAT", f"raw PNG size {len(raw)} != {expected}")

    image = np.zeros((height, width, channels), dtype=np.uint8)
    previous = bytearray(stride)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        if filter_type == 1:
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif filter_type == 2:
            for x in range(stride):
                line[x] = (line[x] + previous[x]) & 0xFF
        elif filter_type == 3:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + previous[x]) // 2)) & 0xFF
        elif filter_type == 4:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                above_left = previous[x - channels] if x >= channels else 0
                line[x] = (
                    line[x] + _paeth(left, previous[x], above_left)
                ) & 0xFF
        elif filter_type != 0:
            raise PngError("E_PNG_FILTER", f"unsupported PNG filter {filter_type}")
        image[y] = np.frombuffer(bytes(line), dtype=np.uint8).reshape(width, channels)
        previous = line
    return image
