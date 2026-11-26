from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class GrayImage:
    width: int
    height: int
    pixels: np.ndarray
    bit_depth: int = 8
    color_type: int = 0

    @property
    def mode(self) -> str:
        if self.bit_depth == 8 and self.color_type == 0:
            return "L"
        return f"CT{self.color_type}-BD{self.bit_depth}"


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def encode_png(pixels: np.ndarray) -> bytes:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("encode_png expects a 2-D grayscale array")
    height, width = array.shape
    raw = bytearray()
    for row in array:
        raw.append(0)
        raw.extend(row.tobytes())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", zlib.compress(bytes(raw), 6)),
            _chunk(b"IEND", b""),
        ]
    )


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


def decode_png(data: bytes, verify_crc: bool = True) -> GrayImage:
    if not data.startswith(PNG_SIGNATURE):
        raise PngDecodeError("bad PNG container signature")
    pos = len(PNG_SIGNATURE)
    width = height = 0
    bit_depth = color_type = 0
    idat = bytearray()
    saw_iend = False
    chunks = 0

    while pos < len(data):
        if pos + 8 > len(data):
            raise PngDecodeError("truncated PNG chunk header")
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        chunk_type = data[pos + 4 : pos + 8]
        body_start = pos + 8
        body_end = body_start + length
        crc_end = body_end + 4
        if crc_end > len(data):
            raise PngDecodeError(f"truncated {chunk_type!r} chunk")
        body = data[body_start:body_end]
        crc_stored = struct.unpack(">I", data[body_end:crc_end])[0]
        if verify_crc:
            crc_actual = zlib.crc32(chunk_type + body) & 0xFFFFFFFF
            if crc_stored != crc_actual:
                raise PngDecodeError(f"CRC mismatch in {chunk_type!r}")
        pos = crc_end
        chunks += 1

        if chunk_type == b"IHDR":
            if length != 13:
                raise PngDecodeError("IHDR length must be 13")
            width, height, bit_depth, color_type, compression, filt, interlace = (
                struct.unpack(">IIBBBBB", body)
            )
            if compression != 0 or filt != 0:
                raise PngDecodeError("unsupported PNG compression/filter method")
            if interlace != 0:
                raise PngDecodeError("interlaced PNG is not supported")
        elif chunk_type == b"IDAT":
            idat.extend(body)
        elif chunk_type == b"IEND":
            saw_iend = True
            break

    if chunks == 0 or width <= 0 or height <= 0:
        raise PngDecodeError("missing IHDR")
    if not saw_iend:
        raise PngDecodeError("missing IEND / truncated data stream")
    if bit_depth != 8 or color_type != 0:
        raise PngDecodeError("only 8-bit grayscale PNG is supported")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise PngDecodeError(f"invalid IDAT zlib stream: {exc}") from exc

    stride = width
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise PngDecodeError("decompressed PNG data is truncated")
    raw = raw[:expected]
    pixels = np.zeros((height, width), dtype=np.uint8)
    previous = np.zeros(width, dtype=np.uint8)
    src = 0
    for y in range(height):
        filter_type = raw[src]
        src += 1
        line = np.frombuffer(raw[src : src + stride], dtype=np.uint8).copy()
        src += stride
        if filter_type == 0:
            reconstructed = line
        elif filter_type == 1:
            for x in range(width):
                left = int(reconstructed[x - 1]) if x else 0
                line[x] = (int(line[x]) + left) & 0xFF
            reconstructed = line
        elif filter_type == 2:
            reconstructed = (line.astype(np.uint16) + previous.astype(np.uint16)).astype(np.uint8)
        elif filter_type == 3:
            for x in range(width):
                left = int(reconstructed[x - 1]) if x else 0
                up = int(previous[x])
                line[x] = (int(line[x]) + (left + up) // 2) & 0xFF
            reconstructed = line
        elif filter_type == 4:
            for x in range(width):
                a = int(reconstructed[x - 1]) if x else 0
                b = int(previous[x])
                c = int(previous[x - 1]) if x else 0
                line[x] = (int(line[x]) + _paeth(a, b, c)) & 0xFF
            reconstructed = line
        else:
            raise PngDecodeError(f"unsupported row filter {filter_type}")
        pixels[y] = reconstructed
        previous = reconstructed
    return GrayImage(width, height, pixels)
