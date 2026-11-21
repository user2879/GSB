"""Minimal deterministic PNG decoder for the project fixtures.

Only non-interlaced, 8-bit RGB/RGBA PNG files are accepted. The decoder checks
the signature, every chunk CRC, IHDR constraints, and the exact decompressed
scanline length. This keeps the example independent of Pillow.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PNGError(ValueError):
    pass


@dataclass(frozen=True)
class PNGImage:
    width: int
    height: int
    pixels: np.ndarray
    color_type: int


def _read_chunks(data: bytes) -> list[tuple[str, bytes]]:
    if len(data) < len(PNG_SIGNATURE) or data[:8] != PNG_SIGNATURE:
        raise PNGError("bad PNG signature")
    offset = 8
    chunks: list[tuple[str, bytes]] = []
    while offset < len(data):
        if offset + 12 > len(data):
            raise PNGError("truncated chunk header")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type_raw = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise PNGError("truncated chunk body")
        try:
            chunk_type = chunk_type_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PNGError("invalid chunk type") from exc
        payload = data[offset + 8 : offset + 8 + length]
        crc_expected = struct.unpack(">I", data[offset + 8 + length : end])[0]
        crc_actual = binascii.crc32(chunk_type_raw + payload) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise PNGError(f"CRC mismatch in {chunk_type}")
        chunks.append((chunk_type, payload))
        offset = end
    if not chunks or chunks[0][0] != "IHDR":
        raise PNGError("IHDR must be the first chunk")
    if chunks[-1][0] != "IEND" or chunks[-1][1]:
        raise PNGError("IEND must terminate the PNG")
    return chunks


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distance_left = abs(estimate - left)
    distance_above = abs(estimate - above)
    distance_upper_left = abs(estimate - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def _unfilter(raw: bytes, width: int, height: int, bytes_per_pixel: int) -> bytes:
    row_length = width * bytes_per_pixel
    expected = height * (row_length + 1)
    if len(raw) != expected:
        raise PNGError(f"scanline length mismatch: {len(raw)} != {expected}")
    output = bytearray(height * row_length)
    source_offset = 0
    for row in range(height):
        filter_type = raw[source_offset]
        source_offset += 1
        scanline = raw[source_offset : source_offset + row_length]
        source_offset += row_length
        target_offset = row * row_length
        previous_offset = (row - 1) * row_length
        for column, value in enumerate(scanline):
            left = output[target_offset + column - bytes_per_pixel] if column >= bytes_per_pixel else 0
            above = output[previous_offset + column] if row > 0 else 0
            upper_left = (
                output[previous_offset + column - bytes_per_pixel]
                if row > 0 and column >= bytes_per_pixel
                else 0
            )
            if filter_type == 0:
                reconstructed = value
            elif filter_type == 1:
                reconstructed = value + left
            elif filter_type == 2:
                reconstructed = value + above
            elif filter_type == 3:
                reconstructed = value + ((left + above) // 2)
            elif filter_type == 4:
                reconstructed = value + _paeth(left, above, upper_left)
            else:
                raise PNGError(f"unsupported filter type {filter_type}")
            output[target_offset + column] = reconstructed & 0xFF
    return bytes(output)


def decode_png_bytes(data: bytes) -> PNGImage:
    chunks = _read_chunks(data)
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise PNGError("IHDR must be 13 bytes")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width <= 0 or height <= 0:
        raise PNGError("PNG dimensions must be positive")
    if bit_depth != 8:
        raise PNGError("only 8-bit PNG is supported")
    if color_type not in (2, 6):
        raise PNGError("only RGB and RGBA PNG are supported")
    if compression != 0 or filter_method != 0 or interlace != 0:
        raise PNGError("unsupported PNG compression/filter/interlace mode")
    idat = b"".join(payload for chunk_type, payload in chunks if chunk_type == "IDAT")
    if not idat:
        raise PNGError("missing IDAT")
    try:
        decompressed = zlib.decompress(idat)
    except zlib.error as exc:
        raise PNGError(f"invalid IDAT stream: {exc}") from exc
    channels = 3 if color_type == 2 else 4
    scanlines = _unfilter(decompressed, width, height, channels)
    array = np.frombuffer(scanlines, dtype=np.uint8).reshape(height, width, channels)
    if channels == 3:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return PNGImage(width=width, height=height, pixels=array.copy(), color_type=color_type)


def decode_png(path: str | Path) -> PNGImage:
    return decode_png_bytes(Path(path).read_bytes())
