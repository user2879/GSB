"""Small, dependency-free PNG codec for deterministic sample data and tests.

Only the PNG features needed by this project are implemented: non-interlaced
8-bit grayscale, RGB, grayscale+alpha and RGBA. All PNG filter types are
supported when reading. Writing always uses filter type 0 and RGBA.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import E_TILE_DECODE, SegmentationError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS_BY_COLOR_TYPE = {0: 1, 2: 3, 4: 2, 6: 4}
_ALLOWED_BIT_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 4: (8, 16), 6: (8, 16)}


@dataclass(frozen=True, slots=True)
class DecodedPNG:
    array: np.ndarray
    width: int
    height: int
    channels: int
    bit_depth: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_png(array: np.ndarray) -> bytes:
    """Encode an HxWxC uint8 array as a deterministic RGBA/RGB/grayscale PNG."""

    value = np.asarray(array)
    if value.ndim == 2:
        value = value[:, :, None]
    if value.ndim != 3:
        raise ValueError("PNG array must be HxW or HxWxC")
    if value.dtype != np.uint8:
        raise ValueError("PNG encoder accepts uint8 only")
    channels = value.shape[2]
    if channels not in (1, 2, 3, 4):
        raise ValueError("PNG supports 1, 2, 3 or 4 channels")
    if channels == 1:
        color_type = 0
    elif channels == 2:
        color_type = 4
    elif channels == 3:
        color_type = 2
    else:
        color_type = 6

    height, width, _ = value.shape
    scanlines = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), value.reshape(height, width * channels)],
        axis=1,
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    compressed = zlib.compress(scanlines.tobytes(order="C"), level=9)
    return b"".join(
        [
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", compressed),
            _chunk(b"IEND", b""),
        ]
    )


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(array))


def _unfilter_scanline(filtered: bytes, previous: bytearray, bpp: int) -> bytearray:
    filter_type = filtered[0]
    data = bytearray(filtered[1:])
    length = len(data)
    if filter_type == 0:
        return data
    if filter_type == 1:
        for index in range(bpp, length):
            data[index] = (data[index] + data[index - bpp]) & 0xFF
        return data
    if filter_type == 2:
        for index in range(length):
            data[index] = (data[index] + previous[index]) & 0xFF
        return data
    if filter_type == 3:
        for index in range(length):
            left = data[index - bpp] if index >= bpp else 0
            data[index] = (data[index] + ((left + previous[index]) >> 1)) & 0xFF
        return data
    if filter_type == 4:
        for index in range(length):
            left = data[index - bpp] if index >= bpp else 0
            up = previous[index]
            up_left = previous[index - bpp] if index >= bpp else 0
            predictor = left + up - up_left
            pa = abs(predictor - left)
            pb = abs(predictor - up)
            pc = abs(predictor - up_left)
            if pa <= pb and pa <= pc:
                nearest = left
            elif pb <= pc:
                nearest = up
            else:
                nearest = up_left
            data[index] = (data[index] + nearest) & 0xFF
        return data
    raise ValueError(f"unsupported PNG filter type {filter_type}")


def decode_png(data: bytes) -> DecodedPNG:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("missing PNG signature")
    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    interlace = None
    compressed_parts: list[bytes] = []
    saw_iend = False

    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise ValueError("truncated PNG payload")
        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        actual_crc = zlib.crc32(kind)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError(f"PNG chunk CRC mismatch in {kind!r}")
        offset = crc_end
        if kind == b"IHDR":
            if length != 13:
                raise ValueError("invalid IHDR length")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if width <= 0 or height <= 0:
                raise ValueError("invalid PNG dimensions")
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("unsupported PNG compression/filter/interlace")
        elif kind == b"IDAT":
            compressed_parts.append(payload)
        elif kind == b"IEND":
            saw_iend = True
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("missing PNG IHDR")
    if not saw_iend:
        raise ValueError("missing PNG IEND")
    if color_type not in _CHANNELS_BY_COLOR_TYPE:
        raise ValueError(f"unsupported PNG color type {color_type}")
    if bit_depth not in _ALLOWED_BIT_DEPTHS[color_type]:
        raise ValueError(f"unsupported bit depth {bit_depth}")
    if interlace != 0:
        raise ValueError("interlaced PNG is unsupported")
    if not compressed_parts:
        raise ValueError("missing PNG IDAT")

    channels = _CHANNELS_BY_COLOR_TYPE[color_type]
    try:
        raw = zlib.decompress(b"".join(compressed_parts))
    except zlib.error as exc:
        raise ValueError(f"invalid PNG IDAT: {exc}") from exc

    if bit_depth < 8:
        bits_per_pixel = channels * bit_depth
        row_bytes = (width * bits_per_pixel + 7) // 8
        expected = height * (row_bytes + 1)
        if len(raw) != expected:
            raise ValueError("unexpected PNG payload length")
        rows = np.zeros((height, width, channels), dtype=np.uint8)
        for y in range(height):
            start = y * (row_bytes + 1)
            filter_type = raw[start]
            if filter_type != 0:
                raise ValueError("filtering is unsupported for sub-8-bit PNG")
            packed = np.frombuffer(raw[start + 1 : start + 1 + row_bytes], dtype=np.uint8)
            unpacked = np.unpackbits(packed, bitorder="big")
            if bit_depth == 1:
                pixels = unpacked[: width * channels]
            else:
                groups = unpacked.reshape(-1, bit_depth)
                pixels = np.zeros(groups.shape[0], dtype=np.uint8)
                for bit in range(bit_depth):
                    pixels = (pixels << 1) | groups[:, bit]
                pixels = pixels[: width * channels]
            rows[y] = pixels.reshape(width, channels)
        return DecodedPNG(rows, width, height, channels, bit_depth)

    bytes_per_pixel = channels * (bit_depth // 8)
    row_bytes = width * bytes_per_pixel
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError("unexpected PNG payload length")
    decoded = np.zeros((height, row_bytes), dtype=np.uint8)
    previous = bytearray(row_bytes)
    cursor = 0
    for y in range(height):
        filtered = raw[cursor : cursor + row_bytes + 1]
        cursor += row_bytes + 1
        row = _unfilter_scanline(filtered, previous, bytes_per_pixel)
        decoded[y] = np.frombuffer(bytes(row), dtype=np.uint8)
        previous = row
    if bit_depth == 16:
        high = decoded.reshape(height, width, channels, 2)[:, :, :, 0]
        array = high.copy()
    else:
        array = decoded.reshape(height, width, channels).copy()
    return DecodedPNG(array, width, height, channels, bit_depth)


def read_png(path: Path, *, truncate_bytes: int | None = None) -> DecodedPNG:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SegmentationError(E_TILE_DECODE, f"cannot read tile: {exc}", stage="load") from exc
    if truncate_bytes is not None:
        if truncate_bytes < 0:
            raise SegmentationError(E_TILE_DECODE, "negative truncate byte count", stage="load")
        data = data[:truncate_bytes]
    try:
        return decode_png(data)
    except (ValueError, IndexError, struct.error) as exc:
        raise SegmentationError(E_TILE_DECODE, f"invalid PNG: {exc}", stage="load") from exc
