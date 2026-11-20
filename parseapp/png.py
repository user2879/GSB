"""Small, dependency-free PNG codec for deterministic synthetic pages.

Only non-interlaced, 8-bit PNG files are used by this project. The decoder
supports grayscale, RGB, palette, and alpha variants so page validation can
reject unsupported or malformed containers without relying on Pillow.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Any

import numpy as np


class PNGError(ValueError):
    code = "E_PNG_INVALID"


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS_BY_COLOR_TYPE = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _parse_chunks(data: bytes) -> list[tuple[bytes, bytes, int]]:
    if not data.startswith(_PNG_SIGNATURE):
        raise PNGError("invalid PNG signature")
    chunks: list[tuple[bytes, bytes, int]] = []
    offset = len(_PNG_SIGNATURE)
    saw_iend = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise PNGError("truncated PNG chunk header")
        length, chunk_type = struct.unpack(">I4s", data[offset : offset + 8])
        offset += 8
        if offset + length + 4 > len(data):
            raise PNGError("truncated PNG chunk payload or CRC")
        payload = data[offset : offset + length]
        offset += length
        crc = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != crc:
            raise PNGError(f"CRC mismatch in {chunk_type!r}")
        chunks.append((chunk_type, payload, length))
        if chunk_type == b"IEND":
            saw_iend = True
            break
    if not saw_iend:
        raise PNGError("missing IEND")
    return chunks


def _paeth(left: int, up: int, upper_left: int) -> int:
    predictor = left + up - upper_left
    distance_left = abs(predictor - left)
    distance_up = abs(predictor - up)
    distance_upper_left = abs(predictor - upper_left)
    if distance_left <= distance_up and distance_left <= distance_upper_left:
        return left
    if distance_up <= distance_upper_left:
        return up
    return upper_left


def _unfilter_rows(
    filtered: bytes, width: int, height: int, bytes_per_pixel: int
) -> bytes:
    stride = width * bytes_per_pixel
    expected = height * (stride + 1)
    if len(filtered) != expected:
        raise PNGError(
            f"IDAT payload length mismatch: expected {expected}, got {len(filtered)}"
        )
    output = bytearray(height * stride)
    previous = bytearray(stride)
    source_offset = 0
    for row in range(height):
        filter_type = filtered[source_offset]
        source_offset += 1
        row_raw = filtered[source_offset : source_offset + stride]
        source_offset += stride
        current = bytearray(stride)
        if filter_type == 0:
            current[:] = row_raw
        elif filter_type == 1:
            for index, value in enumerate(row_raw):
                left = current[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                current[index] = (value + left) & 0xFF
        elif filter_type == 2:
            for index, value in enumerate(row_raw):
                current[index] = (value + previous[index]) & 0xFF
        elif filter_type == 3:
            for index, value in enumerate(row_raw):
                left = current[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous[index]
                current[index] = (value + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for index, value in enumerate(row_raw):
                left = current[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous[index]
                upper_left = (
                    previous[index - bytes_per_pixel]
                    if index >= bytes_per_pixel
                    else 0
                )
                current[index] = (value + _paeth(left, up, upper_left)) & 0xFF
        else:
            raise PNGError(f"unsupported PNG filter type {filter_type}")
        output[row * stride : (row + 1) * stride] = current
        previous = current
    return bytes(output)


def decode_png(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an 8-bit PNG to an HxW uint8 grayscale array."""

    chunks = _parse_chunks(data)
    if chunks[0][0] != b"IHDR":
        raise PNGError("IHDR must be the first chunk")
    if len(chunks[0][1]) != 13:
        raise PNGError("invalid IHDR length")
    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", chunks[0][1])
    )
    if width <= 0 or height <= 0:
        raise PNGError("PNG dimensions must be positive")
    if bit_depth != 8:
        raise PNGError(f"unsupported PNG bit depth {bit_depth}; expected 8")
    if color_type not in _CHANNELS_BY_COLOR_TYPE:
        raise PNGError(f"unsupported PNG color type {color_type}")
    if compression != 0:
        raise PNGError(f"unsupported PNG compression method {compression}")
    if filter_method != 0:
        raise PNGError(f"unsupported PNG filter method {filter_method}")
    if interlace != 0:
        raise PNGError("interlaced PNG is not supported")

    palette: np.ndarray | None = None
    transparency = b""
    idat_parts: list[bytes] = []
    saw_idat = False
    for chunk_type, payload, _ in chunks[1:]:
        if chunk_type == b"PLTE":
            if len(payload) % 3:
                raise PNGError("invalid PLTE length")
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).copy()
        elif chunk_type == b"tRNS":
            transparency = payload
        elif chunk_type == b"IDAT":
            idat_parts.append(payload)
            saw_idat = True
        elif chunk_type in {b"IHDR", b"IEND"}:
            continue
        elif chunk_type and chunk_type[0:1].isalpha():
            # Uppercase first byte means a critical chunk.
            if 65 <= chunk_type[0] <= 90:
                raise PNGError(f"unsupported critical chunk {chunk_type!r}")
        elif chunk_type and not chunk_type[0:1].isalpha():
            raise PNGError(f"invalid chunk type {chunk_type!r}")

    if not saw_idat:
        raise PNGError("missing IDAT")
    if color_type == 3 and (palette is None or len(palette) == 0):
        raise PNGError("indexed PNG is missing PLTE")

    try:
        filtered = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise PNGError(f"invalid IDAT stream: {exc}") from exc

    bytes_per_pixel = _CHANNELS_BY_COLOR_TYPE[color_type]
    raw = _unfilter_rows(filtered, width, height, bytes_per_pixel)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, bytes_per_pixel)

    if color_type == 0:
        gray = pixels[:, :, 0]
        mode = "L8"
    elif color_type == 2:
        gray = _rgb_to_gray(pixels)
        mode = "RGB8"
    elif color_type == 3:
        indices = pixels[:, :, 0]
        if int(indices.max(initial=0)) >= len(palette):
            raise PNGError("palette index outside PLTE")
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = palette[indices]
        rgba[:, :, 3] = 255
        if transparency:
            alpha = np.ones(len(palette), dtype=np.uint8) * 255
            values = np.frombuffer(transparency, dtype=np.uint8)
            alpha[: min(len(alpha), len(values))] = values[: len(alpha)]
            rgba[:, :, 3] = alpha[indices]
        gray = _rgba_to_gray(rgba)
        mode = "P8"
    elif color_type == 4:
        luminance = pixels[:, :, 0].astype(np.uint16)
        alpha = pixels[:, :, 1].astype(np.uint16)
        gray = ((luminance * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
        mode = "LA8"
    else:
        gray = _rgba_to_gray(pixels)
        mode = "RGBA8"

    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    info = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "pixel_mode": mode,
        "interlace": interlace,
    }
    return gray, info


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.uint32)
    return (
        (77 * values[:, :, 0] + 150 * values[:, :, 1] + 29 * values[:, :, 2] + 128)
        >> 8
    ).astype(np.uint8)


def _rgba_to_gray(rgba: np.ndarray) -> np.ndarray:
    values = rgba.astype(np.uint32)
    alpha = values[:, :, 3]
    red = (values[:, :, 0] * alpha + 255 * (255 - alpha) + 127) // 255
    green = (values[:, :, 1] * alpha + 255 * (255 - alpha) + 127) // 255
    blue = (values[:, :, 2] * alpha + 255 * (255 - alpha) + 127) // 255
    return ((77 * red + 150 * green + 29 * blue + 128) >> 8).astype(np.uint8)


def content_fingerprint(gray: np.ndarray) -> str:
    height, width = gray.shape
    prefix = f"{width}x{height}:L8\n".encode("ascii")
    return hashlib.sha256(prefix + np.ascontiguousarray(gray).tobytes()).hexdigest()


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as a deterministic non-interlaced PNG."""

    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PNGError("encode_png_rgb expects an HxWx3 uint8 array")
    height, width, _ = array.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    filtered = bytearray()
    for row in array:
        filtered.append(0)
        filtered.extend(row.tobytes())
    compressed = zlib.compress(bytes(filtered), level=9)
    return (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
