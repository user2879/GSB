"""Strict RGB PNG decoding/encoding without an external imaging dependency."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .errors import BusinessError
from .util import sha256_hex

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PNGInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int

    @property
    def pixel_mode(self) -> str:
        return {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}.get(self.color_type, f"TYPE_{self.color_type}")


@dataclass(frozen=True)
class RGBImage:
    pixels: np.ndarray
    digest: str

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


def _check_crc(chunk_type: bytes, body: bytes, crc: bytes) -> None:
    expected = zlib.crc32(chunk_type)
    expected = zlib.crc32(body, expected) & 0xFFFFFFFF
    actual = struct.unpack(">I", crc)[0]
    if actual != expected:
        raise BusinessError("E_PNG_CRC", "PNG chunk CRC mismatch", {"chunk": chunk_type.decode("ascii", "replace")})


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def parse_png(data: bytes) -> tuple[PNGInfo, np.ndarray]:
    """Parse an 8-bit, non-interlaced RGB PNG and return its pixels.

    The parser validates signatures, chunk lengths and CRCs, IHDR constraints,
    zlib data, row lengths and PNG filters.  Unsupported formats fail loudly.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise BusinessError("E_PNG_HEADER", "invalid PNG signature")
    offset = len(PNG_SIGNATURE)
    info: PNGInfo | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    first_chunk = True
    while offset < len(data):
        if offset + 12 > len(data):
            raise BusinessError("E_PNG_HEADER", "truncated PNG chunk header")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        body_start = offset + 8
        body_end = body_start + length
        crc_end = body_end + 4
        if crc_end > len(data):
            raise BusinessError("E_PNG_HEADER", "truncated PNG chunk")
        body = data[body_start:body_end]
        _check_crc(chunk_type, body, data[body_end:crc_end])
        offset = crc_end
        if first_chunk and chunk_type != b"IHDR":
            raise BusinessError("E_PNG_HEADER", "IHDR must be the first PNG chunk")
        first_chunk = False
        if chunk_type == b"IHDR":
            if info is not None or length != 13:
                raise BusinessError("E_PNG_HEADER", "invalid IHDR chunk")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            info = PNGInfo(width, height, bit_depth, color_type, compression, filter_method, interlace)
            if width <= 0 or height <= 0:
                raise BusinessError("E_PNG_HEADER", "PNG dimensions must be positive", {"width": width,
                                                                                        "height": height})
            if bit_depth != 8:
                raise BusinessError("E_PNG_PIXEL_MODE", "only 8-bit PNG input is supported",
                                    {"bit_depth": bit_depth})
            if color_type != 2:
                raise BusinessError("E_PNG_PIXEL_MODE", "PNG must use RGB color type 2",
                                    {"color_type": color_type})
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise BusinessError("E_PNG_HEADER", "unsupported PNG compression/filter/interlace",
                                    {"compression": compression, "filter": filter_method,
                                     "interlace": interlace})
        elif chunk_type == b"IDAT":
            idat_parts.append(body)
        elif chunk_type == b"IEND":
            saw_iend = True
            if body:
                raise BusinessError("E_PNG_HEADER", "IEND chunk must be empty")
            break
    if info is None:
        raise BusinessError("E_PNG_HEADER", "PNG has no IHDR")
    if not saw_iend:
        raise BusinessError("E_PNG_HEADER", "PNG has no IEND")
    if not idat_parts:
        raise BusinessError("E_PNG_HEADER", "PNG has no IDAT data")
    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise BusinessError("E_PNG_DATA", "invalid PNG zlib stream", {"error": str(exc)}) from exc
    width, height = info.width, info.height
    stride = width * 3
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise BusinessError("E_PNG_DATA", "PNG scanline data length mismatch",
                            {"expected": expected, "actual": len(raw)})
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.int32)
    position = 0
    for row in range(height):
        filter_type = raw[position]
        position += 1
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=position).astype(np.int32)
        position += stride
        reconstructed = np.empty(stride, dtype=np.int32)
        if filter_type == 0:
            reconstructed[:] = scan
        elif filter_type == 1:
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                reconstructed[index] = (scan[index] + left) & 0xFF
        elif filter_type == 2:
            reconstructed[:] = (scan + previous) & 0xFF
        elif filter_type == 3:
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                reconstructed[index] = (scan[index] + ((left + int(previous[index])) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                upper_left = previous[index - 3] if index >= 3 else 0
                reconstructed[index] = (scan[index] + _paeth(left, int(previous[index]), int(upper_left))) & 0xFF
        else:
            raise BusinessError("E_PNG_DATA", "unsupported PNG row filter", {"filter": int(filter_type),
                                                                               "row": row})
        pixels[row] = reconstructed.reshape(width, 3).astype(np.uint8)
        previous = reconstructed
    digest = sha256_hex(b"RGB8|" + struct.pack(">II", width, height) + pixels.tobytes(order="C"))
    return info, RGBImage(pixels=pixels, digest=digest)


def load_png(path) -> tuple[PNGInfo, RGBImage]:
    from pathlib import Path
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read PNG {path}", {"path": str(path), "error": str(exc)}) from exc
    return parse_png(data)


def _chunk(chunk_type: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(body, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + chunk_type + body + struct.pack(">I", crc)


def encode_rgb_png(pixels: np.ndarray) -> bytes:
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise BusinessError("E_PNG_WRITE", "expected an HxWx3 uint8 array", {"shape": list(pixels.shape)})
    height, width = pixels.shape[:2]
    raw_rows = []
    for row in pixels:
        raw_rows.append(b"\x00" + row.tobytes(order="C"))
    raw = b"".join(raw_rows)
    compressed = zlib.compress(raw, level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def concat_rows(blocks: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(block, dtype=np.uint8) for block in blocks]
    if not arrays:
        raise BusinessError("E_PNG_WRITE", "cannot assemble a PNG from zero block rows")
    width = arrays[0].shape[1]
    if any(array.ndim != 3 or array.shape[1] != width or array.shape[2] != 3 for array in arrays):
        raise BusinessError("E_PNG_WRITE", "block row shapes are incompatible")
    return np.concatenate(arrays, axis=0)
