"""Small dependency-free RGB8 PNG reader/writer used by the demo samples."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import E_PARAM_INVALID, E_PNG_INVALID, diagnostic


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PngImage:
    width: int
    height: int
    rgb: np.ndarray
    raw_sha256: str
    bit_depth: int = 8
    color_type: int = 2


def _chunk_payload(data: bytes, offset: int) -> tuple[bytes, bytes, int]:
    if offset + 12 > len(data):
        raise ValueError("truncated PNG chunk")
    length = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_type = data[offset + 4 : offset + 8]
    end = offset + 12 + length
    if end > len(data):
        raise ValueError("truncated PNG payload")
    payload = data[offset + 8 : offset + 8 + length]
    crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
    if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != crc:
        raise ValueError(f"bad CRC for {chunk_type!r}")
    return chunk_type, payload, end


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


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> bytearray:
    stride = width * bpp
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError("decompressed PNG size does not match IHDR")
    out = bytearray(height * stride)
    source = 0
    for y in range(height):
        filter_type = raw[source]
        source += 1
        row_start = y * stride
        prev_start = row_start - stride
        for x in range(stride):
            value = raw[source]
            source += 1
            left = out[row_start + x - bpp] if x >= bpp else 0
            up = out[prev_start + x] if y else 0
            upper_left = out[prev_start + x - bpp] if y and x >= bpp else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = _paeth(left, up, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}")
            out[row_start + x] = (value + predictor) & 0xFF
    return out


def decode_png(path: str | Path) -> PngImage:
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("missing PNG signature")
    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    while offset < len(data):
        chunk_type, payload, offset = _chunk_payload(data, offset)
        if chunk_type == b"IHDR":
            if ihdr is not None or len(payload) != 13:
                raise ValueError("invalid IHDR")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            saw_iend = True
            break
        elif chunk_type and chunk_type[0] & 0x20 == 0:
            raise ValueError(f"unexpected critical PNG chunk {chunk_type!r}")
    if ihdr is None or not idat_parts or not saw_iend:
        raise ValueError("incomplete PNG")
    width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
    if width <= 0 or height <= 0:
        raise ValueError("invalid PNG dimensions")
    if bit_depth != 8 or color_type != 2:
        raise ValueError("only RGB8 PNG (bit depth 8, colour type 2) is supported")
    if compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError("unsupported PNG compression/filter/interlace")
    raw = zlib.decompress(b"".join(idat_parts))
    pixels = _unfilter(raw, width, height, 3)
    rgb = np.frombuffer(bytes(pixels), dtype=np.uint8).reshape(height, width, 3).copy()
    import hashlib

    return PngImage(width, height, rgb, hashlib.sha256(data).hexdigest())


def probe_png(path: str | Path) -> dict:
    image = decode_png(path)
    return {
        "width": image.width,
        "height": image.height,
        "bit_depth": image.bit_depth,
        "color_type": image.color_type,
        "sha256": image.raw_sha256,
    }


def read_png_checked(
    path: str | Path,
    *,
    expected_width: int,
    expected_height: int,
    view_id: str | None = None,
) -> tuple[PngImage | None, dict | None]:
    try:
        image = decode_png(path)
    except Exception as exc:
        return None, diagnostic(
            E_PNG_INVALID,
            f"PNG validation failed: {exc}",
            stage="load",
            view=view_id,
            detail={"path": str(path)},
        )
    if image.width != expected_width or image.height != expected_height:
        return None, diagnostic(
            E_PARAM_INVALID,
            "image resolution does not match the fixed project resolution",
            stage="load",
            view=view_id,
            detail={
                "path": str(path),
                "expected": [expected_width, expected_height],
                "actual": [image.width, image.height],
            },
        )
    return image, None


class PngWriter:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @staticmethod
    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    def encode(self, rgb: np.ndarray) -> bytes:
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB image must be uint8 with shape (height, width, 3)")
        rows = bytearray()
        for y in range(self.height):
            rows.append(0)
            rows.extend(rgb[y].tobytes())
        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            PNG_SIGNATURE
            + self._chunk(b"IHDR", ihdr)
            + self._chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + self._chunk(b"IEND", b"")
        )

    def write(self, path: str | Path, rgb: np.ndarray) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.encode(rgb))
