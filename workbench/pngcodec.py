"""Minimal deterministic PNG codec (8-bit RGB, color type 2) with no third-party deps."""

import binascii
import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(Exception):
    pass


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as a PNG with filter type 0 scanlines."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PngError("expected HxWx3 uint8 array")
    height, width, _ = rgb.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    idat = zlib.compress(raw, 9)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def read_header(data: bytes):
    """Validate the PNG container header and return (width, height)."""
    if len(data) < 33 or data[:8] != PNG_SIGNATURE:
        raise PngError("bad PNG signature")
    length, tag = struct.unpack(">I4s", data[8:16])
    if tag != b"IHDR" or length < 13:
        raise PngError("missing IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    if bit_depth != 8 or color_type != 2:
        raise PngError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type}")
    return width, height


def _unfilter(raw: bytes, width: int, height: int) -> bytes:
    stride = width * 3
    out = bytearray(height * stride)
    pos = 0
    prev = bytearray(stride)
    for row in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for i in range(3, stride):
                line[i] = (line[i] + line[i - 3]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - 3] if i >= 3 else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        else:
            raise PngError(f"unknown filter type {ftype}")
        out[row * stride:(row + 1) * stride] = line
        prev = line
    return bytes(out)


def decode_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit RGB PNG into an HxWx3 uint8 array."""
    width, height = read_header(data)
    pos = 8
    idat = bytearray()
    seen_iend = False
    while pos + 8 <= len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        payload = data[pos + 8:pos + 8 + length]
        crc_stored = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        if binascii.crc32(tag + payload) & 0xFFFFFFFF != crc_stored:
            raise PngError(f"CRC mismatch in chunk {tag!r}")
        if tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            seen_iend = True
            break
        pos += 12 + length
    if not seen_iend:
        raise PngError("missing IEND")
    raw = zlib.decompress(bytes(idat))
    expected = height * (1 + width * 3)
    if len(raw) != expected:
        raise PngError(f"unexpected raw size {len(raw)} != {expected}")
    pixels = _unfilter(raw, width, height)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
