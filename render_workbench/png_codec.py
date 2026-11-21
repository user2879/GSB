"""Minimal deterministic RGB8 PNG encoder/decoder.

Only the subset needed by the workbench is supported: 8-bit, color type 2
(RGB), filter type 0 (None) on every scanline. Encoding is fully
deterministic: fixed zlib compression level, no ancillary chunks and no
timestamps, so identical pixels always produce identical bytes.
"""

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COMPRESSION_LEVEL = 6


def _chunk(tag, payload):
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_png(pixels):
    """Encode an HxWx3 uint8 array to deterministic PNG bytes."""
    if not isinstance(pixels, np.ndarray):
        pixels = np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("encode_png expects an HxWx3 uint8 array")
    height, width = pixels.shape[0], pixels.shape[1]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = np.empty((height, 1 + width * 3), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = pixels.reshape(height, width * 3)
    compressed = zlib.compress(raw.tobytes(order="C"), _COMPRESSION_LEVEL)
    out = [
        PNG_SIGNATURE,
        _chunk(b"IHDR", ihdr),
        _chunk(b"IDAT", compressed),
        _chunk(b"IEND", b""),
    ]
    return b"".join(out)


def decode_png(data):
    """Decode PNG bytes produced by :func:`encode_png` to HxWx3 uint8."""
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if (zlib.crc32(tag + payload) & 0xFFFFFFFF) != crc:
            raise ValueError("PNG CRC mismatch in %s" % tag)
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = \
                struct.unpack(">IIBBBBB", payload)
            if bit_depth != 8 or color_type != 2:
                raise ValueError("only 8-bit RGB PNGs are supported")
            if comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("unsupported PNG compression/filter/interlace")
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
    if width is None:
        raise ValueError("PNG missing IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError("PNG raw data length mismatch")
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    for row in range(height):
        start = row * (stride + 1)
        if raw[start] != 0:
            raise ValueError("only filter type 0 scanlines are supported")
        line = np.frombuffer(raw, dtype=np.uint8, count=stride,
                             offset=start + 1)
        pixels[row] = line.reshape(width, 3)
    return pixels
