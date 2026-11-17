"""Minimal deterministic PNG reader/writer (8-bit grayscale or RGB).

Only the subset needed by the pipeline is supported, but the reader still
validates the PNG container (signature, IHDR, IDAT CRC, IEND) so that a
corrupted byte stream is reliably detected and mapped to E_PNG_INVALID.
"""

import struct
import zlib

import numpy as np

from .errors import E_PNG_INVALID, PipelineError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type, data):
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png(path, image):
    """Write a uint8 HxW (grayscale) or HxWx3 (RGB) array as PNG."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise PipelineError(E_PNG_INVALID, "only uint8 images are supported")
    if arr.ndim == 2:
        color_type, channels = 0, 1
    elif arr.ndim == 3 and arr.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise PipelineError(E_PNG_INVALID, "unsupported image shape")
    height, width = arr.shape[:2]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = np.zeros((height, 1 + width * channels), dtype=np.uint8)
    raw[:, 1:] = arr.reshape(height, width * channels)
    blob = zlib.compress(raw.tobytes(), level=6)
    with open(path, "wb") as handle:
        handle.write(PNG_SIGNATURE)
        handle.write(_chunk(b"IHDR", ihdr))
        handle.write(_chunk(b"IDAT", blob))
        handle.write(_chunk(b"IEND", b""))


def _read_chunks(blob):
    if blob[:8] != PNG_SIGNATURE:
        raise PipelineError(E_PNG_INVALID, "bad PNG signature")
    offset = 8
    chunks = []
    saw_iend = False
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise PipelineError(E_PNG_INVALID, "truncated chunk header")
        (length,) = struct.unpack(">I", blob[offset : offset + 4])
        chunk_type = blob[offset + 4 : offset + 8]
        end = offset + 8 + length + 4
        if end > len(blob):
            raise PipelineError(E_PNG_INVALID, "truncated chunk data")
        data = blob[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", blob[offset + 8 + length : end])
        if zlib.crc32(chunk_type + data) & 0xFFFFFFFF != crc:
            raise PipelineError(
                E_PNG_INVALID, "CRC mismatch in chunk %s" % chunk_type.decode("latin1")
            )
        chunks.append((chunk_type, data))
        if chunk_type == b"IEND":
            saw_iend = True
            break
        offset = end
    if not saw_iend:
        raise PipelineError(E_PNG_INVALID, "missing IEND chunk")
    return chunks


def read_png(path):
    """Return (grayscale HxW uint8, rgb HxWx3 uint8) arrays."""
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        raise PipelineError(E_PNG_INVALID, "cannot read png: %s" % exc)
    chunks = _read_chunks(blob)
    ihdr = next((d for t, d in chunks if t == b"IHDR"), None)
    if ihdr is None or len(ihdr) != 13:
        raise PipelineError(E_PNG_INVALID, "missing IHDR chunk")
    width, height, bit_depth, color_type, compression, flt, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if bit_depth != 8 or color_type not in (0, 2) or compression != 0 or interlace != 0:
        raise PipelineError(E_PNG_INVALID, "unsupported IHDR parameters")
    channels = 1 if color_type == 0 else 3
    payload = b"".join(d for t, d in chunks if t == b"IDAT")
    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise PipelineError(E_PNG_INVALID, "corrupt IDAT stream: %s" % exc)
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise PipelineError(E_PNG_INVALID, "unexpected scanline length")
    rows = []
    prev = np.zeros(stride, dtype=np.uint8)
    for row_index in range(height):
        start = row_index * (stride + 1)
        filter_byte = raw[start]
        line = np.frombuffer(raw[start + 1 : start + 1 + stride], dtype=np.uint8).copy()
        if filter_byte == 0:
            pass
        elif filter_byte == 1:
            for idx in range(channels, stride):
                line[idx] = (line[idx] + line[idx - channels]) & 0xFF
        elif filter_byte == 2:
            line = (line + prev) & 0xFF
        elif filter_byte == 3:
            for idx in range(stride):
                left = line[idx - channels] if idx >= channels else 0
                line[idx] = (line[idx] + ((left + prev[idx]) >> 1)) & 0xFF
        elif filter_byte == 4:
            for idx in range(stride):
                left = line[idx - channels] if idx >= channels else 0
                up_left = prev[idx - channels] if idx >= channels else 0
                p = int(left) + int(prev[idx]) - int(up_left)
                pa, pb, pc = abs(p - left), abs(p - prev[idx]), abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else (prev[idx] if pb <= pc else up_left)
                line[idx] = (line[idx] + predictor) & 0xFF
        else:
            raise PipelineError(E_PNG_INVALID, "unknown scanline filter")
        rows.append(line)
        prev = line
    image = np.stack(rows).reshape(height, width, channels)
    if channels == 1:
        gray = image[:, :, 0]
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        rgb = image
        gray = (
            (0.299 * image[:, :, 0])
            + (0.587 * image[:, :, 1])
            + (0.114 * image[:, :, 2])
        ).astype(np.uint8)
    return gray, rgb
