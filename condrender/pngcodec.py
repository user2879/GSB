from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .errors import Codes, ValidationError, Failure


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png_rgb(rgb: np.ndarray | list | tuple, compress_level: int = 6) -> bytes:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB PNG input must have shape (height, width, 3)")
    height, width, _ = arr.shape
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(arr[y, :, :].tobytes(order="C"))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
        + _chunk(b"IEND", b"")
    )


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_rgb(data: bytes, expected_width: int | None = None,
                   expected_height: int | None = None,
                   source_path: str | None = None) -> np.ndarray:
    if not data.startswith(PNG_SIGNATURE):
        raise ValidationError(Failure(
            Codes.PNG_SIGNATURE, "PNG signature does not match the fixed container",
            path=source_path,
        ))
    pos = len(PNG_SIGNATURE)
    info: PngInfo | None = None
    idat = bytearray()
    saw_iend = False
    try:
        while pos < len(data):
            if pos + 8 > len(data):
                raise ValidationError(Failure(
                    Codes.PNG_TRUNCATED, "PNG chunk header is truncated", path=source_path
                ))
            length, kind = struct.unpack(">I4s", data[pos:pos + 8])
            body_start = pos + 8
            body_end = body_start + length
            crc_end = body_end + 4
            if crc_end > len(data):
                raise ValidationError(Failure(
                    Codes.PNG_TRUNCATED, f"PNG {kind!r} chunk body or CRC is truncated",
                    path=source_path,
                ))
            body = data[body_start:body_end]
            expected_crc = struct.unpack(">I", data[body_end:crc_end])[0]
            actual_crc = zlib.crc32(kind + body) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise ValidationError(Failure(
                    Codes.PNG_CRC, f"PNG chunk CRC mismatch for {kind!r}", path=source_path
                ))
            if kind == b"IHDR":
                if length != 13:
                    raise ValidationError(Failure(
                        Codes.PNG_HEADER, "IHDR length is not 13", path=source_path
                    ))
                width, height, bit_depth, color_type, compression, filt, interlace = struct.unpack(
                    ">IIBBBBB", body
                )
                if compression != 0 or filt != 0 or interlace != 0:
                    raise ValidationError(Failure(
                        Codes.PNG_HEADER, "unsupported PNG compression/filter/interlace mode",
                        path=source_path,
                    ))
                if bit_depth != 8 or color_type != 2:
                    raise ValidationError(Failure(
                        Codes.PNG_HEADER, "only 8-bit RGB PNG is supported", path=source_path,
                        details={"bit_depth": bit_depth, "color_type": color_type},
                    ))
                info = PngInfo(width, height, bit_depth, color_type)
            elif kind == b"IDAT":
                idat.extend(body)
            elif kind == b"IEND":
                saw_iend = True
                pos = crc_end
                break
            pos = crc_end
    except struct.error as exc:
        raise ValidationError(Failure(
            Codes.PNG_TRUNCATED, "PNG struct field is truncated", path=source_path
        )) from exc

    if info is None:
        raise ValidationError(Failure(Codes.PNG_HEADER, "missing IHDR", path=source_path))
    if not saw_iend:
        raise ValidationError(Failure(Codes.PNG_TRUNCATED, "missing IEND; PNG byte stream is truncated",
                                      path=source_path))
    if expected_width is not None and expected_height is not None:
        if info.width != expected_width or info.height != expected_height:
            raise ValidationError(Failure(
                Codes.PNG_RESOLUTION,
                "PNG resolution does not match the fixed condition resolution",
                path=source_path,
                details={
                    "actual": [info.width, info.height],
                    "expected": [expected_width, expected_height],
                },
            ))
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise ValidationError(Failure(
            Codes.PNG_TRUNCATED, "IDAT zlib stream cannot be decoded", path=source_path,
            details={"zlib_error": str(exc)},
        )) from exc

    channels = 3
    stride = info.width * channels
    expected_len = info.height * (stride + 1)
    if len(raw) != expected_len:
        raise ValidationError(Failure(
            Codes.PNG_TRUNCATED, "PNG scanline payload length is incorrect", path=source_path,
            details={"actual": len(raw), "expected": expected_len},
        ))

    rows = np.zeros((info.height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    for y in range(info.height):
        filter_type = raw[y * (stride + 1)]
        line = np.frombuffer(
            raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)], dtype=np.uint8
        ).copy()
        if filter_type == 0:
            recon = line
        elif filter_type == 1:
            recon = line.copy()
            for x in range(channels, stride):
                recon[x] = (int(recon[x]) + int(recon[x - channels])) & 0xFF
        elif filter_type == 2:
            recon = (line.astype(np.uint16) + previous.astype(np.uint16)) & 0xFF
            recon = recon.astype(np.uint8)
        elif filter_type == 3:
            recon = line.copy()
            for x in range(stride):
                left = int(recon[x - channels]) if x >= channels else 0
                up = int(previous[x])
                recon[x] = (int(recon[x]) + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            recon = line.copy()
            for x in range(stride):
                a = int(recon[x - channels]) if x >= channels else 0
                b = int(previous[x])
                c = int(previous[x - channels]) if x >= channels else 0
                recon[x] = (int(recon[x]) + _paeth(a, b, c)) & 0xFF
        else:
            raise ValidationError(Failure(
                Codes.PNG_HEADER, f"unsupported PNG filter type {filter_type}", path=source_path
            ))
        rows[y, :] = recon
        previous = recon
    return rows.reshape(info.height, info.width, 3)


def read_png_file(path: str, expected_width: int | None = None,
                  expected_height: int | None = None) -> np.ndarray:
    with open(path, "rb") as handle:
        data = handle.read()
    return decode_png_rgb(data, expected_width, expected_height, source_path=path)
