"""Deterministic binary point-cloud / normal file format.

Layout (all multi-byte integers little-endian, all floats IEEE-754):

    magic            4 bytes  b"PCRC"
    format_version   uint16   currently 1
    dtype_code       uint16   1 = float32, 2 = float64
    component_count  uint32   columns per record (3 points / 4 normals)
    record_count     uint64
    payload          record_count * component_count * dtype size
    checksum         32 bytes BLAKE2b-256 over everything preceding it

The format is validated on read; structural problems raise
``PointCloudFormatError`` (E_POINT_CLOUD_FORMAT).
"""
from __future__ import annotations

import os
import struct

import numpy as np

from .errors import PointCloudFormatError
from .hashing import atomic_write_bytes, blake2b_hex

MAGIC = b"PCRC"
VERSION = 1
_DTYPE = {1: np.dtype("<f4"), 2: np.dtype("<f8")}
_DTYPE_CODE = {np.dtype(np.float32): 1, np.dtype(np.float64): 2}
HEADER = struct.Struct("<4sHHIQ")
CHECKSUM_SIZE = 32


def write_array(path: str, array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise PointCloudFormatError("array must be 2-D with 3 or 4 columns")
    base = np.dtype(array.dtype).base
    if base not in _DTYPE_CODE:
        raise PointCloudFormatError(f"unsupported dtype {array.dtype}")
    code = _DTYPE_CODE[base]
    payload = array.astype(_DTYPE[code], copy=False).tobytes(order="C")
    header = HEADER.pack(MAGIC, VERSION, code, array.shape[1], array.shape[0])
    checksum = blake2b_hex(header + payload)
    atomic_write_bytes(path, header + payload + bytes.fromhex(checksum))
    return {
        "path": path,
        "records": int(array.shape[0]),
        "components": int(array.shape[1]),
        "dtype": str(_DTYPE[code]),
        "bytes": len(header) + len(payload) + CHECKSUM_SIZE,
        "blake2b256": checksum,
    }


def read_array(path: str, expected_components: int | None = None) -> np.ndarray:
    if not os.path.exists(path):
        raise PointCloudFormatError(f"file not found: {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < HEADER.size + CHECKSUM_SIZE:
        raise PointCloudFormatError("file truncated below header size")
    magic, version, code, components, count = HEADER.unpack(raw[: HEADER.size])
    if magic != MAGIC:
        raise PointCloudFormatError("bad magic bytes")
    if version != VERSION:
        raise PointCloudFormatError(f"unsupported format version {version}")
    if code not in _DTYPE:
        raise PointCloudFormatError(f"unknown dtype code {code}")
    if expected_components is not None and components != expected_components:
        raise PointCloudFormatError(
            f"expected {expected_components} components, got {components}"
        )
    dtype = _DTYPE[code]
    payload_end = HEADER.size + count * components * dtype.itemsize
    if payload_end + CHECKSUM_SIZE != len(raw):
        raise PointCloudFormatError("payload length does not match header")
    payload = raw[HEADER.size:payload_end]
    stored = raw[payload_end:]
    actual = bytes.fromhex(blake2b_hex(raw[:payload_end]))
    if not _constant_time_equal(stored, actual):
        raise PointCloudFormatError("checksum mismatch")
    return np.frombuffer(payload, dtype=dtype).reshape(count, components).copy()


def _constant_time_equal(a: bytes, b: bytes) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0
