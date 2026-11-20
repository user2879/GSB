"""Versioned, deterministic binary point and normal files."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .errors import ErrorCode, StudioError

POINT_MAGIC = b"PCSTPTS1"
NORMAL_MAGIC = b"PCSTNRM1"
FILE_VERSION = 1
HEADER = struct.Struct("<8sIIQ")
POINT_DTYPE = np.dtype("<f4")
NORMAL_DTYPE = np.dtype("<f4")
NORMAL_COMPONENTS = 4  # nx, ny, nz, confidence


def _header(magic: bytes, flags: int, count: int) -> bytes:
    return HEADER.pack(magic, FILE_VERSION, flags, count)


def _read_header(path: Path, magic: bytes) -> int:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "point-cloud file cannot be read",
            context={"path": str(path), "reason": str(exc)},
        ) from exc
    if len(data) < HEADER.size:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "truncated point-cloud header",
            context={"path": str(path), "size": len(data), "minimum": HEADER.size},
        )
    found_magic, version, flags, count = HEADER.unpack_from(data, 0)
    if found_magic != magic:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "invalid point-cloud magic",
            context={"path": str(path), "found": found_magic.hex(), "expected": magic.hex()},
        )
    if version != FILE_VERSION:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "unsupported point-cloud version",
            context={"path": str(path), "found": version, "expected": FILE_VERSION},
        )
    return count


def encode_points(points: np.ndarray) -> bytes:
    array = np.ascontiguousarray(points, dtype=POINT_DTYPE)
    if array.ndim != 2 or array.shape[1] != 3:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "points must have shape (N, 3)",
            context={"shape": list(array.shape)},
        )
    if not np.isfinite(array).all():
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "points contain NaN or infinity",
            context={"shape": list(array.shape)},
        )
    return _header(POINT_MAGIC, 0, array.shape[0]) + array.tobytes(order="C")


def encode_normals(normals: np.ndarray, confidence: np.ndarray) -> bytes:
    normal_array = np.ascontiguousarray(normals, dtype=NORMAL_DTYPE)
    confidence_array = np.ascontiguousarray(confidence, dtype=NORMAL_DTYPE)
    if normal_array.ndim != 2 or normal_array.shape[1] != 3:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normals must have shape (N, 3)",
            context={"shape": list(normal_array.shape)},
        )
    if confidence_array.shape != (normal_array.shape[0],):
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "confidence must have shape (N,)",
            context={"shape": list(confidence_array.shape)},
        )
    record = np.empty((normal_array.shape[0], NORMAL_COMPONENTS), dtype=NORMAL_DTYPE)
    record[:, :3] = normal_array
    record[:, 3] = confidence_array
    return _header(NORMAL_MAGIC, 1, record.shape[0]) + record.tobytes(order="C")


def load_points(path: str | Path) -> np.ndarray:
    source = Path(path)
    count = _read_header(source, POINT_MAGIC)
    expected = HEADER.size + count * 3 * POINT_DTYPE.itemsize
    data = source.read_bytes()
    if len(data) != expected:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "point-cloud payload size does not match header",
            context={"path": str(source), "size": len(data), "expected": expected},
        )
    points = np.frombuffer(data, dtype=POINT_DTYPE, count=count * 3, offset=HEADER.size)
    points = points.reshape(count, 3).astype(np.float32, copy=True)
    if not np.isfinite(points).all():
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "points contain NaN or infinity",
            context={"path": str(source), "count": count},
        )
    return points


def load_normals(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    count = _read_header(source, NORMAL_MAGIC)
    expected = HEADER.size + count * NORMAL_COMPONENTS * NORMAL_DTYPE.itemsize
    data = source.read_bytes()
    if len(data) != expected:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal payload size does not match header",
            context={"path": str(source), "size": len(data), "expected": expected},
        )
    records = np.frombuffer(
        data,
        dtype=NORMAL_DTYPE,
        count=count * NORMAL_COMPONENTS,
        offset=HEADER.size,
    ).reshape(count, NORMAL_COMPONENTS)
    normals = records[:, :3].astype(np.float32, copy=True)
    confidence = records[:, 3].astype(np.float32, copy=True)
    if not np.isfinite(normals).all() or not np.isfinite(confidence).all():
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normals contain NaN or infinity",
            context={"path": str(source), "count": count},
        )
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    bad = np.flatnonzero(np.abs(lengths - 1.0) > 2.0e-5)
    if bad.size:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal vectors must be unit length",
            context={"path": str(source), "first_bad_index": int(bad[0]), "length": float(lengths[bad[0]])},
        )
    if ((confidence < 0.0) | (confidence > 1.0)).any():
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal confidence must be in [0, 1]",
            context={"path": str(source), "minimum": float(confidence.min()), "maximum": float(confidence.max())},
        )
    return normals, confidence


def write_points(path: str | Path, points: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode_points(points))


def write_normals(path: str | Path, normals: np.ndarray, confidence: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode_normals(normals, confidence))

