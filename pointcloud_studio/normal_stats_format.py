"""Binary sidecar for per-point normal-fit audit information."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .errors import ErrorCode, StudioError

MAGIC = b"PCSTNST1"
HEADER = struct.Struct("<8sII")


def encode_neighbor_stats(
    neighbor_counts: np.ndarray,
    residuals: np.ndarray,
    error_bounds: np.ndarray,
) -> bytes:
    count = int(np.asarray(neighbor_counts).shape[0])
    counts = np.ascontiguousarray(neighbor_counts, dtype="<i4")
    res = np.ascontiguousarray(residuals, dtype="<f4")
    bounds = np.ascontiguousarray(error_bounds, dtype="<f4")
    if counts.shape != (count,) or res.shape != (count,) or bounds.shape != (count,):
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal-stat arrays must have matching one-dimensional shapes",
            context={"count": count, "shapes": [list(counts.shape), list(res.shape), list(bounds.shape)]},
        )
    payload = np.column_stack((counts, res.view(np.int32), bounds.view(np.int32)))
    return HEADER.pack(MAGIC, 1, count) + payload.astype("<i4", copy=False).tobytes(order="C")


def load_neighbor_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = Path(path)
    data = source.read_bytes()
    if len(data) < HEADER.size:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "truncated normal-stat header",
            context={"path": str(source)},
        )
    magic, version, count = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != 1:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "invalid normal-stat header",
            context={"path": str(source), "magic": magic.hex(), "version": version},
        )
    expected = HEADER.size + count * 3 * 4
    if len(data) != expected:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "normal-stat payload size mismatch",
            context={"path": str(source), "size": len(data), "expected": expected},
        )
    raw = np.frombuffer(data, dtype="<i4", count=count * 3, offset=HEADER.size).reshape(count, 3)
    counts = raw[:, 0].astype(np.int32, copy=True)
    residuals = raw[:, 1].view(np.float32).astype(np.float32, copy=True)
    bounds = raw[:, 2].view(np.float32).astype(np.float32, copy=True)
    return counts, residuals, bounds

