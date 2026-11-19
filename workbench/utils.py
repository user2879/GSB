"""Deterministic hashing, fixed-point arithmetic and atomic file helpers.

The same hash definitions drive cache keys on Windows and Linux, and the
fixed-point rules make algorithm outputs bit-identical across machines.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

ALGORITHM_VERSION = "wb-algo-1.0.0"
SCALE_BITS = 8
Q = 1 << SCALE_BITS


def canonical_json(obj) -> bytes:
    """Serialize to canonical JSON bytes (sorted keys, stable separators)."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def digest_obj(obj) -> str:
    return digest_bytes(canonical_json(obj))


def digest_array(arr: np.ndarray) -> str:
    """Content fingerprint of a numpy array, independent of file metadata."""
    return digest_bytes(arr.dtype.str.encode() + b"|" + np.ascontiguousarray(arr).tobytes())


def fp_mul(value: int, weight_q: int) -> int:
    """Signed fixed-point multiplication with symmetric rounding."""
    return (value * weight_q + (Q >> 1)) // Q


def fp_div(num: int, den: int) -> int:
    """Fixed-point division num/den with symmetric rounding (den > 0)."""
    if den <= 0:
        return 0
    return (num + (den >> 1)) // den


def weight_to_q(weight: float) -> int:
    return int(round(float(weight) * Q))


def is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a temp file in the same directory then atomically replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(Path(path), canonical_json(obj))


def load_json(path: Path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def npy_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(arr), allow_pickle=False)
    return buffer.getvalue()


def save_npy(path: Path, arr: np.ndarray) -> None:
    # allow_pickle=False keeps every artifact plain numeric data.
    atomic_write_bytes(Path(path), npy_bytes(arr))


def load_npy(path: Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img), mode="RGB").save(
        buffer, format="PNG", optimize=False, compress_level=6
    )
    return buffer.getvalue()


def png_decode(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def stable_sort_key(sample_name: str, stage: str) -> str:
    """Deterministic task ordering key derived from sample and stage names."""
    return f"{sample_name}::{stage}"


def map_summary(mapping) -> str:
    """Short auditable fingerprint of a recolour / style mapping table."""
    return digest_obj(mapping)[:12]
