"""Canonical serialization, hashing, fixed-point math, and PNG helpers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PIPELINE_VERSION = "fixedvideo-pipeline-v1"
ENCODER_VERSION = "hash-bow-v1"
DENOISE_VERSION = "denoise-fixed-v1"
DECODER_VERSION = "rgb-linear-v1"
ROUNDING_VERSION = "half-away-from-zero-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_object(value: Any, *, domain: str = PIPELINE_VERSION) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\x00" + canonical_bytes(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonical_bytes(list(contiguous.shape)))
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def stable_seed(*parts: Any, bits: int = 63) -> int:
    """Derive a process-independent integer seed from content-addressed parts."""
    raw = hash_object(list(parts), domain="fixedvideo-seed").encode("ascii")
    return int.from_bytes(raw[:16], "big") & ((1 << bits) - 1)


def round_half_away_scalar(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def round_half_away_array(values: np.ndarray, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    values = np.asarray(values, dtype=np.int64)
    positive = (values + step // 2) // step
    negative = -((-values + step // 2) // step)
    return np.where(values >= 0, positive, negative) * step


def float_to_fixed(values: np.ndarray, fractional_bits: int, quant_step: int) -> np.ndarray:
    scale = 1 << fractional_bits
    scaled = np.rint(np.asarray(values, dtype=np.float64) * scale).astype(np.int64)
    return round_half_away_array(scaled, quant_step)


def fixed_to_float(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / float(1 << fractional_bits)


def normalize_prompt(prompt: str) -> tuple[str, list[str]]:
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    return " ".join(tokens), tokens


def json_safe_float(value: float) -> float:
    """Round display values without feeding them back into rendering math."""
    if not math.isfinite(value):
        return value
    return float(f"{value:.12g}")


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "sum": 0.0, "abs_sum": 0.0, "max_abs": 0.0}
    return {
        "count": int(array.size),
        "sum": json_safe_float(float(array.sum())),
        "abs_sum": json_safe_float(float(np.abs(array).sum())),
        "max_abs": json_safe_float(float(np.abs(array).max())),
    }


def encode_png(pixels: np.ndarray, compress_level: int = 9) -> bytes:
    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"unsupported pixel shape for PNG: {array.shape}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=compress_level)
    return buffer.getvalue()


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_bytes_fast(path: str | Path, data: bytes) -> None:
    """Write inside a private temp generation; the whole directory is atomically renamed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(data)


def write_json_fast(path: str | Path, value: Any) -> None:
    write_bytes_fast(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be an object")
    return value


def resolve_relative(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()
