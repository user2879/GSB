"""Small deterministic and filesystem helpers shared by the pipeline."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest_obj(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def stable_task_key(stage: str, frame: int | None = None, step: int | None = None,
                    block: int | None = None, kind: str | None = None) -> str:
    parts = [stage]
    if frame is not None:
        parts.append(f"frame={frame:04d}")
    if step is not None:
        parts.append(f"step={step:04d}")
    if block is not None:
        parts.append(f"row={block:04d}")
    if kind is not None:
        parts.append(kind)
    return "/".join(parts)


def stage_order(stage: str) -> int:
    return {
        "encode": 10,
        "fusion": 20,
        "latent": 30,
        "denoise": 40,
        "decode": 50,
        "block": 60,
    }.get(stage, 999)


def round_div(value: int, divisor: int) -> int:
    """Integer division rounded to nearest, with halves away from zero."""
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    sign = -1 if value < 0 else 1
    value = abs(value)
    return sign * ((value + divisor // 2) // divisor)


def round_div_array(values: np.ndarray, divisor: int) -> np.ndarray:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    values = np.asarray(values, dtype=np.int64)
    signed = np.sign(values).astype(np.int64)
    return signed * ((np.abs(values) + divisor // 2) // divisor)


def quantize_int(value: int, level: int) -> tuple[int, int]:
    """Quantize to a power-of-two integer grid and return (value, residual)."""
    if level < 0:
        raise ValueError("quantization level must be non-negative")
    step = 1 << level
    quantized = round_div(value, step) * step
    return quantized, value - quantized


def quantize_array(values: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    if level < 0:
        raise ValueError("quantization level must be non-negative")
    step = 1 << level
    quantized = round_div_array(values, step) * step
    return quantized, values - quantized


def clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def array_bytes(array: np.ndarray, dtype: str = "<i8") -> bytes:
    return np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes(order="C")


def array_digest(array: np.ndarray, dtype: str = "<i8") -> str:
    header = canonical_json({"shape": list(array.shape), "dtype": dtype})
    return sha256_hex(header.encode("utf-8") + b"|" + array_bytes(array, dtype))


def encode_array(array: np.ndarray, dtype: str = "<i8") -> dict[str, Any]:
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "order": "C",
        "data_b64": base64.b64encode(array_bytes(array, dtype)).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(x) for x in payload["shape"])
        raw = base64.b64decode(payload["data_b64"].encode("ascii"), validate=True)
        array = np.frombuffer(raw, dtype=dtype).copy()
        expected = int(np.prod(shape, dtype=np.int64))
        if array.size != expected:
            raise ValueError("array size does not match shape")
        return array.reshape(shape)
    except Exception as exc:  # noqa: BLE001 - convert untrusted cache data
        raise BusinessError("E_CACHE_ARTIFACT", "invalid array payload in cache", {"error": str(exc)}) from exc


def read_json_file(path: Path, code: str = "E_JSON") -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read {path}", {"path": str(path), "error": str(exc)}) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BusinessError(code, f"invalid JSON in {path}", {"path": str(path), "line": exc.lineno,
                                                             "column": exc.colno, "error": exc.msg}) from exc


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
