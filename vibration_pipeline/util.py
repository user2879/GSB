from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


FLOAT_DECIMALS = 12


def stable_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return value
    rounded = round(value, FLOAT_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_object(value: Any, *, namespace: str = "") -> str:
    digest = hashlib.sha256()
    if namespace:
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def hash_bytes(parts: list[bytes], *, namespace: str = "") -> str:
    digest = hashlib.sha256()
    if namespace:
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\x00")
    for part in parts:
        digest.update(len(part).to_bytes(8, "little", signed=False))
        digest.update(part)
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    return hash_bytes([contiguous.tobytes()], namespace="float64-le")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_half_away(value: float, scale: int, quantum: int) -> int:
    """Quantize a real value to integer ticks in multiples of ``quantum``."""
    units = float(value) * scale / quantum
    if units >= 0:
        quantized_units = math.floor(units + 0.5)
    else:
        quantized_units = math.ceil(units - 0.5)
    return int(quantized_units) * quantum


def stable_key(device_id: str, segment_index: int, algorithm: str, stage: str) -> str:
    return f"{device_id}|{segment_index:010d}|{algorithm}|{stage}"


class ConcurrencyProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_observed = 0

    @contextmanager
    def track(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
            if self._active > self.max_observed:
                self.max_observed = self._active
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
