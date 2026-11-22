"""Small deterministic and filesystem helpers shared by the lab."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert common numerical containers to stable JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float cannot be serialized: {value!r}")
        return 0.0 if value == 0.0 else value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_hash(prefix: str, value: Any) -> str:
    return f"{prefix}_{sha256_text(canonical_json(value))}"


def short_hash(prefix: str, value: Any, length: int = 16) -> str:
    digest = sha256_text(canonical_json(value))[:length]
    return f"{prefix}_{digest}"


def stable_seed(*parts: Any) -> np.random.SeedSequence:
    """Build a NumPy seed sequence without Python's salted hash()."""

    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    words = [int.from_bytes(digest[index : index + 4], "little") for index in range(0, 32, 4)]
    return np.random.SeedSequence(words)


def child_rng(seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(seed, *parts))


def round_half_up_milli(value: float) -> int:
    """Round to a thousandth using decimal half-away-from-zero semantics."""

    if not math.isfinite(float(value)):
        raise ValueError("cannot round a non-finite value")
    return int(Decimal(str(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP) * 1000)


def milli_to_float(value: int) -> float:
    return float(int(value)) / 1000.0


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, pretty: bool = True) -> None:
    if pretty:
        text = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    else:
        text = canonical_json(value)
    atomic_write_text(path, text + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_tree_link_or_copy(source: Path, destination: Path) -> None:
    """Copy a snapshot cheaply, hard-linking files whenever possible."""

    def copy_file(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=copy_file, dirs_exist_ok=False)


def split_batches(sample_count: int, batch_size: int) -> list[tuple[int, int]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [(start, min(start + batch_size, sample_count)) for start in range(0, sample_count, batch_size)]


def coefficient_digest(coefficients: Iterable[float], intercept: float) -> str:
    return sha256_text(canonical_json({"coefficients": list(coefficients), "intercept": intercept}))


def normalize_path(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def safe_filename(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return cleaned or "model"
