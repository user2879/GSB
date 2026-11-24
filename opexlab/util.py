"""Deterministic serialization, hashing and rounding helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, pretty_json(value))


def read_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            values.append(json.loads(line))
    return values


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    text = "".join(canonical_json(value) + "\n" for value in values)
    atomic_write_text(path, text)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def round_half_away(value: float) -> int:
    """Deterministic decimal-style rounding, without banker's rounding."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def div_round_half_away(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize_int_to_step(value: int, step: int) -> int:
    if step <= 1:
        return value
    return div_round_half_away(value, step) * step


def resolve_path(
    value: str,
    *,
    project_root: Path,
    config_dir: Path,
    must_exist: bool = False,
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_candidate = (project_root / candidate).resolve()
    config_candidate = (config_dir / candidate).resolve()
    if root_candidate.exists() or not config_candidate.exists():
        result = root_candidate
    else:
        result = config_candidate
    if must_exist and not result.exists():
        raise FileNotFoundError(str(result))
    return result


def stable_parent_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer")
    return int(value)
