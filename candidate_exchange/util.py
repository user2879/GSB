from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(sanitize_for_json(value))).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    text = json.dumps(
        sanitize_for_json(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, text + "\n")


def decimal_from(value: float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def quantize_scaled(value: float | int | str, decimal_places: int) -> int:
    scale = Decimal(10) ** decimal_places
    return int(
        (decimal_from(value) * scale).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def quantize_int(value: float | int | str) -> int:
    return int(decimal_from(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def quantize_milli(value: float | int | str) -> int:
    return int((decimal_from(value) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def scaled_to_float(value: int, decimal_places: int) -> float:
    return float(Decimal(value) / (Decimal(10) ** decimal_places))


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def histogram(values: list[float], *, places: int = 3) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for value in values:
        key = f"{abs(float(value)):.{places}f}"
        buckets[key] = buckets.get(key, 0) + 1
    return {key: buckets[key] for key in sorted(buckets, key=lambda item: (float(item), item))}


def remove_tree(path: str | Path) -> None:
    import shutil

    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def copy_tree(source: str | Path, destination: str | Path) -> None:
    import shutil

    shutil.copytree(source, destination)


def relative_posix(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
