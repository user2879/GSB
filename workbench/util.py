from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def round_half_away(value: int | float | Decimal) -> int:
    """Round to nearest integer, with exact halves moving away from zero."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError("non-finite value")
    if decimal_value >= 0:
        return int((decimal_value + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR"))
    return -int((-decimal_value + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR"))


def round_div(numerator: int, denominator: int) -> int:
    """Integer division with deterministic half-away-from-zero rounding."""
    if denominator == 0:
        raise ZeroDivisionError("denominator is zero")
    negative = (numerator < 0) != (denominator < 0)
    numerator = abs(numerator)
    denominator = abs(denominator)
    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return -quotient if negative else quotient


def fixed_from_float(value: float | int, scale: int) -> int:
    return round_half_away(Decimal(str(value)) * scale)


def fixed_from_ratio(numerator: int | float, denominator: int | float, scale: int) -> int:
    den = Decimal(str(denominator))
    if den == 0:
        return 0
    return round_half_away(Decimal(str(numerator)) * scale / den)


def fixed_to_string(value: int, places: int) -> str:
    if places == 0:
        return str(value)
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    scale = 10**places
    return f"{sign}{absolute // scale}.{absolute % scale:0{places}d}"


def fixed_to_float(value: int, places: int) -> float:
    return value / (10**places)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def deep_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_patch(result[key], value)
        else:
            result[key] = value
    return result


def ensure_clean_directory(path: str | Path, root: str | Path) -> Path:
    """Clear a directory while refusing to escape the configured workspace root."""
    directory = Path(path).resolve()
    workspace = Path(root).resolve()
    try:
        directory.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"directory {directory} is outside workspace {workspace}") from exc
    if directory == workspace:
        raise ValueError("refusing to clear workspace root")
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
        for child in sorted(directory.iterdir(), key=lambda item: item.name, reverse=True):
            if child.is_dir() and not child.is_symlink():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_publish_directory(staging: str | Path, live: str | Path) -> None:
    """Publish a fully written directory; the live path is never partially updated."""
    staging_path = Path(staging)
    live_path = Path(live)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    backup = live_path.with_name(f".{live_path.name}.previous")
    if backup.exists():
        import shutil

        shutil.rmtree(backup)
    if live_path.exists():
        os.replace(live_path, backup)
    try:
        os.replace(staging_path, live_path)
    except Exception:
        if backup.exists() and not live_path.exists():
            os.replace(backup, live_path)
        raise
    if backup.exists():
        import shutil

        shutil.rmtree(backup)


def sort_dict_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))
