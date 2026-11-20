"""Deterministic JSON, hashing and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import QuantCalError


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuantCalError("E_FORMAT_INVALID", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
        )
    except QuantCalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuantCalError(
            "E_FORMAT_INVALID", f"cannot parse JSON {source}: {exc}"
        ) from exc


def load_jsonl(path: str | Path) -> list[Any]:
    source = Path(path)
    rows: list[Any] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    rows.append(
                        json.loads(text, object_pairs_hook=_no_duplicate_object)
                    )
                except (json.JSONDecodeError, QuantCalError) as exc:
                    raise QuantCalError(
                        "E_FORMAT_INVALID",
                        f"invalid JSONL at {source}:{line_number}: {exc}",
                    ) from exc
    except OSError as exc:
        raise QuantCalError(
            "E_FORMAT_INVALID", f"cannot read JSONL {source}: {exc}"
        ) from exc
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QuantCalError("E_FORMAT_INVALID", f"value is not valid JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise QuantCalError("E_FORMAT_INVALID", f"value is not valid JSON: {exc}") from exc


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json_text(value))


def as_finite_float(value: Any, field_name: str, tensor: str | None = None) -> float:
    if isinstance(value, bool):
        raise QuantCalError(
            "E_FORMAT_INVALID", f"{field_name} must be a number", tensor=tensor
        )
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"nan", "+nan", "-nan"}:
            return math.nan
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        if lowered in {"-inf", "-infinity"}:
            return -math.inf
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QuantCalError(
            "E_FORMAT_INVALID", f"{field_name} must be a number", tensor=tensor
        ) from exc
    return result


def half_away_round(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def clean_float(value: float) -> float:
    """Remove negative zero and tiny binary noise from arithmetic residuals."""
    if value == 0:
        return 0.0
    return float(value)

