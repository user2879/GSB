from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Union


JSONValue = Union[None, bool, int, float, str, list, dict]


def is_int(value: Any) -> bool:
    """JSON integer check that deliberately rejects booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def read_json_strict(path: Union[str, Path]) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text, parse_constant=reject_constant)


def write_json_atomic(path: Union[str, Path], value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_copy(source: Union[str, Path], target: Union[str, Path]) -> None:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    with source.open("rb") as src, temp.open("wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temp, target)


def stable_dict(value: Mapping[str, Any]) -> dict:
    """Round-trip through canonical JSON to make display/audit order stable."""

    return json.loads(canonical_json(dict(value)))


def first_diff(ledger_value: Any, recomputed_value: Any, prefix: str = "") -> tuple[str, Any, Any] | None:
    """Return the first deterministic structural difference."""

    if type(ledger_value) is not type(recomputed_value):
        # int/bool are distinct in JSON semantics; keep the type check strict.
        return (prefix or "$", recomputed_value, ledger_value)
    if isinstance(ledger_value, dict):
        keys = sorted(set(ledger_value) | set(recomputed_value))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in ledger_value:
                return (path, recomputed_value[key], "<missing>")
            if key not in recomputed_value:
                return (path, "<missing>", ledger_value[key])
            diff = first_diff(ledger_value[key], recomputed_value[key], path)
            if diff is not None:
                return diff
        return None
    if isinstance(ledger_value, list):
        if len(ledger_value) != len(recomputed_value):
            return (f"{prefix}.length", len(recomputed_value), len(ledger_value))
        for index, (left, right) in enumerate(zip(ledger_value, recomputed_value)):
            diff = first_diff(left, right, f"{prefix}[{index}]")
            if diff is not None:
                return diff
        return None
    if ledger_value != recomputed_value:
        return (prefix or "$", recomputed_value, ledger_value)
    return None
