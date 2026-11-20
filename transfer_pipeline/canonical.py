"""Canonical serialization, hashing, and report projection helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    """Return a stable compact JSON representation used for all hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_task_sort_key(task_name: str, target_layer: str) -> str:
    return sha256_text(f"{task_name}\x00{target_layer}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def atomic_write_text(path: os.PathLike[str] | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    atomic_write_text(path, pretty_json(value))


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def materialized_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match a rebuild from the same final input state.

    Lifecycle fields (reused/invalidated) and snapshot lineage intentionally
    describe *how* a run was reached, not the materialized model state.
    """
    task_results = []
    for task in report.get("task_results", []):
        clean = dict(task)
        clean.pop("source", None)
        task_results.append(clean)
    base = report.get("base")
    if isinstance(base, dict):
        base = {
            "fingerprint": base.get("fingerprint"),
            "layers": base.get("layers"),
        }
    return {
        "input_state_hash": report.get("input_state_hash"),
        "config_hash": report.get("config_hash"),
        "base": base,
        "task_results": task_results,
        "adapter_summary": report.get("adapter_summary", []),
        "loss_trajectories": report.get("loss_trajectories", []),
        "validation_metrics": report.get("validation_metrics", []),
        "validation_summary": report.get("validation_summary", {}),
        "threshold_failures": report.get("threshold_failures", []),
        "failures": report.get("failures", []),
        "injection_hits": report.get("injection_hits", []),
        "injection_summary": report.get("injection_summary", {}),
    }


def projections_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return canonical_json(materialized_projection(left)) == canonical_json(
        materialized_projection(right)
    )


def first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Find a compact, deterministic first difference between two JSON values."""
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            return f"{path}: keys differ left_only={only_left} right_only={only_right}"
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None
