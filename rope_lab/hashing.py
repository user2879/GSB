"""Canonical content fingerprints and stable task keys."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def _mark_nonfinite(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite__": "NaN"}
        if math.isinf(value):
            return {"__nonfinite__": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, Mapping):
        return {str(key): _mark_nonfinite(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_mark_nonfinite(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize data in a stable, NaN-rejecting canonical form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not canonically serializable: {type(value)!r}")


def fingerprint(value: Any) -> str:
    material = canonical_json(_mark_nonfinite(value))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def short_fingerprint(value: Any, length: int = 12) -> str:
    return fingerprint(value)[:length]


def stable_task_key(sample_id: str, dimension: int, stage: str, scheme: str | None = None) -> str:
    """Build the required stable sort key.

    Scheme is kept as a namespace suffix so scheme-independent ordering still
    starts with sample/dimension/stage as required by the task.
    """
    stage_order = {
        "position": 10,
        "rotation": 20,
        "dot_term": 30,
        "score": 40,
        "normalization": 50,
        "stats": 60,
    }.get(stage, 90)
    base = f"{sample_id}|d{int(dimension):04d}|{stage_order:03d}:{stage}"
    if scheme is None:
        return base
    return f"{base}|scheme={scheme}"


def cache_key(stage: str, references: Mapping[str, Any]) -> str:
    material = {"stage": stage, "references": references}
    return f"{stage}_{fingerprint(material)[:20]}"
