"""Canonical comparison helpers for deterministic acceptance checks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import canonical_json


def normalize_report(
    report: dict[str, Any],
    *,
    cache_status: bool = False,
    runtime: bool = False,
) -> dict[str, Any]:
    """Return the field surface that must be reproducible.

    Cache hit status is execution-history dependent and is therefore removed
    for incremental-vs-blank product comparison.  Cold serial/parallel runs can
    retain it using ``cache_status=True``.
    """

    result = deepcopy(report)
    if not runtime:
        result.pop("snapshot", None)
        concurrency = result.get("concurrency")
        if isinstance(concurrency, dict):
            concurrency["serial"] = "normalized"
            concurrency["observed_max_active_tasks"] = "normalized"

    if not cache_status:
        result.pop("reused_cache_keys", None)
        result.pop("recomputed_cache_keys", None)
        result.pop("cache_summary", None)
        for entry in result.get("cache_ledger", []):
            entry.pop("status", None)
            entry.pop("reason", None)
    return result


def canonical_signature(report: dict[str, Any], **kwargs: Any) -> str:
    return canonical_json(normalize_report(report, **kwargs))


def deep_diff(
    left: Any, right: Any, path: str = "$", limit: int = 20
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    _walk(left, right, path, differences, limit)
    return differences


def _walk(
    left: Any,
    right: Any,
    path: str,
    differences: list[dict[str, Any]],
    limit: int,
) -> None:
    if len(differences) >= limit:
        return
    if type(left) is not type(right):
        differences.append(
            {"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__}
        )
        return
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            differences.append({"path": f"{path}.{key}", "kind": "missing_right"})
        for key in sorted(right_keys - left_keys):
            differences.append({"path": f"{path}.{key}", "kind": "missing_left"})
        for key in sorted(left_keys & right_keys):
            _walk(left[key], right[key], f"{path}.{key}", differences, limit)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            differences.append(
                {"path": path, "kind": "length", "left": len(left), "right": len(right)}
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _walk(left_item, right_item, f"{path}[{index}]", differences, limit)
            if len(differences) >= limit:
                return
        return
    if left != right:
        differences.append({"path": path, "kind": "value", "left": left, "right": right})


def reports_equal(left: dict[str, Any], right: dict[str, Any], **kwargs: Any) -> bool:
    return not deep_diff(normalize_report(left, **kwargs), normalize_report(right, **kwargs))
