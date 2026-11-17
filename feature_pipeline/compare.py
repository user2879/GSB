"""Layer-aware equality checks for incremental and full rebuild states."""

from __future__ import annotations

from typing import Any

from .state import RuntimeState


def compare_states(incremental: RuntimeState, full: RuntimeState) -> dict[str, Any]:
    feature_equal = incremental.features == full.features
    sample_equal = incremental.samples == full.samples
    cache_equal = incremental.cache_ledger == full.cache_ledger
    event_equal = incremental.events == full.events
    mismatches: list[dict[str, Any]] = []
    if not event_equal:
        mismatches.append(_first_mismatch("events", incremental.events, full.events))
    if not feature_equal:
        mismatches.append(_first_mismatch("features", incremental.features, full.features))
    if not cache_equal:
        mismatches.append(_first_mismatch("cache_ledger", incremental.cache_ledger, full.cache_ledger))
    if not sample_equal:
        mismatches.append(_first_mismatch("samples", incremental.samples, full.samples))
    return {
        "equal": event_equal and feature_equal and sample_equal and cache_equal,
        "event_layer_equal": event_equal,
        "feature_layer_equal": feature_equal,
        "sample_layer_equal": sample_equal,
        "cache_ledger_equal": cache_equal,
        "counts": {
            "incremental": {
                "events": len(incremental.events),
                "features": len(incremental.features),
                "samples": len(incremental.samples),
            },
            "full": {
                "events": len(full.events),
                "features": len(full.features),
                "samples": len(full.samples),
            },
        },
        "mismatches": mismatches,
    }


def _first_mismatch(layer: str, left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "layer": layer,
                "index": index,
                "incremental": left[index],
                "full": right[index],
            }
    return {
        "layer": layer,
        "index": limit,
        "incremental": left[limit] if index_in_range(limit, left) else None,
        "full": right[limit] if index_in_range(limit, right) else None,
    }


def index_in_range(index: int, values: list[Any]) -> bool:
    return 0 <= index < len(values)
