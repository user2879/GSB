from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNTIME_LEDGER_KEYS = {
    "disposition",
    "reuse_hit",
    "invalidation_reason",
    "reused_from_global_version",
}
RUNTIME_ARTIFACT_KEYS = {
    "disposition",
    "reuse_hit",
    "invalidation_reason",
    "invalidation_basis",
    "reused_from_global_version",
}


def canonical_state(state: dict[str, Any]) -> dict[str, Any]:
    """Project all deterministic business content and canonical ledger refs."""

    cleaning = []
    for record in state.get("cleaning", {}).get("records", []):
        cleaning.append(_strip(record, RUNTIME_ARTIFACT_KEYS))

    dedup_shards = []
    for shard in state.get("dedup", {}).get("shards", []):
        dedup_shards.append(_strip(shard, RUNTIME_ARTIFACT_KEYS))

    sampling = _strip(state.get("sampling"), RUNTIME_ARTIFACT_KEYS) if state.get("sampling") else None
    summary = _strip(state.get("summary"), RUNTIME_ARTIFACT_KEYS) if state.get("summary") else None

    ledger = []
    for entry in state.get("ledger", {}).get("entries", []):
        ledger.append(
            {
                key: value
                for key, value in entry.items()
                if key not in RUNTIME_LEDGER_KEYS
            }
        )

    return {
        "config_hash": state["manifest"]["config_hash"],
        "dataset_fingerprint": state["manifest"]["dataset_fingerprint"],
        "cleaning": cleaning,
        "dedup_shards": dedup_shards,
        "sampling": sampling,
        "summary": summary,
        "ledger": ledger,
    }


def execution_plan(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Warm-cache disposition list; stable across budget cut points/workers."""

    return [
        {
            "ordinal": entry["ordinal"],
            "entry_id": entry["entry_id"],
            "disposition": entry["disposition"],
            "invalidation_reason": entry.get("invalidation_reason"),
        }
        for entry in sorted(state.get("ledger", {}).get("entries", []), key=lambda item: item["ordinal"])
    ]


def diff_objects(left: Any, right: Any, path: str = "$", limit: int = 20) -> list[str]:
    diffs: list[str] = []
    _diff(left, right, path, diffs, limit)
    return diffs


def _diff(left: Any, right: Any, path: str, diffs: list[str], limit: int) -> None:
    if len(diffs) >= limit:
        return
    if type(left) is not type(right) and not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
        diffs.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{key}: missing in right")
        for key in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{key}: missing in left")
        for key in sorted(left_keys & right_keys):
            _diff(left[key], right[key], f"{path}.{key}", diffs, limit)
            if len(diffs) >= limit:
                return
    elif isinstance(left, list):
        if len(left) != len(right):
            diffs.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _diff(left_item, right_item, f"{path}[{index}]", diffs, limit)
            if len(diffs) >= limit:
                return
    elif left != right:
        diffs.append(f"{path}: {left!r} != {right!r}")


def _strip(value: Any, keys: set[str]) -> Any:
    copied = deepcopy(value)
    if isinstance(copied, dict):
        for key in keys:
            copied.pop(key, None)
    return copied
