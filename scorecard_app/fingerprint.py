from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Iterable


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


def sha256_value(value: Any) -> str:
    return sha256_text(canonical_json(value))


def derive_seed(base_seed: int, *parts: object) -> int:
    material = ":".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def stable_float(value: float, digits: int = 12) -> float:
    result = round(float(value), digits)
    if result == 0.0:
        return 0.0
    return result


def config_fingerprint(experiment: dict[str, Any]) -> str:
    value = copy.deepcopy(experiment)
    if isinstance(value.get("features"), list):
        value["features"] = sorted(value["features"])
    snapshot = value.get("snapshot")
    if isinstance(snapshot, dict):
        snapshot.pop("expected_global_version", None)
        if not snapshot:
            value.pop("snapshot", None)
    return sha256_value({"kind": "experiment", "value": value})


def data_fingerprint(
    columns: dict[str, list[Any]],
    target: list[Any],
    target_name: str,
    schema: dict[str, Any] | None = None,
) -> str:
    schema_value = copy.deepcopy(schema)
    if isinstance(schema_value, dict) and isinstance(schema_value.get("columns"), list):
        schema_value["columns"] = sorted(
            schema_value["columns"],
            key=lambda item: str(item.get("name", "")) if isinstance(item, dict) else str(item),
        )
    payload = {
        "kind": "generated_table",
        "target_name": target_name,
        "target": target,
        "columns": {name: columns[name] for name in sorted(columns)},
        "schema": schema_value,
    }
    return sha256_value(payload)


def column_content_fingerprint(
    name: str,
    column_type: str,
    values: Iterable[Any],
    target: Iterable[Any],
    target_name: str,
    positive_value: Any,
) -> str:
    return sha256_value(
        {
            "kind": "column_content",
            "name": name,
            "type": column_type,
            "values": list(values),
            "target_name": target_name,
            "target": list(target),
            "positive_value": positive_value,
        }
    )


def binning_params_fingerprint(max_bins: int, min_bin_ratio: float, prebins: int) -> str:
    return sha256_value(
        {
            "kind": "binning_params",
            "max_bins": max_bins,
            "min_bin_ratio": stable_float(min_bin_ratio),
            "prebins": prebins,
        }
    )


def direction_priority_fingerprint(
    direction: str,
    conflict_priority: list[str],
    max_rounds: int,
) -> str:
    return sha256_value(
        {
            "kind": "direction_priority",
            "direction": direction,
            "conflict_priority": conflict_priority,
            "max_rounds": max_rounds,
        }
    )


def stage_cache_key(stage: str, *fingerprints: str) -> str:
    return sha256_value({"kind": "stage_cache", "stage": stage, "inputs": list(fingerprints)})


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
