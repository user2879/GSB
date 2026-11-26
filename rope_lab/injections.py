"""Deterministic fault injection and isolation reporting."""
from __future__ import annotations

import copy
import math
from typing import Any

from .codes import (
    E_BLOCK_INCOMPATIBLE,
    E_KEY_POSITION_DUPLICATE,
    E_SCHEMA_UNKNOWN,
    E_VALUE_NONFINITE,
)
from .engine import LabEngine
from .validation import validate_workspace


EXPECTED_CODES = {
    "duplicate_key_positions": E_KEY_POSITION_DUPLICATE,
    "block_incompatible": E_BLOCK_INCOMPATIBLE,
    "width_out_of_range": E_SCHEMA_UNKNOWN,
    "nonfinite_feature": E_VALUE_NONFINITE,
    "undeclared_scheme": E_SCHEMA_UNKNOWN,
}


def _find_sample(samples: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    for sample in samples:
        if sample.get("sample_id") == sample_id:
            return sample
    raise KeyError(sample_id)


def mutate_for_injection(config: dict[str, Any], samples: list[dict[str, Any]], injection: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = copy.deepcopy(config)
    rows = copy.deepcopy(samples)
    kind = injection["kind"]

    if kind == "duplicate_key_positions":
        sample = _find_sample(rows, injection["target_sample"])
        sample["key_positions"].append(sample["key_positions"][0])
        sample["key_vectors"].append(copy.deepcopy(sample["key_vectors"][0]))

    elif kind == "block_incompatible":
        target = injection.get("target_scheme")
        for scheme in cfg["schemes"]:
            if target is None or scheme["name"] == target:
                scheme["block_size"] = int(injection.get("block_size", 3))
                break

    elif kind == "width_out_of_range":
        cfg["fixed_point"]["fraction_bits"] = int(injection["fraction_bits"])

    elif kind == "nonfinite_feature":
        sample = _find_sample(rows, injection["target_sample"])
        path = injection.get("path", "query_vector[0]")
        if path.startswith("query_vector["):
            index = int(path[len("query_vector[") : -1])
            sample["query_vector"][index] = float("nan") if injection.get("fault_value", "NaN") == "NaN" else float("inf")
        else:
            raise ValueError(f"unsupported injection path: {path}")

    elif kind == "undeclared_scheme":
        sample_id = injection.get("target_sample", rows[0]["sample_id"])
        sample = _find_sample(rows, sample_id)
        sample["schemes"] = [injection["target_scheme"]]

    else:
        raise ValueError(f"unknown injection kind: {kind}")

    return cfg, rows


def run_injection(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    injection: dict[str, Any],
) -> dict[str, Any]:
    cfg, rows = mutate_for_injection(config, samples, injection)
    errors = validate_workspace(cfg, rows)
    expected_code = EXPECTED_CODES[injection["kind"]]
    matching = [error for error in errors if error["code"] == expected_code]
    hit = bool(matching)

    isolation: dict[str, Any] = {"strategy": "deepcopy_workspace_per_injection"}
    if injection["kind"] == "nonfinite_feature":
        engine = LabEngine(cfg, rows)
        report = engine.run(parallel=False)
        isolation.update(
            {
                "engine_completed_with_isolated_failures": True,
                "ledger_failed": sum(1 for entry in report["ledger"] if entry["cache_status"] == "failed"),
                "stats_schemes": [item["scheme"] for item in report["stats_records"]],
                "stats_sample_counts": {
                    item["scheme"]: item["categories"]["inside"]["sample_count"]
                    + item["categories"]["extrapolated"]["sample_count"]
                    for item in report["stats_records"]
                },
                "code_counts": report["code_counts"],
            }
        )
    elif injection["kind"] == "block_incompatible":
        bad_scheme = injection.get("target_scheme")
        healthy_config = copy.deepcopy(config)
        healthy_config["schemes"] = [scheme for scheme in healthy_config["schemes"] if scheme["name"] != bad_scheme]
        engine = LabEngine(healthy_config, rows)
        report = engine.run(parallel=False)
        isolation.update(
            {
                "removed_injected_scheme": bad_scheme,
                "healthy_schemes_computed": [scheme["name"] for scheme in healthy_config["schemes"]],
                "stats_schemes": [item["scheme"] for item in report["stats_records"]],
            }
        )

    return {
        "injection_id": injection["injection_id"],
        "kind": injection["kind"],
        "stage": injection.get("stage"),
        "expected_code": expected_code,
        "hit": hit,
        "matching_error_count": len(matching),
        "errors": errors[:12],
        "isolation": isolation,
    }


def run_injections(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    injections: list[dict[str, Any]],
) -> dict[str, Any]:
    results = [run_injection(config, samples, injection) for injection in injections]
    counts: dict[str, int] = {}
    for result in results:
        key = "hit" if result["hit"] else "miss"
        counts[key] = counts.get(key, 0) + 1
        for error in result["errors"]:
            counts[error["code"]] = counts.get(error["code"], 0) + 1
    return {
        "all_hit": all(result["hit"] for result in results),
        "total": len(results),
        "hit_count": sum(1 for result in results if result["hit"]),
        "results": results,
        "code_counts": dict(sorted(counts.items())),
    }
