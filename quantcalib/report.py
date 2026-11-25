"""Diagnostic report assembly."""

from __future__ import annotations

from pathlib import Path
import json


def build_report(result: dict) -> dict:
    layer_diagnostics = []
    for layer in result["layers"]:
        groups = []
        for group in layer["groups"]:
            quant = group["quantization"]
            groups.append({
                "group_index": group["group_index"],
                "scale": group["calibration"]["scale"],
                "zero_point": group["calibration"]["zero_point"],
                "over_tolerance_count": sum(quant["over_tolerance"]),
                "saturation_count": quant["saturation_count"],
                "error_source_counts": _count(quant["error_sources"]),
            })
        layer_diagnostics.append({
            "layer": layer["layer"],
            "bits": layer["bits"],
            "group_granularity": layer["group_granularity"],
            "groups": groups,
        })

    return {
        "global_version": result["global_version"],
        "resumed": result["resumed"],
        "resume_reason": result["resume_reason"],
        "layer_diagnostics": layer_diagnostics,
        "distillation": {
            "temperature": result["distillation"]["temperature"],
            "steps": result["distillation"]["steps"],
            "sample_deviations_before":
                result["distillation"]["sample_deviations_before"],
            "sample_deviations_after":
                result["distillation"]["sample_deviations_after"],
            "flagged_samples": result["distillation"]["flagged_samples"],
        },
        "authoritative_receipts": [
            {
                "layer": entry["layer"],
                "authority_source": entry["authority_source"],
                "rationale": entry["rationale"],
                "coverage_permille": entry["coverage_permille"],
                "group_scale_digest": entry["group_scale_digest"],
            }
            for entry in result["ledger"]
        ],
        "invalidation_boundaries": result["invalidation_boundaries"],
        "reuse": {
            "hits": result["reuse"]["hit_count"],
            "recomputes": result["reuse"]["recompute_count"],
        },
        "schedule": {
            "consumed_budget": result["schedule"]["consumed_budget"],
            "pending_count": len(result["schedule"]["pending"]),
            "failures": result["failures"],
        },
        "fingerprints": result["fingerprints"],
    }


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def write_report(path: str | Path, result: dict) -> None:
    Path(path).write_text(
        json.dumps(build_report(result), indent=2, sort_keys=True,
                   ensure_ascii=False),
        encoding="utf-8",
    )
