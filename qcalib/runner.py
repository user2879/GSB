"""Top-level run orchestration: staged snapshots, recovery and diagnostics."""
from __future__ import annotations

from typing import Any

from .pipeline import Pipeline, PipelineInputs
from .snapshot import SnapshotStore


def run_pipeline(inputs: PipelineInputs, store: SnapshotStore,
                 previous_cache: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline = Pipeline(store)
    result = pipeline.run(inputs, previous_cache=previous_cache)

    content_versions = result["content_versions"]
    config_hash = result["config_hash"]
    store.commit_stage("parse_validate",
                       {"content_versions": content_versions,
                        "layers": sorted(content_versions),
                        "failures": result["failures"]},
                       config_hash, content_versions)
    store.commit_stage("quantile_calibration",
                       {name: artifact["group_calibration"]
                        for name, artifact in result["layers"].items()},
                       config_hash, content_versions)
    store.commit_stage("fixed_point_requantization",
                       {name: {"groups": artifact["group_quantization"],
                               "dequant_weights": artifact["dequant_weights"],
                               "error_summary": artifact["error_summary"]}
                        for name, artifact in result["layers"].items()},
                       config_hash, content_versions)
    store.commit_stage("distillation_correction",
                       {name: {"distillation": artifact["distillation"],
                               "stage_attribution": artifact["stage_attribution"]}
                        for name, artifact in result["layers"].items()},
                       config_hash, content_versions)
    store.commit_stage("ledger_aggregation",
                       {"layers": result["layers"],
                        "receipt_ledger": result["receipt_ledger"],
                        "reuse_ledger": result["reuse_ledger"],
                        "budget": result["budget"],
                        "failures": result["failures"],
                        "cache": result["cache"],
                        "calibration_metadata": result["calibration_metadata"]},
                       config_hash, content_versions)
    return result


def recover_ledger(store: SnapshotStore, config_hash: str,
                   content_versions: dict[str, int]) -> dict[str, Any] | None:
    stages = store.recoverable_stages(config_hash, content_versions)
    if "ledger_aggregation" not in stages:
        return None
    document = store.load_stage("ledger_aggregation")
    return document["payload"] if document else None


def diagnostic_report(result: dict[str, Any]) -> dict[str, Any]:
    layers = result["layers"]
    return {
        "layer_count": len(layers),
        "failure_count": len(result["failures"]),
        "failure_codes": sorted({item["code"] for item in result["failures"]}),
        "budget_consumed": result["budget"]["consumed_cal_points"],
        "quota_aborted_layers": result["budget"]["quota_aborted_layers"],
        "pending_task_count": len(result["budget"]["pending_tasks"]),
        "cache_hits": sum(1 for item in result["reuse_ledger"] if item["hit"]),
        "cache_misses": sum(1 for item in result["reuse_ledger"]
                            if not item["hit"]),
        "layers": {
            name: {
                "mean_abs_error": artifact["error_summary"]["mean_abs_error"],
                "saturated_count": artifact["error_summary"]["saturated_count"],
                "distill_increased": artifact["distillation"]["increased"],
                "first_stage_over_tolerance":
                    artifact["stage_attribution"]["first_stage_over_tolerance"],
            }
            for name, artifact in sorted(layers.items())
        },
    }
