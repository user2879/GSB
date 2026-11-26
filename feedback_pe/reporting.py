from __future__ import annotations

from typing import Any

from .config import config_fingerprint
from .pipeline import LAYER_EXTRAPOLATION, LAYER_INSIDE
from .runner import plan_tasks
from .samples import business_code_counts


def _empty_layer_aggregate() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "point_count": 0,
        "score_mean": 0.0,
        "weight_mean": 0.0,
        "max_score_deviation": 0.0,
        "max_weight_deviation": 0.0,
    }


def build_report(config: dict[str, Any], samples: list[dict[str, Any]],
                 failures: list[dict[str, Any]], workspace: Any,
                 run_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    samples_by_id = {item["id"]: item for item in sorted(samples, key=lambda x: x["id"])}
    desired = plan_tasks(config, samples_by_id)

    stage_results: list[dict[str, Any]] = []
    for sort_key, metadata in desired.items():
        stage = metadata["stage"]
        payload = workspace.stores[stage].get(metadata["cache_key"])
        stage_results.append({
            "sort_key": sort_key,
            "sort_hash": metadata["sort_hash"],
            "sample_id": metadata["sample_id"],
            "scheme": metadata["scheme"],
            "stage": stage,
            "cache_key": metadata["cache_key"],
            "available": payload is not None,
            "payload": payload,
        })

    summaries: list[dict[str, Any]] = []
    layer_aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    cross: dict[str, Any] = {}
    all_flips: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []

    for sample_id in sorted(samples_by_id):
        sample = samples_by_id[sample_id]
        for scheme in sorted(
                [item for item in config["schemes"] if item.get("enabled", True)],
                key=lambda item: item["name"]):
            scheme_name = scheme["name"]
            layer_aggregates.setdefault(scheme_name, {
                LAYER_INSIDE: _empty_layer_aggregate(),
                LAYER_EXTRAPOLATION: _empty_layer_aggregate(),
            })
            cross.setdefault(scheme_name, {
                "first_exceeded_stage_counts": {
                    "position": 0, "score": 0, "weight": 0, "none": 0,
                },
                "stage_attribution": {
                    stage: {
                        "samples_exceeded": 0,
                        "max_relative_error": None,
                        "max_absolute_error": 0.0,
                    }
                    for stage in ("position", "score", "weight")
                },
                "flip_attributed_sources": {},
                "flip_dominant_sources": {},
                "flip_count": 0,
            })

            def payload_for(stage: str) -> dict[str, Any] | None:
                key = desired[f"{sample_id}::{scheme_name}::{stage}"]["cache_key"]
                return workspace.stores[stage].get(key)

            position_payload = payload_for("position")
            stats_payload = payload_for("stats")
            error_payload = payload_for("error")
            position_digest = (
                position_payload["position_digest"] if position_payload else None
            )

            layers = stats_payload["layers"] if stats_payload else {}
            for layer_name in (LAYER_INSIDE, LAYER_EXTRAPOLATION):
                item = layers.get(layer_name)
                if not item:
                    continue
                aggregate = layer_aggregates[scheme_name][layer_name]
                count = int(item["key_count"])
                aggregate["sample_count"] += int(item["sample_count"])
                aggregate["point_count"] += count
                if count:
                    aggregate["score_mean"] += float(item["score_mean"]) * count
                    aggregate["weight_mean"] += float(item["weight_mean"]) * count
                aggregate["max_score_deviation"] = max(
                    aggregate["max_score_deviation"],
                    float(item["max_score_deviation"]),
                )
                aggregate["max_weight_deviation"] = max(
                    aggregate["max_weight_deviation"],
                    float(item["max_weight_deviation"]),
                )

            trace = error_payload["trace"] if error_payload else None
            flips = error_payload["flips"] if error_payload else None
            first_stage = trace["first_exceeded_stage"] if trace else None
            cross[scheme_name]["first_exceeded_stage_counts"][first_stage or "none"] += 1
            if trace:
                for stage_name in ("position", "score", "weight"):
                    stage_trace = trace["stages"][stage_name]
                    attribution = cross[scheme_name]["stage_attribution"][stage_name]
                    if stage_trace["exceeded"]:
                        attribution["samples_exceeded"] += 1
                    rel = stage_trace["max_relative_error"]
                    if rel is not None:
                        attribution["max_relative_error"] = max(
                            attribution["max_relative_error"] or 0.0, rel,
                        )
                    attribution["max_absolute_error"] = max(
                        attribution["max_absolute_error"],
                        float(stage_trace["max_absolute_error"]),
                    )

            flip_records = flips["flips"] if flips else []
            for flip in flip_records:
                all_flips.append(flip)
                cross[scheme_name]["flip_count"] += 1
                source = flip["attributed_source"]
                dominant = flip["dominant_source"]
                cross[scheme_name]["flip_attributed_sources"][source] = (
                    cross[scheme_name]["flip_attributed_sources"].get(source, 0) + 1
                )
                cross[scheme_name]["flip_dominant_sources"][dominant] = (
                    cross[scheme_name]["flip_dominant_sources"].get(dominant, 0) + 1
                )

            distances = [
                abs(int(sample["query_position"]) - int(position))
                for position in sample["key_positions"]
            ]
            distance_margin = min(
                (abs(distance - int(sample["train_length"])) for distance in distances),
                default=None,
            )
            boundary_reasons = []
            if distance_margin is not None and distance_margin <= 2:
                boundary_reasons.append("relative_distance_within_2_of_train_length")
            if first_stage is not None:
                boundary_reasons.append(f"first_tolerance_exceeded:{first_stage}")
            if flip_records:
                boundary_reasons.append("quantization_rank_flip")
            if boundary_reasons:
                boundary.append({
                    "sample_id": sample_id,
                    "scheme": scheme_name,
                    "distance_boundary_margin": distance_margin,
                    "first_exceeded_stage": first_stage,
                    "flip_count": len(flip_records),
                    "reasons": boundary_reasons,
                })

            summaries.append({
                "sample_id": sample_id,
                "scheme": scheme_name,
                "position_digest": position_digest,
                "layers": layers,
                "first_exceeded_stage": first_stage,
                "first_exceeded_metric": trace["first_exceeded_metric"] if trace else None,
                "extrapolation_point_count": (
                    trace["extrapolation_point_count"] if trace else None
                ),
                "flip_count": len(flip_records),
                "cache_keys": {
                    stage: desired[f"{sample_id}::{scheme_name}::{stage}"]["cache_key"]
                    for stage in ("position", "score", "stats", "error")
                },
            })

    for scheme_name, layers in layer_aggregates.items():
        for layer_name, aggregate in layers.items():
            points = aggregate["point_count"]
            if points:
                aggregate["score_mean"] /= points
                aggregate["weight_mean"] /= points

    all_flips.sort(key=lambda item: (
        item["sample_id"], item["scheme"], item["tier_from"], item["tier_to"],
        item["winner_key_index"], item["loser_key_index"],
    ))
    boundary.sort(key=lambda item: (item["sample_id"], item["scheme"]))

    scheduler_codes = {
        entry["code"]: 1
        for entry in workspace.scheduler.get("pending_entries", [])
    }
    code_counts = business_code_counts(failures, scheduler_codes)

    business = {
        "report_schema": "feedback-pe-business-v1",
        "config_fingerprint": config_fingerprint(config),
        "config": config,
        "samples": sorted(samples, key=lambda item: item["id"]),
        "failures": failures,
        "task_manifest": list(desired.values()),
        "stage_results": stage_results,
        "sample_scheme_summaries": summaries,
        "scheme_layer_stats": layer_aggregates,
        "cross_attribution": cross,
        "flip_attribution": all_flips,
        "boundary_samples": boundary,
        "business_code_counts": code_counts,
    }
    execution = {
        "global_version": workspace.global_version,
        "recovery": workspace.recovery,
        "scheduler": workspace.scheduler,
        "latest_run": run_summary and {
            "reuse": run_summary["reuse"],
            "recompute": [{
                "sort_key": item["sort_key"],
                "cache_key": item["cache_key"],
                "status": item["status"],
            } for item in run_summary["recompute"]],
            "invalidations": run_summary["invalidations"],
            "ready": run_summary["ready"],
            "pending": run_summary["pending"],
            "scheduler": run_summary["scheduler"],
        },
        "cumulative_invalidation_count": len(workspace.invalidation_records),
        "cumulative_reuse_hit_count": len(workspace.reuse_records),
    }
    return {"business": business, "execution": execution}
