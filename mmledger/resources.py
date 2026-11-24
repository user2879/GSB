"""Integer fixed-point resource accounting and multimodal quality summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .jsonutil import MISSING, hash_json, is_missing, json_ready
from .models import (
    IMAGE_ONLY,
    IMAGE_TEXT,
    BatchNormalized,
    BatchResource,
    ModelSpec,
    OwnershipDecision,
)


TARGET_UTILIZATION_CAP_PPM = 1_000_000
DURATION_BUCKETS_MS = (5, 10, 20, 50, 100, 250, 500)


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def model_resource_hash(model: ModelSpec, batch_size: int) -> str:
    return hash_json(
        {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "quantization_bits": model.quantization_bits,
            "macs_per_request": model.macs_per_request,
            "quantized_parameter_units": model.quantized_parameter_units,
            "batch_size": batch_size,
        }
    )


def compute_batch_resource(
    batch: BatchNormalized,
    decisions: Sequence[OwnershipDecision],
    *,
    model: ModelSpec,
    batch_size: int,
    cache_hit: bool = False,
) -> BatchResource:
    relevant = [decision for decision in decisions if decision.batch_id == batch.batch_id]
    request_count = len(relevant)
    observed = [int(item.duration_ms) for item in relevant if _strict_int(item.duration_ms)]
    missing_count = request_count - len(observed)
    duration: Any = max(observed) if observed else MISSING
    macs = request_count * model.macs_per_request
    quantized_macs = (macs * model.quantization_bits + 7) // 8
    if _strict_int(duration):
        utilization: Any = min(
            TARGET_UTILIZATION_CAP_PPM,
            (request_count * TARGET_UTILIZATION_CAP_PPM) // (max(1, batch_size) * max(1, duration)),
        )
    else:
        utilization = MISSING
    return BatchResource(
        batch_id=batch.batch_id,
        worker_id=batch.worker_id,
        parameter_version=batch.param_version,
        request_count=request_count,
        duration_ms=duration,
        duration_observed_count=len(observed),
        duration_missing_count=missing_count,
        macs=macs,
        quantized_macs=quantized_macs,
        quantized_parameter_units=model.quantized_parameter_units,
        utilization_ppm=utilization,
        cache_hit=cache_hit,
        task_key=f"resources|{batch.batch_id}|{batch.worker_id}",
    )


def _bucket(value: int) -> str:
    upper = None
    for candidate in DURATION_BUCKETS_MS:
        if value < candidate:
            return f"lt_{candidate}"
        upper = candidate
    return f"gte_{upper}"


def quality_summary(
    decisions: Sequence[OwnershipDecision],
    resources: Sequence[BatchResource],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_authoritative_requests": len(decisions),
        "kinds": {},
        "duration_histogram_ms": {},
        "missing_duration_total": 0,
        "batch_resource_totals": {
            "request_count": sum(item.request_count for item in resources),
            "macs": sum(item.macs for item in resources),
            "quantized_macs": sum(item.quantized_macs for item in resources),
            "observed_batch_count": sum(1 for item in resources if _strict_int(item.duration_ms)),
            "missing_duration_batch_count": sum(1 for item in resources if not _strict_int(item.duration_ms)),
        },
    }
    for kind in (IMAGE_TEXT, IMAGE_ONLY):
        selected = [decision for decision in decisions if decision.kind == kind]
        durations = [int(item.duration_ms) for item in selected if _strict_int(item.duration_ms)]
        hist: dict[str, int] = {}
        for value in DURATION_BUCKETS_MS:
            hist[f"lt_{value}"] = 0
        hist[f"gte_{DURATION_BUCKETS_MS[-1]}"] = 0
        for duration in durations:
            hist[_bucket(duration)] = hist.get(_bucket(duration), 0) + 1
        total = sum(durations)
        summary["kinds"][kind] = {
            "count": len(selected),
            "observed_duration_count": len(durations),
            "missing_duration_count": len(selected) - len(durations),
            "duration_sum_ms": total,
            "mean_duration_us": (total * 1000 // len(durations)) if durations else MISSING,
            "min_duration_ms": min(durations) if durations else MISSING,
            "max_duration_ms": max(durations) if durations else MISSING,
            "duration_histogram_ms": dict(sorted(hist.items())),
            "complete_receipt_count": sum(
                1 for decision in selected if not is_missing(decision.duration_ms)
            ),
        }
        summary["missing_duration_total"] += len(selected) - len(durations)
        for key, value in sorted(hist.items()):
            summary["duration_histogram_ms"][key] = summary["duration_histogram_ms"].get(key, 0) + value
    return json_ready(summary)


def compute_invalidation(
    *,
    workers: Iterable[Any],
    batches: Sequence[BatchNormalized],
    previous_cache: Mapping[str, Any] | None,
    merge_rules_hash: str,
    model_hash: str,
    batch_size: int,
) -> dict[str, Any]:
    current_worker_versions = {worker.worker_id: worker.parameter_version for worker in workers}
    previous_cache = previous_cache or {}
    previous_workers = previous_cache.get("worker_parameter_versions", {})
    previous_batches = previous_cache.get("batches", {})
    if not isinstance(previous_workers, dict):
        previous_workers = {}
    if not isinstance(previous_batches, dict):
        previous_batches = {}
    changed_workers = []
    for worker_id in sorted(current_worker_versions):
        old = previous_workers.get(worker_id, MISSING)
        new = current_worker_versions[worker_id]
        if is_missing(old) or old != new:
            changed_workers.append(
                {
                    "worker_id": worker_id,
                    "previous_parameter_version": json_ready(old),
                    "current_parameter_version": new,
                }
            )
    previous_merge_hash = previous_cache.get("merge_rules_hash")
    previous_model_hash = previous_cache.get("model_hash")
    previous_batch_size = previous_cache.get("batch_size")
    ownership_global_invalidated = bool(previous_cache) and previous_merge_hash != merge_rules_hash
    resource_global_invalidated = bool(previous_cache) and (
        previous_model_hash != model_hash or previous_batch_size != batch_size
    )

    normalization_invalidated = []
    resource_invalidated = []
    reused = []
    reasons_by_batch: dict[str, list[str]] = {}
    for batch in batches:
        reasons: list[str] = []
        previous = previous_batches.get(batch.batch_id)
        if not previous_cache:
            reasons.append("cold_cache")
        elif not isinstance(previous, dict):
            reasons.append("new_batch")
        else:
            prev_worker = previous.get("worker_id")
            prev_pv = previous.get("worker_parameter_version")
            if prev_worker != batch.worker_id:
                reasons.append("batch_worker_changed")
            elif prev_pv != current_worker_versions.get(batch.worker_id):
                reasons.append("worker_parameter_version_changed")
            if previous.get("input_hash") != batch.input_hash:
                reasons.append("receipt_or_request_changed")
        if reasons:
            normalization_invalidated.append(batch.batch_id)
            reasons_by_batch[batch.batch_id] = sorted(set(reasons))
        else:
            reused.append(batch.batch_id)
        if reasons or resource_global_invalidated:
            resource_invalidated.append(batch.batch_id)
            if resource_global_invalidated:
                reasons_by_batch.setdefault(batch.batch_id, []).append("resource_configuration_changed")
                reasons_by_batch[batch.batch_id] = sorted(set(reasons_by_batch[batch.batch_id]))

    return {
        "cold_cache": not bool(previous_cache),
        "changed_workers": changed_workers,
        "ownership_global_invalidated": ownership_global_invalidated,
        "resource_global_invalidated": resource_global_invalidated,
        "normalization_invalidated_batches": sorted(normalization_invalidated),
        "resource_invalidated_batches": sorted(resource_invalidated),
        "reused_batches": sorted(reused),
        "reasons_by_batch": {key: reasons_by_batch[key] for key in sorted(reasons_by_batch)},
    }


def canonical_dependency_plan(
    batches: Sequence[BatchNormalized],
    *,
    invalidation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Stable, input-derived cache eligibility plan for ledger comparisons.

    Physical cache hits are reported separately in the run audit. This plan is
    a function of final dependencies and therefore is equal for a cold full
    rebuild and an incremental build with the same final input.
    """

    reasons = invalidation.get("reasons_by_batch", {})
    plan = []
    for batch in sorted(batches, key=lambda item: (item.batch_id, item.worker_id)):
        reason_list = reasons.get(batch.batch_id, []) if isinstance(reasons, dict) else []
        invalidated = bool(reason_list)
        plan.append(
            {
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "parameter_version": batch.param_version,
                "dependency_hash": batch.input_hash,
                "cache_action": "invalidate" if invalidated else "reusable",
                "reusable_after_commit": not invalidated,
                "local_invalidation_reasons": list(reason_list),
            }
        )
    return json_ready(plan)
