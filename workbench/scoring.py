from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Executor
from typing import Any, Callable

from .config import RuntimeConfig
from .util import (
    fixed_from_float,
    fixed_to_float,
    fixed_to_string,
    round_div,
    round_half_away,
    stable_hash,
)


def _normalize_value(value: float, values: list[float], cfg: RuntimeConfig) -> tuple[float, bool, str]:
    mode = cfg.normalization
    if mode == "none":
        return max(0.0, min(1.0, value)), False, "none"
    if mode == "max":
        denominator = max(values)
        if denominator == 0:
            return 0.0, True, "denominator_zero"
        if denominator < 0:
            return 0.0, True, "denominator_nonpositive"
        return value / denominator, False, "max"
    minimum = min(values)
    maximum = max(values)
    denominator = maximum - minimum
    if denominator == 0:
        return 0.0, True, "denominator_zero"
    return (value - minimum) / denominator, False, "minmax"


def _quantize_normalized(normalized: float, cfg: RuntimeConfig) -> tuple[int, int]:
    bucket = round_half_away(normalized * (cfg.quantization_levels - 1))
    bucket = max(0, min(cfg.quantization_levels - 1, bucket))
    fixed = round_div(bucket * cfg.scale, cfg.quantization_levels - 1)
    return bucket, max(0, min(cfg.scale, fixed))


def _normalization_task(
    request_id: str,
    channel: str,
    records: list[dict[str, Any]],
    cfg: RuntimeConfig,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record["item_id"])
    values = [float(record["authority"]["score"]) for record in ordered]
    entries = []
    denominator_zero = False
    denominator_rule = "normal"
    for record, value in zip(ordered, values):
        normalized, is_zero, rule = _normalize_value(value, values, cfg)
        if is_zero:
            denominator_zero = True
            denominator_rule = rule
        bucket, fixed = _quantize_normalized(normalized, cfg)
        entries.append(
            {
                "item_id": record["item_id"],
                "source_score": value,
                "normalized": normalized,
                "quantized_bucket": bucket,
                "normalized_fixed": fixed,
                "normalized_fixed_string": fixed_to_string(fixed, cfg.places),
                "merge_cache_key": record["cache_key"],
                "references": {
                    "shards": sorted(record["references"]["shards"]),
                    "channels": [channel],
                },
            }
        )
    key = stable_hash(
        {
            "stage": "normalize",
            "request_id": request_id,
            "channel": channel,
            "normalization_fingerprint": cfg.normalization_fingerprint,
            "merge_keys": sorted(record["cache_key"] for record in ordered),
        }
    )
    return {
        "cache_key": key,
        "request_id": request_id,
        "channel": channel,
        "denominator_zero": denominator_zero,
        "denominator_rule": denominator_rule,
        "entries": entries,
    }


def normalize_channels(
    merge_records: list[dict[str, Any]],
    cfg: RuntimeConfig,
    previous_cache: dict[str, dict[str, Any]] | None = None,
    executor: Executor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    previous = previous_cache or {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in merge_records:
        grouped[(record["request_id"], record["channel"])].append(record)

    tasks: list[tuple[tuple[str, str], list[dict[str, Any]], str]] = []
    cache: dict[str, dict[str, Any]] = {}
    reused = 0
    for (request_id, channel), records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda record: record["item_id"])
        key = stable_hash(
            {
                "stage": "normalize",
                "request_id": request_id,
                "channel": channel,
                "normalization_fingerprint": cfg.normalization_fingerprint,
                "merge_keys": sorted(record["cache_key"] for record in ordered),
            }
        )
        if key in previous:
            cache[key] = previous[key]
            reused += 1
        else:
            tasks.append(((request_id, channel), ordered, key))

    def run(task: tuple[tuple[str, str], list[dict[str, Any]], str]) -> dict[str, Any]:
        (request_id, channel), records, _key = task
        return _normalization_task(request_id, channel, records, cfg)

    if executor is not None and len(tasks) > 1:
        results = list(executor.map(run, tasks))
    else:
        results = [run(task) for task in tasks]
    for result in results:
        cache[result["cache_key"]] = result
    channels = sorted(cache.values(), key=lambda result: (result["request_id"], result["channel"]))
    zero_denominators = [
        {
            "request_id": result["request_id"],
            "channel": result["channel"],
            "rule": result["denominator_rule"],
        }
        for result in channels
        if result["denominator_zero"]
    ]
    telemetry = {
        "computed": len(tasks),
        "reused": reused,
        "active_keys": sorted(cache),
        "denominator_zero": zero_denominators,
    }
    return channels, cache, telemetry


def build_fusion_tasks(
    norm_channels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries_by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for channel_result in norm_channels:
        for entry in channel_result["entries"]:
            entries_by_item[(channel_result["request_id"], entry["item_id"])].append(
                {
                    **entry,
                    "channel": channel_result["channel"],
                    "normalization_cache_key": channel_result["cache_key"],
                }
            )
    tasks = []
    for (request_id, item_id), entries in sorted(entries_by_item.items()):
        entries.sort(key=lambda entry: entry["channel"])
        tasks.append(
            {
                "request_id": request_id,
                "item_id": item_id,
                "entries": entries,
            }
        )
    return tasks


def fusion_cache_key(task: dict[str, Any], cfg: RuntimeConfig) -> str:
    return stable_hash(
        {
            "stage": "fusion",
            "request_id": task["request_id"],
            "item_id": task["item_id"],
            "weights_fingerprint": cfg.weights_fingerprint,
            "normalization_keys": [entry["normalization_cache_key"] for entry in task["entries"]],
        }
    )


def _compute_fusion(task: dict[str, Any], cfg: RuntimeConfig) -> dict[str, Any]:
    contributions = []
    numerator = 0
    for entry in sorted(task["entries"], key=lambda item: item["channel"]):
        channel = entry["channel"]
        weight = float(cfg.channel_weights.get(channel, 0.0))
        weight_fixed = fixed_from_float(weight, cfg.scale)
        term = weight_fixed * int(entry["normalized_fixed"])
        numerator += term
        contributions.append(
            {
                "channel": channel,
                "source_score": entry["source_score"],
                "normalized": entry["normalized"],
                "normalized_fixed": entry["normalized_fixed"],
                "normalized_fixed_string": entry["normalized_fixed_string"],
                "quantized_bucket": entry["quantized_bucket"],
                "weight": weight,
                "weight_fixed": weight_fixed,
                "term_fixed": term,
                "merge_cache_key": entry["merge_cache_key"],
                "normalization_cache_key": entry["normalization_cache_key"],
                "references": entry["references"],
                "unconfigured_channel": channel not in cfg.channel_weights,
            }
        )
    temperature_fixed = max(1, fixed_from_float(float(cfg.raw["fusion_temperature"]), cfg.scale))
    bias_fixed = fixed_from_float(float(cfg.raw["fusion_bias"]), cfg.scale)
    raw_fixed = round_div(numerator, temperature_fixed) + bias_fixed
    minimum = fixed_from_float(float(cfg.raw["fusion_score_min"]), cfg.scale)
    maximum = fixed_from_float(float(cfg.raw["fusion_score_max"]), cfg.scale)
    clipped_fixed = max(minimum, min(maximum, raw_fixed))
    edge_flags = []
    if raw_fixed == minimum or raw_fixed == maximum:
        edge_flags.append("fusion_score_exact_clip_boundary")
    elif raw_fixed < minimum or raw_fixed > maximum:
        edge_flags.append("fusion_score_clipped")
    shard_refs = sorted(
        {
            shard_id
            for contribution in contributions
            for shard_id in contribution["references"]["shards"]
        }
    )
    channels = sorted(contribution["channel"] for contribution in contributions)
    return {
        "cache_key": fusion_cache_key(task, cfg),
        "request_id": task["request_id"],
        "item_id": task["item_id"],
        "raw_fixed": raw_fixed,
        "raw_score": fixed_to_float(raw_fixed, cfg.places),
        "fusion_fixed": clipped_fixed,
        "fusion_score": fixed_to_float(clipped_fixed, cfg.places),
        "fusion_fixed_string": fixed_to_string(clipped_fixed, cfg.places),
        "contributions": contributions,
        "references": {"shards": shard_refs, "channels": channels},
        "edge_flags": edge_flags,
    }


def fuse_candidates(
    norm_channels: list[dict[str, Any]],
    cfg: RuntimeConfig,
    previous_cache: dict[str, dict[str, Any]] | None = None,
    pending_items: list[dict[str, str]] | None = None,
    executor: Executor | None = None,
    budget_override: int | None = None,
    round_number: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    previous = previous_cache or {}
    tasks = build_fusion_tasks(norm_channels)
    for task in tasks:
        task["cache_key"] = fusion_cache_key(task, cfg)
    pending_keys = {(entry["request_id"], entry["item_id"]) for entry in (pending_items or [])}
    tasks.sort(
        key=lambda task: (
            0 if (task["request_id"], task["item_id"]) in pending_keys else 1,
            task["request_id"],
            task["item_id"],
        )
    )
    budget = int(cfg.raw["recompute_budget_per_run"] if budget_override is None else budget_override)
    remaining = budget
    cache: dict[str, dict[str, Any]] = {}
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    to_compute: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    deductions: list[dict[str, Any]] = []
    reused = 0
    for task in tasks:
        key = task["cache_key"]
        if key in previous:
            record = previous[key]
            cache[key] = record
            completed[(record["request_id"], record["item_id"])] = record
            reused += 1
            continue
        if remaining <= 0:
            pending.append(
                {
                    "request_id": task["request_id"],
                    "item_id": task["item_id"],
                    "code": "E_RECOMPUTE_BUDGET_PENDING",
                    "cache_key": key,
                    "deferred": True,
                }
            )
            continue
        remaining -= 1
        deductions.append(
            {
                "request_id": task["request_id"],
                "item_id": task["item_id"],
                "points": 1,
                "remaining_after": remaining,
            }
        )
        to_compute.append(task)

    def run(task: dict[str, Any]) -> dict[str, Any]:
        return _compute_fusion(task, cfg)

    if executor is not None and len(to_compute) > 1:
        results = list(executor.map(run, to_compute))
    else:
        results = [run(task) for task in to_compute]
    for record in results:
        cache[record["cache_key"]] = record
        completed[(record["request_id"], record["item_id"])] = record

    records = list(completed.values())
    tie_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        tie_groups[(record["request_id"], record["fusion_fixed"])].append(record)
    boundary_samples = []
    for (request_id, fusion_fixed), group in sorted(tie_groups.items()):
        if len(group) > 1:
            items = sorted(record["item_id"] for record in group)
            for record in group:
                flag = "fusion_score_tie_item_id_order"
                if flag not in record["edge_flags"]:
                    record["edge_flags"].append(flag)
                boundary_samples.append(
                    {
                        "criterion": flag,
                        "request_id": request_id,
                        "item_id": record["item_id"],
                        "fusion_fixed": fusion_fixed,
                        "tied_items": items,
                        "decision": "item_id_lexicographic",
                    }
                )
    for record in records:
        for flag in record["edge_flags"]:
            if flag == "fusion_score_exact_clip_boundary":
                boundary_samples.append(
                    {
                        "criterion": flag,
                        "request_id": record["request_id"],
                        "item_id": record["item_id"],
                        "raw_fixed": record["raw_fixed"],
                        "fusion_fixed": record["fusion_fixed"],
                        "decision": "clip_to_configured_boundary",
                    }
                )
    records.sort(key=lambda record: (record["request_id"], -record["fusion_fixed"], record["item_id"]))
    pending.sort(key=lambda item: (item["request_id"], item["item_id"]))
    audit = {
        "round": round_number,
        "budget": budget,
        "consumed": len(deductions),
        "remaining": remaining,
        "deductions": deductions,
        "computed_count": len(results),
        "reused_count": reused,
        "pending_count": len(pending),
    }
    telemetry = {
        "computed": len(results),
        "reused": reused,
        "active_keys": sorted(cache),
        "pending": pending,
        "budget": audit,
        "boundary_samples": boundary_samples,
    }
    return records, cache, telemetry


def contribution_breakdown(merge_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in merge_records:
        grouped[record["request_id"]].append(record)
    result: dict[str, list[dict[str, Any]]] = {}
    for request_id, records in sorted(grouped.items()):
        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_channel[record["channel"]].append(record)
        result[request_id] = [
            {
                "channel": channel,
                "raw_candidate_count": len(channel_records),
                "occurrence_count": sum(record["occurrences_count"] for record in channel_records),
                "overridden_occurrence_count": sum(len(record["overridden"]) for record in channel_records),
                "items": sorted(record["item_id"] for record in channel_records),
                "shards": sorted(
                    {
                        shard_id
                        for record in channel_records
                        for shard_id in record["references"]["shards"]
                    }
                ),
            }
            for channel, channel_records in sorted(by_channel.items())
        ]
    return result
