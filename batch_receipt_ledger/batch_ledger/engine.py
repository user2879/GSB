from __future__ import annotations

import copy
from typing import Any

from .adjudication import aggregate_batches, adjudicate_requests
from .concurrency import map_batch_tasks
from .conversion import completeness_stats, conversion_roundtrip_audit
from .hashing import sha256_json
from .inputs import (
    assign_batch_workers,
    normalize_failure_order,
    validate_and_split_requests,
    validate_config,
    validate_workers,
)
from .model import MISSING, Failure
from .receipts import parse_and_validate_receipt
from .resources import build_invalidation_view, build_resource_entry
from .state import apply_command, clone_state, deep_default_config


def effective_config(state: dict[str, Any]) -> dict[str, Any]:
    config = deep_default_config()
    config.update(copy.deepcopy(state.get("config", {})))
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_json(config)


def _resource_cache_key(
    *,
    config: dict[str, Any],
    worker_versions: dict[str, int],
    batch_id: str,
    worker_id: str,
    items: list[Any],
) -> str:
    relevant_config = {
        key: config.get(key)
        for key in (
            "model_dim",
            "hidden_dim",
            "quant_bits",
            "budget_ms",
            "capacity_mac_per_ms",
            "fixed_point_scale",
        )
    }
    relevant_config_hash = sha256_json(relevant_config)
    receipt_fingerprint = sha256_json(
        sorted([receipt.to_dict() for receipt in items], key=lambda item: item["receipt_ref"])
    )
    group_has_param_version = any(
        getattr(receipt, "param_version", MISSING) is not MISSING for receipt in items
    )
    return sha256_json(
        {
            "resource_config_hash": relevant_config_hash,
            "worker_version": (
                worker_versions.get(worker_id) if group_has_param_version else "unversioned"
            ),
            "batch_id": batch_id,
            "worker_id": worker_id,
            "receipts": receipt_fingerprint,
        }
    )


def build_report(
    state: dict[str, Any],
    *,
    parallel: bool = False,
    resource_cache: dict[str, dict[str, Any]] | None = None,
    cache_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = effective_config(state)
    config_hash = config_fingerprint(config)
    failures: list[Failure] = []
    failures.extend(validate_config(config))

    raw_workers = list(state.get("workers", []))
    valid_workers, worker_failures = validate_workers(raw_workers, config)
    failures.extend(worker_failures)
    known_worker_ids = {worker["worker_id"] for worker in valid_workers}
    worker_versions = {worker["worker_id"]: int(worker["param_version"]) for worker in valid_workers}

    raw_requests = list(state.get("requests", []))
    valid_requests, request_failures, planned_batches = validate_and_split_requests(
        raw_requests, config.get("batch_size")
    )
    failures.extend(request_failures)
    assign_batch_workers(planned_batches, valid_workers)
    assignment = {
        batch["batch_id"]: batch["worker_id"]
        for batch in planned_batches
        if batch.get("worker_id")
    }

    receipts = []
    for index, raw_receipt in enumerate(state.get("receipts", [])):
        source = str(raw_receipt.get("_source", "state.receipts")) if isinstance(raw_receipt, dict) else "state.receipts"
        record_index = int(raw_receipt.get("_record_index", index)) if isinstance(raw_receipt, dict) else index
        receipt, failure = parse_and_validate_receipt(
            raw_receipt,
            known_worker_ids=known_worker_ids,
            source=source,
            record_index=record_index,
        )
        if failure is not None:
            failures.append(failure)
        if receipt is not None:
            receipts.append(receipt)

    receipts.sort(
        key=lambda receipt: (
            receipt.batch_id,
            receipt.request_id,
            receipt.source_version,
            receipt.receipt_ref,
        )
    )

    merge_rule = str(config.get("merge_rule", "coverage_then_version_v1"))
    decisions = adjudicate_requests(receipts, merge_rule=merge_rule)
    worker_aggregates, batch_stats = aggregate_batches(
        receipts, decisions, batch_worker_assignment=assignment
    )

    receipts_by_batch: dict[str, list[Any]] = {}
    for receipt in receipts:
        receipts_by_batch.setdefault(receipt.batch_id, []).append(receipt)

    def resource_worker(task: dict[str, Any]) -> dict[str, Any]:
        batch_id = task["batch_id"]
        batch_items = task["items"]
        groups: dict[str, list[Any]] = {}
        for receipt in batch_items:
            worker_id = receipt.worker_id
            if worker_id == "$missing":
                worker_id = assignment.get(batch_id, "$missing")
            groups.setdefault(str(worker_id), []).append(receipt)

        entries = []
        events = []
        for worker_id in sorted(groups):
            items = groups[worker_id]
            cache_key = _resource_cache_key(
                config=config,
                worker_versions=worker_versions,
                batch_id=batch_id,
                worker_id=worker_id,
                items=items,
            )
            if resource_cache is not None and cache_key in resource_cache:
                cached = resource_cache[cache_key]
                entries.append(cached["entry"])
                events.append({"type": "hit", "cache_key": cache_key, "resource_key": cached["entry"]["resource_key"]})
                continue
            entry = build_resource_entry(batch_id, worker_id, items, config)
            if resource_cache is not None:
                resource_cache[cache_key] = {"entry": entry}
            entries.append(entry)
            events.append({"type": "miss", "cache_key": cache_key, "resource_key": entry["resource_key"]})
        return {"entries": entries, "cache_events": events}

    task_envelopes = map_batch_tasks(
        stage="resource",
        tasks=[
            {"batch_id": batch_id, "items": items}
            for batch_id, items in sorted(receipts_by_batch.items())
        ],
        worker=resource_worker,
        parallel=parallel,
    )
    resource_entries: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    for envelope in task_envelopes:
        value = envelope["value"]
        resource_entries.extend(value["entries"])
        cache_events.extend(value["cache_events"])

    if cache_stats is not None:
        hits = [event for event in cache_events if event["type"] == "hit"]
        misses = [event for event in cache_events if event["type"] == "miss"]
        cache_stats["hits"] = len(hits)
        cache_stats["misses"] = len(misses)
        cache_stats["hit_resource_keys"] = [event["resource_key"] for event in hits]
        cache_stats["miss_resource_keys"] = [event["resource_key"] for event in misses]

    baseline = state.get("baseline_worker_versions") or {
        worker["worker_id"]: int(worker.get("param_version", 0))
        for worker in valid_workers
    }
    invalidation = build_invalidation_view(
        workers=valid_workers,
        baseline_worker_versions={str(k): int(v) for k, v in baseline.items()},
        resource_entries=resource_entries,
        decisions=decisions,
    )

    failure_dicts = normalize_failure_order(failures)
    failure_counts: dict[str, int] = {}
    for failure in failure_dicts:
        failure_counts[failure["code"]] = failure_counts.get(failure["code"], 0) + 1

    report = {
        "schema": "batch-receipt-ledger-report-v1",
        "global_version": int(state.get("global_version", 1)),
        "config_hash": config_hash,
        "merge_rule": merge_rule,
        "summary": {
            "worker_count": len(valid_workers),
            "valid_request_count": len(valid_requests),
            "planned_batch_count": len(planned_batches),
            "receipt_count": len(receipts),
            "decision_count": len(decisions),
            "duplicate_request_count": sum(1 for item in decisions if item["is_duplicate"]),
            "batch_stat_count": len(batch_stats),
            "resource_entry_count": len(resource_entries),
            "failure_count": len(failure_dicts),
            "failure_code_counts": dict(sorted(failure_counts.items())),
        },
        "workers": sorted(valid_workers, key=lambda item: item["worker_id"]),
        "planned_batches": planned_batches,
        "failures": failure_dicts,
        "receipts": [receipt.to_dict() for receipt in receipts],
        "decisions": decisions,
        "worker_aggregates": worker_aggregates,
        "batch_stats": batch_stats,
        "resource_entries": resource_entries,
        "invalidation": invalidation,
        "normalization": completeness_stats(receipts),
        "conversion_audit": conversion_roundtrip_audit(receipts),
    }
    report["report_hash"] = sha256_json(_without_report_hash(report))
    return report


def _without_report_hash(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "report_hash"}


def reports_field_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _without_report_hash(left) == _without_report_hash(right) and left.get("report_hash") == right.get("report_hash")


class LedgerEngine:
    """Warm-cache incremental engine."""

    def __init__(self, state: dict[str, Any]):
        self.state = state
        self.resource_cache: dict[str, dict[str, Any]] = {}
        self.current_report: dict[str, Any] | None = None
        self.last_cache_stats: dict[str, Any] = {}
        self.command_results: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []

    def run(self, *, parallel: bool = True) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        self.current_report = build_report(
            self.state,
            parallel=parallel,
            resource_cache=self.resource_cache,
            cache_stats=stats,
        )
        self.last_cache_stats = stats
        self.reports.append(copy.deepcopy(self.current_report))
        return self.current_report

    def apply(self, command: dict[str, Any], *, parallel: bool = True) -> dict[str, Any]:
        result = apply_command(self.state, command)
        self.command_results.append(result)
        self.run(parallel=parallel)
        return result

    def full_rebuild(self, *, parallel: bool = False) -> dict[str, Any]:
        return build_report(clone_state(self.state), parallel=parallel)

    def incremental_matches_full(self, *, parallel: bool = True) -> bool:
        if self.current_report is None:
            self.run(parallel=parallel)
        incremental = self.current_report
        full = self.full_rebuild(parallel=False)
        return reports_field_equal(incremental, full)
