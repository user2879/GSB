"""End-to-end staged pipeline for deterministic batch-inference ledgers."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import ConfigLoadResult, known_parameter_versions, injection_map, load_config, receipt_paths
from .errors import E_FRAME_INVALID, Failure, sort_failures
from .jsonutil import MISSING, atomic_write_json, atomic_write_text, decode_missing, hash_json, is_missing, pretty_dumps
from .models import (
    BatchNormalized,
    BatchResource,
    ConfigBundle,
    OwnershipDecision,
    RankConflict,
    Receipt,
    ScheduleResult,
)
from .normalization import (
    batch_input_hash,
    normalize_batch,
    receipt_from_ledger_dict,
    resolve_ownership,
)
from .receipts import ParseResult, parse_receipt_paths
from .resources import (
    canonical_dependency_plan,
    compute_batch_resource,
    compute_invalidation,
    model_resource_hash,
    quality_summary,
)
from .scheduler import BatchPlan, schedule_batches
from .snapshot import SnapshotStore, restore_decision


class PipelineConfigError(RuntimeError):
    def __init__(self, failures: Sequence[Failure]):
        self.failures = tuple(failures)
        message = "; ".join(f"{item.code}: {item.message}" for item in failures[:5])
        super().__init__(message or "invalid configuration")


@dataclass(frozen=True)
class PipelineResult:
    ledger: Mapping[str, Any]
    report: Mapping[str, Any]
    snapshot_version: int | None
    snapshot_path: str | None
    ledger_path: str
    report_json_path: str
    report_markdown_path: str


@dataclass(frozen=True)
class _IndexedTask:
    key: str
    payload: Any


def _run_tasks(
    func: Callable[[Any], Any],
    tasks: Sequence[_IndexedTask],
    jobs: int,
) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    ordered = sorted(tasks, key=lambda item: item.key)
    results: dict[str, Any] = {}
    if jobs == 1 or len(ordered) <= 1:
        for task in ordered:
            results[task.key] = func(task.payload)
        return results
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="mmledger") as executor:
        future_to_key = {executor.submit(func, task.payload): task.key for task in ordered}
        for future, key in future_to_key.items():
            results[key] = future.result()
    return {key: results[key] for key in sorted(results)}


def _batch_receipt_map(receipts: Sequence[Receipt]) -> dict[str, list[Receipt]]:
    grouped: dict[str, list[Receipt]] = {}
    for receipt in receipts:
        grouped.setdefault(receipt.batch_id, []).append(receipt)
    return {key: sorted(value, key=lambda receipt: receipt.log_key) for key, value in grouped.items()}


def _serialize_normalization(batch: BatchNormalized) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "worker_id": batch.worker_id,
        "param_version": batch.param_version,
        "receipts": [receipt.as_ledger_dict() for receipt in batch.receipts],
        "candidates": [receipt.as_ledger_dict() for receipt in batch.candidates],
        "conflicts": [conflict.as_dict() for conflict in batch.conflicts],
        "input_hash": batch.input_hash,
    }


def _deserialize_normalization(raw: Mapping[str, Any]) -> BatchNormalized:
    receipts = tuple(receipt_from_ledger_dict(item) for item in raw.get("receipts", []))
    candidates = tuple(receipt_from_ledger_dict(item) for item in raw.get("candidates", []))
    conflicts = []
    for item in raw.get("conflicts", []):
        conflicts.append(
            RankConflict(
                batch_id=str(item["batch_id"]),
                request_id=str(item["request_id"]),
                worker_a=str(item["worker_a"]),
                worker_b=str(item["worker_b"]),
                rank_a=int(item["rank_a"]),
                rank_b=int(item["rank_b"]),
                winner_worker=str(item["winner_worker"]),
            )
        )
    return BatchNormalized(
        batch_id=str(raw["batch_id"]),
        worker_id=str(raw["worker_id"]),
        param_version=decode_missing(raw.get("param_version", MISSING)),
        receipts=receipts,
        candidates=candidates,
        conflicts=tuple(conflicts),
        input_hash=str(raw["input_hash"]),
        cache_hit=True,
        task_key=f"normalize|{raw['batch_id']}|{raw['worker_id']}",
    )


def _batch_resource_from_dict(raw: Mapping[str, Any]) -> BatchResource:
    return BatchResource(
        batch_id=str(raw["batch_id"]),
        worker_id=str(raw["worker_id"]),
        parameter_version=decode_missing(raw.get("parameter_version", MISSING)),
        request_count=int(raw["request_count"]),
        duration_ms=decode_missing(raw.get("duration_ms", MISSING)),
        duration_observed_count=int(raw["duration_observed_count"]),
        duration_missing_count=int(raw["duration_missing_count"]),
        macs=int(raw["macs"]),
        quantized_macs=int(raw.get("quantized_macs", 0)),
        quantized_parameter_units=int(raw["quantized_parameter_units"]),
        utilization_ppm=decode_missing(raw.get("utilization_ppm", MISSING)),
        cache_hit=True,
        task_key=f"resources|{raw['batch_id']}|{raw['worker_id']}",
    )


def _decision_hash_for_batch(
    batch_id: str,
    decisions: Sequence[OwnershipDecision],
) -> str:
    selected = [decision.as_dict() for decision in decisions if decision.batch_id == batch_id]
    return hash_json(selected)


def _canonical_batch_dict(batch: BatchNormalized) -> dict[str, Any]:
    payload = batch.as_dict()
    payload.pop("cache_hit", None)
    payload.pop("task_key", None)
    return payload


def _canonical_resource_dict(resource: BatchResource) -> dict[str, Any]:
    payload = resource.as_dict()
    payload.pop("cache_hit", None)
    payload.pop("task_key", None)
    return payload


def _dependency_invalidation(
    bundle: ConfigBundle,
    batches: Mapping[str, Sequence[Any]],
    receipts_by_batch: Mapping[str, Sequence[Receipt]],
) -> dict[str, Any]:
    worker_by_id = bundle.workers_by_id
    rows: list[dict[str, Any]] = []
    stale_batches: list[str] = []
    stale_requests: list[str] = []
    for batch_id in sorted(batches):
        requests = batches[batch_id]
        worker_id = requests[0].worker_id
        current_version = worker_by_id[worker_id].parameter_version
        versions = sorted(
            {
                int(receipt.parameter_version)
                for receipt in receipts_by_batch.get(batch_id, ())
                if not is_missing(receipt.parameter_version)
            }
        )
        stale = any(version != current_version for version in versions)
        row = {
            "batch_id": batch_id,
            "worker_id": worker_id,
            "worker_parameter_version": current_version,
            "receipt_parameter_versions": versions,
            "request_ids": sorted(request.request_id for request in requests),
            "stale": stale,
            "boundary": "worker_parameter_version -> batch -> request",
        }
        rows.append(row)
        if stale:
            stale_batches.append(batch_id)
            stale_requests.extend(sorted(request.request_id for request in requests))
    return {
        "worker_parameter_versions": {
            worker.worker_id: worker.parameter_version for worker in bundle.workers
        },
        "batches": rows,
        "stale_batches": sorted(set(stale_batches)),
        "stale_requests": sorted(set(stale_requests)),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    counts = report.get("counts", {})
    actual = report.get("actual_invalidation", {})
    quality = report.get("quality", {})
    lines = [
        "# Multimodal Batch Ledger Diagnostic Report",
        "",
        f"- State hash: `{report.get('state_hash', '')}`",
        f"- Receipts accepted: {counts.get('receipts_ok', 0)}",
        f"- Failures: {counts.get('failures', 0)}",
        f"- Authoritative requests: {counts.get('authoritative_requests', 0)}",
        f"- Snapshot version: {report.get('snapshot', {}).get('global_version', 'not committed')}",
        f"- Restore decision: `{report.get('snapshot', {}).get('restore_decision', {}).get('decision', '')}`",
        "",
        "## Invalidation boundary",
        "",
        f"- Parameter-version changes observed: {len(actual.get('changed_workers', []))}",
        f"- Normalization computed batches: {len(actual.get('normalization_computed_batches', []))}",
        f"- Normalization reused batches: {len(actual.get('normalization_reused_batches', []))}",
        f"- Resource computed batches: {len(actual.get('resource_computed_batches', []))}",
        f"- Resource reused batches: {len(actual.get('resource_reused_batches', []))}",
        "",
        "## Multimodal quality summary",
        "",
    ]
    for kind, kind_summary in quality.get("kinds", {}).items():
        lines.append(
            f"- `{kind}`: count={kind_summary.get('count', 0)}, "
            f"duration_observed={kind_summary.get('observed_duration_count', 0)}, "
            f"duration_missing={kind_summary.get('missing_duration_count', 0)}, "
            f"sum_ms={kind_summary.get('duration_sum_ms', 0)}"
        )
    lines.extend(["", "## Failures", ""])
    failures = report.get("failures", [])
    if not failures:
        lines.append("- None")
    else:
        for failure in failures:
            injection = f" injection={failure['injection_id']}" if failure.get("injection_id") else ""
            lines.append(
                f"- `{failure['code']}` {failure['source']}#{failure.get('ordinal', 0)}: "
                f"{failure['message']}{injection}"
            )
    lines.extend(["", "## Fault-injection hits", ""])
    hits = report.get("injection_hits", [])
    if not hits:
        lines.append("- Not active for this run")
    else:
        for hit in hits:
            lines.append(
                f"- `{hit['injection_id']}` hit={hit['hit']} code={hit.get('code')} source={hit.get('source')}"
            )
    lines.append("")
    return "\n".join(lines)


def _default_output_paths(root: Path, *, fault_drill: bool) -> tuple[Path, Path, Path]:
    output_dir = root / "reports" / ("fault_drill" if fault_drill else "")
    return (
        output_dir / "ledger.json",
        output_dir / "diagnostic_report.json",
        output_dir / "diagnostic_report.md",
    )


def run_pipeline(
    config_path: str | Path,
    *,
    jobs: int = 4,
    full_rebuild: bool = False,
    fault_drill: bool = False,
    normalization_budget: int | None = None,
    worker_quota_points: int | None = None,
    commit_snapshot: bool = True,
) -> PipelineResult:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    config_result: ConfigLoadResult = load_config(config_path)
    if config_result.bundle is None:
        raise PipelineConfigError(config_result.failures)
    bundle = config_result.bundle
    root = Path(bundle.root)
    config_dir = Path(bundle.config_path).parent
    store = SnapshotStore(root / "snapshots")
    current = None if full_rebuild or fault_drill else store.current()
    previous_cache: Mapping[str, Any] = current.cache if current is not None else {}
    injection_config_path = config_dir / bundle.fault_injections_path
    injection_lookup = injection_map(injection_config_path) if fault_drill else {}

    receipt_files = receipt_paths(bundle, fault=fault_drill)
    parse_result = parse_receipt_paths(
        receipt_files,
        base_dir=config_dir,
        known_versions=known_parameter_versions(bundle),
        injections=injection_lookup,
    )
    restore = restore_decision(
        current,
        config_hash=bundle.config_hash,
        receipt_fingerprints=parse_result.file_fingerprints,
    )

    all_batches = bundle.batches
    receipts_by_batch = _batch_receipt_map(parse_result.receipts)
    failures: list[Failure] = list(parse_result.failures)
    known_batch_ids = set(all_batches)
    for batch_id in sorted(receipts_by_batch):
        if batch_id not in known_batch_ids:
            for receipt in receipts_by_batch[batch_id]:
                failures.append(
                    Failure(
                        E_FRAME_INVALID,
                        f"receipt references unknown batch {batch_id!r}",
                        receipt.source_file,
                        receipt.source_index,
                        item=receipt.uid,
                    )
                )

    placeholders: dict[str, BatchNormalized] = {}
    for batch_id in sorted(all_batches):
        requests = all_batches[batch_id]
        batch_receipts = receipts_by_batch.get(batch_id, [])
        placeholders[batch_id] = BatchNormalized(
            batch_id=batch_id,
            worker_id=requests[0].worker_id,
            param_version=bundle.workers_by_id[requests[0].worker_id].parameter_version,
            receipts=tuple(batch_receipts),
            candidates=(),
            conflicts=(),
            input_hash=batch_input_hash(requests, batch_receipts),
            task_key=f"normalize|{batch_id}|{requests[0].worker_id}",
        )

    model_hash = model_resource_hash(bundle.model, bundle.limits.batch_size)
    invalidation = compute_invalidation(
        workers=bundle.workers,
        batches=tuple(placeholders.values()),
        previous_cache=previous_cache,
        merge_rules_hash=bundle.merge_rules_hash,
        model_hash=model_hash,
        batch_size=bundle.limits.batch_size,
    )
    quotas = {
        worker.worker_id: (
            int(worker_quota_points)
            if worker_quota_points is not None
            else int(worker.quota_points if worker.quota_points is not None else bundle.limits.worker_quota_points)
        )
        for worker in bundle.workers
    }
    effective_budget = (
        int(normalization_budget)
        if normalization_budget is not None
        else bundle.limits.normalization_budget
    )
    previous_pending = current.manifest.get("pending_batches", []) if current is not None else []
    if not isinstance(previous_pending, list):
        previous_pending = []
    schedule = schedule_batches(
        (
            BatchPlan(
                batch_id=batch_id,
                worker_id=requests[0].worker_id,
                request_ids=tuple(sorted(request.request_id for request in requests)),
            )
            for batch_id, requests in sorted(all_batches.items())
        ),
        budget=effective_budget,
        worker_quotas=quotas,
        previous_pending=[str(item) for item in previous_pending],
    )
    failures.extend(schedule.failures)
    scheduled_ids = {ticket.batch_id for ticket in schedule.tickets}
    previous_batches = previous_cache.get("batches", {}) if isinstance(previous_cache, dict) else {}
    if not isinstance(previous_batches, dict):
        previous_batches = {}

    normalization_tasks: list[_IndexedTask] = []
    normalized_by_id: dict[str, BatchNormalized] = {}
    normalization_computed: list[str] = []
    normalization_reused: list[str] = []
    for batch_id in sorted(scheduled_ids):
        placeholder = placeholders[batch_id]
        previous_entry = previous_batches.get(batch_id)
        cache_entry = previous_entry.get("normalization") if isinstance(previous_entry, dict) else None
        invalidated = batch_id in set(invalidation["normalization_invalidated_batches"])
        if (
            not invalidated
            and isinstance(previous_entry, dict)
            and isinstance(cache_entry, dict)
            and previous_entry.get("input_hash") == placeholder.input_hash
            and previous_entry.get("worker_id") == placeholder.worker_id
            and previous_entry.get("worker_parameter_version")
            == bundle.workers_by_id[placeholder.worker_id].parameter_version
        ):
            try:
                normalized_by_id[batch_id] = _deserialize_normalization(cache_entry)
                normalization_reused.append(batch_id)
                continue
            except (KeyError, TypeError, ValueError):
                pass
        normalization_tasks.append(
            _IndexedTask(
                key=f"normalize|{batch_id}|{placeholder.worker_id}",
                payload=(batch_id, all_batches[batch_id], placeholder.receipts, placeholder.input_hash),
            )
        )

    def normalize_task(payload: Any) -> BatchNormalized:
        batch_id, requests, receipts, input_hash = payload
        batch, task_failures = normalize_batch(
            batch_id,
            requests,
            receipts,
            merge_rules=bundle.merge_rules,
            input_hash=input_hash,
            cache_hit=False,
        )
        return batch, tuple(task_failures)

    task_results = _run_tasks(normalize_task, normalization_tasks, jobs)
    for key in sorted(task_results):
        batch, task_failures = task_results[key]
        normalized_by_id[batch.batch_id] = batch
        normalization_computed.append(batch.batch_id)
        failures.extend(task_failures)

    ownership = resolve_ownership(
        tuple(normalized_by_id[batch_id] for batch_id in sorted(normalized_by_id)),
        bundle.requests,
        bundle.merge_rules,
    )
    failures.extend(ownership.failures)
    processed_batches = tuple(ownership.batches)

    decision_hash_by_batch = {
        batch.batch_id: _decision_hash_for_batch(batch.batch_id, ownership.decisions)
        for batch in processed_batches
    }
    resource_inputs = {batch.batch_id: batch for batch in processed_batches}
    resource_tasks: list[_IndexedTask] = []
    resources_by_id: dict[str, BatchResource] = {}
    resource_computed: list[str] = []
    resource_reused: list[str] = []
    resource_invalidated = set(invalidation["resource_invalidated_batches"])
    for batch_id in sorted(resource_inputs):
        batch = resource_inputs[batch_id]
        previous_entry = previous_batches.get(batch_id)
        cached_resource = previous_entry.get("resources") if isinstance(previous_entry, dict) else None
        cached_decision_hash = previous_entry.get("decision_hash") if isinstance(previous_entry, dict) else None
        if (
            batch_id not in resource_invalidated
            and isinstance(cached_resource, dict)
            and cached_decision_hash == decision_hash_by_batch[batch_id]
        ):
            try:
                resources_by_id[batch_id] = _batch_resource_from_dict(cached_resource)
                resource_reused.append(batch_id)
                continue
            except (KeyError, TypeError, ValueError):
                pass
        resource_tasks.append(
            _IndexedTask(
                key=f"resources|{batch_id}|{batch.worker_id}",
                payload=(batch, ownership.decisions),
            )
        )

    def resource_task(payload: Any) -> BatchResource:
        batch, decisions = payload
        return compute_batch_resource(
            batch,
            decisions,
            model=bundle.model,
            batch_size=bundle.limits.batch_size,
            cache_hit=False,
        )

    resource_results = _run_tasks(resource_task, resource_tasks, jobs)
    for key in sorted(resource_results):
        resource = resource_results[key]
        resources_by_id[resource.batch_id] = resource
        resource_computed.append(resource.batch_id)

    resources = tuple(resources_by_id[batch_id] for batch_id in sorted(resources_by_id))
    quality = quality_summary(ownership.decisions, resources)
    dependency_invalidation = _dependency_invalidation(bundle, all_batches, receipts_by_batch)

    all_failures = sort_failures(failures)
    canonical_plan = canonical_dependency_plan(
        processed_batches,
        invalidation={
            "reasons_by_batch": {
                batch_id: ["parameter_version_boundary_stale"]
                for batch_id in dependency_invalidation["stale_batches"]
            }
        },
    )
    ledger = {
        "schema_version": "mmledger-ledger-v1",
        "state_hash": bundle.state_hash,
        "config_hash": bundle.config_hash,
        "requests_hash": bundle.requests_hash,
        "merge_rules_hash": bundle.merge_rules_hash,
        "counts": {
            "receipts_ok": len(parse_result.receipts),
            "authoritative_requests": len(ownership.decisions),
            "processed_batches": len(processed_batches),
            "resource_batches": len(resources),
            "failures": len(all_failures),
            "pending_batches": len(schedule.pending_batches),
            "quota_blocked_batches": len(schedule.quota_blocked_batches),
        },
        "receipt_versions": dict(sorted(parse_result.file_versions.items())),
        "failures": [failure.to_dict() for failure in all_failures],
        "ledger": [decision.as_dict() for decision in ownership.decisions],
        "batches": [_canonical_batch_dict(batch) for batch in processed_batches],
        "resources": [_canonical_resource_dict(resource) for resource in resources],
        "quality": quality,
        "dependency_invalidation": dependency_invalidation,
        "invalidation_reuse_plan": canonical_plan,
        "cache_plan": canonical_plan,
        "scheduler": {
            "budget": effective_budget,
            "consumed_budget": schedule.consumed_budget,
            "pending_queue": list(schedule.pending_batches),
            "quota_blocked": list(schedule.quota_blocked_batches),
            "worker_quotas": dict(sorted(quotas.items())),
            "tickets": [ticket.as_dict() for ticket in schedule.tickets],
            "trace": [dict(item) for item in schedule.trace],
        },
    }
    ledger["ledger_digest"] = hash_json({key: value for key, value in ledger.items() if key != "ledger_digest"})

    actual_invalidation = {
        **invalidation,
        "normalization_computed_batches": sorted(normalization_computed),
        "normalization_reused_batches": sorted(normalization_reused),
        "resource_computed_batches": sorted(resource_computed),
        "resource_reused_batches": sorted(resource_reused),
        "normalization_task_count": len(normalization_tasks),
        "resource_task_count": len(resource_tasks),
    }
    completed_workers = sorted(
        {
            bundle.batches[batch_id][0].worker_id
            for batch_id in normalized_by_id
        }
    )
    ledger_path, report_json_path, report_markdown_path = _default_output_paths(
        root, fault_drill=fault_drill
    )
    report: dict[str, Any] = {
        "schema_version": "mmledger-report-v1",
        "state_hash": bundle.state_hash,
        "config_path": bundle.config_path,
        "jobs": jobs,
        "snapshot": {
            "restore_decision": restore,
            "global_version": None,
            "generation": None,
        },
        "counts": dict(ledger["counts"]),
        "receipt_versions": dict(ledger["receipt_versions"]),
        "injection_hits": [dict(item) for item in parse_result.injection_hits],
        "failures": [failure.to_dict() for failure in all_failures],
        "schedule": {
            "budget": effective_budget,
            "consumed_budget": schedule.consumed_budget,
            "pending_batches": list(schedule.pending_batches),
            "quota_blocked_batches": list(schedule.quota_blocked_batches),
            "tickets": [ticket.as_dict() for ticket in schedule.tickets],
            "trace": [dict(item) for item in schedule.trace],
        },
        "actual_invalidation": actual_invalidation,
        "dependency_invalidation": dependency_invalidation,
        "quality": quality,
        "resources": [_canonical_resource_dict(resource) for resource in resources],
        "ledger_digest": ledger["ledger_digest"],
    }

    snapshot_version: int | None = None
    snapshot_path: str | None = None
    if commit_snapshot:
        staging = store.staging()
        try:
            store.write_stage(staging, "parse", parse_result.as_dict())
            store.write_stage(
                staging,
                "ownership",
                {
                    "schedule": schedule.as_dict(),
                    "batches": [_canonical_batch_dict(batch) for batch in processed_batches],
                    "decisions": [decision.as_dict() for decision in ownership.decisions],
                    "failures": [failure.to_dict() for failure in sort_failures(failures)],
                },
            )
            store.write_stage(
                staging,
                "resources",
                {
                    "resources": [_canonical_resource_dict(resource) for resource in resources],
                    "quality": quality,
                    "actual_invalidation": actual_invalidation,
                    "dependency_invalidation": dependency_invalidation,
                },
            )
            store.write_stage(staging, "ledger", ledger)
            cache_payload = {
                "state_hash": bundle.state_hash,
                "worker_parameter_versions": {
                    worker.worker_id: worker.parameter_version for worker in bundle.workers
                },
                "merge_rules_hash": bundle.merge_rules_hash,
                "model_hash": model_hash,
                "batch_size": bundle.limits.batch_size,
                "pending_batches": list(schedule.pending_batches),
                "batches": {
                    batch_id: {
                        "worker_id": normalized_by_id[batch_id].worker_id,
                        "worker_parameter_version": bundle.workers_by_id[
                            normalized_by_id[batch_id].worker_id
                        ].parameter_version,
                        "input_hash": normalized_by_id[batch_id].input_hash,
                        "normalization": _serialize_normalization(normalized_by_id[batch_id]),
                        "decision_hash": decision_hash_by_batch.get(batch_id, ""),
                        "resources": resources_by_id[batch_id].as_dict() if batch_id in resources_by_id else None,
                    }
                    for batch_id in sorted(normalized_by_id)
                },
            }
            store.write_support_file(staging, "cache.json", cache_payload)
            version = store.next_version()
            manifest = {
                "config_hash": bundle.config_hash,
                "requests_hash": bundle.requests_hash,
                "merge_rules_hash": bundle.merge_rules_hash,
                "state_hash": bundle.state_hash,
                "receipt_fingerprints": dict(sorted(parse_result.file_fingerprints.items())),
                "completed_workers": completed_workers,
                "consumed_budget": schedule.consumed_budget,
                "pending_batches": list(schedule.pending_batches),
                "quota_blocked_batches": list(schedule.quota_blocked_batches),
                "fault_injection_hits": [dict(item) for item in parse_result.injection_hits],
                "restore_decision": restore,
                "actual_invalidation": actual_invalidation,
            }
            committed = store.commit(staging, global_version=version, manifest=manifest)
            snapshot_version = committed.global_version
            snapshot_path = str(committed.generation_path)
            report["snapshot"]["global_version"] = snapshot_version
            report["snapshot"]["generation"] = committed.generation_name
        except Exception:
            store.discard(staging)
            raise

    atomic_write_json(Path(ledger_path), ledger)
    atomic_write_json(Path(report_json_path), report)
    atomic_write_text(Path(report_markdown_path), _report_markdown(report))
    return PipelineResult(
        ledger=ledger,
        report=report,
        snapshot_version=snapshot_version,
        snapshot_path=snapshot_path,
        ledger_path=str(ledger_path),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
    )
