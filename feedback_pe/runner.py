from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import codes
from .engine import Workspace, active_schemes, build_task_metadata, diff_input_basis
from .hashing import meter_sample_points, point_cost
from .samples import business_code_counts


def plan_tasks(config: dict[str, Any],
               samples_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    desired: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(samples_by_id):
        sample = samples_by_id[sample_id]
        for scheme in active_schemes(config):
            desired.update(build_task_metadata(sample, scheme, config))
    return dict(sorted(desired.items()))


def _execute_ready(ready: list[str], samples_by_id: dict[str, dict[str, Any]],
                   config: dict[str, Any], needed_map: dict[tuple[str, str], list[str]],
                   reuse_payloads: dict[tuple[str, str, str], dict[str, Any]],
                   desired: dict[str, dict[str, Any]],
                   parallel: bool = True) -> dict[str, dict[str, Any]]:
    def run_sample(sample_id: str) -> dict[tuple[str, str], dict[str, Any]]:
        output = {}
        sample = samples_by_id[sample_id]
        for scheme in active_schemes(config):
            needed = needed_map.get((sample_id, scheme["name"]), [])
            if not needed:
                continue
            cached = {}
            for stage in ("position", "score", "stats", "error"):
                sort_key = f"{sample_id}::{scheme['name']}::{stage}"
                metadata = desired[sort_key]
                payload = reuse_payloads.get((sample_id, scheme["name"], stage))
                if payload is not None:
                    cached[stage] = payload
            from .engine import compute_sample_scheme
            output[(sample_id, scheme["name"])] = compute_sample_scheme(
                sample, scheme, config, needed, cached,
            )
        return output

    merged: dict[str, dict[str, Any]] = {}
    workers = max(1, int(config.get("workers", 1)))
    if not parallel or workers == 1 or len(ready) <= 1:
        for sample_id in ready:
            merged.update(run_sample(sample_id))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(run_sample, ready):
                merged.update(result)
    return merged


def run_batch(workspace: Workspace, config: dict[str, Any],
              samples: list[dict[str, Any]], failures: list[dict[str, Any]],
              batch_budget: int | None = None,
              sample_quota: int | None = None,
              load_payload: dict[str, Any] | None = None,
              parallel: bool = True) -> dict[str, Any]:
    samples_by_id = {item["id"]: item for item in sorted(samples, key=lambda x: x["id"])}
    desired = plan_tasks(config, samples_by_id)

    current_invalidations: list[dict[str, Any]] = []
    for sort_key, old_metadata in sorted(workspace.task_index.items()):
        new_metadata = desired.get(sort_key)
        if new_metadata is None:
            record = {
                "sort_key": sort_key,
                "sample_id": old_metadata.get("sample_id"),
                "scheme": old_metadata.get("scheme"),
                "stage": old_metadata.get("stage"),
                "old_cache_key": old_metadata.get("cache_key"),
                "new_cache_key": None,
                "reason": "task_removed",
                "basis": ["task_not_desired"],
            }
            current_invalidations.append(record)
        elif old_metadata.get("cache_key") != new_metadata["cache_key"]:
            basis = diff_input_basis(
                old_metadata.get("inputs", {}), new_metadata["inputs"],
            )
            record = {
                "sort_key": sort_key,
                "sample_id": new_metadata["sample_id"],
                "scheme": new_metadata["scheme"],
                "stage": new_metadata["stage"],
                "old_cache_key": old_metadata.get("cache_key"),
                "new_cache_key": new_metadata["cache_key"],
                "reason": "fingerprint_changed",
                "basis": basis,
            }
            current_invalidations.append(record)
    workspace.invalidation_records.extend(current_invalidations)
    workspace.ledger_events.extend(
        {"type": "invalidation", **record} for record in current_invalidations
    )

    reuse: list[dict[str, Any]] = []
    needed_map: dict[tuple[str, str], list[str]] = {}
    reuse_payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    recompute_tasks: list[dict[str, Any]] = []

    for sort_key, metadata in desired.items():
        stage = metadata["stage"]
        sample_id = metadata["sample_id"]
        scheme = metadata["scheme"]
        if metadata["cache_key"] in workspace.stores[stage]:
            reuse.append({
                "sort_key": sort_key,
                "sort_hash": metadata["sort_hash"],
                "sample_id": sample_id,
                "scheme": scheme,
                "stage": stage,
                "cache_key": metadata["cache_key"],
                "status": "reuse",
            })
            reuse_payloads[(sample_id, scheme, stage)] = workspace.stores[stage][metadata["cache_key"]]
            workspace.reuse_records.append({
                "sort_key": sort_key,
                "cache_key": metadata["cache_key"],
                "status": "hit",
            })
        else:
            recompute_tasks.append({**metadata, "status": "recompute"})
            needed_map.setdefault((sample_id, scheme), []).append(stage)

    invalid_sample_ids = sorted({
        metadata["sample_id"] for metadata in recompute_tasks
    })
    previous_pending = [
        item for item in workspace.scheduler.get("pending", [])
        if item in samples_by_id and item in invalid_sample_ids
    ]
    candidate_order = list(previous_pending)
    for sample_id in invalid_sample_ids:
        if sample_id not in candidate_order:
            candidate_order.append(sample_id)

    budget_total = int(batch_budget or config["batch_point_budget"])
    quota = int(sample_quota or config["sample_point_quota"])
    remaining = budget_total
    spent = 0
    ready: list[str] = []
    point_costs: dict[str, int] = {}
    pending_entries: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    scheduler_code_counts: dict[str, int] = {}

    for position, sample_id in enumerate(candidate_order):
        sample = samples_by_id[sample_id]
        cost = point_cost(sample)
        point_costs[sample_id] = cost
        meter = meter_sample_points(sample, quota)
        if meter["exceeded"]:
            entry = {
                "code": codes.E_POINT_BUDGET_PENDING,
                "sample_id": sample_id,
                "cost": cost,
                "metered_points_at_abort": meter["exceeded_at"],
                "sample_point_quota": quota,
                "batch_budget_remaining": remaining,
                "reason": "sample_point_quota_exceeded",
                "field": "key_positions",
            }
            pending_entries.append(entry)
            pending_ids.append(sample_id)
            scheduler_code_counts[codes.E_POINT_BUDGET_PENDING] = (
                scheduler_code_counts.get(codes.E_POINT_BUDGET_PENDING, 0) + 1
            )
            continue
        if cost > remaining:
            reason = ("batch_budget_exhausted" if position == 0
                      or remaining < budget_total else "batch_budget_exhausted")
            for tail_id in candidate_order[position:]:
                if tail_id not in pending_ids:
                    tail_cost = point_cost(samples_by_id[tail_id])
                    pending_entries.append({
                        "code": codes.E_POINT_BUDGET_PENDING,
                        "sample_id": tail_id,
                        "cost": tail_cost,
                        "sample_point_quota": quota,
                        "batch_budget_remaining": remaining,
                        "reason": reason if tail_id == sample_id else "queued_behind_budget_pending",
                        "field": "batch_point_budget",
                    })
                    pending_ids.append(tail_id)
                    scheduler_code_counts[codes.E_POINT_BUDGET_PENDING] = (
                        scheduler_code_counts.get(codes.E_POINT_BUDGET_PENDING, 0) + 1
                    )
            break
        remaining -= cost
        spent += cost
        ready.append(sample_id)

    results = _execute_ready(
        ready, samples_by_id, config, needed_map, reuse_payloads, desired,
        parallel=parallel,
    )
    for (sample_id, scheme_name), stages in results.items():
        for stage, payload in stages.items():
            sort_key = task_sort_key_local(sample_id, scheme_name, stage)
            metadata = desired[sort_key]
            workspace.stores[stage][metadata["cache_key"]] = payload

    for sort_key, metadata in desired.items():
        if metadata["sample_id"] in ready:
            workspace.task_index[sort_key] = metadata

    batch_id = int(workspace.scheduler.get("batch_id", 0)) + 1
    last_batch = {
        "batch_id": batch_id,
        "budget_total": budget_total,
        "sample_point_quota": quota,
        "spent_points": spent,
        "remaining_points": remaining,
        "ready_sample_ids": ready,
        "pending_sample_ids": pending_ids,
        "completed_count": len(ready),
        "pending_count": len(pending_ids),
        "point_costs": point_costs,
        "pending_entries": pending_entries,
        "reuse_task_count": len(reuse),
        "recompute_task_count": len(recompute_tasks),
        "executed_task_count": sum(
            len(stages) for stages in results.values()
        ),
    }
    workspace.scheduler = {
        "batch_id": batch_id,
        "pending": pending_ids,
        "pending_entries": pending_entries,
        "last_batch": last_batch,
    }
    workspace.ledger_events.append({"type": "batch", **last_batch})
    load_payload = load_payload or {
        "failure_count": len(failures),
        "failures": failures,
        "accepted_sample_ids": sorted(samples_by_id),
    }
    manifest = workspace.commit(config, load_payload)

    return {
        "manifest": manifest,
        "desired": desired,
        "reuse": reuse,
        "recompute": recompute_tasks,
        "invalidations": current_invalidations,
        "ready": ready,
        "pending": pending_ids,
        "scheduler": last_batch,
        "scheduler_code_counts": scheduler_code_counts,
    }


def task_sort_key_local(sample_id: str, scheme: str, stage: str) -> str:
    return f"{sample_id}::{scheme}::{stage}"


def fresh_full_run(state_root: str | Path, config: dict[str, Any],
                   samples: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    from .reporting import build_report

    root = Path(state_root)
    if root.exists():
        shutil.rmtree(root)
    workspace = Workspace(root)
    run_summary = run_batch(workspace, config, samples, failures)
    report = build_report(config, samples, failures, workspace, run_summary)
    return {"workspace": workspace, "summary": run_summary, "report": report}
