from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import AppError
from .ledger import Ledger
from .models import Budgets, ClusterSpec, Revision, Spec, TaskSpec, Versions
from .orchestrator import NodeManager
from .planner import compare_plans
from .scheduler import schedule_plan
from .snapshot import load_snapshot, replay_snapshot, save_snapshot
from .validation import validate_spec
from .verification import verify_ledger_authoritative


def spec_from_plan(plan: dict[str, Any]) -> Spec:
    """Reconstruct the subset of a spec needed to start replay nodes."""

    shards = plan.get("shards", [])
    clusters: dict[str, ClusterSpec] = {}
    revisions: dict[tuple[str, int], dict[str, Any]] = {}
    for shard in shards:
        cluster_id = shard["cluster_id"]
        clusters.setdefault(
            cluster_id,
            ClusterSpec(
                id=cluster_id,
                host="127.0.0.1",
                port=int(shard["cluster_port"]),
                nodes=tuple(shard.get("nodes", [])),
                concurrency=int(shard["cluster_quota"]),
            ),
        )
        key = (cluster_id, int(shard["revision_version"]))
        revisions.setdefault(key, {}).update(shard.get("config_values", {}))
    tasks = []
    for task in plan.get("tasks", []):
        tasks.append(
            TaskSpec(
                id=task["task_id"],
                config_keys=tuple(task["config_keys"]),
                target_clusters=tuple(task["target_clusters"]),
                shards=int(task["shards"]),
                resource_units=int(task["resource_units"]),
                max_parallel=int(task["max_parallel"]),
            )
        )
    revision_models = [
        Revision(cluster_id=cluster_id, version=version, parent=None, config=dict(config))
        for (cluster_id, version), config in sorted(revisions.items())
    ]
    versions = plan.get("versions", {})
    return Spec(
        format_version=1,
        simulation_seed=int(plan["simulation_seed"]),
        clusters=tuple(sorted(clusters.values(), key=lambda item: item.id)),
        revisions=tuple(revision_models),
        tasks=tuple(sorted(tasks, key=lambda item: item.id)),
        budgets=Budgets(
            global_resource_units=int(plan["global_resource_units"]),
            default_task_parallel=int(plan["default_task_parallel"]),
        ),
        versions=Versions(
            clusters=int(versions.get("clusters", 1)),
            revisions=int(versions.get("revisions", 1)),
            budgets=int(versions.get("budgets", 1)),
        ),
    )


def _existing_plan(ledger: Ledger) -> dict[str, Any] | None:
    try:
        return ledger.run_plan()
    except AppError:
        return None


def execute_run(
    spec: Spec,
    plan: dict[str, Any],
    ledger_path: str | Path,
    snapshot_path: str | Path | None = None,
    handshake_timeout_ms: int = 3000,
    crash_after_commits: int = 0,
) -> dict[str, Any]:
    target_clusters = {shard["cluster_id"] for shard in plan.get("shards", [])}
    with Ledger(ledger_path) as ledger:
        ledger.verify_integrity()
        old_plan = _existing_plan(ledger)
        stale_before = ledger.stale_count()
        if old_plan is None:
            replacement = {
                "affected_shards": [shard["shard_id"] for shard in plan.get("shards", [])],
                "reasons": ["initial run"],
                "old_plan_digest": None,
                "new_plan_digest": plan["plan_digest"],
            }
        else:
            replacement = compare_plans(old_plan, plan)
        shard_by_id = {shard["shard_id"]: shard for shard in plan.get("shards", [])}

        with NodeManager(
            spec,
            handshake_timeout_ms=handshake_timeout_ms,
        ) as nodes:
            if target_clusters:
                nodes.start(target_clusters)
            run_id = ledger.begin_replacement(plan, replacement)
            committed = 0

            def on_complete(trace: dict[str, Any]) -> None:
                nonlocal committed
                ledger.commit_shard(run_id, trace, shard_by_id[trace["shard_id"]])
                committed += 1
                if crash_after_commits and committed >= crash_after_commits:
                    os._exit(17)

            result = schedule_plan(plan, nodes.dispatch_shard, on_complete=on_complete)
        ledger.finish_run(run_id, result)
        snapshot = None
        if snapshot_path is not None:
            snapshot = save_snapshot(snapshot_path, spec, ledger)
        return {
            "run_id": run_id,
            "plan_digest": plan["plan_digest"],
            "counts": result["counts"],
            "peak": result["peak"],
            "limits": result["limits"],
            "replacement": replacement,
            "stale_shards_before": stale_before,
            "stale_shards_after": ledger.stale_count(),
            "ledger": ledger.ledger_versions(),
            "snapshot": None
            if snapshot is None
            else {"path": str(snapshot_path), "content_digest": snapshot["content_digest"]},
            "trace": result["trace"],
        }


def preview_ledger(ledger_path: str | Path) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        ledger.verify_integrity()
        return ledger.preview()


def verify_run(ledger_path: str | Path, use_nodes: bool = True, handshake_timeout_ms: int = 3000) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        ledger.verify_integrity()
        plan = ledger.run_plan()
        if use_nodes:
            spec = spec_from_plan(plan)
            with NodeManager(spec, handshake_timeout_ms=handshake_timeout_ms) as nodes:
                clusters = {shard["cluster_id"] for shard in plan.get("shards", [])}
                if clusters:
                    nodes.start(clusters)
                result = verify_ledger_authoritative(ledger, dispatch=nodes.dispatch_shard)
        else:
            result = verify_ledger_authoritative(ledger)
        result["ledger"] = ledger.ledger_versions()
        return result


def recover_ledger(ledger_path: str | Path, run_id: str | None = None) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        ledger.verify_integrity()
        result = ledger.recover(run_id)
        result["ledger"] = ledger.ledger_versions()
        return result


def save_ledger_snapshot(spec: Spec, ledger_path: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    with Ledger(ledger_path) as ledger:
        ledger.verify_integrity()
        snapshot = save_snapshot(snapshot_path, spec, ledger)
        return {
            "path": str(snapshot_path),
            "content_digest": snapshot["content_digest"],
            "versions": snapshot["versions"],
            "run_id": snapshot["run"]["run_id"],
        }


def restore_snapshot(
    snapshot_path: str | Path,
    current_versions: dict[str, int] | None = None,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path, current_versions=current_versions)
    validate_spec(snapshot["spec"])
    return snapshot


def replay_restored_snapshot(
    snapshot_path: str | Path,
    handshake_timeout_ms: int = 3000,
    current_versions: dict[str, int] | None = None,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path, current_versions=current_versions)
    spec = validate_spec(snapshot["spec"])
    with NodeManager(spec, handshake_timeout_ms=handshake_timeout_ms) as nodes:
        clusters = {shard["cluster_id"] for shard in snapshot["plan"].get("shards", [])}
        if clusters:
            nodes.start(clusters)
        replay = replay_snapshot(snapshot, nodes.dispatch_shard)
    replay["restore"] = snapshot["restore"]
    replay["snapshot_versions"] = snapshot["versions"]
    return replay
