"""Repeatable end-to-end classroom demonstration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .audit import JsonlReport, StructuredLogger
from .canonical import sha256_object
from .ledger import Ledger
from .packing import PackStore, build_pack_tasks, compare_artifact_sets
from .propagation import propagate_invalidation
from .registry import ReplicaRegistry
from .repair import apply_and_report, block_replica_digest
from .sample import generate_sample
from .scheduler import PackScheduler
from .snapshot import SnapshotManager, compare_replay
from .spec import load_spec, spec_manifest_digest


def run_demo(
    root: str | Path = ".",
    *,
    simulate_ms: int = 12,
    timeout_ms: int = 1_500,
    max_retries: int = 1,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    spec_path = generate_sample(root_path, force=True)
    run_root = root_path / "demo-run"
    if run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "logs").mkdir(parents=True)
    (run_root / "snapshots").mkdir(parents=True)
    (run_root / "state").mkdir(parents=True)

    ledger_path = run_root / "ledger.sqlite3"
    ledger = Ledger(ledger_path)
    logger = StructuredLogger(run_root / "logs" / "events.jsonl", ledger)
    spec = load_spec(spec_path)
    spec_digest = spec_manifest_digest(spec)
    registry = ReplicaRegistry.load(spec, logger=logger)
    pack_store = PackStore(run_root / "packs", spec)
    quotas = {node.node_id: node.pack_quota for node in spec.nodes}
    global_slots = 2
    baseline = registry.summary()

    # Stage 1: a damage event followed by a complete repair and full repack.
    injection1 = registry.inject(shard_id="node-a:s2", kind="truncate", sequence=0)
    propagation1 = propagate_invalidation(spec, [injection1.logical_id])
    registry.mark_affected(propagation1.affected)
    repair1 = apply_and_report(spec, registry, logger=logger)
    full_tasks = build_pack_tasks(
        spec,
        spec.topo_order,
        timeout_ms=timeout_ms,
        simulate_ms=simulate_ms,
        max_retries=max_retries,
        quotas=quotas,
    )
    schedule1 = PackScheduler(
        store=pack_store,
        global_slots=global_slots,
        node_quotas=quotas,
        logger=logger,
        ledger=ledger,
        run_id="demo-full-repair",
    ).run(full_tasks)
    registry.clear_pending()
    _require_success(schedule1, "stage-1 full repair")
    manifest_after_full = pack_store.manifest()

    # Stage 2: inject one new damaged replica, propagate only to its descendants,
    # repack only that affected set, and prove unaffected artifacts are untouched.
    before_entries = {
        logical_id: dict(entry) for logical_id, entry in manifest_after_full["artifacts"].items()
    }
    before_files = {
        logical_id: pack_store.artifact_file_digest(logical_id) for logical_id in before_entries
    }
    injection2 = registry.inject(shard_id="node-b:s4", kind="flip", offset=17, sequence=0)
    propagation2 = propagate_invalidation(spec, [injection2.logical_id])
    registry.mark_affected(propagation2.affected)
    incremental_tasks = build_pack_tasks(
        spec,
        propagation2.affected,
        timeout_ms=timeout_ms,
        simulate_ms=simulate_ms,
        max_retries=max_retries,
        quotas=quotas,
    )
    schedule2 = PackScheduler(
        store=pack_store,
        global_slots=global_slots,
        node_quotas=quotas,
        logger=logger,
        ledger=ledger,
        run_id="demo-incremental",
    ).run(incremental_tasks)
    _require_success(schedule2, "stage-2 incremental repack")
    repair2 = apply_and_report(spec, registry, logger=logger)
    registry.clear_pending(propagation2.affected)
    manifest_incremental = pack_store.manifest()

    affected_set = set(propagation2.affected)
    unaffected = sorted(set(before_entries) - affected_set)
    unaffected_checks = []
    for logical_id in unaffected:
        after = manifest_incremental["artifacts"][logical_id]
        unaffected_checks.append(
            {
                "logical_id": logical_id,
                "version_unchanged": before_entries[logical_id]["version"] == after["version"],
                "digest_unchanged": before_entries[logical_id]["content_digest"] == after["content_digest"],
                "artifact_bytes_unchanged": before_files[logical_id]
                == pack_store.artifact_file_digest(logical_id),
            }
        )
    if not all(
        item["version_unchanged"] and item["digest_unchanged"] and item["artifact_bytes_unchanged"]
        for item in unaffected_checks
    ):
        raise RuntimeError("an unaffected pack artifact changed during incremental repack")

    incremental_block_digest = block_replica_digest(spec, registry)
    full_store = PackStore(run_root / "verify-full", spec)
    full_store.repack_all_direct()
    incremental_verification = compare_artifact_sets(
        manifest_incremental,
        full_store.manifest(),
        incremental_registry_digest=incremental_block_digest,
        full_registry_digest=incremental_block_digest,
    )

    # Stage 3: save, create a deliberate current-state drift, restore as a whole
    # replacement, then replay the recorded injection in a clean workspace.
    snapshot_manager = SnapshotManager(run_root / "snapshots" / "session.json")
    budget = {
        "global_slots": global_slots,
        "node_quotas": quotas,
        "timeout_ms": timeout_ms,
        "simulate_ms": simulate_ms,
        "max_retries": max_retries,
    }
    snapshot = snapshot_manager.save(
        spec=spec,
        registry=registry,
        pack_store=pack_store,
        schedule=schedule2,
        repair_report=repair2,
        injections=[injection2],
        budget=budget,
        spec_digest=spec_digest,
        replay={"affected": list(propagation2.affected)},
        block_replica_digest=incremental_block_digest,
    )

    registry.inject(shard_id="node-c:s6", kind="replace", sequence=9)
    drifted_digest = registry.disk_digest()
    restore = snapshot_manager.restore(
        spec=spec,
        registry=registry,
        state_path=run_root / "state" / "current_state.json",
        logger=logger,
    )

    replay_root = run_root / "replay"
    replay_spec_path = generate_sample(replay_root, force=True)
    replay_ledger = Ledger(replay_root / "replay.sqlite3")
    replay_logger = StructuredLogger(replay_root / "logs" / "events.jsonl", replay_ledger)
    replay_spec = load_spec(replay_spec_path)
    replay_registry = ReplicaRegistry.load(replay_spec, logger=replay_logger)
    replay_store = PackStore(replay_root / "packs", replay_spec)
    replay_store.repack_all_direct()  # rebuild the version-1 baseline before replay
    loaded_snapshot = snapshot_manager.load()
    applied_replay_injections = snapshot_manager.apply_recorded_injections(
        loaded_snapshot, replay_registry
    )
    replay_affected = propagate_invalidation(
        replay_spec, [injection.logical_id for injection in applied_replay_injections]
    )
    replay_registry.mark_affected(replay_affected.affected)
    replay_tasks = build_pack_tasks(
        replay_spec,
        replay_affected.affected,
        timeout_ms=budget["timeout_ms"],
        simulate_ms=budget["simulate_ms"],
        max_retries=budget["max_retries"],
        quotas=budget["node_quotas"],
    )
    replay_schedule = PackScheduler(
        store=replay_store,
        global_slots=budget["global_slots"],
        node_quotas=budget["node_quotas"],
        logger=replay_logger,
        ledger=replay_ledger,
        run_id="demo-incremental",
    ).run(replay_tasks)
    _require_success(replay_schedule, "snapshot replay")
    replay_repair = apply_and_report(replay_spec, replay_registry, logger=replay_logger)
    replay_registry.clear_pending(replay_affected.affected)
    replay_block_digest = block_replica_digest(replay_spec, replay_registry)
    replay_observed = {
        "affected": list(replay_affected.affected),
        "pack_manifest": replay_store.manifest(),
        "schedule": replay_schedule.as_dict(),
        "repair_report": replay_repair,
        "registry_digest": replay_registry.state_digest(),
        "block_replica_digest": replay_block_digest,
    }
    replay_comparison = compare_replay(loaded_snapshot, replay_observed)

    stable_facts = {
        "spec_digest": spec_digest,
        "stage1_affected": list(propagation1.affected),
        "stage2_affected": list(propagation2.affected),
        "stage2_task_ids": [task.task_id for task in incremental_tasks],
        "stage2_schedule": [
            {
                "logical_id": entry.logical_id,
                "node_id": entry.node_id,
                "outcome": entry.outcome,
                "version": entry.version,
                "content_digest": entry.content_digest,
            }
            for entry in schedule2.entries
        ],
        "pack_artifacts": _stable_artifacts(manifest_incremental),
        "repair_counts": repair2["counts"],
        "unaffected_checks": unaffected_checks,
        "replay_checked_fields": replay_comparison["checked_fields"],
    }
    stable_digest = sha256_object(stable_facts)

    report = {
        "code": "E_DEMO_OK",
        "workspace": str(root_path),
        "spec_path": str(spec_path.relative_to(root_path)),
        "baseline": baseline,
        "stage1_full_repair": {
            "injection": injection1.as_dict(),
            "affected": list(propagation1.affected),
            "repair_counts": repair1["counts"],
            "packed_shards": [task.logical_id for task in full_tasks],
            "schedule": schedule1.as_dict(),
        },
        "stage2_incremental": {
            "injection": injection2.as_dict(),
            "affected": list(propagation2.affected),
            "propagation": propagation2.as_dict(),
            "packed_shards": [task.logical_id for task in incremental_tasks],
            "schedule": schedule2.as_dict(),
            "repair_counts": repair2["counts"],
            "unaffected_checks": unaffected_checks,
            "incremental_vs_full": incremental_verification,
            "block_replica_digest": incremental_block_digest,
        },
        "snapshot_restore": {
            "snapshot_version": snapshot["snapshot_version"],
            "snapshot_content_digest": snapshot["content_digest"],
            "drifted_disk_digest": drifted_digest,
            "restore": restore,
        },
        "replay": {
            **replay_comparison,
            "injections": [injection.as_dict() for injection in applied_replay_injections],
            "affected": list(replay_affected.affected),
            "schedule": replay_schedule.as_dict(),
            "repair_counts": replay_repair["counts"],
        },
        "stable_facts": stable_facts,
        "stable_digest": stable_digest,
        "artifacts": {
            "pack_manifest": str((run_root / "packs" / "manifest.json").relative_to(root_path)),
            "snapshot": str((run_root / "snapshots" / "session.json").relative_to(root_path)),
            "restored_state": str((run_root / "state" / "current_state.json").relative_to(root_path)),
            "log": str((run_root / "logs" / "events.jsonl").relative_to(root_path)),
            "sqlite": str(ledger_path.relative_to(root_path)),
        },
    }
    JsonlReport.write(run_root / "demo_report.json", report)
    JsonlReport.write(run_root / "stable_facts.json", stable_facts)
    # Leave the teaching sample pristine after the report has recorded the drift.
    generate_sample(root_path, force=True)
    return report


def _require_success(schedule: Any, stage: str) -> None:
    failures = [
        entry.as_dict() for entry in schedule.entries if entry.outcome not in {"success", "recovered"}
    ]
    if failures:
        raise RuntimeError(f"{stage} did not complete successfully: {json.dumps(failures, ensure_ascii=False)}")
    if schedule.max_global_observed > schedule.global_slots:
        raise RuntimeError(f"{stage} exceeded the global slot budget")
    for node_id, observed in schedule.max_per_node_observed.items():
        if observed > schedule.node_quotas.get(node_id, 0):
            raise RuntimeError(f"{stage} exceeded node quota for {node_id}")


def _stable_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        logical_id: {
            "content_digest": entry["content_digest"],
            "version": entry["version"],
            "bytes": entry["bytes"],
        }
        for logical_id, entry in sorted(manifest.get("artifacts", {}).items())
    }


def load_demo_report(root: str | Path = ".") -> dict[str, Any]:
    path = Path(root).resolve() / "demo-run" / "demo_report.json"
    return json.loads(path.read_text(encoding="utf-8"))
