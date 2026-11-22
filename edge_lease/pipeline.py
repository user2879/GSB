"""End-to-end pipeline orchestration and fault-injection entry points."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .converge import reconcile
from .errors import CycleError, GridError, ManifestError
from .fixedpoint import combine_error_rows, validate_scale
from .ledger import Ledger
from .logging_setup import JsonlLogger, mono_ms
from .manifest import load_manifests, load_registry, load_shards
from .models import Receipt
from .protocol import NodeProcess
from .scheduler import Scheduler
from .session import Session


@dataclass
class PipelineConfig:
    workdir: Path
    registry_path: Path
    shards_path: Path
    manifest_path: Path
    scale: int
    content_seed: int
    global_limit: int
    batch_size: int = 4
    startup_timeout_s: float = 15.0
    inject: Optional[str] = None  # corrupt_manifest | force_cycle | bad_grid
    crash_after_batches: Optional[int] = None  # simulate hard kill after N commits


class SimulatedHardKill(RuntimeError):
    """Raised after a committed batch to emulate SIGKILL of the process."""


def inject_fault(config: PipelineConfig) -> None:
    """Parameterized fault injection required by the acceptance criteria."""
    if config.inject == "bad_grid":
        validate_scale(3, location="session.scale")
    if config.inject == "corrupt_manifest":
        raise ManifestError(
            "injected manifest corruption: forced E_MANIFEST",
            "inject.corrupt_manifest",
        )
    if config.inject == "force_cycle":
        raise CycleError(
            "injected derivation cycle: forced E_CYCLE",
            "inject.force_cycle",
            {"cycle": ["a", "b", "a"]},
        )


def load_inputs(config: PipelineConfig):
    registry = load_registry(config.registry_path)
    shard_decls = load_shards(config.shards_path)
    manifests = load_manifests(
        config.manifest_path, registry, shard_decls
    )
    return registry, shard_decls, manifests


def start_nodes(
    registry: Dict[str, Any],
    scale: int,
    workdir: Path,
    logger: JsonlLogger,
    startup_timeout_s: float,
) -> Dict[str, NodeProcess]:
    processes: Dict[str, NodeProcess] = {}
    for node_id in sorted(registry):
        process = NodeProcess(
            registry[node_id],
            scale=scale,
            workdir=workdir,
            logger=logger,
            startup_timeout_s=startup_timeout_s,
        )
        try:
            process.start()
        except Exception:
            processes[node_id] = process
            raise
        processes[node_id] = process
    return processes


def stop_nodes(processes: Dict[str, NodeProcess]) -> None:
    for process in processes.values():
        process.stop()


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(workdir / "events.jsonl")
    inject_fault(config)  # raises E_MANIFEST / E_CYCLE / E_GRID first
    validate_scale(config.scale, location="session.scale")
    if config.global_limit <= 0:
        raise GridError("global concurrency limit must be positive", "session.global_limit")

    registry, _shard_decls, manifests = load_inputs(config)
    ledger = Ledger(workdir / "lease.sqlite", logger=logger)
    session = Session(ledger, logger, config.content_seed, config.scale)
    session.transition("LOADED", "inputs_validated", {"versions": len(manifests)})
    processes: Dict[str, NodeProcess] = {}
    try:
        processes = start_nodes(
            registry,
            config.scale,
            workdir,
            logger,
            config.startup_timeout_s,
        )
        all_results: List[Dict[str, Any]] = []
        for manifest in manifests:
            result = _run_version(
                config, session, ledger, logger, registry, manifest, processes
            )
            all_results.append(result)
        return {
            "status": "ok",
            "workdir": str(workdir),
            "versions": all_results,
        }
    finally:
        stop_nodes(processes)
        ledger.close()


def _run_version(
    config: PipelineConfig,
    session: Session,
    ledger: Ledger,
    logger: JsonlLogger,
    registry: Dict[str, Any],
    manifest,
    processes: Dict[str, NodeProcess],
) -> Dict[str, Any]:
    snapshot_diff = session.check_or_replace_snapshot(manifest)
    if session.state == "REPLACED":
        session.transition("COMPILED", "replacement_complete", {})
    compiled, items, plan_digest = session.compile_and_persist(manifest)
    if session.state == "LOADED":
        session.transition("COMPILED", "plan_compiled", {"items": len(items)})
    elif session.state != "COMPILED":
        session.transition("COMPILED", "plan_recompiled", {"items": len(items)})

    scheduler = Scheduler(
        node_processes=processes,
        registry=registry,
        manifest=manifest,
        content_seed=config.content_seed,
        logger=logger,
        global_limit=config.global_limit,
    )
    batch_summaries = []
    all_receipts: List[Receipt] = []
    for batch_index, start in enumerate(range(0, len(items), config.batch_size)):
        batch_items = items[start : start + config.batch_size]
        batch_id = f"v{manifest.version}-b{batch_index}"
        ledger.begin_batch(batch_id, manifest.version, len(batch_items), mono_ms())
        session.transition(
            "DISPATCHING",
            "batch_begin",
            {"batch_id": batch_id, "items": len(batch_items)},
        )
        result = scheduler.run(batch_items)
        batch_receipts = [
            receipt
            for record in result.records
            for receipt in record.receipts
        ]
        ok_items = len(result.ok_items())
        quota = {
            node_id: outcome.attempts
            for node_id, outcome in sorted(result.outcomes.items())
        }
        done_items = [
            record.item.item_id
            for record in result.records
            if record.final_status == "ok"
        ]
        cursor = done_items[-1] if done_items else batch_items[0].item_id
        ledger.commit_batch(
            batch_id=batch_id,
            receipts=batch_receipts,
            quota=quota,
            cursor_item=cursor,
            done_items=done_items,
            ok_items=ok_items,
            committed_ms=mono_ms(),
        )
        committed_count = len(
            [batch for batch in batch_summaries]
        )
        if (
            config.crash_after_batches is not None
            and committed_count >= config.crash_after_batches
        ):
            logger.error(
                "simulated_hard_kill",
                batch_id=batch_id,
                committed_batches=committed_count,
            )
            raise SimulatedHardKill(
                f"simulated SIGKILL after {committed_count} committed batch(es)"
            )
        all_receipts.extend(
            receipt for record in result.records for receipt in record.receipts
        )
        session.transition(
            "COMMITTED",
            "batch_committed",
            {"batch_id": batch_id, "ok": ok_items},
        )
        batch_summaries.append(
            {
                "batch_id": batch_id,
                "total": len(batch_items),
                "ok": ok_items,
                "unknown_on_reopen": False,
                "timing": [
                    {
                        "item_id": record.item.item_id,
                        "wait_ms": record.waited_ms,
                        "service_ms": max(
                            0, record.service_end_ms - record.service_start_ms
                        ),
                        "attempts": len(record.receipts),
                        "final_status": record.final_status,
                    }
                    for record in result.records
                ],
            }
        )
    receipts_by_item: Dict[str, List[Receipt]] = {}
    for receipt in all_receipts:
        receipts_by_item.setdefault(receipt.item_id, []).append(receipt)
    rows, _deviations, node_error_summary = reconcile(items, receipts_by_item)
    session.transition(
        "CONVERGED",
        "receipts_converged",
        {"items": len(rows), "plan_digest": plan_digest},
    )
    return {
        "version": manifest.version,
        "plan_digest": plan_digest,
        "items": len(items),
        "snapshot": snapshot_diff.to_dict(),
        "batches": batch_summaries,
        "node_outcomes": {
            node_id: outcome.to_dict()
            for node_id, outcome in scheduler.cumulative_outcomes.items()
        },
        "error_summary": node_error_summary,
        "total_error_bound": combine_error_rows(
            [dict(compiled[name].error) for name in sorted(compiled)]
        ),
        "convergence_rows": [row.to_dict() for row in rows],
    }
