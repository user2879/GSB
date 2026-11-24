from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import LeaseError
from .expiry import ExpiryWorker
from .ledger import Ledger
from .node_manager import NodeManager
from .planner import (
    build_decisions,
    canonical_copies,
    per_node_digest_map,
    reuse_plan,
)
from .scheduler import ParallelLeaseScheduler
from .spec import PipelineSpec, load_spec
from .util import StructuredLogger, atomic_write_json, mono_ms


@dataclass
class RunOptions:
    lease_ms: int | None = None
    settle: bool = True
    release_active_on_close: bool = True
    reuse_only: bool = False
    wait_timeout_ms: int = 20000
    node_start_timeout_ms: int = 6000
    request_timeout_ms: int = 6000
    service_delay_ms: int = 0
    interrupt_after_grants: int = 0
    report_path: str | None = None
    settle_timeout_ms: int | None = None


@dataclass
class EngineRun:
    spec: PipelineSpec
    ledger: Ledger
    manager: NodeManager
    logger: StructuredLogger
    options: RunOptions
    runtime_dir: Path
    lease_ms: int
    session: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] | None = None
    digest_sync: dict[str, Any] | None = None
    spec_reconciliation: dict[str, Any] | None = None
    expiry: ExpiryWorker | None = None
    scheduler: ParallelLeaseScheduler | None = None
    scheduler_result: dict[str, Any] | None = None
    initial_reuse_plan: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed: bool = False


def _release_all_active(
    run: EngineRun, reason: str
) -> list[dict[str, Any]]:
    released: list[dict[str, Any]] = []
    for lease in run.ledger.active_leases():
        closed = run.ledger.release_lease(lease["lease_id"], reason)
        if closed is not None:
            released.append(closed)
            try:
                run.manager.client(closed["node_id"]).release(closed["lease_id"])
            except Exception:
                pass
    return released


def _ownership_events(run: EngineRun) -> list[dict[str, Any]]:
    kinds = {
        "LEASE_GRANTED",
        "LEASE_RENEWED",
        "LEASE_EXPIRED",
        "QUOTA_RELEASED",
        "LEASE_UNKNOWN",
        "LEASE_INVALIDATED",
    }
    return [event for event in run.ledger.events() if event["kind"] in kinds]


def _business_snapshot(
    run: EngineRun,
    decisions: list[dict[str, Any]],
    per_node_digests: dict[str, dict[str, str]],
) -> dict[str, Any]:
    sources = run.ledger.sources(valid_only=True)
    valid_by_key: dict[str, list[str]] = {}
    for source in sources:
        valid_by_key.setdefault(source["artifact_key"], []).append(source["node_id"])
    return {
        "format_version": 1,
        "spec_sha256": run.spec.content_sha256,
        "pipeline_id": run.spec.pipeline_id,
        "nodes": [
            {
                "node_id": node["node_id"],
                "port": node["port"],
                "capacity_slots": node["capacity_slots"],
                "lease_limit": node["lease_limit"],
                "used_slots": node["used_slots"],
                "active_leases": node["active_leases"],
                "free_slots": node["free_slots"],
                "digests": per_node_digests.get(node["node_id"], {}),
            }
            for node in run.ledger.nodes()
        ],
        "artifacts": [
            {
                "artifact_key": decision["artifact_key"],
                "canonical_digest": decision["canonical_digest"],
                "references": decision["references"],
                "source_nodes": decision["source_nodes"],
                "valid_source_nodes": sorted(valid_by_key.get(decision["artifact_key"], [])),
                "reusable": decision["reusable"],
                "selected_node": decision["selected_node"],
            }
            for decision in decisions
        ],
        "active_leases": run.ledger.active_leases(),
        "progress": {
            "edit_hash": run.ledger.progress()["edit_hash"],
            "phase": run.ledger.progress()["phase"],
            "completed_keys": run.ledger.progress()["completed_keys"],
        },
    }


def _build_report(
    run: EngineRun,
    decisions: list[dict[str, Any]],
    *,
    status: str,
    released: list[dict[str, Any]],
    elapsed_mono_ms: int,
) -> dict[str, Any]:
    summaries = run.ledger.summaries()
    per_node = per_node_digest_map(summaries)
    artifact_by_key = run.spec.artifact_by_key
    artifacts = []
    for decision in decisions:
        key = decision["artifact_key"]
        artifacts.append(
            {
                **decision,
                "per_node_digests": {
                    node_id: per_node.get(node_id, {}).get(key, "")
                    for node_id in run.spec.sorted_node_ids
                },
            }
        )
    snapshot = _business_snapshot(run, decisions, per_node)
    ledger_meta = run.ledger.meta()
    return {
        "status": status,
        "format_version": 1,
        "run": {
            "lease_ms": run.lease_ms,
            "elapsed_mono_ms": elapsed_mono_ms,
            "settled": bool(run.options.settle),
            "reuse_only": bool(run.options.reuse_only),
            "spec_path": str(run.spec.path),
            "spec_sha256": run.spec.content_sha256,
            "workspace": str(run.runtime_dir.parent),
            "runtime_dir": str(run.runtime_dir),
            "log_path": str(run.logger.path),
        },
        "session": run.session,
        "recovery": run.recovery,
        "digest_sync": run.digest_sync,
        "spec_reconciliation": run.spec_reconciliation,
        "incremental": {
            "initial_plan": run.initial_reuse_plan,
            "reusable_keys": (run.scheduler_result or {}).get("reusable_keys", []),
            "recomputed_keys": (run.scheduler_result or {}).get("recomputed_keys", []),
        },
        "scheduling": run.scheduler_result,
        "nodes": [
            {
                **node,
                "per_node_digests": per_node.get(node["node_id"], {}),
            }
            for node in run.ledger.nodes()
        ],
        "artifacts": artifacts,
        "leases": run.ledger.list_leases(),
        "active_leases": run.ledger.active_leases(),
        "released_on_close": released,
        "expirations": (run.expiry.expirations if run.expiry else []),
        "ownership_events": _ownership_events(run),
        "events_tail": run.ledger.events(limit=300),
        "ledger": {
            "path": str(run.ledger.path),
            "schema_version": ledger_meta.get("schema_version"),
            "format_version": int(ledger_meta.get("format_version", "0")),
            "content_digest": ledger_meta.get("content_digest"),
            "clean_shutdown": ledger_meta.get("clean_shutdown"),
            "progress": run.ledger.progress(),
        },
        "business_snapshot": snapshot,
        "artifacts_by_key": {
            artifact["artifact_key"]: artifact for artifact in artifacts
        },
        "artifact_references": {
            key: list(artifact_by_key[key].references) for key in sorted(artifact_by_key)
        },
    }


def run_engine(
    spec_path: str | Path,
    db_path: str | Path,
    workspace: str | Path,
    *,
    options: RunOptions | None = None,
) -> dict[str, Any]:
    options = options or RunOptions()
    started = mono_ms()
    spec = load_spec(spec_path)
    workspace_path = Path(workspace).resolve()
    runtime_dir = workspace_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "logs" / "engine.jsonl"
    logger = StructuredLogger(log_path, "engine")
    ledger = Ledger(db_path)
    lease_ms = options.lease_ms if options.lease_ms is not None else spec.lease_ms
    if lease_ms <= 0:
        raise LeaseError(
            "E_SPEC",
            "lease duration override must be a positive integer",
            field="lease_ms",
            expected="positive integer",
            actual=lease_ms,
        )
    run = EngineRun(
        spec=spec,
        ledger=ledger,
        manager=NodeManager(
            spec,
            workspace_path,
            runtime_dir,
            logger,
            start_timeout_ms=options.node_start_timeout_ms,
            request_timeout_ms=options.request_timeout_ms,
        ),
        logger=logger,
        options=options,
        runtime_dir=runtime_dir,
        lease_ms=lease_ms,
    )
    report_path = Path(
        options.report_path or (str(Path(db_path).with_suffix("")) + ".report.json")
    ).resolve()
    logger.info(
        "engine run starting",
        spec=str(spec.path),
        spec_sha256=spec.content_sha256,
        ledger=str(ledger.path),
        workspace=str(workspace_path),
        lease_ms=lease_ms,
    )
    try:
        run.session = ledger.start_session(spec.content_sha256)
        if run.session["requires_recovery"] or ledger.has_open_leases():
            # Runtime leases are node-process scoped.  A fresh engine always
            # starts fresh node processes; committed-but-still-active records
            # therefore have unknown physical ownership and cannot be promised.
            run.recovery = ledger.recover_unknown_leases(
                "node runtime restarted; active/pending lease ownership is unknown"
            )
            if run.recovery["recovered"]:
                logger.warning(
                    "unclean committed leases recovered as unknown",
                    recovered=run.recovery["recovered"],
                    affected_keys=run.recovery["affected_keys"],
                )
        run.manager.start_all()
        ledger.sync_node_specs(spec)
        snapshot_before = run.manager.digest_snapshot()
        run.digest_sync = ledger.sync_digests(snapshot_before)
        run.spec_reconciliation = ledger.reconcile_spec(spec)
        logger.info(
            "node digests converged with ledger",
            affected_keys=run.digest_sync["affected_keys"],
            canonical_changed_keys=run.digest_sync["canonical_changed_keys"],
        )
        run.initial_reuse_plan = reuse_plan(spec, ledger)
        run.expiry = ExpiryWorker(ledger, logger)
        run.expiry.start()
        if not options.reuse_only:
            run.scheduler = ParallelLeaseScheduler(
                spec,
                ledger,
                run.manager,
                logger,
                lease_ms=lease_ms,
                wait_timeout_ms=options.wait_timeout_ms,
                service_delay_ms=options.service_delay_ms,
                interrupt_after_grants=options.interrupt_after_grants,
                lease_namespace=str(run.session["session_number"]),
            )
            run.scheduler_result = run.scheduler.run()
            if run.expiry.errors:
                raise run.expiry.errors[0]
            if options.settle:
                timeout = options.settle_timeout_ms or (
                    lease_ms + options.wait_timeout_ms + 3000
                )
                if not run.expiry.wait_until_idle(timeout):
                    raise LeaseError(
                        "E_EXPIRY",
                        "active leases did not reach their local monotonic deadlines",
                        field="expires_mono_ms",
                        expected=f"all leases expired within {timeout} ms",
                        actual=f"{len(ledger.active_leases())} still active",
                    )
            elif options.release_active_on_close:
                released = _release_all_active(run, "quota reclaim before clean close")
                if released:
                    logger.info("active leases reclaimed before close", released=released)
        else:
            run.scheduler_result = {
                "reusable_keys": sorted(
                    key
                    for key, decision in run.initial_reuse_plan.items()
                    if decision["reusable"]
                ),
                "recompute_keys": sorted(
                    key
                    for key, decision in run.initial_reuse_plan.items()
                    if not decision["reusable"]
                ),
                "recomputed_keys": [],
                "requests": [],
                "queue_waits": [],
            }
            if options.release_active_on_close:
                _release_all_active(run, "reuse-only close reclaimed stale active leases")

        decisions = build_decisions(spec, ledger)
        completed = {
            decision["artifact_key"]
            for decision in decisions
            if decision["source_nodes"]
        }
        phase = "reuse_analysis_complete" if options.reuse_only else "run_complete"
        ledger.record_decisions(
            spec, decisions, completed_keys=completed, phase=phase
        )
        if run.expiry is not None:
            if run.expiry.errors:
                raise run.expiry.errors[0]
            run.expiry.stop()
        released = _release_all_active(run, "clean close safety reclaim")
        if released:
            decisions = build_decisions(spec, ledger)
            completed = {
                decision["artifact_key"]
                for decision in decisions
                if decision["source_nodes"]
            }
            ledger.record_decisions(
                spec, decisions, completed_keys=completed, phase=phase
            )
        report = _build_report(
            run,
            decisions,
            status="ok",
            released=released,
            elapsed_mono_ms=mono_ms() - started,
        )
        report["ledger"]["content_digest"] = ledger.meta().get("content_digest")
        close = ledger.close_session(status="complete", report_path=str(report_path))
        report["ledger"] = {
            "path": str(ledger.path),
            "schema_version": ledger.meta().get("schema_version"),
            "format_version": int(ledger.meta().get("format_version", "0")),
            "content_digest": ledger.meta().get("content_digest"),
            "clean_shutdown": ledger.meta().get("clean_shutdown"),
            "progress": ledger.progress(),
            "close_event_seq": close["event_seq"],
        }
        # Decisions were built before SESSION_CLOSED; verifier uses the same
        # persistent projections, so the digest below reflects the final commit.
        report["business_snapshot"] = _business_snapshot(run, decisions, per_node_digest_map(ledger.summaries()))
        report["business_snapshot"]["progress"] = {
            "edit_hash": spec.content_sha256,
            "phase": "run_complete",
            "completed_keys": sorted(completed),
        }
        report["ledger"]["content_digest"] = ledger.meta().get("content_digest")
        atomic_write_json(report_path, report)
        run.closed = True
        logger.info(
            "engine run completed",
            report_path=str(report_path),
            ledger_digest=report["ledger"]["content_digest"],
            elapsed_mono_ms=report["run"]["elapsed_mono_ms"],
        )
        return report
    except BaseException as exc:
        if isinstance(exc, LeaseError):
            logger.error(exc.code, exc.message, **exc.log_fields())
        else:
            logger.error(
                "E_INTERNAL",
                "uncaught engine failure",
                traceback="".join(traceback.format_exception(exc)),
            )
        if run.expiry is not None:
            run.expiry.stop()
        try:
            _release_all_active(run, f"failure cleanup: {type(exc).__name__}")
            ledger.close_session(status="failed")
        except Exception:
            pass
        raise
    finally:
        if run.expiry is not None:
            run.expiry.stop()
        run.manager.stop_all()
        ledger.close()


def compare_business_snapshots(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []

    def visit(a: Any, b: Any, path: str) -> None:
        if len(differences) >= 50:
            return
        if type(a) is not type(b):
            differences.append({"path": path, "left": a, "right": b})
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append({"path": f"{path}.{key}", "left": "<missing>", "right": b[key]})
                elif key not in b:
                    differences.append({"path": f"{path}.{key}", "left": a[key], "right": "<missing>"})
                else:
                    visit(a[key], b[key], f"{path}.{key}" if path else key)
            return
        if isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": path, "left_length": len(a), "right_length": len(b)})
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if a != b:
            differences.append({"path": path, "left": a, "right": b})

    visit(left, right, "")
    return {
        "equal": not differences,
        "difference_count": len(differences),
        "differences": differences,
    }
