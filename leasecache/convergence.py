from __future__ import annotations

from pathlib import Path
from typing import Any

from .ledger import Ledger
from .node_manager import NodeManager
from .planner import build_decisions
from .spec import load_spec
from .util import StructuredLogger, atomic_write_json, mono_ms


def check_restart_convergence(
    spec_path: str | Path,
    db_path: str | Path,
    workspace: str | Path,
    *,
    node_ids: list[str] | None = None,
    report_path: str | Path | None = None,
    node_start_timeout_ms: int = 6000,
) -> dict[str, Any]:
    """Regenerate all node copies, restart nodes, and report stable invalidation.

    The first digest sync compares the regenerated copies against the summaries
    previously committed in the ledger.  If the specification (for example a
    node seed or reference list) changed, affected keys are returned in stable
    order.  Each requested node is then actually terminated and respawned; the
    post-restart sync must be empty because the explicit generation rule is
    deterministic.
    """

    started = mono_ms()
    spec = load_spec(spec_path)
    workspace_path = Path(workspace).resolve()
    runtime_dir = workspace_path / "runtime"
    logger = StructuredLogger(runtime_dir / "logs" / "engine.jsonl", "convergence")
    ledger = Ledger(db_path)
    manager = NodeManager(
        spec,
        workspace_path,
        runtime_dir,
        logger,
        start_timeout_ms=node_start_timeout_ms,
    )
    targets = sorted(node_ids or [node.node_id for node in spec.nodes])
    missing = sorted(set(targets) - set(spec.node_by_id))
    if missing:
        raise ValueError(f"unknown node ids: {missing}")
    try:
        session = ledger.start_session(spec.content_sha256)
        if session["requires_recovery"] or ledger.has_open_leases():
            ledger.recover_unknown_leases("restart convergence session recovered runtime leases")
        manager.start_all()
        ledger.sync_node_specs(spec)
        initial_snapshot = manager.digest_snapshot()
        initial_sync = ledger.sync_digests(initial_snapshot)
        reconciliation = ledger.reconcile_spec(spec)
        restart_results: list[dict[str, Any]] = []
        for node_id in targets:
            affected_for_node = sorted(
                change["artifact_key"]
                for change in initial_sync["summary_changes"]
                if change["node_id"] == node_id
            )
            orphaned = ledger.orphan_node_leases(
                node_id,
                "explicit node restart; prior physical lease ownership is unknown",
                invalidate_keys=set(affected_for_node),
            )
            manager.restart_node(node_id)
            after_snapshot = manager.digest_snapshot()
            after_sync = ledger.sync_digests(after_snapshot)
            restart_results.append(
                {
                    "node_id": node_id,
                    "orphaned_leases": orphaned,
                    "post_restart_affected_keys": after_sync["affected_keys"],
                    "post_restart_canonical_changed_keys": after_sync[
                        "canonical_changed_keys"
                    ],
                }
            )
        final_decisions = build_decisions(spec, ledger)
        report = {
            "status": "ok",
            "format_version": 1,
            "spec_sha256": spec.content_sha256,
            "initial_digest_sync": initial_sync,
            "spec_reconciliation": reconciliation,
            "affected_keys": initial_sync["affected_keys"],
            "restart_results": restart_results,
            "stable_affected_keys": sorted(set(initial_sync["affected_keys"])),
            "reusable_artifacts": sorted(
                decision["artifact_key"]
                for decision in final_decisions
                if decision["reusable"]
            ),
            "elapsed_mono_ms": mono_ms() - started,
        }
        ledger.record_decisions(
            spec,
            final_decisions,
            completed_keys={
                source["artifact_key"] for source in ledger.sources(valid_only=True)
            },
            phase="restart_convergence_complete",
        )
        close = ledger.close_session(status="complete")
        report["ledger"] = {
            "path": str(ledger.path),
            "content_digest": ledger.meta().get("content_digest"),
            "close_event_seq": close["event_seq"],
        }
        target = Path(report_path) if report_path else Path(db_path).with_suffix(
            ".convergence.json"
        )
        atomic_write_json(target, report)
        report["report_path"] = str(target)
        return report
    finally:
        manager.stop_all()
        ledger.close()
