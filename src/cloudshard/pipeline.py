from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .adjudicate import Adjudicator
from .canonical import canonical_text, content_digest
from .cluster import NodeCluster
from .invalidate import build_invalidation_plan
from .ledger import Ledger
from .recovery import recover_inflight
from .scheduler import DispatchEngine
from .snapshot import atomic_write_json, build_snapshot
from .spec import Spec, load_spec


def import_declared_adjudication(spec: Spec, ledger: Ledger) -> dict[str, Any]:
    existing = ledger.conn.execute("SELECT COUNT(*) AS c FROM event_log WHERE stream='declared'").fetchone()["c"]
    adj = Adjudicator.from_spec(spec).replay(list(spec.events))
    if int(existing) == 0:
        for event in spec.events:
            ledger.append_event("declared", event)
        for record in adj.ownership_sequence:
            ledger.record_ownership("declared", record)
        for decision in adj.decisions:
            ledger.record_decision("declared", int(decision["event_seq"]), decision)
    return adj.result_dict()


def execute_pipeline(
    spec_path: str | Path,
    workdir: str | Path,
    *,
    ledger_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    start_nodes: bool = True,
    run_second_revision: bool = True,
    crash_before_commit: str | None = None,
    crash_marker: str | Path | None = None,
) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path else workdir / "ledger.sqlite"
    snapshot_path = Path(snapshot_path) if snapshot_path else workdir / "snapshot.json"
    spec = load_spec(spec_path)

    ledger = Ledger(ledger_path)
    ledger.initialize_meta(content_digest(spec.raw), str(spec_path))
    recovery_marks = recover_inflight(ledger)
    declared = import_declared_adjudication(spec, ledger)

    cluster: NodeCluster | None = None
    try:
        if start_nodes:
            cluster = NodeCluster(spec, spec_path, workdir)
            cluster.start_all()
        engine = DispatchEngine(spec, ledger)
        if crash_before_commit:
            engine.crash_before_commit = crash_before_commit
            engine.crash_marker = Path(crash_marker) if crash_marker else workdir / "crash-marker.json"

        first = engine.run_target(spec.initial_revision, [s.shard_id for s in spec.shards])
        snapshot1 = build_snapshot(spec, ledger, revision=spec.initial_revision, extra={"run": first})
        atomic_write_json(workdir / "snapshot.v1.json", snapshot1)
        ledger.write_snapshot_record(spec.initial_revision, snapshot1)

        invalidation = None
        second = None
        if run_second_revision and len(spec.revisions) > 1:
            revision = spec.revisions[-1].revision
            plan = build_invalidation_plan(spec, revision)
            invalidation = plan.to_dict()
            atomic_write_json(workdir / "invalidation.json", invalidation)
            second = engine.run_target(revision, list(plan.invalid_shard_ids))
            snapshot = build_snapshot(
                spec,
                ledger,
                revision=revision,
                extra={"invalidation": invalidation, "second_run": second, "recovery_marks": recovery_marks},
            )
        else:
            snapshot = build_snapshot(
                spec,
                ledger,
                revision=spec.initial_revision,
                extra={"first_run": first, "recovery_marks": recovery_marks},
            )
        atomic_write_json(snapshot_path, snapshot)
        ledger.write_snapshot_record(snapshot["revision"], snapshot)
        report = {
            "ok": True,
            "spec": str(spec_path),
            "workdir": str(workdir),
            "ledger": str(ledger_path),
            "snapshot": str(snapshot_path),
            "recovery_marks": recovery_marks,
            "first_run": first,
            "second_run": second,
            "declared_ownership": declared["ownership_sequence"],
            "invalidation": invalidation,
            "ledger_digest": ledger.ledger_digest(),
        }
        atomic_write_json(workdir / "run-report.json", report)
        return report
    finally:
        if cluster is not None:
            cluster.stop_all()
        ledger.close()


def replay_declared(spec: Spec, out_path: str | Path | None = None) -> dict[str, Any]:
    adj = Adjudicator.from_spec(spec).replay(list(spec.events))
    result = adj.result_dict()
    result["ownership_canonical_bytes"] = adj.ownership_bytes().decode("utf-8")
    if out_path:
        atomic_write_json(out_path, result)
    return result


def check_batch_boundaries(spec: Spec) -> dict[str, Any]:
    events = list(spec.events)
    full = Adjudicator.from_spec(spec).replay(events).ownership_bytes()
    boundaries: list[dict[str, Any]] = []
    for boundary in range(1, len(events) + 1):
        adj = Adjudicator.from_spec(spec)
        for start in range(0, len(events), boundary):
            adj.replay(events[start : start + boundary])
        actual = adj.ownership_bytes()
        boundaries.append({"batch_size": boundary, "byte_equal": actual == full, "bytes": len(actual)})
    return {
        "all_equal": all(item["byte_equal"] for item in boundaries),
        "boundaries": boundaries,
        "sha256": hashlib.sha256(full).hexdigest(),
    }


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = ["execute_pipeline", "replay_declared", "check_batch_boundaries", "canonical_text"]
