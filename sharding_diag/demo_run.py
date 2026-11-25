"""End-to-end diagnostic driver covering all eight contracted requirements.

Run::

    python3 -m sharding_diag.demo_run

It builds deterministic samples, dispatches the DAG across three node
subprocesses, drives freeze/lease/revocation/capacity evolution, forces a hard
kill and replays recovery, then writes a structured report under artifacts/.
"""

import copy
import json
import os
import shutil

from .clock import ManualClock, MonotonicClock
from .errors import DiagError, LeaseError
from .freeze import FROZEN, UNFROZEN, UPSTREAM_HELD
from .invalidation import diff_declarations, propagate
from .ledger import Ledger
from .model import load_declaration, parse_declaration
from .recovery import rebuild_scheduler, replay_committed, rerun_unknown
from .revocation import (
    Candidate, DISTINGUISHABLE, LEASE_VERSION_CONFLICT,
    TOLERANCE_INDISTINGUISHABLE, UPSTREAM_UNDETERMINED,
    adjudicate_batch, revocation_statistics,
)
from .sampling import (
    compute_summary, generate_sample, materialize_samples, summary_digest,
)
from .scheduler import Scheduler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "declaration.json")
LEDGER_DIR = os.path.join(ROOT, "ledger")
SAMPLES_DIR = os.path.join(ROOT, "samples")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DB_PATH = os.path.join(LEDGER_DIR, "diag.db")


def _write_report(name, payload):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return path


def _evolve(declaration, mutations):
    # Build fully independent plain dicts so the original declaration is never
    # aliased/mutated; diff_declarations then observes every real change.
    raw = {
        "nodes": [{
            "node_id": n.node_id, "host": n.host, "port": n.port,
            "concurrency": n.concurrency, "lease_limit": n.lease_limit,
            "seed": n.seed, "capacity_version": n.capacity_version,
            "fixed_point_scale": n.fixed_point_scale,
        } for n in declaration.nodes],
        "shards": [{"shard_id": s.shard_id, "depends_on": list(s.depends_on),
                    "rows": s.rows, "demand": s.demand, "priority": s.priority}
                   for s in declaration.shards],
        "global": {
            "budget": declaration.global_spec.budget,
            "freeze_threshold_ms": declaration.global_spec.freeze_threshold_ms,
            "revoke_tolerance": declaration.global_spec.revoke_tolerance,
            "lease_ttl_ms": declaration.global_spec.lease_ttl_ms,
            "fixed_point_scale": declaration.global_spec.fixed_point_scale,
        },
    }
    for section, key, field, value in mutations:
        for item in raw[section]:
            if item.get("node_id") == key or item.get("shard_id") == key:
                item[field] = value
    return parse_declaration(raw)


def main():
    report = {"steps": []}

    def step(name, payload):
        report["steps"].append({"step": name, **payload})
        print(f"[ok] {name}")

    # ---- clean prior run ------------------------------------------------
    for path in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.join(LEDGER_DIR, "backups"), ignore_errors=True)
    if os.path.exists(SAMPLES_DIR):
        shutil.rmtree(SAMPLES_DIR)

    declaration = load_declaration(CONFIG)
    manifest = materialize_samples(declaration, SAMPLES_DIR)
    step("samples_materialized", {"manifest": manifest,
                                  "sample_files": len(os.listdir(SAMPLES_DIR)) - 1})

    # ---- requirement 4/2: full DAG dispatch with real subprocesses ------
    clock = MonotonicClock()
    ledger = Ledger(DB_PATH, declaration).open()
    scheduler = Scheduler(declaration, ledger, clock)
    scheduler.start_nodes()
    run_result = scheduler.run()
    outcomes = ledger.outcomes()
    assert len(outcomes) == len(declaration.shards), "every shard must commit"
    assert ledger.verify_chain(), "ledger content digest chain must verify"
    wait_with_bounds = [r for r in run_result["wait_records"] if r.get("wait_ms") is not None]
    step("full_dispatch", {
        "committed": sorted(outcomes),
        "rounds": run_result["rounds"],
        "queue_waits": wait_with_bounds,
        "budget": declaration.global_spec.budget,
    })

    # Freeze history: root shards thaw immediately, downstream thaws after upstreams.
    freeze_history = scheduler.freeze_mgr.history()
    step("freeze_semantics", {
        "frozen_at_end": scheduler.freeze_mgr.frozen_shards(),
        "ready_at_end": scheduler.freeze_mgr.ready_shards(),
        "migrations": freeze_history,
    })

    # ---- requirement 3: lease release uniqueness ------------------------
    release_log = scheduler.lease_mgr.release_log()
    reasons = {}
    for event in release_log:
        reasons[event["reason"]] = reasons.get(event["reason"], 0) + 1
    lease_ids = [event["lease_id"] for event in release_log]
    assert len(lease_ids) == len(set(lease_ids)), "each lease released at most once"
    # Double release attempt must be a no-op.
    sample_lease = release_log[0]
    second = scheduler.lease_mgr.release_on_complete(sample_lease["node_id"], sample_lease["shard_id"])
    assert second is None
    step("lease_release_arbitration", {"release_reasons": reasons,
                                       "double_release_returns": second})

    # ---- requirement 5: revocation adjudication -------------------------
    adj_path = _adjudication_demo(declaration, scheduler, ledger)
    step("revocation_adjudication", {"report": adj_path})

    # ---- requirement 6: capacity/quota/demand evolution + subset rerun --
    inv_path = _invalidation_demo(declaration, scheduler, ledger)
    step("invalidation_subset_recompute", {"report": inv_path})

    # ---- requirement 7: snapshot identity mismatch -> replace -----------
    ledger.checkpoint()
    snap_path = _snapshot_replace_demo(declaration)
    step("snapshot_whole_replace", {"report": snap_path})
    scheduler.stop_nodes()
    ledger.close()

    # ---- requirement 8: hard kill + replay ------------------------------
    kill_path = _hard_kill_demo(declaration)
    step("hard_kill_recovery_replay", {"report": kill_path})

    final_ledger = Ledger(DB_PATH, declaration).open()
    report["final_identity"] = final_ledger.identity()
    final_ledger.close()
    out = _write_report("run_report.json", report)
    print(f"[ok] report -> {out}")
    return out


def _candidate_from_outcome(outcome, version_override=None, total_override=None):
    return Candidate(
        shard_id=outcome["shard_id"],
        node_id=outcome["node_id"],
        version=outcome["lease_version"] if version_override is None else version_override,
        capacity_version=outcome["capacity_version"],
        fixed_total=outcome["fixed_total"] if total_override is None else total_override,
        summary_digest=outcome["summary_digest"],
    )


def _adjudication_demo(declaration, scheduler, ledger):
    outcomes = ledger.outcomes()
    tolerance = declaration.global_spec.revoke_tolerance
    candidates = {}
    expected = {}

    s1 = outcomes["s1"]
    # Tolerance-indistinguishable pair: diff 500 fixed units inside tolerance 1000.
    incumbent = _candidate_from_outcome(s1)
    challenger = Candidate(
        shard_id="s1", node_id="node-b", version=incumbent.version + 1,
        capacity_version=incumbent.capacity_version,
        fixed_total=incumbent.fixed_total + 500,
        summary_digest="sha256:challenger-s1",
    )
    candidates["s1"] = [incumbent, challenger]
    expected["s1"] = incumbent.version

    # Distinguishable pair on s2: diff exceeds tolerance.
    s2 = outcomes["s2"]
    inc2 = _candidate_from_outcome(s2)
    alt2 = Candidate(
        shard_id="s2", node_id="node-c", version=inc2.version + 1,
        capacity_version=inc2.capacity_version,
        fixed_total=inc2.fixed_total + tolerance + 1,
        summary_digest="sha256:challenger-s2",
    )
    candidates["s2"] = [inc2, alt2]
    expected["s2"] = inc2.version

    # Lease version conflict on s3: expected version differs from candidates.
    s3 = outcomes["s3"]
    candidates["s3"] = [_candidate_from_outcome(s3)]
    expected["s3"] = s3["lease_version"] + 5

    # Frozen shard must be excluded from revocation statistics.
    frozen_shard_id = "s8"
    candidates[frozen_shard_id] = [_candidate_from_outcome(outcomes[frozen_shard_id])]
    expected[frozen_shard_id] = None
    scheduler.freeze_mgr.refreeze(frozen_shard_id, "upstream", missing=["s6", "s7"])

    decisions = adjudicate_batch(
        candidates, expected, scheduler.freeze_mgr.frozen_shards(), tolerance
    )
    at_ms = scheduler.clock.now_ms()
    for decision in decisions:
        ledger.record_revocation(decision, at_ms)
    stats = revocation_statistics(decisions)
    cause_map = {d["shard_id"]: d["cause"] for d in decisions}
    assert cause_map["s1"] == TOLERANCE_INDISTINGUISHABLE
    assert cause_map["s2"] == DISTINGUISHABLE
    assert cause_map["s3"] == LEASE_VERSION_CONFLICT
    assert cause_map[frozen_shard_id] == UPSTREAM_UNDETERMINED
    assert stats["frozen_excluded"] == 1
    # Optimistic lease version mismatch must be rejected with E_LEASE.
    lease_error = None
    try:
        scheduler.lease_mgr.revoke(s3["node_id"], "s3", expected["s3"])
    except LeaseError as exc:
        lease_error = exc.as_dict()
    payload = {"tolerance": tolerance, "decisions": decisions, "statistics": stats,
               "lease_version_conflict_error": lease_error}
    return _write_report("revocation_report.json", payload)


def _invalidation_demo(declaration, scheduler, ledger):
    before_hashes = ledger.entry_hashes()
    assignments = {sid: row["node_id"] for sid, row in ledger.outcomes().items()}

    evolved = _evolve(declaration, [
        ("nodes", "node-b", "capacity_version", 2),
        ("nodes", "node-c", "concurrency", 2),
        ("shards", "s2", "demand", 6),
    ])
    changes = diff_declarations(declaration, evolved)
    plan = propagate(evolved, changes, assignments)

    # Scheduler adopts the evolved declaration but only the invalid subset reruns.
    scheduler.declaration = evolved
    rerun = scheduler.run_subset(set(plan["invalidated"]))

    after_hashes = ledger.entry_hashes()
    untouched = plan["untouched"]
    preserved = {sid: before_hashes[sid] == after_hashes[sid] for sid in untouched}
    assert all(preserved.values()), "untouched shard ledger entries must be byte-identical"

    replay = replay_committed(scheduler, declaration)
    payload = {
        "changes": [c.as_dict() for c in changes],
        "invalidation_plan": plan,
        "rerun_shards": plan["invalidated"],
        "untouched_preserved": preserved,
        "replay": replay,
        "rounds": rerun["rounds"],
    }
    return _write_report("invalidation_report.json", payload)


def _snapshot_replace_demo(declaration):
    # Open the existing committed snapshot against a drifted declaration.
    drifted = _evolve(declaration, [("nodes", "node-a", "capacity_version", 9)])
    snap_path = os.path.join(LEDGER_DIR, "snapshot_check.db")
    shutil.copy2(DB_PATH, snap_path)
    ledger = Ledger(snap_path, drifted).open()
    identity = ledger.identity()
    assert ledger.replaced, "identity drift must trigger whole-file replacement"
    assert os.path.exists(ledger.backup_path)
    backup_mode = oct(os.stat(ledger.backup_path).st_mode & 0o777)
    backup_kept = ledger.backup_path
    result = {
        "replace_reason": ledger.replace_reason,
        "backup_path": backup_kept,
        "backup_mode": backup_mode,
        "fresh_identity": identity,
        "fresh_outcome_count": len(ledger.outcomes()),
    }
    ledger.close()
    # Keep the backup read-only (0444); only the temporary replaced DB is removed.
    if os.path.exists(snap_path):
        os.remove(snap_path)
    for suffix in ("-wal", "-shm"):
        side = snap_path + suffix
        if os.path.exists(side):
            os.remove(side)
    return _write_report("snapshot_replace_report.json", result)


def _hard_kill_demo(declaration):
    # Fresh ledger and a scheduler whose node is killed while work is in flight.
    kill_db = os.path.join(LEDGER_DIR, "hard_kill.db")
    for suffix in ("", "-wal", "-shm"):
        path = kill_db + suffix
        if os.path.exists(path):
            os.remove(path)
    clock = MonotonicClock()
    ledger = Ledger(kill_db, declaration).open()
    scheduler = Scheduler(declaration, ledger, clock)
    scheduler.start_nodes()
    scheduler.start_pool()

    shard_map = declaration.shard_map()
    outcomes = {}
    # Commit only the root shards s1/s2, then launch s3 and kill its node mid-flight.
    scheduler.freeze_mgr.initialize(s.shard_id for s in declaration.shards)
    roots = ["s1", "s2"]
    scheduler.run_subset(set(roots))

    target = "s3"
    shard = shard_map[target]
    scheduler.start_pool()
    migrations = scheduler.freeze_mgr.evaluate_round([shard], ledger.outcomes())
    for migration in migrations:
        ledger.record_freeze_migration(migration)
    node_spec = next(n for n in scheduler.declaration.nodes if n.node_id == "node-a")
    scheduler._dispatch(shard, node_spec)
    job = scheduler._inflight[target]
    scheduler._nodes["node-a"].kill9()
    try:
        job["future"].result()
    except Exception as exc:  # connection dies with the subprocess
        connection_error = repr(exc)
    else:
        connection_error = None
    scheduler.shutdown_pool()
    scheduler.start_pool()

    # Recovery trusts committed rows only; the uncommitted lease is reclaimed.
    recovery = rebuild_scheduler(scheduler)
    # The killed node is respawned before the recovered shards are re-dispatched.
    scheduler.start_nodes()
    rerun = rerun_unknown(scheduler)
    replay = replay_committed(scheduler)
    chain_ok = ledger.verify_chain()
    payload = {
        "committed_before_kill": roots,
        "connection_error_on_kill": connection_error,
        "recovery": recovery,
        "rerun": rerun,
        "replay": replay,
        "chain_verified": chain_ok,
        "final_outcomes": sorted(ledger.outcomes()),
    }
    scheduler.shutdown_pool()
    ledger.close()
    scheduler.stop_nodes()
    return _write_report("hard_kill_report.json", payload)


if __name__ == "__main__":
    try:
        main()
    except DiagError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        raise
