"""Crash recovery and deterministic replay verification.

Recovery contract:

* only fully committed SQLite rows are trusted; an attempt whose row was never
  committed (process killed mid-flight) is recorded as ``unknown`` in
  ``recovery_events`` and the owning delivery is re-run;
* an exit report is never produced while unknown deliveries remain.

Replay contract:

* the snapshot's scene version, payload digest and mock failure/slow
  configuration must match the current scene, otherwise the session is
  declared non-reproducible (``E_SNAPSHOT`` with the concrete items);
* the same attempt sequence is re-driven against freshly started deterministic
  mock services; per-attempt outcomes, per-node outcome distributions and the
  aggregated success rate must match exactly.  P95 uses integer milliseconds
  measured over a real loopback, so a documented tolerance of
  max(2 ms, 5%) is applied there;
* the first divergence is reported with node id, delivery/attempt sequence and
  field name under ``E_REPLAY``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Scene
from .errors import ReplayError
from .ledger import AttemptRecord, Ledger
from .metrics import SCALE_RATIO, aggregate
from .scheduler import FanoutScheduler, now_ms
from .snapshot import Snapshot


def expected_jobs(scene: Scene) -> list[tuple[str, int]]:
    return [
        (node.node_id, delivery_seq)
        for delivery_seq in range(scene.task.deliveries_per_node)
        for node in scene.nodes
    ]


def recover(scene: Scene, ledger: Ledger, snapshot: Snapshot) -> list[tuple[str, int]]:
    """Return the subset of delivery jobs that must be re-run after a crash."""
    committed = {
        (row["node_id"], row["delivery_seq"]) for row in ledger.fetch_deliveries()
    }
    missing: list[tuple[str, int]] = []
    for node_id, delivery_seq in expected_jobs(scene):
        if (node_id, delivery_seq) not in committed:
            # The last in-flight attempt of this delivery has no committed row:
            # mark it unknown (idempotently) and require a re-run rather than
            # guessing an outcome or producing an exit report.
            if not ledger.has_recovery_event(node_id, delivery_seq):
                last_seen = ledger.max_committed_attempt(node_id, delivery_seq)
                ledger.record_unknown(
                    node_id=node_id, delivery_seq=delivery_seq,
                    attempt_seq=last_seen + 1,
                    reason="no committed row after forced termination",
                    detected_at_ms=now_ms(),
                )
            missing.append((node_id, delivery_seq))
    return missing


def resume_session(scene: Scene, ledger: Ledger, snapshot: Snapshot,
                   snapshot_dir: str | Path) -> dict[str, Any]:
    """Re-run only deliveries that lack a committed terminal row."""
    missing = recover(scene, ledger, snapshot)
    scheduler = FanoutScheduler(scene, ledger, snapshot_dir=str(snapshot_dir),
                                snapshot_every=0, only_jobs=missing or None)
    scheduler.run()
    records = ledger.fetch_attempts()
    metrics = aggregate(records, scene)
    return {"rerun_jobs": missing, "metrics": metrics, "records": records}


def _replay_attempts_sequentially(scene: Scene,
                                  planned: list[dict[str, Any]]) -> list[AttemptRecord]:
    """Re-drive the recorded attempt sequence one at a time (no concurrency)."""
    from .http_client import post_attempt

    replayed: list[AttemptRecord] = []
    for entry in planned:
        node = scene.node(entry["node_id"])
        started = now_ms()
        response = post_attempt(
            node, entry["attempt_seq"], scene.task.task_id,
            scene.task.payload_bytes, scene.task.attempt_timeout_ms,
        )
        ended = now_ms()
        replayed.append(AttemptRecord(
            node_id=node.node_id, delivery_seq=entry["delivery_seq"],
            attempt_seq=entry["attempt_seq"], wait_start_ms=started,
            wait_end_ms=started, started_ms=started, ended_ms=ended,
            elapsed_ms=max(0, ended - started), outcome=response.outcome,
            budget_consumed_ms=max(0, ended - started), backoff_ms=0,
            http_status=response.http_status, note=response.note,
        ))
    return replayed


def verify_replay(scene: Scene, snapshot: Snapshot) -> dict[str, Any]:
    """Replay the snapshot trajectory and compare field by field."""
    planned: list[dict[str, Any]] = snapshot.body.get("trajectory", [])
    replayed = _replay_attempts_sequentially(scene, planned)

    if len(replayed) != len(planned):
        raise ReplayError(
            "replay attempt count differs from snapshot",
            {"field": "trajectory.length",
             "snapshot": len(planned), "replay": len(replayed)},
        )

    # 1) per-attempt outcome classification; locate the first divergence.
    for index, (entry, record) in enumerate(zip(planned, replayed)):
        identity = {
            "node_id": entry["node_id"],
            "delivery_seq": entry["delivery_seq"],
            "attempt_seq": entry["attempt_seq"],
        }
        if entry["node_id"] != record.node_id:
            raise ReplayError("first mismatch on node_id",
                              {"index": index, "field": "node_id", **identity,
                               "snapshot": entry["node_id"], "replay": record.node_id})
        if entry["attempt_seq"] != record.attempt_seq:
            raise ReplayError("first mismatch on attempt_seq",
                              {"index": index, "field": "attempt_seq", **identity,
                               "snapshot": entry["attempt_seq"], "replay": record.attempt_seq})
        if entry["outcome"] != record.outcome:
            raise ReplayError(
                "first mismatch on attempt outcome",
                {"index": index, "field": "outcome", **identity,
                 "snapshot": entry["outcome"], "replay": record.outcome,
                 "basis": "deterministic mock rule re-driven with identical attempt_seq"},
            )

    # 2) per-node outcome distributions.
    def distribution(rows: Any, key: str) -> dict[str, dict[str, int]]:
        buckets: dict[str, dict[str, int]] = {}
        for row in rows:
            node_id = row["node_id"] if isinstance(row, dict) else row.node_id
            outcome = row[key] if isinstance(row, dict) else row.outcome
            buckets.setdefault(node_id, {})
            buckets[node_id][outcome] = buckets[node_id].get(outcome, 0) + 1
        return {node_id: dict(sorted(counts.items()))
                for node_id, counts in sorted(buckets.items())}

    snap_dist = distribution(planned, "outcome")
    replay_dist = distribution(replayed, "outcome")
    for node_id in sorted(set(snap_dist) | set(replay_dist)):
        if snap_dist.get(node_id) != replay_dist.get(node_id):
            raise ReplayError(
                "first mismatch on node outcome distribution",
                {"field": f"node_outcome_distribution.{node_id}",
                 "snapshot": snap_dist.get(node_id), "replay": replay_dist.get(node_id)},
            )

    # 3) aggregate success rate (exact fixed-point over delivery verdicts)
    #    and P95 (documented jitter over all attempt latencies).
    replay_metrics = aggregate(replayed, scene)
    snap_global = snapshot.body.get("metrics", {}).get("global", {})
    replay_global = replay_metrics["global"]
    if snap_global.get("success_rate_scaled") != replay_global["success_rate_scaled"]:
        raise ReplayError(
            "first mismatch on aggregate delivery success rate",
            {"field": "global.success_rate_scaled",
             "snapshot": snap_global.get("success_rate_scaled"),
             "replay": replay_global["success_rate_scaled"],
             "scale": SCALE_RATIO,
             "basis": "fixed-point success rate over terminal delivery outcomes"},
        )
    snap_p95 = int(snap_global.get("p95_latency_ms", 0))
    replay_p95 = replay_global["p95_latency_ms"]
    # P95 is integer wall-clock ms measured over a real loopback.  Outcome
    # classification and success rate are exact; only P95 carries a documented
    # measurement tolerance of max(2 ms, 5% of the recorded value) to absorb
    # scheduler/OS jitter on very short requests.
    tolerance_ms = max(2, (snap_p95 * 5 + 99) // 100)
    jitter = abs(snap_p95 - replay_p95)
    if jitter > tolerance_ms:
        raise ReplayError(
            "first mismatch on aggregate P95 beyond the documented jitter tolerance",
            {"field": "global.p95_latency_ms",
             "snapshot": snap_p95, "replay": replay_p95,
             "tolerance_ms": tolerance_ms,
             "basis": "integer ms measured over a real loopback; nearest-rank P95; "
                      "tolerance=max(2ms, 5%)"},
        )
    return {
        "matched_attempts": len(replayed),
        "node_distribution": replay_dist,
        "snapshot_p95_ms": snap_p95,
        "replay_p95_ms": replay_p95,
        "replay_success_rate_scaled": replay_global["success_rate_scaled"],
        "p95_jitter_ms": jitter,
        "p95_tolerance_ms": tolerance_ms,
    }
