"""End-to-end orchestration: load, index, lease, snapshot, converge, commit."""

import json
import os
from pathlib import Path

from .canon import canonical_json, key_digest
from .clock import sleep_ms
from .converge import judge_convergence
from .errors import ChainError
from .index import build_index_entries
from .ledger import Ledger
from .leases import LeaseManager
from .log import StructuredLogger
from .scheduler import Scheduler
from .snapshots import (build_snapshot_rows, make_snapshot_id,
                        replay_from_snapshot, shard_state_digest)
from .spec import load_all_rows, load_spec


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _current_head(ledger):
    rows = ledger.query("SELECT snapshot_id, parent_id FROM snapshots")
    if not rows:
        return None
    parents = {r["parent_id"] for r in rows if r["parent_id"]}
    heads = [r["snapshot_id"] for r in rows if r["snapshot_id"] not in parents]
    if len(heads) != 1:
        raise ChainError("snapshot chain must have exactly one head",
                         details={"heads": sorted(heads)})
    return heads[0]


def _next_batch_id(ledger):
    rows = ledger.query("SELECT COALESCE(MAX(batch_id), 0) AS m FROM progress")
    return int(rows[0]["m"]) + 1


def run_pipeline(spec_path, db_path, *, renewals_path=None, session_out=None,
                 batch_size=2, max_workers=2, per_shard_quota=1,
                 crash_before_batch=None, only_shards=None, logger=None,
                 expire_margin_ms=50):
    """Run the full batch pipeline. Business errors propagate with codes."""
    logger = logger or StructuredLogger(Path(db_path).with_suffix(".log.jsonl"))
    spec = load_spec(spec_path)
    rows_by_shard = load_all_rows(spec)
    ledger = Ledger(db_path)
    logger.emit("spec_loaded", code=None, shard_count=len(spec.shards),
                digest=spec.digest)

    renewal_plan = {"renewals": []}
    if renewals_path:
        renewal_plan = json.loads(Path(renewals_path).read_text(encoding="utf-8"))

    committed = set(ledger.committed_shards())
    todo = [sid for sid in (only_shards or spec.shard_ids())
            if sid not in committed]
    lease_mgr = LeaseManager(spec, logger=logger)
    scheduler = Scheduler(max_workers=max_workers,
                          per_shard_quota=per_shard_quota, logger=logger)
    batch_id = _next_batch_id(ledger)
    prev_head = _current_head(ledger)
    report = {"batches": [], "conclusions": {}, "replay_ok": None}

    try:
        for batch in _chunks(todo, batch_size):
            if crash_before_batch is not None and batch_id == crash_before_batch:
                logger.emit("simulated_crash", code=None, batch_id=batch_id)
                ledger.close()
                os._exit(3)  # forced termination: uncommitted batch is lost

            # 1. Deterministic index build for the batch.
            entries_by_shard = {}
            for sid in batch:
                shard = spec.shard(sid)
                entries_by_shard[sid] = build_index_entries(
                    shard, rows_by_shard[sid], batch_id, buckets=spec.buckets)

            # 2. Issue one lease per key; renewal plan uses per-shard ordinals.
            leases_by_shard = {sid: [] for sid in batch}
            for sid in batch:
                seen_keys = sorted({row["key"] for row in rows_by_shard[sid]})
                for key in seen_keys:
                    lease = lease_mgr.issue(sid, key_digest(key),
                                            batch_id=batch_id)
                    leases_by_shard[sid].append(lease)

            # 3. Scheduled renewals under the per-shard in-flight quota.
            futures = []
            for item in renewal_plan.get("renewals", []):
                sid = item["shard_id"]
                if sid not in leases_by_shard or not leases_by_shard[sid]:
                    continue
                ordinal = item["lease_ordinal"] % len(leases_by_shard[sid])
                lease_id = leases_by_shard[sid][ordinal]["lease_id"]

                def renew_task(sid=sid, lease_id=lease_id,
                               expected=item["expected_version"]):
                    lease_mgr.acquire_renewal_slot(sid)
                    try:
                        return lease_mgr.renew(lease_id, expected)
                    finally:
                        lease_mgr.release_renewal_slot(sid)

                futures.append(scheduler.submit(sid, "renew", renew_task))
            for fut in futures:
                fut.result()  # E_LEASE propagates here

            # 4. Parallel replay tasks under global/per-shard quotas.
            def replay_task(sid):
                shard = spec.shard(sid)
                return {
                    "shard_id": sid,
                    "index_pos_lo": lease_mgr.expiry_position(sid),
                    "index_pos_hi": len(rows_by_shard[sid]),
                    "lease_expiry_pos": lease_mgr.expiry_position(sid),
                    "entry_count": len(entries_by_shard[sid]),
                    "tolerance": shard.tolerance,
                }

            replay_futures = [scheduler.submit(sid, "replay", replay_task, sid)
                              for sid in batch]
            replay_infos = {sid: f.result() for sid, f in
                            zip(batch, replay_futures)}

            # 5. Real-clock expiry and reclamation.
            ttl = max(spec.shard(sid).lease.ttl_ms for sid in batch)
            sleep_ms(ttl + expire_margin_ms)
            lease_mgr.expire_due()

            # 6. Shard states, snapshot chain links, convergence.
            shard_states = {}
            for sid in batch:
                expiry_pos = lease_mgr.expiry_position(sid)
                shard_states[sid] = {
                    "index_pos_lo": expiry_pos,
                    "index_pos_hi": len(rows_by_shard[sid]),
                    "lease_expiry_pos": expiry_pos,
                    "content_digest": shard_state_digest(
                        entries_by_shard[sid],
                        [l for l in lease_mgr.all_leases()
                         if l["shard_id"] == sid]),
                }
            parent_of = {}
            parent = prev_head
            for sid in sorted(shard_states):
                parent_of[sid] = parent
                parent = make_snapshot_id(batch_id, sid)
            snapshot_rows = build_snapshot_rows(spec, batch_id, shard_states,
                                                parent_of)
            prev_head = snapshot_rows[-1]["snapshot_id"]

            replayed = {sid: {"index_pos_lo": s["index_pos_lo"]}
                        for sid, s in shard_states.items()}
            expiry_positions = {sid: s["lease_expiry_pos"]
                                for sid, s in shard_states.items()}
            conclusions = judge_convergence(
                spec, replayed, expiry_positions, logger=logger,
                shard_ids=batch)

            # 7. Single-transaction batch commit.
            wait_events = (scheduler.wait_events + lease_mgr.wait_events)
            digest = ledger.commit_batch(
                batch_id,
                index_entries=[e for sid in batch for e in entries_by_shard[sid]],
                leases=[l for sid in batch
                        for l in lease_mgr.all_leases_for(sid)],
                snapshots=snapshot_rows,
                conclusions=list(conclusions.values()),
                progress=[{"shard_id": sid, "state": "committed"}
                          for sid in batch],
                wait_events=wait_events)
            scheduler.wait_events.clear()
            lease_mgr.wait_events.clear()
            logger.emit("batch_committed", code=None, batch_id=batch_id,
                        shards=sorted(batch), content_digest=digest)
            if session_out:
                from .recovery import write_session_snapshot
                write_session_snapshot(ledger, session_out)
            report["batches"].append({"batch_id": batch_id,
                                      "shards": sorted(batch),
                                      "content_digest": digest})
            report["conclusions"].update(conclusions)
            batch_id += 1
    finally:
        scheduler.shutdown()

    # 8. Post-run replay verification against a direct rebuild.
    if not only_shards or set(spec.shard_ids()) <= set(ledger.committed_shards()):
        ok = replay_verify(ledger, spec, rows_by_shard)
        report["replay_ok"] = ok
        logger.emit("replay_verified", code=None, ok=ok)

    if session_out:
        from .recovery import write_session_snapshot
        write_session_snapshot(ledger, session_out)
        logger.emit("session_snapshot_written", code=None, path=str(session_out))
    ledger.close()
    return report


def replay_verify(ledger, spec, rows_by_shard=None):
    """Replay from the snapshot head and compare with a direct rebuild.

    The direct rebuild bypasses snapshots entirely: positions come from the
    committed index entries and lease states. Both sides must be
    byte-identical under canonical serialization.
    """
    if rows_by_shard is None:
        rows_by_shard = load_all_rows(spec)
    head = _current_head(ledger)
    replayed = replay_from_snapshot(ledger, spec, head)
    direct = {}
    leases = ledger.query("SELECT * FROM leases")
    entries = ledger.query("SELECT * FROM index_entries")
    for sid in spec.shard_ids():
        shard_leases = [l for l in leases if l["shard_id"] == sid]
        shard_entries = [e for e in entries if e["shard_id"] == sid]
        expired = sum(1 for l in shard_leases if l["state"] == "expired")
        direct[sid] = {
            "shard_id": sid,
            "index_pos_lo": expired,
            "index_pos_hi": len(rows_by_shard[sid]),
            "lease_expiry_pos": expired,
            "content_digest": shard_state_digest(shard_entries, shard_leases),
        }
    return canonical_json({sid: replayed[sid] for sid in spec.shard_ids()}) == \
        canonical_json(direct)
