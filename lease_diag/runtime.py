"""Runtime orchestration: live demo, persistence, recovery and replay."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .clock import RealClock
from .diagnostics import (DEFAULT_SCALE, global_watermark, pool_watermark,
                          rank_pools)
from .engine import Engine
from .ledger import Ledger
from .loader import load_workspace
from .model import EV_QUEUE
from .replay import (assert_no_gap, compare_lease_ownership,
                     compare_watermarks, replay_engine)
from .server import CoordinatorServer
from .snapshot import (build_snapshot, expected_versions,
                       whole_replace_leases, verify_snapshot)

BATCH_SIZE = 3


def compute_watermarks(engine, pools: dict, queue_by_pool: dict,
                       scale: int = DEFAULT_SCALE) -> list[dict]:
    effective = {pid: len(engine._active_leases(pid)) for pid in pools}
    marks: list[dict] = []
    for pid in sorted(pools):
        mark = pool_watermark(pools[pid], effective[pid],
                              queue_by_pool.get(pid, 0), scale)
        payload = mark.to_dict()
        payload["scope"] = f"pool:{pid}"
        marks.append(payload)
    overall = global_watermark(pools, effective, queue_by_pool, scale)
    payload = overall.to_dict()
    payload["scope"] = "global"
    marks.append(payload)
    return marks


def queue_lengths(events: list[dict], pools: dict) -> dict:
    queued = {pid: 0 for pid in pools}
    for event in events:
        if event["event_type"] == EV_QUEUE and event.get("pool_id"):
            queued[event["pool_id"]] = queued.get(event["pool_id"], 0) + 1
    return queued


def run_session(workspace: dict, workdir: Path, *, scale: int,
                global_cap: int, time_scale: int, socket_timeout_ms: int,
                max_retries: int, batch_size: int = BATCH_SIZE,
                hard_exit_batch: int | None = None,
                resume: bool = False) -> dict:
    from .server import LeaseClient

    pools = workspace["pools"]
    requests = workspace["requests"]
    clock = RealClock()
    engine = Engine(pools, workspace["rollback_targets"])
    ledger = Ledger(workdir / "lease_ledger.sqlite")
    ledger.put_meta("pools", {pid: pool.to_dict()
                              for pid, pool in pools.items()})
    ledger.put_meta("scale", scale)
    ledger.put_meta("pool_set_version", workspace["pool_set_version"])
    ledger.put_meta("rollback_version", workspace["rollback_version"])

    start_index = 0
    reset_info = {"removed_batches": []}
    if resume:
        reset_info = ledger.reset_to_last_committed()
        progress = ledger.progress()
        if progress is not None:
            whole_replace_leases(engine, ledger)
            engine._seq = int(progress["last_event_seq"])
            done = {rid for rid in ledger.request_sequence()}
            start_index = next(
                (idx for idx, req in enumerate(requests)
                 if req.request_id == progress["last_request_id"]), -1) + 1
            assert {req.request_id for req in requests[:start_index]} <= done

    server = CoordinatorServer(engine, clock, pools, global_cap=global_cap)
    server.start()
    client = LeaseClient(server.host, server.port, socket_timeout_ms,
                         max_retries)
    origin_ms = clock.now_ms()
    responses: list[dict] = []
    pending = list(enumerate(requests[start_index:], start=start_index + 1))
    try:
        batches: list[dict] = []
        for position, request in pending:
            index = position
            target_ms = origin_ms + request.arrive_ms // time_scale
            wait = target_ms - clock.now_ms()
            if wait > 0:
                clock.sleep_ms(wait)
            # Expiry is driven by the real monotonic clock via engine.tick.
            engine.tick(clock.now_ms())
            existing = engine.active_index.get((request.pool_id, request.key))
            kind = "RENEW" if existing is not None else "GRANT"
            response = client.send({
                "kind": kind, "request_id": request.request_id,
                "key": request.key, "pool_id": request.pool_id,
                "revision": request.revision, "priority": request.priority,
                "arrive_ms": request.arrive_ms})
            responses.append(response)

            if index % batch_size == 0 or index == len(requests):
                batch_id = ledger.begin_batch(clock.now_ms())
                if hard_exit_batch == batch_id:
                    _hard_exit(workdir, batch_id)
                last_seq = (ledger.progress() or
                            {"last_event_seq": 0}).get("last_event_seq", 0)
                new_events = [event.to_dict() for event in engine.events
                              if event.seq > last_seq]
                queue_by_pool = queue_lengths(new_events, pools)
                marks = compute_watermarks(engine, pools, queue_by_pool, scale)
                leases = {lease_id: lease.to_dict()
                          for lease_id, lease in engine.leases.items()}
                cursor = {"last_request_id": request.request_id,
                          "last_event_seq": engine._seq}
                snapshot = build_snapshot(
                    pool_set_version=workspace["pool_set_version"],
                    rollback_version=workspace["rollback_version"],
                    scale=scale, watermarks=marks, leases=leases,
                    cursor=cursor)
                ledger.commit_batch(
                    batch_id, events=new_events, leases=leases,
                    watermarks=marks, last_request_id=request.request_id,
                    committed_ms=clock.now_ms(), snapshot=snapshot)
                batches.append({"batch_id": batch_id,
                                "events": len(new_events),
                                "last_request_id": request.request_id})
        return {"responses": responses, "batches": batches,
                "resumed_from_request_index": start_index,
                "reset_info": reset_info,
                "port": server.port,
                "final_watermarks": compute_watermarks(
                    engine, pools, {}, scale),
                "ranking": _rank(engine, pools, scale),
                "ledger_path": str(workdir / "lease_ledger.sqlite")}
    finally:
        server.stop()
        ledger.close()


def _rank(engine, pools, scale):
    from .diagnostics import pool_watermark
    marks = [pool_watermark(pool, len(engine._active_leases(pid)), 0, scale)
             for pid, pool in sorted(pools.items())]
    return rank_pools(marks, scale)


def _hard_exit(workdir: Path, batch_id: int) -> None:
    """Simulate SIGKILL: flush nothing and leave the OPEN transaction behind."""
    marker = workdir / "hard_exit.json"
    marker.write_text(json.dumps(
        {"batch_id": batch_id,
         "note": "process force-terminated mid-transaction; no exit report "
                 "is claimed for this run"}))
    os._exit(87)


def recover_session(workspace: dict, workdir: Path, *, scale: int,
                    snapshot_tamper: bool = False) -> dict:
    pools = workspace["pools"]
    requests = workspace["requests"]
    ledger = Ledger(workdir / "lease_ledger.sqlite")
    report = ledger.recovery_report()

    if snapshot_tamper:
        row = ledger.load_snapshot_row()
        if row is None:
            raise RuntimeError("cannot inject E_SNAPSHOT before any snapshot")
        tampered = dict(row["payload"])
        tampered["scale"] = scale + 1
        ledger.replace_snapshot_payload(tampered, row["digest"])

    snapshot_row = ledger.load_snapshot_row()
    snapshot_verdict: dict
    if snapshot_row is None:
        snapshot_verdict = {"status": "absent",
                            "action": "rebuilt from ledger"}
        engine = Engine(pools, workspace["rollback_targets"])
        whole_replace_leases(engine, ledger)
    else:
        expected = expected_versions(
            pool_set_version=workspace["pool_set_version"],
            rollback_version=workspace["rollback_version"], scale=scale)
        try:
            verify_snapshot(snapshot_row, expected)
            snapshot_verdict = {"status": "verified",
                                "action": "restored from snapshot"}
            engine = Engine(pools, workspace["rollback_targets"])
            _restore_from_snapshot(engine, snapshot_row["payload"])
        except Exception as exc:
            snapshot_verdict = {"status": "rejected",
                                "action": "whole-state replacement from ledger",
                                "error": exc.to_dict()
                                if hasattr(exc, "to_dict") else str(exc)}
            engine = Engine(pools, workspace["rollback_targets"])
            whole_replace_leases(engine, ledger)

    # E_GAP is raised here if the current stream lacks ledger-referenced ids.
    assert_no_gap(ledger, requests)
    replay_engine_obj, comparisons = replay_engine(
        ledger, pools, workspace["rollback_targets"], requests)
    compare_lease_ownership(replay_engine_obj, ledger)
    replay_marks = _replay_watermarks(ledger, pools, scale)
    compare_watermarks(ledger, replay_marks)

    result = {"recovery": report, "snapshot": snapshot_verdict,
              "replay": {"requests_compared": len(comparisons),
                         "all_match": all(item["match"]
                                          for item in comparisons)},
              "ownership": {"active_leases": len(engine.active_index),
                            "total_leases": len(engine.leases)},
              "watermarks": ledger.latest_watermarks()}
    ledger.close()
    return result


def _restore_from_snapshot(engine: Engine, payload: dict) -> None:
    from .model import Lease
    engine.leases = {
        lease_id: Lease(**row)
        for lease_id, row in payload["leases"].items()}
    engine.active_index = {
        (lease.pool_id, lease.key): lease.lease_id
        for lease in engine.leases.values() if lease.active}
    engine.key_location = {
        lease.key: lease.pool_id
        for lease in engine.leases.values() if lease.active}


def _replay_watermarks(ledger, pools: dict, scale: int) -> list[dict]:
    marks: list[dict] = []
    for batch in ledger.committed_batches():
        bid = int(batch["batch_id"])
        events = ledger.events_for_batch(bid)
        queues = queue_lengths(events, pools)
        engine = Engine(pools, {})
        engine.rebuild_from_ledger(
            [event for event in ledger.events()
             if event["seq"] <= max((item["seq"] for item in events),
                                    default=0)])
        for mark in compute_watermarks(engine, pools, queues, scale):
            mark["batch_id"] = bid
            marks.append(mark)
    return marks
