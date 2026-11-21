"""Run engine: orchestrates spec, backends, scheduler, ledger, replay."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

from . import health as hm
from . import paths as pathmod
from . import probe as probemod
from .errors import PortError, SpecError
from .health import Health
from .ledger import Ledger
from .logutil import StructuredLogger
from .replay import replay_verify
from .scheduler import Scheduler
from .spec import load_spec, spec_digest

READY_TIMEOUT_S = 5.0
CRASH_ENV = "LBHC_CRASH_AFTER"  # test hook: os._exit after Nth commit


def _check_ports_free(spec):
    """Every declared port must be bindable -> otherwise E_PORT."""
    for pi, pool in enumerate(spec.pools):
        for bi, b in enumerate(pool.backends):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((b.host, b.port))
            except OSError as exc:
                raise PortError(
                    f"port {b.port} for instance {b.id!r} is occupied: {exc}",
                    path=f"pools[{pi}].backends[{bi}].port",
                    detail={"instance_id": b.id, "port": b.port})
            finally:
                s.close()


def _spawn_backends(spec, logger):
    procs = {}
    for pool in spec.pools:
        for b in pool.backends:
            delay_s = max(pool.timeout_ms * 3 / 1000.0, 0.5)
            cmd = [sys.executable, "-m", "lbhc.backend",
                   "--port", str(b.port), "--behavior", b.behavior,
                   "--body-ok", pool.expect_body,
                   "--delay-s", f"{delay_s:.3f}"]
            procs[b.id] = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            logger.emit("backend_spawned", instance_id=b.id, port=b.port,
                        behavior=b.behavior, pid=procs[b.id].pid)
    # wait until every listener accepts connections
    deadline = time.monotonic() + READY_TIMEOUT_S
    for pool in spec.pools:
        for b in pool.backends:
            while True:
                try:
                    with socket.create_connection((b.host, b.port),
                                                  timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise PortError(
                            f"backend {b.id!r} did not start on port "
                            f"{b.port}",
                            detail={"instance_id": b.id, "port": b.port})
                    time.sleep(0.02)
    return procs


def _stop_backends(procs):
    for p in procs.values():
        try:
            p.terminate()
        except OSError:
            pass
    for p in procs.values():
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()


def _diff_views(old_rows, new_rows):
    """Stable diff between the stale view and the replacement view."""
    old = {r["instance_id"]: r for r in old_rows}
    new = {r["instance_id"]: r for r in new_rows}
    diff = []
    for iid in sorted(set(old) | set(new)):
        o, n = old.get(iid), new.get(iid)
        if o is None:
            diff.append({"instance_id": iid, "change": "added",
                         "new_state": n["state"],
                         "new_reason": n["last_reason"]})
        elif n is None:
            diff.append({"instance_id": iid, "change": "removed",
                         "old_state": o["state"],
                         "old_reason": o["last_reason"]})
        elif (o["state"] != n["state"]
              or o["last_reason"] != n["last_reason"]):
            diff.append({"instance_id": iid, "change": "updated",
                         "old_state": o["state"], "new_state": n["state"],
                         "old_reason": o["last_reason"],
                         "new_reason": n["last_reason"]})
    return diff


def run_engine(spec_path, ledger_path, rounds=4, path="auto",
               report_path=None, logger=None, log_name="run"):
    logger = logger or StructuredLogger(name=log_name)
    spec = load_spec(spec_path)  # E_SPEC on any violation
    logger.emit("spec_loaded", version=spec.version,
                instances=spec.instance_count, pools=spec.pool_count,
                digest=spec_digest(spec))

    ledger = Ledger(ledger_path)
    ledger.verify_chain()  # E_LEDGER if the ledger was tampered with
    stored_version = ledger.current_version()

    old_view = None
    resumed = False
    if stored_version is None:
        ledger.set_meta("spec_version", spec.version)
        ledger.init_view(spec.version, spec)
        logger.emit("ledger_initialised", version=spec.version)
    elif spec.version < stored_version:
        raise SpecError(
            f"spec version {spec.version} is not monotonic: ledger already "
            f"at version {stored_version}", path="version")
    elif spec.version > stored_version:
        old_view = ledger.view(stored_version)
        ledger.set_meta("spec_version", spec.version)
        ledger.init_view(spec.version, spec)  # wholesale replace + stale
        logger.emit("spec_evolved", old_version=stored_version,
                    new_version=spec.version)
    else:
        resumed = True
        voided = ledger.void_inflight(spec.version)
        if voided:
            logger.emit("recovery", voided=[
                {"seq": s, "instance_id": i} for _, s, i in voided])

    _check_ports_free(spec)  # E_PORT
    procs = _spawn_backends(spec, logger)
    crash_after = int(os.environ.get(CRASH_ENV, "0") or "0")
    stats = {"committed": 0}
    try:
        health_map = {}
        if resumed:
            # rebuild in-memory state from the committed ledger view
            for row in ledger.view(spec.version):
                health_map[row["instance_id"]] = Health(
                    state=row["state"],
                    consec_success=row["consec_success"],
                    consec_failure=row["consec_failure"],
                    last_reason=row["last_reason"])
        else:
            for pool in spec.pools:
                for b in pool.backends:
                    health_map[b.id] = hm.initial_health(b.role)

        chosen, params = pathmod.select_path(spec, path)
        logger.emit("path_selected", path=chosen, **params)

        scheduler = Scheduler(
            spec.concurrency_budget, spec.rate_limit_per_sec,
            {p.id: p.concurrency_quota for p in spec.pools})
        seq = ledger.next_seq(spec.version)
        start_round = ledger.max_round(spec.version) + 1 if resumed else 0
        dispatched_rounds = []
        sequence = []

        def probe_fn(task):
            return probemod.run_probe(task["protocol"], task["host"],
                                      task["port"], task["timeout_ms"],
                                      task["expect_body"])

        def on_result(task, outcome, reason, detail):
            pool, backend = spec.backend(task["instance_id"])
            h = health_map[backend.id]
            transition = hm.apply_outcome(
                h, outcome, reason, pool.success_threshold,
                pool.failure_threshold)
            rec = {"version": spec.version, "seq": task["seq"],
                   "round": task["round"], "path": task["path"],
                   "pool_id": task["pool_id"],
                   "instance_id": task["instance_id"],
                   "enqueued_ms": task["enqueued_ms"],
                   "started_ms": task["started_ms"],
                   "finished_ms": task["finished_ms"],
                   "wait_ms": task["wait_ms"],
                   "outcome": outcome, "reason": reason, "detail": detail}
            view_row = {"state": h.state, "consec_success": h.consec_success,
                        "consec_failure": h.consec_failure,
                        "last_reason": h.last_reason}
            ledger.record_probe(rec, [transition] if transition else [],
                                view_row)
            stats["committed"] += 1
            if crash_after and stats["committed"] >= crash_after:
                # simulate a hard kill: no cleanup, no atexit, no report
                os._exit(137)

        for round_no in range(start_round, rounds):
            round_start = time.monotonic()
            due = pathmod.due_instances(chosen, spec, health_map, round_no)
            tasks = []
            for pool, b in due:
                ledger.add_inflight(spec.version, seq, b.id)
                tasks.append({"seq": seq, "round": round_no, "path": chosen,
                              "pool_id": pool.id, "instance_id": b.id,
                              "protocol": pool.protocol, "host": b.host,
                              "port": b.port, "timeout_ms": pool.timeout_ms,
                              "expect_body": pool.expect_body})
                seq += 1
            dispatched = scheduler.run_round(tasks, probe_fn, on_result)
            dispatched_rounds.append({"round": round_no,
                                      "dispatched": dispatched})
            sequence.extend(d["instance_id"] for d in dispatched)
            logger.emit("round_done", round=round_no, path=chosen,
                        initiated=[d["instance_id"] for d in dispatched])
            # fixed cadence: next round starts after the pool interval
            interval_s = min(p.interval_ms for p in spec.pools) / 1000.0
            elapsed = time.monotonic() - round_start
            if round_no < rounds - 1 and elapsed < interval_s:
                time.sleep(interval_s - elapsed)

        replay_result = replay_verify(ledger, spec)  # E_REPLAY on mismatch
        view = ledger.view(spec.version)
        transitions = ledger.transitions(spec.version)
        # stable diff between the retired view and its replacement
        stale_diff = (_diff_views(old_view, view)
                      if old_view is not None else [])
        if stale_diff:
            logger.emit("view_replaced", changes=len(stale_diff))
        waits = [d["wait_ms"] for r in dispatched_rounds
                 for d in r["dispatched"]]
        wait_stats = ({} if not waits else {
            "count": len(waits),
            "mean_ms": round(float(np.mean(waits)), 3),
            "p50_ms": round(float(np.percentile(waits, 50)), 3),
            "p95_ms": round(float(np.percentile(waits, 95)), 3),
            "max_ms": round(float(np.max(waits)), 3),
        })
        report = {
            "spec_version": spec.version,
            "spec_digest": spec_digest(spec),
            "path": chosen,
            "path_params": params,
            "resumed": resumed,
            "rounds": dispatched_rounds,
            "initiated_sequence": sequence,
            "queue_wait_stats": wait_stats,
            "budget": {
                "global_budget": spec.concurrency_budget,
                "rate_limit_per_sec": spec.rate_limit_per_sec,
                "peak_global_inflight": scheduler.peak_global,
                "peak_pool_inflight": scheduler.peak_pool,
                "pool_quotas": {p.id: p.concurrency_quota
                                for p in spec.pools},
            },
            "transitions": transitions,
            "final_view": view,
            "stale_diff": stale_diff,
            "replay": replay_result,
        }
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)),
                        exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
        logger.emit("run_complete", version=spec.version, path=chosen,
                    probes=replay_result["probes_replayed"],
                    transitions=replay_result["transitions_verified"])
        return report
    finally:
        _stop_backends(procs)
        ledger.close()
