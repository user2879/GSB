"""Run orchestration: instance subprocesses, probing, arbitration,
ledger persistence, snapshots and replay verification."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from . import ledger as ledger_mod
from .arbiter import run_arbitration
from .errors import PortError, ReplayError
from .probe import run_probe_rounds
from .propagation import affected_for_transition
from .slog import StructuredLogger, NULL_LOGGER
from .snapshot import build_snapshot, write_snapshot
from .spec import Spec

READY_TIMEOUT_S = 5.0


def run_state_path(db_path: str) -> str:
    return db_path + ".run_state.json"


def _write_run_state(db_path: str, fleet: "InstanceFleet") -> None:
    pids = [p.proc.pid for p in fleet.processes if p.proc is not None]
    with open(run_state_path(db_path), "w", encoding="utf-8") as fh:
        json.dump({"pids": pids, "db": db_path}, fh)


def cleanup_orphans(db_path: str, logger=NULL_LOGGER) -> list:
    """Terminate instance child processes recorded by a killed run."""
    path = run_state_path(db_path)
    killed = []
    if not os.path.exists(path):
        return killed
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError):
        state = {}
    for pid in state.get("pids", []):
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            pass
    try:
        os.remove(path)
    except OSError:
        pass
    if killed:
        logger.log("orphan_cleanup", pids=killed)
    return killed


class InstanceProcess:
    """One simulated service instance child process."""

    def __init__(self, instance, logger: StructuredLogger) -> None:
        self.instance = instance
        self.logger = logger
        self.proc = None

    def start(self) -> None:
        inst = self.instance
        cmd = [
            sys.executable, "-m", "quota_workbench.instance_server",
            "--instance-id", inst.instance_id,
            "--host", inst.host,
            "--port", str(inst.port),
            "--behavior", inst.behavior,
            "--delay-ms", str(inst.delay_ms),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_package_root(),
        )
        first_line = _readline_with_timeout(self.proc, READY_TIMEOUT_S)
        if first_line is None or not (
            first_line.startswith("READY ") or first_line.startswith("NOLISTEN")
        ):
            stderr = ""
            try:
                _, stderr = self.proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.logger.log("instance_start_failed", code="E_PORT",
                            instance=inst.instance_id, port=inst.port,
                            stderr=stderr.strip())
            raise PortError(
                f"instance {inst.instance_id!r} failed to bind {inst.host}:{inst.port}: {stderr.strip() or 'no READY handshake'}",
                field="instances[].port",
                details={"instance_id": inst.instance_id, "port": inst.port},
            )
        self.logger.log("instance_started", instance=inst.instance_id, port=inst.port,
                        behavior=inst.behavior, pid=self.proc.pid)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _package_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _readline_with_timeout(proc: subprocess.Popen, timeout_s: float):
    q: queue.Queue = queue.Queue()

    def reader() -> None:
        line = proc.stdout.readline()
        q.put(line.rstrip("\n") if line else None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        return q.get(timeout=timeout_s)
    except queue.Empty:
        return None


class InstanceFleet:
    def __init__(self, spec: Spec, logger: StructuredLogger) -> None:
        self.spec = spec
        self.logger = logger
        self.processes = []

    def __enter__(self) -> "InstanceFleet":
        try:
            for iid in sorted(self.spec.instances):
                proc = InstanceProcess(self.spec.instances[iid], self.logger)
                proc.start()
                self.processes.append(proc)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        for proc in self.processes:
            proc.stop()
        self.processes = []


def execute_run(spec: Spec, requests: list, db_path: str, snapshot_dir: str,
                probe_rounds: int, batch_size: int = 25,
                crash_after_batch=None, logger: StructuredLogger = NULL_LOGGER) -> dict:
    """Full run: start instances -> probe -> persist transitions ->
    arbitrate with batched ledger commits -> snapshot."""
    session = ledger_mod.LedgerSession(
        db_path, spec, batch_size=batch_size,
        crash_after_batch=crash_after_batch, logger=logger)
    try:
        with InstanceFleet(spec, logger) as fleet:
            _write_run_state(db_path, fleet)
            transitions = run_probe_rounds(spec, probe_rounds, logger)
        session.commit_probe_transitions(transitions)
        for t in transitions:
            affected, steps = affected_for_transition(spec, requests, t.instance_id)
            session.record_propagation(
                change=f"health:{t.instance_id}",
                affected=affected,
                basis=[{"node": n, "via": v} for n, v in steps],
            )
        session.start_arbitration()
        result = run_arbitration(spec, requests, transitions, sink=session)
        session.finalize_run()
    finally:
        session.close()

    try:
        os.remove(run_state_path(db_path))
    except OSError:
        pass
    quota_final = {
        tid: {
            "share": spec.tenants[tid].share,
            "consumed": result.tenant_consumed[tid],
            "remaining": spec.tenants[tid].share - result.tenant_consumed[tid],
        }
        for tid in sorted(spec.tenants)
    }
    committed = ledger_mod.load_committed(db_path)
    snapshot = build_snapshot(
        spec, quota_final,
        cursor=len(result.finalize_order),
        content_digest=committed["meta"]["content_digest"],
    )
    snapshot_path = write_snapshot(snapshot, snapshot_dir)
    logger.log("run_complete", requests=len(result.finalize_order),
               transitions=len(transitions), snapshot=snapshot_path)
    return {
        "result": result,
        "snapshot_path": snapshot_path,
        "content_digest": committed["meta"]["content_digest"],
    }


def replay_verify(spec: Spec, requests: list, db_path: str, probe_rounds: int,
                  logger: StructuredLogger = NULL_LOGGER) -> dict:
    """Re-execute probing and arbitration from the declaration, then compare
    every request outcome, every health transition and the final quota
    state against the committed ledger. The first mismatch is reported as
    E_REPLAY with the request id / tenant / field."""
    committed = ledger_mod.load_committed(db_path)
    with InstanceFleet(spec, logger):
        transitions = run_probe_rounds(spec, probe_rounds, logger)
    result = run_arbitration(spec, requests, transitions, sink=None)

    committed_transitions = [tuple(row) for row in committed["transitions"]]
    replayed_transitions = [t.logical_key() for t in transitions]
    if committed_transitions != replayed_transitions:
        for idx, (have, want) in enumerate(zip(committed_transitions, replayed_transitions)):
            if have != want:
                raise ReplayError(
                    f"transition #{idx} differs: ledger={have} replay={want}",
                    field="transitions",
                    details={"index": idx, "ledger": have, "replay": want},
                )
        raise ReplayError(
            f"transition count differs: ledger={len(committed_transitions)} replay={len(replayed_transitions)}",
            field="transitions",
        )

    committed_requests = {row[0]: row for row in committed["requests"]}
    for request_id in sorted(committed_requests):
        row = committed_requests[request_id]
        outcome = result.outcomes.get(request_id)
        if outcome is None:
            raise ReplayError(f"request {request_id} missing in replay",
                              field="requests", details={"request_id": request_id})
        checks = [
            ("outcome", row[5], outcome.outcome),
            ("basis", row[6], outcome.basis_blob()),
            ("wait_start_ms", row[7], _neg_if_none(outcome.wait_start_ms)),
            ("wait_end_ms", row[8], _neg_if_none(outcome.wait_end_ms)),
            ("grant_ms", row[9], _neg_if_none(outcome.grant_ms)),
            ("done_ms", row[10], _neg_if_none(outcome.done_ms)),
        ]
        for field_name, stored, actual in checks:
            if stored != actual:
                raise ReplayError(
                    f"request {request_id} field {field_name} differs: ledger={stored!r} replay={actual!r}",
                    field=f"requests.{field_name}",
                    details={"request_id": request_id, "tenant_id": row[1],
                             "field": field_name, "ledger": stored, "replay": actual},
                )

    committed_tenants = {row[0]: row for row in committed["tenants"]}
    for tid in sorted(committed_tenants):
        stored_consumed = committed_tenants[tid][2]
        actual = result.tenant_consumed.get(tid, 0)
        if stored_consumed != actual:
            raise ReplayError(
                f"tenant {tid} final consumed differs: ledger={stored_consumed} replay={actual}",
                field="tenants.consumed",
                details={"tenant_id": tid, "ledger": stored_consumed, "replay": actual},
            )
    logger.log("replay_ok", requests=len(committed_requests),
               transitions=len(committed_transitions))
    return {"requests": len(committed_requests), "transitions": len(committed_transitions)}


def _neg_if_none(value: int):
    return None if value is None or value < 0 else value
