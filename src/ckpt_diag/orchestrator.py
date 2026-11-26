"""End-to-end orchestration: shard subprocesses, scheduler, arbiter, ledger."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .clock import MonotonicClock
from .compute import compute_batch
from .ledger import Ledger
from .protocol import request
from .quota import QuotaArbiter
from .scheduler import BatchTicket, Scheduler
from .spec import EventSpec, Spec, load_spec


class ShardProcess:
    def __init__(self, spec_path: Path, shard_id: str, port: int, ttl_ms: int) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ckpt_diag.shard_process",
             "--spec", str(spec_path), "--shard-id", shard_id,
             "--ttl-ms", str(ttl_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**os.environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            text=True,
        )
        self.shard_id = shard_id
        self.port = port
        self._call_lock = threading.Lock()
        self._wait_ready()

    def _wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read()
                    raise RuntimeError(f"shard {self.shard_id} exited early: {err}")
                continue
            payload = json.loads(line)
            if payload.get("ready"):
                return
        raise TimeoutError(f"shard {self.shard_id} did not become ready")

    def call(self, message: dict) -> dict:
        # Serialize requests to one shard subprocess; the shard is a single
        # deterministic lease authority and compute engine.
        with self._call_lock:
            return request("127.0.0.1", self.port, message)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Orchestrator:
    def __init__(self, spec: Spec, spec_path: Path, ledger: Ledger,
                 ttl_ms: int = 5000) -> None:
        self.spec = spec
        self.spec_path = spec_path
        self.ledger = ledger
        self.ttl_ms = ttl_ms
        self.clock = MonotonicClock()
        self.processes: dict[str, ShardProcess] = {}
        self.arbiter = QuotaArbiter(
            spec.global_budget,
            {shard.shard_id: shard.lease_cap for shard in spec.shards})
        self.scheduler = Scheduler(
            global_budget=spec.global_budget,
            shard_limits={shard.shard_id: (shard.lease_cap, shard.concurrency_quota)
                          for shard in spec.shards},
            max_workers=max(4, len(spec.shards) * 2),
            clock=self.clock.now_ms)
        self.results: dict[str, dict] = {}
        self.batch_leases: dict[str, dict] = {}
        self.batch_units: dict[str, tuple[int, ...]] = {}
        self.batch_inputs: dict[str, dict[str, dict]] = {}

    def start_shards(self) -> None:
        for shard in self.spec.shards:
            self.processes[shard.shard_id] = ShardProcess(
                self.spec_path, shard.shard_id, shard.port, self.ttl_ms)

    def stop_shards(self) -> None:
        for process in self.processes.values():
            process.stop()
        self.processes.clear()

    def _run_dispatch(self, event: EventSpec) -> dict:
        shard = self.spec.shard(event.shard_id)
        ticket = BatchTicket(event.batch_id, event.shard_id, event.cost_units)

        def job(t: BatchTicket) -> dict:
            proc = self.processes[event.shard_id]
            issue = proc.call({"action": "issue_lease", "batch_id": event.batch_id})
            if not issue.get("ok"):
                raise RuntimeError(issue["error"])
            lease = issue["lease"]
            units = self.arbiter.grant(
                event_seq=event.seq, shard_id=event.shard_id,
                lease_id=lease["lease_id"], lease_version=lease["lease_version"],
                param_version=event.param_version, cost_units=event.cost_units)
            run = proc.call({
                "action": "run_batch",
                "lease_id": lease["lease_id"],
                "batch_id": event.batch_id,
                "param_version": event.param_version,
                "sample_ids": list(event.sample_ids),
                "scope_layers": list(event.scope_layers),
            })
            if not run.get("ok"):
                raise RuntimeError(run["error"])
            result = run["result"]
            result["elapsed_ms"] = run["elapsed_ms"]
            result["lease_id"] = lease["lease_id"]
            result["cost_units"] = event.cost_units
            # Keep the lease alive after the batch so explicit lease_expire /
            # rollback events can arbitrate the same units (unique ownership).
            self.batch_leases[event.batch_id] = {
                **lease, "lease_version": run["lease_version"],
                "param_version": event.param_version, "state": "active"}
            self.batch_units[event.batch_id] = units
            ownership = [self.arbiter.state.units[index].__dict__
                         for index in range(self.arbiter.state.budget)]
            self.ledger.commit_batch(
                seq=event.seq, batch=result,
                lease={**lease, "param_version": event.param_version,
                       "state": "released"},
                ownership_records=ownership,
                decisions=[d.to_dict() for d in self.arbiter.state.decisions
                           if d.event_seq == event.seq],
                checkpoint_states=self._checkpoint_snapshot())
            self.batch_inputs.setdefault(event.shard_id, {})[event.batch_id] = {
                "param_version": event.param_version,
                "sample_ids": list(event.sample_ids),
                "scope_layers": list(event.scope_layers),
            }
            return result

        future = self.scheduler.submit(ticket, job)
        return future.result()

    def _checkpoint_snapshot(self) -> list[dict]:
        return [{"checkpoint_id": cp.checkpoint_id, "shard_id": cp.shard_id,
                 "generation": cp.generation, "param_version": cp.param_version,
                 "loss_baseline": cp.loss_baseline, "state": "frozen"}
                for cp in self.spec.checkpoints]

    def run_event(self, event: EventSpec) -> dict:
        if event.kind == "dispatch":
            result = self._run_dispatch(event)
            self.results[event.batch_id] = result
            return {"event_seq": event.seq, "kind": event.kind, "result": result}
        if event.kind == "quota_change":
            self.arbiter.change_quota(event_seq=event.seq, shard_id=event.shard_id,
                                      new_lease_cap=event.lease_cap)
            self.scheduler.update_quota(event.shard_id, event.lease_cap,
                                        event.concurrency_quota)
            return {"event_seq": event.seq, "kind": event.kind}
        if event.kind == "rollback":
            return self._handle_rollback(event)
        if event.kind == "param_release":
            return self._handle_param_release(event)
        if event.kind == "lease_expire":
            return self._handle_expire(event)
        return {"event_seq": event.seq, "kind": event.kind, "note": "no-op"}

    def _handle_rollback(self, event: EventSpec) -> dict:
        units = self.batch_units.get(event.batch_id, ())
        lease = self.batch_leases.get(event.batch_id)
        moved = self.arbiter.void_param_version(
            event_seq=event.seq, shard_id=event.shard_id,
            param_version=event.param_version,
            lease_id=lease["lease_id"] if lease else None,
            lease_version=lease["lease_version"] if lease else None,
            unit_indices=units)
        return {"event_seq": event.seq, "kind": event.kind,
                "voided_units": list(moved)}

    def _handle_param_release(self, event: EventSpec) -> dict:
        voided = self.arbiter.state.void_units.get(event.param_version, set())
        freed = self.arbiter.reclaim_to_free(
            event_seq=event.seq, unit_indices=tuple(sorted(voided)),
            basis="param_version_recycled_to_free")
        return {"event_seq": event.seq, "kind": event.kind, "freed_units": list(freed)}

    def _handle_expire(self, event: EventSpec) -> dict:
        lease = self.batch_leases.get(event.batch_id)
        if lease is None:
            return {"event_seq": event.seq, "kind": event.kind, "note": "no active lease"}
        active = self.arbiter.state.active_lease_units.get(lease["lease_id"])
        if not active:
            return {"event_seq": event.seq, "kind": event.kind,
                    "note": "units already released/voided"}
        expired = self.arbiter.expire_lease(
            event_seq=event.seq, lease_id=lease["lease_id"],
            lease_version=lease["lease_version"], param_version=event.param_version)
        return {"event_seq": event.seq, "kind": event.kind,
                "expired_units": list(expired)}

    def run_all(self) -> list[dict]:
        summaries: list[dict] = []
        for event in self.spec.events:
            summaries.append(self.run_event(event))
        return summaries
