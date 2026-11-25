"""Parallel dispatch scheduler enforcing the three scheduling constraints.

Real work is projected onto a bounded :class:`ThreadPoolExecutor` whose worker
threads talk to node subprocesses over loopback TCP.  No artificial sleeps are
used to mimic quotas: a shard that cannot obtain budget or a node slot is
queued with monotonic-millisecond wait start/end timestamps and retried in a
later round.

Invariant at every dispatch decision:

* sum(demand of in-flight shards) <= global budget
* active leases per node <= node.lease_limit
* in-flight shards per node <= node.concurrency
"""

import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Lock
from typing import Dict, List, Optional

from .clock import Clock, MonotonicClock
from .errors import LeaseError
from .freeze import FreezeManager
from .leases import LeaseManager
from .node_client import NodeClient
from .sampling import summary_digest


class NodeProcess:
    def __init__(self, spec):
        self.spec = spec
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        params = {
            "node_id": self.spec.node_id,
            "host": self.spec.host,
            "port": self.spec.port,
            "concurrency": self.spec.concurrency,
        }
        cmd = [sys.executable, "-m", "sharding_diag.node_process", json.dumps(params)]
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(os.path.join(log_dir, f"node-{self.spec.node_id}.log"), "a", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        client = NodeClient(self.spec.host, self.spec.port)
        last_error = None
        for attempt in range(200):
            try:
                if client.ping().get("type") == "pong":
                    return self
            except OSError as exc:
                last_error = exc
            self.process.poll()
            if self.process.returncode is not None:
                raise RuntimeError(f"node {self.spec.node_id} exited early: {last_error}")
            # Startup-handshake backoff only; never used to mimic scheduling.
            time.sleep(0.02)
        raise RuntimeError(f"node {self.spec.node_id} did not become ready: {last_error}")

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            NodeClient(self.spec.host, self.spec.port, timeout_ms=2000).shutdown()
        except OSError:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def kill9(self):
        """Hard kill used by the forced-termination recovery demonstration."""
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)


class Scheduler:
    def __init__(self, declaration, ledger, clock: Optional[Clock] = None,
                 node_processes: Optional[Dict[str, NodeProcess]] = None):
        self.declaration = declaration
        self.ledger = ledger
        self.clock = clock or MonotonicClock()
        self.lease_mgr = LeaseManager(self.clock)
        self.freeze_mgr = FreezeManager(self.clock, declaration.global_spec.freeze_threshold_ms)
        self._nodes: Dict[str, NodeProcess] = node_processes or {}
        self._inflight: Dict[str, dict] = {}
        self._enqueued: Dict[str, int] = {}
        self._wait_records: List[dict] = []
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._pool: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------ lifecycle
    def start_nodes(self, only: Optional[List[str]] = None):
        wanted = set(only) if only else {n.node_id for n in self.declaration.nodes}
        # Drop handles whose subprocess was hard-killed so they can be respawned.
        for node_id in list(self._nodes):
            handle = self._nodes[node_id]
            if handle.process is not None and handle.process.poll() is not None:
                self._nodes.pop(node_id, None)
        for spec in self.declaration.nodes:
            if spec.node_id not in wanted or spec.node_id in self._nodes:
                continue
            process = NodeProcess(spec).start()
            self._nodes[spec.node_id] = process

    def restart_node(self, node_id: str):
        handle = self._nodes.pop(node_id, None)
        if handle is not None:
            handle.kill9()
        spec = next(n for n in self.declaration.nodes if n.node_id == node_id)
        self._nodes[node_id] = NodeProcess(spec).start()
        return self._nodes[node_id]

    def stop_nodes(self):
        for process in self._nodes.values():
            process.stop()
        self._nodes.clear()

    def start_pool(self):
        max_workers = max(n.concurrency for n in self.declaration.nodes)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dispatch")

    def shutdown_pool(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------- helpers
    def _node_for(self, shard_id, node_map) -> Optional[object]:
        eligible = [n for n in self.declaration.nodes if self._node_has_room(n)]
        if not eligible:
            return None
        # Stable ownership preference: lowest node id keeps ownership sticky.
        eligible.sort(key=lambda n: n.node_id)
        return eligible[0]

    def _node_has_room(self, node_spec) -> bool:
        active = self.lease_mgr.node_active_count(node_spec.node_id)
        if active >= node_spec.lease_limit:
            return False
        inflight = sum(1 for job in self._inflight.values() if job["node_id"] == node_spec.node_id)
        if inflight >= node_spec.concurrency:
            return False
        return True

    def _budget_available(self, demand) -> bool:
        return self.lease_mgr.budget_used() + demand <= self.declaration.global_spec.budget

    def _drain_expired(self):
        for event in self.lease_mgr.expire_due():
            self.ledger.record_lease_release(event)
            shard_id = event["shard_id"]
            self.freeze_mgr.refreeze(shard_id, "lease_expiry")
            self._enqueued.pop(shard_id, None)

    # --------------------------------------------------------------- rounds
    def run(self, max_rounds: int = 1000) -> dict:
        return self.run_subset({s.shard_id for s in self.declaration.shards}, max_rounds)

    def run_subset(self, target_shard_ids, max_rounds: int = 1000) -> dict:
        targets = set(target_shard_ids)
        self.freeze_mgr.initialize(s.shard_id for s in self.declaration.shards
                                   if s.shard_id in targets and self.freeze_mgr.state_of(s.shard_id) is None)
        self.start_pool()
        shard_map = self.declaration.shard_map()
        outcomes = self.ledger.outcomes()
        rounds = 0
        try:
            while rounds < max_rounds:
                rounds += 1
                self._drain_inflight()
                self._drain_expired()
                migrations = self.freeze_mgr.evaluate_round(
                    [shard_map[sid] for sid in targets], outcomes
                )
                for migration in migrations:
                    self.ledger.record_freeze_migration(migration)
                if self._all_targets_done(targets, outcomes):
                    break
                self._dispatch_round(targets, outcomes)
                self._drain_inflight(wait_for_one=bool(self._inflight))
                outcomes = self.ledger.outcomes()
        finally:
            self.shutdown_pool()
        return {"rounds": rounds, "outcomes": self.ledger.outcomes(),
                "wait_records": list(self._wait_records)}

    def _all_targets_done(self, targets, outcomes) -> bool:
        return all(sid in outcomes and outcomes[sid]["status"] == "COMMITTED" for sid in targets)

    def _dispatch_round(self, targets, outcomes):
        ready = [sid for sid in self.freeze_mgr.ready_shards()
                 if sid in targets and sid not in outcomes and sid not in self._inflight]
        # Priority desc, then shard id asc: deterministic and independent of dict order.
        ready.sort(key=lambda sid: (-self.declaration.shard_map()[sid].priority, sid))
        for shard_id in ready:
            shard = self.declaration.shard_map()[shard_id]
            if not self._budget_available(shard.demand):
                self._mark_waiting(shard_id, "budget_exhausted")
                continue
            node_spec = self._node_for(shard_id, self.declaration.node_map())
            if node_spec is None:
                self._mark_waiting(shard_id, "node_slots_exhausted")
                continue
            self._dispatch(shard, node_spec)

    def _mark_waiting(self, shard_id, reason):
        if shard_id not in self._enqueued:
            self._enqueued[shard_id] = self.clock.now_ms()
            self._wait_records.append({
                "shard_id": shard_id, "reason": reason,
                "enqueue_ms": self.clock.now_ms(), "dequeue_ms": None,
            })

    def _dispatch(self, shard, node_spec):
        ttl = self.declaration.global_spec.lease_ttl_ms
        lease = self.lease_mgr.grant(node_spec.node_id, shard.shard_id, shard.demand, ttl)
        self.ledger.record_lease_grant(lease, lease.grant_ms)
        wait_start = self._enqueued.pop(shard.shard_id, None)
        if wait_start is not None:
            for record in self._wait_records:
                if record["shard_id"] == shard.shard_id and record["dequeue_ms"] is None:
                    record["dequeue_ms"] = self.clock.now_ms()
                    record["wait_ms"] = record["dequeue_ms"] - wait_start
                    self.ledger.record_queue_wait(record)
        request = {
            "type": "compute",
            "request_id": uuid.uuid4().hex,
            "node_id": node_spec.node_id,
            "shard_id": shard.shard_id,
            "rows": shard.rows,
            "node_seed": node_spec.seed,
            "fixed_point_scale": node_spec.fixed_point_scale,
            "capacity_version": node_spec.capacity_version,
        }
        client = NodeClient(node_spec.host, node_spec.port)
        future = self._pool.submit(client.request, request)
        self._inflight[shard.shard_id] = {
            "future": future, "node_id": node_spec.node_id,
            "lease": lease, "demand": shard.demand,
        }

    def _drain_inflight(self, wait_for_one: bool = False):
        if not self._inflight:
            return
        if wait_for_one:
            # Wait on the real futures (no artificial sleep). The first future
            # in iteration order is enough to unblock the round; concurrent
            # futures are harvested in the same pass once complete.
            for job in list(self._inflight.values()):
                if not job["future"].done():
                    job["future"].result()
                    break
        done_ids = []
        for shard_id, job in list(self._inflight.items()):
            future = job["future"]
            if not future.done():
                continue
            done_ids.append(shard_id)
            try:
                response = future.result()
            except Exception as exc:
                # Node vanished (e.g. SIGKILL): recycle lease and requeue.
                response = {"type": "error", "reason": "connection_lost",
                            "exception": repr(exc)}
            self._handle_response(job, response)
        for shard_id in done_ids:
            self._inflight.pop(shard_id, None)

    def _handle_response(self, job, response):
        shard_id = job["lease"].shard_id
        node_id = job["node_id"]
        lease = job["lease"]
        if response.get("type") != "compute_result":
            # Worker-side failure: release the lease and return to the queue.
            event = self.lease_mgr.release_on_complete(node_id, shard_id)
            if event:
                self.ledger.record_lease_release(event)
            self.freeze_mgr.refreeze(shard_id, "worker_error")
            return
        release_event = self.lease_mgr.release_on_complete(node_id, shard_id)
        if release_event:
            self.ledger.record_lease_release(release_event)
        summary = response["summary"]
        outcome = {
            "shard_id": shard_id,
            "status": "COMMITTED",
            "node_id": node_id,
            "lease_id": lease.lease_id,
            "lease_version": lease.version,
            "capacity_version": response["capacity_version"],
            "summary": summary,
            "summary_digest": summary_digest(summary),
            "fixed_total": summary["total_sum_fp"],
            "demand": job["demand"],
            "node_elapsed_ms": response["elapsed_ms"],
            "at_ms": self.clock.now_ms(),
        }
        self.ledger.record_outcome(outcome)
