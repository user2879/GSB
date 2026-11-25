"""Orchestration engine: scheduling, leases, events, invalidation, recovery."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .errors import DuplicateError, HandshakeError, LeaseError, PortError
from .invalidation import propagate_invalidation
from .lease import Lease, check_renew, grant_lease
from .ledger import Ledger
from .process import NodeProcess
from .protocol import NodeClient
from .samples import (
    encode_batch,
    generate_samples,
    sample_digest,
    summary_canonical,
)
from .util import canonical_json, now_ms, sha256_hex

TERMINAL_STATUSES = {"completed", "cancelled", "failed", "unknown"}


class ShardRuntime:
    def __init__(self, shard):
        self.shard = shard
        self.status = "pending"
        self.attempt = 0
        self.lease = None
        self.wait_start_ms = None
        self.wait_end_ms = None
        self.dispatch_ms = None
        self.worker_future = None
        self.cancelled = False
        self.last_summary = None
        self.last_config_version = None
        self.last_capacity_version = None
        self.last_digest = None
        self.unknown = False


class Engine:
    def __init__(self, spec, ledger_path, max_workers=None, start_nodes=True,
                 enable_renewer=True):
        self.spec = spec
        self.ledger_path = ledger_path
        self.max_workers = max_workers or max(
            1, sum(node.concurrency for node in spec.nodes.values())
        )
        self.start_nodes = start_nodes
        self.enable_renewer = enable_renewer
        self.spec_digest = sha256_hex(
            {
                "nodes": [node.__dict__ for node in spec.nodes.values()],
                "shards": [shard.__dict__ for shard in spec.shards.values()],
                "revisions": [revision.__dict__ for revision in
                              spec.revisions.values()],
                "global_budget": spec.global_budget,
                "lease_ttl_ms": spec.lease_ttl_ms,
            }
        )
        self.ledger = Ledger(ledger_path, spec_digest=self.spec_digest)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.runtimes = {
            shard_id: ShardRuntime(shard)
            for shard_id, shard in sorted(spec.shards.items())
        }
        self.active_leases = {node_id: set() for node_id in spec.node_order}
        self.active_shards = {node_id: set() for node_id in spec.node_order}
        self.resource_in_use = 0
        self.current_revision = min(spec.revisions) if spec.revisions else 1
        self.effective_config = spec.revision_values(self.current_revision)
        self.concurrency_override = {}
        self.capacity_versions = {
            node_id: node.capacity_version
            for node_id, node in spec.nodes.items()
        }
        self.invalidated_order = []
        self.invalidation_steps = {}
        self.event_cursor = 0
        self.node_processes = {}
        self.node_clients = {}
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="sf-worker"
        )
        self._stopping = threading.Event()
        self._scheduler_thread = None
        self._reaper_thread = None
        self._renewer_thread = None
        self.errors = []
        self.unknown_shards = set()
        self.recovered = False

    # ------------------------------------------------------------------ nodes
    def start_node_processes(self):
        if not self.start_nodes:
            for node_id in self.spec.node_order:
                node = self.spec.nodes[node_id]
                self.node_clients[node_id] = NodeClient(
                    node, self.spec.handshake_timeout_ms, self.spec.io_timeout_ms
                )
            return
        for node_id in self.spec.node_order:
            node = self.spec.nodes[node_id]
            process = NodeProcess(
                node,
                self.effective_config,
                self.spec.handshake_timeout_ms,
                self.spec.io_timeout_ms,
                workdir=self.ledger_path + ".nodes",
            )
            process.start()
            self.node_processes[node_id] = process
            self.node_clients[node_id] = NodeClient(
                node, self.spec.handshake_timeout_ms, self.spec.io_timeout_ms
            )

    def stop_node_processes(self):
        for process in self.node_processes.values():
            process.stop()
        self.node_processes.clear()

    def restart_node(self, node_id):
        old = self.node_processes.get(node_id)
        if old is not None:
            old.force_kill()
            self.node_processes.pop(node_id, None)
        node = self.spec.nodes[node_id]
        process = NodeProcess(
            node,
            self.effective_config,
            self.spec.handshake_timeout_ms,
            self.spec.io_timeout_ms,
            workdir=self.ledger_path + ".nodes",
        )
        process.start()
        self.node_processes[node_id] = process
        self.node_clients[node_id] = NodeClient(
            node, self.spec.handshake_timeout_ms, self.spec.io_timeout_ms
        )

    def force_kill_node(self, node_id):
        process = self.node_processes.get(node_id)
        if process is not None:
            process.force_kill()

    # ------------------------------------------------------------- scheduling
    def _node_concurrency(self, node_id):
        return self.concurrency_override.get(
            node_id, self.spec.nodes[node_id].concurrency
        )

    def _is_ready(self, runtime):
        if runtime.status != "pending" or runtime.cancelled:
            return False
        shard = runtime.shard
        for dep_id in shard.depends_on:
            dep = self.runtimes[dep_id]
            if dep.cancelled or dep.status == "blocked":
                runtime.status = "blocked"
                return False
            if dep.status != "completed" or dep.last_summary is None:
                return False
        return True

    def _can_grant(self, runtime):
        node_id = runtime.shard.node_id
        units = runtime.shard.resource_units
        node = self.spec.nodes[node_id]
        if self.resource_in_use + units > self.spec.global_budget:
            return False
        if len(self.active_leases[node_id]) >= node.lease_limit:
            return False
        if len(self.active_shards[node_id]) >= self._node_concurrency(node_id):
            return False
        return True

    def _drain_schedule(self):
        """One deterministic scheduling pass; each shard grants at most once."""
        granted = []
        with self.condition:
            for shard_id in self.spec.topo_order:
                runtime = self.runtimes[shard_id]
                if runtime.wait_start_ms is None and runtime.status == "pending":
                    runtime.wait_start_ms = now_ms()
                if not self._is_ready(runtime):
                    continue
                if not self._can_grant(runtime):
                    continue
                self._grant(runtime)
                granted.append(shard_id)
            self._submit_inflight()
            self.condition.notify_all()
        return granted

    def _grant(self, runtime):
        runtime.attempt += 1
        node_id = runtime.shard.node_id
        granted_ms = now_ms()
        lease = grant_lease(
            runtime.shard.id,
            node_id,
            runtime.attempt,
            granted_ms,
            self.spec.lease_ttl_ms,
        )
        runtime.lease = lease
        runtime.status = "inflight"
        runtime.wait_end_ms = granted_ms
        runtime.dispatch_ms = granted_ms
        self.active_leases[node_id].add(runtime.shard.id)
        self.active_shards[node_id].add(runtime.shard.id)
        self.resource_in_use += runtime.shard.resource_units
        payload = lease.to_dict()
        payload.update(
            {
                "config_version": self.current_revision,
                "capacity_version": self.capacity_versions[node_id],
                "resource_units": runtime.shard.resource_units,
                "wait_start_ms": runtime.wait_start_ms,
                "wait_end_ms": runtime.wait_end_ms,
            }
        )
        self.ledger.record_lease_event("grant", payload)

    def _release(self, runtime, kind="release", extra=None):
        node_id = runtime.shard.node_id
        if runtime.shard.id in self.active_leases[node_id]:
            self.active_leases[node_id].discard(runtime.shard.id)
            self.active_shards[node_id].discard(runtime.shard.id)
            self.resource_in_use -= runtime.shard.resource_units
            if self.resource_in_use < 0:
                self.resource_in_use = 0
        payload = {}
        if runtime.lease is not None:
            payload = runtime.lease.to_dict()
        if extra:
            payload.update(extra)
        if kind:
            self.ledger.record_lease_event(kind, payload)
        runtime.lease = None
        runtime.status = "pending"
        runtime.worker_future = None

    def _scheduler_loop(self):
        while not self._stopping.is_set():
            self._drain_schedule()
            with self.condition:
                self.condition.wait(timeout=0.02)
                if self._all_settled_locked():
                    self.condition.notify_all()

    def _all_settled_locked(self):
        if self.event_cursor < len(self.spec.events):
            return False
        for runtime in self.runtimes.values():
            if runtime.status == "inflight":
                return False
            if runtime.status == "pending" and not runtime.cancelled:
                # A pending shard with a cancelled/blocked dependency settles
                # into blocked on the next scheduling pass; wait only while it
                # still has a chance to become ready.
                if not any(
                    self.runtimes[dep].cancelled
                    or self.runtimes[dep].status == "blocked"
                    for dep in runtime.shard.depends_on
                ):
                    return False
        return True

    # ---------------------------------------------------------------- workers
    def _submit_inflight(self):
        for shard_id in self.spec.shard_order:
            runtime = self.runtimes[shard_id]
            if runtime.status == "inflight" and runtime.worker_future is None:
                runtime.worker_future = self.executor.submit(
                    self._worker, shard_id
                )

    def _worker(self, shard_id):
        runtime = self.runtimes[shard_id]
        shard = runtime.shard
        node = self.spec.nodes[shard.node_id]
        client = self.node_clients[shard.node_id]
        groups, values = generate_samples(node.seed, shard.sample_count)
        digest = sample_digest(groups, values)
        lease_snapshot = runtime.lease
        if lease_snapshot is None:
            return
        try:
            receipt = client.dispatch(
                shard_id,
                self.current_revision,
                lease_snapshot.lease_id,
                lease_snapshot.lease_version,
                encode_batch(groups, values),
                digest,
            )
        except (PortError, HandshakeError) as exc:
            self.errors.append(exc.to_dict())
            with self.condition:
                if runtime.status == "inflight":
                    self._release(runtime, kind="node_error",
                                  extra={"error": exc.code})
                    runtime.worker_future = None
                    self.condition.notify_all()
            return
        with self.condition:
            self._accept_receipt(runtime, receipt)
            self.condition.notify_all()

    def _accept_receipt(self, runtime, receipt):
        if runtime.status != "inflight" or runtime.lease is None:
            # Late receipt after expiry/cancel: ignored and never overwrites.
            self.ledger.record_lease_event(
                "late_receipt_ignored",
                {"shard_id": runtime.shard.id, "attempt": runtime.attempt},
            )
            return
        lease = runtime.lease
        if receipt["lease_version"] != lease.lease_version:
            self._release(runtime, kind="stale_receipt",
                          extra={"receipt_version": receipt["lease_version"]})
            return
        if now_ms() >= lease.expires_ms:
            self._release(runtime, kind="expired_late_receipt",
                          extra={"expires_ms": lease.expires_ms})
            return
        summary = receipt["summary"]
        runtime.status = "completed"
        runtime.last_summary = summary
        runtime.last_config_version = receipt["config_version"]
        runtime.last_capacity_version = receipt["capacity_version"]
        runtime.last_digest = summary["digest"]
        runtime.unknown = False
        self.unknown_shards.discard(runtime.shard.id)
        node_id = runtime.shard.node_id
        outcome = {
            "shard_id": runtime.shard.id,
            "attempt": runtime.attempt,
            "status": "completed",
            "node_id": node_id,
            "config_version": receipt["config_version"],
            "capacity_version": receipt["capacity_version"],
            "resource_units": runtime.shard.resource_units,
            "summary": summary,
            "outcome_digest": self._outcome_digest(runtime, summary),
            "lease": lease.to_dict(),
            "wait_start_ms": runtime.wait_start_ms,
            "wait_end_ms": runtime.wait_end_ms,
            "dispatch_ms": runtime.dispatch_ms,
            "completed_ms": now_ms(),
            "node_elapsed_ms": receipt["elapsed_ms"],
        }
        self.ledger.record_lease_event(
            "complete",
            {
                "shard_id": runtime.shard.id,
                "lease": lease.to_dict(),
                "resource_units": runtime.shard.resource_units,
            },
        )
        self.ledger.record_outcome(outcome, self.event_cursor)
        self.active_leases[node_id].discard(runtime.shard.id)
        self.active_shards[node_id].discard(runtime.shard.id)
        self.resource_in_use -= runtime.shard.resource_units
        runtime.lease = None
        runtime.worker_future = None

    @staticmethod
    def _outcome_digest(runtime, summary):
        return sha256_hex(
            {
                "shard_id": runtime.shard.id,
                "node_id": runtime.shard.node_id,
                "attempt": runtime.attempt,
                "config_version": runtime.last_config_version,
                "capacity_version": runtime.last_capacity_version,
                "resource_units": runtime.shard.resource_units,
                "summary_canonical": summary_canonical(summary),
            }
        )

    # ------------------------------------------------------------- lease care
    def _reaper_loop(self):
        while not self._stopping.is_set():
            with self.condition:
                now = now_ms()
                for shard_id in self.spec.shard_order:
                    runtime = self.runtimes[shard_id]
                    lease = runtime.lease
                    if runtime.status != "inflight" or lease is None:
                        continue
                    if now < lease.expires_ms:
                        continue
                    if runtime.cancelled:
                        self._finalize_cancel(runtime)
                    else:
                        self._release(runtime, kind="expire",
                                      extra={"expires_ms": lease.expires_ms})
                self.condition.notify_all()
            self._stopping.wait(0.01)

    def _renewer_loop(self):
        while not self._stopping.is_set():
            self._stopping.wait(max(0.01, self.spec.lease_ttl_ms / 1000.0 / 3.0))
            with self.condition:
                now = now_ms()
                for shard_id in self.spec.shard_order:
                    runtime = self.runtimes[shard_id]
                    lease = runtime.lease
                    if runtime.status != "inflight" or lease is None:
                        continue
                    # Renew around two thirds of the TTL while still alive.
                    if now - lease.renewed_ms < self.spec.lease_ttl_ms * 2 // 3:
                        continue
                    self._attempt_renew(runtime, now)

    def _attempt_renew(self, runtime, now):
        node = self.spec.nodes[runtime.shard.node_id]
        lease = runtime.lease
        if runtime.status != "inflight" or lease is None:
            return
        expected = lease.lease_version
        try:
            check_renew(lease, expected, node.max_renewals, now,
                        self.spec.lease_ttl_ms)
        except LeaseError as exc:
            self.ledger.record_lease_event(
                "renew_rejected_local",
                {"lease": lease.to_dict(), "error": exc.to_dict()},
            )
            return
        try:
            response = self.node_clients[node.id].renew(
                runtime.shard.id, lease.lease_id, expected
            )
        except (PortError, HandshakeError) as exc:
            lease.renewals -= 1
            lease.lease_version -= 1
            lease.expires_ms = lease.renewed_ms + self.spec.lease_ttl_ms
            self.ledger.record_lease_event(
                "renew_network_failed",
                {"lease": lease.to_dict(), "error": exc.code},
            )
            return
        if response.get("type") != "renew_ack":
            # Roll back the local optimistic change; expiry is unchanged.
            lease.renewals -= 1
            lease.lease_version -= 1
            lease.expires_ms = lease.renewed_ms + self.spec.lease_ttl_ms
            self.ledger.record_lease_event(
                "renew_rejected_node",
                {"lease": lease.to_dict(), "response": response},
            )
            return
        runtime.lease.lease_version = response["lease_version"]
        self.ledger.record_lease_event(
            "renew",
            {"lease": runtime.lease.to_dict(), "old_version": expected},
        )

    # ----------------------------------------------------------------- events
    def process_events(self):
        with self.condition:
            while self.event_cursor < len(self.spec.events):
                event = self.spec.events[self.event_cursor]
                self._apply_event(event)
                self.event_cursor += 1
                self._submit_inflight()
                self.condition.notify_all()

    def _apply_event(self, event):
        if event.type == "dispatch":
            runtime = self.runtimes[event.shard_id]
            if runtime.status == "pending" and not runtime.cancelled:
                if runtime.wait_start_ms is None:
                    runtime.wait_start_ms = now_ms()
        elif event.type == "cancel":
            self._apply_cancel(event)
        elif event.type == "quota_change":
            old = self._node_concurrency(event.node_id)
            self.concurrency_override[event.node_id] = event.concurrency
            self.ledger.record_lease_event(
                "quota_change",
                {"node_id": event.node_id, "old": old,
                 "new": event.concurrency, "event_seq": event.index},
            )
        elif event.type == "config_apply":
            self._apply_revision(event)

    def _apply_cancel(self, event):
        runtime = self.runtimes[event.shard_id]
        if runtime.cancelled or runtime.status == "completed":
            raise DuplicateError(
                "shard %s cannot be cancelled again after terminal state"
                % event.shard_id,
                object_id=event.shard_id,
                field="cancel",
            )
        runtime.cancelled = True
        if runtime.status == "inflight":
            lease = runtime.lease
            try:
                self.node_clients[runtime.shard.node_id].cancel(
                    event.shard_id, lease.lease_id
                )
            except (PortError, HandshakeError):
                pass
            self._finalize_cancel(runtime)
        elif runtime.status == "pending":
            self._finalize_cancel(runtime)

    def _finalize_cancel(self, runtime):
        node_id = runtime.shard.node_id
        attempt = max(1, runtime.attempt)
        lease_payload = runtime.lease.to_dict() if runtime.lease else None
        if runtime.lease is not None:
            self.active_leases[node_id].discard(runtime.shard.id)
            self.active_shards[node_id].discard(runtime.shard.id)
            self.resource_in_use -= runtime.shard.resource_units
            runtime.lease = None
        runtime.status = "cancelled"
        self.ledger.record_lease_event(
            "cancel",
            {"shard_id": runtime.shard.id, "lease": lease_payload},
        )
        self.ledger.record_outcome(
            {
                "shard_id": runtime.shard.id,
                "attempt": attempt,
                "status": "cancelled",
                "node_id": node_id,
                "config_version": self.current_revision,
                "capacity_version": self.capacity_versions[node_id],
                "resource_units": runtime.shard.resource_units,
                "summary": None,
                "outcome_digest": None,
                "lease": lease_payload,
                "wait_start_ms": runtime.wait_start_ms,
                "wait_end_ms": runtime.wait_end_ms,
                "dispatch_ms": runtime.dispatch_ms,
                "completed_ms": now_ms(),
                "node_elapsed_ms": None,
            },
            self.event_cursor,
        )

    def _apply_revision(self, event):
        old_revision = self.current_revision
        new_revision = event.revision
        ordered, reasons = propagate_invalidation(
            self.spec,
            old_revision,
            new_revision,
            old_capacity_versions=dict(self.capacity_versions),
            new_capacity_versions=dict(self.capacity_versions),
            changed_shard_params=set(),
        )
        self.current_revision = new_revision
        self.effective_config = self.spec.revision_values(new_revision)
        if self.start_nodes:
            for node_id in self.spec.node_order:
                self.restart_node(node_id)
        self.ledger.record_lease_event(
            "config_applied",
            {"revision": new_revision, "parent": old_revision},
        )
        self.invalidated_order = ordered
        self.invalidation_steps = reasons
        for shard_id in ordered:
            if self.runtimes[shard_id].cancelled:
                # Cancellation is terminal: later revisions must not re-dispatch.
                continue
            basis = reasons[shard_id]
            self.ledger.record_invalidation(
                new_revision, shard_id,
                basis["reasons"][0]["reason"], basis,
            )
            runtime = self.runtimes[shard_id]
            if runtime.status == "inflight":
                self._release(runtime, kind="invalidated",
                              extra={"revision": new_revision})
            runtime.status = "pending"
            runtime.cancelled = False
            runtime.last_summary = None
            runtime.last_config_version = None
            runtime.last_capacity_version = None
            runtime.last_digest = None
            runtime.unknown = False
            runtime.attempt = 0
            runtime.worker_future = None
            runtime.wait_start_ms = None
            runtime.wait_end_ms = None
            runtime.dispatch_ms = None
        self.ledger.add_snapshot(
            "config_apply:%d" % new_revision, now_ms()
        )

    # --------------------------------------------------------------- run / end
    def run(self):
        self.start_node_processes()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="sf-scheduler", daemon=True
        )
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="sf-reaper", daemon=True
        )
        self._scheduler_thread.start()
        self._reaper_thread.start()
        if self.enable_renewer:
            self._renewer_thread = threading.Thread(
                target=self._renewer_loop, name="sf-renewer", daemon=True
            )
            self._renewer_thread.start()
        self.process_events()
        with self.condition:
            while not self._all_settled_locked():
                self.condition.wait(timeout=0.1)
        self.shutdown()
        self._resolve_blocked_runtimes()
        self._record_blocked_outcomes()
        self.ledger.add_snapshot("run_complete", now_ms())
        return self.snapshot()

    def shutdown(self):
        self._stopping.set()
        with self.condition:
            self.condition.notify_all()
        for thread in (self._scheduler_thread, self._reaper_thread,
                       self._renewer_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.stop_node_processes()

    def _record_blocked_outcomes(self):
        for shard_id in self.spec.shard_order:
            runtime = self.runtimes[shard_id]
            if runtime.status != "blocked":
                continue
            node_id = runtime.shard.node_id
            self.ledger.record_outcome(
                {
                    "shard_id": shard_id,
                    "attempt": max(1, runtime.attempt),
                    "status": "blocked",
                    "node_id": node_id,
                    "config_version": self.current_revision,
                    "capacity_version": self.capacity_versions[node_id],
                    "resource_units": runtime.shard.resource_units,
                    "summary": None,
                    "outcome_digest": None,
                    "lease": None,
                    "wait_start_ms": runtime.wait_start_ms,
                    "wait_end_ms": runtime.wait_end_ms,
                    "dispatch_ms": runtime.dispatch_ms,
                    "completed_ms": now_ms(),
                    "node_elapsed_ms": None,
                },
                self.event_cursor,
            )

    def _resolve_blocked_runtimes(self):
        """Deterministically mark pending shards whose deps cannot complete."""
        changed = True
        while changed:
            changed = False
            for shard_id in self.spec.topo_order:
                runtime = self.runtimes[shard_id]
                if runtime.status != "pending":
                    continue
                for dep_id in runtime.shard.depends_on:
                    dep = self.runtimes[dep_id]
                    if dep.cancelled or dep.status == "blocked":
                        runtime.status = "blocked"
                        changed = True
                        break

    def snapshot(self):
        with self.condition:
            return {
                "format_version": self.ledger.format_version(),
                "content_digest": self.ledger.content_digest(),
                "current_revision": self.current_revision,
                "resource_in_use": self.resource_in_use,
                "global_budget": self.spec.global_budget,
                "nodes": {
                    node_id: {
                        "active_leases": sorted(self.active_leases[node_id]),
                        "active_shards": sorted(self.active_shards[node_id]),
                        "concurrency": self._node_concurrency(node_id),
                        "lease_limit": self.spec.nodes[node_id].lease_limit,
                        "capacity_version": self.capacity_versions[node_id],
                    }
                    for node_id in self.spec.node_order
                },
                "shards": {
                    shard_id: {
                        "status": runtime.status,
                        "attempt": runtime.attempt,
                        "cancelled": runtime.cancelled,
                        "unknown": runtime.unknown,
                        "config_version": runtime.last_config_version,
                        "summary_digest": runtime.last_digest,
                        "wait_start_ms": runtime.wait_start_ms,
                        "wait_end_ms": runtime.wait_end_ms,
                    }
                    for shard_id, runtime in sorted(self.runtimes.items())
                },
                "invalidated_order": list(self.invalidated_order),
                "invalidation_steps": self.invalidation_steps,
                "unknown_shards": sorted(self.unknown_shards),
                "snapshots": self.ledger.fetch_snapshots(),
                "errors": list(self.errors),
            }

    # -------------------------------------------------------------- recovery
    def recover(self):
        """Rebuild the in-memory view solely from committed ledger rows."""
        self.recovered = True
        outcomes = self.ledger.fetch_outcomes()
        latest = {}
        for row in outcomes:
            latest[row["shard_id"]] = row
        for shard_id, row in latest.items():
            runtime = self.runtimes[shard_id]
            runtime.attempt = int(row["attempt"])
            runtime.last_config_version = row["config_version"]
            runtime.last_capacity_version = row["capacity_version"]
            runtime.last_digest = row["outcome_digest"]
            if row["status"] == "completed":
                runtime.status = "completed"
                runtime.last_summary = row["summary"]
            elif row["status"] == "cancelled":
                runtime.status = "cancelled"
                runtime.cancelled = True
            elif row["status"] in {"unknown", "failed", "inflight"}:
                runtime.status = "pending"
                runtime.unknown = True
                self.unknown_shards.add(shard_id)
                # Unknown shards are explicitly required to be recomputed.
                runtime.attempt = int(row["attempt"])
        self.event_cursor = self.ledger.progress_cursor()
        invalidations = self.ledger.fetch_invalidations()
        self.invalidated_order = sorted(
            {row["shard_id"] for row in invalidations},
            key=lambda sid: self.spec.topo_order.index(sid),
        )
        steps = {}
        for row in invalidations:
            steps.setdefault(row["shard_id"], {"reasons": []})
            steps[row["shard_id"]]["reasons"].extend(row["basis"]["reasons"])
        self.invalidation_steps = steps
        return {
            "unknown_shards": sorted(self.unknown_shards),
            "progress_cursor": self.event_cursor,
            "recovered_outcomes": len(latest),
        }

    def rerun_unknown(self):
        if not self.unknown_shards:
            return self.snapshot()
        self.start_node_processes()
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="sf-recovery"
        )
        self._stopping.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="sf-rec-sched", daemon=True
        )
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="sf-rec-reaper", daemon=True
        )
        self._scheduler_thread.start()
        self._reaper_thread.start()
        with self.condition:
            self._submit_inflight()
            while any(
                runtime.status == "inflight"
                or (runtime.unknown and runtime.status == "pending")
                for runtime in self.runtimes.values()
            ):
                self.condition.wait(timeout=0.1)
        self.shutdown()
        self.unknown_shards = {
            sid for sid, rt in self.runtimes.items() if rt.unknown
        }
        self.ledger.add_snapshot("recovery_complete", now_ms())
        return self.snapshot()
