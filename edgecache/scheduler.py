"""Concurrent lease scheduler backed by real node subprocesses."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .content import digest_for, manifest_digest
from .errors import EdgeError, HandshakeTimeout, LeaseError, NodeError, PortError
from .ledger import Ledger
from .model import Event, Manifest, NodeSpec
from .protocol import NodeClient, monotonic_ms


@dataclass
class PendingApplication:
    queue_id: str
    event_seq: int
    node_id: str
    content_key: str
    shard_index: int
    byte_count: int
    quota_version: int
    attempt: int
    wait_start_ms: int


@dataclass
class LeaseRuntime:
    lease_id: str
    queue_id: str
    batch_id: str
    node_id: str
    content_key: str
    shard_index: int
    slot_index: int
    byte_count: int
    granted_ms: int
    expires_ms: int
    lease_version: int
    quota_version: int
    digest: str
    checksum: int
    status: str = "RESERVED"
    attempt: int = 0

    def ledger_payload(self, event_seq: int) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "queue_id": self.queue_id,
            "batch_id": self.batch_id,
            "node_id": self.node_id,
            "content_key": self.content_key,
            "shard_index": self.shard_index,
            "slot_index": self.slot_index,
            "bytes": self.byte_count,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "lease_version": self.lease_version,
            "quota_version": self.quota_version,
            "digest": self.digest,
            "checksum": self.checksum,
            "status": self.status,
            "event_seq": event_seq,
            "attempt": self.attempt,
        }


@dataclass
class BatchRuntime:
    batch_id: str
    event_seq: int
    node_id: str
    quota_version: int
    attempt: int
    request: dict[str, Any]
    leases: list[LeaseRuntime] = field(default_factory=list)
    futures: list[Future[Any]] = field(default_factory=list)
    pending_count: int = 0
    finalized: bool = False
    force_unknown: bool = False
    failure_reason: str | None = None
    responses: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkerRuntime:
    node: NodeSpec
    client: NodeClient
    process: subprocess.Popen[str]


class Coordinator:
    def __init__(
        self,
        manifest: Manifest,
        db_path: str | Path,
        *,
        sample_path: str | Path | None = None,
        crash_node: str | None = None,
        crash_after_grants: int = 0,
        retry_unknown: bool = False,
    ) -> None:
        self.manifest = manifest
        self.manifest_digest = manifest_digest(manifest)
        self.db_path = Path(db_path)
        self.sample_path = Path(sample_path or manifest.source_path or "")
        if not self.sample_path:
            raise ValueError("sample_path is required to launch node subprocesses")
        self.crash_node = crash_node
        self.crash_after_grants = crash_after_grants
        self.retry_unknown = retry_unknown
        self.lock = threading.RLock()
        self.ledger: Ledger | None = None
        self.workers: dict[str, WorkerRuntime] = {}
        self.executor: ThreadPoolExecutor | None = None
        self.sweeper: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active: dict[str, LeaseRuntime] = {}
        self.pending: dict[str, PendingApplication] = {}
        self.open_batches: dict[str, BatchRuntime] = {}
        self.current_quota_version = manifest.initial_quota_version
        self.active_nodes = {node.node_id for node in manifest.nodes}
        self.processed_seq = 0
        self.batch_counter = 0
        self.grant_count = 0
        self.crash_done = False
        self.crashed_nodes: set[str] = set()
        self.unknown_requests: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.worker_errors: list[dict[str, Any]] = []
        self.started = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self.lock:
            if self.started:
                return
            self.ledger = Ledger(self.db_path, self.manifest)
            state = self.ledger.state()
            self.current_quota_version = state["current_quota_version"]
            self.active_nodes = set(state["active_nodes"])
            self.processed_seq = state["progress_cursor"]
            self._load_runtime_state_locked()
            if state["requires_rerun"] and not self.unknown_requests and not self.pending:
                self.ledger.mark_rerun_resolved(self.processed_seq)
            for node in self.manifest.nodes:
                self._assert_port_available(node)
            self.executor = ThreadPoolExecutor(
                max_workers=max(1, self.manifest.config.max_workers),
                thread_name_prefix="edgecache-dispatch",
            )
            for node in self.manifest.nodes:
                self.workers[node.node_id] = self._spawn_worker(node)
            self._restore_workers_locked()
            self.sweeper = threading.Thread(target=self._sweeper_loop, name="edgecache-expiry", daemon=True)
            self.sweeper.start()
            self.started = True

    def _load_runtime_state_locked(self) -> None:
        assert self.ledger is not None
        for row in self.ledger.active_lease_rows():
            lease = LeaseRuntime(
                lease_id=row["lease_id"],
                queue_id=row["queue_id"] or "",
                batch_id=row["batch_id"] or "",
                node_id=row["node_id"],
                content_key=row["content_key"],
                shard_index=int(row["shard_index"]),
                slot_index=int(row["slot_index"]),
                byte_count=int(row["bytes"]),
                granted_ms=int(row["granted_ms"]),
                expires_ms=int(row["expires_ms"]),
                lease_version=int(row["lease_version"]),
                quota_version=int(row["quota_version"]),
                digest=row["digest"],
                checksum=int(row["checksum"]),
                status=row["status"],
                attempt=int(row["attempt"]),
            )
            self.active[lease.lease_id] = lease
        for row in self.ledger.waiting_queues():
            content = self.manifest.content_map.get(row["content_key"])
            if content is None or row["node_id"] not in self.manifest.node_map:
                continue
            pending = PendingApplication(
                queue_id=row["queue_id"],
                event_seq=int(row["event_seq"]),
                node_id=row["node_id"],
                content_key=row["content_key"],
                shard_index=int(row["shard_index"]),
                byte_count=content.shard_bytes,
                quota_version=int(row["quota_version"]),
                attempt=int(row["attempt"]),
                wait_start_ms=int(row["wait_start_ms"]),
            )
            self.pending[pending.queue_id] = pending
        self.unknown_requests = self.ledger.unresolved_unknown_requests()

    def _assert_port_available(self, node: NodeSpec) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name != "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((node.host, node.port))
        except OSError as exc:
            raise PortError(
                f"loopback port {node.port} is unavailable for {node.node_id}: {exc}", node_id=node.node_id
            ) from exc
        finally:
            probe.close()

    def _spawn_worker(self, node: NodeSpec) -> WorkerRuntime:
        command = [
            sys.executable,
            "-m",
            "edgecache.worker",
            "--sample",
            str(self.sample_path),
            "--node-id",
            node.node_id,
        ]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        client = NodeClient(node, self.manifest.config)
        deadline = monotonic_ms() + self.manifest.config.handshake_timeout_ms
        last_error: Exception | None = None
        while monotonic_ms() <= deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                if "E_PORT" in stderr:
                    raise PortError(f"node {node.node_id} could not bind port {node.port}: {stderr.strip()}", node_id=node.node_id)
                raise NodeError(f"node {node.node_id} exited during startup: {stderr.strip()}", node_id=node.node_id)
            try:
                hello = client.hello(self.current_quota_version)
                if hello.get("manifest_digest") != self.manifest_digest:
                    raise NodeError(
                        "node loaded a different manifest digest",
                        node_id=node.node_id,
                    )
                if self.current_quota_version != node.quota_version:
                    client.set_quota(self.current_quota_version)
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                return WorkerRuntime(node=node, client=client, process=process)
            except HandshakeTimeout as exc:
                last_error = exc
                time.sleep(0.02)
            except NodeError as exc:
                if "different manifest digest" in exc.message:
                    if process.poll() is None:
                        process.kill()
                    raise
                last_error = exc
                time.sleep(0.02)
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        raise HandshakeTimeout(
            f"node {node.node_id} handshake timed out: {last_error or 'no response'}", node_id=node.node_id
        )

    def _restore_workers_locked(self) -> None:
        for node_id, runtime in self.workers.items():
            leases = [lease for lease in self.active.values() if lease.node_id == node_id]
            if not leases:
                continue
            try:
                runtime.client.restore(
                    [
                        {"lease_id": lease.lease_id, "lease_version": lease.lease_version}
                        for lease in leases
                    ],
                    self.current_quota_version,
                )
            except EdgeError as exc:
                self.worker_errors.append(exc.to_dict())

    def _stop_runtime(self) -> None:
        self.stop_event.set()
        if self.sweeper is not None and self.sweeper.is_alive():
            self.sweeper.join(timeout=2.0)
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
        for runtime in list(self.workers.values()):
            process = runtime.process
            if process.poll() is None:
                try:
                    runtime.client.shutdown(self.current_quota_version)
                except EdgeError:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.workers.clear()
        if self.ledger is not None:
            self.ledger.close()

    def run(self) -> dict[str, Any]:
        try:
            self.start()
        except Exception:
            self._stop_runtime()
            raise
        try:
            resume_after = self.processed_seq
            for event in self.manifest.events:
                if event.seq <= resume_after:
                    continue
                if event.delay_ms:
                    time.sleep(event.delay_ms / 1000.0)
                self.process_event(event)
            self._drain_until_settled()
            if self.retry_unknown and self.unknown_requests:
                self._retry_unknown()
                self._drain_until_settled()
            assert self.ledger is not None
            report = self.ledger.report()
            report["rejections"] = sorted(self.rejections, key=lambda item: (item.get("event_seq", 0), item.get("lease_id") or ""))
            report["worker_errors"] = self.worker_errors
            report["unknown_requests"] = self.unknown_requests
            report["queued_after_drain"] = self.ledger.queued()
            report["status"] = (
                "OK"
                if not self.ledger.queued() and not self.ledger.state()["requires_rerun"]
                else "PARTIAL"
            )
            return report
        finally:
            self._stop_runtime()

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------
    def process_event(self, event: Event) -> None:
        with self.lock:
            self.processed_seq = max(self.processed_seq, event.seq)
        if event.type == "apply":
            self._handle_apply(event)
        elif event.type == "renew":
            self._handle_renew(event)
        elif event.type == "release":
            self._handle_release(event)
        elif event.type == "quota":
            self._handle_quota(event)

    def _handle_apply(self, event: Event) -> None:
        assert event.node_id is not None and event.content_key is not None
        assert self.ledger is not None
        content = self.manifest.content_map[event.content_key]
        now = monotonic_ms()
        with self.lock:
            for shard_index in event.shards:
                queue_id = self._queue_id(event.seq, shard_index, 0)
                pending = PendingApplication(
                    queue_id=queue_id,
                    event_seq=event.seq,
                    node_id=event.node_id,
                    content_key=event.content_key,
                    shard_index=shard_index,
                    byte_count=content.shard_bytes,
                    quota_version=self.current_quota_version,
                    attempt=0,
                    wait_start_ms=now,
                )
                self.pending[queue_id] = pending
                self.ledger.record_queue_start(
                    queue_id,
                    event.seq,
                    event.node_id,
                    event.content_key,
                    shard_index,
                    now,
                    0,
                    self.current_quota_version,
                )
            self._expire_due_locked(monotonic_ms(), event.seq)
            futures = self._pump_locked(monotonic_ms(), event.seq)
        self._wait_futures(futures)

    def _handle_renew(self, event: Event) -> None:
        assert event.lease_id is not None and event.expected_version is not None
        assert self.ledger is not None
        rejected: str | None = None
        with self.lock:
            lease = self.active.get(event.lease_id)
            now = monotonic_ms()
            if lease is None or lease.status != "ACTIVE":
                rejected = "lease is not active"
            elif now >= lease.expires_ms:
                self._claim_locked(lease, "EXPIRED", "monotonic_deadline<=now; expiry_precedes_quota", event.seq, now)
                rejected = "lease expired before renewal"
            else:
                runtime = self.workers.get(lease.node_id)
                if runtime is None or runtime.process.poll() is not None:
                    rejected = "node process is unavailable"
                else:
                    try:
                        response = runtime.client.renew(lease.lease_id, event.expected_version, self.current_quota_version)
                        new_version = int(response.get("lease_version", 0))
                        if new_version != event.expected_version + 1:
                            rejected = "node returned a non-monotonic lease version"
                        else:
                            extend_ms = event.extend_ms or self.manifest.config.lease_ttl_ms
                            new_expiry = now + extend_ms
                            old_expiry = lease.expires_ms
                            self.ledger.record_renew(
                                lease.lease_id,
                                event.seq,
                                lease.lease_version,
                                new_version,
                                old_expiry,
                                new_expiry,
                            )
                            lease.lease_version = new_version
                            lease.expires_ms = new_expiry
                    except LeaseError as exc:
                        rejected = exc.message
                    except EdgeError as exc:
                        rejected = exc.message
                        self.worker_errors.append(exc.to_dict())
            if rejected is not None:
                self.ledger.record_reject(event.lease_id, event.seq, lease.node_id if lease else None, rejected)
                self.rejections.append(
                    {"code": "E_LEASE", "event_seq": event.seq, "lease_id": event.lease_id, "reason": rejected}
                )
            futures = self._pump_locked(now, event.seq)
        self._wait_futures(futures)

    def _handle_release(self, event: Event) -> None:
        assert event.lease_id is not None
        assert self.ledger is not None
        with self.lock:
            lease = self.active.get(event.lease_id)
            now = monotonic_ms()
            reason: str | None = None
            if lease is None or lease.status != "ACTIVE":
                reason = "lease is not active"
            elif now >= lease.expires_ms:
                self._claim_locked(lease, "EXPIRED", "monotonic_deadline<=now; expiry_precedes_quota", event.seq, now)
                reason = "lease expired before release"
            elif event.expected_version is not None and event.expected_version != lease.lease_version:
                reason = f"lease version mismatch: expected {event.expected_version}, current {lease.lease_version}"
            else:
                runtime = self.workers.get(lease.node_id)
                if runtime is not None and runtime.process.poll() is None:
                    try:
                        runtime.client.release(lease.lease_id, self.current_quota_version)
                    except EdgeError as exc:
                        self.worker_errors.append(exc.to_dict())
                self._claim_locked(lease, "RELEASED", "explicit_release", event.seq, now)
            if reason is not None:
                self.ledger.record_reject(
                    event.lease_id, event.seq, lease.node_id if lease else None, reason
                )
                self.rejections.append(
                    {"code": "E_LEASE", "event_seq": event.seq, "lease_id": event.lease_id, "reason": reason}
                )
            futures = self._pump_locked(now, event.seq)
        self._wait_futures(futures)

    def _handle_quota(self, event: Event) -> None:
        assert event.quota_version is not None and event.node_ids
        assert self.ledger is not None
        revoked: list[str] = []
        cancelled: list[str] = []
        old_quota = self.current_quota_version
        with self.lock:
            old_view = self.ledger.snapshot()
            now = monotonic_ms()
            self._expire_due_locked(now, event.seq)
            queue_ids = sorted(self.pending)
            cancelled = self.ledger.cancel_queues(
                queue_ids, event.seq, "queued_before_quota_replacement_became_stale"
            )
            for queue_id in queue_ids:
                self.pending.pop(queue_id, None)
            for lease in sorted(self.active.values(), key=lambda item: item.lease_id):
                if lease.quota_version < event.quota_version:
                    basis = "quota_version_advanced; expiry_precedes_quota"
                    if self.ledger.record_claim(lease.lease_id, "QUOTA_REVOKED", basis, event.seq, mono_ms=now):
                        revoked.append(lease.lease_id)
                    self._remove_lease_locked(lease.lease_id)
            new_version = int(event.quota_version)
            new_nodes = set(event.node_ids)
            added_nodes = sorted(new_nodes - self.active_nodes)
            removed_nodes = sorted(self.active_nodes - new_nodes)
            self.ledger.record_quota_transition(
                event.seq,
                new_version,
                sorted(new_nodes),
                added_nodes,
                removed_nodes,
                [],
                revoked,
                [],
                old_view=old_view,
                old_quota_version=old_quota,
            )
            self.current_quota_version = new_version
            self.active_nodes = new_nodes
            futures = self._pump_locked(now, event.seq)
        self._wait_futures(futures)
        for node_id in sorted(self.workers):
            runtime = self.workers[node_id]
            if runtime.process.poll() is not None:
                continue
            try:
                runtime.client.set_quota(self.current_quota_version)
            except EdgeError as exc:
                self.worker_errors.append(exc.to_dict())
        for lease_id in revoked:
            lease = self.ledger.lease(lease_id)
            if lease is None:
                continue
            runtime = self.workers.get(lease["node_id"])
            if runtime is not None and runtime.process.poll() is None:
                try:
                    runtime.client.release(lease_id, self.current_quota_version)
                except EdgeError as exc:
                    self.worker_errors.append(exc.to_dict())
        with self.lock:
            self.rejections.extend(
                {
                    "code": "E_LEASE",
                    "event_seq": event.seq,
                    "lease_id": queue_id,
                    "reason": "queued application cancelled by quota replacement",
                }
                for queue_id in cancelled
            )

    # ------------------------------------------------------------------
    # allocation and network completion
    # ------------------------------------------------------------------
    def _pump_locked(self, now: int, event_seq: int) -> list[Future[Any]]:
        if not self.pending:
            return []
        assert self.ledger is not None
        cfg = self.manifest.config
        used_slots: dict[str, set[int]] = {node.node_id: set() for node in self.manifest.nodes}
        node_counts: dict[str, int] = {node.node_id: 0 for node in self.manifest.nodes}
        node_bytes: dict[str, int] = {node.node_id: 0 for node in self.manifest.nodes}
        global_bytes = 0
        for lease in self.active.values():
            used_slots.setdefault(lease.node_id, set()).add(lease.slot_index)
            node_counts[lease.node_id] = node_counts.get(lease.node_id, 0) + 1
            node_bytes[lease.node_id] = node_bytes.get(lease.node_id, 0) + lease.byte_count
            global_bytes += lease.byte_count
        global_count = len(self.active)
        allocations: list[LeaseRuntime] = []
        for queue_id, pending in sorted(
            self.pending.items(), key=lambda pair: (pair[1].event_seq, pair[1].node_id, pair[1].shard_index, pair[0])
        ):
            if pending.node_id not in self.active_nodes or pending.quota_version != self.current_quota_version:
                self.ledger.cancel_queues([queue_id], event_seq, "application belongs to a stale quota view")
                self.pending.pop(queue_id, None)
                continue
            node = self.manifest.node_map.get(pending.node_id)
            if node is None:
                self.pending.pop(queue_id, None)
                continue
            if global_count >= cfg.global_lease_limit:
                continue
            if node_counts.get(node.node_id, 0) >= node.lease_limit:
                continue
            if node_counts.get(node.node_id, 0) >= node.slots:
                continue
            if global_bytes + pending.byte_count > cfg.global_byte_budget:
                continue
            occupied = used_slots.setdefault(node.node_id, set())
            if len(occupied) >= node.slots:
                continue
            slot_index = next(index for index in range(node.slots) if index not in occupied)
            shard = digest_for(self.manifest, pending.node_id, pending.content_key, pending.shard_index)
            lease_id = self._lease_id(pending.event_seq, pending.shard_index, pending.attempt)
            lease = LeaseRuntime(
                lease_id=lease_id,
                queue_id=queue_id,
                batch_id="",
                node_id=node.node_id,
                content_key=pending.content_key,
                shard_index=pending.shard_index,
                slot_index=slot_index,
                byte_count=pending.byte_count,
                granted_ms=now,
                expires_ms=now + cfg.lease_ttl_ms,
                lease_version=1,
                quota_version=self.current_quota_version,
                digest=shard.digest,
                checksum=shard.checksum,
                status="RESERVED",
                attempt=pending.attempt,
            )
            occupied.add(slot_index)
            node_counts[node.node_id] = node_counts.get(node.node_id, 0) + 1
            node_bytes[node.node_id] = node_bytes.get(node.node_id, 0) + pending.byte_count
            global_count += 1
            global_bytes += pending.byte_count
            allocations.append(lease)
        if not allocations:
            return []
        groups: dict[tuple[int, str, int, int], list[LeaseRuntime]] = {}
        for lease in allocations:
            groups.setdefault(
                (lease.granted_ms, lease.node_id, lease.quota_version, lease.attempt), []
            ).append(lease)
        batches: list[BatchRuntime] = []
        for (_granted, node_id, quota_version, attempt), group in sorted(groups.items(), key=lambda item: item[0]):
            group.sort(key=lambda lease: lease.lease_id)
            self.batch_counter += 1
            node_tag = hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:8]
            batch_id = f"B{event_seq:06d}-{node_tag}-{self.batch_counter:04d}"
            request = {
                "batch_id": batch_id,
                "event_seq": event_seq,
                "node_id": node_id,
                "quota_version": quota_version,
                "attempt": attempt,
                "items": [
                    {
                        "lease_id": lease.lease_id,
                        "content_key": lease.content_key,
                        "shard_index": lease.shard_index,
                        "expected_digest": lease.digest,
                        "expected_checksum": lease.checksum,
                    }
                    for lease in group
                ],
            }
            self.ledger.record_batch_open(batch_id, event_seq, node_id, request, attempt)
            for lease in group:
                lease.batch_id = batch_id
                self.active[lease.lease_id] = lease
                self.grant_count += 1
                self.ledger.record_grant(lease.ledger_payload(event_seq))
                self.pending.pop(lease.queue_id, None)
            batch = BatchRuntime(
                batch_id=batch_id,
                event_seq=event_seq,
                node_id=node_id,
                quota_version=quota_version,
                attempt=attempt,
                request=request,
                leases=group,
                pending_count=len(group),
            )
            self.open_batches[batch_id] = batch
            batches.append(batch)
        if (
            self.crash_node
            and not self.crash_done
            and self.crash_after_grants > 0
            and self.grant_count >= self.crash_after_grants
        ):
            self.crash_done = True
            self.crashed_nodes.add(self.crash_node)
            for batch in batches:
                if batch.node_id == self.crash_node:
                    batch.force_unknown = True
            runtime = self.workers.get(self.crash_node)
            if runtime is not None and runtime.process.poll() is None:
                runtime.process.kill()
        assert self.executor is not None
        futures: list[Future[Any]] = []
        for batch in batches:
            for lease in batch.leases:
                item = next(item for item in batch.request["items"] if item["lease_id"] == lease.lease_id)
                future = self.executor.submit(self._validate_one, batch.batch_id, lease.lease_id, item)
                batch.futures.append(future)
                futures.append(future)
        return futures

    def _validate_one(self, batch_id: str, lease_id: str, item: dict[str, Any]) -> None:
        with self.lock:
            batch = self.open_batches.get(batch_id)
            if batch is None or batch.finalized:
                return
            if batch.force_unknown:
                self._finalize_unknown_locked(
                    batch,
                    "worker_terminated_before_batch_commit; occupancy reclaimed before rerun",
                    "E_NODE",
                )
                return
            runtime = self.workers.get(batch.node_id)
            lease = self.active.get(lease_id)
            if runtime is None or lease is None:
                self._finalize_unknown_locked(batch, "node or lease disappeared before batch commit", "E_NODE")
                return
            client = runtime.client
            quota_version = batch.quota_version
        try:
            response = client.validate_batch(batch_id, [item], quota_version)
            response_items = response.get("items")
            if not isinstance(response_items, list) or len(response_items) != 1:
                raise NodeError("node returned an invalid validation batch", node_id=batch.node_id)
            result = response_items[0]
            if int(response.get("quota_version", -1)) != batch.quota_version:
                raise NodeError("node response quota version mismatch", node_id=batch.node_id)
            if response.get("manifest_digest") != self.manifest_digest:
                raise NodeError("node response manifest digest mismatch", node_id=batch.node_id)
            if result.get("lease_id") != lease_id:
                raise NodeError("node response lease_id mismatch", node_id=batch.node_id)
            if result.get("digest") != lease.digest or int(result.get("checksum", 0)) != lease.checksum:
                raise NodeError("node fixed-point digest/checksum mismatch", node_id=batch.node_id)
        except Exception as exc:
            edge_error = exc.to_dict() if isinstance(exc, EdgeError) else {"code": "E_NODE", "message": str(exc)}
            with self.lock:
                current = self.open_batches.get(batch_id)
                if current is not None and not current.finalized:
                    self.worker_errors.append(edge_error)
                    self._finalize_unknown_locked(
                        current,
                        f"uncommitted batch failed: {edge_error.get('code')} {edge_error.get('message')}",
                        str(edge_error.get("code", "E_NODE")),
                    )
            return
        with self.lock:
            current = self.open_batches.get(batch_id)
            if current is None or current.finalized:
                return
            current.responses.append(
                {
                    "lease_id": lease_id,
                    "digest": result.get("digest"),
                    "checksum": result.get("checksum"),
                    "sample_count": result.get("sample_count"),
                    "elapsed_mono_ms": int(result.get("elapsed_mono_ms", 0)),
                    "quota_version": int(response.get("quota_version", 0)),
                }
            )
            current.pending_count -= 1
            if current.pending_count == 0:
                assert self.ledger is not None
                self.ledger.record_batch_commit(batch_id, current.event_seq, current.responses)
                for batch_lease in current.leases:
                    runtime_lease = self.active.get(batch_lease.lease_id)
                    if runtime_lease is not None:
                        runtime_lease.status = "ACTIVE"
                current.finalized = True
                self.open_batches.pop(batch_id, None)

    def _finalize_unknown_locked(self, batch: BatchRuntime, basis: str, code: str) -> None:
        if batch.finalized:
            return
        assert self.ledger is not None
        batch.finalized = True
        self.ledger.record_batch_unknown(batch.batch_id, basis, batch.event_seq, error_code=code)
        for lease in batch.leases:
            self._remove_lease_locked(lease.lease_id)
            self.unknown_requests.append(
                {
                    "event_seq": batch.event_seq,
                    "node_id": batch.node_id,
                    "content_key": lease.content_key,
                    "shard_index": lease.shard_index,
                    "quota_version": batch.quota_version,
                    "attempt": batch.attempt + 1,
                }
            )
        self.open_batches.pop(batch.batch_id, None)

    def _claim_locked(
        self,
        lease: LeaseRuntime,
        claim_kind: str,
        basis: str,
        event_seq: int,
        now: int,
    ) -> bool:
        assert self.ledger is not None
        changed = self.ledger.record_claim(lease.lease_id, claim_kind, basis, event_seq, mono_ms=now)
        self._remove_lease_locked(lease.lease_id)
        return changed

    def _remove_lease_locked(self, lease_id: str) -> None:
        self.active.pop(lease_id, None)

    def _expire_due_locked(self, now: int, event_seq: int) -> int:
        expired = 0
        for lease in sorted(self.active.values(), key=lambda item: (item.expires_ms, item.lease_id)):
            if lease.status == "ACTIVE" and lease.expires_ms <= now:
                if self._claim_locked(
                    lease,
                    "EXPIRED",
                    "monotonic_deadline<=now; expiry_precedes_quota",
                    event_seq,
                    now,
                ):
                    expired += 1
        return expired

    def _sweeper_loop(self) -> None:
        while not self.stop_event.wait(0.01):
            try:
                with self.lock:
                    if not self.started:
                        continue
                    now = monotonic_ms()
                    expired = self._expire_due_locked(now, self.processed_seq)
                    futures = self._pump_locked(now, self.processed_seq) if expired or self.pending else []
                if futures:
                    wait(futures)
            except Exception as exc:
                self.worker_errors.append({"code": "E_NODE", "message": f"expiry sweeper: {exc}"})

    def _wait_futures(self, futures: list[Future[Any]]) -> None:
        if futures:
            wait(futures)

    def _drain_until_settled(self) -> None:
        deadline = monotonic_ms() + self.manifest.config.drain_timeout_ms
        while True:
            with self.lock:
                now = monotonic_ms()
                self._expire_due_locked(now, self.processed_seq)
                futures = self._pump_locked(now, self.processed_seq)
                pending = bool(self.pending)
                open_batches = bool(self.open_batches)
            self._wait_futures(futures)
            if not pending and not open_batches:
                return
            if monotonic_ms() >= deadline:
                return
            time.sleep(0.01)

    # ------------------------------------------------------------------
    # crash rerun
    # ------------------------------------------------------------------
    def _retry_unknown(self) -> None:
        stale_retry = False
        with self.lock:
            dead_nodes = sorted(
                node_id
                for node_id, runtime in self.workers.items()
                if runtime.process.poll() is not None
            )
        for node_id in dead_nodes:
            self._restart_worker(node_id)
        assert self.ledger is not None
        with self.lock:
            retry_items = list(self.unknown_requests)
            self.unknown_requests.clear()
            now = monotonic_ms()
            for item in sorted(
                retry_items,
                key=lambda value: (value["event_seq"], value["node_id"], value["shard_index"], value["attempt"]),
            ):
                if item["node_id"] not in self.active_nodes or item["quota_version"] != self.current_quota_version:
                    stale_retry = True
                    self.ledger.record_reject(
                        None,
                        int(item["event_seq"]),
                        item["node_id"],
                        "unknown lease retry belongs to a stale quota view",
                    )
                    self.rejections.append(
                        {
                            "code": "E_LEASE",
                            "event_seq": item["event_seq"],
                            "lease_id": None,
                            "reason": "unknown lease retry belongs to a stale quota view",
                        }
                    )
                    continue
                queue_id = self._queue_id(int(item["event_seq"]), int(item["shard_index"]), int(item["attempt"]))
                content = self.manifest.content_map[item["content_key"]]
                self.pending[queue_id] = PendingApplication(
                    queue_id=queue_id,
                    event_seq=int(item["event_seq"]),
                    node_id=item["node_id"],
                    content_key=item["content_key"],
                    shard_index=int(item["shard_index"]),
                    byte_count=content.shard_bytes,
                    quota_version=int(item["quota_version"]),
                    attempt=int(item["attempt"]),
                    wait_start_ms=now,
                )
                self.ledger.record_queue_start(
                    queue_id,
                    int(item["event_seq"]),
                    item["node_id"],
                    item["content_key"],
                    int(item["shard_index"]),
                    now,
                    int(item["attempt"]),
                    int(item["quota_version"]),
                )
            futures = self._pump_locked(now, self.processed_seq)
        self._wait_futures(futures)
        self._drain_until_settled()
        with self.lock:
            if not self.pending and not self.open_batches and not self.unknown_requests and not stale_retry:
                self.ledger.mark_rerun_resolved(self.processed_seq)

    def _restart_worker(self, node_id: str) -> None:
        existing = self.workers.get(node_id)
        if existing is not None and existing.process.poll() is None:
            return
        node = self.manifest.node_map[node_id]
        self._assert_port_available(node)
        with self.lock:
            runtime = self._spawn_worker(node)
            self.workers[node_id] = runtime
            assert self.ledger is not None
            leases = self.ledger.current_leases(node_id=node_id)
            if leases:
                runtime.client.restore(
                    [
                        {"lease_id": lease["lease_id"], "lease_version": int(lease["lease_version"])}
                        for lease in leases
                    ],
                    self.current_quota_version,
                )

    # ------------------------------------------------------------------
    # identifiers
    # ------------------------------------------------------------------
    @staticmethod
    def _lease_id(event_seq: int, shard_index: int, attempt: int) -> str:
        suffix = f"-R{attempt}" if attempt else ""
        return f"L{event_seq:06d}-S{shard_index:04d}{suffix}"

    @staticmethod
    def _queue_id(event_seq: int, shard_index: int, attempt: int) -> str:
        suffix = f"-R{attempt}" if attempt else ""
        return f"Q{event_seq:06d}-S{shard_index:04d}{suffix}"
