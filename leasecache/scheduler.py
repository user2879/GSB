from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Iterator

from .errors import LeaseError
from .ledger import Ledger
from .node_manager import NodeManager
from .planner import (
    arbitration_for,
    canonical_copies,
    equivalent_source_nodes,
    reuse_plan,
)
from .spec import PipelineSpec
from .util import StructuredLogger, mono_ms


class QueueTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_seq = 1
        self.records: list[dict[str, Any]] = []
        self._active: dict[str, int] = {}
        self.peak: dict[str, int] = {}

    def _new_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def enter(self, key: str) -> int:
        with self._lock:
            value = self._active.get(key, 0) + 1
            self._active[key] = value
            self.peak[key] = max(self.peak.get(key, 0), value)
            return value

    def leave(self, key: str) -> None:
        with self._lock:
            self._active[key] = max(0, self._active.get(key, 0) - 1)

    def record(
        self,
        *,
        kind: str,
        request_seq: int,
        artifact_key: str,
        wait_started_mono_ms: int,
        wait_ended_mono_ms: int,
        node_id: str | None = None,
        stage_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            queue_seq = self._new_seq()
            record = {
                "queue_seq": queue_seq,
                "kind": kind,
                "request_seq": request_seq,
                "artifact_key": artifact_key,
                "node_id": node_id,
                "stage_id": stage_id,
                "enqueued_mono_ms": wait_started_mono_ms,
                "wait_started_mono_ms": wait_started_mono_ms,
                "wait_ended_mono_ms": wait_ended_mono_ms,
                "waited_mono_ms": max(0, wait_ended_mono_ms - wait_started_mono_ms),
            }
            self.records.append(record)
            return record


class ParallelLeaseScheduler:
    def __init__(
        self,
        spec: PipelineSpec,
        ledger: Ledger,
        manager: NodeManager,
        logger: StructuredLogger,
        *,
        lease_ms: int,
        wait_timeout_ms: int = 20000,
        service_delay_ms: int = 0,
        interrupt_after_grants: int = 0,
        lease_namespace: str = "run",
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.manager = manager
        self.logger = logger
        self.lease_ms = lease_ms
        self.wait_timeout_ms = wait_timeout_ms
        self.service_delay_ms = service_delay_ms
        self.interrupt_after_grants = interrupt_after_grants
        self.lease_namespace = lease_namespace
        self.queue = QueueTracker()
        self._global = threading.BoundedSemaphore(spec.global_concurrency)
        self._stages = {
            stage.stage_id: threading.BoundedSemaphore(stage.concurrency)
            for stage in spec.stages
        }
        self._nodes = {
            node.node_id: threading.BoundedSemaphore(node.lease_limit)
            for node in spec.nodes
        }
        self._request_lock = threading.Lock()
        self._grant_lock = threading.Lock()
        self._grant_count = 0
        self.requests: list[dict[str, Any]] = []
        self._stop = threading.Event()

    @contextmanager
    def _permit(
        self,
        semaphore: threading.BoundedSemaphore,
        key: str,
        *,
        kind: str,
        request_seq: int,
        artifact_key: str,
        node_id: str | None = None,
        stage_id: str | None = None,
    ) -> Iterator[None]:
        started = mono_ms()
        acquired = semaphore.acquire(blocking=False)
        if not acquired:
            semaphore.acquire()
            ended = mono_ms()
            self.queue.record(
                kind=kind,
                request_seq=request_seq,
                artifact_key=artifact_key,
                node_id=node_id,
                stage_id=stage_id,
                wait_started_mono_ms=started,
                wait_ended_mono_ms=ended,
            )
        self.queue.enter(key)
        try:
            yield
        finally:
            self.queue.leave(key)
            semaphore.release()

    def _select(self, key: str) -> tuple[Any, Any]:
        canonical = canonical_copies(self.spec, self.ledger.summaries())[key]
        arbitration = arbitration_for(self.spec, self.ledger, canonical, key)
        return canonical, arbitration

    def _wait_for_capacity(self, key: str, request_seq: int, started: int) -> None:
        if mono_ms() - started >= self.wait_timeout_ms:
            raise LeaseError(
                "E_QUEUE",
                "lease request exceeded the capacity queue deadline",
                field="queue",
                expected=f"capacity within {self.wait_timeout_ms} ms",
                actual=f"waited {mono_ms() - started} ms",
                artifact_key=key,
            )
        self._stop.wait(0.02)
        if self._stop.is_set():
            raise LeaseError(
                "E_ABORT",
                "parallel scheduler was stopped",
                field="scheduler",
                expected="active scheduler",
                actual="stopped",
                artifact_key=key,
            )

    def _grant_interrupt(self) -> None:
        with self._grant_lock:
            self._grant_count += 1
            if self.interrupt_after_grants and self._grant_count >= self.interrupt_after_grants:
                # Deliberately bypass finally/atexit/report generation.  SQLite
                # already committed the grant transaction; everything else is
                # treated as an unclean shutdown by the next process.
                self.logger.write(
                    "WARNING",
                    "forced-interruption marker reached; terminating without exit report",
                    code="E_FORCED_INTERRUPTION",
                    grants=self._grant_count,
                )
                os._exit(86)

    def _acquire_one(self, key: str, request_seq: int) -> dict[str, Any]:
        artifact = self.spec.artifact_by_key[key]
        capacity_queue_started: int | None = None
        attempt = 1
        while True:
            canonical, arbitration = self._select(key)
            if arbitration.selected_node is None:
                if capacity_queue_started is None:
                    capacity_queue_started = mono_ms()
                    self.logger.info(
                        "lease request queued for capacity",
                        artifact_key=key,
                        request_seq=request_seq,
                        verdicts=[verdict.to_dict() for verdict in arbitration.verdicts],
                    )
                self._wait_for_capacity(key, request_seq, capacity_queue_started)
                continue
            node_id = arbitration.selected_node
            with self._permit(
                self._global,
                "global",
                kind="global",
                request_seq=request_seq,
                artifact_key=key,
            ):
                with self._permit(
                    self._stages[artifact.stage_id],
                    f"stage:{artifact.stage_id}",
                    kind="stage",
                    request_seq=request_seq,
                    artifact_key=key,
                    stage_id=artifact.stage_id,
                ):
                    with self._permit(
                        self._nodes.get(node_id, threading.BoundedSemaphore(1)),
                        f"node:{node_id}",
                        kind="node",
                        request_seq=request_seq,
                        artifact_key=key,
                        node_id=node_id,
                    ):
                        canonical, arbitration = self._select(key)
                        if arbitration.selected_node is None:
                            if capacity_queue_started is None:
                                capacity_queue_started = mono_ms()
                            self._wait_for_capacity(key, request_seq, capacity_queue_started)
                            continue
                        if arbitration.selected_node != node_id:
                            continue
                        lease_id = (
                            f"{node_id}:lease:{self.lease_namespace}:"
                            f"{request_seq}:{attempt}"
                        )
                        attempt += 1
                        source_nodes = equivalent_source_nodes(
                            self.spec,
                            self.ledger.summaries(),
                            key,
                            canonical,
                        )
                        self.ledger.begin_pending(
                            lease_id,
                            node_id,
                            key,
                            request_seq,
                            canonical.digest,
                            list(artifact.references),
                        )
                        try:
                            ack = self.manager.client(node_id).lease(
                                [key],
                                self.lease_ms,
                                request_seq,
                                service_delay_ms=self.service_delay_ms,
                                lease_id=lease_id,
                            )
                        except LeaseError as exc:
                            self.ledger.cancel_pending(lease_id, f"transport: {exc.code}")
                            if exc.code in {"E_LEASE_LIMIT", "E_CAPACITY"}:
                                if capacity_queue_started is None:
                                    capacity_queue_started = mono_ms()
                                self.logger.info(
                                    "node quota rejected request; queued for retry",
                                    artifact_key=key,
                                    request_seq=request_seq,
                                    node_id=node_id,
                                    reason=exc.code,
                                )
                                self._wait_for_capacity(key, request_seq, capacity_queue_started)
                                continue
                            raise
                        if ack["node_id"] != node_id:
                            try:
                                self.manager.client(node_id).release(ack.get("lease_id", ""))
                            except Exception:
                                pass
                            self.ledger.cancel_pending(lease_id, "ack node identity mismatch")
                            raise LeaseError(
                                "E_ACK",
                                "acknowledgement node id differs from request target",
                                field="node_id",
                                expected=node_id,
                                actual=ack["node_id"],
                                artifact_key=key,
                                node_id=node_id,
                            )
                        if sorted(ack["artifact_keys"]) != [key]:
                            try:
                                self.manager.client(node_id).release(ack["lease_id"])
                            except Exception:
                                pass
                            self.ledger.cancel_pending(lease_id, "ack artifact set mismatch")
                            raise LeaseError(
                                "E_ACK",
                                "acknowledgement artifact set does not match request",
                                field="artifact_keys",
                                expected=[key],
                                actual=ack["artifact_keys"],
                                artifact_key=key,
                                node_id=node_id,
                            )
                        if ack["digests"].get(key) != canonical.digest:
                            try:
                                self.manager.client(node_id).release(ack["lease_id"])
                            except Exception:
                                pass
                            self.ledger.cancel_pending(lease_id, "ack digest mismatch")
                            raise LeaseError(
                                "E_VERIFY",
                                "node digest no longer matches the canonical ledger digest",
                                field="digest",
                                expected=canonical.digest,
                                actual=ack["digests"].get(key),
                                artifact_key=key,
                                node_id=node_id,
                            )
                        try:
                            granted = self.ledger.commit_grant(
                                lease_id,
                                node_id,
                                [key],
                                ack["digests"],
                                ack.get("refs", {key: list(artifact.references)}),
                                int(ack["expires_mono_ms"]),
                                source_nodes=source_nodes,
                                node_capacity_ack=ack["capacity"],
                            )
                        except LeaseError as exc:
                            try:
                                self.manager.client(node_id).release(ack["lease_id"])
                            except Exception:
                                pass
                            self.ledger.cancel_pending(lease_id, f"ledger: {exc.code}")
                            if exc.code in {"E_LEASE_LIMIT", "E_CAPACITY", "E_LEASE_CLAIMED"}:
                                if capacity_queue_started is None:
                                    capacity_queue_started = mono_ms()
                                self._wait_for_capacity(key, request_seq, capacity_queue_started)
                                continue
                            raise
                        if capacity_queue_started is not None:
                            ended = mono_ms()
                            self.queue.record(
                                kind="capacity",
                                request_seq=request_seq,
                                artifact_key=key,
                                node_id=node_id,
                                wait_started_mono_ms=capacity_queue_started,
                                wait_ended_mono_ms=ended,
                            )
                        request_record = {
                            "artifact_key": key,
                            "request_seq": request_seq,
                            "node_id": node_id,
                            "lease_id": lease_id,
                            "ack_elapsed_mono_ms": int(ack["elapsed_mono_ms"]),
                            "expires_mono_ms": int(ack["expires_mono_ms"]),
                            "capacity": ack["capacity"],
                            "event_seq": granted["event_seq"],
                        }
                        with self._request_lock:
                            self.requests.append(request_record)
                        self.logger.info(
                            "real TCP lease request granted",
                            **request_record,
                        )
                        self._grant_interrupt()
                        return request_record

    def run(self) -> dict[str, Any]:
        initial_plan = reuse_plan(self.spec, self.ledger)
        reusable_keys = sorted(
            key for key, decision in initial_plan.items() if decision["reusable"]
        )
        recompute_keys = sorted(
            key for key, decision in initial_plan.items() if not decision["reusable"]
        )
        self.logger.info(
            "incremental lease plan computed",
            reusable_keys=reusable_keys,
            recompute_keys=recompute_keys,
        )
        # Request ids are preassigned in stable topological/key order.
        ordered_keys = [
            key for layer in self.spec.topological_layers() for key in sorted(layer)
        ]
        request_seq_by_key = {
            key: index + 1 for index, key in enumerate(ordered_keys)
        }
        errors: list[tuple[str, LeaseError]] = []
        with ThreadPoolExecutor(
            max_workers=self.spec.global_concurrency,
            thread_name_prefix="lease-worker",
        ) as executor:
            for layer in self.spec.topological_layers():
                layer_keys = [key for key in sorted(layer) if key in recompute_keys]
                if not layer_keys:
                    continue
                futures = {
                    executor.submit(
                        self._acquire_one, key, request_seq_by_key[key]
                    ): key
                    for key in layer_keys
                }
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        future.result()
                    except LeaseError as exc:
                        errors.append((key, exc))
                        self._stop.set()
                    except Exception as exc:  # pragma: no cover - defensive thread boundary
                        errors.append(
                            (
                                key,
                                LeaseError(
                                    "E_INTERNAL",
                                    f"unexpected worker failure: {exc!r}",
                                    field="worker",
                                    expected="completed lease task",
                                    actual=repr(exc),
                                    artifact_key=key,
                                ),
                            )
                        )
                        self._stop.set()
                if errors:
                    break
        if errors:
            key, error = sorted(errors, key=lambda item: item[0])[0]
            self.logger.error(error.code, error.message, artifact_key=key, **error.details)
            raise error
        requests = sorted(self.requests, key=lambda item: item["artifact_key"])
        result = {
            "reusable_keys": reusable_keys,
            "recompute_keys": recompute_keys,
            "recomputed_keys": sorted(item["artifact_key"] for item in requests),
            "queue_waits": list(self.queue.records),
            "requests": requests,
            "global_limit": self.spec.global_concurrency,
            "stage_limits": {
                stage.stage_id: stage.concurrency for stage in self.spec.stages
            },
            "node_limits": {
                node.node_id: node.lease_limit for node in self.spec.nodes
            },
            "peak_global_inflight": self.queue.peak.get("global", 0),
            "peak_stage_inflight": {
                stage.stage_id: self.queue.peak.get(f"stage:{stage.stage_id}", 0)
                for stage in sorted(self.spec.stages, key=lambda item: item.stage_id)
            },
            "peak_node_inflight": {
                node.node_id: self.queue.peak.get(f"node:{node.node_id}", 0)
                for node in sorted(self.spec.nodes, key=lambda item: item.node_id)
            },
        }
        self.logger.info(
            "parallel lease scheduling completed",
            reusable_keys=reusable_keys,
            recomputed_keys=result["recomputed_keys"],
            queue_wait_count=len(result["queue_waits"]),
            request_count=len(requests),
            peak_global_inflight=result["peak_global_inflight"],
        )
        return result
