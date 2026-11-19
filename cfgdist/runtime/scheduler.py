"""Budget-constrained parallel distribution scheduler.

Concurrency is enforced by a single bounded semaphore (never bypassed).
Distribution dependencies are honoured through per-request *success gates*:
a dependent request blocks on its upstreams' gates, which are set only after an
upstream succeeds.  No thread reads shared result state before it is published,
which removes the start-up race entirely.
"""
from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from ..config.planner import DistributionPlan, PlanRequest
from ..errors import E_NODE, ErrorDetail, PlatformError
from ..util.clock import now_ms
from ..util.logging_util import StructuredLogger
from .node_process import NodeProcess, spawn_node
from ..config.manifest import Registry


@dataclass
class AttemptRecord:
    request_id: str
    attempt: int
    node_id: str
    effective_version: int
    started_wait_ms: int
    ended_wait_ms: int
    started_send_ms: int
    ended_send_ms: int
    outcome: str            # ok | timeout | error
    duplicate: bool = False
    sequence: int | None = None
    elapsed_ms: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempt": self.attempt,
            "node_id": self.node_id,
            "effective_version": self.effective_version,
            "started_wait_ms": self.started_wait_ms,
            "ended_wait_ms": self.ended_wait_ms,
            "wait_ms": self.ended_wait_ms - self.started_wait_ms,
            "started_send_ms": self.started_send_ms,
            "ended_send_ms": self.ended_send_ms,
            "send_ms": self.ended_send_ms - self.started_send_ms,
            "outcome": self.outcome,
            "duplicate": self.duplicate,
            "sequence": self.sequence,
            "node_elapsed_ms": self.elapsed_ms,
            "reason": self.reason,
        }


@dataclass
class RequestRecord:
    request_id: str
    node_id: str
    effective_version: int
    final_outcome: str       # ok | exhausted | blocked
    attempts: list[AttemptRecord] = field(default_factory=list)
    receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "effective_version": self.effective_version,
            "final_outcome": self.final_outcome,
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class SchedulerResult:
    records: list[RequestRecord]
    receipts: dict[str, dict[str, Any]]   # node_id -> winning receipt
    concurrency_max: int
    observed_max_inflight: int


class NodePool:
    """Start every required node once and own their lifecycle."""

    def __init__(
        self,
        registry: Registry,
        logger: StructuredLogger,
        *,
        log_dir: str,
        inject_by_node: dict[str, list[str]] | None = None,
        startup_timeout_ms: int = 8000,
        handshake_timeout_ms: int = 5000,
        operation_timeout_ms: int = 10000,
    ) -> None:
        self.registry = registry
        self.logger = logger
        self.log_dir = log_dir
        self.inject_by_node = inject_by_node or {}
        self.processes: dict[str, NodeProcess] = {}
        self.startup_timeout_ms = startup_timeout_ms
        self.handshake_timeout_ms = handshake_timeout_ms
        self.operation_timeout_ms = operation_timeout_ms

    def start(self, node_ids: list[str]) -> None:
        for node_id in sorted(node_ids):
            spec = self.registry.nodes[node_id]
            child_log = f"{self.log_dir}/node-{node_id}.log"
            process = spawn_node(
                spec,
                self.logger,
                log_path=child_log,
                inject=self.inject_by_node.get(node_id),
                startup_timeout_ms=self.startup_timeout_ms,
                handshake_timeout_ms=self.handshake_timeout_ms,
                operation_timeout_ms=self.operation_timeout_ms,
            )
            process.handshake()
            self.processes[node_id] = process

    def get(self, node_id: str) -> NodeProcess:
        if node_id not in self.processes:
            raise PlatformError(
                E_NODE,
                "node was not started for this plan",
                [ErrorDetail(key=node_id, source="scheduler")],
            )
        return self.processes[node_id]

    def shutdown_all(self) -> None:
        for process in self.processes.values():
            try:
                process.shutdown()
            except OSError:
                process.force_reap()


class Scheduler:
    def __init__(
        self,
        pool: NodePool,
        logger: StructuredLogger,
        *,
        concurrency: int,
        max_attempts: int,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency limit must be >= 1")
        self.pool = pool
        self.logger = logger
        self._semaphore = threading.BoundedSemaphore(concurrency)
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self._inflight_lock = threading.Lock()
        self._inflight = 0
        self._observed_max = 0

    def _track_enter(self) -> None:
        with self._inflight_lock:
            self._inflight += 1
            self._observed_max = max(self._observed_max, self._inflight)

    def _track_exit(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1

    def _execute_request(
        self,
        request: PlanRequest,
        success_gates: dict[str, threading.Event],
        finish_events: dict[str, threading.Event],
        records: dict[str, RequestRecord],
        records_lock: threading.Lock,
    ) -> None:
        # Wait for every distribution dependency to *finish* (the finish event
        # is always set, success or failure), then check its success gate.
        for dep in request.depends_on:
            finish_events[dep].wait()

        blocked = not all(success_gates[dep].is_set() for dep in request.depends_on)

        record = RequestRecord(
            request_id=request.request_id,
            node_id=request.node_id,
            effective_version=request.version,
            final_outcome="blocked" if blocked else "exhausted",
        )
        with records_lock:
            records[request.request_id] = record

        if blocked:
            self.logger.log(
                "dispatch_dependency_blocked",
                key=request.node_id,
                source="scheduler",
                request_id=request.request_id,
            )
            finish_events[request.request_id].set()
            return

        process = self.pool.get(request.node_id)
        winning_receipt: dict[str, Any] | None = None
        started_wait = now_ms()
        prev_end = started_wait

        for attempt in range(1, self.max_attempts + 1):
            self._semaphore.acquire()  # never bypass the global cap
            ended_wait = now_ms()
            self._track_enter()
            started_send = now_ms()
            attempt_record = AttemptRecord(
                request_id=request.request_id,
                attempt=attempt,
                node_id=request.node_id,
                effective_version=request.version,
                started_wait_ms=started_wait if attempt == 1 else prev_end,
                ended_wait_ms=ended_wait,
                started_send_ms=started_send,
                ended_send_ms=0,
                outcome="error",
            )
            try:
                receipt = process.request(request.request_id, attempt, request.slice_payload)
                ended_send = now_ms()
                attempt_record.ended_send_ms = ended_send
                attempt_record.outcome = "ok"
                attempt_record.duplicate = bool(receipt.get("duplicate"))
                attempt_record.sequence = receipt.get("sequence")
                attempt_record.elapsed_ms = receipt.get("elapsed_ms")
                winning_receipt = receipt
                self.logger.log(
                    "dispatch_ok",
                    key=request.node_id,
                    source="scheduler",
                    request_id=request.request_id,
                    attempt=attempt,
                    version=request.version,
                    duplicate=attempt_record.duplicate,
                )
                break
            except (socket.timeout, TimeoutError, OSError, ConnectionError) as exc:
                ended_send = now_ms()
                attempt_record.ended_send_ms = ended_send
                attempt_record.outcome = "timeout"
                attempt_record.reason = str(exc)
                self.logger.log(
                    "dispatch_timeout",
                    key=request.node_id,
                    source="scheduler",
                    request_id=request.request_id,
                    attempt=attempt,
                    reason=str(exc),
                )
            finally:
                record.attempts.append(attempt_record)
                self._track_exit()
                self._semaphore.release()
                prev_end = now_ms()

        if winning_receipt is not None:
            record.final_outcome = "ok"
            record.receipt = winning_receipt
            success_gates[request.request_id].set()
        finish_events[request.request_id].set()

    def run(self, plan: DistributionPlan) -> SchedulerResult:
        records: dict[str, RequestRecord] = {}
        records_lock = threading.Lock()
        success_gates = {
            request.request_id: threading.Event() for request in plan.requests
        }
        finish_events = {
            request.request_id: threading.Event() for request in plan.requests
        }
        threads: list[threading.Thread] = []

        for request in plan.requests:
            thread = threading.Thread(
                target=self._execute_request,
                args=(request, success_gates, finish_events, records, records_lock),
                name=f"dispatch-{request.request_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        receipts = {
            record.node_id: record.receipt
            for record in records.values()
            if record.receipt is not None
        }
        ordered = [records[request.request_id] for request in plan.requests]
        return SchedulerResult(
            records=ordered,
            receipts=receipts,
            concurrency_max=self.concurrency,
            observed_max_inflight=self._observed_max,
        )
