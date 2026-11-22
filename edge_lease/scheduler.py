"""Bounded parallel dispatch scheduler.

Three hard constraints are enforced at all times:

* a global semaphore caps running dispatch tasks;
* a per-node semaphore caps tasks targeting the same node (its declared quota);
* every socket conversation uses the node's per-dispatch timeout.

Queue wait and service windows are stamped with the monotonic millisecond
clock.  Timeouts follow the node retry policy with byte-identical request
semantics, and retry receipts are kept distinct from first attempts.  Dead
nodes are excluded from all later scheduling.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .errors import HandshakeError, ReplyError
from .protocol import DispatchTimeout
from .logging_setup import JsonlLogger, mono_ms
from .models import NodeDef, NodeOutcome, PlanItem, Receipt, VersionManifest
from .protocol import NodeClient, NodeProcess


@dataclass
class TaskRecord:
    item: PlanItem
    enqueued_ms: int
    wait_start_ms: int
    wait_end_ms: int = 0
    service_start_ms: int = 0
    service_end_ms: int = 0
    receipts: List[Receipt] = field(default_factory=list)
    final_status: str = "not_dispatched"  # ok|timeout|error|not_dispatched
    error_code: Optional[str] = None

    @property
    def waited_ms(self) -> int:
        return max(0, self.wait_end_ms - self.wait_start_ms)


@dataclass
class SchedulerResult:
    records: List[TaskRecord]
    outcomes: Dict[str, NodeOutcome]
    receipts: List[Receipt]
    unavailable_nodes: List[str]

    def ok_items(self) -> List[TaskRecord]:
        return [record for record in self.records if record.final_status == "ok"]


class Scheduler:
    def __init__(
        self,
        node_processes: Dict[str, NodeProcess],
        registry: Dict[str, NodeDef],
        manifest: VersionManifest,
        content_seed: int,
        logger: JsonlLogger,
        global_limit: int,
    ) -> None:
        self.node_processes = node_processes
        self.registry = registry
        self.manifest = manifest
        self.content_seed = content_seed
        self.logger = logger
        self.global_limit = global_limit
        self._global = threading.BoundedSemaphore(global_limit)
        self._node_slots = {
            node_id: threading.BoundedSemaphore(node.per_node_quota)
            for node_id, node in registry.items()
        }
        self._lock = threading.Lock()
        self.cumulative_outcomes: Dict[str, NodeOutcome] = {
            node_id: NodeOutcome(node_id=node_id, status="not_dispatched")
            for node_id in registry
        }

    def run(self, items: List[PlanItem]) -> SchedulerResult:
        ordered = sorted(items, key=lambda item: (item.priority, item.path, item.node_id))
        records: Dict[str, TaskRecord] = {
            item.item_id: TaskRecord(
                item=item,
                enqueued_ms=mono_ms(),
                wait_start_ms=mono_ms(),
            )
            for item in ordered
        }
        threads = [
            threading.Thread(
                target=self._run_task,
                name=f"dispatch-{item.item_id}",
                args=(records[item.item_id],),
            )
            for item in ordered
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        result = self._summarize(list(records.values()))
        self._accumulate(result)
        return result

    def _accumulate(self, result: SchedulerResult) -> None:
        for node_id, outcome in result.outcomes.items():
            total = self.cumulative_outcomes[node_id]
            total.ok += outcome.ok
            total.timeout += outcome.timeout
            total.failed += outcome.failed
            total.not_dispatched += outcome.not_dispatched
            total.attempts += outcome.attempts
            if node_id in result.unavailable_nodes:
                total.status = "unavailable"
            elif total.timeout and total.ok == 0 and total.failed == 0:
                total.status = "timeout"
            elif total.failed and total.ok == 0:
                total.status = "failed"
            elif total.not_dispatched and total.ok == 0:
                total.status = "not_dispatched"
            elif total.ok and (total.timeout or total.failed):
                total.status = "partial"
            else:
                total.status = "success"

    def _run_task(self, record: TaskRecord) -> None:
        item = record.item
        node = self.registry[item.node_id]
        process = self.node_processes.get(item.node_id)
        if process is None or not process.check_liveness():
            record.wait_end_ms = mono_ms()
            record.final_status = "error"
            record.error_code = "E_HANDSHAKE"
            self.logger.error(
                "task_skipped_dead_node",
                item_id=item.item_id,
                node_id=item.node_id,
                code="E_HANDSHAKE",
            )
            return
        node_slot = self._node_slots[item.node_id]
        acquired_global = False
        acquired_node = False
        try:
            self.logger.info(
                "task_waiting",
                item_id=item.item_id,
                node_id=item.node_id,
                ts_wait_start=record.wait_start_ms,
            )
            self._global.acquire()
            acquired_global = True
            node_slot.acquire()
            acquired_node = True
            record.wait_end_ms = mono_ms()
            record.service_start_ms = mono_ms()
            self.logger.info(
                "task_running",
                item_id=item.item_id,
                node_id=item.node_id,
                ts_start=record.service_start_ms,
                global_limit=self.global_limit,
                node_quota=node.per_node_quota,
            )
            self._with_retries(record, node, process)
        finally:
            record.service_end_ms = mono_ms()
            self.logger.info(
                "task_finished",
                item_id=item.item_id,
                node_id=item.node_id,
                ts_end=record.service_end_ms,
                final_status=record.final_status,
            )
            if acquired_node:
                node_slot.release()
            if acquired_global:
                self._global.release()

    def _with_retries(
        self, record: TaskRecord, node: NodeDef, process: NodeProcess
    ) -> None:
        item = record.item
        client = NodeClient(process, self.logger)
        attempts = node.max_retries + 1 if node.retry_policy == "retry" else 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                receipt = client.dispatch(
                    item, self.manifest, self.content_seed, attempt
                )
            except Exception as exc:  # timeout / handshake / reply / dead node
                last_error = exc
                receipt = Receipt(
                    item_id=item.item_id,
                    node_id=item.node_id,
                    version=item.version,
                    path=item.path,
                    derived_digest=0,
                    shard_hit_digest=0,
                    cost_ms=node.dispatch_timeout_ms,
                    quant_version=item.quant_version,
                    attempt=attempt,
                    status=self._error_status(exc),
                    error_code=getattr(exc, "code", "E_PLATFORM"),
                    waited_ms=record.waited_ms,
                )
                record.receipts.append(receipt)
                self.logger.warning(
                    "dispatch_attempt_failed",
                    item_id=item.item_id,
                    node_id=item.node_id,
                    attempt=attempt,
                    code=receipt.error_code,
                    retry=(attempt < attempts),
                )
                if not process.check_liveness():
                    break
                continue
            receipt.waited_ms = record.waited_ms
            receipt.service_ms = max(0, record.service_end_ms - record.service_start_ms)
            record.receipts.append(receipt)
            record.final_status = "ok"
            record.error_code = None
            self.logger.info(
                "dispatch_ok",
                item_id=item.item_id,
                node_id=item.node_id,
                attempt=attempt,
                cost_ms=receipt.cost_ms,
            )
            return
        record.final_status = (
            "timeout"
            if isinstance(last_error, (TimeoutError, DispatchTimeout))
            or "timed out" in str(last_error).lower()
            else "error"
        )
        record.error_code = getattr(last_error, "code", "E_PLATFORM")

    @staticmethod
    def _error_status(exc: Exception) -> str:
        if isinstance(exc, (TimeoutError, DispatchTimeout)):
            return "timeout"
        if "timed out" in str(exc).lower():
            return "timeout"
        if isinstance(exc, (HandshakeError, ReplyError)):
            return "error"
        return "error"

    def _summarize(self, records: List[TaskRecord]) -> SchedulerResult:
        outcomes: Dict[str, NodeOutcome] = {
            node_id: NodeOutcome(node_id=node_id, status="not_dispatched")
            for node_id in self.registry
        }
        all_receipts: List[Receipt] = []
        unavailable = sorted(
            node_id
            for node_id, process in self.node_processes.items()
            if not process.available
        )
        for record in records:
            outcome = outcomes[record.item.node_id]
            all_receipts.extend(record.receipts)
            outcome.attempts += len(record.receipts)
            if record.final_status == "ok":
                outcome.ok += 1
            elif record.final_status == "timeout":
                outcome.timeout += 1
            elif record.final_status == "error":
                outcome.failed += 1
            else:
                outcome.not_dispatched += 1
        for node_id, outcome in outcomes.items():
            if node_id in unavailable:
                outcome.status = "unavailable"
            elif outcome.timeout and outcome.ok == 0 and outcome.failed == 0:
                outcome.status = "timeout"
            elif outcome.failed and outcome.ok == 0:
                outcome.status = "failed"
            elif outcome.not_dispatched and outcome.ok == 0:
                outcome.status = "not_dispatched"
            elif outcome.ok and (outcome.timeout or outcome.failed):
                outcome.status = "partial"
            else:
                outcome.status = "success"
        return SchedulerResult(
            records=records,
            outcomes=outcomes,
            receipts=all_receipts,
            unavailable_nodes=unavailable,
        )
