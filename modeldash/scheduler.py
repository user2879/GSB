"""Priority, quota-bounded dispatch with shard-boundary preemption and recovery."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import DispatchError, LedgerError, WorkbenchError
from .inference import aggregate_output_summaries
from .nodes import DispatchResult, SocketDispatcher, request_id_for
from .receipts import Receipt, ReceiptBook, build_node_untrust
from .spec import Spec, TaskRecord
from .storage import Ledger
from .util import StructuredLog, canonical_json, digest_json, now_monotonic_ms, utc_now


WAITING_RELEASE = "WAITING_RELEASE"
PENDING = "PENDING"
RUNNING = "RUNNING"
PREEMPT_REQUESTED = "PREEMPT_REQUESTED"
BACKOFF = "BACKOFF"
PAUSED = "PAUSED"
DONE = "DONE"
EXHAUSTED = "EXHAUSTED"
FAILED = "FAILED"
TERMINAL_STATES = {DONE, EXHAUSTED, FAILED}
ACTIVE_STATES = {RUNNING, PREEMPT_REQUESTED}


@dataclass
class RuntimeTask:
    task_id: str
    config_version: int
    node_id: str
    priority: int
    total_shards: int
    release_after_ms: int
    state: str = PENDING
    completed_shards: list[dict[str, Any]] = field(default_factory=list)
    output_summary: str = ""
    error_code: str = ""
    error_detail: str = ""
    retries_by_shard: dict[int, int] = field(default_factory=dict)
    attempts_total: int = 0
    retries_total: int = 0
    wait_ms: int = 0
    wait_started_ms: int | None = None
    last_wait_started_ms: int | None = None
    last_wait_ended_ms: int | None = None
    wait_windows: list[dict[str, int]] = field(default_factory=list)
    used_quota_ms: int = 0
    inflight: dict[str, Any] | None = None
    slots_allocated: bool = False
    preempt_requested_by: str = ""
    preempt_count: int = 0
    recovery_reruns: int = 0
    unknown_shards: list[int] = field(default_factory=list)
    backoff_for_shard: int | None = None
    backoff_index: int | None = None
    backoff_ms: int = 0
    backoff_started_ms: int | None = None
    backoff_ended_ms: int | None = None
    last_output_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config_version": self.config_version,
            "node_id": self.node_id,
            "priority": self.priority,
            "total_shards": self.total_shards,
            "release_after_ms": self.release_after_ms,
            "state": self.state,
            "completed_shards": self.completed_shards,
            "output_summary": self.output_summary,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "retries_by_shard": {str(key): self.retries_by_shard[key] for key in sorted(self.retries_by_shard)},
            "attempts_total": self.attempts_total,
            "retries_total": self.retries_total,
            "wait_ms": self.wait_ms,
            "wait_started_ms": self.wait_started_ms,
            "last_wait_started_ms": self.last_wait_started_ms,
            "last_wait_ended_ms": self.last_wait_ended_ms,
            "wait_windows": self.wait_windows,
            "used_quota_ms": self.used_quota_ms,
            "inflight": self.inflight,
            "preempt_requested_by": self.preempt_requested_by,
            "preempt_count": self.preempt_count,
            "recovery_reruns": self.recovery_reruns,
            "unknown_shards": self.unknown_shards,
            "backoff_for_shard": self.backoff_for_shard,
            "backoff_index": self.backoff_index,
            "backoff_ms": self.backoff_ms,
            "backoff_started_ms": self.backoff_started_ms,
            "backoff_ended_ms": self.backoff_ended_ms,
            "last_output_summary": self.last_output_summary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeTask":
        retries = {int(key): int(item) for key, item in value.get("retries_by_shard", {}).items()}
        return cls(
            task_id=str(value["task_id"]),
            config_version=int(value["config_version"]),
            node_id=str(value["node_id"]),
            priority=int(value["priority"]),
            total_shards=int(value["total_shards"]),
            release_after_ms=int(value.get("release_after_ms", 0)),
            state=str(value["state"]),
            completed_shards=list(value.get("completed_shards", [])),
            output_summary=str(value.get("output_summary", "")),
            error_code=str(value.get("error_code", "")),
            error_detail=str(value.get("error_detail", "")),
            retries_by_shard=retries,
            attempts_total=int(value.get("attempts_total", 0)),
            retries_total=int(value.get("retries_total", 0)),
            wait_ms=int(value.get("wait_ms", 0)),
            wait_started_ms=value.get("wait_started_ms"),
            last_wait_started_ms=value.get("last_wait_started_ms"),
            last_wait_ended_ms=value.get("last_wait_ended_ms"),
            wait_windows=list(value.get("wait_windows", [])),
            used_quota_ms=int(value.get("used_quota_ms", 0)),
            inflight=value.get("inflight"),
            preempt_requested_by=str(value.get("preempt_requested_by", "")),
            preempt_count=int(value.get("preempt_count", 0)),
            recovery_reruns=int(value.get("recovery_reruns", 0)),
            unknown_shards=[int(item) for item in value.get("unknown_shards", [])],
            backoff_for_shard=value.get("backoff_for_shard"),
            backoff_index=value.get("backoff_index"),
            backoff_ms=int(value.get("backoff_ms", 0)),
            backoff_started_ms=value.get("backoff_started_ms"),
            backoff_ended_ms=value.get("backoff_ended_ms"),
            last_output_summary=str(value.get("last_output_summary", "")),
        )


@dataclass
class SchedulerSummary:
    session_id: str
    tasks: list[dict[str, Any]]
    preemptions: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    receipts: list[dict[str, Any]]
    node_untrust: list[dict[str, Any]]
    deterministic_digest: str
    recovered: bool
    recovery_notes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": self.session_id,
            "recovered": self.recovered,
            "recovery_notes": self.recovery_notes,
            "tasks": self.tasks,
            "preemptions": self.preemptions,
            "decisions": self.decisions,
            "receipts": self.receipts,
            "node_untrust": self.node_untrust,
            "deterministic_digest": self.deterministic_digest,
        }


class Scheduler:
    """A deterministic policy layer around genuinely concurrent socket dispatch."""

    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        session_id: str,
        dispatcher: SocketDispatcher,
        *,
        recovered_state: dict[str, Any] | None = None,
        logger: StructuredLog | None = None,
        clock_ms: Callable[[], int] = now_monotonic_ms,
        snapshot_hook: Callable[[dict[str, Any], str], None] | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.session_id = session_id
        self.dispatcher = dispatcher
        self.logger = logger
        self.clock_ms = clock_ms
        self.snapshot_hook = snapshot_hook
        self.cond = threading.Condition(threading.RLock())
        self.tasks: dict[str, RuntimeTask] = {}
        self.preemptions: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.receipt_book = ReceiptBook()
        self.node_untrust: list[dict[str, Any]] = []
        self.cursor = 0
        self.round_seq = 0
        self.node_ids = [node.node_id for node in spec.nodes]
        self.node_quotas = {node.node_id: node.quota for node in spec.nodes}
        self.global_running = 0
        self.node_running: Counter[str] = Counter()
        self.threads: dict[str, threading.Thread] = {}
        self.worker_generation: dict[str, int] = {}
        self.timer_count = 0
        self.decision_keys: set[str] = set()
        self.preemption_serial = 0
        self.decision_serial = 0
        self.recovered = recovered_state is not None
        self.recovery_notes: list[dict[str, Any]] = []
        self.started_at_ms = self.clock_ms()
        self._initialize(recovered_state)

    def _initialize(self, recovered_state: dict[str, Any] | None) -> None:
        if recovered_state is None:
            for definition in self.spec.tasks:
                task = RuntimeTask(
                    task_id=definition.task_id,
                    config_version=definition.config_version,
                    node_id=definition.node_id,
                    priority=definition.priority,
                    total_shards=self.spec.config(definition.config_version).shards,
                    release_after_ms=definition.release_after_ms,
                    state=WAITING_RELEASE if definition.release_after_ms > 0 else PENDING,
                    wait_started_ms=None if definition.release_after_ms > 0 else self.clock_ms(),
                )
                self.tasks[task.task_id] = task
            return

        self.cursor = int(recovered_state.get("cursor", 0))
        self.round_seq = int(recovered_state.get("round_seq", 0))
        for task_id in sorted(recovered_state.get("tasks", {})):
            self.tasks[task_id] = RuntimeTask.from_dict(recovered_state["tasks"][task_id])
        self.preemptions = list(recovered_state.get("preemptions", []))
        self.decisions = list(recovered_state.get("decisions", []))
        self.attempts = list(recovered_state.get("attempts", []))
        for item in recovered_state.get("receipts", []):
            receipt = Receipt.from_dict(item)
            self.receipt_book.add(receipt)
        self.node_untrust = list(recovered_state.get("node_untrust", []))
        for item in self.decisions:
            self.decision_keys.add(
                f"{item.get('high_task_id')}|{item.get('victim_task_id')}|{item.get('verdict')}|{item.get('reason')}"
            )
        for item in self.preemptions:
            try:
                number = int(str(item.get("preemption_id", "p0")).lstrip("p"))
                self.preemption_serial = max(self.preemption_serial, number)
            except ValueError:
                pass
        for item in self.decisions:
            try:
                number = int(str(item.get("decision_id", "d0")).lstrip("d"))
                self.decision_serial = max(self.decision_serial, number)
            except ValueError:
                pass

        now = self.clock_ms()
        interrupted: list[str] = []
        for task_id in sorted(self.tasks):
            task = self.tasks[task_id]
            if task.state in ACTIVE_STATES:
                shard = len(task.completed_shards)
                if shard not in task.unknown_shards:
                    task.unknown_shards.append(shard)
                task.recovery_reruns += 1
                task.state = PENDING
                task.inflight = None
                task.slots_allocated = False
                task.preempt_requested_by = ""
                task.wait_started_ms = now
                interrupted.append(task.task_id)
            elif task.state in {BACKOFF, WAITING_RELEASE}:
                task.state = PENDING
                task.inflight = None
                task.slots_allocated = False
                task.wait_started_ms = now
            elif task.state == PAUSED and task.wait_started_ms is None:
                task.wait_started_ms = now
        for preemption in self.preemptions:
            if preemption.get("status") == "REQUESTED":
                preemption["status"] = "UNKNOWN_RERUN"
        if interrupted:
            note = {
                "kind": "RECOVERY_UNKNOWN",
                "tasks": sorted(interrupted),
                "detail": "uncommitted in-flight shards are unknown and will be re-run from the last committed boundary",
            }
            self.recovery_notes.append(note)
            if self.logger:
                self.logger.write("recovery_unknown", tasks=sorted(interrupted))

    def _can_allocate(self, task: RuntimeTask) -> bool:
        return (
            self.global_running < self.spec.budget.global_concurrency
            and self.node_running[task.node_id] < self.node_quotas[task.node_id]
        )

    def _allocate(self, task: RuntimeTask) -> None:
        if task.slots_allocated:
            return
        self.global_running += 1
        self.node_running[task.node_id] += 1
        task.slots_allocated = True

    def _release(self, task: RuntimeTask) -> None:
        if not task.slots_allocated:
            return
        self.global_running -= 1
        self.node_running[task.node_id] -= 1
        if self.global_running < 0 or self.node_running[task.node_id] < 0:
            raise LedgerError("E_SCHEDULER", "quota", "negative quota counter detected")
        task.slots_allocated = False

    def _ready_tasks(self) -> list[RuntimeTask]:
        return sorted(
            (task for task in self.tasks.values() if task.state in {PENDING, PAUSED}),
            key=lambda task: (-task.priority, task.task_id),
        )

    def _start_task(self, task: RuntimeTask) -> None:
        now = self.clock_ms()
        if task.wait_started_ms is not None:
            task.last_wait_started_ms = task.wait_started_ms
            task.last_wait_ended_ms = now
            duration = max(0, now - task.wait_started_ms)
            task.wait_ms += duration
            task.wait_windows.append(
                {
                    "started_ms": int(task.wait_started_ms),
                    "ended_ms": int(now),
                    "duration_ms": int(duration),
                }
            )
        task.wait_started_ms = None
        task.state = RUNNING
        task.preempt_requested_by = ""
        self._allocate(task)
        generation = self.worker_generation.get(task.task_id, 0) + 1
        self.worker_generation[task.task_id] = generation
        thread = threading.Thread(
            target=self._worker,
            args=(task.task_id, generation),
            name=f"dispatch-{task.task_id}-{generation}",
            daemon=True,
        )
        self.threads[task.task_id] = thread
        thread.start()
        if self.logger:
            self.logger.write(
                "task_started",
                task_id=task.task_id,
                node_id=task.node_id,
                priority=task.priority,
                resume_shard=len(task.completed_shards),
            )

    def _consider_preemption(self, high: RuntimeTask) -> bool:
        decision_count_before = len(self.decisions)
        lower = sorted(
            (
                task
                for task in self.tasks.values()
                if task.priority < high.priority and task.task_id != high.task_id
            ),
            key=lambda task: (task.priority, len(task.completed_shards), task.task_id),
        )
        for task in lower:
            if task.state == BACKOFF:
                self._record_decision(
                    high,
                    task,
                    "WAIT_EXEMPT",
                    "BACKOFF_DOES_NOT_HOLD_QUOTA",
                    False,
                    {"state": task.state, "retry_shard": task.backoff_for_shard, "backoff_ms": task.backoff_ms},
                )
            elif task.state in TERMINAL_STATES:
                self._record_decision(
                    high,
                    task,
                    "TERMINAL_NOT_PREEMPTABLE",
                    "TASK_ALREADY_TERMINAL",
                    False,
                    {"state": task.state, "error_code": task.error_code},
                )
            elif task.state == PAUSED:
                self._record_decision(
                    high,
                    task,
                    "ALREADY_SUSPENDED_NO_QUOTA",
                    "PAUSED_TASK_HOLDS_NO_QUOTA",
                    False,
                    {"completed_shards": len(task.completed_shards)},
                )

        running = [task for task in lower if task.state == RUNNING]
        selected: list[RuntimeTask] = []
        need_node = self.node_running[high.node_id] >= self.node_quotas[high.node_id]
        need_global = self.global_running >= self.spec.budget.global_concurrency
        if need_node:
            on_node = [task for task in running if task.node_id == high.node_id]
            if on_node:
                selected.append(on_node[0])
        if need_global:
            remaining_global = need_global and not any(True for _ in selected)
            if remaining_global:
                candidates = [task for task in running if task not in selected]
                if candidates:
                    selected.append(candidates[0])
        if not selected:
            if len(self.decisions) != decision_count_before:
                self._commit(
                    "PREEMPTION_WAIT_AUDIT",
                    {
                        "high_task_id": high.task_id,
                        "global_running": self.global_running,
                        "global_limit": self.spec.budget.global_concurrency,
                        "node_running": self.node_running[high.node_id],
                        "node_quota": self.node_quotas[high.node_id],
                    },
                )
            return False

        for victim in selected:
            if victim.state == PREEMPT_REQUESTED:
                continue
            self.preemption_serial += 1
            preemption_id = f"p{self.preemption_serial:06d}"
            record = {
                "preemption_id": preemption_id,
                "task_id": victim.task_id,
                "by_task_id": high.task_id,
                "requested_shard": len(victim.completed_shards),
                "suspended_shard": -1,
                "resources": [
                    {"resource": "global_slot", "node_id": victim.node_id, "quota": 1},
                    {
                        "resource": "node_slot",
                        "node_id": victim.node_id,
                        "slot": self.node_running[victim.node_id],
                        "quota": self.node_quotas[victim.node_id],
                    },
                ],
                "status": "REQUESTED",
                "created_at": utc_now(),
            }
            self.preemptions.append(record)
            victim.state = PREEMPT_REQUESTED
            victim.preempt_requested_by = high.task_id
            victim.preempt_count += 1
            self._record_decision(
                high,
                victim,
                "PREEMPTABLE",
                "LOWER_PRIORITY_HOLDS_REQUIRED_QUOTA",
                True,
                {
                    "preemption_id": preemption_id,
                    "victim_priority": victim.priority,
                    "high_priority": high.priority,
                    "victim_shards_completed": len(victim.completed_shards),
                    "checkpoint": "after current shard boundary",
                },
            )
        self._commit(
            "PREEMPTION_REQUESTED",
            {
                "high_task_id": high.task_id,
                "victims": [victim.task_id for victim in selected],
                "resources": [record["resources"] for record in self.preemptions[-len(selected) :]],
            },
        )
        return True

    def _record_decision(
        self,
        high: RuntimeTask,
        victim: RuntimeTask,
        verdict: str,
        reason: str,
        preemptable: bool,
        details: dict[str, Any],
    ) -> None:
        key = f"{high.task_id}|{victim.task_id}|{verdict}|{reason}"
        if key in self.decision_keys:
            return
        self.decision_keys.add(key)
        self.decision_serial += 1
        self.decisions.append(
            {
                "decision_id": f"d{self.decision_serial:06d}",
                "high_task_id": high.task_id,
                "victim_task_id": victim.task_id,
                "verdict": verdict,
                "reason": reason,
                "preemptable": bool(preemptable),
                "details": details,
            }
        )

    def _pause_task(self, task: RuntimeTask, *, status: str, failure: str = "") -> None:
        task.state = PAUSED
        task.wait_started_ms = self.clock_ms()
        task.preempt_requested_by = ""
        task.inflight = None
        self._release(task)
        checkpoint = len(task.completed_shards)
        for item in reversed(self.preemptions):
            if item["task_id"] == task.task_id and item["status"] in {"REQUESTED", "UNKNOWN_RERUN"}:
                item["status"] = status
                item["suspended_shard"] = checkpoint
                item["suspended_output_summary"] = task.last_output_summary
                if failure:
                    item["failure"] = failure
                break
        self._commit(
            "PREEMPTION_SUSPENDED",
            {
                "task_id": task.task_id,
                "suspended_shard": checkpoint,
                "completed_shards": len(task.completed_shards),
                "used_quota_ms": task.used_quota_ms,
                "last_output_summary": task.last_output_summary,
                "status": status,
            },
        )
        if self.logger:
            self.logger.write(
                "task_suspended",
                task_id=task.task_id,
                suspended_shard=checkpoint,
                status=status,
            )
        if self.snapshot_hook is not None:
            self.snapshot_hook(
                self._state_snapshot(self.round_seq, self.cursor),
                f"preemption-{task.task_id}-shard{checkpoint}",
            )

    def _worker(self, task_id: str, generation: int) -> None:
        while True:
            with self.cond:
                task = self.tasks[task_id]
                if self.worker_generation.get(task_id) != generation or task.state != RUNNING:
                    return
                if len(task.completed_shards) >= task.total_shards:
                    self._finish_task(task)
                    self.cond.notify_all()
                    return
                if task.preempt_requested_by:
                    self._pause_task(task, status="SUSPENDED")
                    self.cond.notify_all()
                    return
                shard_index = len(task.completed_shards)
                failures = task.retries_by_shard.get(shard_index, 0)
                attempt_no = failures + 1
                definition = self.spec.task(task.task_id)
                request_id = request_id_for(self.session_id, definition, shard_index)
                started_ms = self.clock_ms()
                attempt_key = f"{request_id}|a{attempt_no}"
                attempt_record = {
                    "attempt_key": attempt_key,
                    "task_id": task.task_id,
                    "shard_index": shard_index,
                    "attempt_no": attempt_no,
                    "request_id": request_id,
                    "error_code": "",
                    "compute_ms": 0,
                    "started_at_ms": started_ms,
                    "finished_at_ms": started_ms,
                }
                self.attempts.append(attempt_record)
                task.attempts_total += 1
                task.inflight = {
                    "shard_index": shard_index,
                    "attempt_no": attempt_no,
                    "request_id": request_id,
                    "attempt_key": attempt_key,
                    "started_at_ms": started_ms,
                }
                self._commit(
                    "DISPATCH_INFLIGHT",
                    {
                        "task_id": task.task_id,
                        "node_id": task.node_id,
                        "shard_index": shard_index,
                        "attempt_no": attempt_no,
                        "request_id": request_id,
                    },
                )

            try:
                result = self.dispatcher.dispatch(definition, shard_index, request_id)
            except DispatchError as exc:
                with self.cond:
                    finished_ms = self.clock_ms()
                    attempt_record["error_code"] = exc.code
                    attempt_record["finished_at_ms"] = finished_ms
                    task.inflight = None
                    task.error_code = exc.code
                    task.error_detail = exc.detail
                    task.retries_by_shard[shard_index] = failures + 1
                    task.retries_total += 1
                    self._release(task)
                    if task.preempt_requested_by:
                        self._pause_task(task, status="SUSPENDED_AT_FAILURE", failure=exc.code)
                        self.cond.notify_all()
                        return
                    if exc.retryable and failures + 1 <= self.spec.budget.max_retries:
                        delay = self.spec.budget.retry_backoff_ms[failures]
                        task.state = BACKOFF
                        task.backoff_for_shard = shard_index
                        task.backoff_index = failures
                        task.backoff_ms = delay
                        task.backoff_started_ms = finished_ms
                        task.backoff_ended_ms = None
                        self._commit(
                            "DISPATCH_BACKOFF",
                            {
                                "task_id": task.task_id,
                                "shard_index": shard_index,
                                "attempt_no": attempt_no,
                                "error_code": exc.code,
                                "backoff_ms": delay,
                                "holds_quota": False,
                            },
                        )
                        self._start_backoff_timer(task.task_id, generation, delay)
                        self.cond.notify_all()
                        return
                    if exc.retryable:
                        task.state = EXHAUSTED
                        task.error_code = "E_EXHAUST"
                        task.error_detail = (
                            f"retry sequence exhausted after {attempt_no} attempts; last error {exc.code}: {exc.detail}"
                        )
                        kind = "TASK_EXHAUSTED"
                    else:
                        task.state = FAILED
                        task.error_code = exc.code
                        task.error_detail = exc.detail
                        kind = "TASK_FAILED"
                    self._commit(
                        kind,
                        {
                            "task_id": task.task_id,
                            "shard_index": shard_index,
                            "attempt_no": attempt_no,
                            "error_code": task.error_code,
                            "detail": task.error_detail,
                        },
                    )
                    self.cond.notify_all()
                    return
            except Exception as exc:  # protect the coordinator from a dispatcher bug
                with self.cond:
                    task.inflight = None
                    task.error_code = "E_INTERNAL"
                    task.error_detail = repr(exc)
                    task.state = FAILED
                    self._release(task)
                    self._commit("TASK_FAILED", {"task_id": task.task_id, "error_code": "E_INTERNAL", "detail": repr(exc)})
                    self.cond.notify_all()
                    return

            with self.cond:
                task = self.tasks[task_id]
                if self.worker_generation.get(task_id) != generation or task.state not in {RUNNING, PREEMPT_REQUESTED}:
                    return
                finished_ms = self.clock_ms()
                attempt_record["finished_at_ms"] = finished_ms
                attempt_record["compute_ms"] = int(result.compute_ms)
                checkpoint = {
                    "shard_index": shard_index,
                    "output_summary": result.output_summary,
                    "input_digest": result.input_digest,
                    "compute_ms": int(result.compute_ms),
                    "dispatched_ms": int(result.dispatched_ms),
                    "attempts": attempt_no,
                    "items_used": sorted(result.items_used),
                }
                task.completed_shards.append(checkpoint)
                task.completed_shards.sort(key=lambda item: int(item["shard_index"]))
                task.last_output_summary = result.output_summary
                task.used_quota_ms += max(0, int(result.dispatched_ms))
                task.inflight = None
                config = self.spec.config(task.config_version)
                self.receipt_book.add_result(
                    node_id=task.node_id,
                    task_id=task.task_id,
                    config_version=task.config_version,
                    shard_index=shard_index,
                    shard_count=config.shards,
                    item_keys=result.items_used,
                    input_digest=result.input_digest,
                    output_summary=result.output_summary,
                    attempt=attempt_no,
                )
                self._commit(
                    "SHARD_CHECKPOINT",
                    {
                        "task_id": task.task_id,
                        "node_id": task.node_id,
                        "shard_index": shard_index,
                        "completed_shards": len(task.completed_shards),
                        "output_summary": result.output_summary,
                        "compute_ms": int(result.compute_ms),
                    },
                )
                if len(task.completed_shards) >= task.total_shards:
                    self._finish_task(task)
                    self.cond.notify_all()
                    return
                if task.preempt_requested_by:
                    self._pause_task(task, status="SUSPENDED")
                    self.cond.notify_all()
                    return
                self.cond.notify_all()

    def _start_backoff_timer(self, task_id: str, generation: int, delay_ms: int) -> None:
        self.timer_count += 1

        def run() -> None:
            try:
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                with self.cond:
                    task = self.tasks.get(task_id)
                    if task is None or task.state != BACKOFF or self.worker_generation.get(task_id) != generation:
                        return
                    task.state = PENDING
                    task.backoff_ended_ms = self.clock_ms()
                    task.wait_started_ms = task.backoff_ended_ms
                    task.inflight = None
                    self._commit(
                        "BACKOFF_EXPIRED",
                        {
                            "task_id": task_id,
                            "backoff_ms": task.backoff_ms,
                            "backoff_index": task.backoff_index,
                            "retry_shard": task.backoff_for_shard,
                            "holds_quota": False,
                        },
                    )
                    self.cond.notify_all()
            finally:
                with self.cond:
                    self.timer_count -= 1
                    self.cond.notify_all()

        threading.Thread(target=run, name=f"backoff-{task_id}-{generation}", daemon=True).start()

    def _start_release_timer(self, task_id: str, delay_ms: int) -> None:
        self.timer_count += 1

        def run() -> None:
            try:
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                with self.cond:
                    task = self.tasks.get(task_id)
                    if task is None or task.state != WAITING_RELEASE:
                        return
                    task.state = PENDING
                    task.wait_started_ms = self.clock_ms()
                    self._commit("TASK_RELEASED", {"task_id": task_id, "release_after_ms": task.release_after_ms})
                    self.cond.notify_all()
            finally:
                with self.cond:
                    self.timer_count -= 1
                    self.cond.notify_all()

        threading.Thread(target=run, name=f"release-{task_id}", daemon=True).start()

    def _finish_task(self, task: RuntimeTask) -> None:
        task.state = DONE
        task.output_summary = aggregate_output_summaries(
            [str(item["output_summary"]) for item in sorted(task.completed_shards, key=lambda row: int(row["shard_index"]))]
        )
        task.error_code = ""
        task.error_detail = ""
        task.inflight = None
        task.preempt_requested_by = ""
        self._release(task)
        for item in reversed(self.preemptions):
            if item["task_id"] == task.task_id and item["status"] in {"REQUESTED", "UNKNOWN_RERUN"}:
                item["status"] = "RETIRED_AT_COMPLETION"
                item["suspended_shard"] = len(task.completed_shards)
                item["suspended_output_summary"] = task.last_output_summary
                break
        self._commit(
            "TASK_DONE",
            {
                "task_id": task.task_id,
                "node_id": task.node_id,
                "output_summary": task.output_summary,
                "wait_ms": task.wait_ms,
                "used_quota_ms": task.used_quota_ms,
            },
        )
        if self.logger:
            self.logger.write("task_done", task_id=task.task_id, output_summary=task.output_summary)

    def _state_snapshot(self, next_round_seq: int, next_cursor: int) -> dict[str, Any]:
        audited = self.receipt_book.replace_with_audited(self.spec.current_version)
        self.node_untrust = build_node_untrust(audited, self.node_ids, self.spec.current_version)
        return {
            "format_version": 1,
            "session_id": self.session_id,
            "spec_version": self.spec.spec_version,
            "spec_digest": self.spec.digest,
            "budget_digest": self.spec.budget_digest,
            "current_version": self.spec.current_version,
            "cursor": next_cursor,
            "round_seq": next_round_seq,
            "tasks": {task_id: self.tasks[task_id].to_dict() for task_id in sorted(self.tasks)},
            "preemptions": self.preemptions,
            "decisions": self.decisions,
            "attempts": self.attempts,
            "receipts": [receipt.to_dict() for receipt in audited],
            "node_untrust": self.node_untrust,
            "recovery_notes": self.recovery_notes,
        }

    def _commit(self, kind: str, event: dict[str, Any]) -> None:
        next_round = self.round_seq + 1
        next_cursor = self.cursor + 1
        state = self._state_snapshot(next_round, next_cursor)
        self.ledger.commit_state(self.session_id, state, kind, event)
        self.round_seq = next_round
        self.cursor = next_cursor

    def run(self) -> SchedulerSummary:
        with self.cond:
            self._commit(
                "SCHEDULER_READY",
                {
                    "recovered": self.recovered,
                    "recovery_notes": self.recovery_notes,
                    "global_concurrency": self.spec.budget.global_concurrency,
                    "node_quotas": {node.node_id: node.quota for node in self.spec.nodes},
                },
            )
            if self.snapshot_hook is not None:
                self.snapshot_hook(self._state_snapshot(self.round_seq, self.cursor), "session-start")
            for task in self.tasks.values():
                if task.state == WAITING_RELEASE:
                    delay = task.release_after_ms
                    self._start_release_timer(task.task_id, delay)

            while True:
                if all(task.state in TERMINAL_STATES for task in self.tasks.values()):
                    break
                made_progress = False
                while True:
                    ready = self._ready_tasks()
                    if not ready:
                        break
                    candidate = ready[0]
                    if self._can_allocate(candidate):
                        self._start_task(candidate)
                        made_progress = True
                        continue
                    self._consider_preemption(candidate)
                    break
                if all(task.state in TERMINAL_STATES for task in self.tasks.values()):
                    break
                if not made_progress and self.global_running == 0 and self.timer_count == 0:
                    blocked = [task.to_dict() for task in sorted(self.tasks.values(), key=lambda item: item.task_id)]
                    raise LedgerError(
                        "E_SCHEDULER",
                        "scheduler.deadlock",
                        "no runnable or timer-backed task remains",
                        {"tasks": blocked},
                    )
                self.cond.wait(timeout=0.02)

            self._commit("SESSION_FINALIZED", {"tasks": len(self.tasks)})
            return self.summary()

    def summary(self) -> SchedulerSummary:
        tasks = [self.tasks[task_id].to_dict() for task_id in sorted(self.tasks)]
        receipts = [receipt.to_dict() for receipt in self.receipt_book.values()]
        deterministic = digest_json(
            {
                "tasks": [
                    {
                        "task_id": item["task_id"],
                        "state": item["state"],
                        "output_summary": item["output_summary"],
                        "completed_shards": [
                            {"shard_index": shard["shard_index"], "output_summary": shard["output_summary"]}
                            for shard in sorted(item["completed_shards"], key=lambda row: int(row["shard_index"]))
                        ],
                        "error_code": item["error_code"],
                    }
                    for item in tasks
                ],
                "receipts": [
                    {
                        "receipt_key": item["receipt_key"],
                        "output_summary": item["output_summary"],
                        "generated_version": item["generated_version"],
                    }
                    for item in sorted(receipts, key=lambda row: row["receipt_key"])
                ],
                "node_untrust": [
                    {
                        "node_id": item["node_id"],
                        "score": item["score"],
                        "rank": item["rank"],
                        "untrusted": item["untrusted_receipt_keys"],
                    }
                    for item in self.node_untrust
                ],
                "preemptions": [
                    {
                        "task_id": item["task_id"],
                        "by_task_id": item["by_task_id"],
                        "requested_shard": item["requested_shard"],
                        "suspended_shard": item["suspended_shard"],
                        "status": item["status"],
                    }
                    for item in self.preemptions
                ],
            }
        )
        return SchedulerSummary(
            session_id=self.session_id,
            tasks=tasks,
            preemptions=self.preemptions,
            decisions=self.decisions,
            receipts=receipts,
            node_untrust=self.node_untrust,
            deterministic_digest=deterministic,
            recovered=self.recovered,
            recovery_notes=self.recovery_notes,
        )
