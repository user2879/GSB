"""Parallel dispatch scheduler constrained by units, concurrency, and leases."""
from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .client import NodeClient
from .errors import BusinessError, StructuredLogger
from .ledger import Ledger
from .leases import Lease, LeaseManager
from .spec import ShardSpec, Spec
from .util import deterministic_samples, monotonic_ms


@dataclass
class TaskState:
    shard: ShardSpec
    status: str = "unrequested"
    requested: bool = False
    requested_generation: int = 1
    result_generation: int = 0
    dispatch_event_seq: int = 0
    revision_version: int = 1
    ttl_ms: int = 0
    summary: dict[str, Any] | None = None
    lease: Lease | None = None
    accounting_held: bool = False
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    attempts: int = 0
    cancel_reason: str | None = None
    error: dict[str, Any] | None = None
    worker_thread: str | None = None
    future: Future[Any] | None = None


@dataclass
class Launch:
    task: TaskState
    lease: Lease
    node_client: NodeClient


class Scheduler:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        clients: Mapping[str, NodeClient],
        logger: StructuredLogger,
        *,
        global_budget: int,
        node_limits: Mapping[str, Mapping[str, int]],
        revision_version: int,
        max_attempts: int = 2,
        after_commit: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.spec = spec
        self.ledger = ledger
        self.clients = dict(clients)
        self.logger = logger
        self.global_budget = int(global_budget)
        self.node_limits = {
            node_id: {
                "concurrency": int(values["concurrency"]),
                "lease_limit": int(values["lease_limit"]),
                "capacity_version": int(values["capacity_version"]),
            }
            for node_id, values in sorted(node_limits.items())
        }
        self.revision_version = int(revision_version)
        self.max_attempts = int(max_attempts)
        self.after_commit = after_commit
        self.tasks: dict[str, TaskState] = {
            shard_id: TaskState(shard=spec.shards[shard_id]) for shard_id in sorted(spec.shards)
        }
        self._cv = threading.Condition(threading.RLock())
        self._leases = LeaseManager(on_expired=self._on_lease_expired, scan_interval_s=0.005)
        self._executor = ThreadPoolExecutor(max_workers=max(1, sum(int(v["concurrency"]) for v in self.node_limits.values())), thread_name_prefix="quota-worker")
        self._closed = False
        self._paused = False
        self._shutdown_requested = False
        self._used_units = 0
        self._node_active = {node_id: 0 for node_id in self.node_limits}
        self._node_units = {node_id: 0 for node_id in self.node_limits}
        self.max_units_in_flight = 0
        self.max_node_active = {node_id: 0 for node_id in self.node_limits}
        self.max_node_units = {node_id: 0 for node_id in self.node_limits}
        self._recent_waits: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._load_ledger()

    # ------------------------------------------------------------- state sync
    def _load_ledger(self) -> None:
        results = self.ledger.results()
        requests = self.ledger.dispatch_requests()
        cancelled = self.ledger.cancelled_shards()
        with self._cv:
            for shard_id, task in self.tasks.items():
                result = results.get(shard_id)
                if result is not None:
                    task.status = "completed"
                    task.result_generation = int(result["generation"])
                    try:
                        task.summary = json.loads(result["summary_json"])
                    except json.JSONDecodeError:
                        task.summary = None
                request = requests.get(shard_id)
                if request is not None:
                    task.dispatch_event_seq = int(request["event_seq"])
                    task.requested_generation = int(request["generation"])
                    task.revision_version = int(request["revision_version"])
                    task.ttl_ms = int(request.get("ttl_ms", 0)) or self.spec.default_lease_ttl_ms
                    task.requested = bool(request["requested"])
                cancel = cancelled.get(shard_id)
                if cancel is not None:
                    task.cancel_reason = str(cancel["reason"])
                    if result is None:
                        task.status = "cancelled"
                if result is not None and (request is None or task.requested_generation <= task.result_generation):
                    task.status = "completed"
                    task.requested = bool(request and request["requested"])
                elif task.requested and result is not None and task.requested_generation > task.result_generation:
                    task.status = "blocked"
                elif task.requested:
                    task.status = "blocked"
                elif result is None and cancel is not None:
                    task.status = "cancelled"
            self._leases.start_monitor()

    def sync_from_ledger(self) -> None:
        """Refresh task state while no workers are active (configuration barrier)."""
        self._load_ledger_state_only()

    def _load_ledger_state_only(self) -> None:
        results = self.ledger.results()
        requests = self.ledger.dispatch_requests()
        cancelled = self.ledger.cancelled_shards()
        with self._cv:
            for shard_id, task in self.tasks.items():
                if task.accounting_held:
                    continue
                result = results.get(shard_id)
                request = requests.get(shard_id)
                cancel = cancelled.get(shard_id)
                if request is not None:
                    task.dispatch_event_seq = int(request["event_seq"])
                    task.requested_generation = int(request["generation"])
                    task.revision_version = int(request["revision_version"])
                    task.ttl_ms = int(request.get("ttl_ms", 0)) or self.spec.default_lease_ttl_ms
                    task.requested = bool(request["requested"])
                task.cancel_reason = str(cancel["reason"]) if cancel is not None else None
                if result is not None:
                    task.result_generation = int(result["generation"])
                    try:
                        task.summary = json.loads(result["summary_json"])
                    except json.JSONDecodeError:
                        task.summary = None
                    if request is None or task.requested_generation <= task.result_generation:
                        task.status = "completed"
                    else:
                        task.status = "blocked"
                elif cancel is not None:
                    task.status = "cancelled"
                elif task.requested:
                    task.status = "blocked"
                else:
                    task.status = "unrequested"
                    task.summary = None
            self._cv.notify_all()

    def update_config(
        self,
        *,
        global_budget: int,
        node_limits: Mapping[str, Mapping[str, int]],
        revision_version: int,
    ) -> None:
        with self._cv:
            self.global_budget = int(global_budget)
            self.revision_version = int(revision_version)
            self.node_limits = {
                node_id: {
                    "concurrency": int(values["concurrency"]),
                    "lease_limit": int(values["lease_limit"]),
                    "capacity_version": int(values["capacity_version"]),
                }
                for node_id, values in sorted(node_limits.items())
            }
            for node_id in self.node_limits:
                self._node_active.setdefault(node_id, 0)
                self._node_units.setdefault(node_id, 0)
                self.max_node_active.setdefault(node_id, 0)
                self.max_node_units.setdefault(node_id, 0)
            self._cv.notify_all()

    # ------------------------------------------------------------- scheduling
    def _deps_complete_locked(self, task: TaskState) -> bool:
        for dep_id in task.shard.dependencies:
            dep = self.tasks[dep_id]
            if dep.status != "completed" or dep.summary is None:
                return False
        return True

    def _candidate_locked(self, task: TaskState) -> bool:
        if task.status in {"completed", "cancelled", "failed", "cancelled_dependency", "running", "committing"}:
            return False
        if not task.requested:
            return False
        if task.result_generation >= task.requested_generation and task.summary is not None:
            return False
        return self._deps_complete_locked(task)

    def _has_capacity_locked(self, task: TaskState) -> bool:
        node = self.node_limits[task.shard.node_id]
        if self._used_units + task.shard.units > self.global_budget:
            return False
        if self._node_active.get(task.shard.node_id, 0) + 1 > node["concurrency"]:
            return False
        if self._leases.node_counts(task.shard.node_id) + 1 > node["lease_limit"]:
            return False
        return True

    def _refresh_ready_locked(self) -> list[TaskState]:
        ready: list[TaskState] = []
        for shard_id in sorted(self.tasks):
            task = self.tasks[shard_id]
            if task.status in {"completed", "cancelled", "failed", "cancelled_dependency", "running", "committing"}:
                continue
            if not task.requested:
                task.status = "unrequested" if task.result_generation == 0 else "completed"
                continue
            if task.result_generation >= task.requested_generation and task.summary is not None:
                task.status = "completed"
                continue
            if not self._deps_complete_locked(task):
                task.status = "blocked"
                continue
            task.status = "queued" if not self._has_capacity_locked(task) else "ready"
            if task.status == "queued" and task.wait_start_ms is None:
                task.wait_start_ms = monotonic_ms()
            ready.append(task)
        return ready

    def pump(self) -> int:
        launches: list[Launch] = []
        with self._cv:
            if self._closed or self._paused or self._shutdown_requested:
                return 0
            ready = self._refresh_ready_locked()
            for task in ready:
                if not self._has_capacity_locked(task):
                    continue
                generation = max(task.requested_generation, task.result_generation + 1)
                task.requested_generation = generation
                lease_id = f"lease:{task.shard.id}:g{generation}"
                try:
                    lease = self._leases.issue(
                        lease_id=lease_id,
                        shard_id=task.shard.id,
                        node_id=task.shard.node_id,
                        config_version=task.revision_version,
                        generation=generation,
                        ttl_ms=task.ttl_ms or self.spec.default_lease_ttl_ms,
                    )
                except BusinessError as exc:
                    task.error = exc.to_dict()
                    self.logger.write_error(exc)
                    continue
                if task.wait_start_ms is not None and task.wait_end_ms is None:
                    task.wait_end_ms = monotonic_ms()
                    self._recent_waits.append(
                        {
                            "shard_id": task.shard.id,
                            "node_id": task.shard.node_id,
                            "wait_start_ms": task.wait_start_ms,
                            "wait_end_ms": task.wait_end_ms,
                        }
                    )
                try:
                    self.ledger.reserve_dispatch(
                        event_seq=max(1, task.dispatch_event_seq),
                        shard_id=task.shard.id,
                        generation=generation,
                        node_id=task.shard.node_id,
                        lease_id=lease.lease_id,
                        granted_ms=lease.granted_ms,
                        expires_ms=lease.expires_ms,
                        config_version=task.revision_version,
                        units=task.shard.units,
                    )
                except BusinessError as exc:
                    self._leases.cancel(lease.lease_id)
                    task.error = exc.to_dict()
                    self.logger.write_error(exc)
                    continue
                task.status = "running"
                task.lease = lease
                task.accounting_held = True
                task.attempts += 1
                self._used_units += task.shard.units
                self._node_active[task.shard.node_id] = self._node_active.get(task.shard.node_id, 0) + 1
                self._node_units[task.shard.node_id] = self._node_units.get(task.shard.node_id, 0) + task.shard.units
                self.max_units_in_flight = max(self.max_units_in_flight, self._used_units)
                self.max_node_active[task.shard.node_id] = max(
                    self.max_node_active.get(task.shard.node_id, 0),
                    self._node_active[task.shard.node_id],
                )
                self.max_node_units[task.shard.node_id] = max(
                    self.max_node_units.get(task.shard.node_id, 0),
                    self._node_units[task.shard.node_id],
                )
                launches.append(Launch(task=task, lease=lease, node_client=self.clients[task.shard.node_id]))
            self._cv.notify_all()
        for launch in launches:
            try:
                launch.task.future = self._executor.submit(self._run_task, launch)
            except RuntimeError as exc:
                self._fail_launch(launch, BusinessError("E_LEDGER", f"worker pool rejected task: {exc}", shard_id=launch.task.shard.id))
        return len(launches)

    def _fail_launch(self, launch: Launch, error: BusinessError) -> None:
        self.ledger.release_reservation(
            shard_id=launch.task.shard.id,
            lease_id=launch.lease.lease_id,
            units=launch.task.shard.units,
            state="failed",
            detail=error.message,
        )
        self._leases.cancel(launch.lease.lease_id)
        with self._cv:
            self._release_accounting_locked(launch.task)
            launch.task.status = "failed"
            launch.task.error = error.to_dict()
            self._errors.append(error.to_dict())
            self._cv.notify_all()

    def _release_accounting_locked(self, task: TaskState) -> None:
        if not task.accounting_held:
            return
        task.accounting_held = False
        self._used_units = max(0, self._used_units - task.shard.units)
        self._node_active[task.shard.node_id] = max(0, self._node_active.get(task.shard.node_id, 0) - 1)
        self._node_units[task.shard.node_id] = max(0, self._node_units.get(task.shard.node_id, 0) - task.shard.units)

    def _run_task(self, launch: Launch) -> None:
        task = launch.task
        lease = launch.lease
        node_client = launch.node_client
        task.worker_thread = threading.current_thread().name
        acquired_node_lease = False
        try:
            with self._cv:
                if task.status != "running" or task.lease is None or task.lease.lease_id != lease.lease_id:
                    return
            node_client.acquire_lease(
                lease_id=lease.lease_id,
                shard_id=task.shard.id,
                ttl_ms=task.ttl_ms or self.spec.default_lease_ttl_ms,
            )
            acquired_node_lease = True
            with self._cv:
                if task.status != "running" or not self._leases.during_lease(lease.lease_id):
                    raise BusinessError("E_LEASE", "lease expired before compute", field="lease_id", shard_id=task.shard.id)
            revision = self.spec.revisions[task.revision_version]
            samples = deterministic_samples(
                self.spec.nodes[task.shard.node_id].seed,
                task.shard.seed_offset,
                task.shard.sample_count,
                int(revision.aggregation["group_modulus"]),
            )
            reply = node_client.compute(
                shard_id=task.shard.id,
                config_version=task.revision_version,
                lease_id=lease.lease_id,
                lease_version=lease.version,
                samples=samples,
                scale=int(revision.aggregation["scale"]),
                group_modulus=int(revision.aggregation["group_modulus"]),
            )
            # Claim the local lease atomically against the expiry monitor before
            # committing. A lease that crossed its real deadline cannot commit.
            with self._cv:
                if task.status != "running" or not self._leases.complete(lease.lease_id, expected_version=lease.version):
                    raise BusinessError("E_LEASE", "lease expired before result commit", field="lease_id", shard_id=task.shard.id)
                task.status = "committing"
            committed = self.ledger.commit_result(
                shard_id=task.shard.id,
                outcome="completed",
                summary=reply["summary"],
                node_id=task.shard.node_id,
                lease_id=lease.lease_id,
                lease_version=lease.version,
                config_version=task.revision_version,
                resource_units=task.shard.units,
                generation=task.requested_generation,
                completion_event_seq=max(1, task.dispatch_event_seq),
                elapsed_monotonic_ms=int(reply["elapsed_monotonic_ms"]),
                wait_start_ms=task.wait_start_ms,
                wait_end_ms=task.wait_end_ms,
            )
            with self._cv:
                task.status = "completed"
                task.summary = reply["summary"]
                task.result_generation = committed["generation"]
                task.requested = False
                self._release_accounting_locked(task)
                task.lease = None
                self._cv.notify_all()
            try:
                node_client.release_lease(lease_id=lease.lease_id)
            except BusinessError:
                pass
            if self.after_commit is not None:
                self.after_commit(
                    {
                        "shard_id": task.shard.id,
                        "generation": committed["generation"],
                        "ledger_digest": committed["ledger_digest"],
                        "result_digest": committed["result_digest"],
                        "monotonic_ms": committed["completion_monotonic_ms"],
                    }
                )
        except BusinessError as exc:
            if acquired_node_lease:
                try:
                    node_client.release_lease(lease_id=lease.lease_id)
                except BusinessError:
                    pass
            self._handle_worker_error(task, lease, exc)
        except Exception as exc:  # unexpected local failure must not strand quota
            self._handle_worker_error(
                task,
                lease,
                BusinessError("E_LEDGER", f"worker failure: {type(exc).__name__}: {exc}", shard_id=task.shard.id),
            )

    def _handle_worker_error(self, task: TaskState, lease: Lease, error: BusinessError) -> None:
        self.logger.write_error(error)
        with self._cv:
            if task.status == "completed":
                return
            if task.status not in {"running", "committing", "lease_expired"}:
                if task.accounting_held:
                    self._release_accounting_locked(task)
                self._cv.notify_all()
                return
            can_retry = (
                not task.cancel_reason
                and task.attempts < self.max_attempts
                and error.code in {"E_TIMEOUT", "E_PORT", "E_LEASE"}
            )
            self._release_accounting_locked(task)
            task.lease = None
            task.error = error.to_dict()
            self._errors.append(error.to_dict())
            task.status = "blocked" if can_retry else "failed"
            if not can_retry:
                task.requested = False
            self._cv.notify_all()
        self.ledger.release_reservation(
            shard_id=task.shard.id,
            lease_id=lease.lease_id,
            units=task.shard.units,
            state="failed",
            detail=error.message,
        )
        self._leases.cancel(lease.lease_id)

    def _on_lease_expired(self, lease: Lease) -> None:
        with self._cv:
            task = self.tasks.get(lease.shard_id)
            if task is None or task.lease is None or task.lease.lease_id != lease.lease_id:
                return
            if task.status not in {"running", "committing"}:
                return
            self._release_accounting_locked(task)
            task.lease = None
            task.error = {
                "code": "E_LEASE",
                "message": "lease expired by real monotonic clock",
                "shard_id": task.shard.id,
                "monotonic_ms": monotonic_ms(),
            }
            self._errors.append(task.error)
            task.status = "blocked"
            self._cv.notify_all()
        self.ledger.release_reservation(
            shard_id=task.shard.id,
            lease_id=lease.lease_id,
            units=task.shard.units,
            state="expired",
            detail="real monotonic lease expiry",
        )
        self.logger.write(
            "E_LEASE",
            "lease expired by real monotonic clock",
            shard_id=task.shard.id,
            extra={"lease_id": lease.lease_id},
        )
        self.pump()

    # ---------------------------------------------------------------- control
    def request_shutdown(self, *, pause: bool = True) -> None:
        with self._cv:
            self._shutdown_requested = True
            if pause:
                self._paused = True
            self._cv.notify_all()

    def resume(self) -> None:
        with self._cv:
            self._paused = False
            self._cv.notify_all()

    def wait_epoch(self, *, stop_event: threading.Event | None = None) -> str:
        """Wait until runnable work in the current event batch is drained."""
        while True:
            self.pump()
            with self._cv:
                running = sum(1 for task in self.tasks.values() if task.status in {"running", "committing"})
                ready = [
                    task
                    for shard_id in sorted(self.tasks)
                    for task in [self.tasks[shard_id]]
                    if self._candidate_locked(task) or task.status == "queued"
                ]
                if stop_event is not None and stop_event.is_set():
                    self.request_shutdown()
                if self._shutdown_requested:
                    if running == 0:
                        return "shutdown"
                elif running == 0 and not ready:
                    return "settled"
                self._cv.wait(timeout=0.05)

    def mark_unreachable(self) -> list[str]:
        marked: list[str] = []
        with self._cv:
            changed = True
            while changed:
                changed = False
                for shard_id in sorted(self.tasks):
                    task = self.tasks[shard_id]
                    if task.status not in {"blocked", "queued", "ready"} or not task.requested:
                        continue
                    if any(self.tasks[dep].status in {"cancelled", "failed", "cancelled_dependency"} for dep in task.shard.dependencies):
                        task.status = "cancelled_dependency"
                        task.requested = False
                        marked.append(shard_id)
                        changed = True
            self._cv.notify_all()
        return marked

    def status_snapshot(self) -> dict[str, Any]:
        with self._cv:
            ready = [
                shard_id
                for shard_id in sorted(self.tasks)
                if self.tasks[shard_id].status == "ready"
            ]
            queued = [
                shard_id
                for shard_id in sorted(self.tasks)
                if self.tasks[shard_id].status == "queued"
            ]
            blocked = [
                shard_id
                for shard_id in sorted(self.tasks)
                if self.tasks[shard_id].status in {"blocked", "cancelled_dependency"}
            ]
            running = [
                {
                    "shard_id": shard_id,
                    "node_id": self.tasks[shard_id].shard.node_id,
                    "lease_id": self.tasks[shard_id].lease.lease_id if self.tasks[shard_id].lease else None,
                    "status": self.tasks[shard_id].status,
                    "worker_thread": self.tasks[shard_id].worker_thread,
                }
                for shard_id in sorted(self.tasks)
                if self.tasks[shard_id].status in {"running", "committing"}
            ]
            task_rows = []
            for shard_id in sorted(self.tasks):
                task = self.tasks[shard_id]
                task_rows.append(
                    {
                        "shard_id": shard_id,
                        "node_id": task.shard.node_id,
                        "status": task.status,
                        "requested": task.requested,
                        "requested_generation": task.requested_generation,
                        "result_generation": task.result_generation,
                        "dependencies": list(task.shard.dependencies),
                        "wait_start_ms": task.wait_start_ms,
                        "wait_end_ms": task.wait_end_ms,
                        "lease": task.lease.public() if task.lease else None,
                        "worker_thread": task.worker_thread,
                        "cancel_reason": task.cancel_reason,
                        "error": task.error,
                    }
                )
            return {
                "ready": ready,
                "queued": queued,
                "blocked": blocked,
                "running": running,
                "tasks": task_rows,
                "global_budget": self.global_budget,
                "used_units": self._used_units,
                "node_active": {node_id: self._node_active.get(node_id, 0) for node_id in sorted(self.node_limits)},
                "node_units": {node_id: self._node_units.get(node_id, 0) for node_id in sorted(self.node_limits)},
                "node_limits": {
                    node_id: dict(values) for node_id, values in sorted(self.node_limits.items())
                },
                "leases": self._leases.snapshot(),
                "max_units_in_flight": self.max_units_in_flight,
                "max_node_active": {node_id: self.max_node_active[node_id] for node_id in sorted(self.max_node_active)},
                "max_node_units": {node_id: self.max_node_units[node_id] for node_id in sorted(self.max_node_units)},
                "wait_events": list(self._recent_waits),
                "errors": list(self._errors),
            }

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._shutdown_requested = True
            self._cv.notify_all()
        self._leases.stop_monitor()
        self._executor.shutdown(wait=True)
