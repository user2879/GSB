from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BusinessError, as_business_error
from .lease import LeaseManager, LeaseRecord
from .logging_utils import StructuredLogger
from .util import monotonic_ms


JobFunction = Callable[[], Any]
LeaseLostCallback = Callable[[LeaseRecord, BusinessError], None]


@dataclass
class JobSpec:
    view_id: str
    kind: str
    param_version: int
    refresh_version: int
    ttl_ms: int
    renew_ms: int
    fn: JobFunction
    batch_id: str | None = None
    per_view_quota: int = 1
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class JobTrace:
    job_id: str
    batch_id: str | None
    view_id: str
    kind: str
    param_version: int
    refresh_version: int
    enqueued_mono_ms: int
    started_mono_ms: int | None = None
    finished_mono_ms: int | None = None
    wait_ms: int | None = None
    run_ms: int | None = None
    status: str = "queued"
    error_code: str | None = None
    error_message: str | None = None
    worker_thread: str | None = None
    lease_state: str | None = None
    lease_expires_mono_ms: int | None = None
    global_limit: int = 0
    per_view_quota: int = 1

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "batch_id": self.batch_id,
            "view_id": self.view_id,
            "kind": self.kind,
            "param_version": self.param_version,
            "refresh_version": self.refresh_version,
            "enqueued_mono_ms": self.enqueued_mono_ms,
            "started_mono_ms": self.started_mono_ms,
            "finished_mono_ms": self.finished_mono_ms,
            "wait_ms": self.wait_ms,
            "run_ms": self.run_ms,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "worker_thread": self.worker_thread,
            "lease_state": self.lease_state,
            "lease_expires_mono_ms": self.lease_expires_mono_ms,
            "global_limit": self.global_limit,
            "per_view_quota": self.per_view_quota,
        }


@dataclass
class JobHandle:
    spec: JobSpec
    future: Future
    trace: JobTrace

    @property
    def job_id(self) -> str:
        return self.spec.job_id


class RefreshScheduler:
    """Real worker-thread scheduler with global and per-view concurrency gates."""

    def __init__(
        self,
        lease_manager: LeaseManager,
        logger: StructuredLogger,
        *,
        global_limit: int,
        per_view_quotas: dict[str, int] | None = None,
        on_lease_lost: LeaseLostCallback | None = None,
        clock: Callable[[], int] = monotonic_ms,
    ):
        if global_limit <= 0:
            raise ValueError("global_limit must be positive")
        self.lease_manager = lease_manager
        self.logger = logger
        self.global_limit = global_limit
        self.per_view_quotas = dict(per_view_quotas or {})
        self.on_lease_lost = on_lease_lost
        self.clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._pending: list[JobHandle] = []
        self._traces: dict[str, JobTrace] = {}
        self._active_total = 0
        self._active_by_view: dict[str, int] = {}
        self._max_active_total = 0
        self._max_active_by_view: dict[str, int] = {}
        self._shutdown = False
        self._executor = ThreadPoolExecutor(max_workers=global_limit, thread_name_prefix="imv-refresh")
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="imv-dispatcher", daemon=True)
        self._dispatcher.start()

    def _quota_for(self, view_id: str) -> int:
        return max(1, int(self.per_view_quotas.get(view_id, 1)))

    def submit(self, spec: JobSpec) -> JobHandle:
        now = self.clock()
        trace = JobTrace(
            job_id=spec.job_id,
            batch_id=spec.batch_id,
            view_id=spec.view_id,
            kind=spec.kind,
            param_version=spec.param_version,
            refresh_version=spec.refresh_version,
            enqueued_mono_ms=now,
            global_limit=self.global_limit,
            per_view_quota=spec.per_view_quota,
        )
        handle = JobHandle(spec=spec, future=Future(), trace=trace)
        with self._condition:
            if self._shutdown:
                raise RuntimeError("scheduler is shut down")
            self._pending.append(handle)
            self._traces[spec.job_id] = trace
            self._condition.notify_all()
        self.logger.emit(
            "INFO",
            "job_enqueued",
            "refresh job enqueued",
            details={"job_id": spec.job_id, "view_id": spec.view_id, "kind": spec.kind, "refresh_version": spec.refresh_version},
        )
        return handle

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown:
                    if self._active_total < self.global_limit and self._pending:
                        selected_index: int | None = None
                        for index, handle in enumerate(self._pending):
                            view_id = handle.spec.view_id
                            quota = self._quota_for(view_id)
                            if self._active_by_view.get(view_id, 0) < quota:
                                selected_index = index
                                break
                        if selected_index is None:
                            self._condition.wait()
                            continue
                        handle = self._pending.pop(selected_index)
                        view_id = handle.spec.view_id
                        self._active_total += 1
                        self._active_by_view[view_id] = self._active_by_view.get(view_id, 0) + 1
                        self._max_active_total = max(self._max_active_total, self._active_total)
                        self._max_active_by_view[view_id] = max(
                            self._max_active_by_view.get(view_id, 0),
                            self._active_by_view[view_id],
                        )
                        handle.trace.status = "running"
                        handle.trace.started_mono_ms = self.clock()
                        handle.trace.wait_ms = handle.trace.started_mono_ms - handle.trace.enqueued_mono_ms
                        self._executor.submit(self._execute, handle)
                        self.logger.emit(
                            "INFO",
                            "job_started",
                            "refresh job started on a worker thread",
                            details={
                                "job_id": handle.spec.job_id,
                                "view_id": view_id,
                                "wait_ms": handle.trace.wait_ms,
                                "active_total": self._active_total,
                                "active_for_view": self._active_by_view[view_id],
                            },
                        )
                    else:
                        self._condition.wait()
                return

    def _execute(self, handle: JobHandle) -> None:
        spec = handle.spec
        trace = handle.trace
        trace.worker_thread = threading.current_thread().name
        renew_stop = threading.Event()
        lost_error: BusinessError | None = None
        lost_lock = threading.Lock()
        lease: LeaseRecord | None = None
        renewer: threading.Thread | None = None

        def renewal_loop(active_lease: LeaseRecord) -> None:
            nonlocal lost_error
            interval_seconds = max(0.001, active_lease.renew_ms / 1000.0)
            while not renew_stop.wait(interval_seconds):
                try:
                    self.lease_manager.renew(spec.view_id, spec.refresh_version)
                except BusinessError as exc:
                    with lost_lock:
                        lost_error = exc
                    if self.on_lease_lost is not None:
                        self.on_lease_lost(active_lease, exc)
                    return

        try:
            lease = self.lease_manager.issue(
                spec.view_id,
                spec.param_version,
                spec.refresh_version,
                spec.ttl_ms,
                spec.renew_ms,
                job_id=spec.job_id,
            )
            trace.lease_expires_mono_ms = lease.expires_mono_ms
            renewer = threading.Thread(
                target=renewal_loop,
                args=(lease,),
                name=f"imv-lease-{spec.view_id}-{spec.job_id[:8]}",
                daemon=True,
            )
            renewer.start()
            try:
                value = spec.fn()
                with lost_lock:
                    if lost_error is not None:
                        raise lost_error
                renew_stop.set()
                renewer.join()
                completed = self.lease_manager.complete(spec.view_id, spec.refresh_version)
                trace.lease_state = completed.state
                trace.lease_expires_mono_ms = completed.expires_mono_ms
            finally:
                renew_stop.set()
            trace.status = "succeeded"
            handle.future.set_result(value)
        except BaseException as exc:  # noqa: BLE001 - worker boundary must capture all failures
            error = as_business_error(exc)
            renew_stop.set()
            if lease is not None:
                failed = self.lease_manager.fail(spec.view_id, spec.refresh_version, error.code)
                trace.lease_state = None if failed is None else failed.state
            trace.status = "failed"
            trace.error_code = error.code
            trace.error_message = error.message
            handle.future.set_exception(error)
            self.logger.emit(
                "ERROR",
                "job_failed",
                "refresh job failed; later refresh versions may retry",
                code=error.code,
                location=error.location,
                details={"job_id": spec.job_id, "view_id": spec.view_id, "error": error.message},
            )
        finally:
            renew_stop.set()
            if renewer is not None and renewer.is_alive():
                renewer.join()
            with self._condition:
                trace.finished_mono_ms = self.clock()
                if trace.started_mono_ms is not None:
                    trace.run_ms = trace.finished_mono_ms - trace.started_mono_ms
                self._active_total -= 1
                active_for_view = self._active_by_view.get(spec.view_id, 0) - 1
                if active_for_view <= 0:
                    self._active_by_view.pop(spec.view_id, None)
                else:
                    self._active_by_view[spec.view_id] = active_for_view
                self._condition.notify_all()
            self.logger.emit(
                "INFO",
                "job_finished",
                "refresh job worker released its concurrency slot",
                details={
                    "job_id": spec.job_id,
                    "view_id": spec.view_id,
                    "status": trace.status,
                    "run_ms": trace.run_ms,
                    "worker_thread": trace.worker_thread,
                },
            )

    def wait(self, handle: JobHandle, timeout: float | None = None) -> Any:
        return handle.future.result(timeout=timeout)

    def wait_all(self, handles: list[JobHandle], timeout: float | None = None) -> tuple[list[Any], list[BusinessError], list[JobHandle]]:
        values: list[Any] = []
        errors: list[BusinessError] = []
        pending: list[JobHandle] = []
        for handle in handles:
            try:
                values.append(handle.future.result(timeout=timeout))
            except TimeoutError:
                pending.append(handle)
            except BusinessError as exc:
                errors.append(exc)
            except BaseException as exc:  # worker should normalize, defensive here
                errors.append(as_business_error(exc))
        return values, errors, pending

    def traces_for(self, job_ids: list[str]) -> list[dict[str, Any]]:
        with self._condition:
            return [self._traces[job_id].payload() for job_id in job_ids if job_id in self._traces]

    def payload(self, *, recent: int = 100) -> dict[str, Any]:
        with self._condition:
            traces = [trace.payload() for trace in self._traces.values()]
            return {
                "global_limit": self.global_limit,
                "active_total": self._active_total,
                "pending_total": len(self._pending),
                "active_by_view": dict(sorted(self._active_by_view.items())),
                "max_active_total_observed": self._max_active_total,
                "max_active_by_view_observed": dict(sorted(self._max_active_by_view.items())),
                "recent_jobs": traces[-recent:],
            }

    def shutdown(self, *, wait: bool = True) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        if self._dispatcher.is_alive():
            self._dispatcher.join(timeout=2.0)
        self._executor.shutdown(wait=wait, cancel_futures=False)
        with self._condition:
            for handle in self._pending:
                if not handle.future.done():
                    handle.trace.status = "failed"
                    handle.trace.error_code = "E_SCHEDULE"
                    handle.trace.error_message = "scheduler shut down before the job started"
                    handle.future.set_exception(BusinessError("E_SCHEDULE", "scheduler shut down before job start", {"view_id": handle.spec.view_id}))
