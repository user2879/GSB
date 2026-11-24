from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from .errors import AppError
from .spec import SessionSpec, SliceSpec

T = TypeVar("T")


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class SliceTask:
    task_id: str
    view_id: str
    env_id: str
    slice_spec: SliceSpec


@dataclass(frozen=True)
class DispatchMetrics:
    enqueued_ms: int
    worker_started_ms: int
    quota_wait_start_ms: int
    quota_wait_end_ms: int
    socket_started_ms: int
    socket_ended_ms: int
    worker_thread: str

    @property
    def executor_wait_ms(self) -> int:
        return self.quota_wait_start_ms - self.enqueued_ms

    @property
    def quota_wait_ms(self) -> int:
        return self.quota_wait_end_ms - self.quota_wait_start_ms

    @property
    def socket_ms(self) -> int:
        return self.socket_ended_ms - self.socket_started_ms

    def to_dict(self) -> dict[str, int]:
        return {
            "enqueued_monotonic_ms": self.enqueued_ms,
            "worker_started_monotonic_ms": self.worker_started_ms,
            "quota_wait_start_monotonic_ms": self.quota_wait_start_ms,
            "quota_wait_end_monotonic_ms": self.quota_wait_end_ms,
            "socket_started_monotonic_ms": self.socket_started_ms,
            "socket_ended_monotonic_ms": self.socket_ended_ms,
            "worker_thread": self.worker_thread,
            "executor_wait_ms": self.executor_wait_ms,
            "quota_wait_ms": self.quota_wait_ms,
            "socket_ms": self.socket_ms,
        }


@dataclass(frozen=True)
class DispatchResult(Generic[T]):
    task: SliceTask
    value: T | None
    error: dict[str, Any] | None
    metrics: DispatchMetrics


class ConcurrencyGate:
    def __init__(self, global_limit: int, per_environment_limit: int) -> None:
        if global_limit < 1 or per_environment_limit < 1:
            raise ValueError("concurrency limits must be >= 1")
        self.global_limit = global_limit
        self.per_environment_limit = per_environment_limit
        self._condition = threading.Condition()
        self._global_in_flight = 0
        self._environment_in_flight: dict[str, int] = {}
        self.peak_global_in_flight = 0
        self.peak_environment_in_flight: dict[str, int] = {}

    def acquire(self, env_id: str) -> None:
        with self._condition:
            while (
                self._global_in_flight >= self.global_limit
                or self._environment_in_flight.get(env_id, 0)
                >= self.per_environment_limit
            ):
                self._condition.wait()
            self._global_in_flight += 1
            current_env = self._environment_in_flight.get(env_id, 0) + 1
            self._environment_in_flight[env_id] = current_env
            self.peak_global_in_flight = max(self.peak_global_in_flight, self._global_in_flight)
            self.peak_environment_in_flight[env_id] = max(
                self.peak_environment_in_flight.get(env_id, 0), current_env
            )

    def release(self, env_id: str) -> None:
        with self._condition:
            if self._global_in_flight <= 0:
                raise RuntimeError("global concurrency gate released too many times")
            current_env = self._environment_in_flight.get(env_id, 0)
            if current_env <= 0:
                raise RuntimeError(f"environment gate for {env_id!r} released too many times")
            self._global_in_flight -= 1
            if current_env == 1:
                self._environment_in_flight.pop(env_id, None)
            else:
                self._environment_in_flight[env_id] = current_env - 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "global_limit": self.global_limit,
                "per_environment_limit": self.per_environment_limit,
                "global_in_flight": self._global_in_flight,
                "environment_in_flight": dict(sorted(self._environment_in_flight.items())),
                "peak_global_in_flight": self.peak_global_in_flight,
                "peak_environment_in_flight": dict(
                    sorted(self.peak_environment_in_flight.items())
                ),
            }


class BoundedScheduler:
    def __init__(self, session: SessionSpec) -> None:
        self.session = session
        self.gate = ConcurrencyGate(
            session.global_concurrency, session.per_environment_concurrency
        )
        self.last_executor_workers = session.global_concurrency

    def run_batch(
        self,
        tasks: list[SliceTask],
        execute: Callable[[SliceTask], T],
    ) -> list[DispatchResult[T]]:
        if not tasks:
            return []
        enqueued = {task.task_id: monotonic_ms() for task in tasks}
        max_workers = max(1, self.session.global_concurrency)
        self.last_executor_workers = max_workers
        futures: list[Future[DispatchResult[T]]] = []
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="arbiter-dispatch"
        ) as executor:
            for task in tasks:
                futures.append(
                    executor.submit(self._run_one, task, enqueued[task.task_id], execute)
                )
            return [future.result() for future in futures]

    def _run_one(
        self,
        task: SliceTask,
        enqueued_ms: int,
        execute: Callable[[SliceTask], T],
    ) -> DispatchResult[T]:
        worker_started = monotonic_ms()
        quota_wait_start = monotonic_ms()
        self.gate.acquire(task.env_id)
        quota_wait_end = monotonic_ms()
        socket_started = monotonic_ms()
        value: T | None = None
        error: dict[str, Any] | None = None
        try:
            value = execute(task)
        except AppError as exc:
            error = exc.to_dict()
        except Exception as exc:  # Worker failures are diagnostics, not process crashes.
            error = {
                "code": "E_FIELD",
                "field": "worker",
                "message": f"unexpected worker failure: {type(exc).__name__}: {exc}",
                "context": {"task_id": task.task_id},
            }
        finally:
            socket_ended = monotonic_ms()
            self.gate.release(task.env_id)
        metrics = DispatchMetrics(
            enqueued_ms=enqueued_ms,
            worker_started_ms=worker_started,
            quota_wait_start_ms=quota_wait_start,
            quota_wait_end_ms=quota_wait_end,
            socket_started_ms=socket_started,
            socket_ended_ms=socket_ended,
            worker_thread=threading.current_thread().name,
        )
        return DispatchResult(task=task, value=value, error=error, metrics=metrics)
