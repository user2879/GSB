"""Budget-aware parallel transcode scheduler with retries and unit isolation."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .canonical import relative_posix
from .config import BudgetConfig, FaultConfig, ProfileConfig, RetryConfig
from .errors import WorkbenchError, require
from .transcode import TranscodeArtifact


TranscodeExecutor = Callable[[], TranscodeArtifact]


@dataclass(frozen=True)
class TranscodeTask:
    profile: ProfileConfig
    sequence: int
    input_video: Path
    input_audio: Path
    output_path: Path

    @property
    def key(self) -> str:
        return f"{self.profile.name}:{self.sequence:05d}"

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.profile.priority, self.sequence, self.profile.name)

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        return {
            "key": self.key,
            "profile": self.profile.name,
            "sequence": self.sequence,
            "input_video": relative_posix(self.input_video, root) if root else self.input_video.as_posix(),
            "input_audio": relative_posix(self.input_audio, root) if root else self.input_audio.as_posix(),
            "output": relative_posix(self.output_path, root) if root else self.output_path.as_posix(),
            "memory_mb": self.profile.memory_mb,
            "cpu_ms": self.profile.cpu_ms,
        }


@dataclass(frozen=True)
class TaskOutcome:
    task: TranscodeTask
    artifact: TranscodeArtifact
    attempts: int
    backoff_total_ms: int
    units_tried: tuple[str, ...]
    ready_clock_ms: int

    def to_dict(self) -> dict[str, Any]:
        value = {
            "key": self.task.key,
            "profile": self.task.profile.name,
            "sequence": self.task.sequence,
            "artifact": self.artifact.to_dict(),
            "attempts": self.attempts,
            "backoff_total_ms": self.backoff_total_ms,
            "units_tried": list(self.units_tried),
            "ready_clock_ms": self.ready_clock_ms,
        }
        return value


@dataclass(frozen=True)
class SchedulerResult:
    outcomes: tuple[TaskOutcome, ...]
    events: tuple[dict[str, Any], ...]
    waves: tuple[dict[str, Any], ...]
    budget_denials: tuple[dict[str, Any], ...]
    degradations: tuple[dict[str, Any], ...]
    retry_events: tuple[dict[str, Any], ...]
    isolation_events: tuple[dict[str, Any], ...]
    reassignment_events: tuple[dict[str, Any], ...]
    requested_parallelism: int
    effective_parallelism: int

    @property
    def artifacts(self) -> list[TranscodeArtifact]:
        return [outcome.artifact for outcome in self.outcomes]

    def ready_times(self) -> dict[str, dict[int, int]]:
        values: dict[str, dict[int, int]] = {}
        for outcome in self.outcomes:
            values.setdefault(outcome.task.profile.name, {})[outcome.task.sequence] = outcome.ready_clock_ms
        return {profile: dict(sorted(entries.items())) for profile, entries in sorted(values.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_parallelism": self.requested_parallelism,
            "effective_parallelism": self.effective_parallelism,
            "waves": list(self.waves),
            "degradations": list(self.degradations),
            "budget_denials": list(self.budget_denials),
            "retry_events": list(self.retry_events),
            "isolation_events": list(self.isolation_events),
            "reassignment_events": list(self.reassignment_events),
            "events": list(self.events),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "ready_times": self.ready_times(),
        }


class InjectedTranscodeFailure(RuntimeError):
    pass


class FaultInjector:
    def __init__(self, config: FaultConfig):
        self.config = config
        self._lock = threading.Lock()
        self._fault_unit: str | None = config.unit_id
        self._counts: dict[tuple[str, str], int] = {}
        self._isolated_units: set[str] = set()

    def should_fail(self, task: TranscodeTask, unit_id: str) -> bool:
        if self.config.failures <= 0 or self.config.task_key not in (None, task.key):
            return False
        with self._lock:
            if self._fault_unit is None:
                self._fault_unit = unit_id
            if unit_id != self._fault_unit or unit_id in self._isolated_units:
                return False
            key = (task.key, unit_id)
            count = self._counts.get(key, 0)
            if count >= self.config.failures:
                return False
            self._counts[key] = count + 1
            return True

    def mark_isolated(self, unit_id: str) -> None:
        with self._lock:
            self._isolated_units.add(unit_id)

    @property
    def fault_unit(self) -> str | None:
        return self._fault_unit


class UnitPool:
    def __init__(self, tasks: list[TranscodeTask], worker_count: int):
        self._units = [f"unit-{index:02d}" for index in range(max(1, worker_count))]
        ordered = sorted(tasks, key=lambda task: task.sort_key)
        self._assignments = {
            task.key: self._units[index % len(self._units)] for index, task in enumerate(ordered)
        }
        self._isolated: set[str] = set()
        self._lock = threading.Lock()

    def unit_for(self, task_key: str) -> str:
        with self._lock:
            return self._assignments[task_key]

    def is_isolated(self, unit_id: str) -> bool:
        with self._lock:
            return unit_id in self._isolated

    def isolate(self, unit_id: str) -> None:
        with self._lock:
            self._isolated.add(unit_id)

    def reassign(self, task_key: str, failed_unit: str) -> str:
        with self._lock:
            current_index = self._units.index(failed_unit)
            for distance in range(1, len(self._units) + 1):
                candidate = self._units[(current_index + distance) % len(self._units)]
                if candidate not in self._isolated:
                    self._assignments[task_key] = candidate
                    return candidate
            raise WorkbenchError(
                "E_RETRY_EXHAUSTED",
                "all transcode units are isolated",
                {"task": task_key, "isolated_units": sorted(self._isolated)},
            )


class BudgetScheduler:
    def __init__(
        self,
        *,
        budget: BudgetConfig,
        retry: RetryConfig,
        fault: FaultConfig,
        root: Path,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        budget.validate()
        retry.validate()
        fault.validate()
        self.budget = budget
        self.retry = retry
        self.fault = fault
        self.root = root
        self.sleep_fn = sleep_fn

    def run(self, tasks: list[TranscodeTask], executor: Callable[[TranscodeTask, Path], TranscodeArtifact]) -> SchedulerResult:
        self._validate_tasks(tasks)
        ordered_tasks = sorted(tasks, key=lambda task: task.sort_key)
        unit_pool = UnitPool(ordered_tasks, self.budget.max_workers)
        fault_injector = FaultInjector(self.fault)
        events: list[dict[str, Any]] = []
        waves: list[dict[str, Any]] = []
        denials: list[dict[str, Any]] = []
        degradations: list[dict[str, Any]] = []
        retry_events: list[dict[str, Any]] = []
        isolation_events: list[dict[str, Any]] = []
        reassignment_events: list[dict[str, Any]] = []
        event_lock = threading.Lock()
        denied_once: set[str] = set()

        def record(event: dict[str, Any]) -> None:
            with event_lock:
                events.append(event)

        pending = list(ordered_tasks)
        outcomes_by_key: dict[str, TaskOutcome] = {}
        wave_index = 0
        logical_clock_ms = 0

        while pending:
            wave: list[TranscodeTask] = []
            remaining: list[TranscodeTask] = []
            memory_used = 0
            cpu_used = 0
            reasons: set[str] = set()
            for task in pending:
                if len(wave) >= self.budget.max_workers:
                    reasons.add("worker_limit")
                    remaining.append(task)
                    self._record_denial(task, denials, denied_once, wave_index, ["worker_limit"])
                    continue
                memory_ok = memory_used + task.profile.memory_mb <= self.budget.memory_budget_mb
                cpu_ok = cpu_used + task.profile.cpu_ms <= self.budget.cpu_budget_ms
                if not memory_ok:
                    reasons.add("memory_budget")
                if not cpu_ok:
                    reasons.add("cpu_budget")
                if memory_ok and cpu_ok:
                    wave.append(task)
                    memory_used += task.profile.memory_mb
                    cpu_used += task.profile.cpu_ms
                else:
                    remaining.append(task)
                    denied_reasons = []
                    if not memory_ok:
                        denied_reasons.append("memory_budget")
                    if not cpu_ok:
                        denied_reasons.append("cpu_budget")
                    self._record_denial(task, denials, denied_once, wave_index, denied_reasons)

            before_parallelism = min(self.budget.max_workers, len(pending))
            forced_override = False
            if not wave:
                forced_override = True
                wave = [remaining.pop(0)]
                memory_used = wave[0].profile.memory_mb
                cpu_used = wave[0].profile.cpu_ms
                reasons = {*reasons, "forced_serial_override"}
            after_parallelism = len(wave)
            if after_parallelism < before_parallelism:
                degradation = {
                    "wave": wave_index,
                    "code": "E_BUDGET_DENIED",
                    "before_parallelism": before_parallelism,
                    "after_parallelism": after_parallelism,
                    "reasons": sorted(reasons),
                    "memory_used_mb": memory_used,
                    "memory_budget_mb": self.budget.memory_budget_mb,
                    "cpu_used_ms": cpu_used,
                    "cpu_budget_ms": self.budget.cpu_budget_ms,
                    "forced_serial_override": forced_override,
                    "priority_order": [item.key for item in wave],
                }
                degradations.append(degradation)
                record({"kind": "degradation", "wave": wave_index, **degradation})

            wave_start_ms = logical_clock_ms
            wave_results: list[TaskOutcome] = []
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="transcode") as pool:
                futures = {
                    pool.submit(
                        self._run_one_with_retries,
                        task,
                        executor,
                        unit_pool,
                        fault_injector,
                        wave_index,
                        wave_start_ms,
                        events,
                        retry_events,
                        isolation_events,
                        reassignment_events,
                    ): task
                    for task in wave
                }
                for future in as_completed(futures):
                    outcome = future.result()
                    require(
                        outcome.task.key not in outcomes_by_key,
                        "E_TASK_DUPLICATE",
                        "duplicate successful task key",
                        key=outcome.task.key,
                    )
                    outcomes_by_key[outcome.task.key] = outcome
                    wave_results.append(outcome)

            wave_results.sort(key=lambda item: item.task.sort_key)
            max_ready = max((outcome.ready_clock_ms for outcome in wave_results), default=wave_start_ms)
            wave_record = {
                "wave": wave_index,
                "start_clock_ms": wave_start_ms,
                "end_clock_ms": max_ready,
                "parallelism": after_parallelism,
                "memory_used_mb": memory_used,
                "cpu_used_ms": cpu_used,
                "tasks": [outcome.task.key for outcome in wave_results],
                "forced_serial_override": forced_override,
            }
            waves.append(wave_record)
            logical_clock_ms = max(logical_clock_ms, max_ready)
            pending = remaining
            wave_index += 1

        outcomes = tuple(sorted(outcomes_by_key.values(), key=lambda item: item.task.sort_key))
        max_parallelism = max((wave["parallelism"] for wave in waves), default=0)
        events.sort(key=_event_sort_key)
        budget_denials = tuple(sorted(denials, key=lambda item: (item["wave"], item["task"])))
        degradations.sort(key=lambda item: item["wave"])
        retry_events.sort(key=_event_sort_key)
        isolation_events.sort(key=_event_sort_key)
        reassignment_events.sort(key=_event_sort_key)
        return SchedulerResult(
            outcomes=outcomes,
            events=tuple(events),
            waves=tuple(waves),
            budget_denials=budget_denials,
            degradations=tuple(degradations),
            retry_events=tuple(retry_events),
            isolation_events=tuple(isolation_events),
            reassignment_events=tuple(reassignment_events),
            requested_parallelism=self.budget.max_workers,
            effective_parallelism=max_parallelism,
        )

    def _run_one_with_retries(
        self,
        task: TranscodeTask,
        executor: Callable[[TranscodeTask, Path], TranscodeArtifact],
        unit_pool: UnitPool,
        fault_injector: FaultInjector,
        wave_index: int,
        wave_start_ms: int,
        events: list[dict[str, Any]],
        retry_events: list[dict[str, Any]],
        isolation_events: list[dict[str, Any]],
        reassignment_events: list[dict[str, Any]],
    ) -> TaskOutcome:
        if task.output_path.exists():
            raise WorkbenchError(
                "E_TASK_DUPLICATE",
                "refusing to overwrite an existing successful segment",
                {"task": task.key, "path": task.output_path.name},
            )

        unit_id = unit_pool.unit_for(task.key)
        if unit_pool.is_isolated(unit_id):
            unit_id = self._reassign(task, unit_id, unit_pool, fault_injector, wave_index, reassignment_events)

        units_tried = [unit_id]
        total_attempts = 0
        retries_on_unit = 0
        backoff_total_ms = 0

        while True:
            total_attempts += 1
            temporary = task.output_path.with_name(
                f".{task.output_path.name}.{total_attempts:03d}.tmp"
            )
            temporary.parent.mkdir(parents=True, exist_ok=True)
            try:
                if fault_injector.should_fail(task, unit_id):
                    raise InjectedTranscodeFailure(
                        f"injected failure for {task.key} on {unit_id}"
                    )
                artifact = executor(task, temporary)
                require(
                    temporary.exists() and temporary.stat().st_size > 0,
                    "E_MEDIA_FORMAT",
                    "transcode executor returned without a non-empty temporary file",
                    task=task.key,
                )
                os.replace(temporary, task.output_path)
                artifact = replace(artifact, relative_path=relative_posix(task.output_path, self.root))
                virtual_ready = (
                    wave_start_ms
                    + task.profile.virtual_transcode_ms
                    + backoff_total_ms
                    + max(0, total_attempts - 1) * task.profile.virtual_transcode_ms
                )
                return TaskOutcome(
                    task=task,
                    artifact=artifact,
                    attempts=total_attempts,
                    backoff_total_ms=backoff_total_ms,
                    units_tried=tuple(units_tried),
                    ready_clock_ms=virtual_ready,
                )
            except WorkbenchError as exc:
                temporary.unlink(missing_ok=True)
                if exc.code == "E_TASK_DUPLICATE":
                    raise
                retry_event = {
                    "kind": "retry_failure",
                    "wave": wave_index,
                    "task": task.key,
                    "attempt": total_attempts,
                    "unit": unit_id,
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
                retry_events.append(retry_event)
                events.append(retry_event)
            except Exception as exc:  # injected failures and transient codec errors are retryable
                temporary.unlink(missing_ok=True)
                retry_event = {
                    "kind": "retry_failure",
                    "wave": wave_index,
                    "task": task.key,
                    "attempt": total_attempts,
                    "unit": unit_id,
                    "error_code": "E_RETRY_EXHAUSTED" if retries_on_unit >= self.retry.max_retries else None,
                    "error_message": str(exc),
                }
                retry_events.append(retry_event)
                events.append(retry_event)

            if retries_on_unit < self.retry.max_retries:
                retries_on_unit += 1
                delay_ms = self.retry.delay_for_retry(retries_on_unit)
                backoff_total_ms += delay_ms
                event = {
                    "kind": "retry_backoff",
                    "wave": wave_index,
                    "task": task.key,
                    "attempt": total_attempts,
                    "retry_number": retries_on_unit,
                    "unit": unit_id,
                    "delay_ms": delay_ms,
                }
                retry_events.append(event)
                events.append(event)
                if delay_ms > 0:
                    self.sleep_fn(delay_ms / 1000.0)
                continue

            exhaustion = {
                "kind": "retry_exhausted",
                "wave": wave_index,
                "task": task.key,
                "attempt": total_attempts,
                "unit": unit_id,
                "code": "E_RETRY_EXHAUSTED",
                "backoff_total_ms": backoff_total_ms,
            }
            retry_events.append(exhaustion)
            events.append(exhaustion)

            unit_pool.isolate(unit_id)
            fault_injector.mark_isolated(unit_id)
            isolation = {
                "kind": "unit_isolated",
                "wave": wave_index,
                "task": task.key,
                "unit": unit_id,
                "code": "E_RETRY_EXHAUSTED",
                "reason": "retry budget exhausted",
            }
            isolation_events.append(isolation)
            events.append(isolation)
            unit_id = self._reassign(
                task,
                unit_id,
                unit_pool,
                fault_injector,
                wave_index,
                reassignment_events,
            )
            units_tried.append(unit_id)
            retries_on_unit = 0

    def _reassign(
        self,
        task: TranscodeTask,
        failed_unit: str,
        unit_pool: UnitPool,
        fault_injector: FaultInjector,
        wave_index: int,
        reassignment_events: list[dict[str, Any]],
    ) -> str:
        new_unit = unit_pool.reassign(task.key, failed_unit)
        fault_injector.mark_isolated(failed_unit)
        event = {
            "kind": "task_reassigned",
            "wave": wave_index,
            "task": task.key,
            "from_unit": failed_unit,
            "to_unit": new_unit,
            "code": "E_RETRY_EXHAUSTED",
        }
        reassignment_events.append(event)
        return new_unit

    @staticmethod
    def _record_denial(
        task: TranscodeTask,
        denials: list[dict[str, Any]],
        denied_once: set[str],
        wave_index: int,
        reasons: list[str],
    ) -> None:
        if task.key in denied_once:
            return
        denied_once.add(task.key)
        denials.append(
            {
                "wave": wave_index,
                "task": task.key,
                "code": "E_BUDGET_DENIED",
                "reasons": sorted(set(reasons)),
                "resolution": "deferred_to_next_wave",
                "memory_mb": task.profile.memory_mb,
                "cpu_ms": task.profile.cpu_ms,
            }
        )

    @staticmethod
    def _validate_tasks(tasks: list[TranscodeTask]) -> None:
        keys = [task.key for task in tasks]
        if len(keys) != len(set(keys)):
            duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
            raise WorkbenchError("E_TASK_DUPLICATE", "duplicate task keys submitted", {"keys": duplicates})
        for task in tasks:
            require(task.sequence >= 0, "E_MEDIA_FORMAT", "negative segment sequence")
            require(task.profile.memory_mb > 0, "E_BUDGET_DENIED", "task memory cost must be positive")
            require(task.profile.cpu_ms > 0, "E_BUDGET_DENIED", "task CPU cost must be positive")


def _event_sort_key(event: dict[str, Any]) -> tuple[Any, ...]:
    kind_order = {
        "retry_failure": 0,
        "retry_backoff": 1,
        "retry_exhausted": 2,
        "unit_isolated": 3,
        "task_reassigned": 4,
        "degradation": 5,
    }
    return (
        event.get("wave", -1),
        event.get("task", ""),
        kind_order.get(event.get("kind", ""), 99),
        event.get("attempt", -1),
        event.get("unit", event.get("from_unit", "")),
        event.get("retry_number", -1),
    )
