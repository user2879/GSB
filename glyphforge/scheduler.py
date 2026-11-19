"""Bounded-concurrency deterministic rasterization scheduler."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .errors import budget_denied, task_key_error
from .model import SUPPORTED_SIZES, Glyph
from .raster import (
    RasterConfig,
    RasterResult,
    estimate_raster_work_bytes,
    normalize_fill_rule,
    rasterize_glyph,
    verify_bitmap_hash,
)


@dataclass(frozen=True, order=True)
class RasterTaskKey:
    glyph_id: str
    size: int
    fill_rule: str
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.glyph_id, str) or not self.glyph_id.strip():
            raise task_key_error("task glyph id must be a non-empty string")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise task_key_error("task size must be an integer")
        if self.size not in SUPPORTED_SIZES:
            raise task_key_error(
                "task size is not a supported tier",
                size=self.size,
                allowed=list(SUPPORTED_SIZES),
            )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise task_key_error("task revision must be a non-negative integer")
        normalized = normalize_fill_rule(self.fill_rule)
        object.__setattr__(self, "fill_rule", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "glyph_id": self.glyph_id,
            "size": self.size,
            "fill_rule": self.fill_rule,
            "revision": self.revision,
        }

    def display(self) -> str:
        return f"{self.glyph_id}|{self.size}|{self.fill_rule}|r{self.revision}"


@dataclass(frozen=True)
class RasterTask:
    glyph: Glyph
    size: int
    fill_rule: str = "nonzero"
    revision: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.glyph, Glyph):
            raise task_key_error("RasterTask.glyph must be a Glyph")
        normalized = normalize_fill_rule(self.fill_rule)
        # Validate all key fields immediately, even when the scheduler is not
        # invoked yet.
        key = RasterTaskKey(self.glyph.glyph_id, self.size, normalized, self.revision)
        object.__setattr__(self, "fill_rule", key.fill_rule)

    @property
    def key(self) -> RasterTaskKey:
        return RasterTaskKey(
            self.glyph.glyph_id, self.size, self.fill_rule, self.revision
        )

    def estimated_memory_bytes(self, config: RasterConfig | None = None) -> int:
        return estimate_raster_work_bytes(self.glyph, self.size, config)


@dataclass(frozen=True)
class ThreadDegradation:
    action: str
    reason: str
    requested: int
    effective: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "requested": self.requested,
            "effective": self.effective,
        }


@dataclass(frozen=True)
class ThreadBudgetReport:
    requested_threads: int
    effective_threads: int
    memory_budget_bytes: int
    max_task_memory_bytes: int
    peak_memory_estimate_bytes: int
    task_count: int
    degradations: tuple[ThreadDegradation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_threads": self.requested_threads,
            "effective_threads": self.effective_threads,
            "memory_budget_bytes": self.memory_budget_bytes,
            "max_task_memory_bytes": self.max_task_memory_bytes,
            "peak_memory_estimate_bytes": self.peak_memory_estimate_bytes,
            "task_count": self.task_count,
            "degradations": [decision.to_dict() for decision in self.degradations],
        }


@dataclass(frozen=True)
class ScheduledRasterResult:
    results: tuple[tuple[RasterTaskKey, RasterResult], ...]
    report: ThreadBudgetReport

    @property
    def by_key(self) -> dict[RasterTaskKey, RasterResult]:
        return dict(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_budget": self.report.to_dict(),
            "results": [
                {"task": key.to_dict(), **result.to_dict()}
                for key, result in self.results
            ],
        }


class RasterScheduler:
    """Run raster tasks concurrently while keeping canonical result order."""

    def __init__(self, config: RasterConfig | None = None) -> None:
        self.config = config or RasterConfig()

    def run(
        self,
        tasks: Iterable[RasterTask],
        *,
        worker_budget: int,
        memory_budget_bytes: int,
        expected_hashes: Mapping[RasterTaskKey, str] | None = None,
    ) -> ScheduledRasterResult:
        if (
            isinstance(worker_budget, bool)
            or not isinstance(worker_budget, int)
            or worker_budget < 1
        ):
            raise budget_denied(
                "worker_budget must be a positive integer", worker_budget=worker_budget
            )
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise budget_denied(
                "memory_budget_bytes must be a positive integer",
                memory_budget_bytes=memory_budget_bytes,
            )

        task_list = sorted(tasks, key=lambda item: item.key)
        seen: set[RasterTaskKey] = set()
        for task in task_list:
            if task.key in seen:
                raise task_key_error(
                    "duplicate raster task key", task=task.key.to_dict()
                )
            seen.add(task.key)

        if not task_list:
            report = ThreadBudgetReport(
                requested_threads=worker_budget,
                effective_threads=0,
                memory_budget_bytes=memory_budget_bytes,
                max_task_memory_bytes=0,
                peak_memory_estimate_bytes=0,
                task_count=0,
                degradations=(),
            )
            return ScheduledRasterResult(results=(), report=report)

        estimates = [task.estimated_memory_bytes(self.config) for task in task_list]
        max_task_memory = max(estimates)
        if max_task_memory > memory_budget_bytes:
            raise budget_denied(
                "largest raster task exceeds total thread memory budget",
                required_bytes=max_task_memory,
                memory_budget_bytes=memory_budget_bytes,
            )

        memory_workers = max(1, memory_budget_bytes // max_task_memory)
        effective = min(worker_budget, memory_workers, len(task_list))
        degradations: list[ThreadDegradation] = []
        if effective < worker_budget:
            if memory_workers < worker_budget:
                degradations.append(
                    ThreadDegradation(
                        action="reduce_concurrency",
                        reason="thread_memory_budget",
                        requested=worker_budget,
                        effective=effective,
                    )
                )
            elif len(task_list) < worker_budget:
                degradations.append(
                    ThreadDegradation(
                        action="reduce_concurrency",
                        reason="task_count_less_than_worker_budget",
                        requested=worker_budget,
                        effective=effective,
                    )
                )
        peak = effective * max_task_memory
        report = ThreadBudgetReport(
            requested_threads=worker_budget,
            effective_threads=effective,
            memory_budget_bytes=memory_budget_bytes,
            max_task_memory_bytes=max_task_memory,
            peak_memory_estimate_bytes=peak,
            task_count=len(task_list),
            degradations=tuple(degradations),
        )

        def execute(task: RasterTask) -> tuple[RasterTaskKey, RasterResult]:
            return task.key, rasterize_glyph(
                task.glyph,
                task.size,
                task.fill_rule,
                config=self.config,
            )

        completed: dict[RasterTaskKey, RasterResult] = {}
        failures: list[tuple[RasterTaskKey, BaseException]] = []
        with ThreadPoolExecutor(
            max_workers=effective, thread_name_prefix="glyphforge-raster"
        ) as pool:
            futures = {pool.submit(execute, task): task.key for task in task_list}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result_key, result = future.result()
                except Exception as exc:  # noqa: BLE001 - surfaced after all tasks settle
                    failures.append((key, exc))
                    continue
                if result_key != key:
                    failures.append(
                        (key, task_key_error("worker returned a mismatched task key"))
                    )
                    continue
                completed[key] = result

        if failures:
            _failed_key, exception = min(failures, key=lambda item: item[0])
            raise exception

        missing = [key for key in seen if key not in completed]
        if missing:
            raise task_key_error(
                "one or more raster task results are missing",
                missing_tasks=[key.to_dict() for key in sorted(missing)],
            )

        if expected_hashes is not None:
            for key in sorted(expected_hashes):
                if key not in completed:
                    raise task_key_error(
                        "expected hash refers to an unknown task", task=key.to_dict()
                    )
            for key in sorted(completed):
                if key in expected_hashes:
                    verify_bitmap_hash(completed[key], expected_hashes[key])

        canonical = tuple((key, completed[key]) for key in sorted(completed))
        return ScheduledRasterResult(results=canonical, report=report)
