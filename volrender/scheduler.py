from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .errors import PipelineError
from .geometry import BlockPlan


@dataclass
class ExecutionResult:
    results: dict[int, Any]
    errors: list[tuple[int, BaseException]]
    interrupted: bool
    not_started_indices: list[int]
    peak_sample_bytes: int

    def first_error(self) -> tuple[int, BaseException] | None:
        if not self.errors:
            return None
        return min(self.errors, key=lambda item: item[0])


def execute_blocks(
    plans: Iterable[BlockPlan],
    worker: Callable[[BlockPlan], Any],
    *,
    concurrency: int,
    memory_budget_bytes: int,
    interrupt_event: threading.Event,
) -> ExecutionResult:
    """Schedule row-major block jobs with bounded threads and sample memory."""

    pending = list(plans)
    for plan in pending:
        if plan.sample_buffer_bytes > memory_budget_bytes:
            raise PipelineError(
                "E_MEMORY",
                "memory_budget_bytes",
                "one block's deterministic sampling buffer exceeds the memory budget",
                details={
                    "block_index": plan.index,
                    "block_sample_buffer_bytes": plan.sample_buffer_bytes,
                    "memory_budget_bytes": memory_budget_bytes,
                },
            )

    results: dict[int, Any] = {}
    errors: list[tuple[int, BaseException]] = []
    not_started: list[int] = []
    active: dict[Future[Any], tuple[BlockPlan, int]] = {}
    used_bytes = 0
    peak_bytes = 0
    cursor = 0
    stop_scheduling = False

    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="volrender")
    try:
        while cursor < len(pending) or active:
            interrupted_now = interrupt_event.is_set()
            if interrupted_now:
                stop_scheduling = True

            while (
                not stop_scheduling
                and cursor < len(pending)
                and len(active) < concurrency
            ):
                plan = pending[cursor]
                if used_bytes + plan.sample_buffer_bytes > memory_budget_bytes:
                    if not active:
                        raise PipelineError(
                            "E_MEMORY",
                            "memory_budget_bytes",
                            "block sampling buffer cannot fit in the memory budget",
                            details={
                                "block_index": plan.index,
                                "required_bytes": plan.sample_buffer_bytes,
                                "memory_budget_bytes": memory_budget_bytes,
                            },
                        )
                    break
                future = executor.submit(worker, plan)
                active[future] = (plan, plan.sample_buffer_bytes)
                used_bytes += plan.sample_buffer_bytes
                peak_bytes = max(peak_bytes, used_bytes)
                cursor += 1

            if not active:
                if stop_scheduling:
                    break
                continue

            done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                plan, reserved = active.pop(future)
                used_bytes -= reserved
                if future.cancelled():
                    not_started.append(plan.index)
                    continue
                try:
                    value = future.result()
                except BaseException as exc:
                    errors.append((plan.index, exc))
                    stop_scheduling = True
                    continue
                results[plan.index] = value

            if stop_scheduling:
                for future, (plan, reserved) in list(active.items()):
                    if future.cancel():
                        used_bytes -= reserved
                        active.pop(future)
                        not_started.append(plan.index)

        if stop_scheduling:
            not_started.extend(plan.index for plan in pending[cursor:])
    finally:
        for future in active:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    return ExecutionResult(
        results=results,
        errors=errors,
        interrupted=interrupt_event.is_set(),
        not_started_indices=sorted(set(not_started)),
        peak_sample_bytes=peak_bytes,
    )

