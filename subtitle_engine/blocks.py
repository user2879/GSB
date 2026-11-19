"""Bounded concurrent block scheduling and deterministic frame computation."""

from __future__ import annotations

import time
from concurrent.futures import CancelledError, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from ._numpy import np
from .errors import BlockComputationCancelled, EngineError
from .incremental import IncrementalPlan
from .layout import EventLayout
from .render import compose_frame
from .spec import Spec


@dataclass(frozen=True)
class BlockComputation:
    index: int
    frames: "np.ndarray"
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SchedulerResult:
    completed: tuple[int, ...]
    interrupted: bool
    signal_number: int | None
    reused: tuple[int, ...]
    recomputed: tuple[int, ...]


def estimate_block_memory(spec: Spec, index: int) -> int:
    start, end = _block_frame_range(spec, index)
    frame_count = max(0, end - start)
    frame_bytes = frame_count * spec.frame.width * spec.frame.height
    per_frame_metadata = 64 + max(1, len(spec.events)) * 20
    layout_overhead = 64 * 1024
    # Renderer keeps the final block buffer plus one working frame, and modest
    # metadata while the completion is in flight.
    return frame_bytes * 2 + frame_count * per_frame_metadata + layout_overhead


def _block_frame_range(spec: Spec, index: int) -> tuple[int, int]:
    start = index * spec.frames_per_block
    return start, min(spec.total_frames, start + spec.frames_per_block)


def visible_layouts_for_frame(layouts: dict[str, EventLayout], timestamp_ms: int) -> list[EventLayout]:
    return [
        layouts[event_id]
        for event_id in sorted(layouts)
        if layouts[event_id].event.start_ms <= timestamp_ms < layouts[event_id].event.end_ms
    ]


def compute_block(
    spec: Spec,
    plan: IncrementalPlan,
    index: int,
    *,
    cancel_event: Any,
    test_block_delay_ms: int,
) -> BlockComputation:
    if test_block_delay_ms > 0:
        time.sleep(test_block_delay_ms / 1000.0)
    if cancel_event.is_set():
        raise BlockComputationCancelled()

    start_frame, end_frame = _block_frame_range(spec, index)
    frame_count = end_frame - start_frame
    frames = np.empty(
        (frame_count, spec.frame.height, spec.frame.width), dtype=np.uint8
    )
    per_frame_visible: list[list[str]] = []
    visible_union: set[str] = set()
    critical_union: set[str] = set()

    for local_index, absolute_frame in enumerate(range(start_frame, end_frame)):
        if cancel_event.is_set():
            raise BlockComputationCancelled()
        timestamp = absolute_frame * spec.frame.frame_ms
        visible = visible_layouts_for_frame(plan.layouts, timestamp)
        ids = [layout.event.id for layout in visible]
        per_frame_visible.append(ids)
        visible_union.update(ids)
        critical_union.update(
            layout.event.id for layout in visible if layout.event.critical_sound
        )
        frames[local_index] = compose_frame(
            spec.frame,
            plan.final_style,
            visible,
            absolute_frame,
        )

    metadata = {
        "index": index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": frame_count,
        "start_ms": start_frame * spec.frame.frame_ms,
        "end_ms": end_frame * spec.frame.frame_ms,
        "block_ms": spec.runtime.block_ms,
        "visible_event_ids": sorted(visible_union),
        "critical_event_ids": sorted(critical_union),
        "frame_visible_event_ids": per_frame_visible,
        "event_bboxes": {
            event_id: list(plan.layouts[event_id].bbox.as_list())
            for event_id in sorted(visible_union)
        },
    }
    return BlockComputation(index, frames, metadata)


def schedule_blocks(
    spec: Spec,
    plan: IncrementalPlan,
    pending_indices: list[int],
    *,
    stop_event: Any,
    cancel_event: Any,
    on_block: Callable[[BlockComputation], None],
    test_block_delay_ms: int = 0,
) -> SchedulerResult:
    """Compute pending blocks with bounded threads and strict memory admission."""

    if not pending_indices:
        return SchedulerResult((), False, None, (), ())

    concurrency = max(1, spec.runtime.concurrency)
    budget = spec.runtime.memory_budget_bytes
    costs = {index: estimate_block_memory(spec, index) for index in pending_indices}
    oversized = [index for index in pending_indices if costs[index] > budget]
    if oversized:
        raise EngineError(
            "E_MEMORY_BUDGET",
            "A single block exceeds the configured memory budget",
            details={
                "block_index": oversized[0],
                "block_cost_bytes": costs[oversized[0]],
                "memory_budget_bytes": budget,
            },
        )

    pending = list(pending_indices)
    cursor = 0
    futures: dict[Future[BlockComputation], int] = {}
    reserved = 0
    completed: list[int] = []
    errors: dict[int, BaseException] = {}
    stop_scheduling = False
    interrupted = False
    signal_number: int | None = None

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="subtitle-block") as pool:
        while cursor < len(pending) or futures:
            if stop_event.is_set():
                stop_scheduling = True
                interrupted = True
                signal_number = getattr(stop_event, "signal_number", None)

            while (
                not stop_scheduling
                and not cancel_event.is_set()
                and cursor < len(pending)
                and len(futures) < concurrency
            ):
                index = pending[cursor]
                cost = costs[index]
                if reserved + cost > budget:
                    if futures:
                        break
                    raise EngineError(
                        "E_MEMORY_BUDGET",
                        "Memory budget cannot admit the next block",
                        details={
                            "block_index": index,
                            "reserved_bytes": reserved,
                            "block_cost_bytes": cost,
                            "memory_budget_bytes": budget,
                        },
                    )
                future = pool.submit(
                    compute_block,
                    spec,
                    plan,
                    index,
                    cancel_event=cancel_event,
                    test_block_delay_ms=test_block_delay_ms,
                )
                futures[future] = index
                reserved += cost
                cursor += 1

            if not futures:
                break

            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: futures[item]):
                index = futures.pop(future)
                try:
                    result = future.result()
                except BlockComputationCancelled:
                    reserved -= costs[index]
                    continue
                except CancelledError:
                    reserved -= costs[index]
                    continue
                except BaseException as exc:  # noqa: BLE001 - collected and ranked by block index
                    reserved -= costs[index]
                    errors[index] = exc
                    cancel_event.set()
                    for other in list(futures):
                        other.cancel()
                    continue
                completed.append(index)
                on_block(result)
                reserved -= costs[index]

            if errors:
                stop_scheduling = True

        if interrupted and cursor < len(pending):
            # Do not wait for and persist blocks that were never submitted.
            pass

    if errors:
        actual = [(index, exc) for index, exc in errors.items()]
        first_index, first_error = min(actual, key=lambda item: item[0])
        if isinstance(first_error, EngineError):
            raise first_error
        raise EngineError(
            "E_INTERNAL",
            "Block computation failed",
            details={"block_index": first_index, "reason": str(first_error)},
        )

    return SchedulerResult(
        completed=tuple(sorted(completed)),
        interrupted=interrupted and len(completed) < len(pending_indices),
        signal_number=signal_number,
        reused=(),
        recomputed=tuple(sorted(completed)),
    )
