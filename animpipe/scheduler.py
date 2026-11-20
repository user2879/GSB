"""Frame-block task scheduler over a worker thread pool.

Guarantees
----------
* Tasks are pure functions of ``(block_key, frame_range)``; no worker
  mutates shared state.
* Results are merged into pre-allocated slots keyed by block start frame,
  so completion order cannot affect the assembled output.
* Per-frame deterministic tags derive from frame content only; jitter or
  random terms can never depend on block partitioning.
* ``workers == 1`` uses an inline serial path (no thread pool), which the
  budget module selects when the parallel budget is exhausted.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .hashutil import frame_hash

BlockFn = Callable[[tuple[int, int]], dict[str, np.ndarray]]


@dataclass(frozen=True)
class BlockTask:
    key: int  # start frame of the block
    frame_range: tuple[int, int]
    stage: str


@dataclass
class BlockResult:
    task: BlockTask
    data: dict[str, np.ndarray]
    frame_tags: tuple[str, ...]
    worker: str


@dataclass
class ScheduleReport:
    stage: str
    workers_requested: int
    workers_effective: int
    block_size_requested: int
    block_size_effective: int
    block_count: int
    total_frames: int
    completed_keys: list[int]
    degradation: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)


def split_ranges(total_frames: int, block_size: int) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [
        (start, min(start + block_size, total_frames))
        for start in range(0, total_frames, block_size)
    ]


def make_frame_tags(
    frame_indices: list[int],
    rotations: np.ndarray,
    root_translations: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    return tuple(
        frame_hash(index, rotations[local], root_translations[local], meta)
        for local, index in enumerate(frame_indices)
    )


class FrameBlockScheduler:
    def __init__(self, workers: int):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers

    def run(
        self,
        stage: str,
        total_frames: int,
        block_size: int,
        block_fn: BlockFn,
        *,
        tag_meta: dict[str, Any] | None = None,
    ) -> tuple[dict[int, dict[str, np.ndarray]], ScheduleReport, tuple[str, ...]]:
        ranges = split_ranges(total_frames, block_size)
        tasks = [
            BlockTask(key=start, frame_range=(start, end), stage=stage)
            for start, end in ranges
        ]
        slots: dict[int, dict[str, np.ndarray]] = {}
        tags: dict[int, tuple[str, ...]] = {}
        workers_used: dict[int, str] = {}

        def run_task(task: BlockTask) -> BlockResult:
            data = block_fn(task.frame_range)
            start, end = task.frame_range
            frame_tags = make_frame_tags(
                list(range(start, end)),
                data["rotations"],
                data["root_translations"],
                tag_meta,
            )
            return BlockResult(
                task=task,
                data=data,
                frame_tags=frame_tags,
                worker=f"thread-{task.key}",
            )

        if self.workers == 1 or len(tasks) == 1:
            for task in tasks:
                result = run_task(task)
                slots[result.task.key] = result.data
                tags[result.task.key] = result.frame_tags
                workers_used[result.task.key] = "serial"
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix=f"{stage}-worker",
            ) as pool:
                future_map = {pool.submit(run_task, task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_map):
                    result = future.result()
                    slots[result.task.key] = result.data
                    tags[result.task.key] = result.frame_tags
                    workers_used[result.task.key] = result.worker

        ordered_keys = sorted(slots)
        all_tags: list[str] = []
        for key in ordered_keys:
            all_tags.extend(tags[key])
        report = ScheduleReport(
            stage=stage,
            workers_requested=self.workers,
            workers_effective=self.workers if self.workers > 1 else 1,
            block_size_requested=block_size,
            block_size_effective=block_size,
            block_count=len(tasks),
            total_frames=total_frames,
            completed_keys=ordered_keys,
        )
        return slots, report, tuple(all_tags)
