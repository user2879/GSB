"""任务级并行调度器与固定优先级降级路径。

数学结果只取决于任务携带的样本区间与全局积分块网格，与线程数、缓冲是否
驻留无关。调度器在数学之外额外负责：线程上限校验、内存预算下的 LRU 计划、
单任务耗时预算下的并行度降级，以及按时间轴任务键（而非完成顺序）归并结果。
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .bufferpool import BufferPool, build_cache_plan, raw_chunk_keys
from .errors import (
    ErrorRecord,
    MergeOrderError,
    PeakOverError,
    ThreadRangeError,
)
from .kweight import kweight_impulse, true_peak_fir
from .meter import TaskResult, process_task

MAX_WORKERS_HARD_LIMIT = 64
# 归一化的“计算工作量”到秒的估算系数（确定性、无随机等待）
COST_SECONDS_PER_FLOAT64 = 2.2e-8


@dataclass
class SchedulerConfig:
    max_workers: int = 4
    memory_budget_bytes: Optional[int] = None
    task_time_budget_seconds: Optional[float] = None
    raise_on_peak_over: bool = False
    inject_merge_fault: bool = False


@dataclass
class SchedulerRun:
    results: List[TaskResult]
    cache_plan: Any
    effective_workers: int
    requested_workers: int
    degradation: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[ErrorRecord] = field(default_factory=list)



def validate_workers(max_workers: int) -> None:
    if not isinstance(max_workers, int) or max_workers < 1 or max_workers > MAX_WORKERS_HARD_LIMIT:
        raise ThreadRangeError(
            "线程数越界",
            location={"max_workers": max_workers},
            evidence={"min": 1, "max": MAX_WORKERS_HARD_LIMIT},
        )


def estimate_task_seconds(task, channels: int) -> float:
    floats = channels * task.context_samples
    return floats * COST_SECONDS_PER_FLOAT64


def _load_raw_chunk(samples: np.ndarray, chunk_start: int, chunk_end: int) -> np.ndarray:
    chunk = samples[:, chunk_start:min(chunk_end, samples.shape[1])]
    if chunk_end > samples.shape[1]:
        pad = np.zeros((samples.shape[0], chunk_end - samples.shape[1]), dtype=samples.dtype)
        chunk = np.concatenate([chunk, pad], axis=1)
    return np.ascontiguousarray(chunk)


def run_scheduler(
    samples: np.ndarray,
    tasks: List,
    config: SchedulerConfig,
) -> SchedulerRun:
    validate_workers(config.max_workers)
    channels = samples.shape[0]
    degradation: List[Dict[str, Any]] = []
    errors: List[ErrorRecord] = []

    # 1) 内存预算：先做规范 LRU 计划（与并发无关，结果可复现）
    cache_plan = build_cache_plan(tasks, channels, config.memory_budget_bytes)
    denied_count = len(cache_plan.denied_records)
    if denied_count:
        errors.extend(cache_plan.denied_records)
        degradation.append(
            {
                "priority": 1,
                "axis": "memory",
                "action": "lru_evict_then_uncached_recompute",
                "reason": "memory_budget_pressure",
                "denied_buffers": denied_count,
                "budget_bytes": config.memory_budget_bytes,
            }
        )

    # 2) 单任务耗时预算：确定性估算，超预算则固定降级并行度，最终串行
    worst_task_seconds = max(estimate_task_seconds(t, channels) for t in tasks)
    effective_workers = config.max_workers
    if config.task_time_budget_seconds is not None:
        if worst_task_seconds > config.task_time_budget_seconds:
            degradation.append(
                {
                    "priority": 2,
                    "axis": "time",
                    "action": "clamp_parallelism",
                    "reason": "task_time_budget_exceeded_estimate",
                    "worst_task_seconds": worst_task_seconds,
                    "budget_seconds": config.task_time_budget_seconds,
                }
            )
            # 固定降级公式：每超一倍预算，并行度减半，最低 1（串行）
            over = worst_task_seconds / config.task_time_budget_seconds
            effective_workers = max(1, config.max_workers // int(math.ceil(over)))
            if effective_workers == 1:
                degradation.append(
                    {
                        "priority": 2,
                        "axis": "time",
                        "action": "serial_fallback",
                        "reason": "parallelism_clamped_to_one",
                    }
                )
    effective_workers = min(effective_workers, len(tasks))

    kw_impulse = kweight_impulse()
    tp_fir = true_peak_fir()
    pool = BufferPool(cache_plan)

    def worker(task) -> TaskResult:
        pool.apply_evictions(task.task_key)
        # 共享原始/重采样上下文缓冲（按计划驻留或重算）
        pieces: List[np.ndarray] = []
        for key, cstart, cend in raw_chunk_keys(task.context_start, task.context_end):
            # 恒等重采样：48k 直通，复用同一缓冲；驻留/复用结论统一取自规范账本
            buf, _event = pool.acquire(
                task.task_key, key, lambda s=cstart, e=cend: _load_raw_chunk(samples, s, e)
            )
            pieces.append(buf)
        context = np.concatenate(pieces, axis=1)
        # 网格切块从 floor(context_start) 开始，按声明区间精确裁剪
        grid_start = _grid_floor_safe(task.context_start)
        lead = task.context_start - grid_start
        context = context[:, lead:lead + (task.context_end - task.context_start)]
        assert context.shape[1] == task.context_end - task.context_start

        result = process_task(task, context, kw_impulse, tp_fir)
        plan_stats = cache_plan.task_plans[task.task_key]
        # 驻留/复用结论以规范账本为准：并发只改变执行时机，不改变统计与数学结果
        result.cache_hits = plan_stats.hits
        result.cache_misses = plan_stats.misses
        result.recomputes = plan_stats.recomputes
        result.uncached = plan_stats.uncached
        result.ledger_events = list(plan_stats.events)
        return result

    if effective_workers == 1:
        collected = [worker(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(worker, t) for t in tasks]
            collected = [f.result() for f in futures]

    # 3) 按时间轴任务键归并（默认输入即时间序；这里强制按 index 排序并校验）
    ordered = sorted(collected, key=lambda r: r.index)
    if [r.task_key for r in ordered] != [t.task_key for t in tasks]:
        raise MergeOrderError(
            "归并任务键序列与时间轴清单不一致",
            evidence={
                "got": [r.task_key for r in ordered],
                "expected": [t.task_key for t in tasks],
            },
        )
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.start <= prev.start:
            raise MergeOrderError(
                "归并结果未按时间轴单调递增",
                location={"prev": prev.task_key, "next": nxt.task_key},
                evidence={"prev_start": prev.start, "next_start": nxt.start},
            )

    if config.inject_merge_fault:
        raise MergeOrderError(
            "注入的归并顺序故障",
            location={"injected": True},
            evidence={"task_keys": [r.task_key for r in ordered]},
        )

    if config.raise_on_peak_over:
        for r in ordered:
            if r.true_peak_db > 0.0:
                raise PeakOverError(
                    "真峰值超出 0 dBTP 量程",
                    location={"task_key": r.task_key, "channel": r.true_peak_channel},
                    evidence={"true_peak_db": r.true_peak_db, "oversample_index": r.true_peak_sample},
                )

    return SchedulerRun(
        results=ordered,
        cache_plan=cache_plan,
        effective_workers=effective_workers,
        requested_workers=config.max_workers,
        degradation=degradation,
        errors=errors,
    )


def _grid_floor_safe(x: int) -> int:
    from .bufferpool import RAW_GRID

    return (x // RAW_GRID) * RAW_GRID
