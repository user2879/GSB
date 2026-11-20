"""Fixed-point quota and throughput statistics over committed scheduler state.

All counters are integers collected from the engine trace.  Ratios are scaled
once on exact totals (toward zero) and averages use floor truncation; every
result embeds its absolute error bound.  Pool ranking and window comparison
reuse :mod:`gpu_scheduler.fixedpoint` so error-affected decisions are always
labelled and tie-broken deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import fixedpoint as fp
from .engine import SimulationEngine
from .events import DISPATCH, COMPLETE, PREEMPT
from .model import SUCCESS


@dataclass
class Window:
    """A statistic window bounded by an integer millisecond interval."""

    label: str
    start_ms: int
    end_ms: int


def _busy_series(engine: SimulationEngine) -> List[Tuple[int, int, Dict[str, int]]]:
    """Return ``(interval_start, interval_end, busy_by_pool)`` busy segments.

    Busy time is computed from dispatch/preemption/complete events using only
    integer millisecond arithmetic, so occupancy ratios never touch floats.
    """
    intervals: List[Tuple[int, int, str, int, int]] = []
    active: Dict[Tuple[str, int], Tuple[int, str]] = {}
    for record in engine.rt.rounds:
        for event in record.events:
            key = (event.task_id, event.shard_index)
            if event.kind == DISPATCH:
                active[key] = (event.clock_ms, event.pool_id or "")
            elif event.kind in (COMPLETE, PREEMPT):
                start, pool_id = active.pop(key, (event.clock_ms, ""))
                intervals.append((start, event.clock_ms, pool_id,
                                  event.gpu_count, event.memory_mib))
    return intervals  # type: ignore[return-value]


def _busy_totals_via_numpy(intervals, horizon, start):
    """Aggregate busy GPU/memory milliseconds on an integer NumPy timeline.

    The array uses dtype ``int64`` and millisecond bucket indices, so NumPy is
    used only for deterministic integer accumulation; no floating point enters
    the fixed-point statistics.
    """
    gpu_timeline = np.zeros(horizon, dtype=np.int64)
    mem_timeline = np.zeros(horizon, dtype=np.int64)
    for seg_start, seg_end, pool_id, gpu, memory in intervals:
        lo = max(seg_start, start)
        hi = min(seg_end, horizon)
        if hi > lo:
            gpu_timeline[lo:hi] += np.int64(gpu)
            mem_timeline[lo:hi] += np.int64(memory)
    return int(gpu_timeline.sum()), int(mem_timeline.sum())


def pool_occupancy(engine: SimulationEngine,
                   window: Optional[Window] = None) -> List[dict]:
    """Per-pool GPU and memory occupancy as fixed-point ratios.

    Occupancy uses busy-GPU-milliseconds divided by capacity-GPU-milliseconds
    over the window (the full simulated horizon when no window is supplied).
    """
    declaration = engine.declaration
    scale = declaration.config.fixed_point_scale
    horizon = window.end_ms if window else _horizon_ms(engine)
    start = window.start_ms if window else 0
    busy_gpu_ms: Dict[str, int] = {pid: 0 for pid in declaration.pool_map()}
    busy_mem_ms: Dict[str, int] = {pid: 0 for pid in declaration.pool_map()}
    intervals_by_pool: Dict[str, list] = {
        pid: [] for pid in declaration.pool_map()}
    for seg_start, seg_end, pool_id, gpu, memory in _busy_series(engine):
        if pool_id in intervals_by_pool:
            intervals_by_pool[pool_id].append(
                (seg_start, seg_end, pool_id, gpu, memory))
    for pool_id, intervals in intervals_by_pool.items():
        gpu_total, mem_total = _busy_totals_via_numpy(
            intervals, max(horizon, start + 1), start)
        busy_gpu_ms[pool_id] = gpu_total
        busy_mem_ms[pool_id] = mem_total

    rows: List[dict] = []
    gpu_items: List[Tuple[str, fp.RatioStat]] = []
    mem_items: List[Tuple[str, fp.RatioStat]] = []
    for pool_id in sorted(declaration.pool_map()):
        pool = declaration.pool_map()[pool_id]
        gpu_cap_ms = pool.gpu_count * max(horizon - start, 0)
        mem_cap_ms = pool.total_memory_mib * max(horizon - start, 0)
        gpu_stat = fp.ratio_stat(busy_gpu_ms[pool_id], gpu_cap_ms, scale)
        mem_stat = fp.ratio_stat(busy_mem_ms[pool_id], mem_cap_ms, scale)
        gpu_items.append((pool_id, gpu_stat))
        mem_items.append((pool_id, mem_stat))
        rows.append({
            "pool_id": pool_id,
            "window": window.label if window else "full_horizon",
            "gpu_occupancy": gpu_stat.to_dict(),
            "memory_occupancy": mem_stat.to_dict(),
            "busy_gpu_ms": busy_gpu_ms[pool_id],
            "gpu_capacity_ms": gpu_cap_ms,
            "busy_memory_mib_ms": busy_mem_ms[pool_id],
            "memory_capacity_mib_ms": mem_cap_ms,
        })
    ranking = fp.rank(gpu_items)
    for row in rows:
        for ranked in ranking:
            if ranked["key"] == row["pool_id"]:
                row["gpu_rank"] = ranked["rank"]
                row["gpu_error_affected_against"] = ranked[
                    "error_affected_against"]
    return rows


def _horizon_ms(engine: SimulationEngine) -> int:
    if not engine.rt.rounds:
        return 0
    return max(record.clock_ms for record in engine.rt.rounds)


def task_completion(engine: SimulationEngine) -> dict:
    """Fixed-point task and shard completion ratios with explicit error bounds."""
    declaration = engine.declaration
    scale = declaration.config.fixed_point_scale
    tasks = declaration.tasks
    total_tasks = len(tasks)
    done_tasks = 0
    total_shards = 0
    success_shards = 0
    wait_total = 0
    wait_count = 0
    for shard in declaration.all_shards():
        total_shards += 1
        if shard.state == SUCCESS:
            success_shards += 1
        if shard.wait_duration_ms > 0:
            wait_total += shard.wait_duration_ms
            wait_count += 1
    for task in tasks:
        if all(shard.state == SUCCESS for shard in task.shards):
            done_tasks += 1
    task_ratio = fp.ratio_stat(done_tasks, total_tasks, scale)
    shard_ratio = fp.ratio_stat(success_shards, total_shards, scale)
    wait_avg = fp.average_stat(wait_total, wait_count, scale)
    return {
        "scale": scale,
        "tasks_total": total_tasks,
        "tasks_completed": done_tasks,
        "task_completion_ratio": task_ratio.to_dict(),
        "shards_total": total_shards,
        "shards_succeeded": success_shards,
        "shard_completion_ratio": shard_ratio.to_dict(),
        "average_wait_ms": wait_avg.to_dict(),
        "wait_samples": wait_count,
        "wait_total_ms": wait_total,
    }


def compare_windows(engine: SimulationEngine,
                    left: Window, right: Window) -> List[dict]:
    """Compare the same occupancy metric across two statistic windows."""
    left_rows = {row["pool_id"]: row for row in pool_occupancy(engine, left)}
    right_rows = {row["pool_id"]: row for row in pool_occupancy(engine, right)}
    result = []
    for pool_id in sorted(left_rows):
        for metric in ("gpu_occupancy", "memory_occupancy"):
            lhs = fp.RatioStat(**left_rows[pool_id][metric])
            rhs = fp.RatioStat(**right_rows[pool_id][metric])
            comparison = fp.compare_ratios(
                lhs, rhs, f"{left.label}:{pool_id}",
                f"{right.label}:{pool_id}")
            result.append({
                "pool_id": pool_id,
                "metric": metric,
                "left_window": left.label,
                "right_window": right.label,
                "comparison": comparison.to_dict(),
            })
    return result


def aggregate_pools(engine: SimulationEngine) -> dict:
    """Aggregate occupancy across pools on exact integer totals (bound = 1)."""
    declaration = engine.declaration
    scale = declaration.config.fixed_point_scale
    rows = pool_occupancy(engine)
    gpu_parts = [(row["busy_gpu_ms"], row["gpu_capacity_ms"]) for row in rows]
    mem_parts = [(row["busy_memory_mib_ms"], row["memory_capacity_mib_ms"])
                 for row in rows]
    gpu_agg = fp.aggregate_ratio(gpu_parts, scale)
    mem_agg = fp.aggregate_ratio(mem_parts, scale)
    return {
        "scale": scale,
        "aggregation": "exact integer totals before single truncation",
        "aggregate_gpu_occupancy": gpu_agg.to_dict(),
        "aggregate_memory_occupancy": mem_agg.to_dict(),
        "difference_error_bound_scaled_units": 2,
        "note": "single-value bound=1; comparison of two aggregates bound=2",
    }


def full_report(engine: SimulationEngine,
                windows: Optional[List[Window]] = None) -> dict:
    report = {
        "pool_occupancy": pool_occupancy(engine),
        "task_completion": task_completion(engine),
        "aggregate": aggregate_pools(engine),
        "window_comparison": [],
    }
    if windows and len(windows) >= 2:
        report["window_comparison"] = compare_windows(
            engine, windows[0], windows[1])
    return report
