"""Budget-aware, deterministic tile scheduler."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .errors import ErrorCode, fail
from .models import Scene, TileResult
from .rendering import TileSpec, render_tile, tile_hashes
from .validation import validate_budgets, validate_threads


@dataclass(frozen=True)
class SchedulerConfig:
    thread_limit: int
    queue_memory_budget_bytes: int
    compute_budget_units: int


@dataclass
class SchedulingResult:
    tiles: dict[str, TileResult]
    report: dict[str, Any]


def estimate_task_memory(scene: Scene, spec: TileSpec) -> int:
    pixels = spec.pixels
    steps = scene.volume.max_steps
    bytes_per_pixel = 4 * 4 + 4 + steps * (6 * 4 + 1)
    return int(pixels * bytes_per_pixel + 768)


def estimate_task_cost(scene: Scene, spec: TileSpec) -> int:
    layer_visits = max(1, len(scene.surfaces) * 2)
    volume_visits = scene.volume.max_steps * (scene.light_cone.shadow_samples + 2)
    return int(spec.pixels * (1 + layer_visits + volume_visits))


def _occupancy(workers: int, task_memory: int, task_cost: int) -> dict[str, int]:
    return {
        "workers": int(workers),
        "queue_memory_inflight_bytes": int(workers * task_memory),
        "compute_inflight_units": int(workers * task_cost),
    }


def _admission_error(
    code: ErrorCode,
    message: str,
    *,
    tasks: list[TileSpec],
    config: SchedulerConfig,
    task_memory: int,
    task_cost: int,
    reason: str,
) -> Exception:
    denied = [
        {
            "tile_key": task.key,
            "status": "DENIED",
            "code": str(code),
            "reason": reason,
            "estimated_memory_bytes": task_memory,
            "estimated_cost_units": task_cost,
        }
        for task in tasks
    ]
    return fail(
        code,
        message,
        budget_report={
            "requested_threads": config.thread_limit,
            "queue_memory_budget_bytes": config.queue_memory_budget_bytes,
            "compute_budget_units": config.compute_budget_units,
            "largest_task_memory_bytes": task_memory,
            "largest_task_cost_units": task_cost,
            "denied_tasks": denied,
            "denied_task_count": len(denied),
            "degradation_events": [],
        },
    )


def _build_waves(
    tasks: list[TileSpec],
    memories: dict[str, int],
    costs: dict[str, int],
    memory_budget: int,
    compute_budget: int,
) -> list[list[str]]:
    waves: list[list[str]] = []
    for task in tasks:
        placed = False
        for wave in waves:
            wave_memory = sum(memories[key] for key in wave)
            wave_cost = sum(costs[key] for key in wave)
            if (
                wave_memory + memories[task.key] <= memory_budget
                and wave_cost + costs[task.key] <= compute_budget
            ):
                wave.append(task.key)
                placed = True
                break
        if not placed:
            waves.append([task.key])
    return waves


def execute_tiles(scene: Scene, tasks: list[TileSpec], config: SchedulerConfig) -> SchedulingResult:
    """Render all tiles under resource budgets and merge strictly by tile key."""

    validate_threads(config.thread_limit)
    validate_budgets(config.queue_memory_budget_bytes, config.compute_budget_units)
    ordered_tasks = sorted(tasks, key=lambda task: task.key)
    if not ordered_tasks:
        return SchedulingResult(
            tiles={},
            report={
                "requested_threads": config.thread_limit,
                "effective_threads": 0,
                "serialized": False,
                "degradation_events": [],
                "tile_reports": [],
            },
        )

    memories = {task.key: estimate_task_memory(scene, task) for task in ordered_tasks}
    costs = {task.key: estimate_task_cost(scene, task) for task in ordered_tasks}
    largest_memory = max(memories.values())
    largest_cost = max(costs.values())

    if largest_memory > config.queue_memory_budget_bytes:
        raise _admission_error(
            ErrorCode.BUDGET_DENIED,
            "tile queue memory budget cannot admit the largest tile",
            tasks=ordered_tasks,
            config=config,
            task_memory=largest_memory,
            task_cost=largest_cost,
            reason="QUEUE_MEMORY_INSUFFICIENT",
        )
    if largest_cost > config.compute_budget_units:
        raise _admission_error(
            ErrorCode.BUDGET_DENIED,
            "frame compute budget cannot admit the largest tile",
            tasks=ordered_tasks,
            config=config,
            task_memory=largest_memory,
            task_cost=largest_cost,
            reason="COMPUTE_BUDGET_INSUFFICIENT",
        )

    effective_threads = config.thread_limit
    degradation_events: list[dict[str, Any]] = []

    memory_slots = max(1, config.queue_memory_budget_bytes // largest_memory)
    if effective_threads > memory_slots:
        before = _occupancy(effective_threads, largest_memory, largest_cost)
        effective_threads = memory_slots
        after = _occupancy(effective_threads, largest_memory, largest_cost)
        degradation_events.append(
            {
                "priority": 1,
                "reason": "QUEUE_MEMORY_LIMIT",
                "before": before,
                "after": after,
                "queue_memory_budget_bytes": config.queue_memory_budget_bytes,
            }
        )

    compute_slots = max(1, config.compute_budget_units // largest_cost)
    if effective_threads > compute_slots:
        before = _occupancy(effective_threads, largest_memory, largest_cost)
        effective_threads = compute_slots
        after = _occupancy(effective_threads, largest_memory, largest_cost)
        degradation_events.append(
            {
                "priority": 2,
                "reason": "COMPUTE_BUDGET_LIMIT",
                "before": before,
                "after": after,
                "compute_budget_units": config.compute_budget_units,
            }
        )

    serialized = config.thread_limit > 1 and effective_threads == 1
    if serialized:
        degradation_events.append(
            {
                "priority": 3,
                "reason": "SERIAL_FALLBACK",
                "before": _occupancy(config.thread_limit, largest_memory, largest_cost),
                "after": _occupancy(1, largest_memory, largest_cost),
            }
        )

    waves = _build_waves(
        ordered_tasks,
        memories,
        costs,
        config.queue_memory_budget_bytes,
        config.compute_budget_units,
    )

    def run(task: TileSpec) -> TileResult:
        result = render_tile(scene, task)
        result.status = "COMPLETED_SERIAL" if serialized else "COMPLETED"
        return result

    merged: dict[str, TileResult] = {}
    with ThreadPoolExecutor(max_workers=effective_threads, thread_name_prefix="tile") as executor:
        futures = {executor.submit(run, task): task for task in ordered_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result.key in merged:
                raise RuntimeError(f"duplicate tile result for {result.key}")
            merged[result.key] = result

    expected_keys = [task.key for task in ordered_tasks]
    if sorted(merged) != expected_keys:
        raise RuntimeError("scheduler lost or duplicated a tile")

    tile_reports = []
    for key in expected_keys:
        result = merged[key]
        hashes = tile_hashes(result)
        tile_reports.append(
            {
                "tile_key": key,
                "tx": result.tx,
                "ty": result.ty,
                "bounds": [result.x0, result.y0, result.x1, result.y1],
                "status": result.status,
                "estimated_memory_bytes": memories[key],
                "estimated_cost_units": costs[key],
                **hashes,
            }
        )

    report = {
        "requested_threads": config.thread_limit,
        "effective_threads": effective_threads,
        "serialized": serialized,
        "queue_memory_budget_bytes": config.queue_memory_budget_bytes,
        "compute_budget_units": config.compute_budget_units,
        "largest_task_memory_bytes": largest_memory,
        "largest_task_cost_units": largest_cost,
        "degraded": bool(degradation_events),
        "degradation_events": degradation_events,
        "dispatch_waves": waves,
        "wave_count": len(waves),
        "task_count": len(ordered_tasks),
        "completed_task_count": len(merged),
        "denied_tasks": [],
        "completed_tiles": expected_keys,
        "tile_reports": tile_reports,
    }
    return SchedulingResult(tiles=merged, report=report)


def recommended_budgets(scene: Scene, tile_size: int) -> tuple[int, int]:
    """Convenience budgets used by demos and tests; deterministic by construction."""

    from .rendering import plan_tiles

    specs = plan_tiles(scene.camera.width, scene.camera.height, tile_size)
    largest_memory = max(estimate_task_memory(scene, spec) for spec in specs)
    largest_cost = max(estimate_task_cost(scene, spec) for spec in specs)
    total_cost = sum(estimate_task_cost(scene, spec) for spec in specs)
    memory_budget = math.ceil(largest_memory * 4.25)
    compute_budget = math.ceil(max(largest_cost * 2.5, total_cost * 1.2))
    return int(memory_budget), int(compute_budget)
