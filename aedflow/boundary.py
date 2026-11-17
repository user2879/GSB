"""Boundary continuity decisions and per-frame class decisions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .config import CANONICAL_CLASSES
from .errors import failure_entry
from .models import BoundaryDecision, LoadedShard

BOUNDARY_STAGE = "boundary_decision"
FRAME_STAGE = "frame_decision"


def classify_frames(
    global_grid: np.ndarray, threshold: float
) -> tuple[list[str], dict[str, int]]:
    decisions: list[str] = []
    distribution = {name: 0 for name in CANONICAL_CLASSES}
    distribution["background"] = 0
    # Descending class name tie-break: iterate names in reverse alphabetical
    # order and only replace on a strictly greater posterior.
    tie_order = list(reversed(CANONICAL_CLASSES))
    for row in global_grid:
        best_label = tie_order[0]
        best_value = row[CANONICAL_CLASSES.index(best_label)]
        for name in tie_order[1:]:
            value = row[CANONICAL_CLASSES.index(name)]
            if value > best_value:
                best_value = value
                best_label = name
        if best_value < threshold:
            decisions.append("background")
        else:
            decisions.append(best_label)
        distribution[decisions[-1]] += 1
    return decisions, distribution


def _one_boundary(
    boundary_id: str,
    sort_key_ms: int,
    left: LoadedShard,
    right: LoadedShard,
    global_grid: np.ndarray,
    neighborhood: int,
    mean_min: float,
    diff_max: float,
) -> BoundaryDecision:
    boundary_frame = left.last_frame
    left_indexes = list(range(max(0, boundary_frame - neighborhood + 1), boundary_frame + 1))
    right_indexes = list(range(boundary_frame + 1, min(global_grid.shape[0], boundary_frame + 1 + neighborhood)))
    left_block = global_grid[left_indexes]
    right_block = global_grid[right_indexes]
    left_mean = {
        name: float(np.mean(left_block[:, idx])) for idx, name in enumerate(CANONICAL_CLASSES)
    }
    right_mean = {
        name: float(np.mean(right_block[:, idx])) for idx, name in enumerate(CANONICAL_CLASSES)
    }
    mean_diff = {
        name: abs(left_mean[name] - right_mean[name]) for name in CANONICAL_CLASSES
    }
    # Deterministic dominant class: highest left+right mean, name descending on ties.
    ranked = sorted(
        CANONICAL_CLASSES,
        key=lambda name: (-(left_mean[name] + right_mean[name]), tuple(-ord(ch) for ch in name)),
    )
    dominant = ranked[0]
    continuous = (
        left_mean[dominant] >= mean_min
        and right_mean[dominant] >= mean_min
        and mean_diff[dominant] <= diff_max
    )
    criterion = (
        f"dominant={dominant}; left_mean>={mean_min}; right_mean>={mean_min}; "
        f"abs_mean_diff<={diff_max}"
    )
    evidence = {
        "left_frames": left_indexes,
        "right_frames": right_indexes,
        "dominant_left_mean": round(left_mean[dominant], 6),
        "dominant_right_mean": round(right_mean[dominant], 6),
        "dominant_mean_diff": round(mean_diff[dominant], 6),
    }
    return BoundaryDecision(
        boundary_id=boundary_id,
        sort_key_ms=sort_key_ms,
        left_shard_id=left.shard_id,
        right_shard_id=right.shard_id,
        boundary_ms=boundary_frame * 10,
        left_mean=left_mean,
        right_mean=right_mean,
        mean_diff=mean_diff,
        criterion=criterion,
        continuous=continuous,
        dominant_class=dominant,
        evidence=evidence,
    )


def decide_boundaries(
    loaded_shards: list[LoadedShard],
    global_grid: np.ndarray,
    config: dict[str, Any],
    *,
    inject_failure_boundary: str | None = None,
) -> dict[str, Any]:
    neighborhood = int(config["boundary"]["neighborhood_frames"])
    mean_min = float(config["thresholds"]["boundary_mean_min"])
    diff_max = float(config["thresholds"]["boundary_mean_diff_max"])
    workers = max(1, int(config["runtime"]["max_workers"]))

    ordered = sorted(loaded_shards, key=lambda shard: (shard.start_ms, shard.shard_id))
    tasks = []
    for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        boundary_ms = left.end_ms
        boundary_id = f"b{index:02d}_{boundary_ms:05d}ms"
        tasks.append((boundary_id, boundary_ms, left, right))

    failures: list[dict[str, Any]] = []
    decisions: list[BoundaryDecision] = []

    def runner(task: tuple[Any, ...]) -> BoundaryDecision:
        boundary_id, sort_key_ms, left, right = task
        if inject_failure_boundary == boundary_id:
            raise RuntimeError("injected boundary task failure")
        return _one_boundary(
            boundary_id,
            sort_key_ms,
            left,
            right,
            global_grid,
            neighborhood,
            mean_min,
            diff_max,
        )

    if workers == 1:
        for task in tasks:
            try:
                decisions.append(runner(task))
            except Exception as exc:  # noqa: BLE001 - isolation is a requirement
                failures.append(
                    failure_entry(
                        task[0],
                        BOUNDARY_STAGE,
                        "E_TASK_FAILURE",
                        str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(runner, task): task for task in tasks}
            for future, task in future_map.items():
                try:
                    decisions.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        failure_entry(
                            task[0],
                            BOUNDARY_STAGE,
                            "E_TASK_FAILURE",
                            str(exc),
                            exception_type=type(exc).__name__,
                        )
                    )

    # Stable ordering keyed by the global boundary millisecond position.
    decisions.sort(key=lambda decision: decision.sort_key_ms)
    return {"decisions": decisions, "failures": failures}
