from __future__ import annotations

from typing import Any

from .spec import Spec


def fixed_metric(used: int, capacity: int, scale_power: int) -> dict[str, Any]:
    if not isinstance(used, int) or isinstance(used, bool) or used < 0:
        raise ValueError("used must be a non-negative integer")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if not 0 <= scale_power <= 3:
        raise ValueError("scale_power must be in [0, 3]")
    scale_factor = 10**scale_power
    ratio_fixed = (used * scale_factor) // capacity
    remaining = capacity - used
    return {
        "used": used,
        "capacity": capacity,
        "remaining": remaining,
        "scale_power": scale_power,
        "scale_factor": scale_factor,
        "ratio_fixed": ratio_fixed,
        "ratio_numerator": ratio_fixed,
        "ratio_denominator": scale_factor,
        "truncation_direction": "floor_down_for_nonnegative_values",
        "quant_error_bound": {
            "absolute_ratio_upper_bound_exclusive": f"1/{scale_factor}",
            "absolute_ratio_upper_bound_inclusive": f"1/{scale_factor}",
            "fixed_units": 1,
            "decimal_upper_bound": 1.0 / scale_factor,
            "note": "true_ratio - ratio_fixed/scale_factor is in [0, 1/scale_factor)",
        },
    }


def build_watermarks(
    spec: Spec,
    pool_gpu_used: dict[str, int],
    pool_memory_used: dict[str, int],
    pool_active_tasks: dict[str, int],
    *,
    global_gpu_quota: int | None = None,
) -> dict[str, Any]:
    quota = spec.global_gpu_quota if global_gpu_quota is None else global_gpu_quota
    if quota <= 0:
        raise ValueError("global_gpu_quota must be positive")
    p = spec.scale_power
    pools_out: list[dict[str, Any]] = []
    global_gpu_used = 0
    global_memory_used = 0
    global_memory_capacity = 0
    global_physical_gpu = 0
    global_active_tasks = 0

    for pool in sorted(spec.pools, key=lambda item: item.id):
        gpu_used = int(pool_gpu_used.get(pool.id, 0))
        memory_used = int(pool_memory_used.get(pool.id, 0))
        active_tasks = int(pool_active_tasks.get(pool.id, 0))
        if gpu_used < 0 or memory_used < 0 or active_tasks < 0:
            raise ValueError(f"negative usage for pool {pool.id}")
        if gpu_used > pool.gpu_count:
            raise ValueError(f"GPU usage exceeds physical capacity in {pool.id}")
        if gpu_used > pool.quota_share:
            raise ValueError(f"GPU usage exceeds quota_share in {pool.id}")
        if memory_used > pool.memory_capacity_mib:
            raise ValueError(f"memory usage exceeds capacity in {pool.id}")
        if active_tasks > pool.max_concurrent_tasks:
            raise ValueError(f"concurrent task limit exceeded in {pool.id}")

        global_gpu_used += gpu_used
        global_memory_used += memory_used
        global_memory_capacity += pool.memory_capacity_mib
        global_physical_gpu += pool.gpu_count
        global_active_tasks += active_tasks
        pools_out.append(
            {
                "scope": "pool",
                "id": pool.id,
                "gpu_physical": fixed_metric(gpu_used, pool.gpu_count, p),
                "gpu_quota_share": fixed_metric(gpu_used, pool.quota_share, p),
                "memory_mib": fixed_metric(memory_used, pool.memory_capacity_mib, p),
                "active_tasks": fixed_metric(active_tasks, pool.max_concurrent_tasks, p),
                "quota_share_cards": pool.quota_share,
                "integer_units_per_gpu": pool.integer_units_per_gpu,
            }
        )

    if global_gpu_used > quota:
        raise ValueError("global GPU quota exceeded")
    global_block = {
        "scope": "global",
        "id": "__global__",
        "gpu_quota": fixed_metric(global_gpu_used, quota, p),
        "gpu_physical": fixed_metric(global_gpu_used, global_physical_gpu, p),
        "memory_mib": fixed_metric(global_memory_used, global_memory_capacity, p),
        "active_tasks_sum": global_active_tasks,
        "pool_count": len(pools_out),
    }
    return {"global": global_block, "pools": pools_out}
