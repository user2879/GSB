"""依据数据规模自适应选择默认执行路径。

规则完全确定性：仅依赖租户数 T、请求数 N、资源池数 P 与显式常量，
相同输入永远得到相同选择，并输出三路径预估工作量与取舍理由。
"""
from __future__ import annotations

from . import ALL_PATHS, PATH_BATCH, PATH_MERGED, PATH_REALTIME

# 工作量估算系数（显式常量，可在技术说明中复核）
COST_REALTIME_PER_REQ = 3.0
COST_BATCH_BUCKET = 12.0
COST_BATCH_PER_REQ = 1.4
COST_MERGED_TENANT = 18.0
COST_MERGED_PER_REQ = 1.1
COST_POOL_FACTOR = 0.08

# 两条候选工作量之差低于该比例阈值时，触发稳定决胜
THRESHOLD_RATIO = 0.10


def estimate_work(tenant_count: int, request_count: int, pool_count: int):
    pool_factor = 1.0 + COST_POOL_FACTOR * pool_count
    estimates = {
        PATH_REALTIME: round(
            COST_REALTIME_PER_REQ * request_count * pool_factor, 3
        ),
        PATH_BATCH: round(
            (COST_BATCH_BUCKET * pool_count + COST_BATCH_PER_REQ * request_count)
            * pool_factor,
            3,
        ),
        PATH_MERGED: round(
            (COST_MERGED_TENANT * tenant_count + COST_MERGED_PER_REQ * request_count)
            * pool_factor,
            3,
        ),
    }
    return estimates


def select_default_path(tenant_count: int, request_count: int, pool_count: int):
    estimates = estimate_work(tenant_count, request_count, pool_count)
    # 稳定主序：工作量升序；工作量相同按固定路径名序决胜（realtime < batch < merged）
    name_rank = {name: idx for idx, name in enumerate(ALL_PATHS)}
    ranked = sorted(estimates.items(), key=lambda kv: (kv[1], name_rank[kv[0]]))
    chosen = ranked[0][0]
    runner_up = ranked[1][0]
    best_cost = estimates[chosen]
    second_cost = estimates[runner_up]

    spread = (second_cost - best_cost) / best_cost if best_cost > 0 else 1.0
    threshold_influenced = spread < THRESHOLD_RATIO

    reasons = [
        f"租户数 T={tenant_count}, 请求数 N={request_count}, 资源池数 P={pool_count}",
        f"工作量估算 realtime={estimates[PATH_REALTIME]}, "
        f"batch={estimates[PATH_BATCH]}, merged={estimates[PATH_MERGED]}",
        f"按工作量升序选择 {chosen}，次选 {runner_up}，相对差 {spread:.4f}",
    ]
    if threshold_influenced:
        # 稳定决胜：阈值内按固定路径名序，保证跨进程一致
        tied = [name for name in ALL_PATHS if abs(estimates[name] - best_cost) / best_cost < THRESHOLD_RATIO]
        chosen = sorted(tied, key=lambda name: name_rank[name])[0]
        reasons.append(
            f"候选相对差低于阈值 {THRESHOLD_RATIO:.2f}，触发稳定决胜，"
            f"候选={tied}，最终选择 {chosen}（本判定受阈值影响）"
        )

    return {
        "selected": chosen,
        "estimates": estimates,
        "threshold_ratio": THRESHOLD_RATIO,
        "threshold_influenced": threshold_influenced,
        "scale": {
            "tenant_count": tenant_count,
            "request_count": request_count,
            "pool_count": pool_count,
        },
        "reasons": reasons,
    }
