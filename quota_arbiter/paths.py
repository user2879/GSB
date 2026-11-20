"""三条执行路径：逐请求实时、按到达时刻分批、按租户归并批量。

确定性的根本来源：三条路径都以同一全局规范序 (arrival, rid) 驱动同一个
状态机引擎，引擎内部的完成 / 唤醒事件完全由到达时刻决定，因此结局序列与
配额终态逐字节一致。路径之间真正不同的是“批次边界与统计口径”：

- realtime：每条请求一个批次；
- batch   ：同一到达时刻的请求一个批次；
- merged  ：先按租户聚合，逐租户形成批次，再按排序后的租户标识合并；
            引擎仍按规范序推进，提交边界落在每个租户组的最后一个请求处，
            合并次序显式取排序租户序，绝不依赖 dict / set 遍历顺序。
"""
from __future__ import annotations

from collections import OrderedDict

from . import PATH_BATCH, PATH_MERGED, PATH_REALTIME
from .engine import ArbiterEngine
from .models import RunResult


def canonical_requests(requests):
    return sorted(requests, key=lambda r: (r.arrival, r.rid))


def build_batch_plan(path_name, tenants, ordered):
    """返回批次计划，每批含 rids、请求数、跳过数、合并次序。

    所有批次的 rids 都按全局规范序排列，引擎可直接顺序驱动。
    """
    if path_name == PATH_REALTIME:
        return [
            {
                "batch_index": idx,
                "path": path_name,
                "request_count": 1,
                "skipped_count": 0,
                "rids": [req.rid],
                "merge_order": [],
            }
            for idx, req in enumerate(ordered)
        ]

    if path_name == PATH_BATCH:
        buckets: "OrderedDict[int, list]" = OrderedDict()
        for req in ordered:
            buckets.setdefault(req.arrival, []).append(req)
        plans = []
        for idx, (arrival, group) in enumerate(buckets.items()):
            plans.append(
                {
                    "batch_index": idx,
                    "path": path_name,
                    "arrival": arrival,
                    "request_count": len(group),
                    "skipped_count": 0,
                    "rids": [r.rid for r in group],
                    "merge_order": [],
                }
            )
        return plans

    # merged：排序租户序聚合。每个租户组定义一个提交边界（该组在规范流中的
    # 最后位置），批次取规范流的连续切片，保证引擎始终按 (arrival, rid) 推进。
    # 切片内属于其他租户的请求记为“被跳过请求”，合并次序为排序租户序。
    groups: "OrderedDict[str, list]" = OrderedDict()
    for tenant in sorted(tenants, key=lambda t: t.tid):
        groups[tenant.tid] = []
    for req in ordered:
        groups[req.tid].append(req)
    merge_order = [tid for tid in groups if groups[tid]]
    global_position = {req.rid: idx for idx, req in enumerate(ordered)}
    group_rids = {tid: {r.rid for r in groups[tid]} for tid in merge_order}

    # 各租户组在规范流中的最后位置，按位置升序确定互不重叠的连续切片。
    group_end = {
        tid: max(global_position[r.rid] for r in groups[tid]) for tid in merge_order
    }
    ordered_groups = sorted(merge_order, key=lambda tid: (group_end[tid], tid))
    plans = []
    start = 0
    for seq, tid in enumerate(ordered_groups):
        end = group_end[tid]
        slice_requests = ordered[start:end + 1]
        skipped = sum(1 for r in slice_requests if r.rid not in group_rids[tid])
        plans.append(
            {
                "batch_index": seq,
                "path": path_name,
                "tenant_id": tid,
                "request_count": len(slice_requests),
                "tenant_owned_count": len(groups[tid]),
                "skipped_count": skipped,
                "rids": [r.rid for r in slice_requests],
                "merge_order": list(ordered_groups),
            }
        )
        start = end + 1
    return plans


def _execute(path_name, tenants, pools, requests):
    ordered = canonical_requests(requests)
    plan = build_batch_plan(path_name, tenants, ordered)
    by_rid = {req.rid: req for req in ordered}
    engine = ArbiterEngine(tenants, pools)
    for batch in plan:
        for rid in batch["rids"]:
            engine.process_request(by_rid[rid])
    # 完成 / 唤醒事件只取决于到达时刻，在整条流处理完后统一排空，与批次边界无关。
    engine.finalize()
    last_rid = ordered[-1].rid if ordered else ""
    result = RunResult(
        outcomes=engine.ordered_outcomes(ordered),
        queue_events=list(engine.queue_events),
        final_tenant_quota=dict(engine.quota_remaining),
        final_tenant_burst=dict(engine.burst_remaining),
        final_pool_capacity=dict(engine.pool_remaining),
        final_inflight=dict(engine.inflight),
        batches=plan,
        last_rid=last_rid,
    )
    return result, ordered, plan


def run_realtime(tenants, pools, requests):
    return _execute(PATH_REALTIME, tenants, pools, requests)[0]


def run_batch(tenants, pools, requests):
    return _execute(PATH_BATCH, tenants, pools, requests)[0]


def run_merged(tenants, pools, requests):
    return _execute(PATH_MERGED, tenants, pools, requests)[0]


PATH_RUNNERS = {
    PATH_REALTIME: run_realtime,
    PATH_BATCH: run_batch,
    PATH_MERGED: run_merged,
}


def run_path(path_name, tenants, pools, requests):
    if path_name not in PATH_RUNNERS:
        raise ValueError(f"未知执行路径: {path_name}")
    return PATH_RUNNERS[path_name](tenants, pools, requests)
