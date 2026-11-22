from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from .errors import AppError


def monotonic_ms() -> int:
    return time.perf_counter_ns() // 1_000_000


@dataclass
class _Running:
    shard: dict[str, Any]
    trace: dict[str, Any]


def schedule_plan(
    plan: dict[str, Any],
    dispatch: Callable[[dict[str, Any]], dict[str, Any]],
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run shards under global, cluster and task concurrency budgets.

    The dispatcher loop only starts work when every constraint can be checked
    atomically in the coordinator, so there is no lock-order dependency and no
    way for a shard to exceed a budget while queued work is reconsidered.
    """

    global_budget = int(plan["global_resource_units"])
    shards = sorted(
        plan.get("shards", []),
        key=lambda item: (item["cluster_id"], item["task_id"], item["shard_index"]),
    )
    plan_order = {shard["shard_id"]: index for index, shard in enumerate(shards)}
    started_at = monotonic_ms()
    traces: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for shard in shards:
        trace = {
            "shard_id": shard["shard_id"],
            "cluster_id": shard["cluster_id"],
            "task_id": shard["task_id"],
            "shard_index": shard["shard_index"],
            "revision_version": shard["revision_version"],
            "resource_units": shard["resource_units"],
            "outcome": "queued",
            "error_code": None,
            "error_message": None,
            "wait_started_ms": started_at,
            "wait_ended_ms": None,
            "started_ms": None,
            "finished_ms": None,
            "elapsed_ms": None,
            "aggregate": None,
            "node_pid": None,
        }
        traces[shard["shard_id"]] = trace
        reason = None
        if shard["cluster_quota"] <= 0:
            reason = "cluster concurrency quota is zero"
        elif shard["max_parallel"] <= 0:
            reason = "task concurrency limit is zero"
        elif shard["resource_units"] > global_budget:
            reason = "shard resource requirement exceeds global budget"
        if reason is None:
            pending.append(shard)
        else:
            trace["outcome"] = "blocked"
            trace["error_code"] = "E_BUDGET"
            trace["error_message"] = reason
            trace["wait_ended_ms"] = monotonic_ms()

    running: dict[Future[dict[str, Any]], _Running] = {}
    running_resource = 0
    running_cluster: dict[str, int] = {}
    running_task: dict[str, int] = {}
    peak_resource = 0
    peak_cluster: dict[str, int] = {}
    peak_task: dict[str, int] = {}
    max_workers = max(1, len(pending))
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shard")

    def completed_trace(shard: dict[str, Any], result: dict[str, Any] | None, exc: BaseException | None) -> None:
        trace = traces[shard["shard_id"]]
        trace["finished_ms"] = monotonic_ms()
        if exc is not None:
            trace["outcome"] = "failed"
            if isinstance(exc, AppError):
                trace["error_code"] = exc.code
                trace["error_message"] = exc.message
            else:
                trace["error_code"] = "E_INTERNAL"
                trace["error_message"] = str(exc)
        else:
            assert result is not None
            trace["outcome"] = "completed"
            trace["aggregate"] = result["aggregate"]
            trace["elapsed_ms"] = int(result.get("elapsed_ms", 0))
            trace["node_pid"] = result.get("pid")
        if on_complete is not None:
            on_complete(trace)

    if on_complete is not None:
        for shard in shards:
            trace = traces[shard["shard_id"]]
            if trace["outcome"] == "blocked":
                on_complete(trace)

    try:
        while pending or running:
            for shard in list(pending):
                cluster_id = shard["cluster_id"]
                task_id = shard["task_id"]
                if running_cluster.get(cluster_id, 0) >= shard["cluster_quota"]:
                    continue
                if running_task.get(task_id, 0) >= shard["max_parallel"]:
                    continue
                if running_resource + shard["resource_units"] > global_budget:
                    continue
                pending.remove(shard)
                now = monotonic_ms()
                trace = traces[shard["shard_id"]]
                trace["wait_ended_ms"] = now
                trace["started_ms"] = now
                running_resource += shard["resource_units"]
                running_cluster[cluster_id] = running_cluster.get(cluster_id, 0) + 1
                running_task[task_id] = running_task.get(task_id, 0) + 1
                peak_resource = max(peak_resource, running_resource)
                peak_cluster[cluster_id] = max(peak_cluster.get(cluster_id, 0), running_cluster[cluster_id])
                peak_task[task_id] = max(peak_task.get(task_id, 0), running_task[task_id])
                future = executor.submit(dispatch, shard)
                running[future] = _Running(shard=shard, trace=trace)

            if not running and pending:
                now = monotonic_ms()
                for shard in pending:
                    trace = traces[shard["shard_id"]]
                    trace["outcome"] = "blocked"
                    trace["error_code"] = "E_BUDGET"
                    trace["error_message"] = "no queued shard can satisfy the combined budget constraints"
                    trace["wait_ended_ms"] = now
                    if on_complete is not None:
                        on_complete(trace)
                pending.clear()
                break
            if not running:
                break

            completed, _ = wait(set(running), return_when=FIRST_COMPLETED, timeout=1.0)
            if not completed:
                continue
            for future in sorted(
                completed,
                key=lambda item: plan_order[running[item].shard["shard_id"]],
            ):
                context = running.pop(future)
                shard = context.shard
                running_resource -= shard["resource_units"]
                running_cluster[shard["cluster_id"]] -= 1
                running_task[shard["task_id"]] -= 1
                result: dict[str, Any] | None = None
                exc: BaseException | None = None
                try:
                    result = future.result()
                except BaseException as caught:  # Keep one shard failure from cancelling peers.
                    exc = caught
                completed_trace(shard, result, exc)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    ordered_traces = [traces[shard["shard_id"]] for shard in shards]
    completed_count = sum(1 for trace in ordered_traces if trace["outcome"] == "completed")
    failed_count = sum(1 for trace in ordered_traces if trace["outcome"] == "failed")
    blocked_count = sum(1 for trace in ordered_traces if trace["outcome"] == "blocked")
    return {
        "run_id": "run-" + str(plan["plan_digest"])[:20],
        "plan_digest": plan["plan_digest"],
        "started_ms": started_at,
        "finished_ms": monotonic_ms(),
        "trace": ordered_traces,
        "peak": {
            "resource_units": peak_resource,
            "cluster_concurrency": dict(sorted(peak_cluster.items())),
            "task_concurrency": dict(sorted(peak_task.items())),
        },
        "limits": {
            "global_resource_units": global_budget,
            "cluster_quotas": {
                cluster_id: quota
                for cluster_id, quota in sorted(
                    {shard["cluster_id"]: shard["cluster_quota"] for shard in shards}.items()
                )
            },
            "task_parallel_limits": {
                task_id: limit
                for task_id, limit in sorted(
                    {shard["task_id"]: shard["max_parallel"] for shard in shards}.items()
                )
            },
        },
        "counts": {
            "total": len(ordered_traces),
            "completed": completed_count,
            "failed": failed_count,
            "blocked": blocked_count,
        },
    }
