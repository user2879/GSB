from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .canonical import monotonic_ms, monotonic_ns
from .errors import BudgetError
from .pipeline import aggregate_shard_result, flatten_docs, plan_shards, weighted_sampling_stage
from .rpc import ComputeClient, WorkerHandle, start_worker_subprocess


def build_domain_tasks(bundle: Any, shard_ids: set[str] | list[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weighted = weighted_sampling_stage(bundle)
    plans = plan_shards(weighted["selected"], int(bundle.config["shard_size"]))
    docs_by_id = {doc["doc_id"]: doc for doc in flatten_docs(bundle)}
    wanted = set(shard_ids) if shard_ids is not None else None
    tasks: list[dict[str, Any]] = []
    used_plans: list[dict[str, Any]] = []
    for plan in plans:
        if wanted is not None and plan["shard_id"] not in wanted:
            continue
        used_plans.append(plan)
        groups: dict[str, list[dict[str, Any]]] = {}
        for doc_id in plan["member_ids"]:
            doc = docs_by_id[doc_id]
            groups.setdefault(doc["domain_id"], []).append(doc)
        for domain in plan["domains"]:
            docs = groups[domain]
            tasks.append(
                {
                    "type": "task",
                    "task_id": f"{plan['shard_id']}:{domain}",
                    "shard_id": plan["shard_id"],
                    "domain_id": domain,
                    "points": len(docs),
                    "domain_points": {domain: len(docs)},
                    "similarity_threshold_bp": int(bundle.config["similarity_threshold_bp"]),
                    "stage_params": bundle.config["stage_params"]["dedup"],
                    "docs": docs,
                }
            )
    return tasks, used_plans


def schedule_tcp_tasks(
    tasks: list[dict[str, Any]],
    *,
    host: str,
    port: int,
    timeout_ms: int = 2_000,
    global_budget_points: int,
    domain_quota_points: int | dict[str, int],
    max_workers: int = 8,
) -> dict[str, Any]:
    condition = threading.Condition()
    active_global = 0
    active_domain: dict[str, int] = {}
    peak_global = 0
    peak_domain: dict[str, int] = {}
    invariant_violations: list[dict[str, Any]] = []
    wait_records: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    global_budget = int(global_budget_points)
    if global_budget <= 0:
        raise BudgetError("global_budget_points must be positive")

    def quota_for(domain: str) -> int:
        if isinstance(domain_quota_points, dict):
            return int(domain_quota_points.get(domain, domain_quota_points.get("*", 0)))
        return int(domain_quota_points)

    for task in tasks:
        cost = int(task.get("points", 1))
        if cost > global_budget:
            raise BudgetError("task point cost exceeds global budget", task_id=task["task_id"], cost=cost, budget=global_budget)
        for domain, domain_cost in task.get("domain_points", {task["domain_id"]: cost}).items():
            if int(domain_cost) > quota_for(str(domain)):
                raise BudgetError(
                    "task point cost exceeds domain quota and can never be scheduled",
                    task_id=task["task_id"],
                    domain_id=domain,
                    cost=int(domain_cost),
                    quota=quota_for(str(domain)),
                )

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_global, peak_global
        cost = int(task.get("points", 1))
        domain_points = {str(key): int(value) for key, value in task.get("domain_points", {task["domain_id"]: cost}).items()}
        start_ns: int | None = None
        wait_start_ms: int | None = None
        acquired_ns: int | None = None

        with condition:
            while True:
                domain_ok = all(active_domain.get(domain, 0) + domain_cost <= quota_for(domain) for domain, domain_cost in domain_points.items())
                if active_global + cost <= global_budget and domain_ok:
                    break
                if start_ns is None:
                    start_ns = monotonic_ns()
                    wait_start_ms = monotonic_ms()
                condition.wait(timeout=max(0.01, timeout_ms / 1000.0))

            active_global += cost
            for domain, domain_cost in domain_points.items():
                active_domain[domain] = active_domain.get(domain, 0) + domain_cost
            acquired_ns = monotonic_ns()
            peak_global = max(peak_global, active_global)
            for domain, value in active_domain.items():
                peak_domain[domain] = max(peak_domain.get(domain, 0), value)
            if active_global > global_budget:
                invariant_violations.append({"task_id": task["task_id"], "kind": "global", "active": active_global, "budget": global_budget})
            for domain, value in active_domain.items():
                if value > quota_for(domain):
                    invariant_violations.append({"task_id": task["task_id"], "kind": "domain", "domain_id": domain, "active": value, "quota": quota_for(domain)})

        if start_ns is not None and wait_start_ms is not None and acquired_ns is not None:
            end_ns = monotonic_ns()
            record = {
                "task_id": task["task_id"],
                "shard_id": task.get("shard_id"),
                "domain_id": task.get("domain_id"),
                "wait_start_monotonic_ms": wait_start_ms,
                "wait_end_monotonic_ms": monotonic_ms(),
                "wait_ms_real": round((end_ns - start_ns) / 1_000_000.0, 3),
                "wait_ms_scheduling": round((acquired_ns - start_ns) / 1_000_000.0, 3),
            }
            with lock:
                wait_records.append(record)

        try:
            with ComputeClient(host, port, timeout_ms=timeout_ms) as client:
                result = client.submit(task)
            with lock:
                results[task["task_id"]] = result
            return result
        finally:
            with condition:
                active_global -= cost
                for domain, domain_cost in domain_points.items():
                    active_domain[domain] -= domain_cost
                    if active_domain[domain] <= 0:
                        active_domain.pop(domain, None)
                condition.notify_all()

    workers = max(1, min(max_workers, len(tasks) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reconflow-rpc") as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in futures:
            future.result()

    wait_records.sort(key=lambda item: (item["wait_start_monotonic_ms"], str(item["task_id"])))
    return {
        "task_count": len(tasks),
        "results": results,
        "wait_records": wait_records,
        "peak_global_points": peak_global,
        "peak_domain_points": dict(sorted(peak_domain.items())),
        "global_budget_points": global_budget,
        "domain_quota_points": domain_quota_points,
        "invariant_violations": invariant_violations,
    }


def recompute_shards_tcp(
    bundle: Any,
    *,
    shard_ids: set[str] | list[str] | None = None,
    worker: WorkerHandle | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    start_server: bool = True,
) -> dict[str, Any]:
    tasks, plans = build_domain_tasks(bundle, shard_ids)
    config = bundle.config
    timeout_ms = int(config["stage_params"]["dedup"].get("rpc_timeout_ms", 2_000))
    own_worker: WorkerHandle | None = None
    if worker is None and start_server:
        own_worker = start_worker_subprocess(host=host, port=port or 0, timeout_ms=timeout_ms)
        worker = own_worker
    if worker is None:
        raise RuntimeError("TCP recompute requires a WorkerHandle or start_server=True")
    try:
        scheduling = schedule_tcp_tasks(
            tasks,
            host=worker.host,
            port=worker.port,
            timeout_ms=timeout_ms,
            global_budget_points=int(config["global_sampling_point_budget"]),
            domain_quota_points=int(config["domain_sampling_point_quota"]),
            max_workers=max(2, min(8, len(tasks))),
        )
    finally:
        if own_worker is not None:
            own_worker.close()

    shard_results: list[dict[str, Any]] = []
    for plan in plans:
        domain_results = [scheduling["results"][f"{plan['shard_id']}:{domain}"] for domain in plan["domains"]]
        shard_results.append(aggregate_shard_result(plan, domain_results, threshold_bp=int(config["similarity_threshold_bp"])))
    return {"shard_results": shard_results, "plans": plans, "scheduling": scheduling}
