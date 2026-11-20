"""Declaration loading and strong validation.

The loader reads three local JSON documents (session, pools, tasks), performs
every structural and semantic check required by the specification and rejects
the *whole* declaration on any failure.  Normal loads surface a top-level
``E_SPEC`` while each individual :class:`Problem` keeps a granular sub-code
(``E_QUOTA`` / ``E_MEMORY`` / ``E_CYCLE`` / ``E_DEPEND``) so the fault
injection layer can reclassify a scenario to its precise business code.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .errors import (
    E_CYCLE,
    E_DEPEND,
    E_MEMORY,
    E_QUOTA,
    E_SPEC,
    Problem,
    SchedulerError,
)
from .model import (
    Declaration,
    Pool,
    SessionConfig,
    Shard,
    Task,
    content_digest,
)


class _Collector:
    """Accumulates validation problems with stable field paths."""

    def __init__(self) -> None:
        self.problems: List[Problem] = []

    def add(
        self,
        path: str,
        expected: str,
        actual: Any,
        code: str = E_SPEC,
    ) -> None:
        self.problems.append(Problem(path, expected, actual, code))

    def fail(self, top_code: str = E_SPEC, message: str = "declaration rejected") -> None:
        raise SchedulerError(top_code, message, sorted(
            self.problems, key=lambda problem: (problem.path, problem.code)))

    @property
    def has_problems(self) -> bool:
        return bool(self.problems)


def _read_json(path: str, collector: _Collector) -> Optional[dict]:
    if not os.path.isfile(path):
        collector.add(path, "existing readable JSON file", "missing")
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        collector.add(path, "valid JSON syntax", f"json error: {exc.msg}")
        return None


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; reject it explicitly to avoid True == 1.
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    collector: _Collector,
    container: dict,
    key: str,
    path: str,
    *,
    minimum: int,
    code: str = E_SPEC,
) -> Optional[int]:
    if key not in container:
        collector.add(f"{path}.{key}", f"integer >= {minimum}", "missing", code)
        return None
    value = container[key]
    if not _is_int(value) or value < minimum:
        collector.add(f"{path}.{key}", f"integer >= {minimum}", value, code)
        return None
    return value


def _require_str(
    collector: _Collector,
    container: dict,
    key: str,
    path: str,
) -> Optional[str]:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        collector.add(f"{path}.{key}", "non-empty string", value)
        return None
    return value


def _require_int_list(
    collector: _Collector,
    container: dict,
    key: str,
    path: str,
    *,
    minimum: int = 0,
) -> Optional[List[int]]:
    value = container.get(key, [])
    if not isinstance(value, list) or not all(_is_int(item) for item in value):
        collector.add(f"{path}.{key}", "list of integers", value)
        return None
    if any(item < minimum for item in value):
        collector.add(f"{path}.{key}", f"list of integers >= {minimum}", value)
        return None
    return list(value)


def _parse_session(raw: Optional[dict], collector: _Collector) -> Optional[SessionConfig]:
    if not isinstance(raw, dict):
        collector.add("session", "object with session parameters", type(raw).__name__)
        return None
    path = "session"
    total_shares = _require_int(collector, raw, "total_quota_shares", path, minimum=1)
    gpu_cap = _require_int(collector, raw, "global_gpu_cap", path, minimum=1)
    budget = _require_int(collector, raw, "default_memory_budget_mib", path,
                          minimum=1, code=E_MEMORY)
    scale = _require_int(collector, raw, "fixed_point_scale", path, minimum=1)
    seed = _require_int(collector, raw, "seed", path, minimum=0)
    origin = _require_int(collector, raw, "clock_origin_ms", path, minimum=0)
    if collector.has_problems:
        return None
    if scale not in (1, 10, 100, 1000, 10000, 100000, 1000000):
        collector.add(f"{path}.fixed_point_scale",
                      "power of ten between 1 and 1_000_000", scale)
        return None
    return SessionConfig(
        total_quota_shares=total_shares,
        global_gpu_cap=gpu_cap,
        default_memory_budget_mib=budget,
        fixed_point_scale=scale,
        seed=seed,
        clock_origin_ms=origin or 0,
    )


def _parse_pools(raw: Optional[dict], collector: _Collector,
                 total_shares: Optional[int]) -> List[Pool]:
    pools: List[Pool] = []
    if not isinstance(raw, dict) or "pools" not in raw:
        collector.add("pools", "object containing a pools list", raw)
        return pools
    entries = raw["pools"]
    if not isinstance(entries, list) or not entries:
        collector.add("pools", "non-empty list of resource pools", entries)
        return pools

    seen: Dict[str, int] = {}
    share_sum = 0
    gpu_sum = 0
    for position, entry in enumerate(entries):
        path = f"pools[{position}]"
        if not isinstance(entry, dict):
            collector.add(path, "pool object", type(entry).__name__)
            continue
        pool_id = _require_str(collector, entry, "pool_id", path)
        gpu_count = _require_int(collector, entry, "gpu_count", path, minimum=1)
        memory = _require_int(collector, entry, "memory_per_gpu_mib", path,
                              minimum=1, code=E_MEMORY)
        compute = _require_int(collector, entry, "compute_units_per_gpu",
                               path, minimum=1)
        shares = _require_int(collector, entry, "quota_shares", path,
                              minimum=1, code=E_QUOTA)
        concurrency = _require_int(collector, entry, "max_concurrent_tasks",
                                   path, minimum=1)
        if pool_id is not None:
            if pool_id in seen:
                collector.add(f"{path}.pool_id",
                              f"unique pool id (duplicate of pools[{seen[pool_id]}])",
                              pool_id)
            else:
                seen[pool_id] = position
        if None in (gpu_count, memory, compute, shares, concurrency, pool_id):
            continue
        if shares % gpu_count != 0:
            collector.add(f"{path}.quota_shares",
                          f"quota_shares divisible by gpu_count ({gpu_count})",
                          shares, code=E_QUOTA)
        share_sum += shares
        gpu_sum += gpu_count
        pools.append(Pool(
            pool_id=pool_id,
            gpu_count=gpu_count,
            memory_per_gpu_mib=memory,
            compute_units_per_gpu=compute,
            quota_shares=shares,
            max_concurrent_tasks=concurrency,
        ))

    if total_shares is not None and share_sum != total_shares:
        collector.add("pools.quota_shares",
                      f"sum == session.total_quota_shares ({total_shares})",
                      share_sum, code=E_QUOTA)
    return pools


def _parse_tasks(
    raw: Optional[dict],
    collector: _Collector,
    config: Optional[SessionConfig],
    pools: List[Pool],
) -> List[Task]:
    tasks: List[Task] = []
    if not isinstance(raw, dict) or "tasks" not in raw:
        collector.add("tasks", "object containing a tasks list", raw)
        return tasks
    entries = raw["tasks"]
    if not isinstance(entries, list) or not entries:
        collector.add("tasks", "non-empty list of tasks", entries)
        return tasks

    seen: Dict[str, int] = {}
    valid_ids: set = set()
    max_gpu = max((pool.gpu_count for pool in pools), default=0)

    for position, entry in enumerate(entries):
        path = f"tasks[{position}]"
        if not isinstance(entry, dict):
            collector.add(path, "task object", type(entry).__name__)
            continue
        task_id = _require_str(collector, entry, "task_id", path)
        kind = _require_str(collector, entry, "kind", path)
        gpu_need = _require_int(collector, entry, "gpu_need", path, minimum=1)
        memory = _require_int(collector, entry, "memory_per_gpu_mib", path,
                              minimum=1, code=E_MEMORY)
        priority = _require_int(collector, entry, "priority", path, minimum=0)
        duration = _require_int(collector, entry, "estimated_ms", path, minimum=1)
        arrival = _require_int(collector, entry, "arrival_ms", path, minimum=0)
        shard_count = _require_int(collector, entry, "shard_count", path, minimum=1)
        # Task-level dependencies are string ids; validate shape here and
        # resolve existence once every id is known.
        raw_task_deps = entry.get("depends_on_tasks", [])
        if not isinstance(raw_task_deps, list) or not all(
            isinstance(item, str) for item in raw_task_deps
        ):
            collector.add(f"{path}.depends_on_tasks", "list of task id strings",
                         raw_task_deps)
            raw_task_deps = []
        fail_once = _require_int_list(collector, entry, "fail_once_shards",
                                      path, minimum=0)
        if kind is not None and kind not in ("training", "inference"):
            collector.add(f"{path}.kind",
                          "one of 'training' or 'inference'", kind)
        if gpu_need is not None and gpu_need > max_gpu:
            collector.add(f"{path}.gpu_need",
                          f"<= largest pool gpu_count ({max_gpu})", gpu_need,
                          code=E_QUOTA)
        budget = entry.get("memory_budget_mib")
        if budget is None:
            budget = config.default_memory_budget_mib if config else 0
        elif not _is_int(budget) or budget < 1:
            collector.add(f"{path}.memory_budget_mib", "integer >= 1", budget,
                          code=E_MEMORY)
            budget = 0
        if task_id is not None:
            if task_id in seen:
                collector.add(f"{path}.task_id",
                              f"unique task id (duplicate of tasks[{seen[task_id]}])",
                              task_id)
            else:
                seen[task_id] = position
                valid_ids.add(task_id)
        if None in (gpu_need, memory, shard_count) or budget == 0:
            continue
        # A single shard already exceeds the declared task memory budget.
        if gpu_need * memory > budget:
            collector.add(f"{path}.memory_budget_mib",
                          f">= gpu_need * memory_per_gpu_mib "
                          f"({gpu_need * memory})", budget, code=E_MEMORY)
        task = Task(
            task_id=task_id or f"<invalid-{position}>",
            kind=kind or "training",
            gpu_need=gpu_need,
            memory_per_gpu_mib=memory,
            priority=priority or 0,
            estimated_ms=duration or 1,
            shard_count=shard_count,
            arrival_ms=arrival if arrival is not None else 0,
            memory_budget_mib=budget,
            depends_on_tasks=list(raw_task_deps),
            fail_once_shards=list(fail_once or []),
        )
        tasks.append(task)

    _build_shards(entries, tasks, collector)
    _validate_task_graph(tasks, collector, valid_ids)
    return tasks


def _build_shards(entries: List[dict], tasks: List[Task],
                  collector: _Collector) -> None:
    by_position = {task.task_id: task for task in tasks}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("task_id")
        task = by_position.get(task_id)
        if task is None:
            continue
        path = f"tasks[{position}].shards"
        raw_shards = entry.get("shards", [])
        if not isinstance(raw_shards, list) or len(raw_shards) != task.shard_count:
            collector.add(path,
                          f"list of {task.shard_count} shard declarations",
                          raw_shards if isinstance(raw_shards, list) else type(
                              raw_shards).__name__)
            continue
        used_indices: set = set()
        for shard_position, shard_entry in enumerate(raw_shards):
            spath = f"{path}[{shard_position}]"
            if not isinstance(shard_entry, dict):
                collector.add(spath, "shard object", type(shard_entry).__name__)
                continue
            shard_index = _require_int(collector, shard_entry, "shard_index",
                                       spath, minimum=0)
            deps = _require_int_list(collector, shard_entry, "depends_on", spath)
            if shard_index is not None:
                if shard_index in used_indices:
                    collector.add(f"{spath}.shard_index", "unique shard index",
                                  shard_index)
                used_indices.add(shard_index)
                if shard_index >= task.shard_count:
                    collector.add(f"{spath}.shard_index",
                                  f"< shard_count {task.shard_count}", shard_index)
            if deps is None:
                deps = []
            if any(dep >= task.shard_count for dep in deps):
                collector.add(f"{spath}.depends_on",
                              f"indices < shard_count {task.shard_count}", deps,
                              code=E_DEPEND)
            # Self dependency is always a cycle.
            if shard_index is not None and shard_index in deps:
                collector.add(f"{spath}.depends_on",
                              "no self dependency", deps, code=E_CYCLE)
            shard = Shard(
                task_id=task.task_id,
                shard_index=shard_index if shard_index is not None else
                shard_position,
                gpu_need=task.gpu_need,
                memory_per_gpu_mib=task.memory_per_gpu_mib,
                duration_ms=task.estimated_ms,
                depends_on=list(deps),
            )
            task.shards.append(shard)
        task.shards.sort(key=lambda shard: shard.shard_index)


def _kahn_cycle(nodes: List[str], edges: Dict[str, List[str]]) -> List[str]:
    """Return nodes that remain after Kahn's algorithm (i.e. cycle members)."""
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in edges.get(source, []):
            if target in indegree:
                indegree[target] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    visited: List[str] = []
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for target in sorted(edges.get(node, [])):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
        queue.sort()
    return sorted(node for node in nodes if node not in set(visited))


def _validate_task_graph(tasks: List[Task], collector: _Collector,
                         valid_ids: set) -> None:
    # Resolve task-level dependency references.
    edges: Dict[str, List[str]] = {}
    for task in tasks:
        resolved: List[str] = []
        for dep in task.depends_on_tasks:
            if dep not in valid_ids:
                collector.add(
                    f"task:{task.task_id}.depends_on_tasks",
                    "reference to an existing task id", dep, code=E_DEPEND)
            elif dep == task.task_id:
                collector.add(f"task:{task.task_id}.depends_on_tasks",
                              "no self dependency", dep, code=E_CYCLE)
            else:
                resolved.append(dep)
        task.depends_on_tasks = sorted(set(resolved))
        edges[task.task_id] = list(task.depends_on_tasks)

    cycle_nodes = _kahn_cycle([task.task_id for task in tasks], edges)
    if cycle_nodes:
        collector.add("tasks.depends_on_tasks",
                      "acyclic task dependency graph",
                      cycle_nodes, code=E_CYCLE)


def _validate_shard_graphs(tasks: List[Task], collector: _Collector) -> None:
    for task in tasks:
        nodes = [shard.shard_index for shard in task.shards]
        edges = {shard.shard_index: list(shard.depends_on)
                 for shard in task.shards}
        # Missing references within the task graph.
        for shard in task.shards:
            for dep in shard.depends_on:
                if dep not in nodes:
                    collector.add(
                        f"task:{task.task_id}.shard:{shard.shard_index}.depends_on",
                        "reference to an existing shard index", dep,
                        code=E_DEPEND)
        cycle_nodes = _kahn_cycle(nodes, edges)
        if cycle_nodes:
            collector.add(f"task:{task.task_id}.shards.depends_on",
                          "acyclic intra-task shard graph",
                          cycle_nodes, code=E_CYCLE)

        # Inherit task-level cross-task dependencies on every shard so the
        # runtime only has to inspect the shard object.
        for shard in task.shards:
            merged = sorted(set(shard.depends_on_tasks) |
                            set(task.depends_on_tasks))
            shard.depends_on_tasks = merged


def load_declaration(session_path: str, pools_path: str,
                     tasks_path: str) -> Declaration:
    """Load and fully validate the three declaration documents.

    Raises :class:`SchedulerError` (``E_SPEC``) with field paths and expected
    constraints if any rule is violated.  Nothing is persisted on failure.
    """
    collector = _Collector()
    session_raw = _read_json(session_path, collector)
    pools_raw = _read_json(pools_path, collector)
    tasks_raw = _read_json(tasks_path, collector)
    if collector.has_problems:
        collector.fail(message="unreadable declaration file")

    config = _parse_session(session_raw, collector)
    total_shares = config.total_quota_shares if config else None
    pools = _parse_pools(pools_raw, collector, total_shares)
    tasks = _parse_tasks(tasks_raw, collector, config, pools)
    _validate_shard_graphs(tasks, collector)

    # fail_once indices must reference real shards.
    for task in tasks:
        for index in task.fail_once_shards:
            if index >= task.shard_count:
                collector.add(f"task:{task.task_id}.fail_once_shards",
                              f"index < shard_count {task.shard_count}", index,
                              code=E_DEPEND)

    if collector.has_problems:
        # Preserve the specific sub-codes inside each problem but the load as a
        # whole is an E_SPEC rejection per requirement 1.
        collector.fail(message="declaration failed strong validation")

    pool_version = content_digest([pool.to_dict() for pool in pools])
    task_payload = []
    for task in tasks:
        item = task.to_dict()
        task_payload.append(item)
    task_version = content_digest(task_payload)
    config_version = content_digest(config.to_dict())
    return Declaration(
        config=config,
        pools=pools,
        tasks=tasks,
        pool_version=pool_version,
        task_version=task_version,
        config_version=config_version,
    )
