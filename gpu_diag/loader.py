"""Declaration loading with fail-fast-all validation.

Validation is *atomic*: every violation is collected first; if any exists the
whole declaration is rejected with a single business exception and nothing is
partially loaded.  The specific ``E_*`` code is chosen deterministically by the
highest-severity failure (cycle > quota > memory > generic declaration).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import E_CYCLE, E_DECL, E_MEMORY, E_QUOTA, DiagError
from .models import Declaration, Pool, SessionParams, Shard, ShardState, Task

_POSITIVE = {"gpu_count", "memory_per_gpu_mib", "compute_units", "quota_share"}
_NONNEG = {"priority", "duration_ms", "submit_time_ms", "memory_budget_mib"}


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _cycle_members(graph: dict[str, tuple[str, ...]]) -> list[str]:
    """Return deterministically ordered node ids that participate in a cycle."""
    color: dict[str, int] = {node: 0 for node in graph}
    on_stack: list[str] = []
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        color[node] = 1
        on_stack.append(node)
        for nxt in graph.get(node, ()):  # edges already validated to exist
            if color.get(nxt) == 1:
                idx = on_stack.index(nxt)
                cyclic.update(on_stack[idx:])
            elif color.get(nxt) == 0:
                visit(nxt)
        on_stack.pop()
        color[node] = 2

    for node in sorted(graph):
        if color[node] == 0:
            visit(node)
    return sorted(cyclic)


def load_declaration(pools_path: str | Path, tasks_path: str | Path,
                     params_path: str | Path) -> Declaration:
    pools_raw = _load_json(pools_path)
    tasks_raw = _load_json(tasks_path)
    params_raw = _load_json(params_path)

    violations: list[dict[str, Any]] = []
    codes: set[str] = set()

    def add(code: str, entity: str, field_name: str, value: Any, expected: str) -> None:
        codes.add(code)
        violations.append({
            "code": code, "entity": entity, "field": field_name,
            "value": value, "expected": expected,
        })

    pools: dict[str, Pool] = {}
    for entry in pools_raw.get("pools", []):
        pid = entry.get("pool_id")
        if not isinstance(pid, str) or not pid:
            add(E_DECL, str(entry), "pool_id", pid, "non-empty unique string")
            continue
        if pid in pools:
            add(E_DECL, f"pool:{pid}", "pool_id", pid, "unique identifier")
        for field_name in _POSITIVE:
            if not _is_pos_int(entry.get(field_name)):
                add(E_DECL, f"pool:{pid}", field_name, entry.get(field_name),
                    "positive integer (>0)")
        if all(_is_pos_int(entry.get(f)) for f in _POSITIVE):
            pools[pid] = Pool(
                pool_id=pid,
                gpu_count=entry["gpu_count"],
                memory_per_gpu_mib=entry["memory_per_gpu_mib"],
                compute_units=entry["compute_units"],
                quota_share=entry["quota_share"],
            )

    # Quota shares must sum to the declared global share cap (default 100).
    share_cap = int(params_raw.get("quota_share_cap", 100))
    share_sum = sum(p.quota_share for p in pools.values())
    if share_sum > share_cap:
        codes.add(E_QUOTA)
        violations.append({
            "code": E_QUOTA, "entity": "global", "field": "quota_share_sum",
            "value": share_sum, "expected": f"<= quota_share_cap ({share_cap})",
        })

    tasks: dict[str, Task] = {}
    for entry in tasks_raw.get("tasks", []):
        tid = entry.get("task_id")
        if not isinstance(tid, str) or not tid:
            add(E_DECL, str(entry), "task_id", tid, "non-empty unique string")
            continue
        if tid in tasks:
            add(E_DECL, f"task:{tid}", "task_id", tid, "unique identifier")
        if not _is_pos_int(entry.get("gpu_need")):
            add(E_DECL, f"task:{tid}", "gpu_need", entry.get("gpu_need"),
                "positive integer (>0)")
        if not _is_pos_int(entry.get("memory_per_gpu_mib")):
            add(E_DECL, f"task:{tid}", "memory_per_gpu_mib",
                entry.get("memory_per_gpu_mib"), "positive integer (>0)")
        for field_name in _NONNEG:
            if not _is_nonneg_int(entry.get(field_name)):
                add(E_DECL, f"task:{tid}", field_name, entry.get(field_name),
                    "non-negative integer (>=0)")
        shard_count = entry.get("shard_count")
        if not _is_pos_int(shard_count):
            add(E_DECL, f"task:{tid}", "shard_count", shard_count,
                "positive integer (>0)")
        deps = tuple(entry.get("deps", ()))
        if not isinstance(entry.get("deps", ()), list) or not all(
                isinstance(d, str) for d in deps):
            add(E_DECL, f"task:{tid}", "deps", entry.get("deps"),
                "list of existing task ids")
        budget = entry.get("memory_budget_mib", 0)
        if _is_nonneg_int(budget) and _is_pos_int(entry.get("memory_per_gpu_mib")):
            per_gpu = entry["memory_per_gpu_mib"]
            if budget and budget < per_gpu:
                codes.add(E_MEMORY)
                violations.append({
                    "code": E_MEMORY, "entity": f"task:{tid}",
                    "field": "memory_budget_mib", "value": budget,
                    "expected": f">= memory_per_gpu_mib ({per_gpu})",
                })
        if all(_is_pos_int(entry.get(f)) for f in
               ("gpu_need", "memory_per_gpu_mib", "shard_count")) and \
                all(_is_nonneg_int(entry.get(f)) for f in _NONNEG) and \
                isinstance(deps, tuple):
            tasks[tid] = Task(
                task_id=tid, gpu_need=entry["gpu_need"],
                memory_per_gpu_mib=entry["memory_per_gpu_mib"],
                priority=entry["priority"], duration_ms=entry["duration_ms"],
                shard_count=entry["shard_count"],
                submit_time_ms=entry["submit_time_ms"],
                memory_budget_mib=budget, deps=deps,
            )

    # Task dependency existence + cycle.
    task_graph = {tid: t.deps for tid, t in tasks.items()}
    for tid, deps in task_graph.items():
        for dep in deps:
            if dep not in tasks:
                add(E_DECL, f"task:{tid}", "deps", dep, "existing task id")
    if all(d in tasks for deps in task_graph.values() for d in deps):
        for member in _cycle_members(task_graph):
            add(E_CYCLE, f"task:{member}", "deps", tasks[member].deps,
                "acyclic task dependency graph")

    # Build shards: explicit per-shard dependency refs validated for existence.
    shards: dict[str, Shard] = {}
    shard_graph: dict[str, tuple[str, ...]] = {}
    explicit = tasks_raw.get("shards", [])
    for tid, task in tasks.items():
        listed = [s for s in explicit if s.get("task_id") == tid]
        for index in range(task.shard_count):
            match = next((s for s in listed if s.get("index") == index), None)
            sid = f"{tid}#{index}"
            refs = tuple(match.get("deps", ())) if match else ()
            if match is not None and (not isinstance(match.get("deps", []), list)
                                      or not all(isinstance(d, str) for d in refs)):
                add(E_DECL, f"shard:{sid}", "deps", match.get("deps"),
                    "list of existing shard global ids (task#index)")
                refs = ()
            shards[sid] = Shard(shard_id=sid, task_id=tid, index=index,
                                deps=refs, state=ShardState.PENDING)
            shard_graph[sid] = refs
    for sid, refs in shard_graph.items():
        for ref in refs:
            if ref not in shards:
                add(E_DECL, f"shard:{sid}", "deps", ref,
                    "existing shard global id (task#index)")
    if all(r in shards for refs in shard_graph.values() for r in refs):
        for member in _cycle_members(shard_graph):
            add(E_CYCLE, f"shard:{member}", "deps", shards[member].deps,
                "acyclic shard dependency graph")

    # Memory fit against every target pool capacity.
    for tid, task in tasks.items():
        for pool in pools.values():
            if task.memory_per_gpu_mib > pool.memory_per_gpu_mib:
                codes.add(E_MEMORY)
                violations.append({
                    "code": E_MEMORY, "entity": f"task:{tid}",
                    "field": "memory_per_gpu_mib",
                    "value": task.memory_per_gpu_mib,
                    "expected": f"<= pool {pool.pool_id} memory_per_gpu_mib "
                                f"({pool.memory_per_gpu_mib})",
                })
                break

    params = _parse_params(params_raw, add)

    if violations:
        severity = [E_CYCLE, E_QUOTA, E_MEMORY, E_DECL]
        primary = next(code for code in severity if code in codes)
        raise DiagError(primary, "declaration rejected atomically",
                        sorted(violations, key=lambda v: (v["entity"], v["field"])))
    return Declaration(pools=pools, tasks=tasks, shards=shards, params=params)


def _parse_params(raw: dict[str, Any], add) -> SessionParams:
    required = ("global_gpu_quota", "pool_concurrency_limit")
    for field_name in required:
        if not _is_pos_int(raw.get(field_name)):
            add(E_DECL, "session", field_name, raw.get(field_name),
                "positive integer (>0)")
    scale = raw.get("scale_factor", 1000)
    if not (_is_pos_int(scale) and _is_power_of_ten(scale)):
        add(E_DECL, "session", "scale_factor", scale,
            "positive integer power of ten (10^k, k>=0)")
    return SessionParams(
        global_gpu_quota=raw.get("global_gpu_quota", 0),
        pool_concurrency_limit=raw.get("pool_concurrency_limit", 0),
        scale_factor=scale,
        truncation="floor",
        ledger_format_version=int(raw.get("ledger_format_version", 1)),
        pool_version=str(raw.get("pool_version", "pool-v1")),
        task_version=str(raw.get("task_version", "task-v1")),
        scale_version=str(raw.get("scale_version", "scale-v1")),
    )


def _is_power_of_ten(value: int) -> bool:
    while value > 1:
        if value % 10 != 0:
            return False
        value //= 10
    return value == 1
