from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .errors import SpecError
from .hashing import digest_obj

ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHARD_RE = re.compile(r"^(?P<task>[A-Za-z0-9_.-]+)#(?P<index>\d+)$")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(
    obj: dict[str, Any],
    key: str,
    *,
    entity_type: str,
    entity_id: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if key not in obj:
        raise SpecError(
            f"{entity_type} {entity_id!r} missing field {key!r}",
            locator={"entity_type": entity_type, "entity_id": entity_id, "field": key},
        )
    value = obj[key]
    if not is_int(value):
        raise SpecError(
            f"{entity_type} {entity_id!r} field {key!r} must be an integer",
            locator={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "field": key,
                "value_type": type(value).__name__,
            },
        )
    if positive and value <= 0:
        raise SpecError(
            f"{entity_type} {entity_id!r} field {key!r} must be a positive integer",
            locator={"entity_type": entity_type, "entity_id": entity_id, "field": key, "value": value},
        )
    if nonnegative and value < 0:
        raise SpecError(
            f"{entity_type} {entity_id!r} field {key!r} must be a non-negative integer",
            locator={"entity_type": entity_type, "entity_id": entity_id, "field": key, "value": value},
        )
    return value


def require_str(obj: dict[str, Any], key: str, *, entity_type: str, entity_id: str) -> str:
    if key not in obj or not isinstance(obj[key], str) or not obj[key]:
        raise SpecError(
            f"{entity_type} {entity_id!r} field {key!r} must be a non-empty string",
            locator={"entity_type": entity_type, "entity_id": entity_id, "field": key},
        )
    value = obj[key]
    if not ID_RE.match(value):
        raise SpecError(
            f"{entity_type} {entity_id!r} field {key!r} contains unsupported characters",
            locator={"entity_type": entity_type, "entity_id": entity_id, "field": key, "value": value},
        )
    return value


@dataclass(frozen=True)
class PoolSpec:
    id: str
    host: str
    port: int
    gpu_count: int
    memory_mib_per_gpu: int
    integer_units_per_gpu: int
    max_concurrent_tasks: int
    quota_share: int

    @property
    def memory_capacity_mib(self) -> int:
        return self.gpu_count * self.memory_mib_per_gpu


@dataclass(frozen=True)
class TaskSpec:
    id: str
    pool: str
    gpu_count: int
    memory_mib_per_gpu: int
    priority: int
    estimated_duration_ms: int
    shard_count: int
    submitted_monotonic_ms: int
    depends_on: tuple[str, ...] = ()
    shard_dependencies: tuple[tuple[int, tuple[str, ...]], ...] = ()
    admission_after_running: str | None = None
    duration_overrides_ms: tuple[int, ...] | None = None

    @property
    def memory_per_shard_mib(self) -> int:
        return self.gpu_count * self.memory_mib_per_gpu

    def shard_id(self, index: int) -> str:
        return f"{self.id}#{index}"

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(self.shard_id(i) for i in range(self.shard_count))


@dataclass(frozen=True)
class Spec:
    path: Path | None
    raw: dict[str, Any]
    format_version: int
    global_gpu_quota: int
    scale_power: int
    pools: tuple[PoolSpec, ...]
    tasks: tuple[TaskSpec, ...]
    pool_by_id: dict[str, PoolSpec] = field(default_factory=dict)
    task_by_id: dict[str, TaskSpec] = field(default_factory=dict)

    @property
    def scale_factor(self) -> int:
        return 10**self.scale_power

    @property
    def spec_digest(self) -> str:
        return digest_obj(self.raw)

    @property
    def pool_version(self) -> dict[str, Any]:
        payload = {"format_version": self.format_version, "pools": self.raw["pools"]}
        return {"version": self.format_version, "digest": digest_obj(payload)}

    @property
    def task_version(self) -> dict[str, Any]:
        payload = {"format_version": self.format_version, "tasks": self.raw["tasks"]}
        return {"version": self.format_version, "digest": digest_obj(payload)}

    @property
    def scale_version(self) -> dict[str, Any]:
        from . import SCALE_CALIBER_VERSION

        payload = {"caliber": SCALE_CALIBER_VERSION, "scale_power": self.scale_power}
        return {"version": 1, "caliber": SCALE_CALIBER_VERSION, "digest": digest_obj(payload)}


def parse_shard_ref(ref: Any, current_task: TaskSpec | None = None) -> str:
    if is_int(ref) and current_task is not None:
        return current_task.shard_id(ref)  # bounds checked by caller after context
    if isinstance(ref, str):
        match = SHARD_RE.match(ref)
        if match:
            return ref
        if current_task is not None and re.match(r"^\d+$", ref):
            return current_task.shard_id(int(ref))
    raise SpecError(
        "shard dependency reference must be an integer or 'task-id#index'",
        locator={"field": "shard_dependencies.requires", "value": ref},
    )


def load_spec(path: str | Path, *, validate_graph: bool = True) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"spec file not found: {path}", locator={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"spec JSON is invalid: {exc.msg}",
            locator={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(raw, dict):
        raise SpecError("top-level spec must be a JSON object", locator={"path": str(path)})

    if "format_version" not in raw or not is_int(raw["format_version"]) or raw["format_version"] != 1:
        raise SpecError(
            "format_version must be integer 1",
            locator={"entity_type": "spec", "entity_id": path.name, "field": "format_version"},
        )
    global_quota = require_int(
        raw, "global_gpu_quota", entity_type="spec", entity_id=path.name, positive=True
    )
    scale_power = raw.get("scale_power", 2)
    if not is_int(scale_power) or not 0 <= scale_power <= 3:
        raise SpecError(
            "scale_power must be an integer in [0, 3]",
            locator={"entity_type": "spec", "entity_id": path.name, "field": "scale_power", "value": scale_power},
        )

    pools_raw = raw.get("pools")
    if not isinstance(pools_raw, list) or not pools_raw:
        raise SpecError(
            "pools must be a non-empty array",
            locator={"entity_type": "spec", "entity_id": path.name, "field": "pools"},
        )

    pools: list[PoolSpec] = []
    seen_pool_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item in enumerate(pools_raw):
        if not isinstance(item, dict):
            raise SpecError(
                f"pools[{index}] must be an object",
                locator={"entity_type": "pool", "entity_id": f"pools[{index}]", "field": "<object>"},
            )
        entity_id = str(item.get("id", f"pools[{index}]"))
        pool_id = require_str(item, "id", entity_type="pool", entity_id=entity_id)
        if pool_id in seen_pool_ids:
            raise SpecError(
                f"duplicate pool id {pool_id!r}",
                locator={"entity_type": "pool", "entity_id": pool_id, "field": "id"},
            )
        seen_pool_ids.add(pool_id)
        host = require_str(item, "host", entity_type="pool", entity_id=pool_id)
        if host != "127.0.0.1":
            raise SpecError(
                "host must be the literal 127.0.0.1",
                locator={"entity_type": "pool", "entity_id": pool_id, "field": "host", "value": host},
            )
        port = require_int(item, "port", entity_type="pool", entity_id=pool_id, positive=True)
        if not 1024 <= port <= 65535:
            raise SpecError(
                "port must be in the high user-port range 1024..65535",
                locator={"entity_type": "pool", "entity_id": pool_id, "field": "port", "value": port},
            )
        if port in seen_ports:
            raise SpecError(
                f"loopback port {port} is duplicated",
                locator={"entity_type": "pool", "entity_id": pool_id, "field": "port", "value": port},
            )
        seen_ports.add(port)
        gpu_count = require_int(item, "gpu_count", entity_type="pool", entity_id=pool_id, positive=True)
        memory = require_int(item, "memory_mib_per_gpu", entity_type="pool", entity_id=pool_id, positive=True)
        units = require_int(item, "integer_units_per_gpu", entity_type="pool", entity_id=pool_id, positive=True)
        concurrency = require_int(item, "max_concurrent_tasks", entity_type="pool", entity_id=pool_id, positive=True)
        share = require_int(item, "quota_share", entity_type="pool", entity_id=pool_id, positive=True)
        if share > gpu_count:
            raise SpecError(
                "quota_share is a GPU quota in cards and cannot exceed gpu_count",
                locator={"entity_type": "pool", "entity_id": pool_id, "field": "quota_share", "value": share},
            )
        pools.append(PoolSpec(pool_id, host, port, gpu_count, memory, units, concurrency, share))

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise SpecError(
            "tasks must be a non-empty array",
            locator={"entity_type": "spec", "entity_id": path.name, "field": "tasks"},
        )

    pool_ids = {pool.id for pool in pools}
    tasks: list[TaskSpec] = []
    seen_task_ids: set[str] = set()
    for index, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise SpecError(
                f"tasks[{index}] must be an object",
                locator={"entity_type": "task", "entity_id": f"tasks[{index}]", "field": "<object>"},
            )
        entity_id = str(item.get("id", f"tasks[{index}]"))
        task_id = require_str(item, "id", entity_type="task", entity_id=entity_id)
        if task_id in seen_task_ids:
            raise SpecError(
                f"duplicate task id {task_id!r}",
                locator={"entity_type": "task", "entity_id": task_id, "field": "id"},
            )
        seen_task_ids.add(task_id)
        pool_id = require_str(item, "pool", entity_type="task", entity_id=task_id)
        if pool_id not in pool_ids:
            raise SpecError(
                f"task {task_id!r} refers to unknown pool {pool_id!r}",
                locator={"entity_type": "task", "entity_id": task_id, "field": "pool", "value": pool_id},
            )
        gpu_count = require_int(item, "gpu_count", entity_type="task", entity_id=task_id, positive=True)
        memory = require_int(item, "memory_mib_per_gpu", entity_type="task", entity_id=task_id, positive=True)
        priority = require_int(item, "priority", entity_type="task", entity_id=task_id)
        duration = require_int(
            item, "estimated_duration_ms", entity_type="task", entity_id=task_id, nonnegative=True
        )
        shard_count = require_int(item, "shard_count", entity_type="task", entity_id=task_id, positive=True)
        submitted = require_int(
            item, "submitted_monotonic_ms", entity_type="task", entity_id=task_id, nonnegative=True
        )
        pool = next(p for p in pools if p.id == pool_id)
        if gpu_count > pool.gpu_count:
            raise SpecError(
                "task gpu_count cannot exceed pool gpu_count",
                locator={"entity_type": "task", "entity_id": task_id, "field": "gpu_count", "value": gpu_count},
            )
        if memory > pool.memory_mib_per_gpu:
            raise SpecError(
                "task memory_mib_per_gpu cannot exceed pool per-card memory",
                locator={
                    "entity_type": "task",
                    "entity_id": task_id,
                    "field": "memory_mib_per_gpu",
                    "value": memory,
                },
            )
        depends_on_raw = item.get("depends_on", [])
        if not isinstance(depends_on_raw, list) or not all(isinstance(x, str) for x in depends_on_raw):
            raise SpecError(
                "depends_on must be an array of task id strings",
                locator={"entity_type": "task", "entity_id": task_id, "field": "depends_on"},
            )
        shard_deps: list[tuple[int, tuple[str, ...]]] = []
        deps_raw = item.get("shard_dependencies", [])
        if not isinstance(deps_raw, list):
            raise SpecError(
                "shard_dependencies must be an array",
                locator={"entity_type": "task", "entity_id": task_id, "field": "shard_dependencies"},
            )
        for dep_index, dep in enumerate(deps_raw):
            if not isinstance(dep, dict):
                raise SpecError(
                    f"shard_dependencies[{dep_index}] must be an object",
                    locator={
                        "entity_type": "task",
                        "entity_id": task_id,
                        "field": f"shard_dependencies[{dep_index}]",
                    },
                )
            shard_index = dep.get("shard")
            if not is_int(shard_index) or not 0 <= shard_index < shard_count:
                raise SpecError(
                    "shard index must be an integer within shard_count",
                    locator={
                        "entity_type": "task",
                        "entity_id": task_id,
                        "field": f"shard_dependencies[{dep_index}].shard",
                        "value": shard_index,
                    },
                )
            requires_raw = dep.get("requires", [])
            if not isinstance(requires_raw, list):
                raise SpecError(
                    "requires must be an array",
                    locator={
                        "entity_type": "task",
                        "entity_id": task_id,
                        "field": f"shard_dependencies[{dep_index}].requires",
                    },
                )
            refs: list[str] = []
            ref_context = SimpleNamespace(shard_id=lambda index, task_id=task_id: f"{task_id}#{index}")
            for ref_index, ref in enumerate(requires_raw):
                try:
                    resolved = parse_shard_ref(ref, ref_context)  # type: ignore[arg-type]
                except SpecError as exc:
                    exc.locator.update(
                        {
                            "entity_type": "task",
                            "entity_id": task_id,
                            "field": f"shard_dependencies[{dep_index}].requires[{ref_index}]",
                        }
                    )
                    raise
                if "#" in resolved:
                    ref_task_id, ref_index_text = resolved.rsplit("#", 1)
                    if ref_task_id == task_id:
                        if not 0 <= int(ref_index_text) < shard_count:
                            raise SpecError(
                                "shard dependency reference index is out of range",
                                locator={
                                    "entity_type": "task",
                                    "entity_id": task_id,
                                    "field": f"shard_dependencies[{dep_index}].requires[{ref_index}]",
                                    "value": resolved,
                                },
                            )
                refs.append(resolved)
            shard_deps.append((int(shard_index), tuple(refs)))

        admission_after_running = None
        admission = item.get("admission")
        if admission is not None:
            if not isinstance(admission, dict) or not isinstance(admission.get("after_task_running"), str):
                raise SpecError(
                    "admission.after_task_running must be a task id string",
                    locator={"entity_type": "task", "entity_id": task_id, "field": "admission"},
                )
            admission_after_running = admission["after_task_running"]

        duration_overrides_raw = item.get("duration_overrides_ms")
        duration_overrides: tuple[int, ...] | None = None
        if duration_overrides_raw is not None:
            if not isinstance(duration_overrides_raw, list) or len(duration_overrides_raw) != shard_count:
                raise SpecError(
                    "duration_overrides_ms must be an array with one non-negative integer per shard",
                    locator={"entity_type": "task", "entity_id": task_id, "field": "duration_overrides_ms"},
                )
            overrides: list[int] = []
            for override_index, value in enumerate(duration_overrides_raw):
                if not is_int(value) or value < 0:
                    raise SpecError(
                        "duration override must be a non-negative integer",
                        locator={
                            "entity_type": "task",
                            "entity_id": task_id,
                            "field": f"duration_overrides_ms[{override_index}]",
                            "value": value,
                        },
                    )
                overrides.append(int(value))
            duration_overrides = tuple(overrides)

        tasks.append(
            TaskSpec(
                id=task_id,
                pool=pool_id,
                gpu_count=gpu_count,
                memory_mib_per_gpu=memory,
                priority=priority,
                estimated_duration_ms=duration,
                shard_count=shard_count,
                submitted_monotonic_ms=submitted,
                depends_on=tuple(depends_on_raw),
                shard_dependencies=tuple(shard_deps),
                admission_after_running=admission_after_running,
                duration_overrides_ms=duration_overrides,
            )
        )

    task_ids = {task.id for task in tasks}
    for task in tasks:
        for dep_id in task.depends_on:
            if dep_id not in task_ids:
                raise SpecError(
                    f"task {task.id!r} depends on missing task {dep_id!r}",
                    locator={"entity_type": "task", "entity_id": task.id, "field": "depends_on", "value": dep_id},
                )
        if task.admission_after_running is not None:
            target = task.admission_after_running
            if target not in task_ids:
                raise SpecError(
                    f"task {task.id!r} admission refers to missing task {target!r}",
                    locator={
                        "entity_type": "task",
                        "entity_id": task.id,
                        "field": "admission.after_task_running",
                        "value": target,
                    },
                )
            if target == task.id:
                raise SpecError(
                    "admission trigger cannot refer to the same task",
                    locator={
                        "entity_type": "task",
                        "entity_id": task.id,
                        "field": "admission.after_task_running",
                        "value": target,
                    },
                )
        for _shard, refs in task.shard_dependencies:
            for ref in refs:
                if "#" not in ref:
                    # Integer refs were resolved to local task#index by parse_shard_ref
                    # only when current_task supplied. Re-parse local raw integer case
                    # through the task-aware helper below.
                    continue
                ref_task, ref_idx_text = ref.rsplit("#", 1)
                if ref_task not in task_ids:
                    raise SpecError(
                        f"shard dependency refers to missing task {ref_task!r}",
                        locator={"entity_type": "task", "entity_id": task.id, "field": "shard_dependencies", "value": ref},
                    )
                ref_task_obj = next(t for t in tasks if t.id == ref_task)
                if not ref_idx_text.isdigit() or not 0 <= int(ref_idx_text) < ref_task_obj.shard_count:
                    raise SpecError(
                        "shard dependency reference is out of range",
                        locator={"entity_type": "task", "entity_id": task.id, "field": "shard_dependencies", "value": ref},
                    )

    spec = Spec(
        path=path,
        raw=raw,
        format_version=1,
        global_gpu_quota=global_quota,
        scale_power=int(scale_power),
        pools=tuple(pools),
        tasks=tuple(tasks),
        pool_by_id={p.id: p for p in pools},
        task_by_id={t.id: t for t in tasks},
    )
    if validate_graph:
        from .graph import DependencyGraph

        DependencyGraph.build(spec).ensure_acyclic()
    return spec
