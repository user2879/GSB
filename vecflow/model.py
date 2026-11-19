"""Immutable, validated columnar data and task definitions."""

from __future__ import annotations

import copy
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import digest_value, load_json_file, loads_strict, write_json_atomic
from .errors import AppError
from .expression import ParsedExpression, parse_expression


FORMAT_VERSION = 1
TYPE_ALIASES = {
    "integer": "integer",
    "int": "integer",
    "int64": "integer",
    "float": "float",
    "double": "float",
    "float64": "float",
    "boolean": "boolean",
    "bool": "boolean",
}
AGGREGATION_MODES = {"concat", "sum", "count", "min", "max", "mean", "any", "all", "first", "last"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise AppError("E_DATA", f"{field_name} 必须是整数", {"field": field_name, "value": value})
    if minimum is not None and value < minimum:
        raise AppError("E_DATA", f"{field_name} 必须 >= {minimum}", {"field": field_name, "value": value})
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppError("E_DATA", f"{field_name} 必须是非空字符串", {"field": field_name, "value": value})
    return value.strip()


def _normalize_type(raw: Any, context: str) -> str:
    if not isinstance(raw, str):
        raise AppError("E_DATA", f"{context} 的列类型必须是字符串")
    normalized = TYPE_ALIASES.get(raw.strip().lower())
    if normalized is None:
        raise AppError(
            "E_DATA",
            f"{context} 的列类型 {raw!r} 不受支持；允许 integer/float/boolean",
            {"context": context, "allowed": ["integer", "float", "boolean"]},
        )
    return normalized


def _coerce_value(raw: Any, col_type: str, context: str) -> Any:
    if raw is None:
        return None
    if col_type == "integer":
        if not _is_int(raw):
            raise AppError("E_DATA", f"{context} 需要 integer 样本值，实际 {type(raw).__name__}", {"value": raw})
        return int(raw)
    if col_type == "float":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise AppError("E_DATA", f"{context} 需要 float 样本值，实际 {type(raw).__name__}", {"value": raw})
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            raise AppError("E_DATA", f"{context} 不允许非有限浮点值", {"value": raw})
        return value
    if isinstance(raw, bool):
        return bool(raw)
    if _is_int(raw) and raw in (0, 1):
        return bool(raw)
    raise AppError("E_DATA", f"{context} 需要 boolean 样本值", {"value": raw})


def _parse_null_bitmap(raw: Any, *, row_count: int, block_id: str, column_name: str) -> tuple[int, ...]:
    if isinstance(raw, str):
        if any(ch not in "01" for ch in raw):
            raise AppError(
                "E_NULLMAP",
                f"空值位图只能包含 0/1: block={block_id} column={column_name}",
                {"block_id": block_id, "column": column_name},
            )
        values = [int(ch) for ch in raw]
    elif isinstance(raw, list):
        values = []
        for item in raw:
            if isinstance(item, bool):
                values.append(int(item))
            elif _is_int(item) and item in (0, 1):
                values.append(item)
            else:
                raise AppError(
                    "E_NULLMAP",
                    f"空值位图只能包含 0/1: block={block_id} column={column_name}",
                    {"block_id": block_id, "column": column_name, "value": item},
                )
    else:
        raise AppError(
            "E_NULLMAP",
            f"空值位图必须是数组或 0/1 字符串: block={block_id} column={column_name}",
            {"block_id": block_id, "column": column_name},
        )
    if len(values) != row_count:
        raise AppError(
            "E_NULLMAP",
            f"空值位图长度 {len(values)} 与行数 {row_count} 不一致: block={block_id} column={column_name}",
            {
                "block_id": block_id,
                "column": column_name,
                "expected": row_count,
                "actual": len(values),
            },
        )
    return tuple(values)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    values: tuple[Any, ...]
    null_bitmap: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "values": list(self.values),
            "null_bitmap": list(self.null_bitmap),
        }


@dataclass(frozen=True)
class BlockSpec:
    id: str
    row_count: int
    columns: tuple[ColumnSpec, ...]

    def column_map(self) -> dict[str, ColumnSpec]:
        return {column.name: column for column in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True)
class DataCollection:
    blocks: tuple[BlockSpec, ...]
    format_version: int = FORMAT_VERSION

    def by_id(self) -> dict[str, BlockSpec]:
        return {block.id: block for block in self.blocks}

    def to_dict(self) -> dict[str, Any]:
        return {"format_version": self.format_version, "blocks": [block.to_dict() for block in self.blocks]}


@dataclass(frozen=True)
class ShardSpec:
    start: int
    end: int
    resources: int

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "resources": self.resources}


@dataclass(frozen=True)
class TaskSpec:
    id: str
    block_id: str
    expression: str
    quota: int
    aggregation: str
    parsed: ParsedExpression = field(compare=False, repr=False)
    shards: tuple[ShardSpec, ...] | None = None
    shard_rows: int | None = None
    resources_per_shard: tuple[int, ...] | int | None = None
    delay_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "block": self.block_id,
            "expression": self.expression,
            "quota": self.quota,
            "aggregation": self.aggregation,
        }
        if self.shards is not None:
            payload["shards"] = [shard.to_dict() for shard in self.shards]
        else:
            payload["shard_rows"] = self.shard_rows
            if isinstance(self.resources_per_shard, tuple):
                payload["resources_per_shard"] = list(self.resources_per_shard)
            else:
                payload["resources_per_shard"] = self.resources_per_shard
        if self.delay_ms:
            payload["delay_ms"] = self.delay_ms
        return payload

    def semantic_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["expression"] = self.parsed.normalized
        payload["expression_structure"] = self.parsed.structural_digest
        return payload


@dataclass(frozen=True)
class TaskCollection:
    tasks: tuple[TaskSpec, ...]
    parameter_version: int = 1
    global_budget: int = 1
    threads: int = 4
    format_version: int = FORMAT_VERSION

    def by_id(self) -> dict[str, TaskSpec]:
        return {task.id: task for task in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "parameter_version": self.parameter_version,
            "global_budget": self.global_budget,
            "threads": self.threads,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def parse_data_payload(payload: Any) -> DataCollection:
    if isinstance(payload, list):
        payload = {"format_version": FORMAT_VERSION, "blocks": payload}
    if not isinstance(payload, dict):
        raise AppError("E_DATA", "数据定义顶层必须是对象")
    version = payload.get("format_version", FORMAT_VERSION)
    if not _is_int(version) or version != FORMAT_VERSION:
        raise AppError("E_DATA", f"数据格式版本不受支持: {version!r}", {"format_version": version})
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise AppError("E_DATA", "数据定义必须包含非空 blocks 数组")
    blocks: list[BlockSpec] = []
    seen_blocks: set[str] = set()
    for block_index, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise AppError("E_DATA", f"blocks[{block_index}] 必须是对象")
        block_id = _require_str(raw_block.get("id"), f"blocks[{block_index}].id")
        if block_id in seen_blocks:
            raise AppError("E_DATA", f"数据块标识重复: {block_id}", {"block_id": block_id})
        seen_blocks.add(block_id)
        raw_columns = raw_block.get("columns")
        if not isinstance(raw_columns, list) or not raw_columns:
            raise AppError("E_DATA", f"数据块 {block_id} 必须包含非空 columns 数组")

        normalized_columns: list[ColumnSpec] = []
        seen_columns: set[str] = set()
        inferred_rows: int | None = None
        for column_index, raw_column in enumerate(raw_columns):
            if not isinstance(raw_column, dict):
                raise AppError("E_DATA", f"数据块 {block_id} columns[{column_index}] 必须是对象")
            name = _require_str(raw_column.get("name"), f"数据块 {block_id} 列名")
            if name in seen_columns:
                raise AppError("E_DATA", f"列名重复: block={block_id} column={name}", {"block_id": block_id, "column": name})
            seen_columns.add(name)
            col_type = _normalize_type(raw_column.get("type"), f"block={block_id} column={name}")
            values_raw = raw_column.get("values")
            if not isinstance(values_raw, list):
                raise AppError("E_DATA", f"列 {block_id}.{name} 的 values 必须是数组")
            if inferred_rows is None:
                inferred_rows = len(values_raw)
            elif len(values_raw) != inferred_rows:
                raise AppError(
                    "E_DATA",
                    f"列 {block_id}.{name} 的长度 {len(values_raw)} 与首个列长度 {inferred_rows} 不一致",
                    {"block_id": block_id, "column": name},
                )
            values = tuple(_coerce_value(value, col_type, f"{block_id}.{name}[{idx}]") for idx, value in enumerate(values_raw))
            null_bitmap = _parse_null_bitmap(
                raw_column.get("null_bitmap"),
                row_count=len(values),
                block_id=block_id,
                column_name=name,
            )
            for row_index, (bit, value) in enumerate(zip(null_bitmap, values)):
                if bit == 0 and value is None:
                    raise AppError(
                        "E_DATA",
                        f"列 {block_id}.{name} 第 {row_index} 行为 null，但空值位图为 0",
                        {"block_id": block_id, "column": name, "row": row_index},
                    )
            normalized_columns.append(ColumnSpec(name=name, type=col_type, values=values, null_bitmap=null_bitmap))

        row_count_raw = raw_block.get("row_count", inferred_rows)
        row_count = _require_int(row_count_raw, f"数据块 {block_id} row_count", minimum=0)
        for column in normalized_columns:
            if len(column.values) != row_count:
                raise AppError(
                    "E_DATA",
                    f"列 {block_id}.{column.name} 的值长度 {len(column.values)} 与 row_count {row_count} 不一致",
                    {"block_id": block_id, "column": column.name, "expected": row_count, "actual": len(column.values)},
                )
        blocks.append(BlockSpec(id=block_id, row_count=row_count, columns=tuple(normalized_columns)))
    return DataCollection(blocks=tuple(blocks), format_version=version)


def _parse_shard(raw: Any, task_id: str, index: int) -> ShardSpec:
    if not isinstance(raw, dict):
        raise AppError("E_DATA", f"任务 {task_id} 的 shards[{index}] 必须是对象")
    start = _require_int(raw.get("start", raw.get("start_row")), f"任务 {task_id} shards[{index}].start")
    end = _require_int(raw.get("end", raw.get("end_row")), f"任务 {task_id} shards[{index}].end")
    resources = _require_int(
        raw.get("resources", raw.get("resource_units")),
        f"任务 {task_id} shards[{index}].resources",
        minimum=1,
    )
    return ShardSpec(start=start, end=end, resources=resources)


def _parse_aggregation(raw: Any, task_id: str) -> str:
    if isinstance(raw, dict):
        raw = raw.get("mode")
    if not isinstance(raw, str):
        raise AppError("E_DATA", f"任务 {task_id} 的 aggregation 必须是字符串或 mode 对象")
    mode = raw.strip().lower()
    if mode not in AGGREGATION_MODES:
        raise AppError(
            "E_DATA",
            f"任务 {task_id} 的聚合口径 {raw!r} 不受支持",
            {"task_id": task_id, "allowed": sorted(AGGREGATION_MODES)},
        )
    return mode


def _parse_task(raw: Any, index: int) -> TaskSpec:
    if not isinstance(raw, dict):
        raise AppError("E_DATA", f"tasks[{index}] 必须是对象")
    task_id = _require_str(raw.get("id", raw.get("task_id")), f"tasks[{index}].id")
    block_id = _require_str(raw.get("block", raw.get("block_id", raw.get("data_block"))), f"任务 {task_id} block")
    expression_text = _require_str(raw.get("expression", raw.get("expr")), f"任务 {task_id} expression")
    parsed = parse_expression(expression_text)
    quota = _require_int(
        raw.get("quota", raw.get("concurrency", raw.get("max_concurrency"))),
        f"任务 {task_id} quota",
        minimum=1,
    )
    aggregation = _parse_aggregation(raw.get("aggregation", "concat"), task_id)
    delay_ms = _require_int(raw.get("delay_ms", 0), f"任务 {task_id} delay_ms", minimum=0)

    has_shards = "shards" in raw
    has_shard_rows = "shard_rows" in raw
    if has_shards == has_shard_rows:
        raise AppError("E_DATA", f"任务 {task_id} 必须二选一声明 shards 或 shard_rows")

    shards: tuple[ShardSpec, ...] | None = None
    shard_rows: int | None = None
    resources_per_shard: tuple[int, ...] | int | None = None
    if has_shards:
        raw_shards = raw.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise AppError("E_DATA", f"任务 {task_id} 的 shards 必须是非空数组")
        shards = tuple(_parse_shard(item, task_id, shard_index) for shard_index, item in enumerate(raw_shards))
    else:
        shard_rows = _require_int(raw.get("shard_rows"), f"任务 {task_id} shard_rows", minimum=1)
        raw_resources = raw.get("resources_per_shard", raw.get("resources"))
        if _is_int(raw_resources):
            resources_per_shard = _require_int(raw_resources, f"任务 {task_id} resources_per_shard", minimum=1)
        elif isinstance(raw_resources, list) and raw_resources:
            resources_per_shard = tuple(
                _require_int(item, f"任务 {task_id} resources_per_shard[{i}]", minimum=1)
                for i, item in enumerate(raw_resources)
            )
        else:
            raise AppError("E_DATA", f"任务 {task_id} 需要正整数 resources_per_shard 或其数组")

    return TaskSpec(
        id=task_id,
        block_id=block_id,
        expression=expression_text,
        quota=quota,
        aggregation=aggregation,
        shards=shards,
        shard_rows=shard_rows,
        resources_per_shard=resources_per_shard,
        delay_ms=delay_ms,
        parsed=parsed,
    )


def parse_tasks_payload(payload: Any) -> TaskCollection:
    if not isinstance(payload, dict):
        raise AppError("E_DATA", "任务定义顶层必须是对象")
    version = payload.get("format_version", FORMAT_VERSION)
    if not _is_int(version) or version != FORMAT_VERSION:
        raise AppError("E_DATA", f"任务格式版本不受支持: {version!r}", {"format_version": version})
    parameter_version = _require_int(payload.get("parameter_version", 1), "parameter_version", minimum=1)
    global_budget = _require_int(payload.get("global_budget", payload.get("budget")), "global_budget", minimum=0)
    threads = _require_int(payload.get("threads", payload.get("worker_threads", 4)), "threads", minimum=1)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise AppError("E_DATA", "任务定义必须包含非空 tasks 数组")
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        task = _parse_task(raw_task, index)
        if task.id in seen:
            raise AppError("E_DATA", f"任务标识重复: {task.id}", {"task_id": task.id})
        seen.add(task.id)
        tasks.append(task)
    return TaskCollection(
        tasks=tuple(tasks),
        parameter_version=parameter_version,
        global_budget=global_budget,
        threads=threads,
        format_version=version,
    )


@dataclass(frozen=True)
class WorkspaceState:
    parameter_version: int
    blocks: DataCollection
    tasks: TaskCollection
    global_budget: int
    threads: int

    def by_block(self) -> dict[str, BlockSpec]:
        return self.blocks.by_id()

    def content_digest(self) -> str:
        return digest_value(
            {
                "format_version": FORMAT_VERSION,
                "blocks": self.blocks.to_dict()["blocks"],
                "tasks": [task.semantic_dict() for task in self.tasks.tasks],
                "global_budget": self.global_budget,
                "threads": self.threads,
            }
        )

    def validate(self) -> None:
        block_map = self.by_block()
        if len(block_map) != len(self.blocks.blocks):
            raise AppError("E_DATA", "数据块标识不唯一")
        task_ids: set[str] = set()
        for task in self.tasks.tasks:
            if task.id in task_ids:
                raise AppError("E_DATA", f"任务标识重复: {task.id}", {"task_id": task.id})
            task_ids.add(task.id)
            block = block_map.get(task.block_id)
            if block is None:
                raise AppError(
                    "E_DATA",
                    f"任务 {task.id} 引用了不存在的数据块 {task.block_id}",
                    {"task_id": task.id, "block_id": task.block_id},
                )
            columns = block.column_map()
            missing = sorted(column for column in task.parsed.columns if column not in columns)
            if missing:
                raise AppError(
                    "E_DATA",
                    f"任务 {task.id} 引用了不存在的列: {', '.join(missing)}",
                    {"task_id": task.id, "missing_columns": missing, "block_id": block.id},
                )

    def export_payload(self) -> dict[str, Any]:
        tasks_payload = self.tasks.to_dict()
        tasks_payload["parameter_version"] = self.parameter_version
        tasks_payload["global_budget"] = self.global_budget
        tasks_payload["threads"] = self.threads
        return {
            "format_version": FORMAT_VERSION,
            "parameter_version": self.parameter_version,
            "data": self.blocks.to_dict(),
            "tasks": tasks_payload,
        }


def workspace_state_from_parts(
    blocks: DataCollection,
    tasks: TaskCollection,
    *,
    parameter_version: int | None = None,
    global_budget: int | None = None,
    threads: int | None = None,
) -> WorkspaceState:
    state = WorkspaceState(
        parameter_version=parameter_version if parameter_version is not None else tasks.parameter_version,
        blocks=blocks,
        tasks=tasks,
        global_budget=global_budget if global_budget is not None else tasks.global_budget,
        threads=threads if threads is not None else tasks.threads,
    )
    state.validate()
    return state


def workspace_state_from_export(payload: Any) -> WorkspaceState:
    if not isinstance(payload, dict):
        raise AppError("E_DATA", "会话状态顶层必须是对象")
    version = payload.get("format_version", FORMAT_VERSION)
    if not _is_int(version) or version != FORMAT_VERSION:
        raise AppError("E_DATA", "会话状态格式版本不受支持")
    blocks = parse_data_payload(payload.get("data"))
    tasks = parse_tasks_payload(payload.get("tasks"))
    state = workspace_state_from_parts(
        blocks,
        tasks,
        parameter_version=_require_int(payload.get("parameter_version", tasks.parameter_version), "parameter_version", minimum=1),
        global_budget=tasks.global_budget,
        threads=tasks.threads,
    )
    return state


@dataclass(frozen=True)
class EditResult:
    ok: bool
    parameter_version: int
    content_digest: str
    changed: bool
    affected_running: tuple[str, ...] = ()
    affected_queued: tuple[str, ...] = ()
    stale_run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "parameter_version": self.parameter_version,
            "content_digest": self.content_digest,
            "changed": self.changed,
            "affected_running": list(self.affected_running),
            "affected_queued": list(self.affected_queued),
            "stale_run_ids": list(self.stale_run_ids),
        }


class Workspace:
    """Thread-safe workspace with transactional, versioned edits."""

    def __init__(self, state: WorkspaceState, *, data_path: str | os.PathLike[str] | None = None, tasks_path: str | os.PathLike[str] | None = None) -> None:
        state.validate()
        self._state = state
        self.data_path = Path(data_path) if data_path else None
        self.tasks_path = Path(tasks_path) if tasks_path else None
        self._lock = threading.RLock()
        self._active_runs: list[Any] = []
        self.current_run: Any = None

    @classmethod
    def load(cls, data_path: str | os.PathLike[str], tasks_path: str | os.PathLike[str]) -> "Workspace":
        blocks = parse_data_payload(load_json_file(data_path))
        tasks = parse_tasks_payload(load_json_file(tasks_path))
        state = workspace_state_from_parts(blocks, tasks)
        return cls(state, data_path=data_path, tasks_path=tasks_path)

    @classmethod
    def from_export_payload(cls, payload: Any, *, data_path: str | None = None, tasks_path: str | None = None) -> "Workspace":
        return cls(workspace_state_from_export(payload), data_path=data_path, tasks_path=tasks_path)

    @property
    def parameter_version(self) -> int:
        with self._lock:
            return self._state.parameter_version

    @property
    def content_digest(self) -> str:
        with self._lock:
            return self._state.content_digest()

    @property
    def state(self) -> WorkspaceState:
        with self._lock:
            return self._state

    @property
    def global_budget(self) -> int:
        return self.state.global_budget

    @property
    def threads(self) -> int:
        return self.state.threads

    def data_payload(self) -> dict[str, Any]:
        return self.state.blocks.to_dict()

    def tasks_payload(self) -> dict[str, Any]:
        state = self.state
        payload = state.tasks.to_dict()
        payload["parameter_version"] = state.parameter_version
        payload["global_budget"] = state.global_budget
        payload["threads"] = state.threads
        return payload

    def data_json(self) -> str:
        return json.dumps(self.data_payload(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def tasks_json(self) -> str:
        return json.dumps(self.tasks_payload(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def export_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._state.export_payload()

    def register_run(self, control: Any) -> None:
        with self._lock:
            self._active_runs.append(control)

    def unregister_run(self, control: Any) -> None:
        with self._lock:
            try:
                self._active_runs.remove(control)
            except ValueError:
                pass

    def _invalidate_active(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        running: list[str] = []
        queued: list[str] = []
        run_ids: list[str] = []
        current = self.current_run
        if current is not None:
            is_stale = bool(current.get("stale")) if isinstance(current, dict) else bool(getattr(current, "stale", False))
            if not is_stale:
                marker = getattr(current, "mark_stale", None)
                if callable(marker):
                    marker()
                elif isinstance(current, dict):
                    current["stale"] = True
                    current["status"] = "stale"
                run_id = current.get("run_id") if isinstance(current, dict) else getattr(current, "run_id", None)
                if run_id:
                    run_ids.append(str(run_id))
        for control in list(self._active_runs):
            affected = control.invalidate("parameters changed")
            run_ids.append(str(affected.get("run_id", "")))
            running.extend(affected.get("running", []))
            queued.extend(affected.get("queued", []))
        return tuple(sorted(filter(None, running))), tuple(sorted(filter(None, queued))), tuple(sorted(filter(None, run_ids)))

    def _commit_candidate(self, candidate: WorkspaceState, *, persist: str | None) -> EditResult:
        with self._lock:
            previous = self._state
            previous_digest = previous.content_digest()
            candidate_digest = candidate.content_digest()
            changed = previous_digest != candidate_digest
            if not changed:
                return EditResult(True, previous.parameter_version, previous_digest, False)

            # Persist before publishing the candidate so a failed disk write cannot
            # leave memory and files disagreeing.
            try:
                if persist == "data" and self.data_path is not None:
                    write_json_atomic(self.data_path, candidate.blocks.to_dict())
                elif persist == "tasks" and self.tasks_path is not None:
                    tasks_payload = candidate.tasks.to_dict()
                    tasks_payload["parameter_version"] = candidate.parameter_version
                    tasks_payload["global_budget"] = candidate.global_budget
                    tasks_payload["threads"] = candidate.threads
                    write_json_atomic(self.tasks_path, tasks_payload)
            except OSError as exc:
                raise AppError("E_DATA", f"保存定义失败: {exc}") from exc

            running, queued, run_ids = self._invalidate_active()
            self._state = candidate
            return EditResult(
                True,
                candidate.parameter_version,
                candidate_digest,
                True,
                affected_running=running,
                affected_queued=queued,
                stale_run_ids=run_ids,
            )

    def _with_next_version(self, blocks: DataCollection, tasks: TaskCollection, global_budget: int, threads: int) -> WorkspaceState:
        candidate = WorkspaceState(
            parameter_version=self._state.parameter_version + 1,
            blocks=blocks,
            tasks=tasks,
            global_budget=global_budget,
            threads=threads,
        )
        candidate.validate()
        return candidate

    def replace_data_json(self, text: str, *, persist: bool = True) -> EditResult:
        try:
            payload = loads_strict(text, source="<data editor>")
            blocks = parse_data_payload(payload)
            with self._lock:
                candidate = self._with_next_version(blocks, self._state.tasks, self._state.global_budget, self._state.threads)
                return self._commit_candidate(candidate, persist="data" if persist else None)
        except AppError as exc:
            if exc.code == "E_DATA" and exc.details.get("surface") == "editor":
                raise
            raise AppError("E_DATA", f"数据编辑被整体拒绝: {exc.message}", {**exc.details, "surface": "editor", "cause_code": exc.code}) from exc

    def replace_tasks_json(self, text: str, *, persist: bool = True) -> EditResult:
        try:
            payload = loads_strict(text, source="<task editor>")
            tasks = parse_tasks_payload(payload)
            with self._lock:
                candidate = self._with_next_version(
                    self._state.blocks,
                    tasks,
                    tasks.global_budget,
                    tasks.threads,
                )
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            if exc.code == "E_DATA" and exc.details.get("surface") == "editor":
                raise
            raise AppError("E_DATA", f"任务编辑被整体拒绝: {exc.message}", {**exc.details, "surface": "editor", "cause_code": exc.code}) from exc

    def update_global_budget(self, budget: int, *, persist: bool = True) -> EditResult:
        try:
            budget_value = _require_int(budget, "global_budget", minimum=0)
            with self._lock:
                candidate = self._with_next_version(self._state.blocks, self._state.tasks, budget_value, self._state.threads)
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            raise AppError("E_DATA", f"预算修改被整体拒绝: {exc.message}", {**exc.details, "cause_code": exc.code}) from exc

    def update_threads(self, threads: int, *, persist: bool = True) -> EditResult:
        try:
            threads_value = _require_int(threads, "threads", minimum=1)
            with self._lock:
                candidate = self._with_next_version(self._state.blocks, self._state.tasks, self._state.global_budget, threads_value)
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            raise AppError("E_DATA", f"线程数修改被整体拒绝: {exc.message}", {**exc.details, "cause_code": exc.code}) from exc

    def update_runtime_settings(self, budget: int, threads: int, *, persist: bool = True) -> EditResult:
        try:
            budget_value = _require_int(budget, "global_budget", minimum=0)
            threads_value = _require_int(threads, "threads", minimum=1)
            with self._lock:
                candidate = self._with_next_version(
                    self._state.blocks,
                    self._state.tasks,
                    budget_value,
                    threads_value,
                )
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            raise AppError("E_DATA", f"预算/线程修改被整体拒绝: {exc.message}", {**exc.details, "cause_code": exc.code}) from exc

    def update_task_quota(self, task_id: str, quota: int, *, persist: bool = True) -> EditResult:
        try:
            quota_value = _require_int(quota, f"任务 {task_id} quota", minimum=1)
            with self._lock:
                tasks = list(self._state.tasks.tasks)
                found = False
                for index, task in enumerate(tasks):
                    if task.id == task_id:
                        tasks[index] = TaskSpec(
                            id=task.id,
                            block_id=task.block_id,
                            expression=task.expression,
                            quota=quota_value,
                            aggregation=task.aggregation,
                            shards=task.shards,
                            shard_rows=task.shard_rows,
                            resources_per_shard=task.resources_per_shard,
                            delay_ms=task.delay_ms,
                            parsed=task.parsed,
                        )
                        found = True
                        break
                if not found:
                    raise AppError("E_DATA", f"任务不存在: {task_id}", {"task_id": task_id})
                task_collection = TaskCollection(
                    tasks=tuple(tasks),
                    parameter_version=self._state.parameter_version + 1,
                    global_budget=self._state.global_budget,
                    threads=self._state.threads,
                )
                candidate = self._with_next_version(self._state.blocks, task_collection, self._state.global_budget, self._state.threads)
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            raise AppError("E_DATA", f"任务配额修改被整体拒绝: {exc.message}", {**exc.details, "cause_code": exc.code}) from exc

    def update_expression(self, task_id: str, expression: str, *, persist: bool = True) -> EditResult:
        try:
            parsed = parse_expression(expression)
            with self._lock:
                block_id = None
                tasks = list(self._state.tasks.tasks)
                for index, task in enumerate(tasks):
                    if task.id == task_id:
                        block = self._state.by_block()[task.block_id]
                        missing = [column for column in parsed.columns if column not in block.column_map()]
                        if missing:
                            raise AppError("E_DATA", f"表达式引用了不存在的列: {', '.join(missing)}", {"missing_columns": missing})
                        tasks[index] = TaskSpec(
                            id=task.id,
                            block_id=task.block_id,
                            expression=expression,
                            quota=task.quota,
                            aggregation=task.aggregation,
                            shards=task.shards,
                            shard_rows=task.shard_rows,
                            resources_per_shard=task.resources_per_shard,
                            delay_ms=task.delay_ms,
                            parsed=parsed,
                        )
                        block_id = task.block_id
                        break
                if block_id is None:
                    raise AppError("E_DATA", f"任务不存在: {task_id}", {"task_id": task_id})
                task_collection = TaskCollection(
                    tasks=tuple(tasks),
                    parameter_version=self._state.parameter_version + 1,
                    global_budget=self._state.global_budget,
                    threads=self._state.threads,
                )
                candidate = self._with_next_version(self._state.blocks, task_collection, self._state.global_budget, self._state.threads)
                return self._commit_candidate(candidate, persist="tasks" if persist else None)
        except AppError as exc:
            raise AppError("E_DATA", f"表达式修改被整体拒绝: {exc.message}", {**exc.details, "cause_code": exc.code}) from exc

    def restore_state(self, state: WorkspaceState) -> None:
        """Replace all workspace definitions atomically (used by snapshot restore)."""
        state.validate()
        with self._lock:
            self._invalidate_active()
            self._state = state
            self.current_run = None

    def is_current(self, parameter_version: int, content_digest: str) -> bool:
        with self._lock:
            return self._state.parameter_version == parameter_version and self._state.content_digest() == content_digest

    def publish_run_if_current(self, report: Any, parameter_version: int, content_digest: str) -> bool:
        """Publish a completed run only if its parameter snapshot is still current."""
        with self._lock:
            if self._state.parameter_version != parameter_version or self._state.content_digest() != content_digest:
                return False
            self.current_run = report
            return True

    def persist(self) -> None:
        """Persist the current validated state to the configured source files."""
        with self._lock:
            state = self._state
            try:
                if self.data_path is not None:
                    write_json_atomic(self.data_path, state.blocks.to_dict())
                if self.tasks_path is not None:
                    tasks_payload = state.tasks.to_dict()
                    tasks_payload["parameter_version"] = state.parameter_version
                    tasks_payload["global_budget"] = state.global_budget
                    tasks_payload["threads"] = state.threads
                    write_json_atomic(self.tasks_path, tasks_payload)
            except OSError as exc:
                raise AppError("E_DATA", f"保存工作区状态失败: {exc}") from exc

    def next_parameter_version(self) -> int:
        return self.parameter_version + 1


def clone_payload(payload: Any) -> Any:
    return copy.deepcopy(payload)
