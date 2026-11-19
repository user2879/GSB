"""Compile validated tasks into deterministic, contiguous row-shard plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import digest_value
from .errors import AppError
from .expression import ParsedExpression
from .model import BlockSpec, TaskSpec, WorkspaceState


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    task_id: str
    index: int
    start_row: int
    end_row: int
    resources: int

    @property
    def row_count(self) -> int:
        return self.end_row - self.start_row

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "task_id": self.task_id,
            "index": self.index,
            "start": self.start_row,
            "end": self.end_row,
            "resources": self.resources,
        }


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    block_id: str
    expression_text: str
    expression_normalized: str
    expression_digest: str
    aggregation: str
    quota: int
    delay_ms: int
    shards: tuple[ShardPlan, ...]
    plan_digest: str
    parsed_expression: ParsedExpression

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "block_id": self.block_id,
            "expression": self.expression_normalized,
            "expression_digest": self.expression_digest,
            "aggregation": self.aggregation,
            "quota": self.quota,
            "delay_ms": self.delay_ms,
            "shards": [shard.to_dict() for shard in self.shards],
            "plan_digest": self.plan_digest,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    tasks: tuple[TaskPlan, ...]
    parameter_version: int
    content_digest: str
    global_budget: int
    threads: int
    plan_digest: str

    @property
    def shard_count(self) -> int:
        return sum(len(task.shards) for task in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_version": self.parameter_version,
            "content_digest": self.content_digest,
            "global_budget": self.global_budget,
            "threads": self.threads,
            "tasks": [task.to_dict() for task in self.tasks],
            "plan_digest": self.plan_digest,
        }


def _range_error(task: TaskSpec, message: str, **details: Any) -> AppError:
    return AppError("E_RANGE", f"任务 {task.id} 行区间非法: {message}", {"task_id": task.id, **details})


def _compile_explicit_shards(task: TaskSpec, block: BlockSpec) -> tuple[ShardPlan, ...]:
    assert task.shards is not None
    raw = sorted(task.shards, key=lambda shard: (shard.start, shard.end, shard.resources))
    expected_start = 0
    output: list[ShardPlan] = []
    for index, shard in enumerate(raw):
        if shard.start < 0:
            raise _range_error(task, "start 不能为负数", start=shard.start, end=shard.end, index=index)
        if shard.start >= shard.end:
            raise _range_error(task, "start 必须小于 end", start=shard.start, end=shard.end, index=index)
        if shard.start != expected_start:
            kind = "重叠" if shard.start < expected_start else "不连续"
            raise _range_error(
                task,
                f"区间{kind}，期望 start={expected_start}，实际 start={shard.start}",
                expected_start=expected_start,
                actual_start=shard.start,
                index=index,
            )
        if shard.end > block.row_count:
            raise _range_error(
                task,
                f"分片终点 {shard.end} 超出总行数 {block.row_count}",
                start=shard.start,
                end=shard.end,
                row_count=block.row_count,
                index=index,
            )
        output.append(
            ShardPlan(
                shard_id=f"{task.id}#{index:04d}",
                task_id=task.id,
                index=index,
                start_row=shard.start,
                end_row=shard.end,
                resources=shard.resources,
            )
        )
        expected_start = shard.end
    if expected_start != block.row_count:
        raise _range_error(
            task,
            f"分片未覆盖全部行，最后结束于 {expected_start}，总行数为 {block.row_count}",
            covered_end=expected_start,
            row_count=block.row_count,
        )
    return tuple(output)


def _compile_generated_shards(task: TaskSpec, block: BlockSpec) -> tuple[ShardPlan, ...]:
    assert task.shard_rows is not None
    count = math.ceil(block.row_count / task.shard_rows) if block.row_count else 0
    if count == 0:
        raise _range_error(task, "数据块没有可切分的行", row_count=block.row_count)
    if isinstance(task.resources_per_shard, tuple):
        if len(task.resources_per_shard) != count:
            raise AppError(
                "E_DATA",
                f"任务 {task.id} 的 resources_per_shard 数量 {len(task.resources_per_shard)} 与分片数 {count} 不一致",
                {"task_id": task.id, "expected": count, "actual": len(task.resources_per_shard)},
            )
        resources = task.resources_per_shard
    else:
        resources = tuple(int(task.resources_per_shard) for _ in range(count))
    shards: list[ShardPlan] = []
    for index in range(count):
        start = index * task.shard_rows
        end = min(start + task.shard_rows, block.row_count)
        shards.append(
            ShardPlan(
                shard_id=f"{task.id}#{index:04d}",
                task_id=task.id,
                index=index,
                start_row=start,
                end_row=end,
                resources=resources[index],
            )
        )
    return tuple(shards)


def compile_task(task: TaskSpec, block: BlockSpec) -> TaskPlan:
    missing = [column for column in task.parsed.columns if column not in block.column_map()]
    if missing:
        raise AppError(
            "E_DATA",
            f"任务 {task.id} 引用了不存在的列: {', '.join(missing)}",
            {"task_id": task.id, "missing_columns": missing},
        )
    if task.shards is not None:
        shards = _compile_explicit_shards(task, block)
    else:
        shards = _compile_generated_shards(task, block)
    task_payload = {
        "task_id": task.id,
        "block_id": task.block_id,
        "expression": task.parsed.normalized,
        "expression_structure": task.parsed.structural_digest,
        "aggregation": task.aggregation,
        "quota": task.quota,
        "delay_ms": task.delay_ms,
        "shards": [shard.to_dict() for shard in shards],
    }
    return TaskPlan(
        task_id=task.id,
        block_id=task.block_id,
        expression_text=task.expression,
        expression_normalized=task.parsed.normalized,
        expression_digest=task.parsed.structural_digest,
        aggregation=task.aggregation,
        quota=task.quota,
        delay_ms=task.delay_ms,
        shards=shards,
        plan_digest=digest_value(task_payload),
        parsed_expression=task.parsed,
    )


def compile_workspace(state: WorkspaceState) -> ExecutionPlan:
    state.validate()
    block_map = state.by_block()
    plans = tuple(compile_task(task, block_map[task.block_id]) for task in state.tasks.tasks)
    payload = {
        "format_version": 1,
        "parameter_version": state.parameter_version,
        "content_digest": state.content_digest(),
        "global_budget": state.global_budget,
        "threads": state.threads,
        "tasks": [plan.to_dict() for plan in plans],
    }
    return ExecutionPlan(
        tasks=plans,
        parameter_version=state.parameter_version,
        content_digest=state.content_digest(),
        global_budget=state.global_budget,
        threads=state.threads,
        plan_digest=digest_value(payload),
    )
