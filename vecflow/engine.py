"""NumPy vector evaluation and deterministic, order-aware task aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json_bytes, digest_bytes, digest_value
from .errors import AppError
from .expression import Binary, ColumnRef, Literal, Ternary, Unary
from .model import BlockSpec, WorkspaceState
from .planner import ShardPlan, TaskPlan

try:  # pragma: no cover - the error branch is environment diagnostics
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None


def _numpy() -> Any:
    if np is None:
        raise AppError("E_DATA", f"NumPy 不可用，请执行 pip install numpy: {_NUMPY_IMPORT_ERROR}")
    return np


@dataclass(frozen=True)
class ColumnTensor:
    values: Any
    null_mask: Any
    dtype: str


@dataclass(frozen=True)
class VectorValue:
    values: Any
    null_mask: Any
    divzero_mask: Any
    dtype: str


@dataclass(frozen=True)
class ShardResult:
    task_id: str
    shard_id: str
    start_row: int
    end_row: int
    outcome: str
    value_type: str
    values: tuple[Any, ...] | None
    null_mask: tuple[int, ...] | None
    division_by_zero_mask: tuple[int, ...] | None
    result_digest: str | None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "shard_id": self.shard_id,
            "start": self.start_row,
            "end": self.end_row,
            "outcome": self.outcome,
            "value_type": self.value_type,
            "values": list(self.values) if self.values is not None else None,
            "null_mask": list(self.null_mask) if self.null_mask is not None else None,
            "division_by_zero_mask": list(self.division_by_zero_mask) if self.division_by_zero_mask is not None else None,
            "result_digest": self.result_digest,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    expression: str
    aggregation: str
    payload: dict[str, Any]
    canonical_bytes: bytes
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "expression": self.expression,
            "aggregation": self.aggregation,
            "payload": self.payload,
            "result_digest": self.result_digest,
        }


def _dtype_name(dtype: Any) -> str:
    numpy = _numpy()
    dtype = numpy.dtype(dtype)
    if numpy.issubdtype(dtype, numpy.bool_):
        return "boolean"
    if numpy.issubdtype(dtype, numpy.integer):
        return "integer"
    return "float"


def _zero_for(dtype: str) -> Any:
    return False if dtype == "boolean" else (0 if dtype == "integer" else 0.0)


def _masked_zero(values: Any, null_mask: Any, dtype: str) -> Any:
    numpy = _numpy()
    zero = _zero_for(dtype)
    return numpy.where(null_mask, zero, values).astype(_numpy_dtype(dtype))


def _numpy_dtype(dtype: str) -> Any:
    numpy = _numpy()
    if dtype == "boolean":
        return numpy.bool_
    if dtype == "integer":
        return numpy.int64
    return numpy.float64


def _promote(left: str, right: str) -> str:
    if left == "float" or right == "float":
        return "float"
    if left == "integer" or right == "integer":
        return "integer"
    return "boolean"


class BlockTensorBank:
    def __init__(self, state: WorkspaceState) -> None:
        self._blocks: dict[str, dict[str, ColumnTensor]] = {}
        self._row_counts: dict[str, int] = {}
        for block in state.blocks.blocks:
            self._blocks[block.id] = self._build_block(block)
            self._row_counts[block.id] = block.row_count

    def _build_block(self, block: BlockSpec) -> dict[str, ColumnTensor]:
        numpy = _numpy()
        columns: dict[str, ColumnTensor] = {}
        for column in block.columns:
            null_mask = numpy.asarray(column.null_bitmap, dtype=numpy.bool_)
            dtype = _numpy_dtype(column.type)
            if column.type == "boolean":
                raw = [False if value is None else bool(value) for value in column.values]
            elif column.type == "integer":
                raw = [0 if value is None else int(value) for value in column.values]
            else:
                raw = [0.0 if value is None else float(value) for value in column.values]
            values = numpy.asarray(raw, dtype=dtype)
            values = _masked_zero(values, null_mask, column.type)
            columns[column.name] = ColumnTensor(values=values, null_mask=null_mask, dtype=column.type)
        return columns

    def slice(self, block_id: str, start: int, end: int) -> dict[str, ColumnTensor]:
        return {
            name: ColumnTensor(
                values=tensor.values[start:end],
                null_mask=tensor.null_mask[start:end],
                dtype=tensor.dtype,
            )
            for name, tensor in self._blocks[block_id].items()
        }


def _evaluate(node: Any, columns: dict[str, ColumnTensor], row_count: int) -> VectorValue:
    numpy = _numpy()
    if isinstance(node, Literal):
        if node.literal_kind == "null":
            values = numpy.zeros(row_count, dtype=numpy.float64)
            return VectorValue(values, numpy.ones(row_count, dtype=numpy.bool_), numpy.zeros(row_count, dtype=numpy.bool_), "float")
        if node.literal_kind == "float":
            dtype = "float"
            raw = float(node.value)  # type: ignore[arg-type]
        elif node.literal_kind == "integer":
            dtype = "integer"
            raw = int(node.value)  # type: ignore[arg-type]
        else:
            dtype = "boolean"
            raw = bool(node.value)
        values = numpy.full(row_count, raw, dtype=_numpy_dtype(dtype))
        return VectorValue(values, numpy.zeros(row_count, dtype=numpy.bool_), numpy.zeros(row_count, dtype=numpy.bool_), dtype)

    if isinstance(node, ColumnRef):
        tensor = columns[node.name]
        return VectorValue(
            values=tensor.values.copy(),
            null_mask=tensor.null_mask.copy(),
            divzero_mask=numpy.zeros(row_count, dtype=numpy.bool_),
            dtype=tensor.dtype,
        )

    if isinstance(node, Unary):
        operand = _evaluate(node.operand, columns, row_count)
        if node.op == "not":
            values = numpy.logical_not(operand.values.astype(numpy.bool_))
            dtype = "boolean"
        else:
            values = operand.values.astype(numpy.int64 if operand.dtype != "float" else numpy.float64)
            dtype = operand.dtype if operand.dtype != "boolean" else "integer"
            if node.op == "-":
                values = -values
        null = operand.null_mask.copy()
        return VectorValue(_masked_zero(values, null, dtype), null, operand.divzero_mask.copy(), dtype)

    if isinstance(node, Binary):
        left = _evaluate(node.left, columns, row_count)
        right = _evaluate(node.right, columns, row_count)
        null_pre = numpy.logical_or(left.null_mask, right.null_mask)
        divzero = numpy.logical_or(left.divzero_mask, right.divzero_mask)
        invalid = numpy.zeros(row_count, dtype=numpy.bool_)

        if node.op in {"and", "or"}:
            dtype = "boolean"
            left_values = left.values.astype(numpy.bool_)
            right_values = right.values.astype(numpy.bool_)
            values = numpy.logical_and(left_values, right_values) if node.op == "and" else numpy.logical_or(left_values, right_values)
        elif node.op in {"<", "<=", ">", ">=", "==", "!="}:
            dtype = "boolean"
            if node.op == "<":
                values = numpy.less(left.values, right.values)
            elif node.op == "<=":
                values = numpy.less_equal(left.values, right.values)
            elif node.op == ">":
                values = numpy.greater(left.values, right.values)
            elif node.op == ">=":
                values = numpy.greater_equal(left.values, right.values)
            elif node.op == "==":
                values = numpy.equal(left.values, right.values)
            else:
                values = numpy.not_equal(left.values, right.values)
        elif node.op in {"/", "%"}:
            denominator_zero = numpy.logical_and(numpy.equal(right.values, 0), numpy.logical_not(right.null_mask))
            invalid = denominator_zero.copy()
            valid = numpy.logical_not(numpy.logical_or(null_pre, denominator_zero))
            if node.op == "/":
                dtype = "float"
                left_values = left.values.astype(numpy.float64)
                right_values = right.values.astype(numpy.float64)
                values = numpy.zeros(row_count, dtype=numpy.float64)
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    numpy.divide(left_values, right_values, out=values, where=valid)
            else:
                dtype = "float" if left.dtype == "float" or right.dtype == "float" else "integer"
                dtype_np = _numpy_dtype(dtype)
                left_values = left.values.astype(dtype_np)
                right_values = right.values.astype(dtype_np)
                values = numpy.zeros(row_count, dtype=dtype_np)
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    if dtype == "float":
                        numpy.remainder(left_values, right_values, out=values, where=valid)
                    else:
                        numpy.remainder(left_values, right_values, out=values, where=valid)
        else:
            dtype = _promote(left.dtype, right.dtype)
            if dtype == "boolean":
                dtype = "integer"
            dtype_np = _numpy_dtype(dtype)
            left_values = left.values.astype(dtype_np)
            right_values = right.values.astype(dtype_np)
            if node.op == "+":
                values = numpy.add(left_values, right_values)
            elif node.op == "-":
                values = numpy.subtract(left_values, right_values)
            elif node.op == "*":
                values = numpy.multiply(left_values, right_values)
            else:
                raise AppError("E_DATA", f"未知二元算子 {node.op}")

        null = numpy.logical_or(null_pre, invalid)
        values = _masked_zero(values, null, dtype)
        return VectorValue(values, null, numpy.logical_or(divzero, invalid), dtype)

    if isinstance(node, Ternary):
        condition = _evaluate(node.condition, columns, row_count)
        when_true = _evaluate(node.when_true, columns, row_count)
        when_false = _evaluate(node.when_false, columns, row_count)
        dtype = _promote(when_true.dtype, when_false.dtype)
        dtype_np = _numpy_dtype(dtype)
        choose_true = numpy.logical_and(condition.values.astype(numpy.bool_), numpy.logical_not(condition.null_mask))
        true_values = when_true.values.astype(dtype_np)
        false_values = when_false.values.astype(dtype_np)
        values = numpy.where(choose_true, true_values, false_values).astype(dtype_np)
        null = numpy.logical_or(
            condition.null_mask,
            numpy.where(choose_true, when_true.null_mask, when_false.null_mask),
        )
        divzero = numpy.where(choose_true, when_true.divzero_mask, when_false.divzero_mask)
        values = _masked_zero(values, null, dtype)
        return VectorValue(values, null, divzero, dtype)

    raise AppError("E_DATA", f"未知表达式节点 {type(node).__name__}")


def _clean_scalar(value: Any, dtype: str) -> Any:
    if value is None:
        return None
    if dtype == "boolean":
        return bool(value)
    if dtype == "integer":
        return int(value)
    numeric = float(value)
    if numeric == 0:
        return 0.0
    return numeric


def _clean_vector(values: Any, null_mask: Any) -> tuple[Any, ...]:
    output: list[Any] = []
    for raw, is_null in zip(values.tolist(), null_mask.tolist()):
        if is_null:
            output.append(None)
        elif isinstance(raw, bool):
            output.append(bool(raw))
        elif isinstance(raw, int):
            output.append(int(raw))
        else:
            numeric = float(raw)
            output.append(0.0 if numeric == 0 else numeric)
    return tuple(output)


def evaluate_shard(plan: TaskPlan, shard: ShardPlan, bank: BlockTensorBank) -> ShardResult:
    row_count = shard.row_count
    columns = bank.slice(plan.block_id, shard.start_row, shard.end_row)
    try:
        vector = _evaluate(plan.parsed_expression.root, columns, row_count)
        values = _clean_vector(vector.values, vector.null_mask)
        null_mask = tuple(int(item) for item in vector.null_mask.tolist())
        divzero_mask = tuple(int(item) for item in vector.divzero_mask.tolist())
        payload = {
            "task_id": plan.task_id,
            "shard_id": shard.shard_id,
            "start": shard.start_row,
            "end": shard.end_row,
            "value_type": vector.dtype,
            "values": list(values),
            "null_mask": list(null_mask),
            "division_by_zero_mask": list(divzero_mask),
        }
        return ShardResult(
            task_id=plan.task_id,
            shard_id=shard.shard_id,
            start_row=shard.start_row,
            end_row=shard.end_row,
            outcome="success",
            value_type=vector.dtype,
            values=values,
            null_mask=null_mask,
            division_by_zero_mask=divzero_mask,
            result_digest=digest_value(payload),
        )
    except AppError as exc:
        return ShardResult(
            task_id=plan.task_id,
            shard_id=shard.shard_id,
            start_row=shard.start_row,
            end_row=shard.end_row,
            outcome="error",
            value_type="unknown",
            values=None,
            null_mask=None,
            division_by_zero_mask=None,
            result_digest=None,
            error_code=exc.code,
            error_message=exc.message,
        )
    except Exception as exc:  # defensive shard isolation
        return ShardResult(
            task_id=plan.task_id,
            shard_id=shard.shard_id,
            start_row=shard.start_row,
            end_row=shard.end_row,
            outcome="error",
            value_type="unknown",
            values=None,
            null_mask=None,
            division_by_zero_mask=None,
            result_digest=None,
            error_code="E_INTERNAL",
            error_message=f"{exc.__class__.__name__}: {exc}",
        )


def _aggregate_scalar(mode: str, dtype: str, values: list[Any], null_mask: list[bool]) -> tuple[Any, bool]:
    non_null = [value for value, is_null in zip(values, null_mask) if not is_null]
    if mode == "count":
        return len(non_null), False
    if mode in {"any", "all"}:
        if not non_null:
            return None, True
        if mode == "any":
            return any(bool(value) for value in non_null), False
        return all(bool(value) for value in non_null), False
    if mode in {"first", "last"}:
        if not values:
            return None, True
        index = 0 if mode == "first" else len(values) - 1
        if null_mask[index]:
            return None, True
        return _clean_scalar(values[index], dtype), False
    if not non_null:
        return None, True
    if mode == "sum":
        if dtype == "float":
            return math.fsum(float(value) for value in non_null), False
        if dtype == "boolean":
            return sum(1 for value in non_null if bool(value)), False
        return sum(int(value) for value in non_null), False
    if mode == "mean":
        return math.fsum(float(value) for value in non_null) / len(non_null), False
    if mode == "min":
        return _clean_scalar(min(non_null), dtype), False
    if mode == "max":
        return _clean_scalar(max(non_null), dtype), False
    raise AppError("E_DATA", f"未知聚合口径 {mode}")


def aggregate_task(plan: TaskPlan, shard_results: list[ShardResult], total_rows: int) -> TaskResult:
    ordered = sorted(shard_results, key=lambda result: (result.start_row, result.end_row))
    if any(result.outcome != "success" for result in ordered):
        raise AppError("E_DATA", f"任务 {plan.task_id} 存在失败分片，不能合流")
    expected_start = 0
    for result in ordered:
        if result.start_row != expected_start:
            raise AppError(
                "E_RANGE",
                f"任务 {plan.task_id} 合流分片不连续: 期望 {expected_start}，实际 {result.start_row}",
                {"task_id": plan.task_id, "shard_id": result.shard_id},
            )
        expected_start = result.end_row
    if expected_start != total_rows:
        raise AppError(
            "E_RANGE",
            f"任务 {plan.task_id} 合流未覆盖全部行: {expected_start}/{total_rows}",
            {"task_id": plan.task_id, "covered": expected_start, "row_count": total_rows},
        )

    values: list[Any] = []
    null_mask: list[bool] = []
    divzero_mask: list[bool] = []
    dtype = ordered[0].value_type if ordered else "float"
    for result in ordered:
        assert result.values is not None and result.null_mask is not None and result.division_by_zero_mask is not None
        values.extend(result.values)
        null_mask.extend(bool(item) for item in result.null_mask)
        divzero_mask.extend(bool(item) for item in result.division_by_zero_mask)

    payload: dict[str, Any] = {
        "task_id": plan.task_id,
        "expression": plan.expression_normalized,
        "aggregation": plan.aggregation,
        "row_count": total_rows,
        "null_count": sum(1 for item in null_mask if item),
        "division_by_zero_count": sum(1 for item in divzero_mask if item),
        "shard_ids": [result.shard_id for result in ordered],
    }
    if plan.aggregation == "concat":
        payload["payload"] = {
            "value_type": dtype,
            "values": values,
            "null_mask": [int(item) for item in null_mask],
            "division_by_zero_mask": [int(item) for item in divzero_mask],
        }
    else:
        scalar, is_null = _aggregate_scalar(plan.aggregation, dtype, values, null_mask)
        payload["payload"] = {
            "value_type": dtype,
            "value": scalar,
            "null": is_null,
            "non_null_count": sum(1 for item in null_mask if not item),
        }
    canonical = canonical_json_bytes(payload)
    return TaskResult(
        task_id=plan.task_id,
        expression=plan.expression_normalized,
        aggregation=plan.aggregation,
        payload=payload,
        canonical_bytes=canonical,
        result_digest=digest_bytes(canonical),
    )
