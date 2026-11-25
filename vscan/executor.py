from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .attribution import AttributionTotals, batch_attribution
from .canonical import clamp_int64, canonical_json, digest_value, trunc_div_toward_zero
from .codec import DECODED_CELL_BYTES, ColumnData, read_rows
from .planner import BatchPlan
from .spec import AggregateSpec, QuerySpec, ScanSpec
from .store import StoreState, column_path


def evaluate_predicate(
    predicate: dict[str, Any] | None,
    columns: Mapping[str, ColumnData],
    rows: int,
) -> np.ndarray:
    """Evaluate a canonical predicate as one batch-level NumPy boolean mask."""

    if predicate is None:
        return np.ones(rows, dtype="?")
    op = predicate["op"]
    if op == "and":
        result = np.ones(rows, dtype="?")
        for arg in predicate["args"]:
            result &= evaluate_predicate(arg, columns, rows)
        return result
    if op == "or":
        result = np.zeros(rows, dtype="?")
        for arg in predicate["args"]:
            result |= evaluate_predicate(arg, columns, rows)
        return result
    if op == "not":
        return ~evaluate_predicate(predicate["arg"], columns, rows)

    column = columns[predicate["column"]]
    values = column.values
    null_mask = column.null_mask
    valid = ~null_mask
    if op == "is_null":
        return null_mask.copy()
    if op == "is_not_null":
        return valid.copy()
    if op == "in":
        allowed = np.asarray(predicate["values"], dtype="<i8")
        return valid & np.isin(values, allowed)
    target = int(predicate["value"])
    if op == "eq":
        comparison = values == target
    elif op == "ne":
        comparison = values != target
    elif op == "lt":
        comparison = values < target
    elif op == "le":
        comparison = values <= target
    elif op == "gt":
        comparison = values > target
    elif op == "ge":
        comparison = values >= target
    else:
        raise ValueError(f"unsupported predicate operator {op!r}")
    return valid & comparison


@dataclass(frozen=True)
class AggregateState:
    name: str
    op: str
    column: str | None
    count_rows: int
    non_null_count: int
    null_count: int
    sum_raw: int
    min_value: int | None
    max_value: int | None

    @classmethod
    def initial(cls, spec: AggregateSpec) -> "AggregateState":
        return cls(
            name=spec.name,
            op=spec.op,
            column=spec.column,
            count_rows=0,
            non_null_count=0,
            null_count=0,
            sum_raw=0,
            min_value=None,
            max_value=None,
        )

    def merge(self, other: "AggregateState") -> "AggregateState":
        if (self.name, self.op, self.column) != (other.name, other.op, other.column):
            raise ValueError("cannot merge incompatible aggregate states")
        if self.min_value is None:
            minimum = other.min_value
        elif other.min_value is None:
            minimum = self.min_value
        else:
            minimum = min(self.min_value, other.min_value)
        if self.max_value is None:
            maximum = other.max_value
        elif other.max_value is None:
            maximum = self.max_value
        else:
            maximum = max(self.max_value, other.max_value)
        return AggregateState(
            name=self.name,
            op=self.op,
            column=self.column,
            count_rows=self.count_rows + other.count_rows,
            non_null_count=self.non_null_count + other.non_null_count,
            null_count=self.null_count + other.null_count,
            sum_raw=self.sum_raw + other.sum_raw,
            min_value=minimum,
            max_value=maximum,
        )

    def value(self) -> int | None:
        if self.op == "count":
            return self.non_null_count if self.column is not None else self.count_rows
        if self.op == "sum":
            return clamp_int64(self.sum_raw)
        if self.op == "min":
            return self.min_value
        if self.op == "max":
            return self.max_value
        if self.op == "avg":
            if self.non_null_count == 0:
                return 0
            return clamp_int64(trunc_div_toward_zero(self.sum_raw, self.non_null_count))
        raise ValueError(f"unsupported aggregate operator {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "column": self.column,
            "count_rows": self.count_rows,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "sum_raw": self.sum_raw,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "value": self.value(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AggregateState":
        return cls(
            name=str(data["name"]),
            op=str(data["op"]),
            column=None if data.get("column") is None else str(data["column"]),
            count_rows=int(data["count_rows"]),
            non_null_count=int(data["non_null_count"]),
            null_count=int(data["null_count"]),
            sum_raw=int(data["sum_raw"]),
            min_value=None if data.get("min_value") is None else int(data["min_value"]),
            max_value=None if data.get("max_value") is None else int(data["max_value"]),
        )


@dataclass(frozen=True)
class GroupState:
    key: tuple[int | None, ...]
    aggregates: tuple[AggregateState, ...]

    def merge(self, other: "GroupState") -> "GroupState":
        if self.key != other.key:
            raise ValueError("cannot merge different groups")
        other_by_name = {state.name: state for state in other.aggregates}
        merged = tuple(state.merge(other_by_name[state.name]) for state in self.aggregates)
        return GroupState(self.key, merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "aggregates": [state.to_dict() for state in self.aggregates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroupState":
        return cls(
            key=tuple(data["key"]),
            aggregates=tuple(
                AggregateState.from_dict(state) for state in data["aggregates"]
            ),
        )


def _aggregate_group(
    rows_in_group: np.ndarray,
    columns: Mapping[str, ColumnData],
    aggregate_specs: Iterable[AggregateSpec],
) -> tuple[AggregateState, ...]:
    states: list[AggregateState] = []
    for spec in aggregate_specs:
        if spec.column is None:
            states.append(
                AggregateState(
                    name=spec.name,
                    op=spec.op,
                    column=None,
                    count_rows=int(rows_in_group.size),
                    non_null_count=int(rows_in_group.size),
                    null_count=0,
                    sum_raw=0,
                    min_value=None,
                    max_value=None,
                )
            )
            continue
        data = columns[spec.column]
        nulls = data.null_mask[rows_in_group]
        valid_rows = rows_in_group[~nulls]
        values = data.values[valid_rows]
        non_null = int(values.size)
        null_count = int(rows_in_group.size - non_null)
        if spec.op in ("sum", "avg"):
            sum_raw = int(sum(int(value) for value in values))
        else:
            sum_raw = 0
        minimum = int(np.min(values)) if spec.op == "min" and non_null else None
        maximum = int(np.max(values)) if spec.op == "max" and non_null else None
        states.append(
            AggregateState(
                name=spec.name,
                op=spec.op,
                column=spec.column,
                count_rows=int(rows_in_group.size),
                non_null_count=non_null,
                null_count=null_count,
                sum_raw=sum_raw,
                min_value=minimum,
                max_value=maximum,
            )
        )
    return tuple(states)


def _group_selected_rows(
    query: QuerySpec, columns: Mapping[str, ColumnData], selected_count: int
) -> tuple[list[tuple[int | None, ...]], list[np.ndarray]]:
    if not query.group_by:
        return [()], [np.arange(selected_count, dtype=np.int64)]
    if selected_count == 0:
        return [], []
    structured = np.empty(selected_count, dtype=[(f"g{index}", "<i8") for index in range(len(query.group_by))])
    for index, column_name in enumerate(query.group_by):
        data = columns[column_name]
        valid = ~data.null_mask
        ranks = np.zeros(selected_count, dtype="<i8")
        if np.any(valid):
            _, inverse = np.unique(data.values[valid], return_inverse=True)
            ranks[valid] = inverse.astype("<i8") + 1
        structured[f"g{index}"] = ranks
    _, first_indices, inverse = np.unique(
        structured, return_index=True, return_inverse=True
    )
    group_count = int(first_indices.size)
    keys: list[tuple[int | None, ...]] = []
    for first_index in first_indices:
        key: list[int | None] = []
        for column_name in query.group_by:
            data = columns[column_name]
            key.append(None if bool(data.null_mask[first_index]) else int(data.values[first_index]))
        keys.append(tuple(key))
    group_rows = [np.flatnonzero(inverse == group_id) for group_id in range(group_count)]
    return keys, group_rows


@dataclass
class BatchComputation:
    passed_rows: int
    groups: tuple[GroupState, ...]
    aggregate_digest: str
    actual_decoded_bytes: int
    attribution: dict[str, Any]
    decoded_columns: tuple[str, ...]
    predicate_mask_digest: str

    def aggregate_payload(self) -> dict[str, Any]:
        return {
            "passed_rows": self.passed_rows,
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.aggregate_payload(),
            "aggregate_digest": self.aggregate_digest,
            "actual_decoded_bytes": self.actual_decoded_bytes,
            "attribution": self.attribution,
            "decoded_columns": list(self.decoded_columns),
            "predicate_mask_digest": self.predicate_mask_digest,
        }


@dataclass
class BatchResult:
    plan: BatchPlan
    computation: BatchComputation
    fingerprint: str = ""
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    wait_ms: int = 0
    worker_thread: str = ""
    dispatch_count: int = 1

    @property
    def key(self) -> tuple[str, str, int]:
        return self.plan.key

    @property
    def aggregate_digest(self) -> str:
        return self.computation.aggregate_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.plan.query_id,
            "segment_id": self.plan.segment_id,
            "batch_no": self.plan.batch_no,
            "start_row": self.plan.start_row,
            "end_row": self.plan.end_row,
            "row_count": self.plan.rows,
            "decode_columns": list(self.plan.decode_columns),
            "pruned_columns": list(self.plan.pruned_columns),
            "batch_plan_digest": self.plan.plan_digest,
            **self.computation.to_dict(),
            "scheduling": {
                "wait_start_monotonic_ms": self.wait_start_ms,
                "wait_end_monotonic_ms": self.wait_end_ms,
                "wait_ms": self.wait_ms,
                "worker_thread": self.worker_thread,
                "dispatch_count": self.dispatch_count,
            },
        }


def batch_fingerprint(
    spec: ScanSpec,
    store: StoreState,
    data_root: str | Path,
    batch: BatchPlan,
) -> str:
    segment = store.segment_map()[batch.segment_id]
    column_map = segment.column_map()
    row_indices = np.arange(batch.start_row, batch.end_row, dtype=np.int64)
    slice_digests: dict[str, str] = {}
    for column in batch.decode_columns:
        data = read_rows(
            column_path(data_root, batch.segment_id, column), row_indices
        )
        digest = hashlib.sha256()
        digest.update(b"values:")
        digest.update(data.values.tobytes(order="C"))
        digest.update(b"nulls:")
        digest.update(np.packbits(data.null_mask, bitorder="little").tobytes())
        slice_digests[column] = digest.hexdigest()
    return digest_value(
        {
            "batch_plan_digest": batch.plan_digest,
            "start_row": batch.start_row,
            "end_row": batch.end_row,
            "scale": spec.execution.scale,
            "segment_id": batch.segment_id,
            "segment_rows": segment.rows,
            "decode_column_slices": slice_digests,
            "decode_column_encodings": {
                column: column_map[column].encoding for column in batch.decode_columns
            },
        }
    )


def execute_batch_for_query(
    spec: ScanSpec,
    store: StoreState,
    data_root: str | Path,
    query: QuerySpec,
    batch: BatchPlan,
) -> BatchComputation:
    root = Path(data_root)
    predicate_cache: dict[str, ColumnData] = {}
    batch_row_indices = np.arange(batch.start_row, batch.end_row, dtype=np.int64)
    for column_name in batch.predicate_columns:
        predicate_cache[column_name] = read_rows(
            column_path(root, batch.segment_id, column_name), batch_row_indices
        )
    mask = evaluate_predicate(batch.predicate, predicate_cache, batch.rows)
    selected_local = np.flatnonzero(mask).astype(np.int64)
    passed_rows = int(selected_local.size)

    output_data: dict[str, ColumnData] = {}
    for column_name in batch.output_columns:
        if column_name in predicate_cache:
            data = predicate_cache[column_name]
            output_data[column_name] = ColumnData(
                data.values[selected_local], data.null_mask[selected_local]
            )
        else:
            global_indices = selected_local + batch.start_row
            output_data[column_name] = read_rows(
                column_path(root, batch.segment_id, column_name), global_indices
            )

    keys, group_rows = _group_selected_rows(query, output_data, passed_rows)
    groups = tuple(
        GroupState(
            key=key,
            aggregates=_aggregate_group(row_indices, output_data, query.aggregates),
        )
        for key, row_indices in zip(keys, group_rows, strict=True)
    )
    groups = tuple(sorted(groups, key=lambda group: canonical_group_key(group.key)))
    aggregate_payload = {
        "passed_rows": passed_rows,
        "groups": [group.to_dict() for group in groups],
    }
    output_only_count = len(set(batch.output_columns) - set(batch.predicate_columns))
    actual_decoded_bytes = (
        batch.rows * len(batch.predicate_columns) * DECODED_CELL_BYTES
        + passed_rows * output_only_count * DECODED_CELL_BYTES
    )
    attribution = batch_attribution(
        rows=batch.rows,
        all_column_count=len(batch.all_columns),
        decode_column_count=len(batch.decode_columns),
        predicate_column_count=len(batch.predicate_columns),
        passed_rows=passed_rows,
        predicate_leaf_count=batch.predicate_leaf_count,
        decoded_cell_bytes=DECODED_CELL_BYTES,
        tolerance_ppm=spec.execution.attribution_tolerance_ppm,
    )
    return BatchComputation(
        passed_rows=passed_rows,
        groups=groups,
        aggregate_digest=digest_value(aggregate_payload),
        actual_decoded_bytes=actual_decoded_bytes,
        attribution=attribution,
        decoded_columns=batch.decode_columns,
        predicate_mask_digest=digest_value(np.packbits(mask, bitorder="little").tobytes().hex()),
    )


def canonical_group_key(key: tuple[int | None, ...]) -> str:
    return canonical_json(list(key))


@dataclass
class QueryResult:
    query_id: str
    group_by: tuple[str, ...]
    groups: tuple[GroupState, ...]
    aggregate_digest: str
    batches: tuple[BatchResult, ...]
    attribution: AttributionTotals
    tolerance_ppm: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "group_by": list(self.group_by),
            "groups": [group.to_dict() for group in self.groups],
            "aggregate_digest": self.aggregate_digest,
            "batch_aggregates": [batch.to_dict() for batch in self.batches],
            "attribution": self.attribution.to_dict(self.tolerance_ppm),
        }


def merge_query_result(
    spec: ScanSpec,
    query: QuerySpec,
    batches: Iterable[BatchResult],
) -> QueryResult:
    ordered = tuple(sorted(batches, key=lambda batch: (batch.plan.segment_id, batch.plan.batch_no)))
    merged: dict[str, GroupState] = {}
    totals = AttributionTotals()
    for batch in ordered:
        totals.add(batch.computation.attribution)
        for group in batch.computation.groups:
            key = canonical_group_key(group.key)
            if key in merged:
                merged[key] = merged[key].merge(group)
            else:
                merged[key] = group
    if not query.group_by and not merged:
        merged[canonical_group_key(())] = GroupState(
            key=(),
            aggregates=tuple(AggregateState.initial(agg) for agg in query.aggregates),
        )
    groups = tuple(merged[key] for key in sorted(merged))
    payload = {
        "query_id": query.id,
        "group_by": list(query.group_by),
        "groups": [group.to_dict() for group in groups],
    }
    return QueryResult(
        query_id=query.id,
        group_by=query.group_by,
        groups=groups,
        aggregate_digest=digest_value(payload),
        batches=ordered,
        attribution=totals,
        tolerance_ppm=spec.execution.attribution_tolerance_ppm,
    )
