from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import itertools
import json
import time
from typing import Any, Callable, Iterable, Mapping

from .cost import PlanEstimate
from .errors import LabError
from .model import ColumnRef, FilterPredicate, JoinPredicate, Plan, Query, Table, Workload
from .scheduler import ConstrainedScheduler, ScheduleResult, TaskRunResult, TaskSpec
from .stats import evaluate_predicate


@dataclass(frozen=True)
class Relation:
    schema: tuple[tuple[str, str], ...]
    rows: tuple[tuple[Any, ...], ...]
    width_bytes: int

    def index(self, table: str, column: str) -> int:
        try:
            return self.schema.index((table, column))
        except ValueError as exc:
            raise LabError(
                "E_EXEC",
                f"关系缺少列 {table}.{column}",
                table=table,
                column=column,
                field="relation.schema",
            ) from exc

    @property
    def peak_memory_bytes(self) -> int:
        return max(0, len(self.rows) * self.width_bytes)


@dataclass(frozen=True)
class QueryResult:
    raw_schema: tuple[str, ...]
    raw_rows: tuple[tuple[Any, ...], ...]
    group_fields: tuple[str, ...]
    group_rows: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    group_key_fields: tuple[str, ...]
    group_key_positions: tuple[int, ...]
    raw_bytes: bytes
    grouped_bytes: bytes
    raw_digest: str
    grouped_digest: str


@dataclass(frozen=True)
class ExecutionRun:
    plan: Plan
    result: QueryResult
    relation: Relation
    schedule: ScheduleResult
    elapsed_ms: float

    def summary(self, scale: int) -> dict[str, Any]:
        return {
            "candidate_id": self.plan.candidate_id,
            "plan_signature": self.plan.signature,
            "origin": self.plan.origin,
            "elapsed_ms": self.elapsed_ms,
            "elapsed_ns": int(round(self.elapsed_ms * 1_000_000)),
            "raw_row_count": len(self.result.raw_rows),
            "raw_digest": self.result.raw_digest,
            "grouped_digest": self.result.grouped_digest,
            "group_row_count": len(self.result.group_rows),
            "raw_bytes_length": len(self.result.raw_bytes),
            "grouped_bytes_length": len(self.result.grouped_bytes),
            "result_preview": _preview(self.result),
            "scheduler": {
                "constraints": self.schedule.constraints,
                "tasks": self.schedule.trace_dicts(),
            },
        }


def execute_plan(
    workload: Workload,
    plan: Plan,
    estimate: PlanEstimate,
    *,
    fault_mode: str | None = None,
) -> ExecutionRun:
    tables = workload.table_map()
    tuning = workload.tuning
    tasks: list[TaskSpec] = []
    scan_ids: dict[str, str] = {}
    for table_name in plan.table_order:
        table = tables[table_name]
        task_id = f"scan:{plan.candidate_id}:{table_name}"
        scan_ids[table_name] = task_id
        tasks.append(
            TaskSpec(
                task_id=task_id,
                kind="scan",
                operator_id=f"scan:{table_name}",
                tables=(table_name,),
                memory_reservation_bytes=_scan_reservation(table, tuning),
                dependencies=(),
                runner=_scan_runner(workload, table_name),
            )
        )

    previous_task = scan_ids[plan.table_order[0]]
    previous_relation: Relation | None = None
    joined = [plan.table_order[0]]
    for index, step in enumerate(plan.steps):
        right_scan = scan_ids[step.right_table]
        task_id = f"join:{plan.candidate_id}:{index}:{step.right_table}"
        left_task = previous_task
        reserved_tables = tuple(joined + [step.right_table])
        left_tables = tuple(sorted(joined))
        tasks.append(
            TaskSpec(
                task_id=task_id,
                kind="join",
                operator_id=f"join:{index}:{step.right_table}",
                tables=reserved_tables,
                memory_reservation_bytes=_join_reservation(
                    reserved_tables, tables, tuning
                ),
                dependencies=(left_task, right_scan),
                runner=_join_runner(
                    left_task,
                    right_scan,
                    left_tables,
                    step.right_table,
                    step.predicates,
                    step.algorithm,
                ),
            )
        )
        previous_task = task_id
        joined.append(step.right_table)

    scheduler = ConstrainedScheduler(
        global_concurrency=tuning.global_concurrency,
        per_table_quota=tuning.per_table_scan_quota,
        memory_budget_bytes=tuning.memory_budget_bytes,
    )
    started = time.monotonic_ns()
    schedule = scheduler.run(tasks)
    over_reserved = [
        trace.task_id
        for trace in schedule.traces
        if trace.peak_memory_bytes > trace.memory_reservation_bytes
    ]
    if over_reserved:
        raise LabError(
            "E_RESOURCE",
            "任务实测峰值内存超过调度预留，拒绝绕过内存预算",
            field="tasks.peak_memory_bytes",
            context={"tasks": over_reserved},
        )
    elapsed_ms = (time.monotonic_ns() - started) / 1_000_000.0
    final_relation = schedule.values[previous_task]
    if not isinstance(final_relation, Relation):
        raise LabError("E_EXEC", "最终任务未产生关系", field="execution.final_relation")
    if fault_mode == "drop_last_row" and final_relation.rows:
        final_relation = Relation(
            final_relation.schema,
            final_relation.rows[:-1],
            final_relation.width_bytes,
        )
    result = project_and_aggregate(workload.query, final_relation)
    return ExecutionRun(
        plan=plan,
        result=result,
        relation=final_relation,
        schedule=schedule,
        elapsed_ms=elapsed_ms,
    )


def _scan_runner(workload: Workload, table_name: str) -> Callable[[dict[str, Any]], TaskRunResult]:
    table = workload.table_map()[table_name]
    predicates = [
        predicate
        for predicate in workload.query.filters
        if predicate.column.table == table_name
    ]
    indexes = [table.column_index(predicate.column.column) for predicate in predicates]

    def run(_: dict[str, Any]) -> TaskRunResult:
        rows = []
        for row in table.rows:
            if all(
                evaluate_predicate(predicate, row, index)
                for predicate, index in zip(predicates, indexes)
            ):
                rows.append(row)
        schema = tuple((table.name, column.name) for column in table.columns)
        width = _schema_width(schema, workload)
        relation = Relation(schema, tuple(rows), width)
        return TaskRunResult(relation, len(rows), relation.peak_memory_bytes)

    return run


def _join_runner(
    left_task: str,
    right_task: str,
    left_tables: tuple[str, ...],
    right_table: str,
    predicates: tuple[JoinPredicate, ...],
    algorithm: str,
) -> Callable[[dict[str, Any]], TaskRunResult]:
    def run(dependencies: dict[str, Any]) -> TaskRunResult:
        left: Relation = dependencies[left_task]
        right: Relation = dependencies[right_task]
        if algorithm == "HASH":
            output = _hash_join(left, right, predicates, right_table)
        elif algorithm == "MERGE":
            output = _merge_join(left, right, predicates, right_table)
        elif algorithm == "NESTED_LOOP":
            output = _nested_loop_join(left, right, predicates, right_table)
        else:
            raise LabError("E_EXEC", f"未知连接算法 {algorithm}", field="plan.steps.algorithm")
        peak = left.peak_memory_bytes + right.peak_memory_bytes + output.peak_memory_bytes
        return TaskRunResult(output, len(output.rows), peak)

    return run


def _join_positions(
    left: Relation,
    right: Relation,
    predicates: tuple[JoinPredicate, ...],
    right_table: str,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for predicate in predicates:
        if predicate.left.table == right_table:
            new_ref, old_ref = predicate.left, predicate.right
        elif predicate.right.table == right_table:
            new_ref, old_ref = predicate.right, predicate.left
        else:
            raise LabError(
                "E_EXEC",
                "连接谓词没有引用新连接表",
                table=right_table,
                field="plan.steps.predicates",
            )
        positions.append(
            (
                left.index(old_ref.table, old_ref.column),
                right.index(new_ref.table, new_ref.column),
            )
        )
    return positions


def _key(row: tuple[Any, ...], positions: Iterable[tuple[int, int]], side: int) -> tuple[Any, ...] | None:
    values = tuple(row[position[side]] for position in positions)
    if any(value is None for value in values):
        return None
    return values


def _combined_schema(left: Relation, right: Relation) -> tuple[tuple[str, str], ...]:
    return left.schema + right.schema


def _hash_join(
    left: Relation,
    right: Relation,
    predicates: tuple[JoinPredicate, ...],
    right_table: str,
) -> Relation:
    positions = _join_positions(left, right, predicates, right_table)
    buckets: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for row in right.rows:
        key = _key(row, positions, 1)
        if key is not None:
            buckets.setdefault(key, []).append(row)
    output: list[tuple[Any, ...]] = []
    for left_row in left.rows:
        key = _key(left_row, positions, 0)
        if key is None:
            continue
        for right_row in buckets.get(key, ()):
            output.append(left_row + right_row)
    return Relation(_combined_schema(left, right), tuple(output), left.width_bytes + right.width_bytes)


def _merge_join(
    left: Relation,
    right: Relation,
    predicates: tuple[JoinPredicate, ...],
    right_table: str,
) -> Relation:
    positions = _join_positions(left, right, predicates, right_table)
    left_pairs = []
    for row in left.rows:
        key = _key(row, positions, 0)
        if key is not None:
            left_pairs.append((key, row))
    right_pairs = []
    for row in right.rows:
        key = _key(row, positions, 1)
        if key is not None:
            right_pairs.append((key, row))
    left_pairs.sort(key=lambda item: item[0])
    right_pairs.sort(key=lambda item: item[0])
    output: list[tuple[Any, ...]] = []
    i = 0
    j = 0
    while i < len(left_pairs) and j < len(right_pairs):
        left_key = left_pairs[i][0]
        right_key = right_pairs[j][0]
        if left_key < right_key:
            i += 1
            continue
        if left_key > right_key:
            j += 1
            continue
        i_end = i + 1
        while i_end < len(left_pairs) and left_pairs[i_end][0] == left_key:
            i_end += 1
        j_end = j + 1
        while j_end < len(right_pairs) and right_pairs[j_end][0] == right_key:
            j_end += 1
        for li in range(i, i_end):
            for rj in range(j, j_end):
                output.append(left_pairs[li][1] + right_pairs[rj][1])
        i = i_end
        j = j_end
    return Relation(_combined_schema(left, right), tuple(output), left.width_bytes + right.width_bytes)


def _nested_loop_join(
    left: Relation,
    right: Relation,
    predicates: tuple[JoinPredicate, ...],
    right_table: str,
) -> Relation:
    positions = _join_positions(left, right, predicates, right_table)
    output: list[tuple[Any, ...]] = []
    for left_row in left.rows:
        left_key = _key(left_row, positions, 0)
        if left_key is None:
            continue
        for right_row in right.rows:
            if _key(right_row, positions, 1) == left_key:
                output.append(left_row + right_row)
    return Relation(_combined_schema(left, right), tuple(output), left.width_bytes + right.width_bytes)


def project_and_aggregate(query: Query, relation: Relation) -> QueryResult:
    raw_order = sorted(range(len(relation.schema)), key=lambda index: relation.schema[index])
    raw_schema = tuple(f"{relation.schema[index][0]}.{relation.schema[index][1]}" for index in raw_order)
    canonical_rows = tuple(
        tuple(row[index] for index in raw_order) for row in relation.rows
    )
    raw_sorted = tuple(
        sorted(canonical_rows, key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    )
    raw_doc = {"schema": raw_schema, "rows": raw_sorted}
    raw_bytes = _canonical_bytes(raw_doc)

    group_positions = [relation.index(ref.table, ref.column) for ref in query.group_by]
    original_to_canonical = {original: canonical for canonical, original in enumerate(raw_order)}
    canonical_group_positions = tuple(
        original_to_canonical[position] for position in group_positions
    )
    group_key_fields = tuple(raw_schema[position] for position in canonical_group_positions)
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not query.group_by:
        groups[()] = _initial_group()
    for row in relation.rows:
        group_key = tuple(row[position] for position in group_positions)
        state = groups.setdefault(group_key, _initial_group())
        for index, item in enumerate(query.select):
            if item.aggregate is None:
                continue
            if item.count_star:
                state["count_star"][index] += 1
                continue
            assert item.column is not None
            value = row[relation.index(item.column.table, item.column.column)]
            aggregate = item.aggregate
            if aggregate == "COUNT":
                if value is not None:
                    state["count"][index] += 1
            elif aggregate == "SUM":
                if value is not None:
                    state["sum"][index] += Decimal(str(value))
                    state["seen"][index] += 1
                    state["float_seen"][index] = state["float_seen"][index] or isinstance(value, float)
            elif aggregate == "AVG":
                if value is not None:
                    state["sum"][index] += Decimal(str(value))
                    state["seen"][index] += 1
                    state["float_seen"][index] = state["float_seen"][index] or isinstance(value, float)
            elif aggregate == "MIN":
                if value is not None and (
                    state["extreme"][index] is None
                    or (aggregate == "MIN" and value < state["extreme"][index])  # type: ignore[operator]
                    or (aggregate == "MAX" and value > state["extreme"][index])  # type: ignore[operator]
                ):
                    state["extreme"][index] = value
            elif aggregate == "MAX":
                if value is not None and (
                    state["extreme"][index] is None
                    or (aggregate == "MAX" and value > state["extreme"][index])  # type: ignore[operator]
                ):
                    state["extreme"][index] = value

    group_results: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
    for group_key, state in groups.items():
        output: list[Any] = []
        for index, item in enumerate(query.select):
            if item.aggregate is None:
                assert item.column is not None
                output.append(group_key[group_positions.index(relation.index(item.column.table, item.column.column))])
            elif item.count_star:
                output.append(state["count_star"][index])
            elif item.aggregate == "COUNT":
                output.append(state["count"][index])
            elif item.aggregate == "SUM":
                if not state["seen"][index]:
                    output.append(None)
                else:
                    total = state["sum"][index]
                    if state["float_seen"][index]:
                        output.append(float(total))
                    elif total == total.to_integral_value():
                        output.append(int(total))
                    else:
                        output.append(float(total))
            elif item.aggregate == "AVG":
                count = state["seen"][index]
                output.append(float(state["sum"][index] / count) if count else None)
            else:
                output.append(state["extreme"][index])
        group_results.append((group_key, tuple(output)))
    group_results.sort(
        key=lambda item: json.dumps(item[0], ensure_ascii=False, separators=(",", ":"))
    )
    group_fields = tuple(item.alias for item in query.select)
    grouped_doc = {
        "fields": group_fields,
        "rows": [
            {
                "group_key": list(key),
                "values": list(values),
            }
            for key, values in group_results
        ],
    }
    grouped_bytes = _canonical_bytes(grouped_doc)
    return QueryResult(
        raw_schema=raw_schema,
        raw_rows=raw_sorted,
        group_fields=group_fields,
        group_rows=tuple(group_results),
        group_key_fields=group_key_fields,
        group_key_positions=canonical_group_positions,
        raw_bytes=raw_bytes,
        grouped_bytes=grouped_bytes,
        raw_digest=hashlib.sha256(raw_bytes).hexdigest(),
        grouped_digest=hashlib.sha256(grouped_bytes).hexdigest(),
    )


def _initial_group() -> dict[str, dict[int, Any]]:
    from collections import defaultdict

    return {
        "count_star": defaultdict(int),
        "count": defaultdict(int),
        "sum": defaultdict(int),
        "seen": defaultdict(int),
        "float_seen": defaultdict(bool),
        "extreme": defaultdict(lambda: None),
    }


def compare_execution_results(
    best: ExecutionRun,
    baseline: ExecutionRun,
) -> dict[str, Any]:
    best_result = best.result
    baseline_result = baseline.result
    raw_equal = best_result.raw_bytes == baseline_result.raw_bytes
    grouped_equal = best_result.grouped_bytes == baseline_result.grouped_bytes
    if raw_equal and grouped_equal:
        return {
            "consistent": True,
            "raw_equal": True,
            "grouped_equal": True,
            "best_candidate_id": best.plan.candidate_id,
            "baseline_candidate_id": baseline.plan.candidate_id,
        }

    difference = _first_difference(best_result, baseline_result)
    raise LabError(
        "E_DIVERGE",
        "最优计划与未优化顺序的真实执行结果不一致",
        candidate_id=best.plan.candidate_id,
        field=difference["field"],
        expected=difference.get("expected"),
        actual=difference.get("actual"),
        context={
            "group_key": difference.get("group_key"),
            "best_candidate_id": best.plan.candidate_id,
            "baseline_candidate_id": baseline.plan.candidate_id,
            "raw_equal": raw_equal,
            "grouped_equal": grouped_equal,
        },
    )


def _first_difference(best: QueryResult, baseline: QueryResult) -> dict[str, Any]:
    best_map = dict(best.group_rows)
    baseline_map = dict(baseline.group_rows)
    all_keys = sorted(
        set(best_map) | set(baseline_map),
        key=lambda key: json.dumps(key, ensure_ascii=False, separators=(",", ":")),
    )
    for key in all_keys:
        if key not in best_map or key not in baseline_map:
            return {
                "group_key": list(key),
                "field": "row_presence",
                "expected": list(best_map[key]) if key in best_map else None,
                "actual": list(baseline_map[key]) if key in baseline_map else None,
            }
        for field, best_value, baseline_value in zip(
            best.group_fields, best_map[key], baseline_map[key]
        ):
            if _value_bytes(best_value) != _value_bytes(baseline_value):
                return {
                    "group_key": list(key),
                    "field": field,
                    "expected": best_value,
                    "actual": baseline_value,
                }
    if best.raw_schema != baseline.raw_schema:
        return {
            "group_key": None,
            "field": "raw_schema",
            "expected": list(best.raw_schema),
            "actual": list(baseline.raw_schema),
        }
    limit = min(len(best.raw_rows), len(baseline.raw_rows))
    for row_index in range(limit):
        if _value_bytes(best.raw_rows[row_index]) == _value_bytes(baseline.raw_rows[row_index]):
            continue
        for field_index, (best_value, baseline_value) in enumerate(
            zip(best.raw_rows[row_index], baseline.raw_rows[row_index])
        ):
            if _value_bytes(best_value) != _value_bytes(baseline_value):
                group_key = (
                    tuple(best.raw_rows[row_index][position] for position in best.group_key_positions)
                    if best.group_key_positions
                    else None
                )
                return {
                    "group_key": list(group_key) if group_key is not None else None,
                    "field": best.raw_schema[field_index],
                    "expected": best_value,
                    "actual": baseline_value,
                    "raw_row_index": row_index,
                }
    return {
        "group_key": (
            list(tuple(best.raw_rows[-1][position] for position in best.group_key_positions))
            if best.raw_rows and best.group_key_positions
            else None
        ),
        "field": "raw_row_count",
        "expected": len(best.raw_rows),
        "actual": len(baseline.raw_rows),
    }


def bias_report(
    estimate: PlanEstimate,
    execution: ExecutionRun,
    *,
    threshold_ratio: float,
    scale: int,
    raise_on_exceed: bool = True,
) -> dict[str, Any]:
    # Plan estimates already carry the fixed-point scale in their serialized
    # form; use ratio of fixed value to a fixed millisecond proxy only as a
    # stable, explicitly-declared experiment metric.
    estimated_units = estimate.total_cost_fx / scale
    actual_ms = max(execution.elapsed_ms, 1e-6)
    ratio = estimated_units / actual_ms
    deviation = ratio if ratio >= 1.0 else 1.0 / ratio
    dominant = max(estimate.operators, key=lambda operator: operator.cost_fx)
    report = {
        "estimated_cost_units": estimated_units,
        "actual_elapsed_ms": execution.elapsed_ms,
        "cost_to_time_ratio": ratio,
        "deviation_factor": deviation,
        "threshold_ratio": threshold_ratio,
        "exceeds_threshold": deviation > threshold_ratio,
        "dominant_operator_id": dominant.operator_id,
        "dominant_operator_cost_fx": dominant.cost_fx,
        "dominant_share": dominant.cost_fx / estimate.total_cost_fx if estimate.total_cost_fx else 0.0,
        "evidence": "dominant operator selected by the largest single-operator fixed cost contribution",
    }
    if report["exceeds_threshold"] and raise_on_exceed:
        raise LabError(
            "E_BIAS",
            "最优计划估算代价与真实耗时偏差超过声明阈值",
            candidate_id=estimate.plan.candidate_id,
            operator_id=dominant.operator_id,
            field="cost_to_time_ratio",
            expected=f"deviation <= {threshold_ratio}",
            actual=deviation,
            context=report,
        )
    return report


def _scan_reservation(table: Table, tuning: Any) -> int:
    width = len(table.columns) * 8 + tuning.row_overhead_bytes
    return max(64 * 1024, table.row_count * width * 2 + 8192)


def _join_reservation(table_names: tuple[str, ...], tables: dict[str, Table], tuning: Any) -> int:
    product = 1
    width = 0
    for name in table_names:
        table = tables[name]
        product *= table.row_count
        width += len(table.columns) * 8 + tuning.row_overhead_bytes
    return max(64 * 1024, product * width * 2 + 8192)


def _schema_width(schema: tuple[tuple[str, str], ...], workload: Workload) -> int:
    width = workload.tuning.row_overhead_bytes
    for table_name, column_name in schema:
        column = workload.table_map()[table_name].column(column_name)
        width += 8
    return width


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _value_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")


def _preview(result: QueryResult, limit: int = 5) -> dict[str, Any]:
    return {
        "fields": list(result.group_fields),
        "rows": [
            {"group_key": list(key), "values": list(values)}
            for key, values in result.group_rows[:limit]
        ],
    }
