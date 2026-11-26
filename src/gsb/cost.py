from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import digest_json
from .graph import build_join_graph
from .planner import CandidateSet, Plan, PlanOperator, enumerate_candidates
from .schema import BlockGroupDecl, CostModel, FilterPredicate, JoinPredicate, Workload

MAX_INT64 = 2**63 - 1
COST_VECTOR_WIDTH = 5


def sat_add(left: int, right: int) -> int:
    return min(MAX_INT64, left + right)


def sat_add_many(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total = sat_add(total, value)
    return total


def sat_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return min(MAX_INT64, left * right)


def sat_mul_div(left: int, right: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if left == 0 or right == 0:
        return 0
    return min(MAX_INT64, (left * right) // divisor)


def quantize_rows(rows: int, batch_rows: int) -> int:
    if rows <= 0:
        return 0
    quantized = ((rows + batch_rows - 1) // batch_rows) * batch_rows
    return min(MAX_INT64, quantized)


def _table_width(group: BlockGroupDecl) -> int:
    return group.row_width_bytes


def _distinct_bins(group: BlockGroupDecl, column_name: str) -> int:
    return max(1, sum(1 for count in group.column(column_name).histogram.counts if count > 0))


def estimate_filter_rows(group: BlockGroupDecl, column_name: str, operator: str, value: int | tuple[int, ...]) -> int:
    column = group.column(column_name)
    histogram = column.histogram
    bounds = histogram.bounds
    counts = histogram.counts
    row_count = group.row_count

    def equality_count(target: int) -> int:
        for index, count in enumerate(counts):
            low, high = bounds[index], bounds[index + 1]
            if low <= target < high:
                width = high - low
                if width == 1:
                    return count
                return count // width
        return 0

    if operator == "=":
        return equality_count(int(value))
    if operator == "in":
        values = {int(item) for item in value}  # type: ignore[union-attr]
        return min(row_count, sum(equality_count(item) for item in values))
    if operator == "!=":
        return max(0, row_count - equality_count(int(value)))

    target = int(value)  # type: ignore[arg-type]
    if operator == "<":
        range_low, range_high = bounds[0], target
    elif operator == "<=":
        range_low, range_high = bounds[0], min(MAX_INT64, target + 1)
    elif operator == ">":
        range_low, range_high = min(MAX_INT64, target + 1), bounds[-1]
    elif operator == ">=":
        range_low, range_high = target, bounds[-1]
    else:
        raise ValueError(f"unsupported filter operator {operator}")

    selected = 0
    for index, count in enumerate(counts):
        low, high = bounds[index], bounds[index + 1]
        overlap_low = max(low, range_low)
        overlap_high = min(high, range_high)
        if overlap_high <= overlap_low:
            continue
        width = high - low
        if overlap_low == low and overlap_high == high:
            selected += count
        else:
            selected += (count * (overlap_high - overlap_low)) // width
    return min(row_count, selected)


def estimate_group_rows(group: BlockGroupDecl, filters: tuple[FilterPredicate, ...], batch_rows: int) -> tuple[int, list[dict[str, Any]]]:
    selected = group.row_count
    trace: list[dict[str, Any]] = []
    for predicate in filters:
        eligible = estimate_filter_rows(
            group,
            predicate.column.column,
            predicate.operator,
            predicate.value,
        )
        selected = sat_mul_div(selected, eligible, group.row_count)
        trace.append(
            {
                "column": predicate.column.qualified,
                "operator": predicate.operator,
                "eligible_rows": eligible,
                "remaining_rows": selected,
            }
        )
    return quantize_rows(selected, batch_rows), trace


def _equality_distinct_denominator(
    workload: Workload,
    prefix: tuple[str, ...],
    table: str,
    predicates: tuple[JoinPredicate, ...],
) -> int:
    denominator = 1
    prefix_set = set(prefix)
    for predicate in predicates:
        if not predicate.is_equality:
            continue
        if predicate.right.block_group == table and predicate.left.block_group in prefix_set:
            ref = predicate.right
        elif predicate.left.block_group == table and predicate.right.block_group in prefix_set:
            ref = predicate.left
        else:
            continue
        group = workload.block_group(ref.block_group)
        denominator = sat_mul(denominator, _distinct_bins(group, ref.column))
    return max(1, denominator)


def _comparison_cost(algorithm: str, left_rows: int, right_rows: int, output_rows: int, predicate_count: int) -> int:
    predicate_count = max(1, predicate_count)
    if algorithm == "hash":
        return sat_mul(sat_add(left_rows, right_rows), predicate_count)
    if algorithm == "sort_merge":
        left_sort = sat_mul(left_rows, max(1, left_rows.bit_length()))
        right_sort = sat_mul(right_rows, max(1, right_rows.bit_length()))
        merge = sat_mul(sat_add(left_rows, right_rows), predicate_count)
        return sat_add_many((left_sort, right_sort, merge))
    if algorithm == "nested_loop":
        return sat_mul(sat_mul(left_rows, right_rows), predicate_count)
    raise ValueError(f"unsupported join algorithm {algorithm}")


def _chunk_rows(rows: int, batch_rows: int) -> int:
    return 0 if rows <= 0 else min(rows, batch_rows)


def _memory_peak(
    algorithm: str,
    left_rows: int,
    right_rows: int,
    output_rows: int,
    left_width: int,
    right_width: int,
    batch_rows: int,
) -> int:
    out_width = sat_add(left_width, right_width)
    right_chunk = _chunk_rows(right_rows, batch_rows)
    left_chunk = _chunk_rows(left_rows, batch_rows)
    output_chunk = _chunk_rows(output_rows, batch_rows)
    if algorithm == "hash":
        return sat_mul(sat_add(right_chunk, output_chunk), out_width)
    if algorithm == "sort_merge":
        return sat_add(
            sat_mul(sat_add(left_chunk, right_chunk), out_width),
            sat_mul(output_chunk, out_width),
        )
    if algorithm == "nested_loop":
        return sat_mul(sat_add(right_chunk, output_chunk), out_width)
    raise ValueError(f"unsupported join algorithm {algorithm}")


@dataclass(frozen=True)
class CostEstimate:
    plan_id: str
    cost_vector: tuple[int, int, int, int, int]
    measurements: dict[str, int]
    component_costs: dict[str, int]
    intermediates: tuple[dict[str, Any], ...]
    arithmetic_contract: dict[str, str]

    @property
    def total_cost(self) -> int:
        return self.cost_vector[-1]

    @property
    def cost_digest(self) -> str:
        return digest_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "cost_vector": list(self.cost_vector),
            "measurements": self.measurements,
            "component_costs": self.component_costs,
            "intermediates": list(self.intermediates),
            "arithmetic_contract": self.arithmetic_contract,
        }


@dataclass(frozen=True)
class Adjudication:
    selected_plan_id: str
    other_plan_id: str
    selected_cost_vector: tuple[int, int, int, int, int]
    other_cost_vector: tuple[int, int, int, int, int]
    total_delta: int
    rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_plan_id": self.selected_plan_id,
            "other_plan_id": self.other_plan_id,
            "selected_cost_vector": list(self.selected_cost_vector),
            "other_cost_vector": list(self.other_cost_vector),
            "total_delta": self.total_delta,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class PlanSelection:
    selected_plan: Plan
    selected_cost: CostEstimate
    cost_floor_plan: Plan
    cost_floor: CostEstimate
    finalists: tuple[Plan, ...]
    error_affected: bool
    error_band: int
    adjudications: tuple[Adjudication, ...]
    all_evaluations: tuple[tuple[Plan, CostEstimate], ...]

    def as_dict(self, *, include_all_evaluations: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "selected_plan": self.selected_plan.as_dict(),
            "selected_cost": self.selected_cost.as_dict(),
            "cost_floor_plan_id": self.cost_floor_plan.plan_id,
            "cost_floor": self.cost_floor.as_dict(),
            "finalist_plan_ids": [plan.plan_id for plan in self.finalists],
            "error_affected": self.error_affected,
            "error_band": self.error_band,
            "adjudications": [item.as_dict() for item in self.adjudications],
        }
        if include_all_evaluations:
            result["all_evaluations"] = [
                {"plan": plan.as_dict(), "cost": cost.as_dict()}
                for plan, cost in self.all_evaluations
            ]
        return result


def estimate_plan(workload: Workload, plan: Plan) -> CostEstimate:
    model = workload.cost_model
    query_groups = {group.id: group for group in workload.query_block_groups()}
    access_rows = sum(group.row_count for group in query_groups.values())

    rows_by_table: dict[str, int] = {}
    scan_filters: dict[str, tuple[FilterPredicate, ...]] = {}
    scan_traces: dict[str, list[dict[str, Any]]] = {}
    for operator in plan.operators:
        scan_filters[operator.table or ""] = operator.filters
    for table in plan.tables:
        group = query_groups[table]
        rows_by_table[table], scan_traces[table] = estimate_group_rows(
            group,
            scan_filters.get(table, ()),
            model.batch_rows,
        )

    current_rows = rows_by_table[plan.tables[0]]
    current_width = _table_width(query_groups[plan.tables[0]])
    prefix: tuple[str, ...] = (plan.tables[0],)
    comparisons = 0
    memory_peak = 0
    intermediate_rows = 0
    intermediates: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "scan",
            "table": plan.tables[0],
            "base_rows": query_groups[plan.tables[0]].row_count,
            "estimated_rows": current_rows,
            "filter_trace": scan_traces[plan.tables[0]],
            "row_width_bytes": current_width,
        }
    ]

    for operator in plan.operators:
        if operator.kind != "join_right":
            continue
        assert operator.table is not None and operator.algorithm is not None
        table = operator.table
        right_group = query_groups[table]
        right_rows = rows_by_table[table]
        right_width = _table_width(right_group)
        denominator = _equality_distinct_denominator(workload, prefix, table, operator.predicates)
        output_rows = sat_mul_div(current_rows, right_rows, denominator)
        for predicate in operator.predicates:
            if not predicate.is_equality:
                output_rows = sat_mul_div(output_rows, 1, 2)
        output_rows = quantize_rows(output_rows, model.batch_rows)

        step_comparisons = _comparison_cost(
            operator.algorithm,
            current_rows,
            right_rows,
            output_rows,
            len(operator.predicates),
        )
        comparisons = sat_add(comparisons, step_comparisons)
        step_memory = _memory_peak(
            operator.algorithm,
            current_rows,
            right_rows,
            output_rows,
            current_width,
            right_width,
            model.batch_rows,
        )
        memory_peak = max(memory_peak, step_memory)
        intermediate_rows = sat_add(intermediate_rows, output_rows)
        intermediates.append(
            {
                "step": len(intermediates),
                "kind": "join",
                "table": table,
                "algorithm": operator.algorithm,
                "left_rows": current_rows,
                "right_base_rows": right_group.row_count,
                "right_rows": right_rows,
                "equality_distinct_denominator": denominator,
                "output_rows": output_rows,
                "comparisons": step_comparisons,
                "memory_peak_bytes": step_memory,
                "row_width_bytes": sat_add(current_width, right_width),
                "predicates": [predicate.as_dict() for predicate in operator.predicates],
            }
        )
        prefix = prefix + (table,)
        current_rows = output_rows
        current_width = sat_add(current_width, right_width)

    access_cost = sat_mul_div(access_rows, model.access_row_weight, model.fixed_scale)
    comparison_cost = sat_mul_div(comparisons, model.comparison_weight, model.fixed_scale)
    memory_cost = sat_mul_div(memory_peak, model.memory_byte_weight, model.fixed_scale)
    intermediate_cost = sat_mul_div(intermediate_rows, model.intermediate_row_weight, model.fixed_scale)
    total = sat_add_many((access_cost, comparison_cost, memory_cost, intermediate_cost))

    measurements = {
        "access_rows": access_rows,
        "comparisons": comparisons,
        "memory_peak_bytes": memory_peak,
        "intermediate_rows": intermediate_rows,
        "final_rows": current_rows,
    }
    component_costs = {
        "access": access_cost,
        "comparison": comparison_cost,
        "memory": memory_cost,
        "intermediate": intermediate_cost,
    }
    vector = (access_cost, comparison_cost, memory_cost, intermediate_cost, total)
    return CostEstimate(
        plan_id=plan.plan_id,
        cost_vector=vector,
        measurements=measurements,
        component_costs=component_costs,
        intermediates=tuple(intermediates),
        arithmetic_contract=model.arithmetic_contract,
    )


def evaluate_candidates(candidate_set: CandidateSet, workload: Workload) -> tuple[tuple[Plan, CostEstimate], ...]:
    evaluations = tuple((plan, estimate_plan(workload, plan)) for plan in candidate_set.plans)
    return tuple(
        sorted(evaluations, key=lambda item: (item[1].total_cost, item[0].plan_id, item[0].operator_digest))
    )


def select_plan(evaluations: Iterable[tuple[Plan, CostEstimate]], error_band: int) -> PlanSelection:
    ordered = tuple(
        sorted(evaluations, key=lambda item: (item[1].total_cost, item[0].plan_id, item[0].operator_digest))
    )
    if not ordered:
        raise ValueError("at least one plan evaluation is required")
    if error_band < 0:
        raise ValueError("error_band must be non-negative")
    floor_plan, floor_cost = ordered[0]
    finalists = tuple(
        plan for plan, cost in ordered if cost.total_cost - floor_cost.total_cost <= error_band
    )
    selected_plan = min(finalists, key=lambda plan: (plan.plan_id, plan.operator_digest))
    selected_cost = next(cost for plan, cost in ordered if plan.plan_id == selected_plan.plan_id)
    error_affected = len(finalists) > 1
    adjudications: list[Adjudication] = []
    for plan in finalists:
        if plan.plan_id == selected_plan.plan_id:
            continue
        other_cost = next(cost for other, cost in ordered if other.plan_id == plan.plan_id)
        adjudications.append(
            Adjudication(
                selected_plan_id=selected_plan.plan_id,
                other_plan_id=plan.plan_id,
                selected_cost_vector=selected_cost.cost_vector,
                other_cost_vector=other_cost.cost_vector,
                total_delta=other_cost.total_cost - selected_cost.total_cost,
                rule="abs(total_delta)<=error_band:min(plan_id,operator_digest)",
            )
        )
    return PlanSelection(
        selected_plan=selected_plan,
        selected_cost=selected_cost,
        cost_floor_plan=floor_plan,
        cost_floor=floor_cost,
        finalists=finalists,
        error_affected=error_affected,
        error_band=error_band,
        adjudications=tuple(adjudications),
        all_evaluations=ordered,
    )


def plan_and_select(workload: Workload) -> tuple[CandidateSet, PlanSelection]:
    plans = enumerate_candidates(workload)
    candidates = CandidateSet(
        query_id=workload.query.id,
        graph=build_join_graph(workload.query),
        plans=plans.plans,
    )
    evaluations = evaluate_candidates(candidates, workload)
    selection = select_plan(evaluations, workload.cost_model.error_band)
    return candidates, selection
