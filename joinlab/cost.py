from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import json
import math
from typing import Any, Iterable

from .model import Histogram, JoinPredicate, Plan, Table, Tuning, Workload
from .stats import FilterStats, StatisticsBundle


getcontext().prec = 40
MAX_FIXED = (1 << 63) - 1
SATURATION_RULE = "nonnegative values are truncated toward zero after multiplying by fixed_point_scale; values above 2^63-1 saturate"


@dataclass(frozen=True)
class OperatorEstimate:
    operator_id: str
    kind: str
    algorithm: str | None
    tables: tuple[str, ...]
    input_rows: Decimal
    output_rows: Decimal
    comparisons: Decimal
    io_pages: Decimal
    peak_memory_bytes: Decimal
    raw_cost: Decimal
    output_rows_fx: int
    comparisons_fx: int
    io_pages_fx: int
    peak_memory_bytes_fx: int
    cost_fx: int
    rows_low_fx: int
    rows_high_fx: int
    hist_versions: tuple[str, ...]
    stats_versions: tuple[str, ...]
    confidence_level: float | None
    quantization_bound_fx: int = 1
    saturated: bool = False

    def as_dict(self, scale: int) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "kind": self.kind,
            "algorithm": self.algorithm,
            "tables": list(self.tables),
            "input_rows_fx": fixed(self.input_rows, scale),
            "output_rows_fx": self.output_rows_fx,
            "comparisons_fx": self.comparisons_fx,
            "io_pages_fx": self.io_pages_fx,
            "peak_memory_bytes_fx": self.peak_memory_bytes_fx,
            "cost_fx": self.cost_fx,
            "rows_low_fx": self.rows_low_fx,
            "rows_high_fx": self.rows_high_fx,
            "input_rows": float(self.input_rows),
            "output_rows": float(self.output_rows),
            "comparisons": float(self.comparisons),
            "io_pages": float(self.io_pages),
            "peak_memory_bytes": float(self.peak_memory_bytes),
            "cost": float(self.raw_cost),
            "hist_versions": list(self.hist_versions),
            "stats_versions": list(self.stats_versions),
            "confidence_level": self.confidence_level,
            "quantization_error_bound_fx": self.quantization_bound_fx,
            "saturated": self.saturated,
        }


@dataclass(frozen=True)
class PlanEstimate:
    plan: Plan
    operators: tuple[OperatorEstimate, ...]
    total_cost_fx: int
    access_rows_fx: int
    comparisons_fx: int
    io_pages_fx: int
    peak_memory_bytes_fx: int
    output_rows_fx: int
    plan_quantization_bound_fx: int
    hist_versions: tuple[str, ...]
    stats_versions: tuple[str, ...]
    saturated: bool
    digest: str

    def as_dict(self, scale: int) -> dict[str, Any]:
        return {
            "candidate_id": self.plan.candidate_id,
            "plan_signature": self.plan.signature,
            "origin": self.plan.origin,
            "baseline_reason": self.plan.baseline_reason,
            "table_order": list(self.plan.table_order),
            "steps": [
                {
                    "right_table": step.right_table,
                    "algorithm": step.algorithm,
                    "predicates": [item.canonical() for item in step.predicates],
                }
                for step in self.plan.steps
            ],
            "total_cost_fx": self.total_cost_fx,
            "total_cost": self.total_cost_fx / scale,
            "access_rows_fx": self.access_rows_fx,
            "access_rows": self.access_rows_fx / scale,
            "comparisons_fx": self.comparisons_fx,
            "comparisons": self.comparisons_fx / scale,
            "io_pages_fx": self.io_pages_fx,
            "io_pages": self.io_pages_fx / scale,
            "peak_memory_bytes_fx": self.peak_memory_bytes_fx,
            "peak_memory_bytes": self.peak_memory_bytes_fx / scale,
            "output_rows_fx": self.output_rows_fx,
            "output_rows": self.output_rows_fx / scale,
            "fixed_point": {
                "scale": scale,
                "truncation": "toward_zero",
                "saturation": SATURATION_RULE,
                "per_operator_error_bound_fx": 1,
                "plan_error_bound_fx": self.plan_quantization_bound_fx,
                "plan_error_bound": self.plan_quantization_bound_fx / scale,
            },
            "hist_versions": list(self.hist_versions),
            "stats_versions": list(self.stats_versions),
            "saturated": self.saturated,
            "operators": [operator.as_dict(scale) for operator in self.operators],
            "estimate_digest": self.digest,
        }


def estimate_plan(
    plan: Plan,
    workload: Workload,
    stats: StatisticsBundle,
) -> PlanEstimate:
    tuning = workload.tuning
    tables = workload.table_map()
    filter_map = stats.selected_map()
    operators: list[OperatorEstimate] = []
    joined: set[str] = set()
    current_rows = Decimal(0)
    current_width = 0
    current_low = Decimal(0)
    current_high = Decimal(0)
    plan_hist_versions: set[str] = set()
    plan_stats_versions: set[str] = set()

    first = plan.table_order[0]
    first_scan = _scan_estimate(first, tables[first], filter_map, tuning)
    operators.append(first_scan)
    joined.add(first)
    current_rows = first_scan.output_rows
    current_width = _table_width(tables[first], tuning)
    current_low = _from_fixed(first_scan.rows_low_fx, tuning.fixed_point_scale)
    current_high = _from_fixed(first_scan.rows_high_fx, tuning.fixed_point_scale)
    plan_hist_versions.update(first_scan.hist_versions)
    plan_stats_versions.update(first_scan.stats_versions)

    for index, step in enumerate(plan.steps):
        right_table = tables[step.right_table]
        right_scan = _scan_estimate(step.right_table, right_table, filter_map, tuning)
        operators.append(right_scan)
        plan_hist_versions.update(right_scan.hist_versions)
        plan_stats_versions.update(right_scan.stats_versions)

        left_tables = tuple(sorted(joined))
        output_width = current_width + _table_width(right_table, tuning)
        join_selectivity, join_hist_versions = _join_selectivity(
            step.predicates, tables
        )
        plan_hist_versions.update(join_hist_versions)
        join_rows = current_rows * right_scan.output_rows * join_selectivity
        join_low = current_low * _from_fixed(
            right_scan.rows_low_fx, tuning.fixed_point_scale
        ) * join_selectivity
        join_high = current_high * _from_fixed(
            right_scan.rows_high_fx, tuning.fixed_point_scale
        ) * join_selectivity
        operator = _join_estimate(
            operator_id=f"join:{index}:{step.right_table}",
            algorithm=step.algorithm,
            left_tables=left_tables,
            right_table=step.right_table,
            left_rows=current_rows,
            right_rows=right_scan.output_rows,
            output_rows=join_rows,
            left_width=current_width,
            right_width=_table_width(right_table, tuning),
            output_width=output_width,
            tuning=tuning,
            hist_versions=join_hist_versions,
            rows_low=join_low,
            rows_high=join_high,
        )
        operators.append(operator)
        joined.add(step.right_table)
        current_rows = join_rows
        current_width = output_width
        current_low = join_low
        current_high = join_high

    total_cost_fx = _sat_add(operator.cost_fx for operator in operators)
    access_rows_fx = _sat_add(
        fixed(operator.input_rows, tuning.fixed_point_scale)
        for operator in operators
        if operator.kind == "scan"
    )
    comparisons_fx = _sat_add(operator.comparisons_fx for operator in operators)
    io_pages_fx = _sat_add(operator.io_pages_fx for operator in operators)
    peak_memory_fx = max((operator.peak_memory_bytes_fx for operator in operators), default=0)
    output_rows_fx = operators[-1].output_rows_fx if operators else 0
    saturated = any(operator.saturated for operator in operators)
    bound = MAX_FIXED if saturated else len(operators)
    digest_body = {
        "plan": plan.canonical(),
        "operators": [operator.as_dict(tuning.fixed_point_scale) for operator in operators],
        "stats": stats.digest,
        "scale": tuning.fixed_point_scale,
    }
    digest = hashlib.sha256(_canonical_json(digest_body).encode("utf-8")).hexdigest()
    return PlanEstimate(
        plan=plan,
        operators=tuple(operators),
        total_cost_fx=total_cost_fx,
        access_rows_fx=access_rows_fx,
        comparisons_fx=comparisons_fx,
        io_pages_fx=io_pages_fx,
        peak_memory_bytes_fx=peak_memory_fx,
        output_rows_fx=output_rows_fx,
        plan_quantization_bound_fx=bound,
        hist_versions=tuple(sorted(plan_hist_versions)),
        stats_versions=tuple(sorted(plan_stats_versions)),
        saturated=saturated,
        digest=digest,
    )


def _scan_estimate(
    table_name: str,
    table: Table,
    filter_map: dict[str, FilterStats],
    tuning: Tuning,
) -> OperatorEstimate:
    predicates = [predicate for predicate in table_predicates(table_name, filter_map)]
    selectivity = Decimal(1)
    low_selectivity = Decimal(1)
    high_selectivity = Decimal(1)
    stats_versions: list[str] = []
    confidence_levels: list[float] = []
    for item in predicates:
        profile = item.selected
        selectivity *= profile.selectivity
        low_selectivity *= profile.confidence_low
        high_selectivity *= profile.confidence_high
        stats_versions.append(f"{item.predicate_key}@{profile.version}")
        confidence_levels.append(profile.confidence_level)
    row_count = Decimal(table.row_count)
    output_rows = row_count * selectivity
    width = _table_width(table, tuning)
    io_pages = _pages(row_count * width, tuning.page_size_bytes)
    comparisons = row_count * Decimal(len(predicates))
    peak_memory = output_rows * Decimal(width)
    raw_cost = _raw_cost(comparisons, io_pages, peak_memory, tuning)
    hist_versions = tuple(
        f"{table_name}.{column.name}@v{table.histograms[column.name].version}:{table.histograms[column.name].digest[:12]}"
        for column in table.columns
    )
    return _make_operator(
        operator_id=f"scan:{table_name}",
        kind="scan",
        algorithm=None,
        tables=(table_name,),
        input_rows=row_count,
        output_rows=output_rows,
        comparisons=comparisons,
        io_pages=io_pages,
        peak_memory=peak_memory,
        raw_cost=raw_cost,
        rows_low=row_count * low_selectivity,
        rows_high=row_count * high_selectivity,
        hist_versions=hist_versions,
        stats_versions=tuple(stats_versions),
        confidence_level=min(confidence_levels) if confidence_levels else None,
        tuning=tuning,
    )


def table_predicates(table_name: str, filter_map: dict[str, FilterStats]) -> Iterable[FilterStats]:
    for key in sorted(filter_map):
        item = filter_map[key]
        if item.table == table_name:
            yield item


def _join_estimate(
    *,
    operator_id: str,
    algorithm: str,
    left_tables: tuple[str, ...],
    right_table: str,
    left_rows: Decimal,
    right_rows: Decimal,
    output_rows: Decimal,
    left_width: int,
    right_width: int,
    output_width: int,
    tuning: Tuning,
    hist_versions: tuple[str, ...],
    rows_low: Decimal,
    rows_high: Decimal,
) -> OperatorEstimate:
    if algorithm == "HASH":
        comparisons = left_rows + right_rows
        working = right_rows * Decimal(right_width) * Decimal("1.5") + left_rows * Decimal(left_width)
        io_pages = _pages((left_rows * left_width) + (right_rows * right_width) + (output_rows * output_width), tuning.page_size_bytes)
    elif algorithm == "MERGE":
        comparisons = (
            left_rows * Decimal(max(1.0, math.log2(max(1.0, float(left_rows)))))
            + right_rows * Decimal(max(1.0, math.log2(max(1.0, float(right_rows)))))
            + left_rows
            + right_rows
        )
        working = (left_rows * Decimal(left_width)) + (right_rows * Decimal(right_width))
        sort_spill = _pages(working * Decimal(2), tuning.page_size_bytes) * Decimal(2)
        io_pages = _pages((left_rows * left_width) + (right_rows * right_width) + (output_rows * output_width), tuning.page_size_bytes) + sort_spill
    elif algorithm == "NESTED_LOOP":
        comparisons = left_rows * right_rows
        working = left_rows * Decimal(left_width) + right_rows * Decimal(right_width)
        left_pages = _pages(left_rows * left_width, tuning.page_size_bytes)
        right_pages = _pages(right_rows * right_width, tuning.page_size_bytes)
        io_pages = left_pages + right_pages + (left_pages * right_pages)
    else:
        raise ValueError(f"unknown join algorithm {algorithm}")
    peak_memory = max(working, output_rows * Decimal(output_width))
    raw_cost = _raw_cost(comparisons, io_pages, peak_memory, tuning)
    return _make_operator(
        operator_id=operator_id,
        kind="join",
        algorithm=algorithm,
        tables=tuple(sorted((*left_tables, right_table))),
        input_rows=left_rows + right_rows,
        output_rows=output_rows,
        comparisons=comparisons,
        io_pages=io_pages,
        peak_memory=peak_memory,
        raw_cost=raw_cost,
        rows_low=rows_low,
        rows_high=rows_high,
        hist_versions=hist_versions,
        stats_versions=(),
        confidence_level=None,
        tuning=tuning,
    )


def _make_operator(
    *,
    operator_id: str,
    kind: str,
    algorithm: str | None,
    tables: tuple[str, ...],
    input_rows: Decimal,
    output_rows: Decimal,
    comparisons: Decimal,
    io_pages: Decimal,
    peak_memory: Decimal,
    raw_cost: Decimal,
    rows_low: Decimal,
    rows_high: Decimal,
    hist_versions: tuple[str, ...],
    stats_versions: tuple[str, ...],
    confidence_level: float | None,
    tuning: Tuning,
) -> OperatorEstimate:
    values = [output_rows, comparisons, io_pages, peak_memory, raw_cost]
    fixed_values = [quantize(value, tuning.fixed_point_scale) for value in values]
    saturated = any(
        value > 0 and quantize(value, tuning.fixed_point_scale) == MAX_FIXED for value in values
    )
    return OperatorEstimate(
        operator_id=operator_id,
        kind=kind,
        algorithm=algorithm,
        tables=tables,
        input_rows=input_rows,
        output_rows=output_rows,
        comparisons=comparisons,
        io_pages=io_pages,
        peak_memory_bytes=peak_memory,
        raw_cost=raw_cost,
        output_rows_fx=fixed_values[0],
        comparisons_fx=fixed_values[1],
        io_pages_fx=fixed_values[2],
        peak_memory_bytes_fx=fixed_values[3],
        cost_fx=fixed_values[4],
        rows_low_fx=quantize(rows_low, tuning.fixed_point_scale),
        rows_high_fx=quantize(rows_high, tuning.fixed_point_scale),
        hist_versions=hist_versions,
        stats_versions=stats_versions,
        confidence_level=confidence_level,
        quantization_bound_fx=0 if saturated else 1,
        saturated=saturated,
    )


def _join_selectivity(
    predicates: tuple[JoinPredicate, ...], tables: dict[str, Table]
) -> tuple[Decimal, tuple[str, ...]]:
    selectivity = Decimal(1)
    versions: list[str] = []
    for predicate in predicates:
        left_table = tables[predicate.left.table]
        right_table = tables[predicate.right.table]
        left_hist = left_table.histograms[predicate.left.column]
        right_hist = right_table.histograms[predicate.right.column]
        selectivity *= _histogram_equality_selectivity(left_hist, right_hist)
        versions.extend(
            [
                f"{predicate.left.key}@v{left_hist.version}:{left_hist.digest[:12]}",
                f"{predicate.right.key}@v{right_hist.version}:{right_hist.digest[:12]}",
            ]
        )
    return min(Decimal(1), max(Decimal(0), selectivity)), tuple(sorted(set(versions)))


def _histogram_equality_selectivity(left: Histogram, right: Histogram) -> Decimal:
    left_total = sum(bin.count for bin in left.bins)
    right_total = sum(bin.count for bin in right.bins)
    if left_total == 0 or right_total == 0:
        return Decimal(0)
    matched = Decimal(0)
    for left_bin in left.bins:
        left_width = left_bin.hi - left_bin.lo
        left_density = Decimal(left_bin.count) / left_width
        for right_bin in right.bins:
            overlap = min(left_bin.hi, right_bin.hi) - max(left_bin.lo, right_bin.lo)
            if overlap <= 0:
                continue
            right_width = right_bin.hi - right_bin.lo
            right_density = Decimal(right_bin.count) / right_width
            matched += overlap * left_density * right_density
    selectivity = matched / (Decimal(left_total) * Decimal(right_total))
    return min(Decimal(1), max(Decimal(0), selectivity))


def _table_width(table: Table, tuning: Tuning) -> int:
    return len(table.columns) * 8 + tuning.row_overhead_bytes


def _pages(value_bytes: Decimal, page_size: int) -> Decimal:
    if value_bytes <= 0:
        return Decimal(0)
    return Decimal(math.ceil(float(value_bytes) / page_size))


def _raw_cost(
    comparisons: Decimal, io_pages: Decimal, peak_memory: Decimal, tuning: Tuning
) -> Decimal:
    memory_pages = Decimal(0) if peak_memory <= 0 else peak_memory / Decimal(tuning.page_size_bytes)
    return (
        comparisons * Decimal(tuning.comparison_weight)
        + io_pages * Decimal(tuning.io_page_weight)
        + memory_pages * Decimal(tuning.memory_weight)
    )


def quantize(value: Decimal, scale: int) -> int:
    if value is None:
        return 0
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite():
        return MAX_FIXED
    if value <= 0:
        return 0
    result = int(value * scale)
    return min(MAX_FIXED, result)


def fixed(value: Decimal | int | float, scale: int) -> int:
    return quantize(Decimal(str(value)) if not isinstance(value, Decimal) else value, scale)


def _from_fixed(value: int, scale: int) -> Decimal:
    return Decimal(value) / Decimal(scale)


def _sat_add(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total = min(MAX_FIXED, total + max(0, value))
    return total


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
