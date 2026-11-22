from __future__ import annotations

import math
import sqlite3
from typing import Any, Iterable

import numpy as np

from .canonical import content_digest, normalize
from .data import quote_identifier, stable_hash
from .spec import FilterSpec, JoinSpec, QuerySpec, Spec, StatsConfig, TableSpec


STATS_MODEL_VERSION = "qplan-stats-model-v1"
HISTOGRAM_MODEL_VERSION = "equi-width-cdf-v1"
BLEND_HINT_WEIGHT = 0.25
BLEND_HISTOGRAM_WEIGHT = 0.75


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("statistics produced a non-finite value")
    return round(value, 12)


def _percentile_z(confidence: float) -> float:
    """Normal quantile without scipy.

    The default 95% value is the conventional 1.959963984540054.  Other
    confidence values are solved with a deterministic bisection on the normal
    CDF so no extra dependency is needed.
    """

    target = 0.5 + confidence / 2.0
    low, high = 0.0, 8.0
    for _ in range(80):
        middle = (low + high) / 2.0
        cdf = 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0)))
        if cdf < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def wilson_interval(probability: float, sample_size: int, confidence: float, population_size: int) -> tuple[float, float, float]:
    if sample_size <= 0:
        return 0.0, 1.0, 1.0
    p = min(max(probability, 0.0), 1.0)
    n = float(sample_size)
    z = _percentile_z(confidence)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    if population_size > 1 and sample_size < population_size:
        half *= math.sqrt((population_size - sample_size) / (population_size - 1))
    return max(0.0, center - half), min(1.0, center + half), half


def deterministic_sample_indices(row_count: int, sample_size: int, seed: int, table_name: str) -> list[int]:
    size = min(row_count, sample_size)
    if size <= 0:
        return []
    if size == row_count:
        return list(range(1, row_count + 1))
    stride = row_count // size
    offset = stable_hash(seed, "systematic-sample", table_name) % max(1, stride)
    base = (np.arange(size, dtype=np.int64) * int(row_count)) // int(size)
    indices = np.sort((base + int(offset)) % int(row_count)) + 1
    return [int(item) for item in indices]


def _histogram(values: list[Any], minimum: Any, maximum: Any, buckets: int, valid_count: int) -> list[dict[str, Any]]:
    if valid_count <= 0 or minimum is None or maximum is None:
        return []
    if maximum == minimum or buckets <= 1:
        return [
            {
                "lower": minimum,
                "upper": maximum,
                "count": valid_count,
                "fraction": 1.0 if valid_count else 0.0,
            }
        ]
    width = (float(maximum) - float(minimum)) / float(buckets)
    counts = [0 for _ in range(buckets)]
    for value in values:
        if value is None:
            continue
        if value == maximum:
            index = buckets - 1
        else:
            index = int((float(value) - float(minimum)) / width)
            if index < 0:
                index = 0
            elif index >= buckets:
                index = buckets - 1
        counts[index] += 1
    result = []
    for index, count in enumerate(counts):
        lower = float(minimum) + index * width
        upper = float(maximum) if index == buckets - 1 else float(minimum) + (index + 1) * width
        result.append(
            {
                "lower": _round(lower),
                "upper": _round(upper),
                "count": count,
                "fraction": _round(count / valid_count),
            }
        )
    return result


def _value_counts(values: Iterable[Any], limit: int) -> tuple[list[list[Any]], bool]:
    counts: dict[Any, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
        if len(counts) > limit:
            return [], True
    ordered = sorted(counts.items(), key=lambda item: (item[0],))
    return [[value, count] for value, count in ordered], False


def _column_block(
    *,
    column_type: str,
    hint: float,
    row_count: int,
    values: list[Any],
    sample_values: list[Any],
    sample_indices: list[int],
    config: StatsConfig,
    sample_seed: int,
) -> dict[str, Any]:
    exact_valid = [value for value in values if value is not None]
    sample_valid = [value for value in sample_values if value is not None]
    exact_min = min(exact_valid) if exact_valid else None
    exact_max = max(exact_valid) if exact_valid else None
    sample_min = min(sample_valid) if sample_valid else None
    sample_max = max(sample_valid) if sample_valid else None
    exact_counts, exact_truncated = _value_counts(exact_valid, config.value_count_limit)
    sample_counts, sample_truncated = _value_counts(sample_valid, config.value_count_limit)
    exact_hist = _histogram(exact_valid, exact_min, exact_max, config.histogram_buckets, len(exact_valid))
    sample_hist = _histogram(sample_valid, exact_min, exact_max, config.histogram_buckets, len(sample_valid))
    confidence = config.confidence
    z = _percentile_z(confidence)
    fpc = 1.0
    if row_count > 1 and len(sample_indices) < row_count:
        fpc = math.sqrt((row_count - len(sample_indices)) / (row_count - 1))
    return {
        "type": column_type,
        "hint_selectivity": _round(hint),
        "exact": {
            "method": "full_scan",
            "row_count": row_count,
            "null_count": row_count - len(exact_valid),
            "valid_count": len(exact_valid),
            "distinct_count": len({value for value in exact_valid}),
            "min": exact_min,
            "max": exact_max,
            "histogram_version": HISTOGRAM_MODEL_VERSION,
            "histogram": exact_hist,
            "value_counts": exact_counts,
            "value_counts_truncated": exact_truncated,
        },
        "sampled": {
            "method": "deterministic_systematic_splitmix64",
            "population_size": row_count,
            "sample_size": len(sample_indices),
            "sample_seed": sample_seed,
            "sample_indices_digest": content_digest(sample_indices),
            "null_count": len(sample_indices) - len(sample_valid),
            "valid_count": len(sample_valid),
            "distinct_count": len({value for value in sample_valid}),
            "min": sample_min,
            "max": sample_max,
            "histogram_version": HISTOGRAM_MODEL_VERSION,
            "histogram": sample_hist,
            "value_counts": sample_counts,
            "value_counts_truncated": sample_truncated,
            "confidence": confidence,
            "z_value": _round(z),
            "finite_population_correction": _round(fpc),
        },
    }


def collect_statistics(spec: Spec, connection: sqlite3.Connection, config: StatsConfig | None = None) -> dict[str, Any]:
    config = config or spec.stats
    tables: dict[str, Any] = {}
    for table in spec.tables:
        table_stats = _collect_table_statistics(spec, connection, table, config)
        tables[table.name] = table_stats
    return {
        "statistics_model_version": STATS_MODEL_VERSION,
        "histogram_model_version": HISTOGRAM_MODEL_VERSION,
        "stats_version": config.version,
        "stats_caliber_version": config.caliber_version,
        "config": config.to_dict(),
        "tables": tables,
    }


def _collect_table_statistics(
    spec: Spec,
    connection: sqlite3.Connection,
    table: TableSpec,
    config: StatsConfig,
) -> dict[str, Any]:
    actual_row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table.name)}").fetchone()[0])
    sample_indices = deterministic_sample_indices(
        actual_row_count, config.sample_size, config.random_seed, table.name
    )
    sample_set = set(sample_indices)
    columns: dict[str, Any] = {}
    for column in table.columns:
        sql = (
            f"SELECT rowid, {quote_identifier(column.name)} "
            f"FROM {quote_identifier(table.name)} ORDER BY rowid"
        )
        values: list[Any] = []
        sample_values: list[Any] = []
        for rowid, value in connection.execute(sql):
            values.append(value)
            if int(rowid) in sample_set:
                sample_values.append(value)
        columns[column.name] = _column_block(
            column_type=column.type,
            hint=column.selectivity_hint,
            row_count=actual_row_count,
            values=values,
            sample_values=sample_values,
            sample_indices=sample_indices,
            config=config,
            sample_seed=config.random_seed,
        )
    page_rows = config.page_rows
    return {
        "row_count": actual_row_count,
        "declared_row_count": table.row_count,
        "page_rows": page_rows,
        "table_page_count": max(1, math.ceil(actual_row_count / page_rows)),
        "adaptive_method": "exact" if actual_row_count < config.row_threshold else "sampled",
        "sample_size": len(sample_indices),
        "sample_indices_digest": content_digest(sample_indices),
        "columns": columns,
    }


def _lookup_value_count(block: dict[str, Any], value: Any) -> int | None:
    for candidate, count in block.get("value_counts", []):
        if candidate == value:
            return int(count)
    return None


def _equality_selectivity(
    column_stats: dict[str, Any],
    block_name: str,
    value: Any,
    table_row_count: int,
) -> tuple[float, str, bool]:
    block = column_stats[block_name]
    valid_count = int(block["valid_count"])
    if valid_count <= 0:
        return 0.0, "no_valid_values", False
    if not block.get("value_counts_truncated", False):
        count = _lookup_value_count(block, value)
        if count is not None:
            return min(1.0, count / valid_count), "value_frequency", True
        return 0.0, "value_frequency_absent", True
    distinct = max(1, int(block["distinct_count"]))
    if block_name == "sampled" and distinct >= 0.95 * int(block["sample_size"]) and table_row_count > int(block["sample_size"]):
        distinct = max(distinct, table_row_count)
    return min(1.0, 1.0 / distinct), "distinct_count_inverse", False


def _histogram_cdf(histogram: list[dict[str, Any]], threshold: float, valid_count: int) -> float:
    if valid_count <= 0 or not histogram:
        return 0.0
    limit = threshold
    accumulated = 0.0
    for bucket in histogram:
        lower = float(bucket["lower"])
        upper = float(bucket["upper"])
        count = float(bucket["count"])
        if limit <= lower:
            continue
        if limit >= upper or upper <= lower:
            accumulated += count
            continue
        accumulated += count * (limit - lower) / (upper - lower)
    return min(1.0, max(0.0, accumulated / valid_count))


def _range_selectivity(
    column_stats: dict[str, Any],
    block_name: str,
    op: str,
    value: Any,
) -> float:
    block = column_stats[block_name]
    valid_count = int(block["valid_count"])
    integer_column = column_stats["type"] == "integer"
    histogram = block["histogram"]
    if op == "lt":
        threshold = float(value) - 0.5 if integer_column else float(value)
        return _histogram_cdf(histogram, threshold, valid_count)
    if op == "le":
        threshold = float(value) + 0.5 if integer_column else float(value)
        return _histogram_cdf(histogram, threshold, valid_count)
    if op == "gt":
        threshold = float(value) + 0.5 if integer_column else float(value)
        return 1.0 - _histogram_cdf(histogram, threshold, valid_count)
    if op == "ge":
        threshold = float(value) - 0.5 if integer_column else float(value)
        return 1.0 - _histogram_cdf(histogram, threshold, valid_count)
    if op == "between":
        lower, upper = value
        upper_threshold = float(upper) + 0.5 if integer_column else float(upper)
        lower_threshold = float(lower) - 0.5 if integer_column else float(lower)
        upper_cdf = _histogram_cdf(histogram, upper_threshold, valid_count)
        lower_cdf = _histogram_cdf(histogram, lower_threshold, valid_count)
        return max(0.0, min(1.0, upper_cdf - lower_cdf))
    raise ValueError(f"not a range operator: {op}")


def _raw_selectivity(column_stats: dict[str, Any], block_name: str, op: str, value: Any, rows: int) -> tuple[float, str, bool]:
    if op == "eq":
        return _equality_selectivity(column_stats, block_name, value, rows)
    if op == "ne":
        eq, basis, exactish = _equality_selectivity(column_stats, block_name, value, rows)
        return max(0.0, 1.0 - eq), basis + "_complement", exactish
    if op == "in":
        total = 0.0
        bases = []
        exactish = True
        for item in value:
            part, basis, part_exactish = _equality_selectivity(column_stats, block_name, item, rows)
            total += part
            bases.append(basis)
            exactish = exactish and part_exactish
        return min(1.0, total), "in:" + "+".join(sorted(set(bases))), exactish
    return _range_selectivity(column_stats, block_name, op, value), "histogram_cdf", False


def _blend(raw: float, hint: float) -> float:
    return min(1.0, max(1e-12, BLEND_HISTOGRAM_WEIGHT * raw + BLEND_HINT_WEIGHT * hint))


def estimate_filter_selectivity(
    spec: Spec,
    statistics: dict[str, Any],
    filter_spec: FilterSpec,
) -> dict[str, Any]:
    table_stats = statistics["tables"][filter_spec.table]
    column_stats = table_stats["columns"][filter_spec.column]
    rows = int(table_stats["row_count"])
    config = spec.stats
    exact_raw, exact_basis, exact_value_counts = _raw_selectivity(
        column_stats, "exact", filter_spec.op, filter_spec.value, rows
    )
    sampled_raw, sampled_basis, sampled_value_counts = _raw_selectivity(
        column_stats, "sampled", filter_spec.op, filter_spec.value, rows
    )
    hint = float(column_stats["hint_selectivity"])
    exact_blended = _blend(exact_raw, hint)
    sampled_blended = _blend(sampled_raw, hint)
    sample_block = column_stats["sampled"]
    _, _, ci_half = wilson_interval(
        sampled_raw,
        int(sample_block["sample_size"]),
        config.confidence,
        rows,
    )
    if int(sample_block["sample_size"]) >= rows:
        ci_half = 0.0
    error_bound = ci_half
    adaptive_method = "exact" if rows < config.row_threshold else "sampled"
    within = abs(exact_blended - sampled_blended) <= error_bound + 1e-15
    if within:
        # Preserve the adaptive caliber while making the tie rule explicit and
        # reproducible.  The disagreement is retained and flagged as affected
        # by sampling error instead of silently switching calibers.
        final_method = adaptive_method
        tiebreak_rule = "within_error_bound:prefer_declared_adaptive_method_then_method_name"
        error_affected = True
    else:
        final_method = adaptive_method
        tiebreak_rule = "outside_error_bound:use_adaptive_method"
        error_affected = False
    final_selectivity = exact_blended if final_method == "exact" else sampled_blended
    final_error = 0.0 if final_method == "exact" else error_bound
    return {
        "table": filter_spec.table,
        "column": filter_spec.column,
        "op": filter_spec.op,
        "value": normalize(filter_spec.value),
        "adaptive_method": adaptive_method,
        "declared_stats_method": adaptive_method,
        "final_stats_method": final_method,
        "used_stats_method": final_method,
        "tiebreak_applied": within,
        "error_affected": error_affected,
        "tiebreak_rule": tiebreak_rule,
        "hint_selectivity": _round(hint),
        "hint_weight": BLEND_HINT_WEIGHT,
        "histogram_weight": BLEND_HISTOGRAM_WEIGHT,
        "exact": {
            "raw_selectivity": _round(exact_raw),
            "basis": exact_basis,
            "value_count_based": exact_value_counts,
            "blended_selectivity": _round(exact_blended),
            "histogram_version": column_stats["exact"]["histogram_version"],
            "histogram_bucket_fractions": [bucket["fraction"] for bucket in column_stats["exact"]["histogram"]],
        },
        "sampled": {
            "raw_selectivity": _round(sampled_raw),
            "basis": sampled_basis,
            "value_count_based": sampled_value_counts,
            "blended_selectivity": _round(sampled_blended),
            "sample_size": sample_block["sample_size"],
            "confidence": sample_block["confidence"],
            "ci_half_width": _round(ci_half),
            "finite_population_correction": sample_block["finite_population_correction"],
            "histogram_version": sample_block["histogram_version"],
            "histogram_bucket_fractions": [bucket["fraction"] for bucket in sample_block["histogram"]],
        },
        "selectivity": _round(final_selectivity),
        "error_upper_bound": _round(final_error),
        "error_bound": _round(final_error),
        "sampled_error_upper_bound": _round(error_bound),
        "stats_version": statistics["stats_version"],
        "stats_caliber_version": statistics["stats_caliber_version"],
    }


def estimate_table_filters(
    spec: Spec,
    statistics: dict[str, Any],
    query: QuerySpec,
    table_name: str,
) -> dict[str, Any]:
    filters = query.filters_for(table_name)
    details = [estimate_filter_selectivity(spec, statistics, item) for item in filters]
    combined = 1.0
    for item in details:
        combined *= float(item["selectivity"])
    rows = int(statistics["tables"][table_name]["row_count"])
    combined = min(1.0, max(0.0, combined))
    error = math.sqrt(sum(float(item["error_upper_bound"]) ** 2 for item in details)) if details else 0.0
    return {
        "table": table_name,
        "row_count": rows,
        "predicates": details,
        "combined_selectivity": _round(combined),
        "estimated_rows": int(min(rows, max(1, round(rows * combined)))) if filters else rows,
        "combined_error_upper_bound": _round(min(1.0, error)),
        "adaptive_method": statistics["tables"][table_name]["adaptive_method"],
        "stats_caliber_version": statistics["stats_caliber_version"],
    }


def _column_distinct_estimate(
    statistics: dict[str, Any],
    table_name: str,
    column_name: str,
) -> dict[str, Any]:
    table_stats = statistics["tables"][table_name]
    column_stats = table_stats["columns"][column_name]
    rows = int(table_stats["row_count"])
    exact = max(1, int(column_stats["exact"]["distinct_count"]))
    sampled_block = column_stats["sampled"]
    sampled_distinct = max(1, int(sampled_block["distinct_count"]))
    if sampled_distinct >= 0.95 * int(sampled_block["sample_size"]) and rows > int(sampled_block["sample_size"]):
        sampled_distinct = max(sampled_distinct, rows)
    adaptive = table_stats["adaptive_method"]
    final = exact if adaptive == "exact" else sampled_distinct
    return {
        "table": table_name,
        "column": column_name,
        "adaptive_method": adaptive,
        "exact_distinct": exact,
        "sampled_distinct": sampled_distinct,
        "selected_distinct": final,
        "stats_method": adaptive,
    }


def estimate_join_selectivity(
    statistics: dict[str, Any],
    join: JoinSpec,
) -> dict[str, Any]:
    left = _column_distinct_estimate(statistics, join.left_table, join.left_column)
    right = _column_distinct_estimate(statistics, join.right_table, join.right_column)
    denominator = max(1, int(left["selected_distinct"]), int(right["selected_distinct"]))
    selectivity = min(1.0, 1.0 / denominator)
    return {
        "left": left,
        "right": right,
        "denominator": denominator,
        "selectivity": _round(selectivity),
        "stats_caliber_version": statistics["stats_caliber_version"],
    }


def estimate_query_join(
    statistics: dict[str, Any],
    query: QuerySpec,
) -> dict[str, Any]:
    joins = [estimate_join_selectivity(statistics, join) for join in query.joins]
    selectivity = 1.0
    for join in joins:
        selectivity *= float(join["selectivity"])
    return {
        "joins": joins,
        "combined_selectivity": _round(selectivity),
        "stats_caliber_version": statistics["stats_caliber_version"],
    }


def statistics_digest(statistics: dict[str, Any]) -> str:
    return content_digest(statistics)
