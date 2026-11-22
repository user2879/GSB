from __future__ import annotations

import math
from typing import Any

from .canonical import content_digest
from .paths import AccessPath
from .spec import QuerySpec, Spec
from .stats import estimate_join_selectivity, estimate_table_filters


COST_MODEL_VERSION = "qplan-cost-v1"


def _ceil(value: float) -> int:
    return max(0, int(math.ceil(value - 1e-12)))


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("cost model produced a non-finite value")
    return round(value, 12)


def _index_access(
    spec: Spec,
    query: QuerySpec,
    path: AccessPath,
    table_stats: dict[str, Any],
    table_filter_detail: dict[str, Any],
) -> dict[str, Any]:
    row_count = int(table_stats["row_count"])
    page_rows = int(table_stats["page_rows"])
    first_column = path.index_columns[0]
    first_predicates = [
        item for item in table_filter_detail["predicates"] if item["column"] == first_column
    ]
    access_selectivity = 1.0
    for item in first_predicates:
        access_selectivity *= float(item["selectivity"])
    access_selectivity = min(1.0, max(1e-12, access_selectivity))
    access_rows = max(1, _ceil(row_count * access_selectivity))
    matched_rows = int(table_filter_detail["estimated_rows"])
    residual_predicates = max(1, len(table_filter_detail["predicates"]) - len(first_predicates))
    comparisons = math.ceil(math.log2(row_count + 1.0)) + access_rows * residual_predicates + matched_rows
    index_pages = max(1, math.ceil(row_count / page_rows))
    index_io = 1 + _ceil(index_pages * min(1.0, access_selectivity))
    lookups = 0 if path.covering else matched_rows
    lookup_pages = 0 if path.covering else _ceil(lookups * 0.05)
    return {
        "access_selectivity": _round(access_selectivity),
        "access_rows": access_rows,
        "matched_rows": matched_rows,
        "comparisons": int(comparisons),
        "lookups": int(lookups),
        "io_pages": int(index_io + lookup_pages),
        "index_page_count": index_pages,
        "access_basis": "first_index_column_predicates" if first_predicates else "full_index_scan",
        "first_column_predicates": [item["column"] + ":" + item["op"] for item in first_predicates],
        "residual_predicate_count": residual_predicates,
    }


def _scan_access(table_stats: dict[str, Any], table_filter_detail: dict[str, Any]) -> dict[str, Any]:
    row_count = int(table_stats["row_count"])
    predicate_count = len(table_filter_detail["predicates"])
    comparisons = row_count * max(1, predicate_count)
    return {
        "access_selectivity": 1.0,
        "access_rows": row_count,
        "matched_rows": int(table_filter_detail["estimated_rows"]),
        "comparisons": int(comparisons),
        "lookups": 0,
        "io_pages": int(table_stats["table_page_count"]),
        "index_page_count": 0,
        "access_basis": "table_scan",
        "first_column_predicates": [],
        "residual_predicate_count": predicate_count,
    }


def _join_key_is_indexed(spec: Spec, table_name: str, column_name: str) -> bool:
    return any(
        index.columns and index.columns[0] == column_name
        for index in spec.table(table_name).candidate_indexes
    )


def _estimate_join(
    spec: Spec,
    statistics: dict[str, Any],
    query: QuerySpec,
    path: AccessPath,
    driving_rows: int,
    table_filter_details: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int, int, int, int]:
    current_rows = float(driving_rows)
    processed = {path.driving_table}
    join_probe_rows = 0
    comparisons = 0
    lookups = 0
    io_pages = 0
    steps: list[dict[str, Any]] = []
    for right_table in path.join_order:
        if right_table == path.driving_table:
            continue
        table_stats = statistics["tables"][right_table]
        right_row_count = int(table_stats["row_count"])
        right_filter = table_filter_details[right_table]
        connecting = []
        for join in query.joins:
            if join.left_table in processed and join.right_table == right_table:
                connecting.append((join, join.left_column, join.right_column))
            elif join.right_table in processed and join.left_table == right_table:
                connecting.append((join, join.right_column, join.left_column))
        if not connecting:
            raise ValueError(f"join order cannot connect table {right_table}")
        join_selectivity = 1.0
        join_details = []
        indexed = True
        for join, left_col, right_col in connecting:
            detail = estimate_join_selectivity(statistics, join)
            join_selectivity *= float(detail["selectivity"])
            index_available = indexed and _join_key_is_indexed(spec, right_table, right_col)
            indexed = indexed and index_available
            join_details.append(
                {
                    "left_table": join.left_table,
                    "left_column": join.left_column,
                    "right_table": join.right_table,
                    "right_column": join.right_column,
                    "selectivity": detail["selectivity"],
                    "right_join_column_index_available": index_available,
                }
            )
        right_filtered_rows = int(right_filter["estimated_rows"])
        probe_rows = max(1, _ceil(current_rows))
        join_probe_rows += probe_rows
        if indexed:
            per_probe_comparisons = math.ceil(math.log2(right_row_count + 1.0)) + 2
            comparisons += probe_rows * per_probe_comparisons + right_filtered_rows
            lookups += probe_rows if right_filtered_rows > 0 else 0
            io_pages += max(1, _ceil(probe_rows * 0.1)) + int(table_stats["table_page_count"])
        else:
            comparisons += probe_rows * (right_row_count + 1) + right_filtered_rows
            io_pages += probe_rows * int(table_stats["table_page_count"])
        output_rows = current_rows * max(0.0, float(right_filtered_rows)) * join_selectivity
        steps.append(
            {
                "right_table": right_table,
                "input_rows": _round(current_rows),
                "right_row_count": right_row_count,
                "right_filter_selectivity": right_filter["combined_selectivity"],
                "right_filtered_rows": right_filtered_rows,
                "join_selectivity": _round(join_selectivity),
                "join_details": join_details,
                "join_key_indexed_after_filters": indexed,
                "probe_rows": probe_rows,
                "output_rows": _round(output_rows),
            }
        )
        current_rows = output_rows
        processed.add(right_table)
    return (
        {
            "steps": steps,
            "estimated_output_rows": max(0, _ceil(current_rows)),
            "join_probe_rows": join_probe_rows,
            "comparisons": comparisons,
            "lookups": lookups,
            "io_pages": io_pages,
        },
        join_probe_rows,
        comparisons,
        lookups,
        io_pages,
    )


def estimate_path(spec: Spec, statistics: dict[str, Any], query: QuerySpec, path: AccessPath) -> dict[str, Any]:
    table_filter_details = {
        table: estimate_table_filters(spec, statistics, query, table) for table in query.tables
    }
    driving_stats = statistics["tables"][path.driving_table]
    driving_filters = table_filter_details[path.driving_table]
    if path.uses_index:
        base = _index_access(spec, query, path, driving_stats, driving_filters)
    else:
        base = _scan_access(driving_stats, driving_filters)
    join_estimate, join_probes, join_comparisons, join_lookups, join_io = _estimate_join(
        spec,
        statistics,
        query,
        path,
        int(base["matched_rows"]),
        table_filter_details,
    )
    total_access_rows = int(base["access_rows"]) + int(join_probes)
    total_comparisons = int(base["comparisons"]) + int(join_comparisons)
    total_lookups = int(base["lookups"]) + int(join_lookups)
    total_io_pages = int(base["io_pages"]) + int(join_io)
    weights = spec.cost_weights.to_dict()
    contributions = {
        "rows": weights["rows"] * total_access_rows,
        "comparisons": weights["comparisons"] * total_comparisons,
        "lookups": weights["lookups"] * total_lookups,
        "io_pages": weights["io_pages"] * total_io_pages,
    }
    total_cost = int(sum(contributions.values()))
    stats_methods = {
        table: {
            "adaptive_method": table_filter_details[table]["adaptive_method"],
            "combined_error_upper_bound": table_filter_details[table]["combined_error_upper_bound"],
            "predicate_final_methods": [item["final_stats_method"] for item in table_filter_details[table]["predicates"]],
            "tiebreak_applied": [item["tiebreak_applied"] for item in table_filter_details[table]["predicates"]],
            "error_affected": [item["error_affected"] for item in table_filter_details[table]["predicates"]],
        }
        for table in query.tables
    }
    maximum_error = max(
        [float(table_filter_details[table]["combined_error_upper_bound"]) for table in query.tables] or [0.0]
    )
    estimate = {
        "cost_model_version": COST_MODEL_VERSION,
        "weights": weights,
        "access_rows": int(base["access_rows"]),
        "join_probe_rows": int(join_probes),
        "total_access_rows": int(total_access_rows),
        "comparisons": int(total_comparisons),
        "lookups": int(total_lookups),
        "io_pages": int(total_io_pages),
        "estimated_output_rows": int(join_estimate["estimated_output_rows"]),
        "base_access": base,
        "join": join_estimate,
        "cost_components": contributions,
        "cost": int(total_cost),
        "stats_provenance": {
            "stats_version": statistics["stats_version"],
            "stats_caliber_version": statistics["stats_caliber_version"],
            "statistics_model_version": statistics["statistics_model_version"],
            "histogram_model_version": statistics["histogram_model_version"],
            "tables": stats_methods,
            "maximum_error_upper_bound": _round(maximum_error),
        },
        "filter_details": table_filter_details,
    }
    estimate["total_cost"] = int(total_cost)
    estimate["estimate_digest"] = content_digest(estimate)
    return estimate


def estimate_query_paths(
    spec: Spec,
    statistics: dict[str, Any],
    query: QuerySpec,
    paths: list[AccessPath],
) -> dict[str, Any]:
    estimates = {path.id: estimate_path(spec, statistics, query, path) for path in paths}
    return {"query_id": query.id, "estimates": estimates}
