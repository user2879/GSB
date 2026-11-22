from __future__ import annotations

from typing import Any

from .canonical import content_digest, normalize
from .spec import Spec


INVALIDATION_MODEL_VERSION = "qplan-invalidation-v1"


def _table_map(spec_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {table["name"]: table for table in spec_json.get("tables", [])}


def _query_map(spec_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {query["id"]: query for query in spec_json.get("queries", [])}


def _column_map(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {column["name"]: column for column in table.get("columns", [])}


def _index_signature(table: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    return sorted(
        (index["name"], tuple(index["columns"])) for index in table.get("candidate_indexes", [])
    )


def _query_references_table(query: dict[str, Any], table: str) -> bool:
    return table in query.get("tables", [])


def _query_references_column(query: dict[str, Any], table: str, column: str) -> bool:
    for item in query.get("filters", []):
        if item.get("table") == table and item.get("column") == column:
            return True
    for item in query.get("joins", []):
        if (item.get("left_table") == table and item.get("left_column") == column) or (
            item.get("right_table") == table and item.get("right_column") == column
        ):
            return True
    return False


def compute_invalidation(
    previous: dict[str, Any] | None,
    current_spec: Spec,
    current_statistics: dict[str, Any],
) -> dict[str, Any]:
    current_spec_json = normalize(current_spec.to_dict())
    query_reasons: dict[str, list[str]] = {query.id: [] for query in current_spec.queries}
    changed: list[dict[str, Any]] = []
    if previous is None:
        for query in current_spec.queries:
            query_reasons[query.id].append("initial_snapshot")
        result = {
            "invalidation_model_version": INVALIDATION_MODEL_VERSION,
            "change_kind": "initial_snapshot",
            "changed": False,
            "changes": [],
            "affected_queries": [
                {"query_id": query.id, "reasons": ["initial_snapshot"]} for query in current_spec.queries
            ],
            "unaffected_queries": [],
            "stale_scope": "no previous committed conclusion view",
        }
        result["invalidation_digest"] = content_digest(result)
        return result

    previous_spec = previous["spec"]
    previous_stats = previous["stats"]
    previous_tables = _table_map(previous_spec)
    current_tables = _table_map(current_spec_json)
    previous_queries = _query_map(previous_spec)
    current_queries = _query_map(current_spec_json)

    def reason_all(reason: str) -> None:
        changed.append({"kind": "all_queries", "reason": reason})
        for query_id in query_reasons:
            query_reasons[query_id].append(reason)

    def reason_table(table: str, reason: str) -> None:
        changed.append({"kind": "table", "table": table, "reason": reason})
        for query_id, query in current_queries.items():
            if _query_references_table(query, table):
                query_reasons[query_id].append(reason)

    def reason_column(table: str, column: str, reason: str) -> None:
        changed.append({"kind": "column", "table": table, "column": column, "reason": reason})
        for query_id, query in current_queries.items():
            if _query_references_column(query, table, column):
                query_reasons[query_id].append(reason)

    for table_name in sorted(set(previous_tables) | set(current_tables)):
        previous_table = previous_tables.get(table_name)
        current_table = current_tables.get(table_name)
        if previous_table is None:
            reason_table(table_name, f"table_added:{table_name}")
            continue
        if current_table is None:
            changed.append({"kind": "table", "table": table_name, "reason": f"table_removed:{table_name}"})
            continue
        previous_row_count = int(previous_table.get("row_count", -1))
        current_row_count = int(current_table.get("row_count", -1))
        previous_stats_rows = previous_stats.get("tables", {}).get(table_name, {}).get("row_count")
        current_stats_rows = current_statistics.get("tables", {}).get(table_name, {}).get("row_count")
        if previous_row_count != current_row_count:
            reason_table(table_name, f"data_row_count_changed:{table_name}")
        elif previous_stats_rows != current_stats_rows:
            reason_table(table_name, f"materialized_row_count_changed:{table_name}")
        previous_columns = _column_map(previous_table)
        current_columns = _column_map(current_table)
        for column_name in sorted(set(previous_columns) | set(current_columns)):
            old_column = previous_columns.get(column_name)
            new_column = current_columns.get(column_name)
            if old_column is None:
                reason_table(table_name, f"column_added:{table_name}.{column_name}")
                continue
            if new_column is None:
                reason_table(table_name, f"column_removed:{table_name}.{column_name}")
                continue
            if old_column.get("type") != new_column.get("type"):
                reason_table(table_name, f"column_type_changed:{table_name}.{column_name}")
            if old_column.get("selectivity_hint") != new_column.get("selectivity_hint"):
                reason_column(table_name, column_name, f"selectivity_hint_changed:{table_name}.{column_name}")
        if _index_signature(previous_table) != _index_signature(current_table):
            reason_table(table_name, f"candidate_index_set_changed:{table_name}")

    previous_stats_config = previous_stats.get("config", {})
    current_stats_config = current_statistics.get("config", {})
    if previous_stats_config != current_stats_config or previous_stats.get("stats_caliber_version") != current_statistics.get("stats_caliber_version"):
        reason_all("stats_caliber_or_config_changed")
    previous_weights = previous.get("weights", {})
    current_weights = current_spec.cost_weights.to_dict()
    if previous_weights != current_weights:
        reason_all("cost_weights_changed")

    previous_search = previous_spec.get("search", {})
    current_search = current_spec.search.to_dict()
    if previous_search != current_search:
        reason_all("index_search_config_changed")
    previous_bias = previous_spec.get("bias", {})
    current_bias = current_spec.bias.to_dict()
    if previous_bias != current_bias:
        reason_all("bias_threshold_changed")

    for query_id in sorted(set(previous_queries) | set(current_queries)):
        old_query = previous_queries.get(query_id)
        new_query = current_queries.get(query_id)
        if old_query is None:
            if query_id in query_reasons:
                query_reasons[query_id].append(f"query_added:{query_id}")
            changed.append({"kind": "query", "query_id": query_id, "reason": f"query_added:{query_id}"})
        elif new_query is None:
            changed.append({"kind": "query", "query_id": query_id, "reason": f"query_removed:{query_id}"})
        elif old_query != new_query:
            query_reasons[query_id].append(f"query_definition_changed:{query_id}")
            changed.append(
                {"kind": "query", "query_id": query_id, "reason": f"query_definition_changed:{query_id}"}
            )

    affected = []
    unaffected = []
    for query_id in sorted(query_reasons):
        reasons = sorted(set(query_reasons[query_id]))
        if reasons:
            affected.append({"query_id": query_id, "reasons": reasons})
        else:
            unaffected.append(query_id)
    change_kind = "content_changed" if changed else "same_content"
    result = {
        "invalidation_model_version": INVALIDATION_MODEL_VERSION,
        "change_kind": change_kind,
        "changed": bool(changed),
        "changes": changed,
        "affected_queries": affected,
        "unaffected_queries": unaffected,
        "stale_scope": "all prior estimates, selections and bias conclusions are marked stale when changed",
        "replacement_semantics": "current view is the new snapshot only; prior rows are never merged",
    }
    result["invalidation_digest"] = content_digest(result)
    return result
