"""Dependency graph invalidation between blocks, indexes, predicates and queries."""

from __future__ import annotations

from typing import Any

from .hashing import canonical_hash


DATA_FIELDS = ("seed", "modulus", "offset", "null_every", "values", "generator")
CODEC_FIELDS = ("encoding", "compression")


def _columns_by_name(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {column["name"]: column for column in group.get("columns", [])}


def _query_references_column(query: dict[str, Any], column: str) -> bool:
    if any(predicate.get("column") == column for predicate in query.get("predicates", [])):
        return True
    if column in query.get("projection", []):
        return True
    return any(aggregation.get("column") == column for aggregation in query.get("aggregations", []))


def diff_block_groups(
    old_spec: dict[str, Any],
    new_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    old_groups = {group["id"]: group for group in old_spec.get("block_groups", [])}
    new_groups = {group["id"]: group for group in new_spec.get("block_groups", [])}
    changes: list[dict[str, Any]] = []

    for block_group_id in sorted(set(old_groups) | set(new_groups)):
        old_group = old_groups.get(block_group_id)
        new_group = new_groups.get(block_group_id)
        if old_group is None:
            changes.append(
                {
                    "block_group_id": block_group_id,
                    "kind": "added",
                    "fields": [],
                    "changed_columns": sorted(_columns_by_name(new_group)),
                    "changed_index_columns": [],
                    "row_count_changed": True,
                    "redecode": True,
                }
            )
            continue
        if new_group is None:
            changes.append(
                {
                    "block_group_id": block_group_id,
                    "kind": "removed",
                    "fields": [],
                    "changed_columns": [],
                    "changed_index_columns": [],
                    "row_count_changed": True,
                    "redecode": False,
                }
            )
            continue

        fields: set[str] = set()
        changed_columns: set[str] = set()
        row_count_changed = old_group.get("row_count") != new_group.get("row_count")
        if row_count_changed:
            fields.add("row_count")
            changed_columns.update(_columns_by_name(new_group))

        old_columns = _columns_by_name(old_group)
        new_columns = _columns_by_name(new_group)
        for column in sorted(set(old_columns) | set(new_columns)):
            old_column = old_columns.get(column)
            new_column = new_columns.get(column)
            if old_column is None or new_column is None:
                fields.add(f"columns[{column}]")
                changed_columns.add(column)
                continue
            for field in DATA_FIELDS + CODEC_FIELDS:
                if old_column.get(field) != new_column.get(field):
                    fields.add(f"columns[{column}].{field}")
                    changed_columns.add(column)

        old_index = old_group.get("skip_index", {}) or {}
        new_index = new_group.get("skip_index", {}) or {}
        changed_index_columns: set[str] = set()
        for column in sorted(set(old_index) | set(new_index)):
            old_stats = old_index.get(column, {}) or {}
            new_stats = new_index.get(column, {}) or {}
            for field in ("min", "max", "null_count"):
                if old_stats.get(field) != new_stats.get(field):
                    fields.add(f"skip_index[{column}].{field}")
                    changed_index_columns.add(column)

        if fields:
            redecode = bool(changed_columns)
            changes.append(
                {
                    "block_group_id": block_group_id,
                    "kind": "changed",
                    "fields": sorted(fields),
                    "changed_columns": sorted(changed_columns),
                    "changed_index_columns": sorted(changed_index_columns),
                    "row_count_changed": row_count_changed,
                    "redecode": redecode,
                }
            )
    return changes


def plan_invalidation(
    old_spec: dict[str, Any] | None,
    new_spec: dict[str, Any],
    *,
    old_spec_digest: str | None = None,
    new_spec_digest: str | None = None,
) -> dict[str, Any]:
    if old_spec is None:
        old_spec = {"block_groups": [], "queries": []}
    direct_changes = diff_block_groups(old_spec, new_spec)
    changed_by_id = {change["block_group_id"]: change for change in direct_changes}
    invalidated_block_groups = sorted(
        block_group_id
        for block_group_id, change in changed_by_id.items()
        if change["kind"] != "removed" or block_group_id in {g["id"] for g in new_spec.get("block_groups", [])}
    )
    redecode_block_groups = sorted(
        block_group_id for block_group_id, change in changed_by_id.items() if change.get("redecode") and change["kind"] != "removed"
    )

    old_queries = {query["id"]: query for query in old_spec.get("queries", [])}
    new_queries = {query["id"]: query for query in new_spec.get("queries", [])}
    changed_query_ids = sorted(
        query_id
        for query_id in set(old_queries) | set(new_queries)
        if canonical_hash(old_queries.get(query_id)) != canonical_hash(new_queries.get(query_id))
    )

    steps: list[dict[str, Any]] = []
    step = 1
    for change in direct_changes:
        block_group_id = change["block_group_id"]
        steps.append(
            {
                "step": step,
                "kind": "direct_block_group",
                "node": f"block_group:{block_group_id}",
                "basis": {
                    "kind": change["kind"],
                    "fields": change["fields"],
                    "redecode": change["redecode"],
                },
            }
        )

    step = 2
    column_nodes: dict[str, list[str]] = {}
    for change in direct_changes:
        block_group_id = change["block_group_id"]
        for column in change["changed_columns"]:
            node = f"block_group:{block_group_id}:column:{column}"
            column_nodes.setdefault(column, []).append(node)
            steps.append(
                {
                    "step": step,
                    "kind": "column",
                    "node": node,
                    "basis": [f"block_group:{block_group_id}"],
                }
            )
        for column in change["changed_index_columns"]:
            node = f"index:{block_group_id}:{column}"
            steps.append(
                {
                    "step": step,
                    "kind": "skip_index",
                    "node": node,
                    "basis": [f"block_group:{block_group_id}"],
                }
            )

    invalidated_queries: set[str] = set(changed_query_ids)
    step = 3
    predicate_nodes_by_query: dict[str, list[str]] = {}
    for query in new_spec.get("queries", []):
        query_id = query["id"]
        for predicate_index, predicate in enumerate(query.get("predicates", [])):
            column = predicate.get("column")
            affected_blocks = [
                block_group_id
                for block_group_id, change in changed_by_id.items()
                if column in change.get("changed_columns", []) or column in change.get("changed_index_columns", [])
            ]
            if affected_blocks:
                node = f"predicate:{query_id}:{column}"
                predicate_nodes_by_query.setdefault(query_id, []).append(node)
                invalidated_queries.add(query_id)
                steps.append(
                    {
                        "step": step,
                        "kind": "predicate",
                        "node": node,
                        "basis": sorted(affected_blocks),
                    }
                )

    step = 4
    for query in new_spec.get("queries", []):
        query_id = query["id"]
        basis = list(predicate_nodes_by_query.get(query_id, []))
        for column in sorted({c for change in direct_changes for c in change.get("changed_columns", [])}):
            if _query_references_column(query, column):
                for block_group_id, change in changed_by_id.items():
                    if column in change.get("changed_columns", []):
                        node = f"block_group:{block_group_id}:column:{column}"
                        if node not in basis:
                            basis.append(node)
                        invalidated_queries.add(query_id)
        if query_id in changed_query_ids and not basis:
            basis.append(f"query_spec:{query_id}")
        if invalidated_queries and query_id in invalidated_queries:
            steps.append(
                {
                    "step": step,
                    "kind": "query",
                    "node": f"query:{query_id}",
                    "basis": sorted(set(basis)),
                }
            )

    step = 5
    for query_id in sorted(invalidated_queries):
        steps.append(
            {
                "step": step,
                "kind": "result",
                "node": f"result:{query_id}",
                "basis": [f"query:{query_id}"],
            }
        )

    all_query_ids = [query["id"] for query in new_spec.get("queries", [])]
    invalidated_query_list = sorted(qid for qid in invalidated_queries if qid in set(all_query_ids))
    unaffected_queries = [qid for qid in all_query_ids if qid not in set(invalidated_query_list)]
    all_group_ids = [group["id"] for group in new_spec.get("block_groups", [])]
    unaffected_groups = [gid for gid in all_group_ids if gid not in set(invalidated_block_groups)]

    return {
        "old_spec_digest": old_spec_digest or canonical_hash(old_spec),
        "new_spec_digest": new_spec_digest or canonical_hash(new_spec),
        "direct_changed_block_groups": sorted(direct_changes, key=lambda item: item["block_group_id"]),
        "invalidated_block_groups": sorted(set(invalidated_block_groups) & set(all_group_ids)),
        "redecode_block_groups": redecode_block_groups,
        "invalidated_queries": invalidated_query_list,
        "unaffected_queries": unaffected_queries,
        "unaffected_block_groups": unaffected_groups,
        "propagation_steps": steps,
        "summary": {
            "changed_block_groups": len(direct_changes),
            "redecode_block_groups": len(redecode_block_groups),
            "invalidated_queries": len(invalidated_query_list),
            "unaffected_queries": len(unaffected_queries),
        },
    }
