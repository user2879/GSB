"""Query execution, dual verification, false-skip and false-retain detection."""

from __future__ import annotations

from typing import Any

from .attribution import attribute_decision, sort_attribution_entries
from .codec import actual_index_stats
from .errors import E_FALSE_SKIP
from .hashing import canonical_hash
from .planner import compile_query_plan, row_matches


def block_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    names = sorted(block["columns"])
    return [{name: block["columns"][name][index] for name in names} for index in range(block["row_count"])]


def matching_row_indexes(predicates: list[dict[str, Any]], block: dict[str, Any]) -> list[int]:
    rows = block_rows(block)
    return [index for index, row in enumerate(rows) if row_matches(predicates, row)]


def project_rows(rows_with_index: list[tuple[str, int, dict[str, Any]]], projection: list[str]) -> list[dict[str, Any]]:
    if projection and "*" not in projection:
        return [{column: row.get(column) for column in projection} for _, _, row in rows_with_index]
    return [dict(row) for _, _, row in rows_with_index]


def aggregate_values(function: str, values: list[Any]) -> Any:
    function = function.lower()
    if function == "count_star":
        return len(values)
    if function == "count":
        return sum(value is not None for value in values)
    non_null = [value for value in values if value is not None]
    if function == "sum":
        return sum(non_null) if non_null else 0
    if function == "avg":
        return (sum(non_null) / len(non_null)) if non_null else None
    if function == "min":
        return min(non_null) if non_null else None
    if function == "max":
        return max(non_null) if non_null else None
    raise ValueError(f"unsupported aggregate {function}")


def compute_aggregates(query: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for aggregation in query.get("aggregations", []):
        alias = aggregation["alias"]
        function = aggregation["function"]
        if function == "count_star":
            values = rows
        else:
            column = aggregation.get("column")
            values = [row.get(column) for row in rows]
        output[alias] = aggregate_values(function, values)
    return output


def actual_block_group_index(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for column, values in block["columns"].items():
        stats = actual_index_stats(values)
        result[column] = stats
    return result


def verify_query(
    query: dict[str, Any],
    spec_block_groups: list[dict[str, Any]],
    blocks_by_id: dict[str, dict[str, Any]],
    *,
    global_decoded_block_groups: int | None = None,
    global_decoded_rows: int | None = None,
    used_predicate_order: list[dict[str, Any]] | None = None,
    force_null_order: bool = False,
) -> dict[str, Any]:
    plan = compile_query_plan(query, spec_block_groups)
    groups_by_id = {group["id"]: group for group in spec_block_groups}
    group_plans = {item["block_group_id"]: item for item in plan["block_groups"]}
    predicates = plan["predicate_order"]

    all_tagged_rows: list[tuple[str, int, dict[str, Any]]] = []
    pushdown_tagged_rows: list[tuple[str, int, dict[str, Any]]] = []
    issues: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    skipped_verified = 0

    for block_group_id in sorted(blocks_by_id):
        block = blocks_by_id[block_group_id]
        group = groups_by_id[block_group_id]
        group_plan = group_plans[block_group_id]
        rows = block_rows(block)
        matched_indexes = [index for index, row in enumerate(rows) if row_matches(predicates, row)]
        tagged = [(block_group_id, index, rows[index]) for index in matched_indexes]
        all_tagged_rows.extend(tagged)

        if group_plan["decision"] == "verify":
            pushdown_tagged_rows.extend(tagged)

        if group_plan["decision"] == "skip":
            skipped_verified += 1
            if matched_indexes:
                issue = {
                    "code": E_FALSE_SKIP,
                    "kind": "false_skip",
                    "query_id": query["id"],
                    "block_group_id": block_group_id,
                    "predicate": group_plan["witness_predicate"],
                    "skip_basis": group_plan["skip_basis"],
                    "missed_rows": len(matched_indexes),
                    "missed_row_indexes": matched_indexes,
                }
                issues.append(issue)
                entry = attribute_decision(
                    kind="false_skip",
                    query=query,
                    block_group=group,
                    block=block,
                    group_plan=group_plan,
                    used_predicate_order=used_predicate_order,
                    force_null_order=force_null_order,
                )
                attribution.append(entry)
        elif not matched_indexes:
            actual_group = dict(group)
            actual_group["skip_index"] = actual_block_group_index(block)
            actual_plan = compile_query_plan(query, [actual_group])["block_groups"][0]
            if actual_plan["decision"] == "skip":
                issue = {
                    "code": "E_FALSE_RETAIN",
                    "kind": "false_retain",
                    "query_id": query["id"],
                    "block_group_id": block_group_id,
                    "predicate": actual_plan["witness_predicate"],
                    "retain_reason": "declared_index_inconclusive",
                    "actual_skip_basis": actual_plan["skip_basis"],
                    "missed_rows": 0,
                    "missed_row_indexes": [],
                }
                issues.append(issue)
                attribution.append(
                    attribute_decision(
                        kind="false_retain",
                        query=query,
                        block_group=group,
                        block=block,
                        group_plan=actual_plan,
                        used_predicate_order=used_predicate_order,
                        force_null_order=force_null_order,
                    )
                )

    all_rows = [row for _, _, row in all_tagged_rows]
    pushdown_rows = [row for _, _, row in pushdown_tagged_rows]
    projection = query.get("projection", [])
    result = {
        "matching_rows": len(all_rows),
        "missed_rows": len(all_rows) - len(pushdown_rows),
        "projection": project_rows(all_tagged_rows, projection),
        "aggregates": compute_aggregates(query, all_rows),
        "pushdown_matching_rows": len(pushdown_rows),
        "pushdown_aggregates": compute_aggregates(query, pushdown_rows),
    }
    verification = {
        "mode": "full",
        "planned_skip_block_groups": len(plan["skip_block_groups"]),
        "planned_verify_block_groups": len(plan["verify_block_groups"]),
        "skipped_block_groups_verified": skipped_verified,
        "decoded_block_groups": global_decoded_block_groups if global_decoded_block_groups is not None else len(blocks_by_id),
        "decoded_rows": global_decoded_rows if global_decoded_rows is not None else sum(block["row_count"] for block in blocks_by_id.values()),
    }
    attribution = sort_attribution_entries(attribution)
    issues.sort(key=lambda issue: (issue["block_group_id"], issue["query_id"], canonical_hash(issue.get("predicate"))))
    payload = {
        "query_id": query["id"],
        "predicate_order": plan["predicate_order"],
        "plan": plan,
        "plan_digest": plan["digest"],
        "result": result,
        "verification": verification,
        "issues": issues,
        "attribution": attribution,
    }
    payload["digest"] = canonical_hash(payload)
    return payload


def verification_summary(blocks_by_id: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    from .attribution import compare_index_statistics

    distorted = 0
    groups_by_id = {group["id"]: group for group in spec["block_groups"]}
    for block_group_id, block in blocks_by_id.items():
        group = groups_by_id[block_group_id]
        if compare_index_statistics(group.get("skip_index", {}), block.get("column_meta", {})):
            distorted += 1
    return {
        "mode": "full",
        "actual_decoded_block_groups": len(blocks_by_id),
        "actual_decoded_rows": sum(block["row_count"] for block in blocks_by_id.values()),
        "index_distorted_block_groups": distorted,
    }
