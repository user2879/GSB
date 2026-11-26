"""Stable three-source attribution for skip/retain deviations."""

from __future__ import annotations

from typing import Any

from .hashing import canonical_hash
from .planner import canonical_predicates


SOURCE_ORDER = ("index_distortion", "compression_mismatch", "predicate_null_order")


def compare_index_statistics(
    declared_index: dict[str, Any],
    actual_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for column in sorted(set(declared_index) | set(actual_meta)):
        declared = declared_index.get(column, {}) or {}
        actual = actual_meta.get(column, {}) or {}
        declared_nulls = int(declared.get("null_count", 0) or 0)
        actual_nulls = int(actual.get("null_count", 0) or 0)
        row_count = actual.get("row_count")
        all_null_sentinel = (
            actual.get("actual_min") is None
            and actual.get("actual_max") is None
            and actual_nulls == declared_nulls
            and (row_count is None or declared_nulls == row_count)
        )
        for field, actual_key in (("min", "actual_min"), ("max", "actual_max"), ("null_count", "null_count")):
            declared_value = declared.get(field)
            actual_value = actual.get(actual_key)
            if field in {"min", "max"} and all_null_sentinel and declared_value == 0:
                continue
            if declared_value != actual_value:
                mismatches.append(
                    {
                        "column": column,
                        "field": field,
                        "declared": declared_value,
                        "actual": actual_value,
                    }
                )
    mismatches.sort(key=lambda item: (item["column"], item["field"]))
    return mismatches


def compare_compression_declaration(
    block_group: dict[str, Any],
    block: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compression_mismatches: list[dict[str, Any]] = []
    encoding_mismatches: list[dict[str, Any]] = []
    specs = {column["name"]: column for column in block_group.get("columns", [])}
    for column in sorted(set(specs) | set(block.get("column_meta", {}))):
        declared = specs.get(column, {})
        actual = block.get("column_meta", {}).get(column, {})
        if declared.get("compression") != actual.get("compression"):
            compression_mismatches.append(
                {
                    "column": column,
                    "declared": declared.get("compression"),
                    "actual": actual.get("compression"),
                }
            )
        if declared.get("encoding") != actual.get("encoding"):
            encoding_mismatches.append(
                {
                    "column": column,
                    "declared": declared.get("encoding"),
                    "actual": actual.get("encoding"),
                }
            )
    return compression_mismatches, encoding_mismatches


def null_order_evidence(
    query: dict[str, Any],
    block_group: dict[str, Any],
    block: dict[str, Any],
    group_plan: dict[str, Any] | None = None,
    *,
    used_order: list[dict[str, Any]] | None = None,
    forced: bool = False,
    reason: str | None = None,
) -> tuple[float, dict[str, Any]]:
    canonical_order = canonical_predicates(query.get("predicates", []))
    if used_order is None:
        used_order = canonical_order
    used_order = canonical_predicates(used_order)
    order_differs = canonical_hash(used_order) != canonical_hash(canonical_order)

    declared_null_counts = {
        column: int((block_group.get("skip_index", {}).get(column, {}) or {}).get("null_count", 0) or 0)
        for column in sorted(block_group.get("skip_index", {}))
    }
    actual_null_counts = {
        column: int((meta or {}).get("null_count", 0) or 0)
        for column, meta in sorted(block.get("column_meta", {}).items())
    }
    null_count_mismatches = [
        {"column": column, "declared": declared_null_counts.get(column, 0), "actual": actual_null_counts.get(column, 0)}
        for column in sorted(set(declared_null_counts) | set(actual_null_counts))
        if declared_null_counts.get(column, 0) != actual_null_counts.get(column, 0)
    ]
    nullable_columns = sorted(column for column, count in actual_null_counts.items() if count)
    witness = group_plan.get("witness_predicate") if group_plan else None
    witness_fields = group_plan.get("witness_index_fields", []) if group_plan else []
    null_sensitive_witness = bool(
        witness
        and (
            witness.get("op") in {"is_null", "is_not_null"}
            or "null_count" in witness_fields
            or witness.get("column") in nullable_columns
        )
    )
    null_count_interaction = bool(null_count_mismatches and null_sensitive_witness)
    order_interaction = bool((order_differs and null_sensitive_witness) or (order_differs and nullable_columns))
    evidence = {
        "semantics": "SQL three-valued AND with canonical predicate order",
        "predicate_order": canonical_order,
        "used_predicate_order": used_order,
        "order_differs_from_canonical": order_differs,
        "nullable_columns": nullable_columns,
        "declared_null_counts": declared_null_counts,
        "actual_null_counts": actual_null_counts,
        "null_count_mismatches": null_count_mismatches,
        "null_sensitive_witness": null_sensitive_witness,
    }
    if reason:
        evidence["reason"] = reason
    if forced or order_interaction:
        evidence["caused"] = True
        weight = 1.0
    elif null_count_interaction:
        evidence["caused"] = True
        evidence["null_count_interaction"] = True
        weight = 0.8
    else:
        weight = 0.0
    return weight, evidence


def attribute_decision(
    *,
    kind: str,
    query: dict[str, Any],
    block_group: dict[str, Any],
    block: dict[str, Any],
    group_plan: dict[str, Any] | None = None,
    witness_predicate: dict[str, Any] | None = None,
    index_mismatches: list[dict[str, Any]] | None = None,
    compression_mismatches: list[dict[str, Any]] | None = None,
    encoding_mismatches: list[dict[str, Any]] | None = None,
    null_order_weight: float | None = None,
    null_order_evidence_value: dict[str, Any] | None = None,
    used_predicate_order: list[dict[str, Any]] | None = None,
    force_null_order: bool = False,
) -> dict[str, Any]:
    if index_mismatches is None:
        index_mismatches = compare_index_statistics(block_group.get("skip_index", {}), block.get("column_meta", {}))
    if compression_mismatches is None or encoding_mismatches is None:
        compression_mismatches, encoding_mismatches = compare_compression_declaration(block_group, block)

    if null_order_evidence_value is None:
        computed_weight, null_evidence = null_order_evidence(
            query,
            block_group,
            block,
            group_plan,
            used_order=used_predicate_order,
            forced=force_null_order,
        )
    else:
        null_evidence = null_order_evidence_value
        computed_weight = 1.0 if null_evidence.get("caused") else 0.0
    if null_order_weight is not None:
        computed_weight = float(null_order_weight)

    index_weight = round(min(1.2, 0.6 * len(index_mismatches)), 3) if index_mismatches else 0.0
    compression_weight = 1.0 if (compression_mismatches or encoding_mismatches) else 0.0
    weights = {
        "index_distortion": index_weight,
        "compression_mismatch": compression_weight,
        "predicate_null_order": computed_weight,
    }
    total = sum(weights.values())
    if total <= 0:
        # A deviation with no concrete source evidence is still attributable to
        # the index decision boundary; keep this explicit rather than emitting
        # NaN ratios.
        weights["index_distortion"] = 1.0
        total = 1.0

    contributions = []
    ratios: dict[str, float] = {}
    for source in SOURCE_ORDER:
        ratio = weights[source] / total if total else 0.0
        ratio = round(ratio, 6)
        ratios[source] = ratio
    rounding_gap = round(1.0 - sum(ratios.values()), 6)
    if rounding_gap:
        dominant_for_rounding = max(SOURCE_ORDER, key=lambda source: (ratios[source], -SOURCE_ORDER.index(source)))
        ratios[dominant_for_rounding] = round(ratios[dominant_for_rounding] + rounding_gap, 6)

    for source in SOURCE_ORDER:
        evidence: dict[str, Any]
        if source == "index_distortion":
            witness_fields = group_plan.get("witness_index_fields", []) if group_plan else []
            evidence = {"index_mismatches": index_mismatches, "witness_index_fields": witness_fields}
        elif source == "compression_mismatch":
            evidence = {
                "compression_mismatches": compression_mismatches,
                "encoding_mismatches": encoding_mismatches,
            }
        else:
            evidence = null_evidence
        contributions.append(
            {
                "source": source,
                "ratio": ratios[source],
                "raw_weight": weights[source],
                "evidence": evidence,
            }
        )

    dominant_source = max(SOURCE_ORDER, key=lambda source: (ratios[source], -SOURCE_ORDER.index(source)))
    entry = {
        "kind": kind,
        "query_id": query["id"],
        "block_group_id": block_group["id"],
        "witness_predicate": witness_predicate or (group_plan or {}).get("witness_predicate"),
        "dominant_source": dominant_source,
        "contributions": contributions,
    }
    entry["digest"] = canonical_hash(entry)
    return entry


def sort_attribution_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.get("block_group_id", ""),
            entry.get("query_id", ""),
            canonical_hash(entry.get("witness_predicate")),
            entry.get("kind", ""),
        ),
    )
