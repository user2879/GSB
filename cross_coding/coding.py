from __future__ import annotations

from collections import Counter
from typing import Any

from .constants import ALLOWED_ENCODING_WIDTH_BITS, ALLOWED_QUANTIZATION_LEVELS
from .hashing import fingerprint, hash_value_to_width

MISSING_SENTINEL = "\x00__CROSS_CODE_MISSING__"


def row_value_tuple(table: dict[str, Any], columns: list[str], row_index: int) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in columns:
        column = table["columns"][name]
        value = column["values"][row_index]
        if value is None or column["missing_bitmap"][row_index]:
            values.append(MISSING_SENTINEL)
        else:
            values.append(value)
    return tuple(values)


def profile_for_width(policy: dict[str, Any], width_bits: int) -> dict[str, Any]:
    return {
        "encoding_width_bits": int(width_bits),
        "fixed_point_bits": int(policy["fixed_point_bits"]),
        "quantization_levels": int(policy["quantization_levels"]),
    }


def profile_for_quant_levels(policy: dict[str, Any], levels: int) -> dict[str, Any]:
    return {
        "encoding_width_bits": int(policy["encoding_width_bits"]),
        "fixed_point_bits": int(policy["fixed_point_bits"]),
        "quantization_levels": int(levels),
    }


def effective_combo_fingerprint(combo: dict[str, Any], profile: dict[str, Any]) -> str:
    return fingerprint(
        "combination",
        {"content_fingerprint": combo["content_fingerprint"], "encoding_profile": profile},
    )


def encode_with_profile(
    combo: dict[str, Any],
    table: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    width_bits = int(profile["encoding_width_bits"])
    combo_fp = effective_combo_fingerprint(combo, profile)
    codes: list[int] = []
    value_to_code: dict[str, int] = {}
    for row_index in range(table["row_count"]):
        value_tuple = row_value_tuple(table, combo["columns"], row_index)
        value_key = repr(value_tuple)
        if value_key in value_to_code:
            code = value_to_code[value_key]
        else:
            code = hash_value_to_width(value_tuple, combo_fp, width_bits)
            value_to_code[value_key] = code
        codes.append(code)

    code_counts = Counter(codes)
    total = len(codes)
    unique = len(code_counts)
    collision_count = total - unique
    collision_pairs = sum(count * (count - 1) // 2 for count in code_counts.values())
    return {
        "profile": profile,
        "combo_fingerprint": combo_fp,
        "codes": codes,
        "code_counts": {str(code): count for code, count in sorted(code_counts.items())},
        "value_dictionary": value_to_code,
        "dictionary_digest": fingerprint(
            "code-dictionary",
            {"combo_fingerprint": combo_fp, "counts": dict(sorted(code_counts.items()))},
        ),
        "total_count": total,
        "unique_count": unique,
        "collision_count": collision_count,
        "collision_rate": (collision_count / total) if total else 0.0,
        "collision_pair_count": collision_pairs,
        "colliding_code_count": sum(1 for count in code_counts.values() if count > 1),
        "width_capacity": 1 << width_bits,
    }


def cardinality_decision(encoding: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    budget = int(policy["cardinality_budget"])
    strategy = str(policy["over_cardinality_strategy"])
    natural_unique = int(encoding["unique_count"])
    total = int(encoding["total_count"])
    over_budget = natural_unique > budget
    width_bits = int(encoding["profile"]["encoding_width_bits"])

    if not over_budget:
        final_codes = list(encoding["codes"])
        final_unique = natural_unique
        action = "retain"
        bucket_count = natural_unique
        truncated = False
        reason = "within_cardinality_budget"
    elif strategy == "skip" or budget == 0:
        final_codes = []
        final_unique = 0
        action = "skip"
        bucket_count = 0
        truncated = False
        reason = "cardinality_budget_exceeded" if strategy == "skip" else "zero_cardinality_budget"
    else:
        bucket_count = budget
        final_codes = [int(code) % bucket_count for code in encoding["codes"]]
        final_unique = len(set(final_codes))
        action = "truncate_hash_buckets"
        truncated = True
        reason = "cardinality_budget_exceeded"

    final_collision = total - final_unique if action != "skip" else None
    return {
        "cardinality_budget": budget,
        "strategy": strategy,
        "over_budget": over_budget,
        "action": action,
        "reason": reason,
        "truncated": truncated,
        "requested_bucket_count": budget if over_budget and strategy == "truncate_hash_buckets" else None,
        "effective_bucket_count": bucket_count,
        "bucket_count": bucket_count,
        "raw_width_bits": width_bits,
        "raw_bucket_space": 1 << width_bits,
        "natural_unique_count": natural_unique,
        "natural_collision_count": encoding["collision_count"],
        "natural_collision_rate": encoding["collision_rate"],
        "final_unique_count": final_unique,
        "final_collision_count": final_collision,
        "final_collision_rate": (final_collision / total) if final_collision is not None and total else None,
        "final_codes": final_codes,
        "verdict": (
            "CARDINALITY_RETAINED"
            if action == "retain"
            else "CARDINALITY_TRUNCATED"
            if action == "truncate_hash_buckets"
            else "CARDINALITY_SKIPPED"
        ),
    }


def point_plan(
    combinations_list: list[dict[str, Any]],
    table: dict[str, Any],
    point_budget: int,
    width_bits: int,
) -> dict[str, Any]:
    row_count = table["row_count"]
    unit_cost = row_count * int(width_bits)
    spent = 0
    records: dict[str, dict[str, Any]] = {}
    admitted: list[str] = []
    pending: list[str] = []
    cutoff_reached = False
    for rank, combo in enumerate(combinations_list):
        fp = combo["combo_fingerprint"]
        cost = unit_cost
        before = spent
        fits = (not cutoff_reached) and spent + cost <= int(point_budget)
        if fits:
            spent += cost
            status = "admitted"
            admitted.append(fp)
        else:
            cutoff_reached = True
            status = "pending"
            pending.append(fp)
        records[fp] = {
            "rank": rank,
            "canonical_key": combo["canonical_key"],
            "combo": combo["columns"],
            "point_cost": cost,
            "cumulative_before": before,
            "cumulative_after": spent if fits else before + cost,
            "point_budget": int(point_budget),
            "point_status": status,
            "code": None if fits else "E_POINT_BUDGET_PENDING",
        }
    return {
        "width_bits": int(width_bits),
        "row_count": row_count,
        "unit_cost": unit_cost,
        "point_budget": int(point_budget),
        "spent_points": spent,
        "admitted_fingerprints": admitted,
        "pending_fingerprints": pending,
        "records": records,
    }


def adjacent_widths(width_bits: int) -> list[tuple[str, int]]:
    widths = list(ALLOWED_ENCODING_WIDTH_BITS)
    result: list[tuple[str, int]] = []
    if width_bits in widths:
        index = widths.index(width_bits)
        if index > 0:
            result.append(("width_down", widths[index - 1]))
        if index + 1 < len(widths):
            result.append(("width_up", widths[index + 1]))
        else:
            result.append(("width_up", width_bits * 2))
    else:
        sorted_widths = sorted(widths + [width_bits])
        index = sorted_widths.index(width_bits)
        if index > 0:
            result.append(("width_down", sorted_widths[index - 1]))
        if index + 1 < len(sorted_widths):
            result.append(("width_up", sorted_widths[index + 1]))
    return result


def adjacent_quant_levels(levels: int) -> list[tuple[str, int]]:
    allowed = list(ALLOWED_QUANTIZATION_LEVELS)
    result: list[tuple[str, int]] = []
    if levels in allowed:
        index = allowed.index(levels)
        if index > 0:
            result.append(("quant_down", allowed[index - 1]))
        if index + 1 < len(allowed):
            result.append(("quant_up", allowed[index + 1]))
    return result


def reference_width(width_bits: int) -> int:
    adjacent = dict(adjacent_widths(width_bits))
    if "width_up" in adjacent:
        return adjacent["width_up"]
    return width_bits * 2
