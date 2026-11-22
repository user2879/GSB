"""Integer fixed-point ratios with explicit quantization and comparison bounds."""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any, Iterable


def fixed_ratio(numerator: int, denominator: int, scale_factor: int) -> dict[str, Any]:
    if numerator < 0 or denominator < 0:
        raise ValueError("fixed_ratio accepts non-negative integers only")
    if denominator == 0:
        return {
            "defined": False,
            "numerator": numerator,
            "denominator": denominator,
            "fixed": 0,
            "exact_ratio": None,
        }
    fixed = (numerator * scale_factor) // denominator
    return {
        "defined": True,
        "numerator": numerator,
        "denominator": denominator,
        "fixed": fixed,
        "exact_ratio": f"{numerator}/{denominator}",
    }


def error_envelope(scale_power: int) -> dict[str, Any]:
    scale_factor = 10 ** scale_power
    return {
        "scale_power": scale_power,
        "scale_factor": scale_factor,
        "truncation": "floor_toward_negative_infinity",
        "non_negative_equivalent": "truncate_toward_zero",
        "single_value": {
            "fixed_units_strictly_less_than": 1,
            "ratio_strictly_less_than": f"1/{scale_factor}",
            "formula": "0 <= exact*scale - fixed < 1",
        },
        "ratio_comparison": {
            "fixed_units_strictly_less_than": 2,
            "ratio_strictly_less_than": f"2/{scale_factor}",
            "formula": "abs((a-b)*scale) < 2 for two floored values",
            "decision_rule": "if abs(fixed_a-fixed_b) <= 2 mark affected and use topic lexical tie-break",
        },
    }


def _topic_compare(left: dict[str, Any], right: dict[str, Any], bound_units: int = 2) -> int:
    difference = int(left["fixed_exact_once_ratio"]) - int(right["fixed_exact_once_ratio"])
    if abs(difference) <= bound_units:
        if left["topic"] < right["topic"]:
            return -1
        if left["topic"] > right["topic"]:
            return 1
        return 0
    return -1 if difference > 0 else 1


def _affected_pairs(rows: list[dict[str, Any]], bound_units: int = 2) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            difference = abs(
                int(left["fixed_exact_once_ratio"]) - int(right["fixed_exact_once_ratio"])
            )
            if difference <= bound_units:
                topic_a, topic_b = sorted((left["topic"], right["topic"]))
                pairs.append(
                    {
                        "topic_a": topic_a,
                        "topic_b": topic_b,
                        "fixed_difference_units": difference,
                        "comparison_bound_units": bound_units,
                    }
                )
    pairs.sort(key=lambda item: (item["topic_a"], item["topic_b"]))
    return pairs


def build_stats(
    group_facts: Iterable[dict[str, Any]],
    group_conclusions: Iterable[dict[str, Any]],
    conflicts: Iterable[dict[str, Any]],
    *,
    scale_power: int,
) -> dict[str, Any]:
    scale_factor = 10 ** scale_power
    facts = sorted(
        group_facts,
        key=lambda row: (row["topic"], row["consumer_group"], row["message_seq"], row["message_id"]),
    )
    conclusions = sorted(
        group_conclusions,
        key=lambda row: (row["topic"], row["message_seq"], row["consumer_group"], row["message_id"]),
    )
    conflict_rows = list(conflicts)

    cells: dict[tuple[str, str], dict[str, int]] = {}
    for row in conclusions:
        key = (str(row["topic"]), str(row["consumer_group"]))
        cell = cells.setdefault(
            key,
            {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0},
        )
        cell["messages"] += 1
        if row["verdict"] == "exactly_once":
            cell["exact"] += 1
    for row in facts:
        key = (str(row["topic"]), str(row["consumer_group"]))
        cell = cells.setdefault(
            key,
            {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0},
        )
        cell["retry_count"] += int(row["retry_count_max"])
        cell["attempts"] += int(row["attempts"])
    for conflict in conflict_rows:
        topic = str(conflict["topic"])
        for group in conflict.get("consumer_groups", []):
            key = (topic, str(group))
            cell = cells.setdefault(
                key,
                {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0},
            )
            cell["conflicts"] += 1

    by_topic_group: list[dict[str, Any]] = []
    for (topic, group), values in sorted(cells.items()):
        exact_ratio = fixed_ratio(values["exact"], values["messages"], scale_factor)
        retry_ratio = fixed_ratio(values["retry_count"], values["attempts"], scale_factor)
        by_topic_group.append(
            {
                "topic": topic,
                "consumer_group": group,
                "scale_factor": scale_factor,
                "single_error_lt_units": 1,
                "comparison_error_lt_units": 2,
                "message_count": values["messages"],
                "exact_once_count": values["exact"],
                "exact_once_ratio": exact_ratio,
                "retry_count": values["retry_count"],
                "attempt_count": values["attempts"],
                "retry_rate": retry_ratio,
                "semantic_conflict_count": values["conflicts"],
            }
        )

    topic_totals: dict[str, dict[str, int]] = {}
    for row in by_topic_group:
        topic = row["topic"]
        total = topic_totals.setdefault(
            topic,
            {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0},
        )
        total["messages"] += int(row["message_count"])
        total["exact"] += int(row["exact_once_count"])
        total["retry_count"] += int(row["retry_count"])
        total["attempts"] += int(row["attempt_count"])
        total["conflicts"] += int(row["semantic_conflict_count"])

    ranking: list[dict[str, Any]] = []
    for topic, values in sorted(topic_totals.items()):
        ratio = fixed_ratio(values["exact"], values["messages"], scale_factor)
        ranking.append(
            {
                "topic": topic,
                "numerator_exact": values["exact"],
                "denominator_messages": values["messages"],
                "fixed_exact_once_ratio": ratio["fixed"],
                "exact_once_ratio": ratio,
                "semantic_conflict_count": values["conflicts"],
                "tie_break": "topic_lexicographic_ascending",
            }
        )

    ranking.sort(key=cmp_to_key(_topic_compare))
    pairs = _affected_pairs(ranking)
    affected_topics = sorted({item["topic_a"] for item in pairs} | {item["topic_b"] for item in pairs})
    for index, row in enumerate(ranking, start=1):
        row["stable_rank"] = index
        row["error_affected"] = row["topic"] in affected_topics

    return {
        "scale_power": scale_power,
        "scale_factor": scale_factor,
        "truncation": "floor_toward_negative_infinity",
        "error_envelope": error_envelope(scale_power),
        "by_topic_group": by_topic_group,
        "topic_ranking": ranking,
        "comparison_affected_pairs": pairs,
    }
