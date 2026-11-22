"""Independent normalizer, semantics derivation, and statistics verifier.

This module deliberately does not import mwlab.records, mwlab.semantics or
mwlab.stats. It reconstructs conclusions directly from raw JSONL and compares
canonical bytes with the main pipeline outputs.
"""

from __future__ import annotations

import json
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, canonical_lines, read_json, sha256_bytes, sha256_obj
from .config import Declarations, load_declarations
from .errors import LabError, format_error, io_error, state_error


FORMAT_ALLOWED = {
    "v1": {"message_id", "topic", "acknowledged"},
    "v2": {
        "message_id",
        "topic",
        "acknowledged",
        "delivery_time_ms",
        "retry_count",
        "consumer_group",
    },
    "v3": {
        "message_id",
        "topic",
        "acknowledged",
        "delivery_time_ms",
        "retry_count",
        "consumer_group",
        "idempotency_key",
        "delivery_sequence_range",
        "semantic_verdict",
    },
}
VERDICTS = {"at_least_once", "at_most_once", "exactly_once"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(record: dict[str, Any], field: str, line_no: int, minimum: int | None = None) -> int:
    value = record.get(field)
    if not _is_int(value):
        raise format_error(line_no, f"record.{field}", "must be an integer", value=value)
    if minimum is not None and value < minimum:
        raise format_error(line_no, f"record.{field}", f"must be >= {minimum}", value=value)
    return value


def _normalize_records(raw_path: str | Path, declarations: Declarations) -> list[dict[str, Any]]:
    path = Path(raw_path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        raise io_error("record file not found", path=str(path))
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise format_error(line_no, "$", "invalid JSON", reason=exc.msg, column=exc.colno)
        if not isinstance(envelope, dict):
            raise format_error(line_no, "$", "record line must be an object")
        for field in ("format_version", "consumer_group", "record"):
            if field not in envelope:
                raise format_error(line_no, field, "required envelope field is missing")
        version = envelope["format_version"]
        if version not in FORMAT_ALLOWED:
            raise format_error(line_no, "format_version", "unknown format version", value=version)
        group_name = envelope["consumer_group"]
        if not isinstance(group_name, str) or group_name not in declarations.groups:
            raise format_error(line_no, "consumer_group", "consumer group is not declared", value=group_name)
        group = declarations.groups[group_name]
        if group.format_version != version:
            raise format_error(
                line_no,
                "format_version",
                "record version does not match consumer group declaration",
                value=version,
                expected=group.format_version,
            )
        payload = envelope["record"]
        if not isinstance(payload, dict):
            raise format_error(line_no, "record", "must be an object")
        allowed = FORMAT_ALLOWED[version]
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise format_error(
                line_no,
                f"record.{unknown[0]}",
                "field is not allowed in this format version",
                format_version=version,
            )
        for field in sorted(allowed):
            if field not in payload:
                raise format_error(line_no, f"record.{field}", "required field is missing")
        if not isinstance(payload["acknowledged"], bool):
            raise format_error(line_no, "record.acknowledged", "must be a boolean")
        topic_name = payload.get("topic")
        if not isinstance(topic_name, str) or topic_name not in declarations.topics:
            raise format_error(line_no, "record.topic", "topic is not declared", value=topic_name)
        topic = declarations.topics[topic_name]
        message_id = payload.get("message_id")
        if not isinstance(message_id, str):
            raise format_error(line_no, "record.message_id", "must be a string")
        seq = topic.sequence_for_id(message_id)
        if seq is None:
            raise format_error(
                line_no,
                "record.message_id",
                "message id does not satisfy the topic sequence rule",
                value=message_id,
            )
        delivery_time = None
        retry_count = 0
        idempotency_key = None
        range_start = None
        range_end = None
        semantic_verdict = None
        if version in {"v2", "v3"}:
            delivery_time = _require_int(payload, "delivery_time_ms", line_no, 0)
            retry_count = _require_int(payload, "retry_count", line_no, 0)
            payload_group = payload.get("consumer_group")
            if not isinstance(payload_group, str) or payload_group != group_name:
                raise format_error(
                    line_no,
                    "record.consumer_group",
                    "must match envelope consumer_group",
                    value=payload_group,
                    expected=group_name,
                )
        if version == "v3":
            idem = payload.get("idempotency_key")
            if not isinstance(idem, str) or not idem:
                raise format_error(line_no, "record.idempotency_key", "must be a non-empty string")
            idempotency_key = idem
            range_value = payload.get("delivery_sequence_range")
            if not isinstance(range_value, dict) or set(range_value) != {"start", "end"}:
                raise format_error(
                    line_no,
                    "record.delivery_sequence_range",
                    "must contain exactly start and end",
                )
            if not _is_int(range_value["start"]) or not _is_int(range_value["end"]):
                raise format_error(
                    line_no,
                    "record.delivery_sequence_range",
                    "start and end must be integers",
                    value=range_value,
                )
            range_start = int(range_value["start"])
            range_end = int(range_value["end"])
            if range_start < 0:
                raise format_error(
                    line_no,
                    "record.delivery_sequence_range.start",
                    "must be >= 0",
                    value=range_start,
                )
            if range_start > range_end:
                raise format_error(
                    line_no,
                    "record.delivery_sequence_range",
                    "start must be <= end",
                    start=range_start,
                    end=range_end,
                )
            semantic_verdict = payload.get("semantic_verdict")
            if semantic_verdict not in VERDICTS:
                raise format_error(
                    line_no,
                    "record.semantic_verdict",
                    "unknown semantic verdict",
                    value=semantic_verdict,
                )
        output.append(
            {
                "source_line": line_no,
                "source_format": version,
                "source_group": group_name,
                "message_id": message_id,
                "topic": topic_name,
                "message_seq": seq,
                "acknowledged": payload["acknowledged"],
                "delivery_time_ms": delivery_time,
                "retry_count": retry_count,
                "consumer_group": group_name,
                "idempotency_key": idempotency_key,
                "range_start": range_start,
                "range_end": range_end,
                "semantic_verdict": semantic_verdict,
            }
        )
    if not output:
        raise format_error(1, "$", "record file contains no records")
    return output


def _model_dict(record: dict[str, Any]) -> dict[str, Any]:
    has_range = record["range_start"] is not None and record["range_end"] is not None
    return {
        "acknowledged": record["acknowledged"],
        "consumer_group": record["consumer_group"],
        "delivery_sequence_range": [record["range_start"], record["range_end"]] if has_range else None,
        "delivery_time_ms": record["delivery_time_ms"],
        "idempotency_key": record["idempotency_key"],
        "message_id": record["message_id"],
        "message_seq": record["message_seq"],
        "retry_count": record["retry_count"],
        "semantic_verdict": record["semantic_verdict"],
        "source_format": record["source_format"],
        "source_group": record["source_group"],
        "topic": record["topic"],
    }


def _model_digest(records: Iterable[dict[str, Any]]) -> str:
    rows = [_model_dict(record) for record in records]
    rows.sort(key=lambda row: canonical_json(row))
    return sha256_obj({"schema_version": 1, "records": rows})


def _merge_ranges(ranges: list[list[int]], upper: int) -> list[list[int]]:
    clipped = [
        (max(0, start), min(upper, end))
        for start, end in ranges
        if end >= 0 and start <= upper
    ]
    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [[start, end] for start, end in merged]


def _missing_ranges(covered: list[list[int]], upper: int) -> list[list[int]]:
    result: list[list[int]] = []
    cursor = 0
    for start, end in covered:
        if cursor < start:
            result.append([cursor, start - 1])
        cursor = end + 1
    if cursor <= upper:
        result.append([cursor, upper])
    return result


def _cell(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    record_count = len(records)
    retry_max = max((int(record["retry_count"]) for record in records), default=0)
    attempts = max(record_count, retry_max + 1)
    upper = attempts - 1
    ranges = sorted(
        {
            (int(record["range_start"]), int(record["range_end"]))
            for record in records
            if record["range_start"] is not None and record["range_end"] is not None
        }
    )
    covered = _merge_ranges([[start, end] for start, end in ranges], upper)
    acknowledged = any(bool(record["acknowledged"]) for record in records)
    idem_keys = sorted({record["idempotency_key"] for record in records if record["idempotency_key"]})
    formats = sorted({record["source_format"] for record in records})
    markers = sorted({record["semantic_verdict"] for record in records if record["semantic_verdict"]})
    range_present = bool(ranges)
    uncovered = _missing_ranges(covered, upper) if range_present else []
    range_complete = range_present and not uncovered
    ack_ok = acknowledged or not policy["exactly_once_requires_ack"]
    idem_ok = bool(idem_keys) or not policy["exactly_once_requires_idempotency"]
    range_ok = range_complete or not policy["exactly_once_requires_complete_range"]
    if ack_ok and idem_ok and range_ok:
        verdict = "exactly_once"
        rule = "ack_ok AND idempotency_ok AND range_ok"
    elif acknowledged or attempts > 1 or uncovered:
        verdict = "at_least_once"
        rule = "acknowledged OR attempts>1 OR uncovered_range"
    else:
        verdict = "at_most_once"
        rule = "not acknowledged AND attempts==1 AND no uncovered range"
    marker_applied = False
    if policy.get("semantic_marker_authoritative") and len(markers) == 1:
        verdict = markers[0]
        marker_applied = True
        rule = "authoritative_v3_semantic_marker"
    return {
        "verdict": verdict,
        "rule": rule,
        "marker_applied": marker_applied,
        "basis": {
            "acknowledged": acknowledged,
            "attempts": attempts,
            "covered_intervals": covered,
            "delivery_ranges": [[start, end] for start, end in ranges],
            "expected_attempt_interval": [0, upper],
            "idempotency_keys": idem_keys,
            "range_complete": range_complete,
            "range_present": range_present,
            "record_count": record_count,
            "retry_count_max": retry_max,
            "semantic_markers": markers,
            "source_formats": formats,
            "uncovered_intervals": uncovered,
        },
    }


def _conflict(
    message_id: str,
    seq: int,
    topic: str,
    conflict_type: str,
    groups: Iterable[str],
    formats: Iterable[str],
    basis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "topic": topic,
        "message_seq": seq,
        "message_id": message_id,
        "conflict_type": conflict_type,
        "consumer_groups": sorted(set(groups)),
        "format_versions": sorted(set(formats)),
        "basis": basis,
    }


def _analyze(records: list[dict[str, Any]], declarations: Declarations) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        key = (record["topic"], int(record["message_seq"]), record["message_id"])
        grouped.setdefault(key, {}).setdefault(record["consumer_group"], []).append(record)
    group_conclusions: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    conflicts: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    def add_conflict(row: dict[str, Any]) -> None:
        key = (row["topic"], row["message_seq"], row["message_id"], row["conflict_type"])
        previous = conflicts.get(key)
        if previous is None:
            conflicts[key] = row
            return
        previous["consumer_groups"] = sorted(set(previous["consumer_groups"]) | set(row["consumer_groups"]))
        previous["format_versions"] = sorted(set(previous["format_versions"]) | set(row["format_versions"]))
        previous["basis"].setdefault("merged_group_causes", []).append(row["basis"])
        previous["basis"]["merged_group_causes"].sort(
            key=lambda item: (item.get("consumer_group", ""), canonical_json(item))
        )

    for topic, seq, message_id in sorted(grouped):
        topic_policy = declarations.topics[topic].policy
        per_group = grouped[(topic, seq, message_id)]
        cells: list[dict[str, Any]] = []
        for group in sorted(per_group):
            derived = _cell(per_group[group], topic_policy)
            row = {
                "topic": topic,
                "message_seq": seq,
                "message_id": message_id,
                "consumer_group": group,
                "verdict": derived["verdict"],
                "basis": derived["basis"],
                "marker_applied": derived["marker_applied"],
                "rule": derived["rule"],
            }
            cells.append(row)
            group_conclusions.append(row)
            facts.append(
                {
                    "topic": topic,
                    "message_seq": seq,
                    "message_id": message_id,
                    "consumer_group": group,
                    "verdict": derived["verdict"],
                    "retry_count_max": derived["basis"]["retry_count_max"],
                    "attempts": derived["basis"]["attempts"],
                }
            )
        verdicts = [cell["verdict"] for cell in cells]
        if all(verdict == "exactly_once" for verdict in verdicts):
            global_verdict = "exactly_once"
            global_rule = "all_consumer_groups_exactly_once"
        elif all(verdict == "at_most_once" for verdict in verdicts):
            global_verdict = "at_most_once"
            global_rule = "all_consumer_groups_at_most_once"
        else:
            global_verdict = "at_least_once"
            global_rule = "one_or_more_groups_at_least_once_or_mixed_verdicts"
        ranges = sorted(
            {tuple(value) for cell in cells for value in cell["basis"]["delivery_ranges"]}
        )
        conclusions.append(
            {
                "topic": topic,
                "message_seq": seq,
                "message_id": message_id,
                "verdict": global_verdict,
                "basis": {
                    "consumer_group_verdicts": [
                        {"consumer_group": cell["consumer_group"], "verdict": cell["verdict"]}
                        for cell in sorted(cells, key=lambda item: item["consumer_group"])
                    ],
                    "delivery_ranges": [[start, end] for start, end in ranges],
                    "idempotency_keys": sorted(
                        {key for cell in cells for key in cell["basis"]["idempotency_keys"]}
                    ),
                    "retry_count_max": max(cell["basis"]["retry_count_max"] for cell in cells),
                    "attempts_max": max(cell["basis"]["attempts"] for cell in cells),
                    "source_formats": sorted(
                        {fmt for cell in cells for fmt in cell["basis"]["source_formats"]}
                    ),
                    "source_groups": sorted(cell["consumer_group"] for cell in cells),
                    "uncovered_group_count": sum(
                        bool(cell["basis"]["uncovered_intervals"]) for cell in cells
                    ),
                },
                "rule": global_rule,
            }
        )

        uncovered_causes = []
        retry_causes = []
        marker_causes = []
        for cell in cells:
            basis = cell["basis"]
            if (
                basis["acknowledged"]
                and basis["range_present"]
                and basis["uncovered_intervals"]
                and not topic_policy["allow_ack_with_uncovered_range"]
            ):
                uncovered_causes.append(
                    {
                        "consumer_group": cell["consumer_group"],
                        "delivery_ranges": basis["delivery_ranges"],
                        "expected_attempt_interval": basis["expected_attempt_interval"],
                        "source_formats": basis["source_formats"],
                        "uncovered_intervals": basis["uncovered_intervals"],
                    }
                )
            if basis["retry_count_max"] > topic_policy["max_allowed_retries"]:
                retry_causes.append(
                    {
                        "consumer_group": cell["consumer_group"],
                        "max_allowed_retries": topic_policy["max_allowed_retries"],
                        "retry_count_max": basis["retry_count_max"],
                        "source_formats": basis["source_formats"],
                    }
                )
            for marker in basis["semantic_markers"]:
                if marker != cell["verdict"]:
                    marker_causes.append(
                        {
                            "consumer_group": cell["consumer_group"],
                            "derived_verdict": cell["verdict"],
                            "semantic_marker": marker,
                            "source_formats": basis["source_formats"],
                        }
                    )
        if uncovered_causes:
            add_conflict(
                _conflict(
                    message_id,
                    seq,
                    topic,
                    "ACK_WITH_UNCOVERED_RANGE",
                    [cause["consumer_group"] for cause in uncovered_causes],
                    [fmt for cause in uncovered_causes for fmt in cause["source_formats"]],
                    {
                        "rule": "acknowledged and delivery range does not cover every expected attempt",
                        "causes": sorted(uncovered_causes, key=lambda item: item["consumer_group"]),
                    },
                )
            )
        if retry_causes:
            add_conflict(
                _conflict(
                    message_id,
                    seq,
                    topic,
                    "RETRY_LIMIT_EXCEEDED",
                    [cause["consumer_group"] for cause in retry_causes],
                    [fmt for cause in retry_causes for fmt in cause["source_formats"]],
                    {
                        "rule": "retry_count_max exceeds topic confirmation policy",
                        "causes": sorted(retry_causes, key=lambda item: item["consumer_group"]),
                    },
                )
            )
        if marker_causes:
            add_conflict(
                _conflict(
                    message_id,
                    seq,
                    topic,
                    "SEMANTIC_MARKER_MISMATCH",
                    [cause["consumer_group"] for cause in marker_causes],
                    [fmt for cause in marker_causes for fmt in cause["source_formats"]],
                    {
                        "rule": "v3 semantic verdict marker disagrees with derived group verdict",
                        "causes": sorted(
                            marker_causes,
                            key=lambda item: (item["consumer_group"], item["semantic_marker"]),
                        ),
                    },
                )
            )

    key_occurrences: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record["idempotency_key"]
        if key:
            key_occurrences.setdefault(key, []).append(
                {
                    "topic": record["topic"],
                    "message_seq": record["message_seq"],
                    "message_id": record["message_id"],
                    "consumer_group": record["consumer_group"],
                    "source_format": record["source_format"],
                }
            )
    for idem_key in sorted(key_occurrences):
        occurrences = sorted(
            key_occurrences[idem_key],
            key=lambda item: (
                item["consumer_group"],
                item["topic"],
                item["message_seq"],
                item["message_id"],
                item["source_format"],
            ),
        )
        groups = sorted({item["consumer_group"] for item in occurrences})
        if len(groups) < 2:
            continue
        messages: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for occurrence in occurrences:
            key = (occurrence["topic"], occurrence["message_seq"], occurrence["message_id"])
            messages.setdefault(key, []).append(occurrence)
        for topic, seq, message_id in sorted(messages):
            add_conflict(
                _conflict(
                    message_id,
                    seq,
                    topic,
                    "DUP_IDEMPOTENCY_ACROSS_GROUPS",
                    groups,
                    [item["source_format"] for item in occurrences],
                    {
                        "rule": "one idempotency key appears in more than one consumer group",
                        "idempotency_key": idem_key,
                        "occurrences": occurrences,
                    },
                )
            )

    group_conclusions.sort(
        key=lambda row: (row["topic"], row["message_seq"], row["consumer_group"], row["message_id"])
    )
    conclusions.sort(key=lambda row: (row["topic"], row["message_seq"], row["message_id"]))
    conflict_rows = sorted(
        conflicts.values(),
        key=lambda row: (row["message_seq"], row["conflict_type"], row["topic"], row["message_id"]),
    )
    stats = _stats(facts, group_conclusions, conflict_rows, declarations.session.scale_power)
    return {
        "conclusions": conclusions,
        "group_conclusions": group_conclusions,
        "conflicts": conflict_rows,
        "stats": stats,
    }


def _fixed(numerator: int, denominator: int, scale: int) -> dict[str, Any]:
    if denominator == 0:
        return {
            "defined": False,
            "numerator": numerator,
            "denominator": denominator,
            "fixed": 0,
            "exact_ratio": None,
        }
    fixed = (numerator * scale) // denominator
    return {
        "defined": True,
        "numerator": numerator,
        "denominator": denominator,
        "fixed": fixed,
        "exact_ratio": f"{numerator}/{denominator}",
    }


def _stats(
    facts: list[dict[str, Any]],
    conclusions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    scale_power: int,
) -> dict[str, Any]:
    scale = 10 ** scale_power
    cells: dict[tuple[str, str], dict[str, int]] = {}
    for row in conclusions:
        key = (row["topic"], row["consumer_group"])
        cell = cells.setdefault(
            key, {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0}
        )
        cell["messages"] += 1
        cell["exact"] += 1 if row["verdict"] == "exactly_once" else 0
    for row in facts:
        key = (row["topic"], row["consumer_group"])
        cell = cells.setdefault(
            key, {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0}
        )
        cell["retry_count"] += row["retry_count_max"]
        cell["attempts"] += row["attempts"]
    for conflict in conflicts:
        for group in conflict["consumer_groups"]:
            key = (conflict["topic"], group)
            cell = cells.setdefault(
                key, {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0}
            )
            cell["conflicts"] += 1
    by_topic_group = []
    for (topic, group), values in sorted(cells.items()):
        by_topic_group.append(
            {
                "topic": topic,
                "consumer_group": group,
                "scale_factor": scale,
                "single_error_lt_units": 1,
                "comparison_error_lt_units": 2,
                "message_count": values["messages"],
                "exact_once_count": values["exact"],
                "exact_once_ratio": _fixed(values["exact"], values["messages"], scale),
                "retry_count": values["retry_count"],
                "attempt_count": values["attempts"],
                "retry_rate": _fixed(values["retry_count"], values["attempts"], scale),
                "semantic_conflict_count": values["conflicts"],
            }
        )
    topic_totals: dict[str, dict[str, int]] = {}
    for row in by_topic_group:
        total = topic_totals.setdefault(
            row["topic"], {"messages": 0, "exact": 0, "retry_count": 0, "attempts": 0, "conflicts": 0}
        )
        total["messages"] += row["message_count"]
        total["exact"] += row["exact_once_count"]
        total["retry_count"] += row["retry_count"]
        total["attempts"] += row["attempt_count"]
        total["conflicts"] += row["semantic_conflict_count"]
    ranking = []
    for topic, values in sorted(topic_totals.items()):
        ratio = _fixed(values["exact"], values["messages"], scale)
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

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        difference = left["fixed_exact_once_ratio"] - right["fixed_exact_once_ratio"]
        if abs(difference) <= 2:
            return -1 if left["topic"] < right["topic"] else (1 if left["topic"] > right["topic"] else 0)
        return -1 if difference > 0 else 1

    ranking.sort(key=cmp_to_key(compare))
    affected_pairs = []
    for index, left in enumerate(ranking):
        for right in ranking[index + 1 :]:
            difference = abs(left["fixed_exact_once_ratio"] - right["fixed_exact_once_ratio"])
            if difference <= 2:
                topic_a, topic_b = sorted((left["topic"], right["topic"]))
                affected_pairs.append(
                    {
                        "topic_a": topic_a,
                        "topic_b": topic_b,
                        "fixed_difference_units": difference,
                        "comparison_bound_units": 2,
                    }
                )
    affected_pairs.sort(key=lambda item: (item["topic_a"], item["topic_b"]))
    affected_topics = sorted(
        {item["topic_a"] for item in affected_pairs} | {item["topic_b"] for item in affected_pairs}
    )
    for index, row in enumerate(ranking, start=1):
        row["stable_rank"] = index
        row["error_affected"] = row["topic"] in affected_topics
    return {
        "scale_power": scale_power,
        "scale_factor": scale,
        "truncation": "floor_toward_negative_infinity",
        "error_envelope": {
            "scale_power": scale_power,
            "scale_factor": scale,
            "truncation": "floor_toward_negative_infinity",
            "non_negative_equivalent": "truncate_toward_zero",
            "single_value": {
                "fixed_units_strictly_less_than": 1,
                "ratio_strictly_less_than": f"1/{scale}",
                "formula": "0 <= exact*scale - fixed < 1",
            },
            "ratio_comparison": {
                "fixed_units_strictly_less_than": 2,
                "ratio_strictly_less_than": f"2/{scale}",
                "formula": "abs((a-b)*scale) < 2 for two floored values",
                "decision_rule": "if abs(fixed_a-fixed_b) <= 2 mark affected and use topic lexical tie-break",
            },
        },
        "by_topic_group": by_topic_group,
        "topic_ranking": ranking,
        "comparison_affected_pairs": affected_pairs,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise io_error("comparison file not found", path=str(path))
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise state_error("comparison JSONL is invalid", path=str(path), line=line_no, reason=exc.msg)
        if not isinstance(value, dict):
            raise state_error("comparison JSONL row is not an object", path=str(path), line=line_no)
        rows.append(value)
    return rows


def _first_row_diff(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    section: str,
) -> dict[str, Any] | None:
    def index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            result[key] = row
        return result

    left = index(expected)
    right = index(actual)
    for key in sorted(set(left) | set(right), key=lambda item: tuple(str(part) for part in item)):
        before = left.get(key)
        after = right.get(key)
        if before is None or after is None:
            row = before or after or {}
            return {
                "section": section,
                "message_id": row.get("message_id", "<non-message-row>"),
                "field": "row",
                "left": before,
                "right": after,
            }
        if canonical_json(before) != canonical_json(after):
            for field in sorted(set(before) | set(after)):
                if canonical_json(before.get(field)) != canonical_json(after.get(field)):
                    return {
                        "section": section,
                        "message_id": before.get("message_id", "<non-message-row>"),
                        "field": field,
                        "left": before.get(field),
                        "right": after.get(field),
                    }
    return None


def verify(
    config_path: str | Path,
    records_path: str | Path,
    model_digest_path: str | Path,
    conclusions_path: str | Path,
    group_conclusions_path: str | Path,
    conflicts_path: str | Path,
    stats_path: str | Path,
    *,
    inject_divergence: bool = False,
) -> dict[str, Any]:
    declarations = load_declarations(config_path)
    records = _normalize_records(records_path, declarations)
    analysis = _analyze(records, declarations)
    if inject_divergence and analysis["conclusions"]:
        first = analysis["conclusions"][0]
        first["verdict"] = "at_most_once" if first["verdict"] != "at_most_once" else "exactly_once"
    independent_digest = _model_digest(records)
    try:
        expected_digest = Path(model_digest_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise io_error("model digest file not found", path=str(model_digest_path))
    expected_conclusions = _read_jsonl(conclusions_path)
    expected_groups = _read_jsonl(group_conclusions_path)
    expected_conflicts = _read_jsonl(conflicts_path)
    expected_conclusion_bytes = Path(conclusions_path).read_bytes()
    expected_group_bytes = Path(group_conclusions_path).read_bytes()
    expected_conflict_bytes = Path(conflicts_path).read_bytes()
    try:
        expected_stats_bytes = Path(stats_path).read_bytes()
    except FileNotFoundError:
        raise io_error("stats file not found", path=str(stats_path))
    try:
        expected_stats = read_json(stats_path)
    except LabError as exc:
        if exc.code == "E_IO":
            raise
        raise state_error("stats artifact is not valid JSON", path=str(stats_path))

    origin = "format_normalization" if expected_digest != independent_digest else "semantic_derivation"
    mismatch = None
    if expected_digest != independent_digest:
        first_message_id = (
            expected_conclusions[0].get("message_id", "<normalized-model>")
            if expected_conclusions
            else "<normalized-model>"
        )
        mismatch = {
            "section": "model_digest",
            "message_id": first_message_id,
            "field": "model_digest",
            "left": expected_digest,
            "right": independent_digest,
        }
    mismatch = mismatch or _first_row_diff(
        expected_conclusions,
        analysis["conclusions"],
        key_fields=("topic", "message_seq", "message_id"),
        section="conclusions",
    )
    mismatch = mismatch or _first_row_diff(
        expected_groups,
        analysis["group_conclusions"],
        key_fields=("topic", "message_seq", "message_id", "consumer_group"),
        section="group_conclusions",
    )
    mismatch = mismatch or _first_row_diff(
        expected_conflicts,
        analysis["conflicts"],
        key_fields=("topic", "message_seq", "message_id", "conflict_type"),
        section="conflicts",
    )
    if mismatch is None and canonical_json(expected_stats) != canonical_json(analysis["stats"]):
        for field in sorted(set(expected_stats) | set(analysis["stats"])):
            if canonical_json(expected_stats.get(field)) != canonical_json(analysis["stats"].get(field)):
                mismatch = {
                    "section": "stats",
                    "message_id": "<statistics>",
                    "field": field,
                    "left": expected_stats.get(field),
                    "right": analysis["stats"].get(field),
                }
                break
    actual_conclusion_bytes = canonical_lines(analysis["conclusions"])
    actual_group_bytes = canonical_lines(analysis["group_conclusions"])
    actual_conflict_bytes = canonical_lines(analysis["conflicts"])
    actual_stats_bytes = canonical_json(analysis["stats"]).encode("utf-8") + b"\n"
    if mismatch is None:
        byte_checks = (
            ("conclusions", expected_conclusion_bytes, actual_conclusion_bytes),
            ("group_conclusions", expected_group_bytes, actual_group_bytes),
            ("conflicts", expected_conflict_bytes, actual_conflict_bytes),
            ("stats", expected_stats_bytes, actual_stats_bytes),
        )
        for section, expected_bytes, actual_bytes in byte_checks:
            if expected_bytes != actual_bytes:
                mismatch = {
                    "section": section,
                    "message_id": (
                        expected_conclusions[0].get("message_id", "<bytes>")
                        if expected_conclusions
                        else "<bytes>"
                    ),
                    "field": "canonical_bytes",
                    "left": {"sha256": sha256_bytes(expected_bytes), "length": len(expected_bytes)},
                    "right": {"sha256": sha256_bytes(actual_bytes), "length": len(actual_bytes)},
                }
                break
    if mismatch is not None:
        raise LabError(
            "E_DIVERGE",
            "independent replay diverged from main pipeline",
            {
                "first_mismatch": mismatch,
                "origin": origin,
                "model_digest": {
                    "main": expected_digest,
                    "independent": independent_digest,
                },
            },
        )
    return {
        "status": "ok",
        "origin": "none",
        "record_count": len(records),
        "conclusion_count": len(analysis["conclusions"]),
        "model_digest": independent_digest,
        "conclusions_sha256": sha256_obj(analysis["conclusions"]),
        "conclusions_bytes": len(actual_conclusion_bytes),
        "byte_identical": True,
        "stats_byte_identical": True,
        "conflicts_byte_identical": True,
    }
