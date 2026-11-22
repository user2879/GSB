"""Deterministic delivery-semantics derivation and conflict attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import Declarations, TopicConfig
from .records import NormalizedRecord
from .stats import build_stats


@dataclass
class _Cell:
    topic: TopicConfig
    message_id: str
    message_seq: int
    consumer_group: str
    records: list[NormalizedRecord]

    @property
    def sorted_records(self) -> list[NormalizedRecord]:
        return sorted(
            self.records,
            key=lambda record: (
                record.source_format,
                record.message_id,
                record.consumer_group,
                record.retry_count,
                record.idempotency_key or "",
                record.range_start if record.range_start is not None else -1,
                record.range_end if record.range_end is not None else -1,
                record.acknowledged,
            ),
        )

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def retry_count_max(self) -> int:
        return max((record.retry_count for record in self.records), default=0)

    @property
    def attempts(self) -> int:
        return max(self.record_count, self.retry_count_max + 1)

    @property
    def acknowledged(self) -> bool:
        return any(record.acknowledged for record in self.records)

    @property
    def idempotency_keys(self) -> list[str]:
        return sorted({record.idempotency_key for record in self.records if record.idempotency_key})

    @property
    def ranges(self) -> list[list[int]]:
        values = {
            (int(record.range_start), int(record.range_end))
            for record in self.records
            if record.range_present
        }
        return [[start, end] for start, end in sorted(values)]

    @property
    def source_formats(self) -> list[str]:
        return sorted({record.source_format for record in self.records})

    @property
    def semantic_markers(self) -> list[str]:
        return sorted({record.semantic_verdict for record in self.records if record.semantic_verdict})


def _merge_intervals(intervals: Iterable[tuple[int, int]], upper: int) -> list[list[int]]:
    clipped: list[tuple[int, int]] = []
    for start, end in intervals:
        if end < 0 or start > upper:
            continue
        clipped.append((max(0, start), min(upper, end)))
    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [[start, end] for start, end in merged]


def _uncovered_intervals(covered: list[list[int]], upper: int) -> list[list[int]]:
    if upper < 0:
        return []
    uncovered: list[list[int]] = []
    cursor = 0
    for start, end in covered:
        if cursor < start:
            uncovered.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= upper:
        uncovered.append([cursor, upper])
    return uncovered


def _derive_cell(cell: _Cell) -> dict[str, Any]:
    policy = cell.topic.policy
    upper = cell.attempts - 1
    covered = _merge_intervals(
        ((start, end) for start, end in cell.ranges),
        upper,
    )
    range_present = bool(cell.ranges)
    uncovered = _uncovered_intervals(covered, upper) if range_present else []
    range_complete = range_present and not uncovered
    ack_ok = cell.acknowledged or not bool(policy["exactly_once_requires_ack"])
    idem_ok = bool(cell.idempotency_keys) or not bool(policy["exactly_once_requires_idempotency"])
    range_ok = range_complete or not bool(policy["exactly_once_requires_complete_range"])

    if ack_ok and idem_ok and range_ok:
        verdict = "exactly_once"
        rule = "ack_ok AND idempotency_ok AND range_ok"
    elif cell.acknowledged or cell.attempts > 1 or bool(uncovered):
        verdict = "at_least_once"
        rule = "acknowledged OR attempts>1 OR uncovered_range"
    else:
        verdict = "at_most_once"
        rule = "not acknowledged AND attempts==1 AND no uncovered range"

    marker_applied = False
    if bool(policy["semantic_marker_authoritative"]) and len(cell.semantic_markers) == 1:
        verdict = cell.semantic_markers[0]
        marker_applied = True
        rule = "authoritative_v3_semantic_marker"

    basis = {
        "acknowledged": cell.acknowledged,
        "attempts": cell.attempts,
        "covered_intervals": covered,
        "delivery_ranges": cell.ranges,
        "expected_attempt_interval": [0, upper],
        "idempotency_keys": cell.idempotency_keys,
        "range_complete": range_complete,
        "range_present": range_present,
        "record_count": cell.record_count,
        "retry_count_max": cell.retry_count_max,
        "semantic_markers": cell.semantic_markers,
        "source_formats": cell.source_formats,
        "uncovered_intervals": uncovered,
    }
    return {
        "topic": cell.topic.name,
        "message_seq": cell.message_seq,
        "message_id": cell.message_id,
        "consumer_group": cell.consumer_group,
        "verdict": verdict,
        "basis": basis,
        "marker_applied": marker_applied,
        "rule": rule,
    }


def _derive_global(message_id: str, seq: int, topic: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [cell["verdict"] for cell in cells]
    if all(verdict == "exactly_once" for verdict in verdicts):
        verdict = "exactly_once"
        rule = "all_consumer_groups_exactly_once"
    elif all(verdict == "at_most_once" for verdict in verdicts):
        verdict = "at_most_once"
        rule = "all_consumer_groups_at_most_once"
    else:
        verdict = "at_least_once"
        rule = "one_or_more_groups_at_least_once_or_mixed_verdicts"

    group_verdicts = [
        {"consumer_group": cell["consumer_group"], "verdict": cell["verdict"]}
        for cell in sorted(cells, key=lambda item: item["consumer_group"])
    ]
    idempotency_keys = sorted(
        {key for cell in cells for key in cell["basis"]["idempotency_keys"]}
    )
    ranges = sorted({tuple(value) for cell in cells for value in cell["basis"]["delivery_ranges"]})
    basis = {
        "consumer_group_verdicts": group_verdicts,
        "delivery_ranges": [[start, end] for start, end in ranges],
        "idempotency_keys": idempotency_keys,
        "retry_count_max": max((cell["basis"]["retry_count_max"] for cell in cells), default=0),
        "attempts_max": max((cell["basis"]["attempts"] for cell in cells), default=0),
        "source_formats": sorted({fmt for cell in cells for fmt in cell["basis"]["source_formats"]}),
        "source_groups": sorted(cell["consumer_group"] for cell in cells),
        "uncovered_group_count": sum(bool(cell["basis"]["uncovered_intervals"]) for cell in cells),
    }
    return {
        "topic": topic,
        "message_seq": seq,
        "message_id": message_id,
        "verdict": verdict,
        "basis": basis,
        "rule": rule,
    }


def _conflict_row(
    *,
    topic: str,
    message_seq: int,
    message_id: str,
    conflict_type: str,
    consumer_groups: Iterable[str],
    format_versions: Iterable[str],
    basis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "topic": topic,
        "message_seq": message_seq,
        "message_id": message_id,
        "conflict_type": conflict_type,
        "consumer_groups": sorted(set(consumer_groups)),
        "format_versions": sorted(set(format_versions)),
        "basis": basis,
    }


def analyze(records: Iterable[NormalizedRecord], declarations: Declarations) -> dict[str, Any]:
    records = list(records)
    cells_by_message: dict[tuple[str, int, str], dict[str, _Cell]] = {}
    for record in records:
        topic = declarations.topics[record.topic]
        key = (record.topic, record.message_seq, record.message_id)
        groups = cells_by_message.setdefault(key, {})
        group_cell = groups.get(record.consumer_group)
        if group_cell is None:
            group_cell = _Cell(
                topic=topic,
                message_id=record.message_id,
                message_seq=record.message_seq,
                consumer_group=record.consumer_group,
                records=[],
            )
            groups[record.consumer_group] = group_cell
        group_cell.records.append(record)

    group_conclusions: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    group_facts: list[dict[str, Any]] = []
    for key in sorted(cells_by_message, key=lambda item: (item[0], item[1], item[2])):
        topic, seq, message_id = key
        cells = cells_by_message[key]
        derived_cells = [_derive_cell(cells[group]) for group in sorted(cells)]
        group_conclusions.extend(derived_cells)
        for cell in derived_cells:
            group_facts.append(
                {
                    "topic": topic,
                    "message_seq": seq,
                    "message_id": message_id,
                    "consumer_group": cell["consumer_group"],
                    "verdict": cell["verdict"],
                    "retry_count_max": cell["basis"]["retry_count_max"],
                    "attempts": cell["basis"]["attempts"],
                }
            )
        conclusions.append(_derive_global(message_id, seq, topic, derived_cells))

    group_conclusions.sort(
        key=lambda row: (row["topic"], row["message_seq"], row["consumer_group"], row["message_id"])
    )
    conclusions.sort(key=lambda row: (row["topic"], row["message_seq"], row["message_id"]))

    conflicts: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    def add_conflict(row: dict[str, Any]) -> None:
        key = (row["topic"], row["message_seq"], row["message_id"], row["conflict_type"])
        existing = conflicts.get(key)
        if existing is None:
            conflicts[key] = row
            return
        existing["consumer_groups"] = sorted(
            set(existing["consumer_groups"]) | set(row["consumer_groups"])
        )
        existing["format_versions"] = sorted(
            set(existing["format_versions"]) | set(row["format_versions"])
        )
        existing["basis"]["merged_group_causes"] = sorted(
            existing["basis"].get("merged_group_causes", []) + [row["basis"]],
            key=lambda item: (item.get("consumer_group", ""), str(item)),
        )

    for key in sorted(cells_by_message, key=lambda item: (item[0], item[1], item[2])):
        topic, seq, message_id = key
        cells = cells_by_message[key]
        topic_config = declarations.topics[topic]
        uncovered_causes = []
        retry_causes = []
        marker_causes = []
        for group in sorted(cells):
            cell = cells[group]
            derived = _derive_cell(cell)
            basis = derived["basis"]
            if (
                basis["acknowledged"]
                and basis["range_present"]
                and basis["uncovered_intervals"]
                and not bool(topic_config.policy["allow_ack_with_uncovered_range"])
            ):
                uncovered_causes.append(
                    {
                        "consumer_group": group,
                        "delivery_ranges": basis["delivery_ranges"],
                        "expected_attempt_interval": basis["expected_attempt_interval"],
                        "source_formats": basis["source_formats"],
                        "uncovered_intervals": basis["uncovered_intervals"],
                    }
                )
            if basis["retry_count_max"] > int(topic_config.policy["max_allowed_retries"]):
                retry_causes.append(
                    {
                        "consumer_group": group,
                        "max_allowed_retries": int(topic_config.policy["max_allowed_retries"]),
                        "retry_count_max": basis["retry_count_max"],
                        "source_formats": basis["source_formats"],
                    }
                )
            for marker in basis["semantic_markers"]:
                if marker != derived["verdict"]:
                    marker_causes.append(
                        {
                            "consumer_group": group,
                            "derived_verdict": derived["verdict"],
                            "semantic_marker": marker,
                            "source_formats": basis["source_formats"],
                        }
                    )
        if uncovered_causes:
            add_conflict(
                _conflict_row(
                    topic=topic,
                    message_seq=seq,
                    message_id=message_id,
                    conflict_type="ACK_WITH_UNCOVERED_RANGE",
                    consumer_groups=[cause["consumer_group"] for cause in uncovered_causes],
                    format_versions=[fmt for cause in uncovered_causes for fmt in cause["source_formats"]],
                    basis={
                        "rule": "acknowledged and delivery range does not cover every expected attempt",
                        "causes": sorted(uncovered_causes, key=lambda item: item["consumer_group"]),
                    },
                )
            )
        if retry_causes:
            add_conflict(
                _conflict_row(
                    topic=topic,
                    message_seq=seq,
                    message_id=message_id,
                    conflict_type="RETRY_LIMIT_EXCEEDED",
                    consumer_groups=[cause["consumer_group"] for cause in retry_causes],
                    format_versions=[fmt for cause in retry_causes for fmt in cause["source_formats"]],
                    basis={
                        "rule": "retry_count_max exceeds topic confirmation policy",
                        "causes": sorted(retry_causes, key=lambda item: item["consumer_group"]),
                    },
                )
            )
        if marker_causes:
            add_conflict(
                _conflict_row(
                    topic=topic,
                    message_seq=seq,
                    message_id=message_id,
                    conflict_type="SEMANTIC_MARKER_MISMATCH",
                    consumer_groups=[cause["consumer_group"] for cause in marker_causes],
                    format_versions=[fmt for cause in marker_causes for fmt in cause["source_formats"]],
                    basis={
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
        if record.idempotency_key:
            key_occurrences.setdefault(record.idempotency_key, []).append(
                {
                    "topic": record.topic,
                    "message_seq": record.message_seq,
                    "message_id": record.message_id,
                    "consumer_group": record.consumer_group,
                    "source_format": record.source_format,
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
                _conflict_row(
                    topic=topic,
                    message_seq=seq,
                    message_id=message_id,
                    conflict_type="DUP_IDEMPOTENCY_ACROSS_GROUPS",
                    consumer_groups=groups,
                    format_versions=[item["source_format"] for item in occurrences],
                    basis={
                        "rule": "one idempotency key appears in more than one consumer group",
                        "idempotency_key": idem_key,
                        "occurrences": occurrences,
                    },
                )
            )

    conflict_rows = sorted(
        conflicts.values(),
        key=lambda row: (
            row["message_seq"],
            row["conflict_type"],
            row["topic"],
            row["message_id"],
        ),
    )
    stats = build_stats(
        group_facts,
        group_conclusions,
        conflict_rows,
        scale_power=declarations.session.scale_power,
    )
    format_counts: dict[str, int] = {version: 0 for version in ("v1", "v2", "v3")}
    for record in records:
        format_counts[record.source_format] += 1
    verdict_counts: dict[str, int] = {verdict: 0 for verdict in ("at_least_once", "at_most_once", "exactly_once")}
    for row in conclusions:
        verdict_counts[row["verdict"]] += 1
    conflict_counts: dict[str, int] = {}
    for row in conflict_rows:
        conflict_counts[row["conflict_type"]] = conflict_counts.get(row["conflict_type"], 0) + 1
    conflict_counts = dict(sorted(conflict_counts.items()))

    return {
        "conclusions": conclusions,
        "group_conclusions": group_conclusions,
        "conflicts": conflict_rows,
        "stats": stats,
        "group_facts": sorted(
            group_facts,
            key=lambda row: (row["topic"], row["message_seq"], row["consumer_group"], row["message_id"]),
        ),
        "record_count": len(records),
        "format_counts": format_counts,
        "verdict_counts": verdict_counts,
        "conflict_counts": conflict_counts,
    }
