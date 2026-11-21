"""Deterministic JSONL evidence loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ERROR_DUPLICATE_EVIDENCE,
    ERROR_EVIDENCE_TYPE_UNKNOWN,
    ERROR_JSON_INVALID,
    ERROR_ROLE_UNKNOWN,
    ERROR_TIMESTAMP_INVALID,
)
from .model import Evidence, EvidenceLoadResult, Rules


def _failure(
    code: str,
    message: str,
    *,
    ticket_id: str,
    line_no: int,
    evidence_seq: int | None = None,
) -> dict:
    return {
        "ticket_id": ticket_id,
        "line_no": line_no,
        "evidence_seq": evidence_seq,
        "stage": "load",
        "code": code,
        "exception_type": None,
        "message": message,
    }


def _parse_jsonl_lines(lines: Iterable[str]) -> tuple[list[tuple[int, Any]], list[dict]]:
    records: list[tuple[int, Any]] = []
    failures: list[dict] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            failures.append(
                _failure(
                    ERROR_JSON_INVALID,
                    "blank evidence line is not valid JSON",
                    ticket_id=f"@line{line_no:06d}",
                    line_no=line_no,
                )
            )
            continue
        try:
            records.append((line_no, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            failures.append(
                _failure(
                    ERROR_JSON_INVALID,
                    f"invalid JSON: {exc.msg}",
                    ticket_id=f"@line{line_no:06d}",
                    line_no=line_no,
                )
            )
    return records, failures


def validate_evidence_records(parsed_records: list[tuple[int, Any]], rules: Rules) -> EvidenceLoadResult:
    failures: list[dict] = []
    rows_by_ticket: dict[str, list[Evidence]] = {}
    ticket_bad: set[str] = set()
    seen_pairs: dict[tuple[str, int], int] = {}

    for line_no, raw in parsed_records:
        if not isinstance(raw, dict):
            failures.append(
                _failure(
                    ERROR_JSON_INVALID,
                    "evidence record must be a JSON object",
                    ticket_id=f"@line{line_no:06d}",
                    line_no=line_no,
                )
            )
            continue

        ticket_raw = raw.get("ticket_id")
        seq_raw = raw.get("evidence_seq")
        ticket_id = ticket_raw if isinstance(ticket_raw, str) and ticket_raw else f"@line{line_no:06d}"
        seq_for_failure = seq_raw if isinstance(seq_raw, int) and not isinstance(seq_raw, bool) else None

        row_errors: list[tuple[str, str]] = []
        if not isinstance(ticket_raw, str) or not ticket_raw:
            row_errors.append((ERROR_JSON_INVALID, "ticket_id must be a non-empty string"))
        if isinstance(seq_raw, bool) or not isinstance(seq_raw, int) or seq_raw < 0:
            row_errors.append((ERROR_JSON_INVALID, "evidence_seq must be a non-negative integer"))
        evidence_type = raw.get("evidence_type")
        if not isinstance(evidence_type, str) or not evidence_type:
            row_errors.append((ERROR_JSON_INVALID, "evidence_type must be a non-empty string"))
        source_role = raw.get("source_role")
        if not isinstance(source_role, str) or not source_role:
            row_errors.append((ERROR_JSON_INVALID, "source_role must be a non-empty string"))
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            row_errors.append((ERROR_TIMESTAMP_INVALID, "timestamp must be a non-negative integer"))
        terms = raw.get("terms")
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            row_errors.append((ERROR_JSON_INVALID, "terms must be an array of strings"))

        if isinstance(evidence_type, str) and evidence_type and evidence_type not in rules.weights:
            row_errors.append((ERROR_EVIDENCE_TYPE_UNKNOWN, f"unknown evidence type {evidence_type!r}"))
        if isinstance(source_role, str) and source_role and source_role not in rules.trusts:
            row_errors.append((ERROR_ROLE_UNKNOWN, f"unknown source role {source_role!r}"))

        key = None
        if isinstance(ticket_raw, str) and ticket_raw and isinstance(seq_raw, int) and not isinstance(seq_raw, bool) and seq_raw >= 0:
            key = (ticket_raw, seq_raw)
            if key in seen_pairs:
                row_errors.append(
                    (
                        ERROR_DUPLICATE_EVIDENCE,
                        f"duplicate ticket_id/evidence_seq pair; first seen on line {seen_pairs[key]}",
                    )
                )
            else:
                seen_pairs[key] = line_no

        if row_errors:
            ticket_bad.add(ticket_id)
            for code, message in row_errors:
                failures.append(
                    _failure(
                        code,
                        message,
                        ticket_id=ticket_id,
                        line_no=line_no,
                        evidence_seq=seq_for_failure,
                    )
                )
            continue

        assert key is not None
        evidence = Evidence(
            ticket_id=ticket_raw,
            evidence_seq=seq_raw,
            evidence_type=evidence_type,
            source_role=source_role,
            timestamp=timestamp,
            terms=tuple(terms),
            line_no=line_no,
        )
        rows_by_ticket.setdefault(ticket_raw, []).append(evidence)

    for ticket_id, rows in rows_by_ticket.items():
        if ticket_id in ticket_bad or not rows:
            continue
        timestamps = {row.timestamp for row in rows}
        if len(timestamps) <= 1:
            first = min(rows, key=lambda row: (row.line_no, row.evidence_seq))
            failures.append(
                _failure(
                    ERROR_TIMESTAMP_INVALID,
                    "all evidence timestamps in a ticket must not be identical",
                    ticket_id=ticket_id,
                    line_no=first.line_no,
                    evidence_seq=first.evidence_seq,
                )
            )
            ticket_bad.add(ticket_id)

    evidence_by_ticket: dict[str, tuple[Evidence, ...]] = {}
    for ticket_id, rows in rows_by_ticket.items():
        if ticket_id in ticket_bad:
            continue
        ordered = tuple(sorted(rows, key=lambda row: row.evidence_seq))
        evidence_by_ticket[ticket_id] = ordered

    failures.sort(
        key=lambda item: (
            str(item["ticket_id"]),
            int(item["line_no"]),
            str(item["stage"]),
            str(item["code"]),
        )
    )
    ticket_ids = sorted(set(rows_by_ticket) | {str(item["ticket_id"]) for item in failures if not str(item["ticket_id"]).startswith("@")})
    return EvidenceLoadResult(evidence_by_ticket=evidence_by_ticket, failures=failures, ticket_ids=ticket_ids)


def load_evidence_text(text: str, rules: Rules) -> EvidenceLoadResult:
    parsed, parse_failures = _parse_jsonl_lines(text.splitlines())
    result = validate_evidence_records(parsed, rules)
    result.failures.extend(parse_failures)
    result.failures.sort(
        key=lambda item: (
            str(item["ticket_id"]),
            int(item["line_no"]),
            str(item["stage"]),
            str(item["code"]),
        )
    )
    result.ticket_ids = sorted(
        set(result.evidence_by_ticket)
        | {str(item["ticket_id"]) for item in result.failures if not str(item["ticket_id"]).startswith("@")}
    )
    return result


def load_evidence_file(path: str | Path, rules: Rules) -> EvidenceLoadResult:
    text = Path(path).read_text(encoding="utf-8")
    return load_evidence_text(text, rules)


def evidence_lines_to_jsonl(lines: Iterable[str | dict]) -> str:
    output: list[str] = []
    for line in lines:
        if isinstance(line, str):
            output.append(line.rstrip("\n"))
        else:
            output.append(json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(output) + ("\n" if output else "")

