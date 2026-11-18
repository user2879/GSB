"""JSONL loading with per-line validation and structured failure records."""

from __future__ import annotations

import json
from datetime import datetime

from .errors import E_BAD_JSON, E_DUP_ID, E_EMPTY_TEXT, PipelineError

PREFERENCE_FIELDS = ("pair_id", "prompt", "chosen", "rejected", "annotator_id", "timestamp")
EVAL_FIELDS = ("sample_id", "context", "model_answer", "reference_answer")


def parse_timestamp(value: str) -> datetime:
    """Parse the fixed ISO-8601 timestamp scheme produced by the generator."""
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _add_failure(failures: list, line_no: int, code: str, message: str, stage: str, sample_id=None) -> None:
    record = {
        "code": code,
        "message": message,
        "exception_type": "PipelineError",
        "stage": stage,
        "line_no": line_no,
    }
    if sample_id is not None:
        record["sample_id"] = sample_id
    failures.append(record)


def _iter_jsonl(path: str, stage: str, failures: list):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_failure(failures, line_no, E_BAD_JSON, f"malformed JSON: {exc}", stage)


def load_preferences(path: str, failures: list) -> list:
    stage = "preference_load"
    records = []
    seen = set()
    for line_no, row in _iter_jsonl(path, stage, failures):
        pair_id = row.get("pair_id")
        missing = [name for name in PREFERENCE_FIELDS if name not in row]
        if missing:
            _add_failure(failures, line_no, E_BAD_JSON, f"missing fields {missing}", stage, pair_id)
            continue
        if pair_id in seen:
            _add_failure(failures, line_no, E_DUP_ID, f"duplicate pair_id {pair_id}", stage, pair_id)
            continue
        empty_fields = [
            name
            for name in ("prompt", "chosen", "rejected")
            if not isinstance(row.get(name), str) or not row[name].strip()
        ]
        if empty_fields:
            _add_failure(
                failures,
                line_no,
                E_EMPTY_TEXT,
                f"empty text in fields {empty_fields}",
                stage,
                pair_id,
            )
            continue
        try:
            parse_timestamp(row["timestamp"])
        except ValueError as exc:
            _add_failure(failures, line_no, E_BAD_JSON, f"unparsable timestamp: {exc}", stage, pair_id)
            continue
        seen.add(pair_id)
        records.append(dict(row))
    return records


def load_evaluations(path: str, failures: list) -> list:
    stage = "evaluation_load"
    records = []
    seen = set()
    for line_no, row in _iter_jsonl(path, stage, failures):
        sample_id = row.get("sample_id")
        missing = [name for name in EVAL_FIELDS if name not in row]
        if missing:
            _add_failure(failures, line_no, E_BAD_JSON, f"missing fields {missing}", stage, sample_id)
            continue
        if sample_id in seen:
            _add_failure(failures, line_no, E_DUP_ID, f"duplicate sample_id {sample_id}", stage, sample_id)
            continue
        empty_fields = [
            name
            for name in ("context", "model_answer", "reference_answer")
            if not isinstance(row.get(name), str) or not row[name].strip()
        ]
        if empty_fields:
            _add_failure(
                failures,
                line_no,
                E_EMPTY_TEXT,
                f"empty text in fields {empty_fields}",
                stage,
                sample_id,
            )
            continue
        seen.add(sample_id)
        records.append(dict(row))
    return records


def sorted_failures(failures: list) -> list:
    """Canonical ordering of the failure list used in every report."""
    return sorted(
        failures,
        key=lambda item: (
            str(item.get("sample_id", "")),
            str(item.get("stage", "")),
            str(item.get("code", "")),
            int(item.get("line_no", -1)),
            str(item.get("message", "")),
        ),
    )
