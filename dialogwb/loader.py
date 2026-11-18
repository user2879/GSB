from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .lexicon import Lexicon
from .util import fingerprint


BUSINESS_CODES = {
    "E_EMPTY_UTTERANCE",
    "E_BAD_JSON",
    "E_UNKNOWN_TOKEN",
    "E_TURN_GAP",
    "E_BAD_TIMESTAMP",
    "E_EMPTY_SESSION",
    "E_BAD_EVENT",
    "E_EDIT_TARGET",
    "E_STAGE_EXCEPTION",
}


@dataclass
class LoadedInput:
    records: list[dict[str, Any]]
    events: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    parse_failures: list[dict[str, Any]]

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            [
                {
                    "session_id": event["session_id"],
                    "turn": event["turn"],
                    "speaker": event["speaker"],
                    "text": event["text"],
                    "timestamp": event["timestamp"],
                }
                for event in self.events
            ]
        )


def load_jsonl(path: str | Path, lexicon: Lexicon) -> LoadedInput:
    records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    try:
        handle = open(path, encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read events JSONL: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_failures.append(
                    _failure(
                        stage="load",
                        code="E_BAD_JSON",
                        message="invalid JSON",
                        line=line_number,
                        exception_type=type(exc).__name__,
                    )
                )
                continue
            if not isinstance(value, dict):
                parse_failures.append(
                    _failure(
                        stage="load",
                        code="E_BAD_EVENT",
                        message="event must be a JSON object",
                        line=line_number,
                    )
                )
                continue
            record = _coerce_record(value, line_number=line_number, order=line_number)
            if isinstance(record, dict) and record.get("__invalid_session__"):
                parse_failures.append(
                    _failure(
                        stage="load",
                        code="E_EMPTY_SESSION",
                        message="session ID must be a non-empty string",
                        line=line_number,
                    )
                )
                continue
            if isinstance(record, dict) and record.get("__invalid_turn__"):
                parse_failures.append(
                    _failure(
                        stage="load",
                        code="E_BAD_EVENT",
                        message="turn must be a positive integer",
                        line=line_number,
                        session_id=record.get("session_id"),
                    )
                )
                continue
            if record is not None:
                records.append(record)

    return normalize_records(records, lexicon, parse_failures=parse_failures)


def normalize_records(
    records: list[dict[str, Any]],
    lexicon: Lexicon,
    parse_failures: list[dict[str, Any]] | None = None,
) -> LoadedInput:
    parse_failures = list(parse_failures or [])
    failures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_seen: dict[str, int] = {}

    indexed = sorted(
        enumerate(records),
        key=lambda item: (
            item[1].get("_order", item[0]),
            item[1].get("_line") if item[1].get("_line") is not None else 10**12,
            item[0],
        ),
    )

    for _, record in indexed:
        session_id = record.get("session_id")
        turn = record.get("turn")
        line = record.get("_line")
        if not isinstance(session_id, str) or not session_id:
            failures.append(
                _failure(
                    stage="load",
                    code="E_EMPTY_SESSION",
                    message="session ID must be a non-empty string",
                    line=line,
                    turn=_optional_int(turn),
                )
            )
            continue
        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_EVENT",
                    message="turn must be a positive integer",
                    line=line,
                    session_id=session_id,
                )
            )
            continue

        expected = last_seen.get(session_id, 0) + 1
        if turn != expected:
            failures.append(
                _failure(
                    stage="load",
                    code="E_TURN_GAP",
                    message=f"turn sequence violation; expected {expected}, got {turn}",
                    line=line,
                    session_id=session_id,
                    turn=turn,
                )
            )
            # The rejected line still advances the sequence cursor. This keeps
            # the remaining rows processable while making the hole visible.
            last_seen[session_id] = turn
            continue
        last_seen[session_id] = turn

        text = record.get("text", record.get("utterance"))
        if not isinstance(text, str) or not text.strip():
            failures.append(
                _failure(
                    stage="load",
                    code="E_EMPTY_UTTERANCE",
                    message="utterance text is empty",
                    line=line,
                    session_id=session_id,
                    turn=turn,
                )
            )
            continue

        speaker = record.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            speaker = "unknown"

        timestamp_value = record.get("timestamp", record.get("time"))
        parsed_timestamp = _parse_timestamp(timestamp_value)
        if parsed_timestamp is None:
            failures.append(
                _failure(
                    stage="load",
                    code="E_BAD_TIMESTAMP",
                    message="timestamp is missing or not parseable",
                    line=line,
                    session_id=session_id,
                    turn=turn,
                )
            )
            continue

        try:
            tokens, unknown = lexicon.split(text)
        except Exception as exc:  # defensive isolation for malformed rules
            failures.append(
                _failure(
                    stage="split",
                    code="E_STAGE_EXCEPTION",
                    message=str(exc) or "split failed",
                    line=line,
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                )
            )
            continue
        if unknown:
            for item in unknown:
                failures.append(
                    _failure(
                        stage="split",
                        code="E_UNKNOWN_TOKEN",
                        message=f"{item['token']}@{item['start']}:{item['end']}",
                        line=line,
                        session_id=session_id,
                        turn=turn,
                    )
                )
            continue

        token_dicts = [token.as_dict() for token in tokens]
        canonical = {
            "session_id": session_id,
            "turn": turn,
            "speaker": speaker,
            "text": text,
            "timestamp": parsed_timestamp,
            "tokens": token_dicts,
        }
        events.append(
            {
                **canonical,
                "line": line,
                "order": record.get("_order", 0),
                "tokens": token_dicts,
                "content_fingerprint": fingerprint(canonical),
            }
        )

    events.sort(key=lambda item: (item["session_id"], item["turn"]))
    all_failures = _sort_failures([*parse_failures, *failures])
    return LoadedInput(
        records=records,
        events=events,
        failures=all_failures,
        parse_failures=parse_failures,
    )


def records_as_jsonl(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in sorted(events, key=lambda item: (item["session_id"], item["turn"])):
        lines.append(
            json.dumps(
                {
                    "session_id": event["session_id"],
                    "turn": event["turn"],
                    "speaker": event["speaker"],
                    "text": event["text"],
                    "timestamp": event["timestamp"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _coerce_record(
    value: dict[str, Any], *, line_number: int, order: int
) -> dict[str, Any] | None:
    session_id = value.get("session_id", value.get("session"))
    if not isinstance(session_id, str) or not session_id.strip():
        return {"__invalid_session__": True}
    turn = value.get("turn", value.get("round"))
    if isinstance(turn, str) and turn.isdigit():
        turn = int(turn)
    if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
        return {"__invalid_turn__": True, "session_id": session_id}
    text = value.get("text", value.get("utterance"))
    speaker = value.get("speaker", value.get("role", "unknown"))
    timestamp = value.get("timestamp", value.get("time"))
    return {
        "session_id": session_id,
        "turn": turn,
        "speaker": speaker,
        "text": text,
        "timestamp": timestamp,
        "_line": line_number,
        "_order": order,
    }


def _parse_timestamp(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _failure(
    *,
    stage: str,
    code: str,
    message: str,
    line: int | None = None,
    session_id: str | None = None,
    turn: int | None = None,
    exception_type: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "turn": turn,
        "stage": stage,
        "code": code,
        "exception_type": exception_type,
        "message": message,
        "line": line,
    }


def _sort_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_order = {"load": 0, "split": 1, "extraction": 2, "merge": 3, "rewrite": 4}
    return sorted(
        failures,
        key=lambda item: (
            item.get("line") if item.get("line") is not None else 10**12,
            stage_order.get(str(item.get("stage")), 99),
            item.get("session_id") or "",
            item.get("turn") if item.get("turn") is not None else -1,
            item.get("code") or "",
            item.get("message") or "",
        ),
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
