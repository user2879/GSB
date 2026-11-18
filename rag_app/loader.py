"""JSONL loaders with per-record validation and failure collection.

Invalid records never abort the load; they are appended to the failure
list with code :data:`rag_app.errors.E_REC_INVALID` and processing of the
remaining records continues.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .errors import E_REC_INVALID
from .models import Command, Document, Failure, QueryTurn, turn_sort_key


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    rows: list[tuple[int, dict[str, Any] | None, str | None]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append((line_no, None, f"json decode error: {exc}"))
                continue
            if not isinstance(data, dict):
                rows.append((line_no, None, "record is not a JSON object"))
                continue
            rows.append((line_no, data, None))
    return rows


def _fail(code: str, scope: str, message: str) -> Failure:
    return Failure(code=code, scope=scope, message=message, sort_key=scope)


def load_documents(path: str | Path) -> tuple[list[Document], list[Failure]]:
    failures: list[Failure] = []
    docs: list[Document] = []
    seen: set[str] = set()
    for line_no, data, parse_error in _read_jsonl(Path(path)):
        scope = f"doc:L{line_no}"
        if parse_error is not None:
            failures.append(_fail(E_REC_INVALID, scope, parse_error))
            continue
        assert data is not None
        missing = [key for key in ("doc_id", "body") if key not in data]
        if missing:
            failures.append(
                _fail(E_REC_INVALID, scope, f"missing fields: {missing}")
            )
            continue
        doc_id = str(data["doc_id"])
        if not doc_id:
            failures.append(_fail(E_REC_INVALID, scope, "empty doc_id"))
            continue
        if doc_id in seen:
            failures.append(
                _fail(E_REC_INVALID, scope, f"duplicate doc_id: {doc_id}")
            )
            continue
        if not str(data["body"]).strip():
            failures.append(
                _fail(E_REC_INVALID, scope, f"empty body: {doc_id}")
            )
            continue
        seen.add(doc_id)
        docs.append(Document.from_dict(data))
    return docs, failures


def load_query_stream(path: str | Path) -> tuple[list[QueryTurn], list[Failure]]:
    failures: list[Failure] = []
    raw_turns: list[QueryTurn] = []
    seen_keys: set[tuple[str, int]] = set()
    for line_no, data, parse_error in _read_jsonl(Path(path)):
        scope = f"query:L{line_no}"
        if parse_error is not None:
            failures.append(_fail(E_REC_INVALID, scope, parse_error))
            continue
        assert data is not None
        missing = [
            key
            for key in ("session_id", "turn_no", "query", "is_followup")
            if key not in data
        ]
        if missing:
            failures.append(
                _fail(E_REC_INVALID, scope, f"missing fields: {missing}")
            )
            continue
        try:
            turn = QueryTurn.from_dict(data)
        except (TypeError, ValueError) as exc:
            failures.append(_fail(E_REC_INVALID, scope, f"bad type: {exc}"))
            continue
        if not str(turn.session_id):
            failures.append(_fail(E_REC_INVALID, scope, "empty session_id"))
            continue
        if not turn.query.strip():
            failures.append(
                _fail(
                    E_REC_INVALID,
                    scope,
                    f"empty query: {turn.session_id}#{turn.turn_no}",
                )
            )
            continue
        if not isinstance(turn.turn_no, int) or turn.turn_no <= 0:
            failures.append(
                _fail(
                    E_REC_INVALID,
                    scope,
                    f"turn_no must be positive int: {turn.turn_no}",
                )
            )
            continue
        key = (turn.session_id, turn.turn_no)
        if key in seen_keys:
            failures.append(
                _fail(
                    E_REC_INVALID,
                    scope,
                    f"duplicate turn key: {turn.session_id}#{turn.turn_no}",
                )
            )
            continue
        if turn.is_followup and not turn.context_ref:
            failures.append(
                _fail(
                    E_REC_INVALID,
                    scope,
                    f"followup without context_ref: {turn.session_id}#{turn.turn_no}",
                )
            )
            continue
        if not turn.is_followup and turn.context_ref:
            failures.append(
                _fail(
                    E_REC_INVALID,
                    scope,
                    f"non-followup carries context_ref: {turn.session_id}#{turn.turn_no}",
                )
            )
            continue
        seen_keys.add(key)
        raw_turns.append(turn)

    # Session-level ordering: strictly increasing turn numbers without gaps
    # and context references must point at earlier turns in the same session.
    by_session: dict[str, list[QueryTurn]] = {}
    for turn in raw_turns:
        by_session.setdefault(turn.session_id, []).append(turn)
    valid: list[QueryTurn] = []
    for session_id, turns in by_session.items():
        ordered = sorted(turns, key=lambda item: item.turn_no)
        prior: set[int] = set()
        expected = 1
        for turn in ordered:
            scope = f"query:{turn_sort_key(session_id, turn.turn_no)}"
            if turn.turn_no != expected:
                failures.append(
                    _fail(
                        E_REC_INVALID,
                        scope,
                        f"turn sequence gap: expected {expected}, got {turn.turn_no}",
                    )
                )
                continue
            if any(ref not in prior for ref in turn.context_ref):
                failures.append(
                    _fail(
                        E_REC_INVALID,
                        scope,
                        f"context_ref points to unknown/earlier turn: {list(turn.context_ref)}",
                    )
                )
                continue
            prior.add(turn.turn_no)
            expected += 1
            valid.append(turn)
    valid.sort(key=lambda item: turn_sort_key(item.session_id, item.turn_no))
    return valid, failures


def load_commands(path: str | Path) -> tuple[list[Command], list[Failure]]:
    failures: list[Failure] = []
    commands: list[Command] = []
    seen_seqs: set[int] = set()
    for line_no, data, parse_error in _read_jsonl(Path(path)):
        scope = f"cmd:L{line_no}"
        if parse_error is not None:
            failures.append(_fail(E_REC_INVALID, scope, parse_error))
            continue
        assert data is not None
        if "seq" not in data or "op" not in data:
            failures.append(
                _fail(E_REC_INVALID, scope, "missing seq/op fields")
            )
            continue
        try:
            command = Command.from_dict(data)
        except (TypeError, ValueError) as exc:
            failures.append(_fail(E_REC_INVALID, scope, f"bad type: {exc}"))
            continue
        if command.seq in seen_seqs:
            failures.append(
                _fail(E_REC_INVALID, scope, f"duplicate command seq: {command.seq}")
            )
            continue
        seen_seqs.add(command.seq)
        commands.append(command)
    commands.sort(key=lambda item: item.seq)
    return commands, failures

