"""Loading and validation for event streams, FSM definitions and configs.

Stage name used by every failure raised here is ``load``.  Structural
problems (bad JSON, empty session id, non-increasing / hole-ridden turns,
illegal operations) are reported with ``E_STREAM_INVALID``; the offending
event is dropped from the materialised stream but processing continues.
"""

import json

from .errors import PipelineError
from .models import Event, FailureRecord, TransitionRule

VALID_OPS = {"append", "replace", "delete"}


def _record_failure(failures, session_id, turn, code, message,
                    batch_id="", injection_id=None, stage="load"):
    rec = FailureRecord(
        session_id=session_id if session_id else "<unknown>",
        turn=turn,
        stage=stage,
        code=code,
        message=message,
        batch_id=batch_id or "",
        injection_id=injection_id,
    )
    failures.append(rec)
    return rec


def parse_event_stream(path):
    """Parse a JSONL event file into raw :class:`Event` objects.

    Each line must be a JSON object.  Lines that are not valid JSON produce an
    ``E_STREAM_INVALID`` failure with the line number instead of an event.
    """
    events = []
    failures = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _record_failure(failures, "", None, "E_STREAM_INVALID",
                                f"line {line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                _record_failure(failures, "", None, "E_STREAM_INVALID",
                                f"line {line_no}: event must be a JSON object")
                continue
            session_id = obj.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                _record_failure(failures, str(session_id), obj.get("turn"),
                                "E_STREAM_INVALID",
                                f"line {line_no}: session_id must be a non-empty string",
                                batch_id=str(obj.get("batch_id", "")))
                continue
            op = obj.get("op", "append")
            if op not in VALID_OPS:
                _record_failure(failures, session_id, obj.get("turn"),
                                "E_STREAM_INVALID",
                                f"line {line_no}: unknown op {op!r}",
                                batch_id=str(obj.get("batch_id", "")),
                                injection_id=obj.get("injection_id"))
                continue
            slot_updates = obj.get("slot_updates", []) or []
            if not isinstance(slot_updates, list) or any(
                    not isinstance(item, dict) for item in slot_updates):
                _record_failure(failures, session_id, obj.get("turn"),
                                "E_STREAM_INVALID",
                                f"line {line_no}: slot_updates must be a list of objects",
                                batch_id=str(obj.get("batch_id", "")),
                                injection_id=obj.get("injection_id"))
                continue
            events.append(Event(
                session_id=session_id,
                op=op,
                turn=obj.get("turn"),
                utterance=str(obj.get("utterance", "")),
                action=obj.get("action"),
                slot_updates=tuple(slot_updates),
                batch_id=str(obj.get("batch_id", "")),
                source_turn=obj.get("source_turn"),
                injection_id=obj.get("injection_id"),
                line_no=line_no,
            ))
    return events, failures


def materialize(events):
    """Apply append/replace/delete operations per session.

    Returns ``(turns_by_session, failures)`` where each materialised turn is a
    plain dict with dense 1-based turn numbers.  Rules:

    * ``append`` extends the session; its declared turn must equal the next
      dense position (no gaps, no backwards jumps).
    * ``replace`` targets an existing turn and swaps its payload in place.
    * ``delete`` removes an existing turn; later turns shift down (dense
      renumbering), keeping their payload.
    """
    sessions = {}
    order = []
    failures = []
    for event in events:
        sid = event.session_id
        if sid not in sessions:
            sessions[sid] = []
            order.append(sid)
        seq = sessions[sid]
        try:
            if event.op == "append":
                expected = len(seq) + 1
                if event.turn is None:
                    turn_no = expected
                else:
                    turn_no = event.turn
                if turn_no != expected:
                    raise PipelineError(
                        "E_STREAM_INVALID",
                        f"append turn {turn_no} is not the next dense turn "
                        f"{expected} for session {sid!r}",
                        "load", injection_id=event.injection_id)
                seq.append(_turn_payload(event, turn_no))
            elif event.op == "replace":
                if event.turn is None or not (1 <= event.turn <= len(seq)):
                    raise PipelineError(
                        "E_STREAM_INVALID",
                        f"replace targets missing turn {event.turn} in session {sid!r}",
                        "load", injection_id=event.injection_id)
                seq[event.turn - 1] = _turn_payload(event, event.turn)
            else:  # delete
                if event.turn is None or not (1 <= event.turn <= len(seq)):
                    raise PipelineError(
                        "E_STREAM_INVALID",
                        f"delete targets missing turn {event.turn} in session {sid!r}",
                        "load", injection_id=event.injection_id)
                del seq[event.turn - 1]
                for new_no, payload in enumerate(seq, start=1):
                    payload["turn"] = new_no
        except PipelineError as exc:
            _record_failure(failures, sid, event.turn, exc.code, str(exc),
                            batch_id=event.batch_id,
                            injection_id=exc.injection_id)
    materialised = {sid: sessions[sid] for sid in order}
    _assert_dense_turns(materialised, failures)
    return materialised, failures


def _turn_payload(event, turn_no):
    return {
        "session_id": event.session_id,
        "turn": turn_no,
        "utterance": event.utterance,
        "action": event.action,
        "slot_updates": [dict(item) for item in event.slot_updates],
        "batch_id": event.batch_id,
        "injection_id": event.injection_id,
        "source_turn": event.source_turn,
        "line_no": event.line_no,
    }


def _assert_dense_turns(materialised, failures):
    for sid, turns in materialised.items():
        for idx, payload in enumerate(turns, start=1):
            if payload["turn"] != idx:
                _record_failure(failures, sid, payload["turn"],
                                "E_STREAM_INVALID",
                                f"hole detected: expected turn {idx}, got {payload['turn']}")


def load_event_stream(path):
    """Convenience wrapper: parse then materialise, merging failure lists."""
    raw, parse_failures = parse_event_stream(path)
    materialised, mat_failures = materialize(raw)
    return materialised, parse_failures + mat_failures
