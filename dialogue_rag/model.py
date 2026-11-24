"""Typed domain state and configuration loading/validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .codes import Code, failure
from .fixedpoint import ALLOWED_GEARS, ALLOWED_WIDTHS, parse_gear, parse_width
from .hashing import fingerprint, fingerprint_lenient


DEFAULT_TOPIC = {
    "similarity_threshold": Decimal("0.62"),
    "equal_rule": "merge",
}
DEFAULT_SCORING = {
    "distance_decay": Decimal("0.82"),
    "cross_topic_penalty": Decimal("0.55"),
    "normalization_weight": Decimal("3.0"),
    "answer_threshold": Decimal("0.60"),
    "error_tolerance": Decimal("0.000001"),
}


@dataclass
class Turn:
    session_id: str
    turn_id: str
    seq: int
    query: str

    @property
    def content_fingerprint(self) -> str:
        return fingerprint(
            {
                "object": "turn_content",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "seq": self.seq,
                "query": self.query,
            }
        )


@dataclass
class Evidence:
    evidence_id: str
    text: str
    weight: Decimal

    @property
    def content_fingerprint(self) -> str:
        return fingerprint(
            {"object": "evidence_content", "evidence_id": self.evidence_id, "text": self.text}
        )

    @property
    def weight_fingerprint(self) -> str:
        return fingerprint({"evidence_id": self.evidence_id, "weight": self.weight})


@dataclass
class Channel:
    channel_id: str
    weight: Decimal

    @property
    def weight_fingerprint(self) -> str:
        return fingerprint({"channel_id": self.channel_id, "weight": self.weight})


@dataclass
class Hit:
    channel_id: str
    evidence_id: str
    raw_score: Decimal
    mentions: int = 1


@dataclass
class RuntimeState:
    config_path: str | None = None
    seed: int = 236
    schema_version: str = "1.0"
    session_order: list[str] = field(default_factory=list)
    turns: dict[str, list[Turn]] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    channels: dict[str, Channel] = field(default_factory=dict)
    hits: dict[tuple[str, str], list[Hit]] = field(default_factory=dict)
    max_evidence_per_turn: int = 5
    topic_threshold: Decimal = DEFAULT_TOPIC["similarity_threshold"]
    equal_rule: str = "merge"
    distance_decay: Decimal = DEFAULT_SCORING["distance_decay"]
    cross_topic_penalty: Decimal = DEFAULT_SCORING["cross_topic_penalty"]
    normalization_weight: Decimal = DEFAULT_SCORING["normalization_weight"]
    answer_threshold: Decimal = DEFAULT_SCORING["answer_threshold"]
    error_tolerance: Decimal = DEFAULT_SCORING["error_tolerance"]
    fixed_width: int = 16
    quant_gear: str = "G1"
    rounding: str = "half_up"
    budget_points: int | None = None
    workers: int = 1
    config_hash: str = ""
    source_config: dict[str, Any] = field(default_factory=dict)

    def sorted_turns(self, session_id: str) -> list[Turn]:
        return sorted(self.turns.get(session_id, []), key=lambda item: (item.seq, item.turn_id))

    def all_turn_keys(self) -> list[tuple[str, int, str]]:
        keys: list[tuple[str, int, str]] = []
        for session_id in sorted(self.session_order):
            for turn in self.sorted_turns(session_id):
                keys.append((session_id, turn.seq, turn.turn_id))
        return keys

    def find_turn(self, session_id: str, turn_id: str) -> Turn | None:
        for turn in self.turns.get(session_id, []):
            if turn.turn_id == turn_id:
                return turn
        return None

    def evidence_fingerprint(self) -> str:
        return fingerprint(
            {eid: {"text": item.text, "weight": item.weight} for eid, item in sorted(self.evidence.items())}
        )


def read_json_file(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [failure(Code.INVALID_JSON, f"cannot read {path}: {exc}", field=str(path))]
    try:
        return json.loads(text, parse_float=Decimal), []
    except json.JSONDecodeError as exc:
        return None, [
            failure(
                Code.INVALID_JSON,
                f"invalid JSON: {exc.msg}",
                field=f"line {exc.lineno} column {exc.colno}",
            )
        ]


def _is_real_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, Decimal))


def decimal_value(
    value: Any,
    *,
    field_name: str,
    failures: list[dict[str, Any]],
    session_id: str | None = None,
    turn_id: str | None = None,
    nonnegative: bool = True,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal | None:
    if isinstance(value, str):
        try:
            result = Decimal(value)
        except Exception:
            failures.append(
                failure(Code.FIELD_TYPE, f"{field_name} must be a finite number", field=field_name)
            )
            return None
    elif _is_real_number(value):
        try:
            result = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
        except Exception:
            failures.append(
                failure(Code.FIELD_TYPE, f"{field_name} cannot be parsed as decimal", field=field_name)
            )
            return None
    else:
        failures.append(
            failure(
                Code.FIELD_TYPE,
                f"{field_name} must be a finite number",
                field=field_name,
                session_id=session_id,
                turn_id=turn_id,
                rejected_type=type(value).__name__,
            )
        )
        return None
    if not result.is_finite() or math.isinf(float(result)) or math.isnan(float(result)):
        failures.append(
            failure(Code.FIELD_NON_FINITE, f"{field_name} must be finite", field=field_name, rejected_value=str(value))
        )
        return None
    if nonnegative and result < 0:
        failures.append(
            failure(Code.FIELD_OUT_OF_RANGE, f"{field_name} must be non-negative", field=field_name, rejected_value=str(result))
        )
        return None
    if minimum is not None and result < minimum:
        failures.append(
            failure(Code.FIELD_OUT_OF_RANGE, f"{field_name} must be >= {minimum}", field=field_name, rejected_value=str(result))
        )
        return None
    if maximum is not None and result > maximum:
        failures.append(
            failure(Code.FIELD_OUT_OF_RANGE, f"{field_name} must be <= {maximum}", field=field_name, rejected_value=str(result))
        )
        return None
    return result


def positive_int(
    value: Any,
    *,
    field_name: str,
    failures: list[dict[str, Any]],
    session_id: str | None = None,
    turn_id: str | None = None,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        failures.append(
            failure(
                Code.FIELD_TYPE if not isinstance(value, int) or isinstance(value, bool) else Code.FIELD_OUT_OF_RANGE,
                f"{field_name} must be a positive integer",
                field=field_name,
                session_id=session_id,
                turn_id=turn_id,
                rejected_value=value if isinstance(value, (str, int, float)) else type(value).__name__,
            )
        )
        return None
    return value


def nonempty_str(
    value: Any,
    *,
    field_name: str,
    failures: list[dict[str, Any]],
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        failures.append(
            failure(Code.FIELD_TYPE, f"{field_name} must be a non-empty string", field=field_name, session_id=session_id, turn_id=turn_id)
        )
        return None
    return value


def load_config(path: str | Path) -> tuple[RuntimeState, list[dict[str, Any]], str]:
    raw, failures = read_json_file(path)
    if raw is None:
        state = RuntimeState()
        return state, failures, ""
    state, build_failures = build_state(raw)
    failures.extend(build_failures)
    config_hash = fingerprint(raw)
    state.config_hash = config_hash
    state.source_config = raw if isinstance(raw, dict) else {}
    return state, failures, config_hash




def _safe_config_fingerprint(raw: Any) -> str:
    try:
        return fingerprint(raw)
    except ValueError:
        return fingerprint_lenient(raw)


def build_state(raw: Any) -> tuple[RuntimeState, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return RuntimeState(), [failure(Code.FIELD_TYPE, "configuration root must be an object", field="$")]

    state = RuntimeState()
    state.source_config = raw
    state.schema_version = str(raw.get("schema_version", "1.0"))
    seed = raw.get("seed", 236)
    state.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else 236

    _load_evidence(raw, state, failures)
    _load_channels(raw, state, failures)
    _load_sessions(raw, state, failures)
    _load_declarations(raw, state, failures)
    _load_parameters(raw, state, failures)
    state.config_hash = _safe_config_fingerprint(raw)
    return state, failures


def _load_evidence(raw: dict[str, Any], state: RuntimeState, failures: list[dict[str, Any]]) -> None:
    rows = raw.get("evidence", raw.get("evidence_library", []))
    if not isinstance(rows, list):
        failures.append(failure(Code.FIELD_TYPE, "evidence must be a list", field="evidence"))
        return
    for index, row in enumerate(rows):
        field_name = f"evidence[{index}]"
        if not isinstance(row, dict):
            failures.append(failure(Code.FIELD_TYPE, f"{field_name} must be an object", field=field_name))
            continue
        evidence_id = nonempty_str(row.get("evidence_id", row.get("id")), field_name=f"{field_name}.evidence_id", failures=failures)
        if evidence_id is None:
            continue
        if evidence_id in state.evidence:
            failures.append(failure(Code.DUPLICATE_ID, "evidence_id must be globally unique", field="evidence_id", evidence_id=evidence_id))
            continue
        text = row.get("text", row.get("content", ""))
        if not isinstance(text, str):
            failures.append(failure(Code.FIELD_TYPE, "evidence text must be a string", field=f"{field_name}.text", evidence_id=evidence_id))
            text = ""
        weight = decimal_value(row.get("weight", Decimal("1.0")), field_name=f"{field_name}.weight", failures=failures)
        if weight is None:
            continue
        state.evidence[evidence_id] = Evidence(evidence_id, text, weight)


def _load_channels(raw: dict[str, Any], state: RuntimeState, failures: list[dict[str, Any]]) -> None:
    rows = raw.get("channels", raw.get("retrieval_channels", []))
    if not isinstance(rows, list):
        failures.append(failure(Code.FIELD_TYPE, "channels must be a list", field="channels"))
        return
    for index, row in enumerate(rows):
        field_name = f"channels[{index}]"
        if not isinstance(row, dict):
            failures.append(failure(Code.FIELD_TYPE, f"{field_name} must be an object", field=field_name))
            continue
        channel_id = nonempty_str(row.get("channel_id", row.get("id")), field_name=f"{field_name}.channel_id", failures=failures)
        if channel_id is None:
            continue
        if channel_id in state.channels:
            failures.append(failure(Code.DUPLICATE_ID, "channel_id must be unique", field="channel_id", channel_id=channel_id))
            continue
        weight = decimal_value(row.get("weight", Decimal("1.0")), field_name=f"{field_name}.weight", failures=failures)
        if weight is None:
            continue
        state.channels[channel_id] = Channel(channel_id, weight)


def _load_sessions(raw: dict[str, Any], state: RuntimeState, failures: list[dict[str, Any]]) -> None:
    if "sessions" in raw:
        rows = raw["sessions"]
        if not isinstance(rows, list):
            failures.append(failure(Code.FIELD_TYPE, "sessions must be a list", field="sessions"))
            return
    else:
        rows = [{"session_id": raw.get("session_id", "session-1"), "turns": raw.get("turns", [])}]

    seen_sessions: set[str] = set()
    for s_index, session in enumerate(rows):
        field_prefix = f"sessions[{s_index}]"
        if not isinstance(session, dict):
            failures.append(failure(Code.FIELD_TYPE, f"{field_prefix} must be an object", field=field_prefix))
            continue
        session_id = nonempty_str(session.get("session_id", session.get("id")), field_name=f"{field_prefix}.session_id", failures=failures)
        if session_id is None:
            session_id = f"__invalid_session_{s_index}"
        if session_id in seen_sessions:
            failures.append(failure(Code.DUPLICATE_ID, "session_id must be unique", field="session_id", session_id=session_id))
            continue
        seen_sessions.add(session_id)
        state.session_order.append(session_id)
        state.turns[session_id] = []
        turn_ids: set[str] = set()
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            failures.append(failure(Code.FIELD_TYPE, f"{field_prefix}.turns must be a list", field=f"{field_prefix}.turns", session_id=session_id))
            continue
        expected_seq = 1
        previous_seq: int | None = None
        for t_index, row in enumerate(turns):
            t_prefix = f"{field_prefix}.turns[{t_index}]"
            if not isinstance(row, dict):
                failures.append(failure(Code.FIELD_TYPE, f"{t_prefix} must be an object", field=t_prefix, session_id=session_id))
                continue
            turn_id = nonempty_str(row.get("turn_id", row.get("id")), field_name=f"{t_prefix}.turn_id", failures=failures, session_id=session_id)
            seq = positive_int(row.get("seq"), field_name=f"{t_prefix}.seq", failures=failures, session_id=session_id, turn_id=turn_id)
            query = nonempty_str(row.get("query"), field_name=f"{t_prefix}.query", failures=failures, session_id=session_id, turn_id=turn_id)
            if turn_id is None or seq is None or query is None:
                continue
            if turn_id in turn_ids:
                failures.append(failure(Code.DUPLICATE_ID, "turn_id must be unique in session", field="turn_id", session_id=session_id, turn_id=turn_id))
                continue
            turn_ids.add(turn_id)
            if previous_seq is not None and seq <= previous_seq:
                failures.append(failure(Code.TURN_SEQ_ORDER, "turn seq must be strictly increasing", field="seq", session_id=session_id, turn_id=turn_id, previous_seq=previous_seq, seq=seq))
            elif seq != expected_seq:
                failures.append(failure(Code.TURN_SEQ_GAP, "turn seq must have no holes and start at 1", field="seq", session_id=session_id, turn_id=turn_id, expected_seq=expected_seq, seq=seq))
            previous_seq = seq
            expected_seq = seq + 1 if seq >= expected_seq else expected_seq + 1
            state.turns[session_id].append(Turn(session_id, turn_id, seq, query))

            embedded = row.get("channel_hits", row.get("recall_declarations"))
            if embedded is not None:
                _load_hit_rows(embedded, state, failures, session_id, turn_id, f"{t_prefix}.channel_hits")


def _load_declarations(raw: dict[str, Any], state: RuntimeState, failures: list[dict[str, Any]]) -> None:
    rows = raw.get("recall_declarations", raw.get("declarations", []))
    if rows is None:
        return
    if not isinstance(rows, list):
        failures.append(failure(Code.FIELD_TYPE, "recall_declarations must be a list", field="recall_declarations"))
        return
    for index, declaration in enumerate(rows):
        prefix = f"recall_declarations[{index}]"
        if not isinstance(declaration, dict):
            failures.append(failure(Code.FIELD_TYPE, f"{prefix} must be an object", field=prefix))
            continue
        session_id = declaration.get("session_id")
        if session_id is None and len(state.session_order) == 1:
            session_id = state.session_order[0]
        session_id = nonempty_str(session_id, field_name=f"{prefix}.session_id", failures=failures)
        turn_id = nonempty_str(declaration.get("turn_id"), field_name=f"{prefix}.turn_id", failures=failures, session_id=session_id)
        channel_id = nonempty_str(declaration.get("channel_id"), field_name=f"{prefix}.channel_id", failures=failures, session_id=session_id, turn_id=turn_id)
        if session_id is None or turn_id is None or channel_id is None:
            continue
        if session_id not in state.turns:
            failures.append(failure(Code.UNKNOWN_REFERENCE, "declaration references unknown session", field="session_id", session_id=session_id, turn_id=turn_id))
            continue
        if state.find_turn(session_id, turn_id) is None:
            failures.append(failure(Code.UNKNOWN_REFERENCE, "declaration references unknown turn", field="turn_id", session_id=session_id, turn_id=turn_id))
            continue
        if channel_id not in state.channels:
            failures.append(failure(Code.UNKNOWN_REFERENCE, "declaration references unknown channel", field="channel_id", session_id=session_id, turn_id=turn_id, channel_id=channel_id))
            continue
        hits = declaration.get("hits", [])
        _load_hit_rows(hits, state, failures, session_id, turn_id, f"{prefix}.hits", default_channel=channel_id)


def _load_hit_rows(
    rows: Any,
    state: RuntimeState,
    failures: list[dict[str, Any]],
    session_id: str,
    turn_id: str,
    prefix: str,
    default_channel: str | None = None,
) -> None:
    if not isinstance(rows, list):
        failures.append(failure(Code.FIELD_TYPE, f"{prefix} must be a list", field=prefix, session_id=session_id, turn_id=turn_id))
        return
    key = (session_id, turn_id)
    state.hits.setdefault(key, [])
    for index, row in enumerate(rows):
        item_prefix = f"{prefix}[{index}]"
        if not isinstance(row, dict):
            failures.append(failure(Code.FIELD_TYPE, f"{item_prefix} must be an object", field=item_prefix, session_id=session_id, turn_id=turn_id))
            continue
        channel_id = row.get("channel_id", default_channel)
        channel_id = nonempty_str(channel_id, field_name=f"{item_prefix}.channel_id", failures=failures, session_id=session_id, turn_id=turn_id)
        evidence_id = nonempty_str(row.get("evidence_id", row.get("id")), field_name=f"{item_prefix}.evidence_id", failures=failures, session_id=session_id, turn_id=turn_id)
        if channel_id is None or evidence_id is None:
            continue
        if channel_id not in state.channels:
            failures.append(failure(Code.UNKNOWN_REFERENCE, "hit references unknown channel", field="channel_id", session_id=session_id, turn_id=turn_id, channel_id=channel_id))
            continue
        if evidence_id not in state.evidence:
            failures.append(failure(Code.UNKNOWN_REFERENCE, "hit references unknown evidence", field="evidence_id", session_id=session_id, turn_id=turn_id, evidence_id=evidence_id))
            continue
        raw_score = decimal_value(row.get("raw_score", row.get("score", Decimal("0"))), field_name=f"{item_prefix}.raw_score", failures=failures, session_id=session_id, turn_id=turn_id)
        if raw_score is None:
            continue
        mentions = row.get("mentions", row.get("count", 1))
        mention_count = positive_int(mentions, field_name=f"{item_prefix}.mentions", failures=failures, session_id=session_id, turn_id=turn_id)
        if mention_count is None:
            continue
        state.hits[key].append(Hit(channel_id, evidence_id, raw_score, mention_count))


def _load_parameters(raw: dict[str, Any], state: RuntimeState, failures: list[dict[str, Any]]) -> None:
    recall = raw.get("recall", {})
    if not isinstance(recall, dict):
        failures.append(failure(Code.FIELD_TYPE, "recall must be an object", field="recall"))
        recall = {}
    cap_value = raw.get("max_evidence_per_turn", recall.get("max_evidence_per_turn", 5))
    cap = positive_int(cap_value, field_name="recall.max_evidence_per_turn", failures=failures)
    if cap is not None:
        state.max_evidence_per_turn = cap

    topic = raw.get("topic", {})
    if not isinstance(topic, dict):
        failures.append(failure(Code.FIELD_TYPE, "topic must be an object", field="topic"))
        topic = {}
    threshold = decimal_value(topic.get("similarity_threshold", DEFAULT_TOPIC["similarity_threshold"]), field_name="topic.similarity_threshold", failures=failures, minimum=Decimal("0"), maximum=Decimal("1"))
    if threshold is not None:
        state.topic_threshold = threshold
    equal_rule = topic.get("equal_rule", "merge")
    if equal_rule not in ("merge", "split"):
        failures.append(failure(Code.FIELD_NOT_ALLOWED, "topic.equal_rule must be merge or split", field="topic.equal_rule", allowed=["merge", "split"], rejected_value=equal_rule))
    else:
        state.equal_rule = equal_rule

    scoring = raw.get("scoring", raw.get("fixed_point", {}))
    fixed = raw.get("fixed_point", {})
    if not isinstance(scoring, dict):
        scoring = {}
    if not isinstance(fixed, dict):
        fixed = {}
    width_value = fixed.get("fixed_width", fixed.get("width", scoring.get("fixed_width", 16)))
    try:
        state.fixed_width = parse_width(width_value)
        if state.fixed_width not in ALLOWED_WIDTHS:
            raise ValueError
    except (ValueError, TypeError):
        failures.append(failure(Code.FIELD_NOT_ALLOWED, "fixed_point.fixed_width must be one of 8,16,32/F8/F16/F32", field="fixed_point.fixed_width", allowed=list(ALLOWED_WIDTHS), rejected_value=str(width_value)))
    gear_value = fixed.get("quant_gear", fixed.get("gear", scoring.get("quant_gear", "G1")))
    try:
        state.quant_gear = parse_gear(gear_value)
        if state.quant_gear not in ALLOWED_GEARS:
            raise ValueError
    except (ValueError, TypeError):
        failures.append(failure(Code.FIELD_NOT_ALLOWED, "fixed_point.quant_gear must be one of G0,G1,G2,G3", field="fixed_point.quant_gear", allowed=list(ALLOWED_GEARS), rejected_value=str(gear_value)))

    for key, attr, default, maximum in [
        ("distance_decay", "distance_decay", DEFAULT_SCORING["distance_decay"], Decimal("1")),
        ("cross_topic_penalty", "cross_topic_penalty", DEFAULT_SCORING["cross_topic_penalty"], Decimal("1")),
        ("normalization_weight", "normalization_weight", DEFAULT_SCORING["normalization_weight"], None),
        ("answer_threshold", "answer_threshold", DEFAULT_SCORING["answer_threshold"], Decimal("1")),
        ("error_tolerance", "error_tolerance", DEFAULT_SCORING["error_tolerance"], Decimal("1")),
    ]:
        value = decimal_value(scoring.get(key, raw.get(key, default)), field_name=f"scoring.{key}", failures=failures, minimum=Decimal("0"), maximum=maximum)
        if value is not None:
            setattr(state, attr, value)

    budget = raw.get("budget_points", scoring.get("budget_points"))
    if budget is not None:
        parsed_budget = positive_int(budget, field_name="budget_points", failures=failures)
        state.budget_points = parsed_budget
    workers = raw.get("workers", 1)
    if isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1:
        state.workers = workers
    else:
        failures.append(failure(Code.FIELD_OUT_OF_RANGE, "workers must be a positive integer", field="workers", rejected_value=workers))


def load_commands(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw, failures = read_json_file(path)
    if raw is None:
        return [], failures
    if isinstance(raw, dict) and "commands" in raw:
        commands = raw["commands"]
    else:
        commands = raw
    if not isinstance(commands, list):
        return [], [failure(Code.FIELD_TYPE, "commands must be a list or an object with commands", field="commands")]
    normalized: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            failures.append(failure(Code.FIELD_TYPE, f"commands[{index}] must be an object", field=f"commands[{index}]"))
            continue
        command = dict(command)
        command.setdefault("index", index)
        normalized.append(command)
    return normalized, failures
