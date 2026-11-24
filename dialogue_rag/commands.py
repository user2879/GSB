"""Deterministic command mutations and command validation."""

from __future__ import annotations

from typing import Any
from decimal import Decimal

from .codes import Code, failure
from .fixedpoint import parse_gear, parse_width
from .model import RuntimeState, Turn, _load_hit_rows, decimal_value, nonempty_str, positive_int


MUTATING_OPS = {
    "add_turn",
    "replace_query",
    "replace_turn_query",
    "delete_turn",
    "remove_turn",
    "set_topic_rule",
    "update_topic_rule",
    "set_channel_weight",
    "update_channel_weight",
    "set_evidence_weight",
    "update_evidence_weight",
    "set_fixed_point",
    "update_fixed_point",
    "set_budget",
    "set_workers",
}


def apply_command(state: RuntimeState, command: dict[str, Any]) -> dict[str, Any]:
    index = command.get("index")
    op = command.get("op", command.get("operation"))
    result = {
        "command_index": index,
        "op": op,
        "applied": False,
        "failures": [],
        "changes": {},
        "batch_budget_points": command.get("budget_points"),
    }
    failures: list[dict[str, Any]] = result["failures"]
    if not isinstance(op, str) or not op:
        failures.append(failure(Code.UNKNOWN_COMMAND, "command requires op", field="op", command_index=index))
        return result

    if op in ("run_batch", "run_ready"):
        budget = command.get("budget_points", state.budget_points)
        if budget is not None:
            parsed = positive_int(budget, field_name="budget_points", failures=failures)
            if parsed is not None:
                state.budget_points = parsed
                result["batch_budget_points"] = parsed
        result["applied"] = not failures
        return result

    if op == "add_turn":
        _add_turn(state, command, result)
    elif op in ("replace_query", "replace_turn_query"):
        _replace_query(state, command, result)
    elif op in ("delete_turn", "remove_turn"):
        _delete_turn(state, command, result)
    elif op in ("set_topic_rule", "update_topic_rule"):
        _set_topic_rule(state, command, result)
    elif op in ("set_channel_weight", "update_channel_weight"):
        _set_weight(state, command, result, "channel")
    elif op in ("set_evidence_weight", "update_evidence_weight"):
        _set_weight(state, command, result, "evidence")
    elif op in ("set_fixed_point", "update_fixed_point"):
        _set_fixed_point(state, command, result)
    elif op == "set_budget":
        budget = command.get("budget_points", command.get("budget"))
        if budget is None:
            state.budget_points = None
            result["changes"]["budget_points"] = None
        else:
            parsed = positive_int(budget, field_name="budget_points", failures=failures)
            if parsed is not None:
                state.budget_points = parsed
                result["changes"]["budget_points"] = parsed
        result["applied"] = not failures
    elif op == "set_workers":
        workers = positive_int(command.get("workers", 1), field_name="workers", failures=failures)
        if workers is not None:
            state.workers = workers
            result["changes"]["workers"] = workers
        result["applied"] = not failures
    else:
        failures.append(failure(Code.UNKNOWN_COMMAND, f"unsupported command op: {op}", field="op", command_index=index, allowed=sorted(MUTATING_OPS | {"run_batch"})))
    return result


def _session_and_turn(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]):
    failures = result["failures"]
    session_id = command.get("session_id")
    turn_id = command.get("turn_id")
    session_id = nonempty_str(session_id, field_name="session_id", failures=failures)
    turn_id = nonempty_str(turn_id, field_name="turn_id", failures=failures, session_id=session_id)
    if session_id is None or turn_id is None:
        return None, None
    if session_id not in state.turns:
        failures.append(failure(Code.COMMAND_TARGET_MISSING, "session not found", field="session_id", session_id=session_id, turn_id=turn_id, command_index=result["command_index"]))
        return session_id, None
    turn = state.find_turn(session_id, turn_id)
    if turn is None:
        failures.append(failure(Code.COMMAND_TARGET_MISSING, "turn not found", field="turn_id", session_id=session_id, turn_id=turn_id, command_index=result["command_index"]))
    return session_id, turn


def _add_turn(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]) -> None:
    failures = result["failures"]
    session_id = nonempty_str(command.get("session_id"), field_name="session_id", failures=failures)
    turn_payload = command.get("turn", command)
    if not isinstance(turn_payload, dict):
        failures.append(failure(Code.FIELD_TYPE, "turn must be an object", field="turn", session_id=session_id))
        return
    if session_id is None:
        return
    if session_id not in state.turns:
        failures.append(failure(Code.COMMAND_TARGET_MISSING, "session not found", field="session_id", session_id=session_id))
        return
    turn_id = nonempty_str(turn_payload.get("turn_id", turn_payload.get("id", command.get("turn_id"))), field_name="turn.turn_id", failures=failures, session_id=session_id)
    query = nonempty_str(turn_payload.get("query"), field_name="turn.query", failures=failures, session_id=session_id, turn_id=turn_id)
    if turn_id is None or query is None:
        return
    if state.find_turn(session_id, turn_id) is not None:
        failures.append(failure(Code.DUPLICATE_ID, "turn already exists", field="turn_id", session_id=session_id, turn_id=turn_id))
        return
    existing = state.sorted_turns(session_id)
    next_seq = max((turn.seq for turn in existing), default=0) + 1
    seq_value = turn_payload.get("seq", command.get("seq", next_seq))
    seq = positive_int(seq_value, field_name="turn.seq", failures=failures, session_id=session_id, turn_id=turn_id)
    if seq is None:
        return
    if seq != next_seq:
        failures.append(failure(Code.TURN_SEQ_GAP, "add_turn only appends and must preserve gap-free seq", field="seq", session_id=session_id, turn_id=turn_id, expected_seq=next_seq, seq=seq))
        return
    turn = Turn(session_id, turn_id, seq, query)
    state.turns[session_id].append(turn)
    state.hits[(session_id, turn_id)] = []
    hits = turn_payload.get("hits", turn_payload.get("channel_hits", command.get("hits", [])))
    _load_hit_rows(hits, state, failures, session_id, turn_id, "add_turn.hits")
    if failures:
        state.turns[session_id] = [item for item in state.turns[session_id] if item.turn_id != turn_id]
        state.hits.pop((session_id, turn_id), None)
        return
    result["applied"] = True
    result["changes"] = {"added_turn": turn_id, "seq": seq}


def _replace_query(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]) -> None:
    session_id, turn = _session_and_turn(state, command, result)
    if turn is None:
        return
    query = nonempty_str(command.get("query"), field_name="query", failures=result["failures"], session_id=session_id, turn_id=turn.turn_id)
    if query is None:
        return
    old = turn.query
    turn.query = query
    result["applied"] = True
    result["changes"] = {"turn_id": turn.turn_id, "old_query": old, "new_query": query}


def _delete_turn(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]) -> None:
    session_id, turn = _session_and_turn(state, command, result)
    if turn is None:
        return
    before = state.sorted_turns(session_id)
    resequence = command.get("resequence", True)
    state.turns[session_id] = [item for item in state.turns[session_id] if item.turn_id != turn.turn_id]
    state.hits.pop((session_id, turn.turn_id), None)
    mapping = {}
    if resequence is not False:
        for new_seq, item in enumerate(state.sorted_turns(session_id), start=1):
            if item.seq != new_seq:
                mapping[item.turn_id] = {"old_seq": item.seq, "new_seq": new_seq}
                item.seq = new_seq
    result["applied"] = True
    result["changes"] = {"deleted_turn": turn.turn_id, "resequenced": mapping}


def _set_topic_rule(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]) -> None:
    failures = result["failures"]
    rule = command.get("topic", command)
    if not isinstance(rule, dict):
        failures.append(failure(Code.FIELD_TYPE, "topic rule must be an object", field="topic"))
        return
    if "similarity_threshold" in rule or "threshold" in rule:
        threshold = decimal_value(
            rule.get("similarity_threshold", rule.get("threshold")),
            field_name="topic.similarity_threshold",
            failures=failures,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if threshold is not None:
            state.topic_threshold = threshold
            result["changes"]["similarity_threshold"] = threshold
    equal_rule = rule.get("equal_rule", command.get("equal_rule"))
    if equal_rule is not None:
        if equal_rule not in ("merge", "split"):
            failures.append(failure(Code.FIELD_NOT_ALLOWED, "equal_rule must be merge or split", field="topic.equal_rule", allowed=["merge", "split"], rejected_value=equal_rule))
        else:
            state.equal_rule = equal_rule
            result["changes"]["equal_rule"] = equal_rule
    result["applied"] = not failures


def _set_weight(state: RuntimeState, command: dict[str, Any], result: dict[str, Any], kind: str) -> None:
    failures = result["failures"]
    target_id = command.get(f"{kind}_id")
    target_id = nonempty_str(target_id, field_name=f"{kind}_id", failures=failures)
    if target_id is None:
        return
    registry = state.channels if kind == "channel" else state.evidence
    if target_id not in registry:
        failures.append(failure(Code.COMMAND_TARGET_MISSING, f"{kind} not found", field=f"{kind}_id", **{f"{kind}_id": target_id}))
        return
    weight = decimal_value(command.get("weight"), field_name="weight", failures=failures)
    if weight is None:
        return
    old = registry[target_id].weight
    registry[target_id].weight = weight
    result["applied"] = True
    result["changes"] = {f"{kind}_id": target_id, "old_weight": old, "new_weight": weight}


def _set_fixed_point(state: RuntimeState, command: dict[str, Any], result: dict[str, Any]) -> None:
    failures = result["failures"]
    fixed = command.get("fixed_point", command)
    width_value = fixed.get("fixed_width", fixed.get("width", state.fixed_width))
    gear_value = fixed.get("quant_gear", fixed.get("gear", state.quant_gear))
    try:
        width = parse_width(width_value)
    except (ValueError, TypeError):
        failures.append(failure(Code.FIELD_NOT_ALLOWED, "fixed_width must be one of 8,16,32/F8/F16/F32", field="fixed_width", rejected_value=str(width_value)))
        width = state.fixed_width
    try:
        gear = parse_gear(gear_value)
    except (ValueError, TypeError):
        failures.append(failure(Code.FIELD_NOT_ALLOWED, "quant_gear must be one of G0,G1,G2,G3", field="quant_gear", rejected_value=str(gear_value)))
        gear = state.quant_gear
    if failures:
        return
    state.fixed_width = width
    state.quant_gear = gear
    result["applied"] = True
    result["changes"] = {"fixed_width": f"F{width}", "quant_gear": gear}
