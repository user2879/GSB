from __future__ import annotations

import copy
from typing import Any


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _find_token_index(state: dict[str, Any], command: dict[str, Any]) -> int:
    if command.get("token_uid") is not None:
        uid = command["token_uid"]
        for index, token in enumerate(state["tokens"]):
            if token.get("uid") == uid:
                return index
        raise KeyError(f"unknown token_uid: {uid}")
    seq = int(command.get("token_seq", command.get("seq", -1)))
    for index, token in enumerate(state["tokens"]):
        if int(token.get("seq", index)) == seq or index == seq:
            return index
    raise KeyError(f"unknown token_seq: {seq}")


def _repair_boundaries_after_delete(boundaries: list[int], deleted_index: int) -> list[int]:
    counts: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        count = max(0, end - start - (1 if start <= deleted_index < end else 0))
        if count:
            counts.append(count)
    repaired = [0]
    for count in counts:
        repaired.append(repaired[-1] + count)
    if repaired[-1] != max(0, boundaries[-1] - 1):
        repaired[-1] = max(0, boundaries[-1] - 1)
    return repaired


def _renumber_operators(operators: list[dict[str, Any]]) -> None:
    ordered = sorted(operators, key=lambda op: int(op.get("seq", 0)))
    old_to_new = {int(op["seq"]): index for index, op in enumerate(ordered)}
    for index, op in enumerate(ordered):
        old_inputs = op.get("inputs", [])
        op["inputs"] = sorted({old_to_new[int(ref)] for ref in old_inputs if int(ref) in old_to_new and old_to_new[int(ref)] != index})
        op["seq"] = index
    operators[:] = ordered


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_command(state: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    command = dict(command)
    op_name = command.get("op") or command.get("command")
    summary = {"op": op_name}

    if op_name == "replace_token":
        index = _find_token_index(state, command)
        old = state["tokens"][index].get("token")
        new_value = command.get("token", command.get("token_id", command.get("value")))
        state["tokens"][index]["token"] = int(new_value)
        summary.update(token_uid=state["tokens"][index]["uid"], old=old, new=int(new_value))

    elif op_name in ("delete_token", "remove_token"):
        index = _find_token_index(state, command)
        removed = state["tokens"].pop(index)
        state["boundaries"] = _repair_boundaries_after_delete(list(state["boundaries"]), index)
        for new_index, token in enumerate(state["tokens"]):
            token["seq"] = new_index
        summary.update(token_uid=removed.get("uid"), token_seq=index)

    elif op_name == "add_token":
        token = command["token"]
        state["tokens"].append(token)
        for index, item in enumerate(state["tokens"]):
            item["seq"] = index
        if state["boundaries"][-1] == len(state["tokens"]) - 1:
            state["boundaries"][-1] = len(state["tokens"])
        else:
            state["boundaries"][-1] = len(state["tokens"])
        summary.update(token_uid=token.get("uid"))

    elif op_name == "set_segments":
        boundaries = command.get("boundaries")
        if boundaries is None and "segments" in command:
            boundaries = [0]
            for segment in command["segments"]:
                boundaries.append(int(segment["end"]))
        state["boundaries"] = [int(value) for value in boundaries]
        summary.update(boundaries=state["boundaries"])

    elif op_name == "update_op_param":
        op_seq = int(command["op_seq"])
        op = next(item for item in state["operators"] if int(item["seq"]) == op_seq)
        if command.get("path"):
            _set_nested(op, command["path"], command.get("value"))
            if command["path"] == "scale_float":
                from fractions import Fraction

                scaled = Fraction(str(command["value"])) * 1000
                op["scale_permille"] = int(scaled + Fraction(1, 2))
        if isinstance(command.get("params"), dict):
            op.setdefault("params", {}).update(command["params"])
        if isinstance(command.get("values"), dict):
            op.update(command["values"])
        summary.update(op_seq=op_seq, path=command.get("path"))

    elif op_name in ("add_operator", "insert_operator"):
        operator = copy.deepcopy(command["operator"])
        state["operators"].append(operator)
        _renumber_operators(state["operators"])
        summary.update(operator_count=len(state["operators"]))

    elif op_name in ("remove_operator", "delete_operator"):
        op_seq = int(command["op_seq"])
        state["operators"] = [op for op in state["operators"] if int(op["seq"]) != op_seq]
        _renumber_operators(state["operators"])
        for chain_key in ("chain", "chain_e2"):
            if isinstance(state.get(chain_key), dict):
                state[chain_key]["operators"] = state["operators"]
        summary.update(removed_op_seq=op_seq, operator_count=len(state["operators"]))

    elif op_name == "set_quant":
        if "quant_tier" in command:
            state["config"]["quant_tier"] = command["quant_tier"]
        if "frac_bits" in command:
            state["config"]["frac_bits"] = int(command["frac_bits"])
        summary.update(quant_tier=state["config"].get("quant_tier"))

    elif op_name == "set_frac_bits":
        state["config"]["frac_bits"] = int(command["frac_bits"])
        summary.update(frac_bits=state["config"]["frac_bits"])

    elif op_name in ("set_version", "switch_version"):
        version = command.get("exchange_version", command.get("version"))
        version = str(version).lower().split("-")[0]
        state["config"]["exchange_version"] = version
        state["exchange_version"] = version
        summary.update(exchange_version=version)

    elif op_name == "set_limits":
        for key in ("point_budget", "operator_quota", "max_workers"):
            if key in command:
                state["config"][key] = command[key]
        summary.update(limits={key: state["config"].get(key) for key in ("point_budget", "operator_quota", "max_workers")})

    else:
        raise ValueError(f"unknown edit command: {op_name}")
    return summary


def apply_commands(state: dict[str, Any], commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for command in commands:
        summaries.append(apply_command(state, command))
    return summaries
