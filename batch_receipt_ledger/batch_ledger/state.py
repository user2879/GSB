from __future__ import annotations

import copy
from typing import Any

from .codes import E_BATCH_SIZE_INVALID, E_WORKER_INVALID
from .inputs import deep_default_config


def new_state() -> dict[str, Any]:
    config = deep_default_config()
    return {
        "schema": "batch-receipt-workspace-v1",
        "global_version": 1,
        "config": config,
        "workers": [],
        "requests": [],
        "receipts": [],
        "sidecar": {"format": "receipt-sidecar-v1", "entries": {}},
        "baseline_worker_versions": {},
        "command_log": [],
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def apply_command(state: dict[str, Any], command: dict[str, Any], command_index: int | None = None) -> dict[str, Any]:
    """Apply one deterministic edit. Invalid edits return a business failure."""

    index = len(state.get("command_log", [])) if command_index is None else command_index
    kind = command.get("type")
    result: dict[str, Any] = {"command_index": index, "type": kind, "ok": True}

    if kind == "replace_receipt":
        match = command.get("match", {})
        replacement = command.get("receipt")
        replaced = 0
        for pos, receipt in enumerate(state["receipts"]):
            if all(receipt.get(key) == value for key, value in match.items()):
                new_receipt = copy.deepcopy(replacement)
                new_receipt.setdefault("_source", receipt.get("_source", "command:replace_receipt"))
                new_receipt["_record_index"] = receipt.get("_record_index", pos)
                state["receipts"][pos] = new_receipt
                replaced += 1
                break
        result["replaced"] = replaced
        if replaced == 0:
            result["ok"] = False
            result["code"] = "E_RECEIPT_NOT_FOUND"

    elif kind == "add_batch":
        batch = command["batch"]
        state["requests"].extend(copy.deepcopy(batch.get("requests", [])))
        start = len(state["receipts"])
        for offset, receipt in enumerate(batch.get("receipts", [])):
            item = copy.deepcopy(receipt)
            item.setdefault("_source", "command:add_batch")
            item["_record_index"] = start + offset
            state["receipts"].append(item)
        result["batch_id"] = batch.get("batch_id")
        result["receipt_count"] = len(batch.get("receipts", []))

    elif kind == "switch_param_version":
        worker_id = command.get("worker_id")
        version = command.get("param_version")
        for worker in state["workers"]:
            if worker.get("worker_id") == worker_id:
                if not _is_int(version) or version < 0:
                    result.update(ok=False, code=E_WORKER_INVALID)
                    break
                worker["param_version"] = int(version)
                if command.get("param_digest") is not None:
                    worker["param_digest"] = command["param_digest"]
                result["worker_id"] = worker_id
                result["param_version"] = version
                break
        else:
            result.update(ok=False, code=E_WORKER_INVALID, worker_id=worker_id)

    elif kind == "set_batch_size":
        size = command.get("batch_size")
        if not _is_int(size) or size <= 0:
            result.update(ok=False, code=E_BATCH_SIZE_INVALID, batch_size=size)
        else:
            state["config"]["batch_size"] = int(size)
            result["batch_size"] = int(size)

    elif kind == "set_merge_rule":
        rule = command.get("merge_rule")
        allowed = state["config"].get("allowed_merge_rules", [])
        if allowed and rule not in allowed:
            result.update(ok=False, code="E_MERGE_RULE_INVALID", merge_rule=rule)
        else:
            state["config"]["merge_rule"] = rule
            result["merge_rule"] = rule

    else:
        result.update(ok=False, code="E_COMMAND_UNKNOWN")

    logged = copy.deepcopy(command)
    logged["_result"] = {key: value for key, value in result.items() if key != "command_index"}
    state.setdefault("command_log", []).append(logged)
    return result


def apply_commands(state: dict[str, Any], commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_command(state, command, index) for index, command in enumerate(commands)]


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)
