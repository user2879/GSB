from __future__ import annotations

import copy
from typing import Any

from .errors import SchemaViolation
from .schema import is_int, issue, validate_bundle


VALID_OPS = {
    "replace_ticket",
    "upsert_neg_trigger",
    "remove_neg_trigger",
    "upsert_conflict",
    "remove_conflict",
    "set_channel_weight",
    "update_config",
}


def validate_command_shape(command: Any, index: int) -> list[dict[str, Any]]:
    ident = command.get("seq", index) if isinstance(command, dict) else index
    if not isinstance(command, dict):
        return [issue("edit_command", index, "$", "command must be an object")]
    issues: list[dict[str, Any]] = []
    op = command.get("op")
    if op not in VALID_OPS:
        issues.append(issue("edit_command", ident, "op", f"must be one of {sorted(VALID_OPS)}"))
        return issues
    if op == "replace_ticket" and not isinstance(command.get("ticket"), dict):
        issues.append(issue("edit_command", ident, "ticket", "replace_ticket requires a ticket object"))
    if op in {"upsert_neg_trigger", "upsert_conflict"} and not isinstance(command.get("entry"), dict) and not isinstance(
        command.get("trigger") if op == "upsert_neg_trigger" else command.get("conflict"), dict
    ):
        issues.append(issue("edit_command", ident, "entry", f"{op} requires an entry object"))
    if op in {"remove_neg_trigger", "remove_conflict"} and not command.get("entry_id"):
        issues.append(issue("edit_command", ident, "entry_id", f"{op} requires entry_id"))
    if op == "set_channel_weight":
        if not isinstance(command.get("channel_id"), str):
            issues.append(issue("edit_command", ident, "channel_id", "must be a string"))
        weight = command.get("weight_permille")
        if not is_int(weight) or not 0 <= weight <= 1000:
            issues.append(issue("edit_command", ident, "weight_permille", "must be an integer in [0, 1000]"))
    if op == "update_config" and not isinstance(command.get("patch"), dict):
        issues.append(issue("edit_command", ident, "patch", "update_config requires a patch object"))
    return issues


def validate_command_stream(commands: list[Any]) -> None:
    issues: list[dict[str, Any]] = []
    for index, command in enumerate(commands, 1):
        issues.extend(validate_command_shape(command, index))
    if issues:
        raise SchemaViolation(issues)


def deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def reindex_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    bundle["vocab"] = {row["term"]: row for row in bundle["vocab_list"]}
    bundle["triggers"] = {row["term"]: dict(row) for row in bundle["triggers_doc"]["triggers"]}
    bundle["trigger_ids"] = {row["trigger_id"]: dict(row) for row in bundle["triggers_doc"]["triggers"]}
    bundle["conflicts"] = {row["term"]: dict(row) for row in bundle["conflicts_doc"]["conflicts"]}
    bundle["conflict_ids"] = {row["conflict_id"]: dict(row) for row in bundle["conflicts_doc"]["conflicts"]}
    bundle["channels"] = {
        row["channel_id"]: int(row["weight_permille"]) for row in bundle["channels_doc"]["channels"]
    }
    return bundle


def _entry_from_command(command: dict[str, Any]) -> dict[str, Any]:
    if "entry" in command:
        return copy.deepcopy(command["entry"])
    key = "trigger" if command["op"] == "upsert_neg_trigger" else "conflict"
    return copy.deepcopy(command[key])


def apply_command(bundle: dict[str, Any], command: dict[str, Any], *, increment_version: bool = True) -> dict[str, Any]:
    op = command["op"]
    if op == "replace_ticket":
        ticket = copy.deepcopy(command["ticket"])
        tickets = bundle["tickets_doc"]["tickets"]
        for index, existing in enumerate(tickets):
            if existing["ticket_id"] == ticket["ticket_id"]:
                tickets[index] = ticket
                break
        else:
            tickets.append(ticket)
        bundle["tickets"] = tickets
    elif op == "upsert_neg_trigger":
        entry = _entry_from_command(command)
        rows = bundle["triggers_doc"]["triggers"]
        for index, row in enumerate(rows):
            if row["trigger_id"] == entry["trigger_id"]:
                rows[index] = entry
                break
        else:
            rows.append(entry)
    elif op == "remove_neg_trigger":
        entry_id = command["entry_id"]
        bundle["triggers_doc"]["triggers"] = [
            row for row in bundle["triggers_doc"]["triggers"] if row["trigger_id"] != entry_id
        ]
    elif op == "upsert_conflict":
        entry = _entry_from_command(command)
        rows = bundle["conflicts_doc"]["conflicts"]
        for index, row in enumerate(rows):
            if row["conflict_id"] == entry["conflict_id"]:
                rows[index] = entry
                break
        else:
            rows.append(entry)
    elif op == "remove_conflict":
        entry_id = command["entry_id"]
        bundle["conflicts_doc"]["conflicts"] = [
            row for row in bundle["conflicts_doc"]["conflicts"] if row["conflict_id"] != entry_id
        ]
    elif op == "set_channel_weight":
        channel_id = command["channel_id"]
        for row in bundle["channels_doc"]["channels"]:
            if row["channel_id"] == channel_id:
                row["weight_permille"] = int(command["weight_permille"])
                break
        else:
            bundle["channels_doc"]["channels"].append({
                "channel_id": channel_id,
                "weight_permille": int(command["weight_permille"]),
            })
    elif op == "update_config":
        deep_merge(bundle["config"], copy.deepcopy(command["patch"]))
    else:  # pragma: no cover - shape validation prevents this
        raise SchemaViolation([issue("edit_command", command.get("seq"), "op", "unsupported op")])

    if increment_version:
        bundle["config"]["global_version"] = int(bundle["config"].get("global_version", 0)) + 1
    reindex_bundle(bundle)
    validate_bundle(bundle)
    return bundle


def dry_apply_command(bundle: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(bundle)
    validate_command_stream([command])
    apply_command(candidate, command)
    return candidate
