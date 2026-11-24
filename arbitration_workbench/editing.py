"""Deterministic edit commands for evidence and rule package files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .codes import BusinessFault, E_EDIT_COMMAND


def load_edit_commands(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessFault(E_EDIT_COMMAND, "edit command file is unreadable", "global") from exc
    if isinstance(raw, dict):
        raw = raw.get("commands")
    if not isinstance(raw, list):
        raise BusinessFault(E_EDIT_COMMAND, "edit command file must contain an array", "global")
    commands: list[dict[str, Any]] = []
    for index, command in enumerate(raw):
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            raise BusinessFault(E_EDIT_COMMAND, f"edit command {index} is invalid", "global")
        commands.append(command)
    return commands


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessFault(E_EDIT_COMMAND, f"cannot read {path.name}", "global") from exc
    if not isinstance(value, dict):
        raise BusinessFault(E_EDIT_COMMAND, f"{path.name} root must be an object", "global")
    return value


def _read_evidence(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BusinessFault(E_EDIT_COMMAND, "cannot read evidence.jsonl", "global") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BusinessFault(E_EDIT_COMMAND, "evidence.jsonl is not valid JSONL", "global") from exc
        if not isinstance(row, dict):
            raise BusinessFault(E_EDIT_COMMAND, "evidence record must be an object", "global")
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("workorder_id", "")), int(row.get("evidence_seq", 0))))
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_evidence(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (str(row.get("workorder_id", "")), int(row.get("evidence_seq", 0))))
    text = "".join(canonical_json(row) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _write_rules(path: Path, rules: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(rules, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _next_input_version(data_dir: Path) -> int:
    path = data_dir / "input_meta.json"
    if path.exists():
        raw = _read_json(path)
        version = raw.get("version", 0)
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise BusinessFault(E_EDIT_COMMAND, "input_meta.json version is invalid", "global")
    else:
        version = 0
    next_version = version + 1
    _atomic_write_text(
        path,
        json.dumps({"version": next_version}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return next_version


def _relation_kind(command: dict[str, Any]) -> str:
    relation = command.get("relation")
    if not isinstance(relation, dict):
        raise BusinessFault(E_EDIT_COMMAND, "add_relation requires a relation object", "global")
    raw_kind = command.get("kind", relation.get("kind"))
    kind = str(raw_kind or "").strip().lower()
    if kind in {"mutex", "exclusion", "mutual-exclusion", "mutual_exclusion"}:
        return "mutual_exclusion"
    if kind in {"support", "bonus", "support_bonus"}:
        return "support"
    raise BusinessFault(E_EDIT_COMMAND, "relation kind must be mutual_exclusion or support", "global")


def apply_edit(data_dir: Path, command: dict[str, Any]) -> dict[str, Any]:
    """Apply one edit command and increment the input version."""

    data_dir = Path(data_dir)
    op = command.get("op")
    evidence_path = data_dir / "evidence.jsonl"
    rules_path = data_dir / "rules.json"
    evidence = _read_evidence(evidence_path)
    rules = _read_json(rules_path)
    changed = False
    detail: dict[str, Any] = {"op": op}

    if op == "replace_evidence":
        workorder_id = command.get("workorder_id")
        evidence_seq = command.get("evidence_seq")
        replacement = command.get("evidence")
        if not isinstance(replacement, dict):
            raise BusinessFault(E_EDIT_COMMAND, "replace_evidence requires an evidence object", "global")
        replacement = dict(replacement)
        replacement.setdefault("workorder_id", workorder_id)
        replacement.setdefault("evidence_seq", evidence_seq)
        if replacement.get("workorder_id") != workorder_id or replacement.get("evidence_seq") != evidence_seq:
            raise BusinessFault(E_EDIT_COMMAND, "replacement workorder_id/evidence_seq does not match command", "global")
        for index, row in enumerate(evidence):
            if row.get("workorder_id") == workorder_id and row.get("evidence_seq") == evidence_seq:
                evidence[index] = replacement
                changed = True
                break
        if not changed:
            raise BusinessFault(E_EDIT_COMMAND, "evidence to replace was not found", "global")
        _write_evidence(evidence_path, evidence)
        detail.update({"workorder_id": workorder_id, "evidence_seq": evidence_seq})

    elif op == "delete_evidence":
        workorder_id = command.get("workorder_id")
        evidence_seq = command.get("evidence_seq")
        filtered = [
            row
            for row in evidence
            if not (row.get("workorder_id") == workorder_id and row.get("evidence_seq") == evidence_seq)
        ]
        if len(filtered) == len(evidence):
            raise BusinessFault(E_EDIT_COMMAND, "evidence to delete was not found", "global")
        _write_evidence(evidence_path, filtered)
        detail.update({"workorder_id": workorder_id, "evidence_seq": evidence_seq})

    elif op == "set_evidence_type_weight":
        evidence_type = command.get("evidence_type")
        if not isinstance(evidence_type, str) or not evidence_type:
            raise BusinessFault(E_EDIT_COMMAND, "set_evidence_type_weight requires evidence_type", "global")
        weights = rules.setdefault("evidence_type_weights", {})
        if not isinstance(weights, dict):
            raise BusinessFault(E_EDIT_COMMAND, "evidence_type_weights is not an object", "global")
        weights[evidence_type] = command.get("value")
        _write_rules(rules_path, rules)
        detail.update({"evidence_type": evidence_type, "value": command.get("value")})

    elif op == "set_role_credibility":
        source_role = command.get("source_role")
        if not isinstance(source_role, str) or not source_role:
            raise BusinessFault(E_EDIT_COMMAND, "set_role_credibility requires source_role", "global")
        roles = rules.setdefault("source_role_credibility", {})
        if not isinstance(roles, dict):
            raise BusinessFault(E_EDIT_COMMAND, "source_role_credibility is not an object", "global")
        roles[source_role] = command.get("value")
        _write_rules(rules_path, rules)
        detail.update({"source_role": source_role, "value": command.get("value")})

    elif op == "add_relation":
        kind = _relation_kind(command)
        relation = dict(command["relation"])
        relation.pop("kind", None)
        if not isinstance(relation.get("id"), str) or not relation["id"]:
            raise BusinessFault(E_EDIT_COMMAND, "relation id is required", "global")
        relation_id = relation["id"]
        all_relations = list(rules.get("mutual_exclusions", [])) + list(rules.get("supports", [])) + list(
            rules.get("relations", [])
        )
        if any(item.get("id", item.get("relation_id")) == relation_id for item in all_relations if isinstance(item, dict)):
            raise BusinessFault(E_EDIT_COMMAND, "relation id already exists", "global")
        target_field = "mutual_exclusions" if kind == "mutual_exclusion" else "supports"
        rules.setdefault(target_field, []).append(relation)
        _write_rules(rules_path, rules)
        detail.update({"relation_id": relation_id, "kind": kind})

    elif op == "delete_relation":
        relation_id = command.get("relation_id", command.get("id"))
        if not isinstance(relation_id, str) or not relation_id:
            raise BusinessFault(E_EDIT_COMMAND, "delete_relation requires relation_id", "global")
        found = False
        for field in ("mutual_exclusions", "supports", "relations"):
            rows = rules.get(field)
            if not isinstance(rows, list):
                continue
            filtered = [
                item
                for item in rows
                if not (isinstance(item, dict) and item.get("id", item.get("relation_id")) == relation_id)
            ]
            if len(filtered) != len(rows):
                found = True
                rules[field] = filtered
        if not found:
            raise BusinessFault(E_EDIT_COMMAND, "relation to delete was not found", "global")
        _write_rules(rules_path, rules)
        detail.update({"relation_id": relation_id})

    elif op == "set_half_life_ms":
        rules["half_life_ms"] = command.get("value")
        _write_rules(rules_path, rules)
        detail.update({"value": command.get("value")})

    elif op == "set_priority_threshold":
        priority = command.get("priority")
        if not isinstance(priority, str) or not priority:
            raise BusinessFault(E_EDIT_COMMAND, "set_priority_threshold requires priority", "global")
        thresholds = rules.get("priority_thresholds")
        if isinstance(thresholds, dict):
            thresholds[priority] = command.get("min_score")
        elif isinstance(thresholds, list):
            found = False
            for item in thresholds:
                if isinstance(item, dict) and item.get("priority") == priority:
                    item["min_score"] = command.get("min_score")
                    found = True
                    break
            if not found:
                thresholds.append({"priority": priority, "min_score": command.get("min_score")})
        else:
            raise BusinessFault(E_EDIT_COMMAND, "priority_thresholds is missing or invalid", "global")
        _write_rules(rules_path, rules)
        detail.update({"priority": priority, "min_score": command.get("min_score")})

    elif op == "set_relation_strength":
        relation_id = command.get("relation_id", command.get("id"))
        if not isinstance(relation_id, str) or not relation_id:
            raise BusinessFault(E_EDIT_COMMAND, "set_relation_strength requires relation_id", "global")
        found = False
        for field in ("mutual_exclusions", "supports", "relations"):
            rows = rules.get(field)
            if not isinstance(rows, list):
                continue
            for item in rows:
                if isinstance(item, dict) and item.get("id", item.get("relation_id")) == relation_id:
                    item["strength"] = command.get("value")
                    found = True
        if not found:
            raise BusinessFault(E_EDIT_COMMAND, "relation was not found", "global")
        _write_rules(rules_path, rules)
        detail.update({"relation_id": relation_id, "value": command.get("value")})

    else:
        raise BusinessFault(E_EDIT_COMMAND, f"unsupported edit op {op}", "global")

    detail["input_version"] = _next_input_version(data_dir)
    return detail


def apply_edit_commands(data_dir: Path, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_edit(data_dir, command) for command in commands]
