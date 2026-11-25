"""Loading and deterministic validation for all local JSON declarations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import (
    E_FORMAT_INVALID,
    E_REVISION_CHAIN,
    E_VALIDATION,
    E_VERSION_DECREASED,
    Issue,
    PipelineValidationError,
)


def load_json_file(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(float(value))


def validate_lexicon(data: Any) -> list[Issue]:
    issues: list[Issue] = []
    if not isinstance(data, dict):
        return [Issue(E_FORMAT_INVALID, "lexicon root must be an object", path="$")]

    tags = data.get("pos_tags")
    tag_names: set[str] = set()
    if not isinstance(tags, list) or not tags:
        issues.append(Issue(E_FORMAT_INVALID, "pos_tags must be a non-empty list", path="pos_tags"))
    else:
        seen_codes: set[int] = set()
        for index, tag in enumerate(tags):
            path = f"pos_tags[{index}]"
            if not isinstance(tag, dict):
                issues.append(Issue(E_FORMAT_INVALID, "POS tag must be an object", path=path))
                continue
            name = tag.get("name")
            code = tag.get("code")
            if not isinstance(name, str) or not name:
                issues.append(Issue(E_FORMAT_INVALID, "POS name must be a non-empty string", path=f"{path}.name"))
            else:
                tag_names.add(name)
            if not is_int(code):
                issues.append(Issue(E_FORMAT_INVALID, "POS code must be an integer", path=f"{path}.code"))
            elif code in seen_codes:
                issues.append(Issue(E_VALIDATION, f"duplicate POS code {code}", path=f"{path}.code"))
            else:
                seen_codes.add(code)

    entries = data.get("entries")
    if not isinstance(entries, list):
        issues.append(Issue(E_FORMAT_INVALID, "entries must be a list", path="entries"))
        return issues

    seen_terms: set[str] = set()
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"entries[{index}]"
        if not isinstance(entry, dict):
            issues.append(Issue(E_FORMAT_INVALID, "entry must be an object", path=path))
            continue
        entry_id = entry.get("id")
        term = entry.get("term")
        if not isinstance(entry_id, str) or not entry_id:
            issues.append(Issue(E_FORMAT_INVALID, "entry id must be a non-empty string", path=f"{path}.id"))
        elif entry_id in seen_ids:
            issues.append(Issue(E_VALIDATION, f"duplicate entry id {entry_id}", path=f"{path}.id"))
        else:
            seen_ids.add(entry_id)

        if not isinstance(term, str) or not term:
            issues.append(Issue(E_FORMAT_INVALID, "term must be a non-empty string", path=f"{path}.term"))
        elif term in seen_terms:
            issues.append(Issue(E_VALIDATION, f"term {term!r} is not unique", field="term", path=f"{path}.term"))
        else:
            seen_terms.add(term)

        frequency = entry.get("frequency")
        if not is_int(frequency) or frequency <= 0:
            issues.append(Issue(E_VALIDATION, "frequency must be a positive integer", field="frequency", path=f"{path}.frequency"))

        version = entry.get("version")
        if not is_int(version) or version < 0:
            issues.append(Issue(E_VALIDATION, "version must be a non-negative integer", field="version", path=f"{path}.version"))

        enabled = entry.get("enabled")
        if not isinstance(enabled, bool):
            issues.append(Issue(E_FORMAT_INVALID, "enabled must be a boolean", field="enabled", path=f"{path}.enabled"))

        pos = entry.get("pos")
        if not isinstance(pos, str) or (tag_names and pos not in tag_names):
            issues.append(Issue(E_VALIDATION, f"pos {pos!r} is not in declared POS set", field="pos", path=f"{path}.pos"))

        pos_tags = entry.get("pos_tags", [pos] if isinstance(pos, str) else [])
        if not isinstance(pos_tags, list) or not pos_tags or not all(isinstance(item, str) for item in pos_tags):
            issues.append(Issue(E_FORMAT_INVALID, "pos_tags must be a list of strings", path=f"{path}.pos_tags"))
        elif tag_names:
            for tag in pos_tags:
                if tag not in tag_names:
                    issues.append(Issue(E_VALIDATION, f"POS tag {tag!r} is not declared", path=f"{path}.pos_tags"))
            if isinstance(pos, str) and pos not in pos_tags:
                issues.append(Issue(E_VALIDATION, "primary pos must be included in pos_tags", path=f"{path}.pos_tags"))

        weights = entry.get("tag_weights", {})
        if weights is not None:
            if not isinstance(weights, dict):
                issues.append(Issue(E_FORMAT_INVALID, "tag_weights must be an object", path=f"{path}.tag_weights"))
            else:
                for tag, weight in weights.items():
                    if not finite_number(weight):
                        issues.append(Issue(E_VALIDATION, f"tag weight for {tag} must be finite", path=f"{path}.tag_weights.{tag}"))

    return issues


def validate_documents(data: Any) -> list[Issue]:
    issues: list[Issue] = []
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        return [Issue(E_FORMAT_INVALID, "documents root must contain a documents list", path="documents")]
    seen: set[str] = set()
    for index, doc in enumerate(data["documents"]):
        path = f"documents[{index}]"
        if not isinstance(doc, dict):
            issues.append(Issue(E_FORMAT_INVALID, "document must be an object", path=path))
            continue
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            issues.append(Issue(E_FORMAT_INVALID, "doc_id must be a non-empty string", path=f"{path}.doc_id"))
        elif doc_id in seen:
            issues.append(Issue(E_VALIDATION, f"duplicate doc_id {doc_id}", path=f"{path}.doc_id"))
        else:
            seen.add(doc_id)
        if not is_int(doc.get("doc_seq")) or doc["doc_seq"] <= 0:
            issues.append(Issue(E_FORMAT_INVALID, "doc_seq must be a positive integer", path=f"{path}.doc_seq"))
        if not isinstance(doc.get("text"), str):
            issues.append(Issue(E_FORMAT_INVALID, "text must be a string", path=f"{path}.text"))
        if "term_count" in doc and (not is_int(doc["term_count"]) or doc["term_count"] <= 0):
            issues.append(Issue(E_FORMAT_INVALID, "term_count must be a positive integer when supplied", path=f"{path}.term_count"))
    return issues


def validate_weights(data: Any) -> list[Issue]:
    if not isinstance(data, dict):
        return [Issue(E_FORMAT_INVALID, "weights root must be an object", path="$")]
    issues: list[Issue] = []
    for field in ("transition_weights",):
        table = data.get(field, {})
        if not isinstance(table, dict):
            issues.append(Issue(E_FORMAT_INVALID, f"{field} must be an object", path=field))
            continue
        for source, row in table.items():
            if not isinstance(row, dict):
                issues.append(Issue(E_FORMAT_INVALID, f"transition row {source} must be an object", path=f"{field}.{source}"))
                continue
            for target, weight in row.items():
                if not finite_number(weight):
                    issues.append(Issue(E_VALIDATION, f"transition weight {source}->{target} must be finite", path=f"{field}.{source}.{target}"))
    if "default_transition_weight" in data and not finite_number(data["default_transition_weight"]):
        issues.append(Issue(E_VALIDATION, "default_transition_weight must be finite", path="default_transition_weight"))
    return issues


def validate_rules(data: Any) -> list[Issue]:
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        return [Issue(E_FORMAT_INVALID, "rules root must contain a rules list", path="rules")]
    issues: list[Issue] = []
    allowed_actions = {"split", "merge"}
    for index, rule in enumerate(data["rules"]):
        path = f"rules[{index}]"
        if not isinstance(rule, dict):
            issues.append(Issue(E_FORMAT_INVALID, "rule must be an object", path=path))
            continue
        if not isinstance(rule.get("rule_id"), str) or not rule["rule_id"]:
            issues.append(Issue(E_FORMAT_INVALID, "rule_id must be a non-empty string", path=f"{path}.rule_id"))
        if not is_int(rule.get("version")):
            issues.append(Issue(E_FORMAT_INVALID, "rule version must be an integer", path=f"{path}.version"))
        if not is_int(rule.get("order")):
            issues.append(Issue(E_FORMAT_INVALID, "rule order must be an integer", path=f"{path}.order"))
        if rule.get("action") not in allowed_actions:
            issues.append(Issue(E_FORMAT_INVALID, "rule action must be split or merge", path=f"{path}.action"))
        if not is_int(rule.get("position")) or rule["position"] < 0:
            issues.append(Issue(E_FORMAT_INVALID, "rule position must be a non-negative integer", path=f"{path}.position"))
        if "doc_id" in rule and not isinstance(rule["doc_id"], str):
            issues.append(Issue(E_FORMAT_INVALID, "doc_id must be a string", path=f"{path}.doc_id"))
        if "enabled" in rule and not isinstance(rule["enabled"], bool):
            issues.append(Issue(E_FORMAT_INVALID, "enabled must be boolean", path=f"{path}.enabled"))
    return issues


def validate_revision_chain(data: Any) -> list[Issue]:
    if not isinstance(data, dict) or not isinstance(data.get("revisions"), list):
        return [Issue(E_REVISION_CHAIN, "revisions root must contain a revisions list", path="revisions")]
    revisions = data["revisions"]
    by_id: dict[int, Any] = {}
    issues: list[Issue] = []
    for index, rev in enumerate(revisions):
        path = f"revisions[{index}]"
        if not isinstance(rev, dict) or not is_int(rev.get("revision")):
            issues.append(Issue(E_REVISION_CHAIN, "revision must have integer revision", path=path))
            continue
        revision = rev["revision"]
        if revision in by_id:
            issues.append(Issue(E_REVISION_CHAIN, f"duplicate revision {revision}", path=f"{path}.revision"))
        by_id[revision] = rev
        parent = rev.get("parent_revision")
        if revision == 0:
            if parent is not None and not is_int(parent):
                issues.append(Issue(E_REVISION_CHAIN, "root parent_revision must be null or integer", path=f"{path}.parent_revision"))
        elif not is_int(parent):
            issues.append(Issue(E_REVISION_CHAIN, "non-root revision must explicitly reference an integer parent_revision", path=f"{path}.parent_revision"))

    for revision, rev in by_id.items():
        if revision == 0:
            continue
        parent = rev.get("parent_revision")
        if is_int(parent) and parent not in by_id:
            issues.append(Issue(E_REVISION_CHAIN, f"revision {revision} references missing parent {parent}", field="parent_revision"))

    visiting: set[int] = set()
    visited: set[int] = set()

    def has_cycle(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited or node not in by_id:
            return False
        visiting.add(node)
        parent = by_id[node].get("parent_revision")
        cycle = is_int(parent) and has_cycle(parent)
        visiting.remove(node)
        visited.add(node)
        return cycle

    for revision in by_id:
        if has_cycle(revision):
            issues.append(Issue(E_REVISION_CHAIN, f"revision chain contains a cycle at {revision}", field="parent_revision"))
            break
    return issues


def validate_command_stream(data: Any) -> list[Issue]:
    if not isinstance(data, dict) or not isinstance(data.get("commands"), list):
        return [Issue(E_FORMAT_INVALID, "command stream must contain commands list", path="commands")]
    issues: list[Issue] = []
    required = {"seq", "op"}
    for index, command in enumerate(data["commands"]):
        path = f"commands[{index}]"
        if not isinstance(command, dict) or not required.issubset(command):
            issues.append(Issue(E_FORMAT_INVALID, "command requires seq and op", path=path))
            continue
        if not is_int(command["seq"]) or command["seq"] <= 0:
            issues.append(Issue(E_FORMAT_INVALID, "seq must be positive integer", path=f"{path}.seq"))
        if not isinstance(command["op"], str):
            issues.append(Issue(E_FORMAT_INVALID, "op must be string", path=f"{path}.op"))
    return issues


def assert_valid(issues: list[Issue]) -> None:
    if issues:
        raise PipelineValidationError(issues)


def load_and_validate_inputs(sample_dir: str | Path) -> dict[str, Any]:
    root = Path(sample_dir)
    loaded = {
        "lexicon": load_json_file(root / "lexicon.json"),
        "documents": load_json_file(root / "documents.json"),
        "weights": load_json_file(root / "weights.json"),
        "rules": load_json_file(root / "rules.json"),
        "revisions": load_json_file(root / "revisions.json"),
        "commands": load_json_file(root / "commands.json"),
    }
    issues = (
        validate_lexicon(loaded["lexicon"])
        + validate_documents(loaded["documents"])
        + validate_weights(loaded["weights"])
        + validate_rules(loaded["rules"])
        + validate_revision_chain(loaded["revisions"])
        + validate_command_stream(loaded["commands"])
    )
    assert_valid(issues)
    return loaded


def ensure_entry_version_monotonic(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> list[Issue]:
    if existing is not None and is_int(incoming.get("version")) and is_int(existing.get("version")):
        if incoming["version"] < existing["version"]:
            return [
                Issue(
                    E_VERSION_DECREASED,
                    f"entry {incoming.get('id')} version {incoming['version']} is below {existing['version']}",
                    field="version",
                )
            ]
    return []
