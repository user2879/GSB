"""Loading and validation for deterministic query-intent datasets."""

from __future__ import annotations

import json
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import read_json, read_jsonl_strict
from .errors import (
    E_CONFIG_INVALID,
    E_QUERY_INVALID,
    E_RULE_INVALID,
    issue,
    sort_issues,
)


SUPPORTED_NORMALIZATION_KINDS = {
    "case_fold",
    "width_fold",
    "whitespace_collapse",
    "token_split",
}


@dataclass
class Dataset:
    queries: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    config_issues: list[dict[str, Any]] = field(default_factory=list)
    rule_issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def config_ok(self) -> bool:
        return not self.config_issues and not self.rule_issues


def _is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _is_positive_decimal(value: Any) -> bool:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return False
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed > 0


def _entity_duplicate_key(entry: dict[str, Any]) -> tuple[str, str]:
    term = str(entry.get("term", ""))
    term = unicodedata.normalize("NFKC", term).casefold()
    term = " ".join(term.split())
    return term, str(entry.get("entity_type", ""))


def validate_rules(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the normalization and rewrite rule table without side effects."""

    issues: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return {}, [issue(E_RULE_INVALID, "rule table must be a JSON object")]

    rules = dict(raw)
    if not isinstance(rules.get("version"), str) or not rules["version"].strip():
        issues.append(issue(E_RULE_INVALID, "rule table version must be non-empty"))

    normalization_rules = rules.get("normalization_rules")
    if not isinstance(normalization_rules, list) or not normalization_rules:
        issues.append(issue(E_RULE_INVALID, "normalization_rules must be a non-empty list"))
        normalization_rules = []
    seen_rule_ids: set[str] = set()
    for index, rule in enumerate(normalization_rules):
        if not isinstance(rule, dict):
            issues.append(issue(E_RULE_INVALID, f"normalization rule {index} must be an object"))
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            issues.append(issue(E_RULE_INVALID, f"normalization rule {index} has invalid id"))
        elif rule_id in seen_rule_ids:
            issues.append(issue(E_RULE_INVALID, f"duplicate normalization rule id: {rule_id}"))
        else:
            seen_rule_ids.add(rule_id)
        if rule.get("kind") not in SUPPORTED_NORMALIZATION_KINDS:
            issues.append(
                issue(
                    E_RULE_INVALID,
                    f"normalization rule {rule_id or index} has unsupported kind",
                )
            )
        if not isinstance(rule.get("enabled"), bool):
            issues.append(
                issue(E_RULE_INVALID, f"normalization rule {rule_id or index} enabled must be boolean")
            )
        if not isinstance(rule.get("revision"), int) or rule.get("revision", 0) < 1:
            issues.append(
                issue(E_RULE_INVALID, f"normalization rule {rule_id or index} revision must be >= 1")
            )

    rewrite_rules = rules.get("rewrite_rules")
    if not isinstance(rewrite_rules, list) or not rewrite_rules:
        issues.append(issue(E_RULE_INVALID, "rewrite_rules must be a non-empty list"))
        rewrite_rules = []
    seen_rewrite_ids: set[str] = set()
    for index, rule in enumerate(rewrite_rules):
        if not isinstance(rule, dict):
            issues.append(issue(E_RULE_INVALID, f"rewrite rule {index} must be an object"))
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            issues.append(issue(E_RULE_INVALID, f"rewrite rule {index} has invalid id"))
        elif rule_id in seen_rewrite_ids:
            issues.append(issue(E_RULE_INVALID, f"duplicate rewrite rule id: {rule_id}"))
        else:
            seen_rewrite_ids.add(rule_id)
        if not isinstance(rule.get("intent"), str) or not rule["intent"]:
            issues.append(issue(E_RULE_INVALID, f"rewrite rule {rule_id or index} has invalid intent"))
        if not isinstance(rule.get("template"), str) or not rule["template"]:
            issues.append(
                issue(E_RULE_INVALID, f"rewrite rule {rule_id or index} has invalid template")
            )
        if not isinstance(rule.get("priority"), int):
            issues.append(
                issue(E_RULE_INVALID, f"rewrite rule {rule_id or index} priority must be integer")
            )
        if not isinstance(rule.get("revision"), int) or rule.get("revision", 0) < 1:
            issues.append(
                issue(E_RULE_INVALID, f"rewrite rule {rule_id or index} revision must be >= 1")
            )

    return rules, issues


def validate_config(raw: Any, rules: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate intent config and cross-check normalization-rule references."""

    if not isinstance(raw, dict):
        return {}, [issue(E_CONFIG_INVALID, "intent config must be a JSON object")]

    config = dict(raw)
    issues: list[dict[str, Any]] = []
    if not isinstance(config.get("version"), str) or not config["version"].strip():
        issues.append(issue(E_CONFIG_INVALID, "config version must be non-empty"))

    vocabulary = config.get("vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary:
        issues.append(issue(E_CONFIG_INVALID, "vocabulary must be a non-empty list"))
        vocabulary = []
    seen_vocabulary: set[str] = set()
    for index, term in enumerate(vocabulary):
        if not isinstance(term, str) or not term:
            issues.append(issue(E_CONFIG_INVALID, f"vocabulary item {index} must be a non-empty string"))
            continue
        if term in seen_vocabulary:
            issues.append(issue(E_CONFIG_INVALID, f"duplicate controlled vocabulary term: {term}"))
        seen_vocabulary.add(term)

    intents = config.get("intent_categories")
    if not isinstance(intents, list) or not intents:
        issues.append(issue(E_CONFIG_INVALID, "intent_categories must be a non-empty list"))
        intents = []
    intent_names: set[str] = set()
    for index, intent in enumerate(intents):
        if not isinstance(intent, dict):
            issues.append(issue(E_CONFIG_INVALID, f"intent category {index} must be an object"))
            continue
        name = intent.get("name")
        if not isinstance(name, str) or not name:
            issues.append(issue(E_CONFIG_INVALID, f"intent category {index} has invalid name"))
            continue
        if name in intent_names:
            issues.append(issue(E_CONFIG_INVALID, f"duplicate intent category: {name}"))
        intent_names.add(name)
        if not _is_positive_finite(intent.get("weight")):
            issues.append(issue(E_CONFIG_INVALID, f"intent {name} weight must be positive and finite"))
        features = intent.get("features")
        if not isinstance(features, list) or not features:
            issues.append(issue(E_CONFIG_INVALID, f"intent {name} features must be a non-empty list"))
            features = []
        seen_features: set[str] = set()
        for feature_index, feature in enumerate(features):
            if not isinstance(feature, dict):
                issues.append(
                    issue(E_CONFIG_INVALID, f"intent {name} feature {feature_index} must be an object")
                )
                continue
            term = feature.get("term")
            if not isinstance(term, str) or not term.strip():
                issues.append(issue(E_CONFIG_INVALID, f"intent {name} feature {feature_index} has invalid term"))
            elif term in seen_features:
                issues.append(issue(E_CONFIG_INVALID, f"intent {name} has duplicate feature: {term}"))
            else:
                seen_features.add(term)
            if not _is_positive_finite(feature.get("weight")):
                issues.append(
                    issue(E_CONFIG_INVALID, f"intent {name} feature {term or feature_index} weight must be positive and finite")
                )

    entity_types = config.get("entity_types")
    if not isinstance(entity_types, list) or not entity_types:
        issues.append(issue(E_CONFIG_INVALID, "entity_types must be a non-empty list"))
        entity_types = []
    seen_entity_types: set[str] = set()
    for index, entity_type in enumerate(entity_types):
        if not isinstance(entity_type, str) or not entity_type:
            issues.append(issue(E_CONFIG_INVALID, f"entity type {index} must be non-empty"))
            continue
        if entity_type in seen_entity_types:
            issues.append(issue(E_CONFIG_INVALID, f"duplicate entity type: {entity_type}"))
        seen_entity_types.add(entity_type)

    entities = config.get("entities")
    if not isinstance(entities, list) or not entities:
        issues.append(issue(E_CONFIG_INVALID, "entities must be a non-empty list"))
        entities = []
    seen_entry_ids: set[str] = set()
    seen_entity_terms: dict[tuple[str, str], str] = {}
    for index, entry in enumerate(entities):
        if not isinstance(entry, dict):
            issues.append(issue(E_CONFIG_INVALID, f"entity entry {index} must be an object"))
            continue
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            issues.append(issue(E_CONFIG_INVALID, f"entity entry {index} has invalid entry_id"))
            continue
        if entry_id in seen_entry_ids:
            issues.append(issue(E_CONFIG_INVALID, f"duplicate entity dictionary entry_id: {entry_id}"))
        seen_entry_ids.add(entry_id)
        term = entry.get("term")
        if not isinstance(term, str) or not term.strip():
            issues.append(issue(E_CONFIG_INVALID, f"entity entry {entry_id} has invalid term"))
            continue
        entity_type = entry.get("entity_type")
        if entity_type not in seen_entity_types:
            issues.append(issue(E_CONFIG_INVALID, f"entity entry {entry_id} references unknown type: {entity_type}"))
        duplicate_key = _entity_duplicate_key(entry)
        if duplicate_key in seen_entity_terms:
            issues.append(
                issue(
                    E_CONFIG_INVALID,
                    f"duplicate entity dictionary entry: {entry_id} matches {seen_entity_terms[duplicate_key]}",
                )
            )
        else:
            seen_entity_terms[duplicate_key] = entry_id
        if not _is_positive_finite(entry.get("priority")):
            issues.append(issue(E_CONFIG_INVALID, f"entity entry {entry_id} priority must be positive and finite"))

    normalization = config.get("normalization")
    if not isinstance(normalization, dict):
        issues.append(issue(E_CONFIG_INVALID, "normalization must be an object"))
        normalization = {}
    rule_ids = normalization.get("rule_ids")
    if not isinstance(rule_ids, list) or not rule_ids:
        issues.append(issue(E_CONFIG_INVALID, "normalization.rule_ids must be a non-empty list"))
        rule_ids = []
    available_rule_ids = {
        rule.get("id")
        for rule in rules.get("normalization_rules", [])
        if isinstance(rule, dict)
    }
    seen_references: set[str] = set()
    for rule_id in rule_ids:
        if not isinstance(rule_id, str):
            issues.append(issue(E_CONFIG_INVALID, "normalization rule reference must be a string"))
        elif rule_id not in available_rule_ids:
            issues.append(issue(E_CONFIG_INVALID, f"unknown normalization rule reference: {rule_id}"))
        elif rule_id in seen_references:
            issues.append(issue(E_CONFIG_INVALID, f"duplicate normalization rule reference: {rule_id}"))
        else:
            seen_references.add(rule_id)

    scoring = config.get("intent_scoring")
    if not isinstance(scoring, dict):
        issues.append(issue(E_CONFIG_INVALID, "intent_scoring must be an object"))
        scoring = {}
    decay = scoring.get("position_decay")
    if not isinstance(decay, (int, float)) or isinstance(decay, bool) or not math.isfinite(float(decay)) or float(decay) < 0:
        issues.append(issue(E_CONFIG_INVALID, "position_decay must be finite and non-negative"))
    precision = scoring.get("decimal_places")
    if not isinstance(precision, int) or not 0 <= precision <= 12:
        issues.append(issue(E_CONFIG_INVALID, "decimal_places must be an integer in [0, 12]"))
    quantum = scoring.get("quantum")
    if not _is_positive_decimal(quantum):
        issues.append(issue(E_CONFIG_INVALID, "scoring quantum must be a positive finite decimal"))

    max_workers = config.get("max_workers")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        issues.append(issue(E_CONFIG_INVALID, "max_workers must be a positive integer"))

    rewrite_intents = {
        rule.get("intent")
        for rule in rules.get("rewrite_rules", [])
        if isinstance(rule, dict)
    }
    for intent_name in sorted(intent_names):
        if intent_name not in rewrite_intents:
            issues.append(issue(E_CONFIG_INVALID, f"intent {intent_name} has no rewrite rule"))

    return config, issues


def validate_query_records(
    rows: list[dict[str, Any]],
    *,
    vocabulary: set[str],
    intent_names: set[str],
    line_errors: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate rows independently, then enforce session turn continuity globally."""

    failures: list[dict[str, Any]] = []
    for error in line_errors or []:
        failures.append(
            issue(
                E_QUERY_INVALID,
                error["message"],
                query_id=f"line:{error['line']:03d}",
            )
        )

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        label = query_id if isinstance(query_id, str) and query_id else f"line:{index:03d}"
        row_issues: list[str] = []
        if not isinstance(query_id, str) or not query_id:
            row_issues.append("query_id must be a non-empty string")
        elif query_id in seen_ids:
            row_issues.append("query_id must be unique")
        else:
            seen_ids.add(query_id)

        terms = row.get("terms")
        if not isinstance(terms, list) or not terms:
            row_issues.append("terms must be a non-empty list")
            terms = []
        for term_index, term in enumerate(terms):
            if not isinstance(term, str) or not term:
                row_issues.append(f"terms[{term_index}] must be a non-empty string")
            elif term not in vocabulary:
                row_issues.append(f"terms[{term_index}] is outside the controlled vocabulary: {term}")

        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            row_issues.append("session_id must be a non-empty string")
        turn = row.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
            row_issues.append("turn must be a positive integer")
        intent_label = row.get("intent_label")
        if not isinstance(intent_label, str) or not intent_label:
            row_issues.append("intent_label must be a non-empty string")
        elif intent_label not in intent_names:
            row_issues.append(f"unknown intent_label: {intent_label}")

        if row_issues:
            failures.append(
                issue(
                    E_QUERY_INVALID,
                    "; ".join(row_issues),
                    query_id=label,
                )
            )
            continue
        candidates.append(
            {
                "query_id": query_id,
                "terms": list(terms),
                "session_id": session_id,
                "turn": turn,
                "intent_label": intent_label,
            }
        )

    duplicate_turns: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for query in candidates:
        duplicate_turns.setdefault((query["session_id"], query["turn"]), []).append(query)
    invalid_ids: set[str] = set()
    for (session_id, turn), duplicates in duplicate_turns.items():
        if len(duplicates) > 1:
            for duplicate in duplicates:
                invalid_ids.add(duplicate["query_id"])
                failures.append(
                    issue(
                        E_QUERY_INVALID,
                        f"duplicate session turn: session {session_id} turn {turn}",
                        query_id=duplicate["query_id"],
                    )
                )

    by_session: dict[str, list[dict[str, Any]]] = {}
    for query in candidates:
        if query["query_id"] not in invalid_ids:
            by_session.setdefault(query["session_id"], []).append(query)
    for session_id, session_queries in by_session.items():
        ordered = sorted(session_queries, key=lambda item: item["turn"])
        expected = 1
        for query in ordered:
            if query["turn"] != expected:
                if query["turn"] > expected:
                    message = (
                        f"session {session_id} turn gap: expected {expected}, got {query['turn']}"
                    )
                else:
                    message = f"session {session_id} turns must be strictly increasing"
                invalid_ids.add(query["query_id"])
                failures.append(
                    issue(E_QUERY_INVALID, message, query_id=query["query_id"])
                )
            expected = max(expected, query["turn"] + 1)

    valid = [query for query in candidates if query["query_id"] not in invalid_ids]
    valid.sort(key=lambda item: item["query_id"])
    return valid, sort_issues(failures)


def load_rules(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        return validate_rules(read_json(path))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [issue(E_RULE_INVALID, f"unable to load rule table: {exc}")]


def load_config(path: str | Path, rules: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        return validate_config(read_json(path), rules)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [issue(E_CONFIG_INVALID, f"unable to load config: {exc}")]


def load_dataset(
    query_path: str | Path,
    config_path: str | Path,
    rules_path: str | Path,
) -> Dataset:
    rules, rule_issues = load_rules(rules_path)
    config, config_issues = load_config(config_path, rules)
    try:
        rows, line_errors = read_jsonl_strict(query_path)
    except OSError as exc:
        rows = []
        line_errors = [{"line": 1, "message": f"unable to read queries: {exc}"}]

    vocabulary = {
        term for term in config.get("vocabulary", []) if isinstance(term, str)
    } if isinstance(config.get("vocabulary"), list) else set()
    intent_names = {
        intent.get("name")
        for intent in config.get("intent_categories", [])
        if isinstance(intent, dict) and isinstance(intent.get("name"), str)
    } if isinstance(config.get("intent_categories"), list) else set()
    queries, query_failures = validate_query_records(
        rows,
        vocabulary=vocabulary,
        intent_names=intent_names,
        line_errors=line_errors,
    )
    return Dataset(
        queries=queries,
        config=config,
        rules=rules,
        failures=query_failures,
        config_issues=config_issues,
        rule_issues=rule_issues,
    )


def load_commands(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the command stream while preserving its declared execution order."""

    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [issue("E_COMMAND_INVALID", f"unable to load command stream: {exc}")]
    if not isinstance(raw, dict) or not isinstance(raw.get("commands"), list):
        return [], [issue("E_COMMAND_INVALID", "command file must contain a commands list")]
    commands = raw["commands"]
    failures: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            failures.append(issue("E_COMMAND_INVALID", f"command {index} must be an object"))
            continue
        command_id = command.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            failures.append(issue("E_COMMAND_INVALID", f"command {index} has invalid command_id"))
            continue
        if command_id in seen_ids:
            failures.append(
                issue("E_COMMAND_INVALID", f"duplicate command_id: {command_id}", command_id=command_id)
            )
            continue
        seen_ids.add(command_id)
        if command.get("type") not in {
            "add_query",
            "remove_query",
            "update_normalization_rule",
            "update_intent_weights",
            "replace_entity_dictionary_entry",
        }:
            failures.append(
                issue(
                    "E_COMMAND_INVALID",
                    f"unknown command type: {command.get('type')}",
                    command_id=command_id,
                )
            )
            continue
        valid.append(command)
    return valid, sort_issues(failures)
