"""Mutable workbench state and command application semantics."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_QUERY_INVALID,
    E_RULE_INVALID,
    issue,
    sort_issues,
)
from .loader import validate_config, validate_query_records, validate_rules


@dataclass
class WorkbenchState:
    queries: list[dict[str, Any]]
    config: dict[str, Any]
    rules: dict[str, Any]
    failures: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "WorkbenchState":
        return WorkbenchState(
            queries=copy.deepcopy(self.queries),
            config=copy.deepcopy(self.config),
            rules=copy.deepcopy(self.rules),
            failures=copy.deepcopy(self.failures),
        )

    def sorted_queries(self) -> list[dict[str, Any]]:
        return sorted(self.queries, key=lambda item: item["query_id"])


def _command_failure(
    command: dict[str, Any],
    message: str,
    *,
    code: str = E_COMMAND_INVALID,
    detail: Any | None = None,
) -> dict[str, Any]:
    return issue(
        code,
        message,
        command_id=str(command.get("command_id", "unknown")),
        detail=detail,
    )


def _append_failure(state: WorkbenchState, failure: dict[str, Any]) -> None:
    state.failures = sort_issues(state.failures + [failure])


def _validated_target_issues(
    config: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    _, rule_issues = validate_rules(rules)
    _, config_issues = validate_config(config, rules)
    return rule_issues + config_issues


def apply_command(
    state: WorkbenchState, command: dict[str, Any]
) -> tuple[WorkbenchState, bool, dict[str, Any]]:
    """Apply one command atomically; failed commands leave business state intact."""

    command_type = command.get("type")
    before_query_ids = [query["query_id"] for query in state.sorted_queries()]
    changed = False
    message = ""

    if command_type == "add_query":
        query = command.get("query")
        if not isinstance(query, dict):
            _append_failure(state, _command_failure(command, "add_query requires a query object"))
            return state, False, {"reason": "invalid_payload"}
        vocabulary = set(state.config.get("vocabulary", []))
        intent_names = {
            intent["name"] for intent in state.config.get("intent_categories", [])
        }
        valid, failures = validate_query_records(
            copy.deepcopy(state.queries) + [copy.deepcopy(query)],
            vocabulary=vocabulary,
            intent_names=intent_names,
        )
        if failures:
            for failure in failures:
                failure = dict(failure)
                failure["command_id"] = command["command_id"]
                _append_failure(state, failure)
            return state, False, {"reason": "query_invalid", "issues": failures}
        state.queries = valid
        changed = True
        message = f"added {query.get('query_id')}"

    elif command_type == "remove_query":
        query_id = command.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            _append_failure(state, _command_failure(command, "remove_query requires query_id"))
            return state, False, {"reason": "invalid_payload"}
        remaining = [query for query in state.queries if query["query_id"] != query_id]
        if len(remaining) == len(state.queries):
            _append_failure(
                state,
                _command_failure(command, f"query does not exist: {query_id}", code=E_QUERY_INVALID),
            )
            return state, False, {"reason": "query_missing"}
        state.queries = remaining
        changed = True
        message = f"removed {query_id}"

    elif command_type == "update_normalization_rule":
        rule_id = command.get("rule_id")
        patch = command.get("patch")
        if not isinstance(rule_id, str) or not isinstance(patch, dict):
            _append_failure(
                state,
                _command_failure(command, "update_normalization_rule requires rule_id and patch"),
            )
            return state, False, {"reason": "invalid_payload"}
        if set(patch) - {"enabled"}:
            _append_failure(
                state,
                _command_failure(command, "only the enabled field may be patched"),
            )
            return state, False, {"reason": "invalid_patch"}
        trial_rules = copy.deepcopy(state.rules)
        target = None
        for rule in trial_rules.get("normalization_rules", []):
            if rule.get("id") == rule_id:
                target = rule
                break
        if target is None:
            _append_failure(
                state,
                _command_failure(command, f"unknown normalization rule: {rule_id}"),
            )
            return state, False, {"reason": "rule_missing"}
        if "enabled" in patch:
            target["enabled"] = patch["enabled"]
        target["revision"] = int(target.get("revision", 1)) + 1
        target_issues = _validated_target_issues(state.config, trial_rules)
        if target_issues:
            _append_failure(
                state,
                _command_failure(
                    command,
                    f"normalization rule update failed validation: {rule_id}",
                    code=E_RULE_INVALID,
                    detail=target_issues,
                ),
            )
            return state, False, {"reason": "rule_invalid", "issues": target_issues}
        state.rules = trial_rules
        changed = True
        message = f"updated normalization rule {rule_id}"

    elif command_type == "update_intent_weights":
        weights = command.get("weights")
        if not isinstance(weights, dict) or not weights:
            _append_failure(
                state,
                _command_failure(command, "update_intent_weights requires non-empty weights"),
            )
            return state, False, {"reason": "invalid_payload"}
        trial_config = copy.deepcopy(state.config)
        known_intents = {
            intent["name"] for intent in trial_config.get("intent_categories", [])
        }
        invalid_weights: list[str] = []
        for intent_name, weight in weights.items():
            if intent_name not in known_intents:
                invalid_weights.append(f"unknown intent: {intent_name}")
                continue
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or float(weight) <= 0
            ):
                invalid_weights.append(f"invalid weight for {intent_name}: {weight!r}")
        if invalid_weights:
            _append_failure(
                state,
                _command_failure(
                    command,
                    "intent weight update failed validation",
                    code=E_CONFIG_INVALID,
                    detail=invalid_weights,
                ),
            )
            return state, False, {"reason": "weight_invalid", "issues": invalid_weights}
        for intent in trial_config.get("intent_categories", []):
            if intent["name"] in weights:
                intent["weight"] = float(weights[intent["name"]])
        target_issues = _validated_target_issues(trial_config, state.rules)
        if target_issues:
            _append_failure(
                state,
                _command_failure(
                    command,
                    "intent weight update failed configuration validation",
                    code=E_CONFIG_INVALID,
                    detail=target_issues,
                ),
            )
            return state, False, {"reason": "config_invalid", "issues": target_issues}
        state.config = trial_config
        changed = True
        message = f"updated weights: {', '.join(sorted(weights))}"

    elif command_type == "replace_entity_dictionary_entry":
        entry_id = command.get("entry_id")
        replacement = command.get("entry")
        if not isinstance(entry_id, str) or not isinstance(replacement, dict):
            _append_failure(
                state,
                _command_failure(
                    command,
                    "replace_entity_dictionary_entry requires entry_id and entry",
                ),
            )
            return state, False, {"reason": "invalid_payload"}
        replacement = copy.deepcopy(replacement)
        if replacement.get("entry_id", entry_id) != entry_id:
            _append_failure(
                state,
                _command_failure(command, "replacement entry_id must match target entry_id"),
            )
            return state, False, {"reason": "entry_id_mismatch"}
        replacement["entry_id"] = entry_id
        trial_config = copy.deepcopy(state.config)
        found = False
        for index, entry in enumerate(trial_config.get("entities", [])):
            if entry.get("entry_id") == entry_id:
                trial_config["entities"][index] = replacement
                found = True
                break
        if not found:
            _append_failure(
                state,
                _command_failure(command, f"unknown entity dictionary entry: {entry_id}"),
            )
            return state, False, {"reason": "entry_missing"}
        target_issues = _validated_target_issues(trial_config, state.rules)
        if target_issues:
            _append_failure(
                state,
                _command_failure(
                    command,
                    f"entity replacement failed validation: {entry_id}",
                    code=E_CONFIG_INVALID,
                    detail=target_issues,
                ),
            )
            return state, False, {"reason": "config_invalid", "issues": target_issues}
        state.config = trial_config
        changed = True
        message = f"replaced entity entry {entry_id}"

    else:
        _append_failure(
            state,
            _command_failure(command, f"unknown command type: {command_type}"),
        )
        return state, False, {"reason": "unknown_type"}

    after_query_ids = [query["query_id"] for query in state.sorted_queries()]
    return state, changed, {
        "message": message,
        "before_query_ids": before_query_ids,
        "after_query_ids": after_query_ids,
    }
