from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import normalize_config
from .errors import E_COMMAND_INVALID, E_TYPE_MISMATCH, E_UNKNOWN_COMMAND, Issue
from .util import deep_copy, merge_dict, read_yaml_mapping, stable_hash


def load_commands(path: str | Path) -> tuple[list[dict[str, Any]], list[Issue]]:
    raw, error = read_yaml_mapping(path)
    if error is not None or raw is None:
        return [], [
            Issue(
                E_COMMAND_INVALID,
                error or "command stream must be a YAML mapping",
                context={"scope": "commands", "path": str(path)},
            )
        ]
    commands = raw.get("commands", [])
    if not isinstance(commands, list):
        return [], [
            Issue(
                E_TYPE_MISMATCH,
                "commands must be a list",
                context={"scope": "commands", "path": str(path)},
            )
        ]
    accepted: list[dict[str, Any]] = []
    issues: list[Issue] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "command must be a mapping",
                    context={"scope": "commands", "command_index": index},
                )
            )
            continue
        normalized = deep_copy(command)
        normalized["_stream_index"] = index
        accepted.append(normalized)
    return accepted, issues


def command_type(command: dict[str, Any]) -> str:
    value = command.get("type", command.get("command", ""))
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_")


def apply_command(
    state: dict[str, Any], command: dict[str, Any]
) -> tuple[dict[str, Any], bool, list[Issue], bool]:
    """Apply one command and return ``(new_state, accepted, issues, changed)``."""

    kind = command_type(command)
    issues: list[Issue] = []
    if kind in {"view", "snapshot", "noop"}:
        return state, True, issues, False
    new_state = deep_copy(state)
    changed = True

    if kind == "add_candidate":
        candidate = _candidate_fragment(command)
        if candidate is None:
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "add_candidate requires id and method",
                    context={"scope": "commands", "command": kind},
                )
            )
            return state, False, issues, False
        if any(item.get("id") == candidate["id"] for item in new_state.get("candidates", [])):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    f"candidate id already exists: {candidate['id']!r}",
                    context={"scope": "commands", "candidate": candidate["id"]},
                )
            )
            return state, False, issues, False
        new_state.setdefault("candidates", []).append(candidate)
        return new_state, True, issues, True

    if kind == "update_candidate":
        identifier = command.get("id", command.get("candidate"))
        candidates = new_state.get("candidates", [])
        index = next((i for i, item in enumerate(candidates) if item.get("id") == identifier), None)
        if index is None:
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    f"unknown candidate id: {identifier!r}",
                    context={"scope": "commands", "candidate": identifier},
                )
            )
            return state, False, issues, False
        candidate = dict(candidates[index])
        if "params" in command:
            if not isinstance(command["params"], dict):
                issues.append(
                    Issue(
                        E_TYPE_MISMATCH,
                        "update_candidate.params must be a mapping",
                        context={"scope": "commands", "candidate": identifier},
                    )
                )
                return state, False, issues, False
            candidate["params"] = merge_dict(candidate.get("params", {}), command["params"])
        if "params_patch" in command and isinstance(command["params_patch"], dict):
            candidate["params"] = merge_dict(candidate.get("params", {}), command["params_patch"])
        if "feature_columns" in command:
            candidate["feature_columns"] = deep_copy(command["feature_columns"])
        if "train_fraction" in command:
            candidate["train_fraction"] = command["train_fraction"]
        if "method" in command:
            candidate["method"] = command["method"]
        candidates[index] = candidate
        new_state["candidates"] = candidates
        return new_state, True, issues, True

    if kind == "remove_candidate":
        identifier = command.get("id", command.get("candidate"))
        candidates = new_state.get("candidates", [])
        retained = [item for item in candidates if item.get("id") != identifier]
        if len(retained) == len(candidates):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    f"unknown candidate id: {identifier!r}",
                    context={"scope": "commands", "candidate": identifier},
                )
            )
            return state, False, issues, False
        new_state["candidates"] = retained
        return new_state, True, issues, True

    if kind in {"set_preprocessing", "update_preprocessing"}:
        patch = command.get("preprocessing")
        if patch is None:
            patch = {key: value for key, value in command.items() if key not in {"type", "command"}}
        if not isinstance(patch, dict):
            issues.append(Issue(E_TYPE_MISMATCH, "preprocessing patch must be a mapping"))
            return state, False, issues, False
        _apply_config_patch(new_state, "preprocessing", patch, issues)
        return new_state, True, issues, True

    if kind in {"set_evaluation", "update_evaluation"}:
        patch = command.get("evaluation", command.get("params"))
        if not isinstance(patch, dict):
            issues.append(Issue(E_TYPE_MISMATCH, "evaluation patch must be a mapping"))
            return state, False, issues, False
        _apply_config_patch(new_state, "evaluation", patch, issues)
        return new_state, True, issues, True

    if kind in {"set_scheduler", "update_scheduler"}:
        patch = command.get("scheduler", command.get("params"))
        if not isinstance(patch, dict):
            issues.append(Issue(E_TYPE_MISMATCH, "scheduler patch must be a mapping"))
            return state, False, issues, False
        _apply_config_patch(new_state, "scheduler", patch, issues)
        return new_state, True, issues, True

    if kind == "replace_column":
        column = command.get("column")
        override = command.get("override")
        if not isinstance(column, str) or not isinstance(override, dict):
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    "replace_column requires column and override mapping",
                    context={"scope": "commands"},
                )
            )
            return state, False, issues, False
        new_state.setdefault("data_overrides", {})[column] = deep_copy(override)
        return new_state, True, issues, True

    if kind == "restore_column":
        column = command.get("column")
        overrides = new_state.setdefault("data_overrides", {})
        if column not in overrides:
            issues.append(
                Issue(
                    E_COMMAND_INVALID,
                    f"column has no override: {column!r}",
                    context={"scope": "commands", "column": column},
                )
            )
            return state, False, issues, False
        overrides.pop(column)
        return new_state, True, issues, True

    if kind == "set_data_size":
        level = command.get("size_level")
        rows = command.get("rows")
        if level is not None:
            new_state["data_description"]["size_level"] = str(level)
            levels = new_state["data_description"].get("size_levels", {})
            if level not in levels:
                issues.append(
                    Issue(
                        E_COMMAND_INVALID,
                        f"unknown size level: {level!r}",
                        context={"scope": "commands", "size_level": level},
                    )
                )
                return state, False, issues, False
            new_state["data_description"]["rows"] = int(levels[level])
        elif isinstance(rows, int) and not isinstance(rows, bool) and rows > 0:
            new_state["data_description"]["rows"] = rows
        else:
            issues.append(
                Issue(E_COMMAND_INVALID, "set_data_size requires size_level or positive rows")
            )
            return state, False, issues, False
        return new_state, True, issues, True

    issues.append(
        Issue(
            E_UNKNOWN_COMMAND,
            f"unknown command type: {command.get('type', command.get('command'))!r}",
            context={"scope": "commands"},
        )
    )
    return state, False, issues, False


def _candidate_fragment(command: dict[str, Any]) -> dict[str, Any] | None:
    identifier = command.get("id", command.get("candidate"))
    method = command.get("method")
    if not isinstance(identifier, str) or not identifier or not isinstance(method, str):
        return None
    fragment: dict[str, Any] = {
        "id": identifier,
        "method": method,
        "params": deep_copy(command.get("params", {})),
    }
    if "feature_columns" in command:
        fragment["feature_columns"] = deep_copy(command["feature_columns"])
    if "train_fraction" in command:
        fragment["train_fraction"] = command["train_fraction"]
    return fragment


def _apply_config_patch(
    state: dict[str, Any], section: str, patch: dict[str, Any], issues: list[Issue]
) -> None:
    raw_config = deep_copy(state.get("raw_config", state.get("config", {})))
    raw_config[section] = merge_dict(raw_config.get(section, {}), patch)
    state["raw_config"] = raw_config
    normalized, validation_issues = normalize_config(raw_config)
    state["config"] = normalized
    state["state_issues"] = [issue.to_dict() for issue in validation_issues]
    issues.extend(validation_issues)


def compute_state_fingerprint(state: dict[str, Any]) -> str:
    config = state.get("config", {})
    schedulable_config = {
        "preprocessing": config.get("preprocessing", {}),
        "evaluation": config.get("evaluation", {}),
        "methods": config.get("methods", {}),
        "scheduler": {
            "global_iteration_budget": config.get("scheduler", {}).get("global_iteration_budget"),
            "per_candidate_iteration_quota": config.get("scheduler", {}).get(
                "per_candidate_iteration_quota"
            ),
        },
    }
    return stable_hash(
        {
            "config": schedulable_config,
            "data_description": state.get("data_description", {}),
            "candidates": state.get("candidates", []),
            "data_overrides": state.get("data_overrides", {}),
            "injections": state.get("injections", []),
            "state_issues": state.get("state_issues", []),
        }
    )
