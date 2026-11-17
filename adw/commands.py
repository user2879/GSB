"""Interactive command parsing and validated configuration patches."""

from __future__ import annotations

import copy
import math
import shlex
from dataclasses import dataclass
from typing import Any

from .config import (
    ALGORITHM_NAMES,
    algorithm_fingerprint,
    enabled_algorithms,
    normalize_config,
    preprocess_fingerprint,
    validate_config,
)
from .errors import ConfigValidationError, Failure, config_failure, sort_failures


HELP_TEXT = """commands:
  algo enable|disable NAME
  param set ALGORITHM KEY VALUE
  preprocess set mode|numeric_fill|categorical_fill VALUE
  threshold set VALUE
  budget set MILLISECONDS
  concurrency set N
  snapshot set N
  run | status | result | settle | report | recover | compare full | quit"""


@dataclass
class CommandResult:
    config: dict[str, Any]
    affected_algorithms: list[str]
    changed_fields: list[str]
    needs_run: bool
    message: str


def _fail(message: str, field: str = "command") -> ConfigValidationError:
    return ConfigValidationError([Failure(**config_failure(field, message).to_dict())])


def _coerce_token(token: str, reference: Any, candidates: list[Any] | None = None) -> Any:
    if isinstance(reference, bool):
        lowered = token.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise _fail(f"expected boolean, got {token!r}", "command.value")
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(token)
        except ValueError as exc:
            raise _fail(f"expected integer, got {token!r}", "command.value") from exc
    if isinstance(reference, float):
        try:
            value = float(token)
        except ValueError as exc:
            raise _fail(f"expected number, got {token!r}", "command.value") from exc
        if not math.isfinite(value):
            raise _fail("number must be finite", "command.value")
        return value
    # Strings may still be selected from a candidate list containing numeric-looking text.
    if candidates and all(isinstance(x, str) for x in candidates):
        return token
    return token


def _verify_candidate(value: Any, candidates: list[Any] | None, field: str) -> None:
    if candidates is None:
        return
    if not any(type(candidate) is type(value) and candidate == value for candidate in candidates):
        raise _fail(f"value {value!r} is not in candidates {candidates!r}", field)


def _affected(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    pre_changed = preprocess_fingerprint(old) != preprocess_fingerprint(new)
    selected = enabled_algorithms(new)
    if pre_changed:
        return list(selected)
    affected: list[str] = []
    for name in ALGORITHM_NAMES:
        old_item = old.get("algorithms", {}).get(name, {})
        new_item = new.get("algorithms", {}).get(name, {})
        old_enabled = bool(old_item.get("enabled", True))
        new_enabled = bool(new_item.get("enabled", True))
        parameters_changed = algorithm_fingerprint(old, name) != algorithm_fingerprint(new, name)
        if (old_enabled != new_enabled or parameters_changed) and (new_enabled or old_enabled):
            affected.append(name)
    return affected


def apply_command(old_config: dict[str, Any], description: dict[str, Any], line: str) -> CommandResult:
    try:
        tokens = shlex.split(line, comments=True)
    except ValueError as exc:
        raise _fail(f"cannot parse command: {exc}") from exc
    if not tokens:
        return CommandResult(copy.deepcopy(old_config), [], [], False, "empty command")
    command = tokens[0].lower()
    if command in ("quit", "exit", "help"):
        return CommandResult(copy.deepcopy(old_config), [], [], False, command)
    new_config = copy.deepcopy(old_config)
    changed: list[str] = []
    affected: list[str] = []
    needs_run = False

    if command in ("algo", "algorithm"):
        if len(tokens) != 3 or tokens[1].lower() not in ("enable", "disable", "set"):
            raise _fail("usage: algo enable|disable NAME")
        name = tokens[2]
        if name not in ALGORITHM_NAMES:
            raise _fail(f"unknown algorithm: {name}")
        if tokens[1].lower() == "set":
            raise _fail("usage: algo enable|disable NAME")
        enabled = tokens[1].lower() == "enable"
        new_config.setdefault("algorithms", {}).setdefault(name, {})["enabled"] = enabled
        changed.append(f"algorithms.{name}.enabled")
        needs_run = True
    elif command in ("param", "hyperparam"):
        if len(tokens) != 5 or tokens[1].lower() != "set":
            raise _fail("usage: param set ALGORITHM KEY VALUE")
        name, key, token = tokens[2], tokens[3], tokens[4]
        if name not in ALGORITHM_NAMES:
            raise _fail(f"unknown algorithm: {name}")
        item = new_config.get("algorithms", {}).get(name)
        if not isinstance(item, dict) or not isinstance(item.get("params"), dict):
            raise _fail(f"algorithm has no parameters: {name}")
        if key not in item["params"]:
            raise _fail(f"unknown hyperparameter: {name}.{key}")
        candidates = item.get("candidates", {}).get(key)
        if not isinstance(candidates, list):
            raise _fail(f"candidate list missing for {name}.{key}")
        value = _coerce_token(token, item["params"][key], candidates)
        _verify_candidate(value, candidates, f"algorithms.{name}.params.{key}")
        item["params"][key] = value
        changed.append(f"algorithms.{name}.params.{key}")
        needs_run = True
    elif command == "preprocess":
        if len(tokens) == 4 and tokens[1].lower() == "set":
            key, token = tokens[2], tokens[3]
        elif len(tokens) == 3 and tokens[1].lower() == "mode":
            key, token = "mode", tokens[2]
        else:
            raise _fail("usage: preprocess set mode|numeric_fill|categorical_fill VALUE")
        pre = new_config.get("preprocessing", {})
        if key not in ("mode", "numeric_fill", "categorical_fill"):
            raise _fail(f"unknown preprocessing field: {key}")
        candidates = pre.get("candidates", {}).get(key)
        if not isinstance(candidates, list):
            raise _fail(f"candidate list missing for preprocessing.{key}")
        value = _coerce_token(token, pre.get(key), candidates)
        _verify_candidate(value, candidates, f"preprocessing.{key}")
        pre[key] = value
        changed.append(f"preprocessing.{key}")
        needs_run = True
    elif command == "threshold":
        if len(tokens) == 3 and tokens[1].lower() == "set":
            token = tokens[2]
        elif len(tokens) == 2:
            token = tokens[1]
        else:
            raise _fail("usage: threshold set VALUE")
        try:
            value = float(token)
        except ValueError as exc:
            raise _fail(f"threshold must be numeric, got {token!r}") from exc
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise _fail("threshold must be in [0, 1]")
        new_config["threshold"] = value
        changed.append("threshold")
        needs_run = True
    elif command in ("budget", "concurrency", "snapshot"):
        if len(tokens) != 3 or tokens[1].lower() != "set":
            raise _fail(f"usage: {command} set VALUE")
        interaction = new_config.setdefault("interaction", {})
        if command == "budget":
            key = "latency_budget_ms"
            try:
                value = float(tokens[2])
            except ValueError as exc:
                raise _fail("budget must be numeric") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise _fail("latency budget must be positive")
        elif command == "concurrency":
            key = "concurrency"
            try:
                value = int(tokens[2])
            except ValueError as exc:
                raise _fail("concurrency must be an integer") from exc
            if value <= 0:
                raise _fail("concurrency must be positive")
        else:
            key = "snapshot_interval"
            try:
                value = int(tokens[2])
            except ValueError as exc:
                raise _fail("snapshot interval must be an integer") from exc
            if value <= 0:
                raise _fail("snapshot interval must be positive")
        interaction[key] = value
        changed.append(f"interaction.{key}")
        # Operational changes do not invalidate content-addressed artifacts.
        needs_run = False
    elif command == "run":
        needs_run = True
    else:
        raise _fail(f"unknown command: {command}. Use 'help'.")

    candidate = normalize_config(new_config)
    _, failures = validate_config(candidate, description)
    if failures:
        raise ConfigValidationError([Failure(**item) for item in sort_failures(failures)])
    affected = _affected(old_config, candidate)
    return CommandResult(
        config=candidate,
        affected_algorithms=affected,
        changed_fields=changed,
        needs_run=needs_run,
        message="configuration updated" if changed else "rerun requested",
    )
