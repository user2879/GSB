from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import is_int, validate_config
from .errors import ErrorCode, make_issue
from .samplegen import write_replacement_image


KNOWN_OPS = {
    "set_grid",
    "set_percentiles",
    "set_thresholds",
    "set_quantization",
    "set_exchange_version",
    "set_concurrency",
    "set_budget",
    "replace_image",
    "delete_image",
}


def load_command_stream(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [make_issue(ErrorCode.CONFIG_INVALID, "command stream could not be parsed", path=str(path), detail=str(exc))]
    if isinstance(raw, dict):
        commands = raw.get("commands")
    else:
        commands = raw
    if not isinstance(commands, list):
        return [], [make_issue(ErrorCode.CONFIG_INVALID, "command stream must contain a commands list")]
    issues: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            issues.append(make_issue(ErrorCode.CONFIG_INVALID, "command must be an object", command_index=index))
        elif command.get("op") not in KNOWN_OPS:
            issues.append(
                make_issue(
                    ErrorCode.CONFIG_INVALID,
                    "unknown command operation",
                    command_index=index,
                    op=command.get("op"),
                    allowed=sorted(KNOWN_OPS),
                )
            )
    return commands, issues


def apply_command(
    command: dict[str, Any],
    config: dict[str, Any],
    exchange_version: str,
    input_dir: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any]]:
    op = command.get("op")
    next_config = dict(config)
    next_version = exchange_version
    issues: list[dict[str, Any]] = []
    mutation: dict[str, Any] = {"op": op}
    input_path = Path(input_dir)

    if op == "set_grid":
        if "rows" in command and "grid_rows" not in command:
            command = {**command, "grid_rows": command["rows"]}
        if "cols" in command and "grid_cols" not in command:
            command = {**command, "grid_cols": command["cols"]}
        next_config.update({k: command[k] for k in ("grid_rows", "grid_cols") if k in command})
    elif op == "set_percentiles":
        mapping = {
            "low": "percentile_low_permille",
            "high": "percentile_high_permille",
            "percentile_low_permille": "percentile_low_permille",
            "percentile_high_permille": "percentile_high_permille",
        }
        for key, target in mapping.items():
            if key in command:
                next_config[target] = command[key]
    elif op == "set_thresholds":
        thresholds = command.get("grade_thresholds_permille", command.get("thresholds"))
        next_config["grade_thresholds_permille"] = thresholds
    elif op == "set_quantization":
        for key in ("fixed_decimals", "quant_levels", "quant_scheme"):
            if key in command:
                next_config[key] = command[key]
        if "levels" in command:
            next_config["quant_levels"] = command["levels"]
        if "decimals" in command:
            next_config["fixed_decimals"] = command["decimals"]
    elif op == "set_exchange_version":
        version = command.get("version")
        if version not in {"v1", "v2"}:
            issues.append(make_issue(ErrorCode.CONFIG_INVALID, "exchange version must be v1 or v2", value=version))
            return config, exchange_version, issues, mutation
        next_version = version
        mutation["version"] = version
        return next_config, next_version, issues, mutation
    elif op == "set_concurrency":
        next_config["max_concurrency"] = command.get("max_concurrency", command.get("value"))
    elif op == "set_budget":
        next_config["sample_budget"] = command.get("sample_budget", command.get("budget"))
    elif op == "replace_image":
        image = command.get("image")
        if not isinstance(image, str):
            issues.append(make_issue(ErrorCode.CONFIG_INVALID, "replace_image requires image", command=command))
            return config, exchange_version, issues, mutation
        destination = input_path / image
        source = command.get("source")
        if source:
            shutil.copyfile(source, destination)
            mutation["source"] = str(source)
        else:
            variant = int(command.get("variant", 9))
            write_replacement_image(input_path, image, variant=variant)
            mutation["variant"] = variant
        mutation["image"] = image
        return next_config, next_version, issues, mutation
    elif op == "delete_image":
        image = command.get("image")
        if not isinstance(image, str):
            issues.append(make_issue(ErrorCode.CONFIG_INVALID, "delete_image requires image", command=command))
            return config, exchange_version, issues, mutation
        destination = input_path / image
        if destination.exists():
            destination.unlink()
        mutation["image"] = image
        return next_config, next_version, issues, mutation
    else:
        issues.append(make_issue(ErrorCode.CONFIG_INVALID, "cannot apply unknown command", op=op))
        return config, exchange_version, issues, mutation

    validated, validation_issues = validate_config(next_config)
    if validation_issues:
        issues.extend(validation_issues)
        return config, exchange_version, issues, mutation
    return validated, next_version, issues, mutation
