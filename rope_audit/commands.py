from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import _validate_scheme, is_positive_int, parse_config
from .errors import ConfigInvalid
from .samples import validate_sample


def parse_command_line(line: str, line_number: int) -> dict[str, Any]:
    try:
        command = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ConfigInvalid(f"invalid command JSON: {exc.msg}", line=line_number, field="$") from exc
    if not isinstance(command, dict):
        raise ConfigInvalid("command must be a JSON object", line=line_number, field="$")
    if not isinstance(command.get("op"), str) or not command["op"]:
        raise ConfigInvalid("command.op must be a non-empty string", line=line_number, field="op")
    command["_line"] = line_number
    return command


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                commands.append(parse_command_line(line, line_number))
    return commands


def _reparse(config: dict[str, Any]) -> dict[str, Any]:
    return parse_config(
        {
            "schema_version": config.get("schema_version", 1),
            "dimension": config["dimension"],
            "seed": config.get("seed", 393),
            "train_length": config["train_length"],
            "target_length": config["target_length"],
            "fixed_point": config["fixed_point"],
            "tolerances": config["tolerances"],
            "budget": config["budget"],
            "concurrency": config["concurrency"],
            "schemes": config["schemes"],
            "active_scheme": config["active_scheme"],
            "snapshot_dir": config.get("snapshot_dir", "artifacts/snapshots"),
        }
    )


def apply_command(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    command: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    line = int(command.get("_line", 0)) or None
    op = command["op"]
    next_config = deepcopy(config)
    next_samples = deepcopy(samples)
    by_id = {sample["id"]: index for index, sample in enumerate(next_samples)}
    edit = {"op": op, "line": line, "affected_sample_ids": [], "affected_scheme_names": []}

    def scheme_index(name: str) -> int:
        for index, scheme in enumerate(next_config["schemes"]):
            if scheme["name"] == name:
                return index
        raise ConfigInvalid(f"unknown scheme {name!r}", line=line, field="name")

    if op in {"add_scheme", "upsert_scheme"}:
        raw_scheme = command.get("scheme") if isinstance(command.get("scheme"), dict) else command
        scheme = _validate_scheme(raw_scheme, next_config["dimension"], None, 0)
        existing = next((i for i, item in enumerate(next_config["schemes"]) if item["name"] == scheme["name"]), None)
        if op == "add_scheme" and existing is not None:
            raise ConfigInvalid(f"scheme {scheme['name']!r} already exists", line=line, field="scheme.name")
        if existing is None:
            next_config["schemes"].append(scheme)
        else:
            next_config["schemes"][existing] = scheme
        edit["affected_scheme_names"].append(scheme["name"])

    elif op in {"remove_scheme", "delete_scheme"}:
        name = command.get("name")
        if not isinstance(name, str):
            raise ConfigInvalid("command.name is required", line=line, field="name")
        index = scheme_index(name)
        if len(next_config["schemes"]) == 1:
            raise ConfigInvalid("at least one scheme is required", line=line, field="schemes")
        next_config["schemes"].pop(index)
        if next_config["active_scheme"] == name:
            next_config["active_scheme"] = next_config["schemes"][0]["name"]
        edit["affected_scheme_names"].append(name)

    elif op == "switch_scheme":
        name = command.get("name")
        if not isinstance(name, str):
            raise ConfigInvalid("command.name is required", line=line, field="name")
        scheme_index(name)
        next_config["active_scheme"] = name
        edit["affected_scheme_names"].append(name)

    elif op in {"update_scheme", "patch_scheme"}:
        name = command.get("name")
        if not isinstance(name, str):
            raise ConfigInvalid("command.name is required", line=line, field="name")
        index = scheme_index(name)
        changes = command.get("params", command.get("changes", {}))
        if not isinstance(changes, dict):
            raise ConfigInvalid("params must be an object", line=line, field="params")
        merged = deepcopy(next_config["schemes"][index])
        merged.update(changes)
        merged["name"] = name
        next_config["schemes"][index] = _validate_scheme(merged, next_config["dimension"], None, index)
        edit["affected_scheme_names"].append(name)

    elif op in {"set_fixed_point", "update_quant"}:
        changes = command.get("fixed_point", command.get("params", command))
        if not isinstance(changes, dict):
            raise ConfigInvalid("fixed_point must be an object", line=line, field="fixed_point")
        fp = next_config["fixed_point"]
        raw = {
            "frac_bits": changes.get("frac_bits", fp["frac_bits"]),
            "quant_level": changes.get("quant_level", fp["quant_level"]),
            "level_shifts": changes.get("level_shifts", fp["level_shifts"]),
        }
        parsed = parse_config(
            {
                "dimension": next_config["dimension"],
                "train_length": next_config["train_length"],
                "target_length": next_config["target_length"],
                "fixed_point": raw,
                "tolerances": next_config["tolerances"],
                "schemes": next_config["schemes"],
                "active_scheme": next_config["active_scheme"],
            }
        )
        next_config["fixed_point"] = parsed["fixed_point"]

    elif op == "set_lengths":
        train_length = command.get("train_length", next_config["train_length"])
        target_length = command.get("target_length", next_config["target_length"])
        if not is_positive_int(train_length):
            raise ConfigInvalid("train_length must be a positive integer", line=line, field="train_length")
        if not is_positive_int(target_length):
            raise ConfigInvalid("target_length must be a positive integer", line=line, field="target_length")
        if int(target_length) < int(train_length):
            raise ConfigInvalid("target_length must be >= train_length", line=line, field="target_length")
        next_config["train_length"] = int(train_length)
        next_config["target_length"] = int(target_length)
        if command.get("sync_samples", True):
            for sample in next_samples:
                sample["train_length"] = int(train_length)
                edit["affected_sample_ids"].append(sample["id"])

    elif op in {"replace_sample", "upsert_sample"}:
        raw_sample = command.get("sample")
        if not isinstance(raw_sample, dict):
            raise ConfigInvalid("command.sample must be an object", line=line, field="sample")
        sample = validate_sample(raw_sample, next_config["dimension"], line)
        if op == "replace_sample" and sample["id"] not in by_id:
            raise ConfigInvalid(
                f"sample {sample['id']!r} does not exist",
                line=line,
                field="sample.id",
                sample_id=sample["id"],
            )
        if sample["id"] in by_id:
            next_samples[by_id[sample["id"]]] = sample
        else:
            next_samples.append(sample)
        edit["affected_sample_ids"].append(sample["id"])

    elif op in {"delete_sample", "remove_sample"}:
        sample_id = command.get("id", command.get("sample_id"))
        if not isinstance(sample_id, str) or sample_id not in by_id:
            raise ConfigInvalid(f"unknown sample {sample_id!r}", line=line, field="id", sample_id=sample_id)
        next_samples.pop(by_id[sample_id])
        edit["affected_sample_ids"].append(sample_id)

    elif op == "set_budget":
        budget = command.get("budget", command)
        if not isinstance(budget, dict):
            raise ConfigInvalid("budget must be an object", line=line, field="budget")
        batch = budget.get("batch_point_budget", next_config["budget"]["batch_point_budget"])
        quota = budget.get("sample_quota_points", next_config["budget"]["sample_quota_points"])
        if not is_positive_int(batch) or not is_positive_int(quota):
            raise ConfigInvalid("budget values must be positive integers", line=line, field="budget")
        next_config["budget"] = {
            "batch_point_budget": int(batch),
            "sample_quota_points": int(quota),
        }

    else:
        raise ConfigInvalid(f"unknown command op {op!r}", line=line, field="op")

    next_config = _reparse(next_config)
    next_samples = sorted(next_samples, key=lambda item: item["id"])
    return next_config, next_samples, edit


def apply_commands(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    edits: list[dict[str, Any]] = []
    for command in commands:
        config, samples, edit = apply_command(config, samples, command)
        edits.append(edit)
    return config, samples, edits
