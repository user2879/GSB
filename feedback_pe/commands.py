from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import codes
from .config import (
    FRAC_BITS_CHOICES, QUANT_TIER_CHOICES, is_finite_number, is_int,
    validate_config,
)
from .samples import validate_sample_record


class CommandError(ValueError):
    def __init__(self, code: str, field: str, message: str):
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message


def load_command_stream(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    commands: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append({
                "line_number": line_number,
                "code": codes.E_COMMAND_PARSE,
                "field": f"line:{line_number}",
                "message": exc.msg,
            })
            continue
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            failures.append({
                "line_number": line_number,
                "code": codes.E_COMMAND_OP,
                "field": "op",
                "message": "command must be an object with string op",
            })
            continue
        command["_line_number"] = line_number
        commands.append(command)
    return commands, failures


def _find_scheme(config: dict[str, Any], name: str | None) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise CommandError(codes.E_COMMAND_FIELD, "scheme", "scheme name is required")
    for scheme in config["schemes"]:
        if scheme["name"] == name:
            return scheme
    raise CommandError(codes.E_COMMAND_FIELD, "scheme", f"unknown scheme {name!r}")


def apply_command(config: dict[str, Any], samples_by_id: dict[str, dict[str, Any]],
                  command: dict[str, Any]) -> dict[str, Any]:
    op = command.get("op")
    line = command.get("_line_number")
    result = {"op": op, "line_number": line, "changed": [], "run_batch": op == "run_batch"}

    if op == "run_batch":
        result["batch_budget"] = command.get("batch_point_budget")
        result["sample_quota"] = command.get("sample_point_quota")
        return result

    if op == "switch_scheme":
        scheme = _find_scheme(config, command.get("name") or command.get("scheme"))
        scheme["enabled"] = True
        config["active_scheme"] = scheme["name"]
        result["changed"] = ["active_scheme", f"schemes.{scheme['name']}.enabled"]

    elif op == "add_scheme":
        scheme = command.get("scheme")
        if not isinstance(scheme, dict):
            raise CommandError(codes.E_COMMAND_FIELD, "scheme", "scheme object required")
        trial = json.loads(json.dumps(config))
        trial["schemes"].append(scheme)
        errors = validate_config(trial)
        scheme_errors = [item for item in errors if item["field"].startswith("schemes")]
        if scheme_errors:
            raise CommandError(scheme_errors[0]["code"], scheme_errors[0]["field"],
                               scheme_errors[0]["message"])
        if any(item["name"] == scheme["name"] for item in config["schemes"]):
            raise CommandError(codes.E_SCHEME, "scheme.name", "scheme already exists")
        config["schemes"].append({
            "name": scheme["name"],
            "kind": scheme["kind"],
            "base_freq": float(scheme["base_freq"]),
            "scale": float(scheme["scale"]),
            "block_size": int(scheme["block_size"]),
            "enabled": bool(scheme.get("enabled", True)),
        })
        result["changed"] = [f"schemes.{scheme['name']}"]

    elif op == "remove_scheme":
        name = command.get("name") or command.get("scheme")
        scheme = _find_scheme(config, name)
        if len(config["schemes"]) <= 1:
            raise CommandError(codes.E_SCHEME, "schemes", "at least one scheme must remain")
        config["schemes"] = [item for item in config["schemes"] if item["name"] != scheme["name"]]
        if config.get("active_scheme") == scheme["name"]:
            config["active_scheme"] = config["schemes"][0]["name"]
        result["changed"] = [f"removed_scheme.{scheme['name']}"]

    elif op in {"set_base_freq", "set_scale", "set_block_size"}:
        field = {
            "set_base_freq": "base_freq",
            "set_scale": "scale",
            "set_block_size": "block_size",
        }[op]
        value = command.get(field, command.get("value"))
        if field == "block_size":
            if not is_int(value) or value <= 0:
                raise CommandError(codes.E_BLOCK_SIZE, field, "must be positive integer")
        elif not is_finite_number(value) or float(value) <= 0:
            code = codes.E_BASE_FREQ if field == "base_freq" else codes.E_SCALE
            raise CommandError(code, field, "must be finite and > 0")
        name = command.get("scheme") or command.get("name")
        if name:
            scheme = _find_scheme(config, name)
            scheme[field] = int(value) if field == "block_size" else float(value)
            result["changed"] = [f"schemes.{scheme['name']}.{field}"]
        else:
            config[field] = int(value) if field == "block_size" else float(value)
            for scheme in config["schemes"]:
                scheme[field] = config[field]
            result["changed"] = [field, "all_schemes." + field]
        errors = validate_config(config)
        if errors:
            raise CommandError(errors[0]["code"], errors[0]["field"], errors[0]["message"])

    elif op == "set_frac_bits":
        value = command.get("frac_bits", command.get("value"))
        if value not in FRAC_BITS_CHOICES:
            raise CommandError(codes.E_ENUM_FRAC_BITS, "frac_bits",
                               f"must be one of {list(FRAC_BITS_CHOICES)}")
        config["frac_bits"] = int(value)
        result["changed"] = ["frac_bits"]

    elif op == "set_quant_tier":
        value = command.get("quant_tier", command.get("value"))
        if value not in QUANT_TIER_CHOICES:
            raise CommandError(codes.E_ENUM_QUANT_TIER, "quant_tier",
                               f"must be one of {list(QUANT_TIER_CHOICES)}")
        config["quant_tier"] = value
        result["changed"] = ["quant_tier"]

    elif op == "set_workers":
        value = command.get("workers", command.get("value"))
        if not is_int(value) or not 1 <= value <= 32:
            raise CommandError(codes.E_WORKERS, "workers", "must be integer in [1,32]")
        config["workers"] = int(value)
        result["changed"] = ["workers"]

    elif op == "set_budget":
        for field in ("batch_point_budget", "sample_point_quota"):
            if field in command:
                value = command[field]
                if not is_int(value) or value <= 0:
                    raise CommandError(codes.E_BUDGET, field, "must be positive integer")
                config[field] = int(value)
                result["changed"].append(field)

    elif op == "set_lengths":
        train = command.get("train_length", config["train_length"])
        target = command.get("target_length", config["target_length"])
        trial = dict(config)
        trial["train_length"] = train
        trial["target_length"] = target
        errors = validate_config(trial)
        length_errors = [item for item in errors if item["field"] in
                         ("train_length", "target_length")]
        if length_errors:
            raise CommandError(length_errors[0]["code"], length_errors[0]["field"],
                               length_errors[0]["message"])
        config["train_length"] = int(train)
        config["target_length"] = int(target)
        sample_id = command.get("sample_id")
        for sample in samples_by_id.values():
            if sample_id and sample["id"] != sample_id:
                continue
            sample["train_length"] = int(train)
            sample["target_length"] = int(target)
        result["changed"] = ["train_length", "target_length"]

    elif op == "replace_sample":
        raw_sample = command.get("sample")
        if not isinstance(raw_sample, dict):
            raise CommandError(codes.E_COMMAND_FIELD, "sample", "sample object required")
        candidate_id = raw_sample.get("id") if isinstance(raw_sample.get("id"), str) else None
        seen_ids = set(samples_by_id)
        if candidate_id in seen_ids:
            seen_ids.remove(candidate_id)
        sample, failure = validate_sample_record(
            raw_sample, int(line or 0), seen_ids, config.get("dimension"),
        )
        if failure:
            first = failure["errors"][0]
            raise CommandError(first["code"], first["field"], first["message"])
        samples_by_id[sample["id"]] = sample
        result["changed"] = [f"sample.{sample['id']}"]

    elif op == "delete_sample":
        sample_id = command.get("id") or command.get("sample_id")
        if sample_id not in samples_by_id:
            raise CommandError(codes.E_COMMAND_FIELD, "id", f"unknown sample {sample_id!r}")
        del samples_by_id[sample_id]
        result["changed"] = [f"deleted_sample.{sample_id}"]

    else:
        raise CommandError(codes.E_COMMAND_OP, "op", f"unsupported op {op!r}")

    errors = validate_config(config)
    if errors:
        raise CommandError(errors[0]["code"], errors[0]["field"], errors[0]["message"])
    return result
