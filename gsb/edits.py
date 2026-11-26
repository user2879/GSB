from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import InputValidationError


ALLOWED_OPS = {
    "set_weight",
    "set_domain_target_ratio",
    "set_config_field",
    "set_threshold",
    "set_shard_size",
    "set_min_tokens",
    "replace_sample",
    "patch_sample",
}


def load_command_stream(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputValidationError(f"cannot read command stream: {exc}", field="edits") from exc

    commands: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                f"invalid edit JSONL at line {line_number}: {exc.msg}",
                field=f"edits.line-{line_number}",
            ) from exc
        if not isinstance(command, dict):
            raise InputValidationError("edit command must be an object", field=f"edits.line-{line_number}")
        command_id = command.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise InputValidationError("command_id must be a nonempty string", field=f"edits[{line_number}].command_id")
        if command_id in seen_ids:
            raise InputValidationError("duplicate command_id", field="edits.command_id", details={"command_id": command_id})
        seen_ids.add(command_id)
        op = command.get("op")
        if op not in ALLOWED_OPS:
            raise InputValidationError(
                f"op must be one of {sorted(ALLOWED_OPS)}",
                field=f"edits.{command_id}.op",
            )
        commands.append(command)
    return commands


def apply_commands(
    config: dict[str, Any],
    batch_files: dict[str, list[dict[str, Any]]],
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    new_config = json.loads(json.dumps(config))
    new_batches = json.loads(json.dumps(batch_files))

    sample_locations = _index_samples(new_batches)

    for command in commands:
        op = command["op"]
        command_id = command["command_id"]
        if op == "set_weight":
            domain = _require_string(command, "domain", command_id)
            weight = command.get("weight")
            _require_number(weight, "weight", command_id)
            domain_obj = _find_domain(new_config, domain, command_id)
            domain_obj["sampling_weight"] = weight
        elif op == "set_domain_target_ratio":
            domain = _require_string(command, "domain", command_id)
            ratio = command.get("target_ratio")
            _require_number(ratio, "target_ratio", command_id)
            _find_domain(new_config, domain, command_id)["target_ratio"] = ratio
        elif op in {"set_config_field", "set_threshold", "set_shard_size", "set_min_tokens"}:
            field = command.get("field")
            value = command.get("value")
            if op == "set_threshold":
                field = "dedup.similarity_threshold_permille"
                value = command.get("threshold_permille")
            elif op == "set_shard_size":
                field = "dedup.shard_size"
                value = command.get("shard_size")
            elif op == "set_min_tokens":
                field = "cleaning.min_total_token_count"
                value = command.get("min_total_token_count")
            if not isinstance(field, str) or not field:
                raise InputValidationError("field must be a nonempty string", field=f"edits.{command_id}.field")
            _set_dotted(new_config, field, value, command_id)
        elif op == "replace_sample":
            sample = command.get("sample")
            if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
                raise InputValidationError(
                    "replace_sample requires a sample object with sample_id",
                    field=f"edits.{command_id}.sample",
                )
            sample_id = sample["sample_id"]
            location = sample_locations.get(sample_id)
            if location is None:
                raise InputValidationError(
                    "cannot replace absent sample; use add_batch for new samples",
                    field=f"edits.{command_id}.sample.sample_id",
                    details={"sample_id": sample_id},
                )
            filename, row_index = location
            new_batches[filename][row_index] = sample
        elif op == "patch_sample":
            sample_id = _require_string(command, "sample_id", command_id)
            patch = command.get("patch")
            if not isinstance(patch, dict):
                raise InputValidationError("patch must be an object", field=f"edits.{command_id}.patch")
            location = sample_locations.get(sample_id)
            if location is None:
                raise InputValidationError(
                    "cannot patch absent sample",
                    field=f"edits.{command_id}.sample_id",
                    details={"sample_id": sample_id},
                )
            filename, row_index = location
            new_batches[filename][row_index].update(patch)
        sample_locations = _index_samples(new_batches)

    return new_config, new_batches


def write_batch_files(root: str | Path, batch_files: dict[str, list[dict[str, Any]]]) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in sorted(batch_files):
        path = root / filename
        rows = batch_files[filename]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _index_samples(batch_files: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[str, int]]:
    locations: dict[str, tuple[str, int]] = {}
    for filename, rows in batch_files.items():
        for index, row in enumerate(rows):
            if isinstance(row, dict) and row.get("record_type", "sample") == "sample" and "sample_id" in row:
                locations[row["sample_id"]] = (filename, index)
    return locations


def _find_domain(config: dict[str, Any], domain: str, command_id: str) -> dict[str, Any]:
    for item in config.get("domains", []):
        if item.get("name") == domain:
            return item
    raise InputValidationError(
        "edit references undeclared domain",
        field=f"edits.{command_id}.domain",
        details={"domain": domain},
    )


def _require_string(command: dict[str, Any], key: str, command_id: str) -> str:
    value = command.get(key)
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{key} must be a nonempty string", field=f"edits.{command_id}.{key}")
    return value


def _require_number(value: Any, key: str, command_id: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{key} must be a finite nonnegative number", field=f"edits.{command_id}.{key}")
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        raise InputValidationError(f"{key} must be a finite nonnegative number", field=f"edits.{command_id}.{key}")


def _set_dotted(config: dict[str, Any], field: str, value: Any, command_id: str) -> None:
    parts = field.split(".")
    target = config
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            raise InputValidationError(
                "config field path does not exist",
                field=f"edits.{command_id}.field",
                details={"field": field},
            )
        target = target[part]
    if parts[-1] not in target:
        raise InputValidationError(
            "refusing to set unknown config field",
            field=f"edits.{command_id}.field",
            details={"field": field},
        )
    target[parts[-1]] = value
