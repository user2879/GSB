from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Category, ConfigError, WorkbenchConfig
from .failures import Failure
from .validation import ImageSpec


@dataclass
class CommandResult:
    config: WorkbenchConfig
    specs: list[ImageSpec]
    failures: list[Failure]
    audit: list[dict[str, Any]]


def read_command_stream(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("commands", payload.get("operations"))
    if not isinstance(payload, list):
        raise ConfigError("command stream must be a list or an object containing commands")
    commands: list[dict[str, Any]] = []
    for index, command in enumerate(payload):
        if not isinstance(command, dict):
            raise ConfigError(f"command {index} must be an object")
        commands.append(command)
    return commands


def apply_commands(
    specs: list[ImageSpec],
    config: WorkbenchConfig,
    commands: list[dict[str, Any]],
    command_base: str | Path | None = None,
) -> CommandResult:
    base = Path(command_base).resolve() if command_base is not None else Path.cwd()
    by_name = {spec.name: spec for spec in specs}
    failures: list[Failure] = []
    audit: list[dict[str, Any]] = []
    current_config = config

    for index, command in enumerate(commands):
        op = str(command.get("op", command.get("operation", ""))).strip()
        item = f"commands[{index}]"
        try:
            if op in {"replace_image", "replace_single_image"}:
                name = _required_string(command, "image", "name")
                old_spec = by_name.get(name)
                image_path = _resolve_path(command.get("source", command.get("image_source")), base)
                truth_path = _resolve_truth_path(command, image_path, base)
                if image_path is None:
                    raise ConfigError("replace_image requires a source image path")
                if old_spec is not None:
                    by_name.pop(old_spec.name)
                spec = ImageSpec(name=name, image_path=image_path, truth_path=truth_path)
                by_name[name] = spec
            elif op in {"add_image", "insert_image"}:
                name = _required_string(command, "image", "name")
                if name in by_name:
                    raise ConfigError(f"image {name!r} already exists")
                image_path = _resolve_path(command.get("source", command.get("image_source")), base)
                if image_path is None:
                    raise ConfigError("add_image requires a source image path")
                truth_path = _resolve_truth_path(command, image_path, base)
                by_name[name] = ImageSpec(name=name, image_path=image_path, truth_path=truth_path)
            elif op in {"remove_image", "delete_image"}:
                name = _required_string(command, "image", "name")
                if name not in by_name:
                    raise ConfigError(f"image {name!r} does not exist")
                by_name.pop(name)
            elif op in {"add_category", "insert_category"}:
                current_config = _add_category(current_config, command)
            elif op in {"remove_category", "delete_category"}:
                current_config = _remove_category(current_config, command)
            elif op in {
                "set_merge_iou",
                "set_iou_threshold",
                "adjust_merge_iou",
                "adjust_iou_threshold",
            }:
                value = _numeric_value(command, "value", "iou", "threshold")
                if "delta" in command:
                    value = current_config.merge_iou + float(command["delta"])
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ConfigError("IoU adjustment requires a numeric value")
                if not 0.0 <= float(value) <= 1.0:
                    raise ConfigError("IoU threshold must be in [0, 1]")
                current_config = replace(current_config, merge_iou=float(value))
            elif op in {
                "set_min_instance_area",
                "set_min_area",
                "adjust_min_instance_area",
                "adjust_min_area",
            }:
                value = _numeric_value(command, "value", "min_instance_area", "min_area")
                if "delta" in command:
                    value = current_config.min_instance_area + int(command["delta"])
                if isinstance(value, bool) or not isinstance(value, int):
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    else:
                        raise ConfigError("minimum instance area must be an integer")
                if int(value) <= 0:
                    raise ConfigError("minimum instance area must be positive")
                current_config = replace(current_config, min_instance_area=int(value))
            elif op in {"set_quant_level", "switch_quant_level"}:
                value = _numeric_value(command, "value", "quant_level", "level")
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ConfigError("quant level must be an integer")
                if value not in current_config.quant_levels:
                    raise ConfigError("quant level is not in the allowed enumeration")
                current_config = replace(current_config, quant_level=value)
            elif op in {"set_max_workers"}:
                value = _numeric_value(command, "value", "max_workers")
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ConfigError("max workers must be a positive integer")
                current_config = replace(current_config, max_workers=value)
            else:
                raise ConfigError(f"unsupported operation {op!r}")
            audit.append(
                {
                    "command": _stable_command(command),
                    "index": index,
                    "op": op,
                    "status": "applied",
                }
            )
        except ConfigError as exc:
            failures.append(
                Failure(
                    code="E_CONFIG_INVALID",
                    item=item,
                    stage="load",
                    message=str(exc),
                    details={"command": _stable_command(command)},
                )
            )

    return CommandResult(
        config=current_config,
        specs=sorted(by_name.values(), key=lambda spec: spec.name),
        failures=failures,
        audit=audit,
    )


def _add_category(config: WorkbenchConfig, command: dict[str, Any]) -> WorkbenchConfig:
    raw = command.get("category")
    if raw is None:
        raw = {
            "code": command.get("code"),
            "name": command.get("name"),
            "weight": command.get("weight", 1.0),
        }
    if not isinstance(raw, dict):
        raise ConfigError("category must be an object")
    code = raw.get("code", raw.get("category_code"))
    name = raw.get("name")
    weight = raw.get("weight", 1.0)
    if isinstance(code, bool) or not isinstance(code, int):
        raise ConfigError("category code must be an integer")
    if code < 0:
        raise ConfigError("category code must be non-negative")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("category name must be a non-empty string")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or float(weight) <= 0.0
    ):
        raise ConfigError("category weight must be positive")
    if any(category.code == code for category in config.categories):
        raise ConfigError(f"category code {code} already exists")
    if any(category.name == name.strip() for category in config.categories):
        raise ConfigError(f"category name {name!r} already exists")
    category = Category(
        code=code,
        name=name.strip(),
        weight_milli=int(round(float(weight) * 1000)),
    )
    return replace(
        config,
        categories=tuple(sorted(config.categories + (category,), key=lambda item: item.code)),
    )


def _remove_category(config: WorkbenchConfig, command: dict[str, Any]) -> WorkbenchConfig:
    value = command.get("code", command.get("category_code", command.get("category", command.get("name"))))
    category: Category | None = None
    if isinstance(value, int) and not isinstance(value, bool):
        category = next((item for item in config.categories if item.code == value), None)
    elif isinstance(value, str):
        category = next((item for item in config.categories if item.name == value), None)
        if category is None:
            try:
                numeric = int(value)
            except ValueError:
                numeric = None
            if numeric is not None:
                category = next(
                    (item for item in config.categories if item.code == numeric), None
                )
    if category is None:
        raise ConfigError("category does not exist")
    remaining = tuple(item for item in config.categories if item.code != category.code)
    if not remaining:
        raise ConfigError("cannot remove the final category")
    return replace(config, categories=remaining)


def _required_string(command: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = command.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ConfigError(f"missing string field {keys[0]}")


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("path must be a non-empty string")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_truth_path(command: dict[str, Any], image_path: Path, base: Path) -> Path:
    truth_value = command.get("truth", command.get("truth_path"))
    if truth_value is not None:
        path = _resolve_path(truth_value, base)
        if path is None:
            raise ConfigError("truth path cannot be empty")
        return path
    return image_path.with_suffix(".json")


def _numeric_value(command: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in command:
            return command[key]
    raise ConfigError(f"missing numeric field {keys[0]}")


def _stable_command(command: dict[str, Any]) -> dict[str, Any]:
    return {key: command[key] for key in sorted(command)}
