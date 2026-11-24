"""Deterministic edit-command stream application."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, E_IO, E_VERSION_ORDER, E_WEIGHT_INVALID, FailureList
from .loading import load_config
from .model import Dataset, Layer
from .utils import flatten_numeric


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _validate_version_order(layers: list[Layer], failures: FailureList) -> bool:
    previous: int | None = None
    for layer in layers:
        if previous is not None and layer.content_version < previous:
            failures.add(
                E_VERSION_ORDER,
                f"edited layer order violates monotonic content version: {layer.content_version} < {previous}",
                "commands",
                layer=layer.name,
            )
            return False
        previous = layer.content_version
    return True


def load_commands(path: Path, failures: FailureList) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        failures.add(E_IO, f"cannot read command stream: {exc}", "commands", path=str(path))
        return []
    commands: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.add(
                E_CONFIG_INVALID,
                f"invalid command JSON: {exc.msg}",
                "commands",
                path=str(path),
                details={"line": line_number},
            )
            continue
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            failures.add(
                E_CONFIG_INVALID,
                "each command must be an object with an op",
                "commands",
                path=str(path),
                details={"line": line_number},
            )
            continue
        command = dict(command)
        command["_line"] = line_number
        commands.append(command)
    return commands


def apply_commands(dataset: Dataset, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = dataset.failures
    applied: list[dict[str, Any]] = []
    for command in commands:
        op = command.get("op")
        line = command.get("_line")
        if op in {"replace_layer", "set_layer_weights"}:
            name = command.get("layer", command.get("name"))
            index = next((i for i, layer in enumerate(dataset.layers) if layer.name == name), None)
            if index is None:
                failures.add(
                    E_WEIGHT_INVALID,
                    f"command references unknown layer {name!r}",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
                continue
            original = dataset.layers[index]
            shape_value = command.get("shape", list(original.shape))
            try:
                shape = tuple(int(value) for value in shape_value)
                weights = flatten_numeric(command.get("weights", command.get("values")))
            except (TypeError, ValueError) as exc:
                failures.add(
                    E_WEIGHT_INVALID,
                    f"invalid replacement layer: {exc}",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
                continue
            if (
                any(dimension <= 0 for dimension in shape)
                or len(weights) != math.prod(shape)
                or any(not math.isfinite(value) for value in weights)
            ):
                failures.add(
                    E_WEIGHT_INVALID,
                    "replacement shape/weights are inconsistent or contain non-finite values",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
                continue
            version = command.get("content_version", original.content_version)
            group = command.get("group_granularity", original.group_granularity)
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                failures.add(
                    E_WEIGHT_INVALID,
                    "replacement content_version must be a non-negative integer",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
                continue
            if isinstance(group, bool) or not isinstance(group, int) or group <= 0 or len(weights) % group:
                failures.add(
                    E_WEIGHT_INVALID,
                    "replacement group_granularity must divide the weight count",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
                continue
            dataset.layers[index] = Layer(
                name=original.name,
                shape=shape,
                weights=tuple(weights),
                content_version=version,
                group_granularity=group,
                source_line=original.source_line,
            )
            applied.append(
                {
                    "line": line,
                    "op": op,
                    "layer": name,
                    "content_version": version,
                    "shape": list(shape),
                    "element_count": len(weights),
                }
            )
            if not _validate_version_order(dataset.layers, failures):
                break
        elif op == "delete_layer":
            name = command.get("layer", command.get("name"))
            before = len(dataset.layers)
            dataset.layers = [layer for layer in dataset.layers if layer.name != name]
            if len(dataset.layers) == before:
                failures.add(
                    E_WEIGHT_INVALID,
                    f"delete command references unknown layer {name!r}",
                    "commands",
                    layer=str(name),
                    details={"line": line},
                )
            else:
                applied.append({"line": line, "op": op, "layer": name})
        elif op in {"patch_config", "set_config"}:
            patch = command.get("set", command.get("patch"))
            if not isinstance(patch, dict):
                failures.add(
                    E_CONFIG_INVALID,
                    "patch_config requires a set object",
                    "commands",
                    details={"line": line},
                )
                continue
            raw = copy.deepcopy(dataset.config.raw) if dataset.config is not None else {}
            _deep_merge(raw, patch)
            from tempfile import NamedTemporaryFile

            # Reuse the canonical config validator without making the command stream
            # depend on the original YAML file.
            with NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                import yaml

                yaml.safe_dump(raw, handle, sort_keys=False)
                temporary_path = Path(handle.name)
            try:
                dataset.config = load_config(temporary_path, failures)
            finally:
                temporary_path.unlink(missing_ok=True)
            applied.append({"line": line, "op": op, "keys": sorted(patch)})
        elif op == "noop":
            applied.append({"line": line, "op": op})
        else:
            failures.add(
                E_CONFIG_INVALID,
                f"unsupported command op {op!r}",
                "commands",
                details={"line": line},
            )
    return applied
