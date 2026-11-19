"""Serializable, non-source-modifying injection configuration."""

from __future__ import annotations

import copy
import re
from typing import Any

import yaml

from .canonical import Coord, coord_to_list, row_major_key
from .errors import InjectionError

_PATH_TOKEN = re.compile(r"([A-Za-z0-9_-]+)(?:\[(\d+)\])?")


def empty_injection() -> dict[str, Any]:
    return {
        "schema": "level-workbench/injection-v1",
        "id": "none",
        "description": "",
        "spec_patches": [],
        "pipeline": {
            "subseed_offsets": {},
            "skip_local_repair": False,
            "skip_resource_repair": False,
            "pre_repair_force_terrain": [],
            "post_generation_force_terrain": [],
            "resource_count_override": None,
        },
        "artifact_tamper": {
            "force_terrain": [],
            "resource_count_override": None,
        },
    }


def load_injection_file(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise InjectionError("E_INJECTION", f"cannot read injection config: {exc}", "$") from exc
    return load_injection_text(text)


def load_injection_text(text: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InjectionError("E_INJECTION", f"invalid YAML injection config: {exc}", "$") from exc
    return normalize_injection(value)


def normalize_injection(value: Any) -> dict[str, Any]:
    if value is None:
        return empty_injection()
    if not isinstance(value, dict):
        raise InjectionError("E_INJECTION", "injection config root must be a mapping", "$")
    result = empty_injection()
    if "id" in value:
        result["id"] = _string(value["id"], "id")
    if "description" in value:
        description = value["description"]
        if not isinstance(description, str):
            raise InjectionError("E_INJECTION", "description must be a string", "description")
        result["description"] = description

    patches = value.get("spec_patches", [])
    if not isinstance(patches, list):
        raise InjectionError("E_INJECTION", "spec_patches must be a list", "spec_patches")
    for index, patch in enumerate(patches):
        path = f"spec_patches[{index}]"
        if not isinstance(patch, dict):
            raise InjectionError("E_INJECTION", "spec patch must be a mapping", path)
        op = patch.get("op", "set")
        if op != "set":
            raise InjectionError("E_INJECTION", "only the set operation is supported", f"{path}.op")
        target = _string(patch.get("path"), f"{path}.path")
        if "value" not in patch:
            raise InjectionError("E_INJECTION", "spec patch needs a value", f"{path}.value")
        _parse_path(target)
        result["spec_patches"].append({"op": "set", "path": target, "value": copy.deepcopy(patch["value"])})

    pipeline_raw = value.get("pipeline", {})
    if not isinstance(pipeline_raw, dict):
        raise InjectionError("E_INJECTION", "pipeline must be a mapping", "pipeline")
    pipeline = result["pipeline"]

    offsets = pipeline_raw.get("subseed_offsets", {})
    if not isinstance(offsets, dict):
        raise InjectionError("E_INJECTION", "pipeline.subseed_offsets must be a mapping", "pipeline.subseed_offsets")
    for purpose in sorted(offsets):
        name = _string(purpose, "pipeline.subseed_offsets key")
        number = offsets[purpose]
        if isinstance(number, bool) or not isinstance(number, int):
            raise InjectionError(
                "E_INJECTION",
                "subseed offset must be an integer",
                f"pipeline.subseed_offsets.{name}",
            )
        pipeline["subseed_offsets"][name] = number
    pipeline["subseed_offsets"] = {
        name: pipeline["subseed_offsets"][name] for name in sorted(pipeline["subseed_offsets"])
    }

    for name in ("skip_local_repair", "skip_resource_repair"):
        if name in pipeline_raw:
            value_bool = pipeline_raw[name]
            if not isinstance(value_bool, bool):
                raise InjectionError("E_INJECTION", f"pipeline.{name} must be boolean", f"pipeline.{name}")
            pipeline[name] = value_bool

    pipeline["pre_repair_force_terrain"] = _parse_forced_terrain(
        pipeline_raw.get("pre_repair_force_terrain", []), "pipeline.pre_repair_force_terrain"
    )
    pipeline["post_generation_force_terrain"] = _parse_forced_terrain(
        pipeline_raw.get("post_generation_force_terrain", []), "pipeline.post_generation_force_terrain"
    )
    pipeline["resource_count_override"] = _parse_optional_count(
        pipeline_raw.get("resource_count_override"), "pipeline.resource_count_override"
    )

    tamper_raw = value.get("artifact_tamper", {})
    if not isinstance(tamper_raw, dict):
        raise InjectionError("E_INJECTION", "artifact_tamper must be a mapping", "artifact_tamper")
    result["artifact_tamper"] = {
        "force_terrain": _parse_forced_terrain(
            tamper_raw.get("force_terrain", []), "artifact_tamper.force_terrain"
        ),
        "resource_count_override": _parse_optional_count(
            tamper_raw.get("resource_count_override"), "artifact_tamper.resource_count_override"
        ),
    }
    return result


def apply_spec_patches(raw_spec: dict[str, Any], injection: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(raw_spec)
    for patch in injection.get("spec_patches", []):
        _set_path(patched, patch["path"], copy.deepcopy(patch["value"]))
    return patched


def forced_terrain_coords(injection: dict[str, Any], key: str) -> list[Coord]:
    entries = injection.get(key, [])
    return sorted({(item["coord"][0], item["coord"][1]) for item in entries}, key=row_major_key)


def forced_terrain_map(injection: dict[str, Any], key: str) -> dict[Coord, str]:
    return {(item["coord"][0], item["coord"][1]): item["terrain"] for item in injection.get(key, [])}


def _parse_forced_terrain(value: Any, path: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InjectionError("E_INJECTION", f"{path} must be a list", path)
    result: list[dict[str, Any]] = []
    seen: set[Coord] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise InjectionError("E_INJECTION", "forced terrain entry must be a mapping", item_path)
        coord_raw = item.get("coord")
        terrain = item.get("terrain")
        if not isinstance(coord_raw, list) or len(coord_raw) != 2:
            raise InjectionError("E_INJECTION", "forced terrain coord must be [x,y]", f"{item_path}.coord")
        if any(isinstance(number, bool) or not isinstance(number, int) for number in coord_raw):
            raise InjectionError("E_INJECTION", "forced terrain coord must contain integers", f"{item_path}.coord")
        coord = (coord_raw[0], coord_raw[1])
        if coord in seen:
            raise InjectionError("E_INJECTION", "duplicate forced terrain coordinate", f"{item_path}.coord")
        seen.add(coord)
        result.append({"coord": coord_to_list(coord), "terrain": _string(terrain, f"{item_path}.terrain")})
    result.sort(key=lambda entry: row_major_key((entry["coord"][0], entry["coord"][1])))
    return result


def _parse_optional_count(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InjectionError("E_INJECTION", "resource count override must be a non-negative integer", path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InjectionError("E_INJECTION", "value must be a non-empty string", path)
    return value


def _parse_path(path: str) -> list[str | int]:
    if not path:
        raise InjectionError("E_INJECTION", "patch path must not be empty", "spec_patches.path")
    tokens: list[str | int] = []
    position = 0
    while position < len(path):
        if path[position] == ".":
            raise InjectionError("E_INJECTION", "invalid patch path", "spec_patches.path")
        match = _PATH_TOKEN.match(path, position)
        if match is None:
            raise InjectionError("E_INJECTION", f"invalid patch path near '{path[position:]}'", "spec_patches.path")
        tokens.append(match.group(1))
        if match.group(2) is not None:
            tokens.append(int(match.group(2)))
        position = match.end()
        if position < len(path):
            if path[position] != ".":
                raise InjectionError("E_INJECTION", "invalid patch path separator", "spec_patches.path")
            position += 1
            if position == len(path):
                raise InjectionError("E_INJECTION", "patch path must not end with '.'", "spec_patches.path")
    return tokens


def _set_path(root: Any, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    current = root
    for index, token in enumerate(tokens):
        last = index == len(tokens) - 1
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise InjectionError("E_INJECTION", f"patch path '{path}' traverses a non-mapping", path)
            if last:
                current[token] = value
                return
            if token not in current:
                raise InjectionError("E_INJECTION", f"patch path '{path}' does not exist", path)
            current = current[token]
        else:
            if not isinstance(current, list) or token >= len(current):
                raise InjectionError("E_INJECTION", f"patch path '{path}' has an invalid list index", path)
            if last:
                current[token] = value
                return
            current = current[token]
