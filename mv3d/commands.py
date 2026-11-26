"""Validated, ordered edit commands for incremental reconstruction."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .constants import ALLOWED_FIXED_BITS, ALLOWED_QUANT_LEVELS, STRATEGY_TABLE
from .errors import Failure, sorted_failures
from .models import ProjectState, parse_camera


def _resolve_path(raw: str, state: ProjectState) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = state.base_dir / path
    return str(path.resolve())


def _find_view(state: ProjectState, view_id: int) -> dict[str, Any] | None:
    matches = [item for item in state.data.get("views", []) if int(item.get("id", -1)) == view_id]
    return matches[0] if len(matches) == 1 else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _apply_one(state: ProjectState, command: dict[str, Any], index: int) -> list[Failure]:
    failures: list[Failure] = []
    artifact = f"cmd:{index:04d}"
    op = command.get("op")
    view_id = _int(command.get("view_id"))
    if not isinstance(op, str):
        return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing op")]
    if op == "replace_view":
        if view_id is None:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing view_id")]
        view = _find_view(state, view_id)
        if view is None:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="view not found")]
        raw_path = command.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="missing path")]
        view["image"] = _resolve_path(raw_path, state)
        return failures
    if op == "update_camera_extrinsics":
        if view_id is None:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing view_id")]
        view = _find_view(state, view_id)
        if view is None:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="view not found")]
        camera = copy.deepcopy(view.get("camera", {}))
        if "rotation" in command:
            camera["rotation"] = command["rotation"]
        if "rotation_scale" in command:
            camera["rotation_scale"] = command["rotation_scale"]
        if "translation" in command:
            camera["translation"] = command["translation"]
        if not any(key in command for key in ("rotation", "rotation_scale", "translation")):
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="no extrinsics fields")]
        try:
            parse_camera(camera)
        except (KeyError, TypeError, ValueError) as error:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail=str(error))]
        view["camera"] = camera
        return failures
    if op == "update_camera_intrinsics":
        if view_id is None:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing view_id")]
        view = _find_view(state, view_id)
        if view is None:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="view not found")]
        if "intrinsic" not in command:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="missing intrinsic")]
        camera = copy.deepcopy(view.get("camera", {}))
        camera["intrinsic"] = command["intrinsic"]
        try:
            parse_camera(camera)
        except (KeyError, TypeError, ValueError) as error:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail=str(error))]
        view["camera"] = camera
        return failures
    if op == "set_min_disparity":
        value = _int(command.get("value"))
        if value is None or value <= 0:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="value must be positive integer")]
        state.data.setdefault("reconstruction", {})["min_disparity"] = value
        return failures
    if op == "set_voxel_size":
        value = _int(command.get("value"))
        if value is None or value <= 0:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="value must be positive integer")]
        state.data.setdefault("reconstruction", {})["voxel_size"] = value
        return failures
    if op == "set_quantization":
        bits = _int(command.get("bits"))
        quant = _int(command.get("quant_level"))
        if bits not in ALLOWED_FIXED_BITS:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="bits not allowed")]
        if quant not in ALLOWED_QUANT_LEVELS:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="quant_level not allowed")]
        state.data.setdefault("fixed", {})["bits"] = bits
        state.data.setdefault("fixed", {})["quant_level"] = quant
        return failures
    if op == "set_budget":
        recon = state.data.setdefault("reconstruction", {})
        if "global_point_budget" in command:
            value = _int(command["global_point_budget"])
            if value is None or value < 0:
                return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="global_point_budget must be nonnegative")]
            recon["global_point_budget"] = value
        if "view_point_quota" in command:
            value = _int(command["view_point_quota"])
            if value is None or value <= 0:
                return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="view_point_quota must be positive")]
            recon["view_point_quota"] = value
        return failures
    if op == "set_workers":
        value = _int(command.get("value"))
        if value is None or value <= 0:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="value must be positive integer")]
        state.data.setdefault("reconstruction", {})["workers"] = value
        return failures
    if op == "set_strategy_override":
        value = command.get("value")
        allowed = {entry["name"] for entry in STRATEGY_TABLE}
        if value is not None and value not in allowed:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="unknown strategy")]
        state.data.setdefault("reconstruction", {})["strategy_override"] = value
        return failures
    if op == "remove_view":
        if view_id is None:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing view_id")]
        view = _find_view(state, view_id)
        if view is None:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="view not found")]
        state.data["views"] = [item for item in state.data.get("views", []) if item is not view]
        state.data["pairs"] = [
            pair for pair in state.data.get("pairs", []) if view_id not in pair
        ]
        return failures
    if op == "add_view":
        if view_id is None:
            return [Failure("E_COMMAND_INVALID", artifact=artifact, detail="missing view_id")]
        if _find_view(state, view_id) is not None:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="view already exists")]
        raw_path = command.get("path")
        raw_camera = command.get("camera")
        if not isinstance(raw_path, str) or not isinstance(raw_camera, dict):
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail="path and camera required")]
        try:
            parse_camera(raw_camera)
        except (KeyError, TypeError, ValueError) as error:
            return [Failure("E_COMMAND_INVALID", view_id=view_id, artifact=artifact, detail=str(error))]
        state.data.setdefault("views", []).append(
            {
                "id": view_id,
                "image": _resolve_path(raw_path, state),
                "camera": raw_camera,
            }
        )
        return failures
    return [Failure("E_COMMAND_INVALID", artifact=artifact, detail=f"unknown op {op}")]


def apply_commands(
    state: ProjectState,
    commands: list[dict[str, Any]],
) -> tuple[ProjectState, list[Failure]]:
    updated = state.clone()
    failures: list[Failure] = []
    for index, command in enumerate(commands, start=1):
        failures.extend(_apply_one(updated, command, index))
    return updated, sorted_failures(failures)

