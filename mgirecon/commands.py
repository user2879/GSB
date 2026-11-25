"""Validated edit-command stream application."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import det3, rotation_is_orthogonal, save_config, voxel_size_vector
from .errors import E_PARAM_INVALID, E_VIEW_UNKNOWN, diagnostic
from .synth import write_view


def _int_matrix(value: Any, rows: int, cols: int) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) != rows:
        return None
    result: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            return None
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in row):
            return None
        result.append(list(row))
    return result


def _view_index(config: dict[str, Any], view_id: str) -> int | None:
    for index, view in enumerate(config.get("views", [])):
        if view.get("id") == view_id:
            return index
    return None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def apply_commands(
    project_dir: str | Path,
    config: dict[str, Any],
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(project_dir)
    updated = copy.deepcopy(config)
    errors: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            errors.append(diagnostic(E_PARAM_INVALID, "command must be an object", stage="command", detail={"index": index}))
            continue
        operation = command.get("op")
        if operation == "replace_view":
            view_id = command.get("view")
            view_index = _view_index(updated, view_id) if isinstance(view_id, str) else None
            if view_index is None:
                errors.append(
                    diagnostic(
                        E_VIEW_UNKNOWN,
                        f"unknown view {view_id!r}; command skipped",
                        stage="command",
                        view=view_id if isinstance(view_id, str) else None,
                        detail={"index": index},
                    )
                )
                continue
            render_seed = command.get("render_seed")
            if not _non_negative_int(render_seed):
                errors.append(
                    diagnostic(
                        E_PARAM_INVALID,
                        "replace_view.render_seed must be a non-negative integer",
                        stage="command",
                        view=view_id,
                        detail={"index": index},
                    )
                )
                continue
            updated["views"][view_index]["render_seed"] = render_seed
            write_view(root, updated, updated["views"][view_index])
            continue
        if operation in {"set_extrinsics", "set_intrinsics"}:
            view_id = command.get("view")
            view_index = _view_index(updated, view_id) if isinstance(view_id, str) else None
            if view_index is None:
                errors.append(
                    diagnostic(
                        E_VIEW_UNKNOWN,
                        f"unknown view {view_id!r}; command skipped",
                        stage="command",
                        view=view_id if isinstance(view_id, str) else None,
                        detail={"index": index},
                    )
                )
                continue
            if operation == "set_extrinsics":
                matrix = _int_matrix(command.get("value"), 3, 4)
                if matrix is None or not rotation_is_orthogonal([row[:3] for row in matrix]):
                    errors.append(
                        diagnostic(
                            E_PARAM_INVALID,
                            "extrinsics must be a 3x4 integer matrix with orthogonal rotation",
                            stage="command",
                            view=view_id,
                            detail={"index": index},
                        )
                    )
                    continue
                updated["views"][view_index]["extrinsics"] = matrix
            else:
                matrix = _int_matrix(command.get("value"), 3, 3)
                if matrix is None or det3(matrix) == 0:
                    errors.append(
                        diagnostic(
                            E_PARAM_INVALID,
                            "intrinsics must be an invertible 3x3 integer matrix",
                            stage="command",
                            view=view_id,
                            detail={"index": index},
                        )
                    )
                    continue
                updated["views"][view_index]["intrinsics"] = matrix
            continue
        if operation == "set_param":
            name = command.get("name")
            value = command.get("value")
            reconstruction = updated.setdefault("reconstruction", {})
            scheduler = updated.setdefault("scheduler", {})
            matching = updated.setdefault("matching", {})
            valid = False
            if name in {"min_disparity", "residual_threshold", "voxel_max_depth"}:
                valid = _positive_int(value)
                if valid:
                    reconstruction[name] = value
            elif name == "min_occupancy_points":
                valid = _non_negative_int(value)
                if valid:
                    reconstruction[name] = value
            elif name == "quantization_level":
                valid = value in reconstruction.get("quantization_levels", [])
                if valid:
                    reconstruction[name] = value
            elif name == "voxel_size":
                normalized = voxel_size_vector(value)
                valid = normalized is not None
                if valid:
                    reconstruction[name] = normalized if isinstance(value, list) else value
            elif name == "global_point_budget":
                valid = _non_negative_int(value)
                if valid:
                    scheduler[name] = value
            elif name == "per_view_point_quota":
                valid = _non_negative_int(value)
                if valid:
                    scheduler[name] = value
            elif name == "max_workers":
                valid = _positive_int(value)
                if valid:
                    scheduler[name] = value
            elif name == "min_inliers":
                valid = _positive_int(value)
                if valid:
                    matching[name] = value
            elif name == "max_pairs":
                valid = _positive_int(value)
                if valid:
                    matching[name] = value
            if not valid:
                errors.append(
                    diagnostic(
                        E_PARAM_INVALID,
                        f"invalid parameter {name!r} or value; command skipped",
                        stage="command",
                        detail={"index": index, "name": name, "value": value},
                    )
                )
            continue
        errors.append(
            diagnostic(
                E_PARAM_INVALID,
                f"unknown command operation {operation!r}; command skipped",
                stage="command",
                detail={"index": index},
            )
        )
    save_config(root, updated)
    return updated, errors


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("commands"), list):
        return value["commands"]
    if not isinstance(value, list):
        raise ValueError("command stream must be a JSON list or {'commands': [...]} object")
    return value
