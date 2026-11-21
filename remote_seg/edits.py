"""Editable example workspace and deterministic edit-sequence semantics."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .errors import E_INDEX_INVALID, E_PATCH_BOUNDS, SegmentationError
from .patches import apply_patch_batch
from .util import atomic_write_json, canonical_hash, deep_set, read_json


def _next_index_version(index: dict[str, Any]) -> str:
    payload = {
        "tiles": index.get("tiles", []),
        "global_bounds": index.get("global_bounds"),
    }
    return "idx-" + canonical_hash(payload, prefix="edited-index-v1|")[:16]


def _tile_by_cell(index: dict[str, Any], row: int, col: int) -> dict[str, Any]:
    for tile in index.get("tiles", []):
        if tile.get("row") == row and tile.get("col") == col:
            return tile
    raise SegmentationError(
        E_INDEX_INVALID,
        f"tile row={row} col={col} does not exist",
        stage="edit",
        row=row,
        col=col,
    )


def _tile_by_id(index: dict[str, Any], tile_id: str) -> dict[str, Any]:
    for tile in index.get("tiles", []):
        if tile.get("tile_id") == tile_id:
            return tile
    raise SegmentationError(
        E_INDEX_INVALID,
        f"tile {tile_id!r} does not exist",
        stage="edit",
        tile_id=tile_id,
    )


def _refresh_global_bounds(index: dict[str, Any]) -> None:
    tiles = index.get("tiles", [])
    if not tiles:
        index["global_bounds"] = [0, 0, 0, 0]
        return
    index["global_bounds"] = [
        min(tile["bounds"][0] for tile in tiles),
        min(tile["bounds"][1] for tile in tiles),
        max(tile["bounds"][2] for tile in tiles),
        max(tile["bounds"][3] for tile in tiles),
    ]


def _remove_tile(index: dict[str, Any], row: int, col: int) -> dict[str, Any]:
    tile = _tile_by_cell(index, row, col)
    tile_id = tile["tile_id"]
    index["tiles"] = [item for item in index["tiles"] if item["tile_id"] != tile_id]
    for item in index["tiles"]:
        item["neighbors"] = [neighbor for neighbor in item.get("neighbors", []) if neighbor != tile_id]
    _refresh_global_bounds(index)
    index["index_version"] = _next_index_version(index)
    return tile


def _add_tile(index: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
    required = ("tile_id", "row", "col", "path", "bounds", "neighbors")
    missing = [name for name in required if name not in operation]
    if missing:
        raise SegmentationError(
            E_INDEX_INVALID, f"add_tile missing fields: {missing}", stage="edit"
        )
    tile_id = operation["tile_id"]
    if any(tile["tile_id"] == tile_id for tile in index["tiles"]):
        raise SegmentationError(E_INDEX_INVALID, f"tile {tile_id!r} already exists", stage="edit")
    row, col = operation["row"], operation["col"]
    if any(tile["row"] == row and tile["col"] == col for tile in index["tiles"]):
        raise SegmentationError(
            E_INDEX_INVALID, f"row/col {row},{col} already exists", stage="edit"
        )
    new_tile = {
        "tile_id": tile_id,
        "row": row,
        "col": col,
        "path": operation["path"],
        "bounds": list(operation["bounds"]),
        "band_order_version": operation.get(
            "band_order_version", index.get("band_order_version", "bov-1")
        ),
        "neighbors": list(operation["neighbors"]),
    }
    for neighbor_id in new_tile["neighbors"]:
        neighbor = _tile_by_id(index, neighbor_id)
        if tile_id not in neighbor.setdefault("neighbors", []):
            neighbor["neighbors"].append(tile_id)
            neighbor["neighbors"].sort()
    index["tiles"].append(new_tile)
    index["tiles"].sort(key=lambda item: (item["row"], item["col"], item["tile_id"]))
    _refresh_global_bounds(index)
    index["index_version"] = _next_index_version(index)
    return new_tile


def _set_adjacency(index: dict[str, Any], row: int, col: int, neighbors: list[str]) -> None:
    tile = _tile_by_cell(index, row, col)
    tile_id = tile["tile_id"]
    requested = sorted(set(neighbors))
    for neighbor_id in requested:
        _tile_by_id(index, neighbor_id)
        if neighbor_id == tile_id:
            raise SegmentationError(E_INDEX_INVALID, "tile cannot neighbor itself", stage="edit")
    old_neighbors = set(tile.get("neighbors", []))
    for other in index["tiles"]:
        if other["tile_id"] == tile_id:
            continue
        other_neighbors = set(other.get("neighbors", []))
        if other["tile_id"] in requested:
            other_neighbors.add(tile_id)
        elif other["tile_id"] in old_neighbors:
            other_neighbors.discard(tile_id)
        other["neighbors"] = sorted(other_neighbors)
    tile["neighbors"] = requested
    index["index_version"] = _next_index_version(index)


def apply_edit_operation(
    workspace: Path,
    operation: dict[str, Any],
    *,
    index_path: Path,
    config_path: Path,
    patch_plan_path: Path,
) -> dict[str, Any]:
    operation_type = operation.get("type")
    if operation_type == "band_patch":
        batch = operation.get("batch")
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise SegmentationError(E_PATCH_BOUNDS, "band_patch.batch must be an integer")
        records = apply_patch_batch(index_path, patch_plan_path, batch)
        return {
            "operation_id": operation.get("id"),
            "type": operation_type,
            "batch": batch,
            "patches": [
                {
                    "tile_id": record.tile_id,
                    "row": record.row,
                    "col": record.col,
                    "rect": list(record.rect),
                    "before_fingerprint": record.before_fingerprint,
                    "after_fingerprint": record.after_fingerprint,
                }
                for record in records
            ],
        }
    if operation_type == "config_patch":
        path = operation.get("path")
        if not isinstance(path, str) or "value" not in operation:
            raise SegmentationError(
                E_INDEX_INVALID, "config_patch requires path and value", stage="edit"
            )
        config = read_json(config_path)
        deep_set(config, path, operation["value"])
        atomic_write_json(config_path, config)
        return {"operation_id": operation.get("id"), "type": operation_type, "path": path}
    if operation_type == "set_fusion_version":
        version = operation.get("version")
        if not isinstance(version, str) or not version:
            raise SegmentationError(
                E_INDEX_INVALID, "set_fusion_version requires version", stage="edit"
            )
        config = read_json(config_path)
        config.setdefault("overlap_fusion", {})["strategy_version"] = version
        atomic_write_json(config_path, config)
        return {"operation_id": operation.get("id"), "type": operation_type, "version": version}

    index = read_json(index_path)
    if operation_type == "delete_tile":
        row, col = operation.get("row"), operation.get("col")
        removed = _remove_tile(index, row, col)
        atomic_write_json(index_path, index)
        return {
            "operation_id": operation.get("id"),
            "type": operation_type,
            "tile_id": removed["tile_id"],
            "row": row,
            "col": col,
        }
    if operation_type == "add_tile":
        source = operation.get("source_path")
        if not isinstance(source, str) or not source:
            raise SegmentationError(E_INDEX_INVALID, "add_tile requires source_path", stage="edit")
        destination = workspace / operation["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / source, destination)
        added = _add_tile(index, operation)
        atomic_write_json(index_path, index)
        return {
            "operation_id": operation.get("id"),
            "type": operation_type,
            "tile_id": added["tile_id"],
            "row": added["row"],
            "col": added["col"],
        }
    if operation_type == "set_adjacency":
        row, col = operation.get("row"), operation.get("col")
        neighbors = operation.get("neighbors")
        if not isinstance(neighbors, list) or not all(isinstance(v, str) for v in neighbors):
            raise SegmentationError(
                E_INDEX_INVALID, "set_adjacency.neighbors must be strings", stage="edit"
            )
        _set_adjacency(index, row, col, neighbors)
        atomic_write_json(index_path, index)
        return {
            "operation_id": operation.get("id"),
            "type": operation_type,
            "row": row,
            "col": col,
            "neighbors": sorted(set(neighbors)),
        }
    raise SegmentationError(
        E_INDEX_INVALID, f"unsupported edit operation {operation_type!r}", stage="edit"
    )


def apply_edit_sequence(
    workspace: Path,
    sequence_path: Path,
    *,
    after_operation: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    sequence = read_json(sequence_path)
    operations = sequence.get("operations")
    if not isinstance(operations, list):
        raise SegmentationError(E_INDEX_INVALID, "edit sequence has no operations", stage="edit")
    index_path = workspace / sequence.get("index", "index.json")
    config_path = workspace / sequence.get("config", "segmentation.json")
    patch_plan_path = workspace / sequence.get("patch_plan", "patch_plan.json")
    outcomes: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise SegmentationError(E_INDEX_INVALID, "edit operation must be an object", stage="edit")
        outcome = apply_edit_operation(
            workspace,
            operation,
            index_path=index_path,
            config_path=config_path,
            patch_plan_path=patch_plan_path,
        )
        outcomes.append(outcome)
        if after_operation is not None:
            after_operation(operation, outcome)
    return outcomes
