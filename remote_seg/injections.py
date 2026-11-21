"""Materialize independent fault-injection profiles on a workspace copy."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .errors import E_INDEX_INVALID, E_PATCH_BOUNDS, SegmentationError
from .patches import apply_patch_batch
from .pngio import encode_png, read_png
from .util import atomic_write_bytes, atomic_write_json, read_json


def _copy_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _find_tile(index: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    row, col = item.get("row"), item.get("col")
    for tile in index.get("tiles", []):
        if tile.get("row") == row and tile.get("col") == col:
            return tile
    raise SegmentationError(
        E_INDEX_INVALID,
        f"injection target row={row} col={col} not found",
        stage="injection",
        row=row if isinstance(row, int) else None,
        col=col if isinstance(col, int) else None,
    )


def materialize_injections(
    source_workspace: Path,
    destination_workspace: Path,
    injection_plan_path: Path,
    *,
    profile: str | None = None,
    index_name: str = "index.json",
    config_name: str = "segmentation.json",
    tile_size: tuple[int, int] = (48, 48),
) -> dict[str, Any]:
    plan = read_json(injection_plan_path)
    profiles = plan.get("profiles") if isinstance(plan, dict) else None
    if profile is not None:
        if not isinstance(profiles, dict) or profile not in profiles:
            raise SegmentationError(
                E_INDEX_INVALID,
                f"injection profile {profile!r} is not declared",
                stage="injection",
            )
        items = profiles[profile]
    elif isinstance(profiles, dict):
        items = plan.get("items", [])
    else:
        items = plan.get("items") if isinstance(plan, dict) else None
    if not isinstance(items, list):
        raise SegmentationError(E_INDEX_INVALID, "injection items must be an array", stage="injection")

    _copy_workspace(source_workspace, destination_workspace)
    index_path = destination_workspace / index_name
    config_path = destination_workspace / config_name
    index = read_json(index_path)
    config = read_json(config_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "items": [],
        "preflight_failures": [],
    }

    for item in items:
        if not isinstance(item, dict):
            raise SegmentationError(E_INDEX_INVALID, "injection item must be an object", stage="injection")
        item_type = item.get("type")
        item_id = item.get("id")
        target_tile_id: str | None = None
        if not isinstance(item_id, str) or not item_id:
            raise SegmentationError(E_INDEX_INVALID, "injection item requires id", stage="injection")
        if item_type == "truncate_png":
            tile = _find_tile(index, item)
            target_tile_id = tile["tile_id"]
            path = destination_workspace / tile["path"]
            data = path.read_bytes()
            limit = item.get("truncate_bytes", max(1, len(data) // 4))
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise SegmentationError(E_INDEX_INVALID, "truncate_bytes must be positive")
            atomic_write_bytes(path, data[: min(limit, len(data) - 1)])
        elif item_type == "band_count_mismatch":
            tile = _find_tile(index, item)
            target_tile_id = tile["tile_id"]
            path = destination_workspace / tile["path"]
            decoded = read_png(path)
            bands = item.get("bands", 3)
            if not isinstance(bands, int) or not 1 <= bands <= 3 or bands >= decoded.channels:
                raise SegmentationError(E_INDEX_INVALID, "injected bands must be 1..3")
            atomic_write_bytes(path, encode_png(decoded.array[:, :, :bands]))
        elif item_type == "unknown_band_version":
            tile = _find_tile(index, item)
            target_tile_id = tile["tile_id"]
            tile["band_order_version"] = item.get("version", "bov-undeclared")
        elif item_type == "unknown_fusion_version":
            config.setdefault("overlap_fusion", {})[
                "strategy_version"
            ] = item.get("version", "f-undeclared")
        elif item_type == "patch_bounds":
            tile = _find_tile(index, item)
            target_tile_id = tile["tile_id"]
            rect = item.get("rect", [0, 0, tile_size[0] + 10, tile_size[1] + 10])
            validation = {
                "schema_version": 1,
                "batches": [
                    {
                        "batch": 1,
                        "operations": [
                            {
                                "tile_id": tile["tile_id"],
                                "row": tile["row"],
                                "col": tile["col"],
                                "rect": rect,
                                "seed": item.get("seed", 1),
                            }
                        ],
                    }
                ],
            }
            temp_plan = destination_workspace / ".injection_patch_plan.json"
            atomic_write_json(temp_plan, validation)
            try:
                apply_patch_batch(index_path, temp_plan, 1)
            except SegmentationError as exc:
                if exc.code != E_PATCH_BOUNDS:
                    raise
                manifest["preflight_failures"].append(
                    {
                        "code": exc.code,
                        "stage": "patch",
                        "message": exc.message,
                        "row": tile["row"],
                        "col": tile["col"],
                        "tile_id": tile["tile_id"],
                        "injection_id": item_id,
                        "details": {"rect": rect},
                    }
                )
            finally:
                temp_plan.unlink(missing_ok=True)
        else:
            raise SegmentationError(
                E_INDEX_INVALID, f"unsupported injection type {item_type!r}", stage="injection"
            )
        manifest["items"].append(
            {
                "id": item_id,
                "type": item_type,
                "row": item.get("row"),
                "col": item.get("col"),
                "tile_id": target_tile_id,
            }
        )

    atomic_write_json(index_path, index)
    atomic_write_json(config_path, config)
    atomic_write_json(destination_workspace / "injection_manifest.json", manifest)
    return manifest
