"""Deterministic edit-command stream application."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .util import deep_set, load_structured, write_json


def _load_band(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        mode = image.mode
        array = np.array(image)
    return array, mode


def _save_band(path: Path, array: np.ndarray, mode: str) -> None:
    if mode == "I;16":
        Image.fromarray(array, mode="I;16").save(path, optimize=False)
    else:
        Image.fromarray(array, mode="L").save(path, optimize=False)


def apply_edit_commands(config_path: Path, edit_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_structured(config_path)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    commands = load_structured(edit_path)
    root = config_path.parent
    manifest_path = (root / config["paths"]["manifest"]).resolve()
    tiles_dir = (root / config["paths"]["tiles_dir"]).resolve()
    applied: list[dict[str, Any]] = []

    for command in commands.get("commands", []):
        operation = command["op"]
        if operation == "set_config":
            deep_set(config, command["path"], command["value"])
            write_json(config_path, config)
            applied.append({"op": operation, "path": command["path"], "value": command["value"]})
            continue
        if operation != "replace_band":
            raise ValueError(f"unsupported edit operation {operation}")
        tile_id = command["tile_id"]
        band = command["band"]
        band_path = tiles_dir / tile_id / f"{band}.png"
        array, mode = _load_band(band_path)
        method = command["method"]
        if method == "offset":
            modified = array.astype(np.int64) + int(command["value"])
        elif method == "patch_add":
            modified = array.astype(np.int64).copy()
            row, col, height, width = command["rect"]
            patch = modified[row : row + height, col : col + width]
            patch += int(command["value"])
        elif method == "replace_file":
            source = Path(command["source"])
            shutil.copyfile(source, band_path)
            applied.append(
                {
                    "op": operation,
                    "tile_id": tile_id,
                    "band": band,
                    "method": method,
                    "source": str(source),
                }
            )
            continue
        else:
            raise ValueError(f"unsupported replace_band method {method}")
        maximum = 65535 if mode == "I;16" else 255
        modified = np.clip(modified, 0, maximum)
        _save_band(band_path, modified.astype(array.dtype), mode)
        applied.append(
            {
                "op": operation,
                "tile_id": tile_id,
                "band": band,
                "method": method,
                "value": command.get("value"),
                "rect": command.get("rect"),
            }
        )
    return {"applied": applied, "config_path": str(config_path), "manifest_path": str(manifest_path)}


def clone_sample(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def write_edit_commands(path: Path, commands: list[dict[str, Any]]) -> None:
    write_json(path, {"schema_version": 1, "commands": commands})

