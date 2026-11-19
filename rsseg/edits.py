"""Apply the deterministic edit sequence to a mutable working sample copy.

Edits operate on a scratch copy of the sample set so the pristine generated
samples stay reusable. Four edit kinds are supported:

* ``config``: deep-merge threshold / weight changes into the config JSON,
* ``adjacency``: remove a symmetric neighbor edge,
* ``restore_adjacency``: re-add a symmetric neighbor edge,
* ``band_patch``: write an integer-only ``.npy`` band patch for one tile.

Every edit is itself deterministic and serialisable; replaying the same
sequence from scratch always yields byte-identical inputs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def copy_sample_set(source: Path, target: Path) -> Path:
    import shutil

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def apply_edit(working_dir: Path, step: dict, *, rng_seed: int = 20261120) -> dict:
    kind = step["type"]
    if kind == "config":
        config_path = working_dir / "segmentation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config = deep_merge(config, step["patch"])
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
        return {"kind": kind, "id": step["id"], "config_hash_patch": step["patch"]}

    if kind in {"adjacency", "restore_adjacency"}:
        index_path = working_dir / "tile_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        edge = step.get("edge") or step.get("remove")
        left, right = tuple(edge[0]), tuple(edge[1])
        entries = {(e["row"], e["col"]): e for e in index["tiles"]}
        if kind == "adjacency":
            entries[left]["neighbors"] = [n for n in entries[left]["neighbors"] if list(right) != n]
            entries[right]["neighbors"] = [n for n in entries[right]["neighbors"] if list(left) != n]
        else:
            if list(right) not in entries[left]["neighbors"]:
                entries[left]["neighbors"].append(list(right))
            if list(left) not in entries[right]["neighbors"]:
                entries[right]["neighbors"].append(list(left))
            for entry in entries.values():
                entry["neighbors"].sort()
        index_path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")
        return {"kind": kind, "id": step["id"], "edge": edge}

    if kind == "band_patch":
        row, col = step["tile"]
        band = int(step["band"])
        patches_dir = working_dir / "patches"
        patches_dir.mkdir(exist_ok=True)
        tile_name = f"tile_r{row:03d}_c{col:03d}"
        patch_path = patches_dir / f"{tile_name}_b{band}.npy"
        rng = np.random.default_rng(rng_seed + int(step.get("seed_offset", 0)))
        existing = np.array(Image.open(working_dir / "tiles" / f"{tile_name}.png"))
        patch = existing[:, :, band].copy()
        delta = rng.integers(-20, 21, size=patch.shape, dtype=np.int16)
        patch = np.clip(patch.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        np.save(patch_path, patch)
        return {"kind": kind, "id": step["id"], "tile": [row, col], "band": band,
                "patch_file": str(patch_path.relative_to(working_dir))}

    raise ValueError(f"unknown edit type {kind}")


def replay_sequence(working_dir: Path, sequence_path: Path) -> list[dict]:
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    trace = []
    for step in sequence["steps"]:
        trace.append(apply_edit(working_dir, step))
    return trace
