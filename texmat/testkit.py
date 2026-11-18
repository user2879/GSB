"""Deterministic helpers shared by the test suite and verification scripts.

These utilities edit sample bundles using only fixed seeds, rebuild engine
inputs from disk, and compare incremental runs against a from-scratch full
reference run field-by-field.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_generation_config, load_materials
from .engine import TextureEngine
from .loader import scan_textures
from .scripts_compat import regenerate_texture


def load_bundle(sample_dir: str | Path) -> tuple[Any, Any, list[dict], Any, list]:
    sample_dir = Path(sample_dir)
    config = load_generation_config(sample_dir / "generation_config.json")
    materials = load_materials(sample_dir / "materials.json", config)
    material_raw = json.loads((sample_dir / "materials.json").read_text(encoding="utf-8"))
    textures, decode_failures = scan_textures(sample_dir, config.active_tier)
    return config, materials, material_raw, textures, decode_failures


def make_engine(sample_dir: str | Path, snapshot_dir: str | Path, *, strict: bool = False) -> TextureEngine:
    config, materials, material_raw, textures, _ = load_bundle(sample_dir)
    return TextureEngine(config, materials, textures, snapshot_dir, material_raw, strict_resume=strict)


def materialize_full_reference(sample_dir: str | Path, snapshot_dir: str | Path) -> dict[str, Any]:
    """From-scratch full run used as the ground truth for equality comparisons."""
    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    engine = make_engine(sample_dir, snapshot_dir)
    result = engine.execute_run()
    while not result.complete and not result.stopped_by_signal:
        engine = make_engine(sample_dir, snapshot_dir)
        result = engine.execute_run()
    return engine.full_reference_run()


def apply_edit(sample_dir: str | Path, edit: dict[str, Any]) -> None:
    sample_dir = Path(sample_dir)
    op = edit["op"]
    if op == "replace_texture":
        regenerate_texture(sample_dir / f"{edit['texture']}.png",
                           size=load_generation_config(sample_dir / "generation_config.json").active_tier,
                           seed=edit["seed"])
        return
    config_path = sample_dir / "generation_config.json"
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    if op == "set_layer":
        layer = config_raw["noise_layers"][edit["layer_index"]]
        for key in ("frequency", "amplitude", "seed", "fingerprint"):
            if key in edit:
                layer[key] = edit[key]
    elif op == "set_config":
        for key, value in edit.items():
            if key == "op":
                continue
            config_raw[key] = value
    else:
        raise ValueError(f"unknown edit op: {op}")
    config_path.write_text(json.dumps(config_raw, indent=2, sort_keys=True), encoding="utf-8")


def run_incremental(sample_dir: str | Path, snapshot_dir: str | Path) -> dict[str, Any]:
    engine = make_engine(sample_dir, snapshot_dir)
    result = engine.execute_run()
    runs = 1
    while not result.complete and not result.stopped_by_signal:
        engine = make_engine(sample_dir, snapshot_dir)
        result = engine.execute_run()
        runs += 1
    return {"engine": engine, "result": result, "runs": runs}


def snapshot_arrays(snapshot_dir: str | Path) -> dict[str, np.ndarray]:
    """Collect every committed npz payload, keyed by relative path, for diffing."""
    snapshot_dir = Path(snapshot_dir)
    out: dict[str, np.ndarray] = {}
    for path in sorted(snapshot_dir.rglob("*.npz")):
        with np.load(path) as data:
            for key in data.files:
                out[f"{path.relative_to(snapshot_dir)}::{key}"] = data[key].copy()
    return out


def assert_array_bundle_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert set(left) == set(right), f"payload key mismatch: {set(left) ^ set(right)}"
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=f"payload differs: {key}")
