"""Shared helpers for the workbench test suite (stdlib unittest only)."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import yaml

from workbench.config import load_config

BASE_CONFIG = {
    "version": "1.0",
    "seed": 4242,
    "width": 40,
    "height": 40,
    "active_tier": "small",
    "workers": 4,
    "latency_budget_ms": 4000.0,
    "quality_weight": 0.6,
    "mask_threshold": 90,
    "edge_threshold": 60,
    "quant_levels": [16, 32, 64],
    "paths": {
        "mask": ["threshold", "edge"],
        "inpaint": ["neighbor_mean", "directional"],
        "style": ["lut", "linear"],
    },
    "tiers": [
        {"name": "small", "sample_count": 6, "primitives": 3, "regions": 2, "instruction_length": 3},
        {"name": "medium", "sample_count": 12, "primitives": 5, "regions": 3, "instruction_length": 5},
        {"name": "large", "sample_count": 20, "primitives": 7, "regions": 4, "instruction_length": 8},
    ],
}


def make_workspace() -> tempfile.TemporaryDirectory:
    workspace = tempfile.TemporaryDirectory(prefix="wb-test-")
    return workspace


def write_config(workspace: tempfile.TemporaryDirectory, overrides: dict | None = None) -> str:
    root = Path(workspace.name)
    data = dict(BASE_CONFIG)
    data.update(
        {
            "samples_dir": str(root / "samples"),
            "outputs_dir": str(root / "outputs"),
            "snapshots_dir": str(root / "snapshots"),
        }
    )
    if overrides:
        data.update(overrides)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return str(path)


def load_test_config(workspace: tempfile.TemporaryDirectory, overrides: dict | None = None):
    return load_config(write_config(workspace, overrides))


def replace_runtime(config, **overrides):
    raw = {**config.raw, **overrides}
    return dataclasses.replace(config, raw=raw, **overrides)
