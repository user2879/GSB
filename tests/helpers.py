from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fixedtex.models import MaterialConfig, RenderConfig, TextureData
from fixedtex.session import Session

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG = ROOT / "samples/config.json"
SAMPLE_CONFIG_LOW = ROOT / "samples/config_low_budget.json"
SAMPLE_MATERIAL = ROOT / "samples/material.json"
SAMPLE_TEXTURE = ROOT / "samples/base_texture.png"
SAMPLE_TEXTURE_V2 = ROOT / "samples/base_texture_v2.png"


def load_sample_config_dict(config_path: Path = SAMPLE_CONFIG) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_config(
    tmp_path: Path,
    *,
    source: Path = SAMPLE_CONFIG,
    overrides: dict[str, Any] | None = None,
    budget: dict[str, int] | None = None,
    preview_budget_ms: int | None = None,
    thread_pool_size: int | None = None,
    commit_interval_commands: int | None = None,
    error_tolerance_quant_steps: int | None = None,
    name: str | None = None,
) -> Path:
    raw = load_sample_config_dict(source)
    raw["texture"] = str(SAMPLE_TEXTURE.resolve())
    if budget is not None:
        raw["budgets"] = budget
    if preview_budget_ms is not None:
        raw["preview_budget_ms"] = preview_budget_ms
    if thread_pool_size is not None:
        raw["thread_pool_size"] = thread_pool_size
    if commit_interval_commands is not None:
        raw["commit_interval_commands"] = commit_interval_commands
    if error_tolerance_quant_steps is not None:
        raw["error_tolerance_quant_steps"] = error_tolerance_quant_steps
    if name is not None:
        raw["name"] = name
    if overrides:
        _deep_update(raw, overrides)
    safe_name = (name or "config").replace("/", "-")
    path = tmp_path / f"config-{safe_name}.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _deep_update(target: dict[str, Any], changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def write_commands(path: Path, commands: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in commands),
        encoding="utf-8",
    )
    return path


def open_session(
    tmp_path: Path,
    *,
    config_path: Path | None = None,
    material_path: Path = SAMPLE_MATERIAL,
    name: str = "state",
    resume: bool = False,
    injections: list[dict[str, Any]] | None = None,
) -> Session:
    return Session.open(
        config_path or SAMPLE_CONFIG,
        material_path,
        tmp_path / name,
        resume=resume,
        injections=injections or [],
    )


def assert_session_matches_full(
    session: Session,
    full_layers: dict[int, Any],
    full_channels: dict[str, Any],
    full_composite: Any,
) -> None:
    assert session.state.composite is not None
    assert sorted(session.state.layers) == sorted(full_layers)
    for index, reference in full_layers.items():
        actual = session.state.layers[index]
        assert actual.key == reference.key
        assert np.array_equal(actual.raw, reference.raw)
        assert np.array_equal(actual.quantized, reference.quantized)
        assert np.array_equal(actual.residual, reference.residual)
        assert np.array_equal(actual.decisions, reference.decisions)
    assert sorted(session.state.channels) == sorted(full_channels)
    for name, reference in full_channels.items():
        actual = session.state.channels[name]
        assert actual.key == reference.key
        assert actual.layer_keys == reference.layer_keys
        assert np.array_equal(actual.array, reference.array)
    assert session.state.composite.key == full_composite.key
    assert np.array_equal(session.state.composite.array, full_composite.array)
