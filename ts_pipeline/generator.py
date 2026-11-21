"""Deterministic local sample/config/injection fixture generator."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, atomic_write_text, read_json
from .yamlutil import load_yaml_file


SEQUENCE_IDS = ("s-alpha", "s-beta", "s-gamma", "s-delta")


@dataclass
class Fixture:
    root: Path
    samples_path: Path
    config_path: Path
    injections_path: Path
    seed: int


def _sequence_values(index: int, count: int, seed: int) -> list[float]:
    rng = random.Random(seed + 1009 * (index + 1))
    base = 12.0 + index * 6.5
    phase = (index + 1) * 0.61
    values: list[float] = []
    for step_index in range(count):
        value = (
            base
            + 0.075 * step_index
            + 2.1 * math.sin(2.0 * math.pi * step_index / 9.0 + phase)
            + 0.65 * math.cos(2.0 * math.pi * step_index / 5.0 + phase * 0.4)
            + rng.uniform(-0.22, 0.22)
        )
        values.append(round(value, 6))
    return values


def _write_samples(path: Path, *, seed: int, count: int = 40) -> None:
    lines: list[str] = []
    missing_by_series = {
        "s-alpha": {5},
        "s-beta": {8},
        "s-gamma": {3, 17},
        "s-delta": set(),
    }
    for index, sequence_id in enumerate(SEQUENCE_IDS):
        values = _sequence_values(index, count, seed)
        missing = [position in missing_by_series[sequence_id] for position in range(count)]
        payload = {
            "sequence_id": sequence_id,
            "timestamps_ms": [
                1_700_000_000_000 + index * 10_000 + position * 1_000
                for position in range(count)
            ],
            "values": [None if missing[position] else values[position] for position in range(count)],
            "missing": missing,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    atomic_write_text(path, "\n".join(lines) + "\n")


def _config_text(raw: dict[str, Any]) -> str:
    backtest = raw["backtest"]
    models = raw["models"]
    quantization = raw["quantization"]
    scheduler = raw["scheduler"]
    snapshot = raw["snapshot"]
    enabled = ", ".join(models["enabled"])
    return (
        "schema_version: 1\n"
        f"samples: {raw['samples']}\n"
        f"injection_manifest: {raw['injection_manifest']}\n"
        "backtest:\n"
        f"  window_length: {backtest['window_length']}\n"
        f"  step: {backtest['step']}\n"
        f"  forecast_horizon: {backtest['forecast_horizon']}\n"
        f"  error_tolerance: {backtest['error_tolerance']}\n"
        "models:\n"
        f"  enabled: [{enabled}]\n"
        f"  moving_average_window: {models['moving_average_window']}\n"
        f"  feature_lag_order: {models['feature_lag_order']}\n"
        f"  autoregressive_regularization: {models['autoregressive_regularization']}\n"
        "quantization:\n"
        f"  fixed_decimal_places: {quantization['fixed_decimal_places']}\n"
        f"  quantization_level: {quantization['quantization_level']}\n"
        "scheduler:\n"
        f"  concurrency: {scheduler['concurrency']}\n"
        f"  batch_point_budget: {scheduler['batch_point_budget']}\n"
        f"  per_series_point_quota: {scheduler['per_series_point_quota']}\n"
        "snapshot:\n"
        f"  commit_interval_windows: {snapshot['commit_interval_windows']}\n"
        f"  version: {snapshot['version']}\n"
    )


def _default_raw() -> dict[str, Any]:
    return {
        "samples": "samples.jsonl",
        "injection_manifest": "injections.json",
        "backtest": {
            "window_length": 12,
            "step": 4,
            "forecast_horizon": 3,
            "error_tolerance": 0.65,
        },
        "models": {
            "enabled": ["naive_persistence", "moving_average", "autoregressive"],
            "moving_average_window": 3,
            "feature_lag_order": 3,
            "autoregressive_regularization": 1.0e-6,
        },
        "quantization": {
            "fixed_decimal_places": 3,
            "quantization_level": 1,
        },
        "scheduler": {
            "concurrency": 4,
            "batch_point_budget": 500_000,
            "per_series_point_quota": 500_000,
        },
        "snapshot": {
            "commit_interval_windows": 2,
            "version": 1,
        },
    }


def _write_config(path: Path, raw: dict[str, Any]) -> None:
    atomic_write_text(path, _config_text(raw))


def _default_injections() -> dict[str, Any]:
    return {
        "injections": [
            {
                "id": "INJ-001",
                "kind": "non_finite_observation",
                "sequence_id": "s-beta",
                "window_index": 1,
                "position": 4,
                "value": "NaN",
            },
            {
                "id": "INJ-002",
                "kind": "irregular_timestamp",
                "sequence_id": "s-gamma",
                "window_index": 2,
                "position": 5,
                "offset_ms": 17,
            },
            {
                "id": "INJ-003",
                "kind": "window_too_long",
                "sequence_id": "s-alpha",
                "window_index": 3,
                "window_length": 99,
            },
            {
                "id": "INJ-004",
                "kind": "quantization_out_of_range",
                "sequence_id": "s-delta",
                "window_index": 1,
                "model": "moving_average",
                "quantization_level": 999,
            },
        ]
    }


def generate_fixture(root: Path, *, seed: int = 20261122, force: bool = False) -> Fixture:
    root = root.resolve()
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    samples_path = root / "samples.jsonl"
    config_path = root / "config.yaml"
    injections_path = root / "injections.json"
    _write_samples(samples_path, seed=seed)
    _write_config(config_path, _default_raw())
    atomic_write_json(injections_path, _default_injections())
    return Fixture(root, samples_path, config_path, injections_path, seed)


def rewrite_config(
    path: Path,
    *,
    backtest: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    quantization: dict[str, Any] | None = None,
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = load_yaml_file(path)
    for key, updates in (
        ("backtest", backtest),
        ("models", models),
        ("quantization", quantization),
        ("scheduler", scheduler),
    ):
        if updates:
            raw[key].update(updates)
    _write_config(path, raw)
    return raw


def replace_sequence_values(samples_path: Path, sequence_id: str, *, seed: int) -> None:
    records = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sequence_index = SEQUENCE_IDS.index(sequence_id)
    replacement = _sequence_values(sequence_index, len(records[sequence_index]["values"]), seed + 777)
    for record in records:
        if record["sequence_id"] != sequence_id:
            continue
        for position, missing in enumerate(record["missing"]):
            if not missing:
                record["values"][position] = replacement[position]
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    atomic_write_text(samples_path, "\n".join(lines) + "\n")


def set_injections(path: Path, items: list[dict[str, Any]]) -> None:
    atomic_write_json(path, {"injections": items})
