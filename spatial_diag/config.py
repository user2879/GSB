"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_hash, json_safe


DEFAULT_CONFIG = Path(__file__).resolve().parent / "assets" / "default_config.yaml"


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    source_path: Path | None = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def tiers(self) -> list[dict[str, Any]]:
        return list(self.data["tiers"])

    def tier(self, name: str) -> dict[str, Any]:
        for tier in self.tiers:
            if tier["name"] == name:
                return dict(tier)
        raise KeyError(f"unknown tier: {name}")

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.data)

    @property
    def public(self) -> dict[str, Any]:
        return json_safe(self.data)

    def audio(self) -> dict[str, Any]:
        return dict(self.data["audio"])

    def room(self) -> dict[str, Any]:
        return dict(self.data["room"])

    def sphere(self) -> dict[str, Any]:
        return dict(self.data["sphere"])

    def projection(self) -> dict[str, Any]:
        return dict(self.data["projection"])

    def validation(self) -> dict[str, Any]:
        return dict(self.data["validation"])

    def consistency(self) -> dict[str, Any]:
        return dict(self.data["consistency"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    data = json_safe(raw)
    _require(data.get("schema") == "spatial-diagnostic-config/1", "unsupported config schema")
    audio = data["audio"]
    _require(int(audio["sample_rate"]) == 48000, "this workbench requires 48 kHz")
    _require(int(audio["bit_depth"]) == 16, "this workbench requires 16-bit PCM")
    _require(int(audio["channels"]) == 4, "this workbench requires four channels")
    _require(int(audio["rir_samples"]) >= 256, "rir_samples is too small")

    room = data["room"]
    dimensions = room["dimensions_m"]
    listener = room["listener_m"]
    _require(len(dimensions) == 3 and all(float(v) > 0 for v in dimensions), "invalid room dimensions")
    _require(len(listener) == 3, "invalid listener position")
    _require(all(0.0 < float(v) < float(dimensions[i]) for i, v in enumerate(listener)), "listener outside room")
    _require(len(room["wall_absorption"]) == 6, "six wall absorption coefficients are required")
    _require(all(0.0 <= float(v) < 1.0 for v in room["wall_absorption"]), "invalid absorption coefficient")
    _require(int(room["reflection_order"]) >= 0, "reflection_order must be non-negative")

    tiers = data["tiers"]
    _require([int(t["source_positions"]) for t in tiers] == [8, 24, 96], "tiers must be 8/24/96 sources")
    _require([int(t["sphere_points"]) for t in tiers] == [32, 98, 200], "tiers must use 32/98/200 points")
    for tier in tiers:
        _require(int(tier["requested_order"]) >= 0, "invalid requested order")
        _require(int(tier["memory_budget_bytes"]) > 0, "invalid memory budget")
        _require(float(tier["direction_resolution_budget_deg"]) > 0, "invalid resolution budget")

    _require(int(data["projection"]["min_effective_samples"]) >= 1, "invalid confidence threshold")
    _require(float(data["projection"]["oversampling_factor"]) > 1.0, "oversampling_factor must be > 1")
    _require(float(data["validation"]["relative_tolerance"]) > 0, "invalid validation tolerance")
    _require(float(data["consistency"]["relative_tolerance"]) > 0, "invalid consistency tolerance")
    return data


def load_config(path: str | Path | None = None) -> Config:
    source = Path(path) if path is not None else DEFAULT_CONFIG
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _require(isinstance(raw, dict), "configuration root must be a mapping")
    return Config(validate_config(raw), source_path=source.resolve())

