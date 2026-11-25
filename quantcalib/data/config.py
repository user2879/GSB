"""Quantization / distillation YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from quantcalib.errors import E_CONFIG_INVALID, Issue


_VALID_ROUNDING = {"half_up", "half_even", "floor"}
_VALID_SATURATION = {"saturate", "wrap"}
_VALID_SCHEME = {"symmetric", "asymmetric"}


@dataclass
class QuantConfig:
    seed: int = 20261125
    default_bits: int = 8
    per_layer_bits: dict[str, int] = field(default_factory=dict)
    default_group_granularity: int = 16
    per_layer_group_granularity: dict[str, int] = field(default_factory=dict)
    q_lo_permille: int = 10
    q_hi_permille: int = 990
    scheme: str = "asymmetric"
    fractional_bits: int = 6
    quantization_levels: int = 256
    rounding_mode: str = "half_even"
    saturation_policy: str = "saturate"
    error_tolerance: float = 0.05
    distill_temperature: float = 2.0
    distill_steps: int = 4
    soft_label_weight: float = 0.5
    distill_learning_rate: float = 0.1
    max_workers: int = 4
    global_point_budget: int = 200000
    per_layer_quota: int = 60000
    batch_point_budget: int = 40000
    snapshot_commit_interval: int = 2
    active_receipt_version: str = "q3"

    def layer_bits(self, layer: str) -> int:
        return int(self.per_layer_bits.get(layer, self.default_bits))

    def layer_granularity(self, layer: str) -> int:
        return int(self.per_layer_group_granularity.get(
            layer, self.default_group_granularity))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> tuple[QuantConfig | None, list[Issue]]:
    issues: list[Issue] = []
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    known = QuantConfig().__dict__.keys()
    unknown = set(raw) - set(known)
    for key in sorted(unknown):
        issues.append(Issue(E_CONFIG_INVALID, f"unknown config key {key!r}"))

    clean = {key: value for key, value in raw.items() if key in known}
    cfg = QuantConfig(**clean)

    if cfg.scheme not in _VALID_SCHEME:
        issues.append(Issue(E_CONFIG_INVALID, f"invalid scheme {cfg.scheme!r}"))
    if cfg.rounding_mode not in _VALID_ROUNDING:
        issues.append(Issue(E_CONFIG_INVALID, f"invalid rounding_mode {cfg.rounding_mode!r}"))
    if cfg.saturation_policy not in _VALID_SATURATION:
        issues.append(Issue(E_CONFIG_INVALID,
                            f"invalid saturation_policy {cfg.saturation_policy!r}"))
    if not (0 <= cfg.q_lo_permille < cfg.q_hi_permille <= 1000):
        issues.append(Issue(E_CONFIG_INVALID,
                            "require 0 <= q_lo_permille < q_hi_permille <= 1000"))
    if cfg.default_bits <= 1 or cfg.quantization_levels <= 1:
        issues.append(Issue(E_CONFIG_INVALID, "bits and levels must be > 1"))
    if cfg.fractional_bits < 0:
        issues.append(Issue(E_CONFIG_INVALID, "fractional_bits must be >= 0"))
    if cfg.max_workers < 1 or cfg.global_point_budget < 1 or cfg.per_layer_quota < 1:
        issues.append(Issue(E_CONFIG_INVALID, "budgets and workers must be >= 1"))
    if cfg.distill_temperature <= 0 or cfg.distill_steps < 0:
        issues.append(Issue(E_CONFIG_INVALID, "distillation parameters are invalid"))

    return (cfg if not issues else None), issues


def write_config(path: str | Path, cfg: QuantConfig) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=True, allow_unicode=True)
