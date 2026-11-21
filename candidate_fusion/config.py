"""Fusion configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import failure
from .io_utils import read_json, sha256_json


VALID_NORMALIZATIONS = {"minmax", "zscore"}
VALID_ROUNDING = {"half_even", "half_up", "truncate"}


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    weight: float
    capacity: int


@dataclass(frozen=True)
class FusionConfig:
    raw: dict[str, Any]
    channels: dict[str, ChannelConfig]
    feature_dim: int
    normalization: str
    quant_bits: int
    rounding: str
    trunc_k: int
    max_trunc_k: int
    max_workers: int
    global_budget: int
    channel_quota: int

    @property
    def quant_step(self) -> float:
        return 2.0 ** (-self.quant_bits)

    @property
    def channel_names(self) -> list[str]:
        return sorted(self.channels)

    def weights(self) -> dict[str, float]:
        return {name: self.channels[name].weight for name in self.channel_names}

    def config_hash(self) -> str:
        return sha256_json(self.raw)

    def structural_hash(self) -> str:
        return sha256_json(
            {
                "channels": {name: self.channels[name].capacity for name in self.channel_names},
                "feature_dim": self.feature_dim,
                "schema": 1,
            }
        )

    def echo(self) -> dict[str, Any]:
        return {
            "channel_quota": self.channel_quota,
            "global_budget": self.global_budget,
            "max_workers": self.max_workers,
            "normalization": self.normalization,
            "quant_bits": self.quant_bits,
            "rounding": self.rounding,
            "trunc_k": self.trunc_k,
            "weights": self.weights(),
        }

    def normalization_parameters(self) -> dict[str, Any]:
        return {
            "normalization": self.normalization,
            "quant_bits": self.quant_bits,
            "rounding": self.rounding,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def config_echo_best_effort(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    channels = raw.get("channels")
    weights: dict[str, float] = {}
    if isinstance(channels, dict):
        for name in sorted(channels):
            item = channels[name]
            if isinstance(item, dict) and _is_finite_number(item.get("weight")):
                weights[str(name)] = float(item["weight"])
    echo: dict[str, Any] = {"weights": weights}
    for key in (
        "channel_quota",
        "global_budget",
        "max_workers",
        "normalization",
        "quant_bits",
        "rounding",
        "trunc_k",
    ):
        echo[key] = raw.get(key)
    return echo


def validate_config(
    raw: Any,
    injection_ids: dict[tuple[str, ...], str] | None = None,
) -> tuple[FusionConfig | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    injection_ids = injection_ids or {}

    def add(detail: str, channel: str | None = None, injection_id: str | None = None) -> None:
        failures.append(
            failure(
                "config",
                "E_CONFIG_INVALID",
                detail,
                channel=channel,
                injection_id=injection_id,
            )
        )

    if not isinstance(raw, dict):
        add("config root must be a JSON object")
        return None, failures

    channels_raw = raw.get("channels")
    channel_configs: dict[str, ChannelConfig] = {}
    if not isinstance(channels_raw, dict) or not channels_raw:
        add("channels must be a non-empty object")
    else:
        for name in sorted(channels_raw):
            entry = channels_raw[name]
            if not isinstance(name, str) or not name:
                add(f"invalid channel name: {name!r}")
                continue
            if not isinstance(entry, dict):
                add(f"channel '{name}' must be an object", channel=name)
                continue
            weight = entry.get("weight")
            capacity = entry.get("capacity")
            valid = True
            if not _is_finite_number(weight):
                add(f"invalid weight for channel '{name}': {weight!r}", channel=name)
                valid = False
            elif float(weight) < 0.0:
                add(f"negative weight for channel '{name}': {weight}", channel=name)
                valid = False
            if not _is_int(capacity) or int(capacity) <= 0:
                add(
                    f"invalid capacity for channel '{name}': {capacity!r}",
                    channel=name,
                    injection_id=injection_ids.get(("channel", name, "capacity")),
                )
                valid = False
            if valid:
                channel_configs[name] = ChannelConfig(
                    name=name,
                    weight=float(weight),
                    capacity=int(capacity),
                )

    feature_dim = raw.get("feature_dim")
    if not _is_int(feature_dim) or int(feature_dim) <= 0:
        add(f"feature_dim must be a positive integer: {feature_dim!r}")

    normalization = raw.get("normalization")
    if normalization not in VALID_NORMALIZATIONS:
        add(f"unsupported normalization: {normalization!r}")

    quant_bits = raw.get("quant_bits")
    if not _is_int(quant_bits) or not 1 <= int(quant_bits) <= 24:
        add(f"quant_bits must be an integer in [1, 24]: {quant_bits!r}")

    rounding = raw.get("rounding", "half_even")
    if rounding not in VALID_ROUNDING:
        add(f"unsupported rounding mode: {rounding!r}")

    max_trunc_k = raw.get("max_trunc_k")
    if not _is_int(max_trunc_k) or int(max_trunc_k) < 1:
        add(f"max_trunc_k must be a positive integer: {max_trunc_k!r}")

    trunc_k = raw.get("trunc_k")
    if not _is_int(trunc_k):
        add(f"trunc_k must be an integer: {trunc_k!r}")
    elif _is_int(max_trunc_k) and not 0 <= int(trunc_k) <= int(max_trunc_k):
        add(f"trunc_k out of range: {trunc_k}", injection_id=injection_ids.get(("config", "trunc_k")))

    max_workers = raw.get("max_workers")
    if not _is_int(max_workers) or int(max_workers) <= 0:
        add(f"max_workers must be a positive integer: {max_workers!r}")

    global_budget = raw.get("global_budget")
    if not _is_int(global_budget) or int(global_budget) < 0:
        add(f"global_budget must be a non-negative integer: {global_budget!r}")

    channel_quota = raw.get("channel_quota")
    if not _is_int(channel_quota) or int(channel_quota) < 0:
        add(f"channel_quota must be a non-negative integer: {channel_quota!r}")

    if failures:
        return None, failures

    config = FusionConfig(
        raw=raw,
        channels=channel_configs,
        feature_dim=int(feature_dim),
        normalization=str(normalization),
        quant_bits=int(quant_bits),
        rounding=str(rounding),
        trunc_k=int(trunc_k),
        max_trunc_k=int(max_trunc_k),
        max_workers=int(max_workers),
        global_budget=int(global_budget),
        channel_quota=int(channel_quota),
    )
    return config, []


def load_raw_config(path: Path) -> tuple[Any | None, list[dict[str, Any]]]:
    try:
        return read_json(path), []
    except FileNotFoundError:
        return None, [failure("config", "E_CONFIG_INVALID", f"config file not found: {path}")]
    except Exception as exc:  # JSONDecodeError and OSError share the failure path
        return None, [
            failure("config", "E_CONFIG_INVALID", f"cannot read config JSON: {exc}")
        ]
