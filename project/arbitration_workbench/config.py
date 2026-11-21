"""Arbitration configuration validation."""

from __future__ import annotations

from typing import Any

from .constants import ERROR_JSON_INVALID, VERSIONS
from .errors import ConfigValidationError
from .model import ArbitrationConfig
from .numeric import ZERO, as_decimal, fingerprint


FUSION_METHODS = ("sum", "max", "mean", "min", "top2_sum")


def parse_config(raw: Any) -> ArbitrationConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError(ERROR_JSON_INVALID, "config must be a JSON object")

    def integer(name: str, *, minimum: int, default: int | None = None) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigValidationError(ERROR_JSON_INVALID, f"{name} must be an integer >= {minimum}")
        return value

    try:
        decimals = integer("fixed_decimal_places", minimum=0)
        if decimals > 9:
            raise ConfigValidationError(ERROR_JSON_INVALID, "fixed_decimal_places must be <= 9")
        quantum = integer("quantization_units", minimum=1)
        half_life = integer("decay_half_life", minimum=1)
        budget = integer("interaction_budget_ms", minimum=0)
        workers = integer("worker_threads", minimum=1)
        if workers > 256:
            raise ConfigValidationError(ERROR_JSON_INVALID, "worker_threads must be <= 256")
        interval = integer("snapshot_commit_interval", minimum=1)
        exclusion = as_decimal(raw.get("exclusion_strength"), field="exclusion_strength")
        support_bonus = as_decimal(raw.get("support_bonus", "0.05"), field="support_bonus")
        threshold = as_decimal(raw.get("decision_threshold"), field="decision_threshold")
    except ValueError as exc:
        raise ConfigValidationError(ERROR_JSON_INVALID, str(exc)) from exc

    if exclusion < ZERO or support_bonus < ZERO:
        raise ConfigValidationError(ERROR_JSON_INVALID, "exclusion_strength and support_bonus must be non-negative")
    if threshold < ZERO:
        raise ConfigValidationError(ERROR_JSON_INVALID, "decision_threshold must be non-negative")
    fusion = str(raw.get("fusion", "sum")).lower()
    if fusion not in FUSION_METHODS:
        raise ConfigValidationError(ERROR_JSON_INVALID, f"fusion must be one of {FUSION_METHODS}")
    target_version = str(raw.get("target_version", "v2")).lower()
    if target_version not in VERSIONS:
        raise ConfigValidationError(ERROR_JSON_INVALID, "target_version must be v1, v2 or v3")
    snapshot_dir = raw.get("snapshot_dir", "snapshots")
    temp_dir = raw.get("temp_dir", "tmp")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise ConfigValidationError(ERROR_JSON_INVALID, "snapshot_dir must be a non-empty string")
    if not isinstance(temp_dir, str) or not temp_dir:
        raise ConfigValidationError(ERROR_JSON_INVALID, "temp_dir must be a non-empty string")
    equality = raw.get("threshold_equality_upgrades", True)
    if not isinstance(equality, bool):
        raise ConfigValidationError(ERROR_JSON_INVALID, "threshold_equality_upgrades must be boolean")

    return ArbitrationConfig(
        fixed_decimal_places=decimals,
        quantization_units=quantum,
        decay_half_life=half_life,
        exclusion_strength=exclusion,
        support_bonus=support_bonus,
        fusion=fusion,
        decision_threshold=threshold,
        interaction_budget_ms=budget,
        worker_threads=workers,
        snapshot_commit_interval=interval,
        snapshot_dir=snapshot_dir,
        temp_dir=temp_dir,
        target_version=target_version,
        threshold_equality_upgrades=equality,
    )


def config_fingerprint(config: ArbitrationConfig) -> str:
    return fingerprint(config.as_dict())

