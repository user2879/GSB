"""Configuration defaults, validation and hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import AppError, E_CONFIG_INVALID
from .util import finite_number, is_int, read_json, sha256_json


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "binarization_threshold": 160,
    "projection_window": 3,
    "min_char_height": 22,
    "max_char_height": 90,
    "split_penalty": 8,
    "split_valley_weight": 3,
    "match_metric": "overlap",
    "fusion_weights": [1, 1, 1],
    "fixed_point_bits": 12,
    "confidence_quantization_steps": 100,
    "ma_point_budget": 2_000_000,
    "thread_pool_workers": 4,
    "snapshot_interval": 1,
    "injection_manifest": "injections.json",
    "expected_width": 600,
    "expected_height": 720,
}


_INT_FIELDS = {
    "schema_version": (1, 1),
    "binarization_threshold": (1, 254),
    "projection_window": (1, 31),
    "min_char_height": (2, 512),
    "max_char_height": (3, 2048),
    "split_penalty": (0, 1_000_000),
    "split_valley_weight": (0, 1_000_000),
    "fixed_point_bits": (4, 20),
    "confidence_quantization_steps": (1, 1_000_000),
    "ma_point_budget": (0, 2**63 - 1),
    "thread_pool_workers": (1, 64),
    "snapshot_interval": (1, 1_000_000),
    "expected_width": (1, 100_000),
    "expected_height": (1, 100_000),
}


def normalize_config(raw: dict[str, Any], variant_count: int = 3) -> dict[str, Any]:
    """Return a validated, canonical configuration.

    Unknown fields are rejected so misspelled tuning parameters cannot silently
    alter cache semantics.
    """

    if not isinstance(raw, dict):
        raise AppError(E_CONFIG_INVALID, "configuration must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise AppError(E_CONFIG_INVALID, f"unknown configuration fields: {', '.join(unknown)}")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw)

    for field, (low, high) in _INT_FIELDS.items():
        value = cfg[field]
        if not is_int(value) or not (low <= value <= high):
            raise AppError(
                E_CONFIG_INVALID,
                f"{field} must be an integer in [{low}, {high}]",
                {"field": field, "value": repr(value)},
            )

    if cfg["min_char_height"] >= cfg["max_char_height"]:
        raise AppError(E_CONFIG_INVALID, "min_char_height must be below max_char_height")
    if cfg["match_metric"] not in {"overlap", "l1"}:
        raise AppError(E_CONFIG_INVALID, "match_metric must be 'overlap' or 'l1'")
    if not isinstance(cfg["fusion_weights"], list) or len(cfg["fusion_weights"]) != variant_count:
        raise AppError(
            E_CONFIG_INVALID,
            f"fusion_weights must contain exactly {variant_count} finite positive values",
        )
    for index, weight in enumerate(cfg["fusion_weights"]):
        if not finite_number(weight) or float(weight) <= 0.0 or float(weight) > 1_000_000.0:
            raise AppError(
                E_CONFIG_INVALID,
                "fusion_weights entries must be finite and positive",
                {"index": index, "value": repr(weight)},
            )
        if float(weight) <= 0.0:
            raise AppError(E_CONFIG_INVALID, "fusion weights must be positive")
    if not isinstance(cfg["injection_manifest"], str) or not cfg["injection_manifest"].strip():
        raise AppError(E_CONFIG_INVALID, "injection_manifest must be a non-empty relative path")
    injection_path = Path(cfg["injection_manifest"])
    if injection_path.is_absolute() or ".." in injection_path.parts:
        raise AppError(E_CONFIG_INVALID, "injection_manifest must stay inside the sample directory")
    return cfg


def load_config(path: Path, variant_count: int = 3) -> dict[str, Any]:
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        raise AppError(E_CONFIG_INVALID, f"cannot read config: {exc}") from exc
    return normalize_config(raw, variant_count=variant_count)


def config_hash(config: dict[str, Any]) -> str:
    return sha256_json(config)


def stage_param_fingerprint(stage: str, config: dict[str, Any]) -> str:
    """Fingerprint only parameters that can influence a given stage."""

    fields = {
        "columns": ("binarization_threshold", "projection_window"),
        "split": (
            "binarization_threshold",
            "projection_window",
            "min_char_height",
            "max_char_height",
        ),
        "border": (
            "min_char_height",
            "max_char_height",
            "split_penalty",
            "split_valley_weight",
            "projection_window",
        ),
        "match": (
            "match_metric",
            "fusion_weights",
            "fixed_point_bits",
            "confidence_quantization_steps",
        ),
    }
    selected = {key: config[key] for key in fields[stage]}
    return sha256_json({"stage": stage, "parameters": selected})
