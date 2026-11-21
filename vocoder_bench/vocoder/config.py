"""Vocoder configuration model and edit-command application."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import yaml_lite

WINDOW_TYPES = ("hann", "hamming", "rect")
NORMALIZATION_RULES = ("cola", "none")
ROUNDING_MODES = ("half_up", "half_even", "trunc")

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "frame_shift_ms": 10.0,
    "frame_length_ms": 20.0,
    "harmonic_limit": 12,
    "phase_bits": 24,
    "phase_quant_step": 1,
    "f0_ratio": 1.0,
    "energy_rounding": "half_up",
    "window_type": "hann",
    "normalization_rule": "cola",
    "max_workers": 2,
    "snapshot_interval": 8,
    "align_search_ms": 12.0,
    "interactive_budget_ms": 50.0,
    "spectral_tol": 0.25,
    "rmse_tol": 0.12,
    "ncc_tol": 0.90,
    "phase_attribution_threshold": 1.5,
}

# Which config field each editable command changes.
EDITABLE_FIELDS = {
    "f0_ratio": "f0_ratio",
    "harmonic_limit": "harmonic_limit",
    "phase_bits": "phase_bits",
    "phase_quant_step": "phase_quant_step",
    "window_type": "window_type",
    "energy_rounding": "energy_rounding",
}


def load_config(path: str | Path) -> dict:
    raw = yaml_lite.load(path)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["sample_rate"] != 16000:
        raise ValueError("sample_rate must be 16000")
    if not (0 < cfg["frame_shift_ms"] <= cfg["frame_length_ms"]):
        raise ValueError("frame_length_ms must be >= frame_shift_ms > 0")
    if cfg["harmonic_limit"] < 1:
        raise ValueError("harmonic_limit must be >= 1")
    if not (4 <= cfg["phase_bits"] <= 48):
        raise ValueError("phase_bits must be within [4, 48]")
    if cfg["phase_quant_step"] < 1:
        raise ValueError("phase_quant_step must be >= 1")
    if cfg["window_type"] not in WINDOW_TYPES:
        raise ValueError(f"window_type must be one of {WINDOW_TYPES}")
    if cfg["normalization_rule"] not in NORMALIZATION_RULES:
        raise ValueError(f"normalization_rule must be one of {NORMALIZATION_RULES}")
    if cfg["energy_rounding"] not in ROUNDING_MODES:
        raise ValueError(f"energy_rounding must be one of {ROUNDING_MODES}")
    if cfg["max_workers"] < 1:
        raise ValueError("max_workers must be >= 1")
    if cfg["align_search_ms"] < 0:
        raise ValueError("align_search_ms must be >= 0")
    if cfg["f0_ratio"] <= 0:
        raise ValueError("f0_ratio must be positive")


def config_fingerprint(cfg: dict) -> str:
    from .hashing import canonical_hash

    return canonical_hash(cfg)


def phase_profile(cfg: dict) -> dict:
    return {
        "phase_bits": cfg["phase_bits"],
        "phase_quant_step": cfg["phase_quant_step"],
    }


def apply_edit(cfg: dict, command: dict) -> set[str]:
    """Apply one edit command in place; return the set of changed field names."""
    kind = command["type"]
    if kind not in EDITABLE_FIELDS:
        raise ValueError(f"unknown edit type: {kind}")
    field = EDITABLE_FIELDS[kind]
    new_value = command["value"]
    changed = set()
    if cfg[field] != new_value:
        cfg[field] = new_value
        changed.add(field)
    validate_config(cfg)
    return changed
