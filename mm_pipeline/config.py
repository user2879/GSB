"""Flow configuration parsing and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import PipelineError, E_CONFIG_INVALID
from .fixedpoint import QUANT_LEVELS, FRAC_BIT_WIDTHS
from .hashing import hash_json

STAGE_LOAD = "load"
STAGE_FEATURE = "feature"
STAGE_FUSION = "fusion"
STAGE_LEDGER = "ledger"
STAGE_ORDER = (STAGE_LOAD, STAGE_FEATURE, STAGE_FUSION, STAGE_LEDGER)

# Bump whenever an on-disk artifact layout changes; stale snapshots are rejected.
SNAPSHOT_FORMAT_VERSION = 3

DEFAULT_CONFIG: dict[str, Any] = {
    "visual_dim": 16,
    "text_dim": 12,
    "grid_cols": 4,
    "grid_rows": 4,
    "quant_level": 12,
    "frac_bits": 16,
    "text_weight": 0.6,
    "visual_weight": 0.8,
    "fusion_temperature": 1.0,
    "confidence_threshold": 0.08,
    "conflict_discount": 0.5,
    "supplement_boost": 1.25,
    "max_workers": 4,
    "snapshot_interval": 2,
}


@dataclass(frozen=True)
class PipelineConfig:
    visual_dim: int
    text_dim: int
    grid_cols: int
    grid_rows: int
    quant_level: int
    frac_bits: int
    text_weight: float
    visual_weight: float
    fusion_temperature: float
    confidence_threshold: float
    conflict_discount: float
    supplement_boost: float
    max_workers: int
    snapshot_interval: int

    @property
    def region_count(self) -> int:
        return self.grid_cols * self.grid_rows

    # Scheduling fields never affect conclusions. Decision-only fields
    # (threshold, discount/boost) affect the final verdict/category but must not
    # perturb feature extraction, so they are also excluded from the fingerprint
    # and the derived RNG seeds. Only genuinely semantic feature/fusion scalars
    # (dims, grid, quantization, frac bits, modality weights, temperature) remain.
    _SCHEDULING_FIELDS = ("max_workers", "snapshot_interval")
    _DECISION_FIELDS = (
        "confidence_threshold",
        "conflict_discount",
        "supplement_boost",
    )

    def semantic_dict(self) -> dict[str, Any]:
        excluded = set(self._SCHEDULING_FIELDS) | set(self._DECISION_FIELDS)
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in excluded
        }

    def fingerprint(self) -> str:
        return hash_json(self.semantic_dict(), prefix="PipelineConfig")

    def rng_seed(self, namespace: str) -> int:
        digest = hash_json(
            {"ns": namespace, "cfg": self.semantic_dict()}, prefix="rng-seed"
        )
        return int(digest[:16], 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_int(raw: dict, key: str, *, positive: bool = True) -> int:
    if key not in raw:
        raise PipelineError(
            E_CONFIG_INVALID, f"missing config field '{key}'", stage=STAGE_LOAD
        )
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(
            E_CONFIG_INVALID,
            f"config field '{key}' must be an integer, got {type(value).__name__}",
            stage=STAGE_LOAD,
        )
    if positive and value <= 0:
        raise PipelineError(
            E_CONFIG_INVALID, f"config field '{key}' must be positive, got {value}",
            stage=STAGE_LOAD,
        )
    return value


def _require_float(raw: dict, key: str) -> float:
    if key not in raw:
        raise PipelineError(
            E_CONFIG_INVALID, f"missing config field '{key}'", stage=STAGE_LOAD
        )
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(
            E_CONFIG_INVALID,
            f"config field '{key}' must be a number, got {type(value).__name__}",
            stage=STAGE_LOAD,
        )
    return float(value)


def parse_config(path: str | Path, *, overrides: Optional[dict] = None) -> PipelineConfig:
    """Load and validate the flow YAML, applying optional in-memory overrides."""
    p = Path(path)
    if not p.exists():
        raise PipelineError(
            E_CONFIG_INVALID, f"config file not found: {p}", stage=STAGE_LOAD
        )
    try:
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineError(
            E_CONFIG_INVALID, f"config YAML parse error: {exc}", stage=STAGE_LOAD
        ) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise PipelineError(
            E_CONFIG_INVALID, "config root must be a mapping", stage=STAGE_LOAD
        )
    if overrides:
        merged = dict(loaded)
        merged.update(overrides)
        loaded = merged
    return build_config(loaded)


def build_config(raw: dict) -> PipelineConfig:
    visual_dim = _require_int(raw, "visual_dim")
    text_dim = _require_int(raw, "text_dim")
    grid_cols = _require_int(raw, "grid_cols")
    grid_rows = _require_int(raw, "grid_rows")

    quant_level = _require_int(raw, "quant_level")
    if quant_level not in QUANT_LEVELS:
        raise PipelineError(
            E_CONFIG_INVALID,
            f"quant_level must be one of {list(QUANT_LEVELS)}, got {quant_level}",
            stage=STAGE_LOAD,
        )
    frac_bits = _require_int(raw, "frac_bits")
    if frac_bits not in FRAC_BIT_WIDTHS:
        raise PipelineError(
            E_CONFIG_INVALID,
            f"frac_bits must be one of {list(FRAC_BIT_WIDTHS)}, got {frac_bits}",
            stage=STAGE_LOAD,
        )
    if quant_level > frac_bits:
        raise PipelineError(
            E_CONFIG_INVALID,
            f"quant_level ({quant_level}) cannot exceed frac_bits ({frac_bits})",
            stage=STAGE_LOAD,
        )

    text_weight = _require_float(raw, "text_weight")
    visual_weight = _require_float(raw, "visual_weight")
    if not (math.isfinite(text_weight) and math.isfinite(visual_weight)):
        raise PipelineError(
            E_CONFIG_INVALID, "evidence weights must be finite", stage=STAGE_LOAD
        )
    if text_weight <= 0 or visual_weight <= 0:
        raise PipelineError(
            E_CONFIG_INVALID, "evidence weights must be positive", stage=STAGE_LOAD
        )

    temperature = _require_float(raw, "fusion_temperature")
    if not math.isfinite(temperature):
        raise PipelineError(
            E_CONFIG_INVALID, "fusion_temperature must be finite", stage=STAGE_LOAD
        )
    if temperature <= 0:
        raise PipelineError(
            E_CONFIG_INVALID, "fusion_temperature must be a positive finite number",
            stage=STAGE_LOAD,
        )

    threshold = _require_float(raw, "confidence_threshold")
    if not (0.0 <= threshold <= 1.0):
        raise PipelineError(
            E_CONFIG_INVALID,
            f"confidence_threshold must lie within [0,1], got {threshold}",
            stage=STAGE_LOAD,
        )

    discount = _require_float(raw, "conflict_discount")
    boost = _require_float(raw, "supplement_boost")
    if not (math.isfinite(discount) and math.isfinite(boost)):
        raise PipelineError(
            E_CONFIG_INVALID,
            "conflict_discount and supplement_boost must be finite",
            stage=STAGE_LOAD,
        )
    if not (0.0 <= discount <= 1.0) or boost <= 0:
        raise PipelineError(
            E_CONFIG_INVALID,
            "conflict_discount must be in [0,1] and supplement_boost positive",
            stage=STAGE_LOAD,
        )

    max_workers = _require_int(raw, "max_workers")
    snapshot_interval = _require_int(raw, "snapshot_interval")

    return PipelineConfig(
        visual_dim=visual_dim,
        text_dim=text_dim,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        quant_level=quant_level,
        frac_bits=frac_bits,
        text_weight=text_weight,
        visual_weight=visual_weight,
        fusion_temperature=temperature,
        confidence_threshold=threshold,
        conflict_discount=discount,
        supplement_boost=boost,
        max_workers=max_workers,
        snapshot_interval=snapshot_interval,
    )
