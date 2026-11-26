"""Render configuration model with strict validation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .codes import BizCode
from .fixedpoint import FpSpec, frac_bits_for, ALLOWED_WIDTHS, ALLOWED_TIERS


@dataclass(frozen=True)
class RenderConfig:
    image_width: int = 64
    image_height: int = 64
    tile_cols: int = 4
    tile_rows: int = 4
    text_dim: int = 12
    image_dim: int = 12
    motion_dim: int = 8
    latent_dim: int = 16
    denoise_steps: int = 6
    seed: int = 20261127
    fp_width: int = 32
    fp_tier: str = "standard"
    # fusion weights, keyed by condition name
    weights: dict = field(default_factory=lambda: {
        "image": 0.30, "motion": 0.20, "text": 0.50})
    guidance_permille: int = 700
    noise_code: int = 1
    points_per_step: int = 4
    batch_budget: int = 18
    exchange_version: int = 2
    max_workers: int = 2
    interact_budget_ms: int = 25
    # tier boundaries keyed by frame count
    small_max_frames: int = 4
    medium_max_frames: int = 12
    merge_low_impact: bool = True

    @property
    def fp_spec(self) -> FpSpec:
        return FpSpec(self.fp_width, self.fp_tier)

    @property
    def tile_width(self) -> int:
        return self.image_width // self.tile_cols

    @property
    def tile_height(self) -> int:
        return self.image_height // self.tile_rows

    def config_fingerprint_material(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RenderConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "RenderConfig":
        if not isinstance(raw, dict):
            raise ValueError(f"{BizCode.E_PARSE.value}: config root must be mapping")
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"{BizCode.E_PARSE.value}: unknown config keys={sorted(unknown)}")
        values = dict(raw)
        positive_int_fields = (
            "image_width", "image_height", "tile_cols", "tile_rows", "text_dim",
            "image_dim", "motion_dim", "latent_dim", "denoise_steps", "seed",
            "guidance_permille", "points_per_step", "batch_budget",
            "max_workers", "interact_budget_ms", "small_max_frames",
            "medium_max_frames",
        )
        for name in positive_int_fields:
            if name in values:
                value = values[name]
                if not (isinstance(value, int) and not isinstance(value, bool)
                        and value > 0):
                    raise ValueError(
                        f"{BizCode.E_POSITIVE_INT.value}: {name}={value!r}")
        if "tile_rows" in values and values["tile_rows"] <= 0:
            raise ValueError(
                f"{BizCode.E_POSITIVE_INT.value}: tile_rows")
        width = values.get("fp_width", 32)
        tier = values.get("fp_tier", "standard")
        if width not in ALLOWED_WIDTHS:
            raise ValueError(f"{BizCode.E_FP_ENUM.value}: fp_width={width!r}")
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"{BizCode.E_FP_ENUM.value}: fp_tier={tier!r}")
        frac_bits_for(width, tier)
        guidance = values.get("guidance_permille", 700)
        if not (0 <= guidance <= 1000):
            raise ValueError(
                f"{BizCode.E_POSITIVE_INT.value}: guidance_permille={guidance}")
        weights = values.get("weights")
        if weights is not None:
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"{BizCode.E_PARSE.value}: weights")
            for key, value in weights.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    raise ValueError(f"{BizCode.E_PARSE.value}: weight {key}")
        cfg = cls(**values)
        if cfg.image_width % cfg.tile_cols != 0:
            raise ValueError(f"{BizCode.E_DIM_MISMATCH.value}: image_width/tile_cols")
        if cfg.image_height % cfg.tile_rows != 0:
            raise ValueError(f"{BizCode.E_DIM_MISMATCH.value}: image_height/tile_rows")
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")
