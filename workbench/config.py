"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import ConfigError
from .utils import digest_obj

STAGE_NAMES = [
    "parse",
    "mask",
    "inpaint",
    "recolor",
    "style",
    "composite",
]

# Fields that may change between runs without invalidating snapshot resume:
# they describe *which* portion of the data to process, not pipeline physics.
RUNTIME_FIELDS = {"active_tier", "quant_override", "run_mode", "workers_override"}
# Local directories describe where data lives, not pipeline physics; keeping
# them out of the fingerprint lets isolated verification stores reuse the same
# algorithm identity while snapshot resume still checks all real parameters.
PATH_FIELDS = {"samples_dir", "outputs_dir", "snapshots_dir"}


@dataclass(frozen=True)
class Tier:
    name: str
    sample_count: int
    primitives: int
    regions: int
    instruction_length: int


@dataclass(frozen=True)
class Config:
    version: str
    seed: int
    width: int
    height: int
    active_tier: str
    workers: int
    workers_override: int | None
    latency_budget_ms: float
    quality_weight: float
    mask_threshold: int
    edge_threshold: int
    quant_levels: tuple
    quant_override: int | None
    tiers: tuple
    paths: dict
    samples_dir: str
    outputs_dir: str
    snapshots_dir: str
    raw: dict

    @property
    def resolution(self):
        return self.width, self.height

    def tier(self, name: str | None = None) -> Tier:
        wanted = name or self.active_tier
        for tier in self.tiers:
            if tier.name == wanted:
                return tier
        raise ConfigError(f"unknown tier '{wanted}'", {"active_tier": wanted})

    def effective_workers(self) -> int:
        return self.workers_override or self.workers

    def effective_quant(self) -> int:
        return self.quant_override or self.quant_levels[0]

    def persistent_fingerprint(self) -> str:
        """Hash of everything that must stay stable for snapshot resume."""
        persistent = {
            key: value
            for key, value in self.raw.items()
            if key not in RUNTIME_FIELDS and key not in PATH_FIELDS
        }
        return digest_obj(persistent)

    def full_fingerprint(self) -> str:
        return digest_obj(self.raw)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, "rb") as handle:
            raw = yaml.safe_load(handle.read().decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return _build_config(raw)


def _require_int(raw, key, minimum=None):
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer", {"got": repr(value)})
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{key}' must be >= {minimum}", {"got": value})
    return value


def _require_float(raw, key):
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number", {"got": repr(value)})
    import math

    if not math.isfinite(value):
        raise ConfigError(f"'{key}' must be finite", {"got": value})
    return float(value)


def _build_config(raw: dict) -> Config:
    version = raw.get("version", "1.0")
    if not isinstance(version, str):
        raise ConfigError("'version' must be a string")

    seed = _require_int(raw, "seed", minimum=0)
    width = _require_int(raw, "width", minimum=8)
    height = _require_int(raw, "height", minimum=8)
    workers = _require_int(raw, "workers", minimum=1)
    latency_budget_ms = _require_float(raw, "latency_budget_ms")
    if latency_budget_ms <= 0:
        raise ConfigError("'latency_budget_ms' must be positive", {"got": latency_budget_ms})
    quality_weight = _require_float(raw, "quality_weight")
    if not 0.0 <= quality_weight <= 1.0:
        raise ConfigError("'quality_weight' must be within [0, 1]", {"got": quality_weight})

    mask_threshold = _require_int(raw, "mask_threshold", minimum=1)
    edge_threshold = _require_int(raw, "edge_threshold", minimum=1)

    quant_levels = raw.get("quant_levels")
    if (
        not isinstance(quant_levels, list)
        or not quant_levels
        or any(not isinstance(q, int) or isinstance(q, bool) or q < 2 or q > 256 for q in quant_levels)
    ):
        raise ConfigError("'quant_levels' must be a non-empty list of ints in [2, 256]")
    if len(set(quant_levels)) != len(quant_levels):
        raise ConfigError("'quant_levels' entries must be unique")

    quant_override = raw.get("quant_override")
    if quant_override is not None:
        if not isinstance(quant_override, int) or quant_override not in quant_levels:
            raise ConfigError("'quant_override' must be one of quant_levels")

    workers_override = raw.get("workers_override")
    if workers_override is not None:
        if not isinstance(workers_override, int) or isinstance(workers_override, bool) or workers_override < 1:
            raise ConfigError("'workers_override' must be a positive integer")

    tiers_raw = raw.get("tiers")
    if not isinstance(tiers_raw, list) or len(tiers_raw) < 2:
        raise ConfigError("'tiers' must be a list of at least two tiers")
    tiers = []
    seen_names = set()
    prev = None
    for entry in tiers_raw:
        if not isinstance(entry, dict):
            raise ConfigError("each tier must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ConfigError("tier name must be a unique non-empty string")
        seen_names.add(name)
        tier = Tier(
            name=name,
            sample_count=_require_int(entry, "sample_count", minimum=1),
            primitives=_require_int(entry, "primitives", minimum=1),
            regions=_require_int(entry, "regions", minimum=1),
            instruction_length=_require_int(entry, "instruction_length", minimum=1),
        )
        if prev is not None:
            for field_name in ("sample_count", "primitives", "regions", "instruction_length"):
                if getattr(tier, field_name) <= getattr(prev, field_name):
                    raise ConfigError(
                        f"tier '{name}' field '{field_name}' must be strictly greater than previous tier",
                        {"previous": getattr(prev, field_name), "got": getattr(tier, field_name)},
                    )
        prev = tier
        tiers.append(tier)

    active_tier = raw.get("active_tier")
    if not isinstance(active_tier, str) or active_tier not in seen_names:
        raise ConfigError("'active_tier' must reference a defined tier")

    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("'paths' must be a mapping")

    for dir_key in ("samples_dir", "outputs_dir", "snapshots_dir"):
        if not isinstance(raw.get(dir_key), str) or not raw[dir_key]:
            raise ConfigError(f"'{dir_key}' must be a non-empty string")

    return Config(
        version=version,
        seed=seed,
        width=width,
        height=height,
        active_tier=active_tier,
        workers=workers,
        workers_override=workers_override,
        latency_budget_ms=latency_budget_ms,
        quality_weight=quality_weight,
        mask_threshold=mask_threshold,
        edge_threshold=edge_threshold,
        quant_levels=tuple(quant_levels),
        quant_override=quant_override,
        tiers=tuple(tiers),
        paths=paths,
        samples_dir=raw["samples_dir"],
        outputs_dir=raw["outputs_dir"],
        snapshots_dir=raw["snapshots_dir"],
        raw=raw,
    )
