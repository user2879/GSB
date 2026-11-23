from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .canonical import normalize_json
from .errors import ConfigError
from .fixed import ALLOWED_FRACTIONAL_BITS, ALLOWED_TIERS, TIER_BITS
from .yaml_lite import load_yaml_file


STAGES: tuple[str, ...] = (
    "feature_quantization",
    "shared_hidden",
    "ctr_tower",
    "cvr_tower",
    "fusion",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("E_CONFIG_INVALID", field, "must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError("E_CONFIG_INVALID", field, "must be an array")
    return value


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    binning: str | None = None
    bins: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "dense":
            result["binning"] = self.binning
            result["bins"] = self.bins
        return result


@dataclass(frozen=True)
class CrossSpec:
    name: str
    left: str
    right: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "left": self.left, "right": self.right, "kind": "cross"}


@dataclass(frozen=True)
class FusionConfig:
    mode: str
    ctr_weight: float = 1.0
    cvr_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "ctr_weight": self.ctr_weight, "cvr_weight": self.cvr_weight}


@dataclass(frozen=True)
class FeatureConfig:
    seed: int
    features: tuple[FeatureSpec, ...]
    crosses: tuple[CrossSpec, ...]
    shared_features: tuple[str, ...]
    tower_features: dict[str, tuple[str, ...]]
    fusion: FusionConfig
    quantization_tier: str
    fixed_fractional_bits: int
    hidden_dim: int
    raw: dict[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted([item.name for item in self.features] + [item.name for item in self.crosses]))

    @property
    def dense_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(item for item in self.features if item.kind == "dense")

    @property
    def sparse_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(item for item in self.features if item.kind == "sparse")

    @property
    def cross_map(self) -> dict[str, CrossSpec]:
        return {item.name: item for item in self.crosses}

    def fingerprint_payload(self) -> dict[str, Any]:
        return normalize_json(self.raw)


@dataclass(frozen=True)
class Tolerances:
    values: dict[str, float]

    def get(self, stage: str) -> float:
        return self.values[stage]

    def to_dict(self) -> dict[str, float]:
        return {stage: self.values[stage] for stage in STAGES}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    learning_rate: float
    batch_size: int
    epochs: int
    positive_weight: float
    negative_weight: float
    conversion_weight: float
    conversion_delay_cutoff_ms: int
    global_train_sample_budget: int
    per_batch_sample_quota: int
    thread_pool_workers: int
    snapshot_commit_interval_batches: int
    initial_event_time_ms: int
    tolerances: Tolerances
    raw: dict[str, Any]

    def fingerprint_payload(self) -> dict[str, Any]:
        return normalize_json(self.raw)


def _parse_feature_config(raw: Any) -> FeatureConfig:
    root = _require_mapping(raw, "feature_config")
    seed = root.get("seed")
    if not _is_int(seed):
        raise ConfigError("E_CONFIG_INVALID", "seed", "must be an integer")

    features_raw = _require_list(root.get("features"), "features")
    if not features_raw:
        raise ConfigError("E_CONFIG_INVALID", "features", "must not be empty")
    features: list[FeatureSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(features_raw):
        obj = _require_mapping(item, f"features[{index}]")
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError("E_CONFIG_INVALID", f"features[{index}].name", "must be non-empty")
        if name in seen:
            raise ConfigError("E_CONFIG_INVALID", f"features[{index}].name", "duplicate feature name")
        kind = obj.get("kind")
        if kind not in ("dense", "sparse"):
            raise ConfigError("E_CONFIG_INVALID", f"features[{index}].kind", "must be dense or sparse")
        if kind == "dense":
            binning = obj.get("binning")
            if binning not in ("equal_width", "equal_frequency"):
                raise ConfigError(
                    "E_CONFIG_INVALID",
                    f"features[{index}].binning",
                    "must be equal_width or equal_frequency",
                )
            bins = obj.get("bins")
            if not _is_int(bins) or bins <= 0:
                raise ConfigError("E_CONFIG_INVALID", f"features[{index}].bins", "must be positive integer")
            features.append(FeatureSpec(name=name, kind=kind, binning=binning, bins=bins))
        else:
            features.append(FeatureSpec(name=name, kind=kind))
        seen.add(name)

    crosses_raw = root.get("crosses", [])
    crosses: list[CrossSpec] = []
    for index, item in enumerate(_require_list(crosses_raw, "crosses")):
        obj = _require_mapping(item, f"crosses[{index}]")
        name = obj.get("name")
        left = obj.get("left")
        right = obj.get("right")
        if not isinstance(name, str) or not name:
            raise ConfigError("E_CONFIG_INVALID", f"crosses[{index}].name", "must be non-empty")
        if name in seen:
            raise ConfigError("E_CONFIG_INVALID", f"crosses[{index}].name", "duplicate feature name")
        if left not in seen or right not in seen:
            raise ConfigError(
                "E_CONFIG_INVALID",
                f"crosses[{index}]",
                "left/right must reference existing base features",
            )
        if left == right:
            raise ConfigError("E_CONFIG_INVALID", f"crosses[{index}]", "cross operands must differ")
        crosses.append(CrossSpec(name=name, left=str(left), right=str(right)))
        seen.add(name)
    valid_names = set(seen)

    shared_raw = _require_list(root.get("shared_features"), "shared_features")
    if not shared_raw:
        raise ConfigError("E_CONFIG_INVALID", "shared_features", "must not be empty")
    shared_features: list[str] = []
    for index, name in enumerate(shared_raw):
        if not isinstance(name, str) or name not in valid_names:
            raise ConfigError(
                "E_CONFIG_INVALID",
                f"shared_features[{index}]",
                "must reference a configured feature",
            )
        if name not in shared_features:
            shared_features.append(name)

    tower_raw = _require_mapping(root.get("tower_features"), "tower_features")
    tower_features: dict[str, tuple[str, ...]] = {}
    for tower in ("ctr", "cvr"):
        names = _require_list(tower_raw.get(tower), f"tower_features.{tower}")
        if not names:
            raise ConfigError("E_CONFIG_INVALID", f"tower_features.{tower}", "must not be empty")
        tower_list: list[str] = []
        for index, name in enumerate(names):
            if not isinstance(name, str) or name not in valid_names:
                raise ConfigError(
                    "E_CONFIG_INVALID",
                    f"tower_features.{tower}[{index}]",
                    "must reference a configured feature",
                )
            if name not in tower_list:
                tower_list.append(name)
        tower_features[tower] = tuple(sorted(tower_list))

    quantization = _require_mapping(root.get("quantization"), "quantization")
    tier = quantization.get("tier")
    bits = quantization.get("fractional_bits")
    if tier not in ALLOWED_TIERS:
        raise ConfigError("E_CONFIG_INVALID", "quantization.tier", f"must be one of {ALLOWED_TIERS}")
    if not _is_int(bits) or bits not in ALLOWED_FRACTIONAL_BITS:
        raise ConfigError(
            "E_CONFIG_INVALID",
            "quantization.fractional_bits",
            f"must be one of {ALLOWED_FRACTIONAL_BITS}",
        )
    if bits > TIER_BITS[tier] - 2:
        raise ConfigError(
            "E_CONFIG_INVALID",
            "quantization",
            "fractional_bits leaves fewer than one sign bit and one integer bit in the tier",
        )

    fusion_raw = _require_mapping(root.get("fusion"), "fusion")
    mode = fusion_raw.get("mode")
    if mode not in ("product", "weighted_sum"):
        raise ConfigError("E_CONFIG_INVALID", "fusion.mode", "must be product or weighted_sum")
    weights = []
    for name in ("ctr_weight", "cvr_weight"):
        value = fusion_raw.get(name, 1.0)
        if not _is_number(value) or float(value) < 0:
            raise ConfigError("E_CONFIG_INVALID", f"fusion.{name}", "must be finite non-negative")
        weights.append(float(value))
    if mode == "weighted_sum" and weights[0] + weights[1] <= 0:
        raise ConfigError("E_CONFIG_INVALID", "fusion", "weighted_sum weights must not both be zero")
    fusion = FusionConfig(mode=mode, ctr_weight=weights[0], cvr_weight=weights[1])

    model_raw = _require_mapping(root.get("model", {}), "model")
    hidden_dim = model_raw.get("hidden_dim", 4)
    if not _is_int(hidden_dim) or hidden_dim <= 0 or hidden_dim > 32:
        raise ConfigError("E_CONFIG_INVALID", "model.hidden_dim", "must be an integer in [1, 32]")

    normalized = normalize_json(root)
    return FeatureConfig(
        seed=seed,
        features=tuple(features),
        crosses=tuple(crosses),
        shared_features=tuple(sorted(shared_features)),
        tower_features=tower_features,
        fusion=fusion,
        quantization_tier=str(tier),
        fixed_fractional_bits=bits,
        hidden_dim=hidden_dim,
        raw=normalized,
    )


def load_feature_config(path: str) -> FeatureConfig:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("E_CONFIG_INVALID", "feature_config", f"cannot load JSON: {exc}") from exc
    return _parse_feature_config(raw)


def feature_config_from_dict(raw: Any) -> FeatureConfig:
    return _parse_feature_config(raw)


def _parse_tolerances(raw: Any) -> Tolerances:
    if raw is None:
        raw = {}
    if isinstance(raw, dict) and "default" in raw:
        default = raw["default"]
        if not _is_number(default) or float(default) <= 0:
            raise ConfigError("E_CONFIG_INVALID", "tolerances.default", "must be positive finite")
        values = {stage: float(raw.get(stage, default)) for stage in STAGES}
    elif isinstance(raw, dict):
        values = {}
        for stage in STAGES:
            value = raw.get(stage, 0.10)
            if not _is_number(value) or float(value) <= 0:
                raise ConfigError("E_CONFIG_INVALID", f"tolerances.{stage}", "must be positive finite")
            values[stage] = float(value)
    elif _is_number(raw) and float(raw) > 0:
        values = {stage: float(raw) for stage in STAGES}
    else:
        raise ConfigError("E_CONFIG_INVALID", "tolerances", "must be a positive number or object")
    return Tolerances(values=values)


def _required_number(root: dict[str, Any], name: str) -> float:
    value = root.get(name)
    if not _is_number(value):
        raise ConfigError("E_CONFIG_INVALID", name, "must be a finite number")
    return float(value)


def _required_int(root: dict[str, Any], name: str, minimum: int = 1) -> int:
    value = root.get(name)
    if not _is_int(value) or value < minimum:
        raise ConfigError("E_CONFIG_INVALID", name, f"must be an integer >= {minimum}")
    return value


def _parse_training_config(raw: Any) -> TrainingConfig:
    root = _require_mapping(raw, "train_config")
    seed = _required_int(root, "seed", 0)
    learning_rate = _required_number(root, "learning_rate")
    if learning_rate <= 0:
        raise ConfigError("E_CONFIG_INVALID", "learning_rate", "must be positive")
    batch_size = _required_int(root, "batch_size")
    epochs = _required_int(root, "epochs")
    positive_weight = _required_number(root, "positive_weight")
    negative_weight = _required_number(root, "negative_weight")
    conversion_weight = _required_number(root, "conversion_weight")
    for name, value in (
        ("positive_weight", positive_weight),
        ("negative_weight", negative_weight),
        ("conversion_weight", conversion_weight),
    ):
        if value <= 0:
            raise ConfigError("E_CONFIG_INVALID", name, "must be positive finite")
    cutoff = _required_int(root, "conversion_delay_cutoff_ms", 0)
    budget = _required_int(root, "global_train_sample_budget", 1)
    quota = _required_int(root, "per_batch_sample_quota", 1)
    workers = _required_int(root, "thread_pool_workers", 1)
    interval = _required_int(root, "snapshot_commit_interval_batches", 1)
    initial_time = root.get("initial_event_time_ms", 0)
    if not _is_int(initial_time) or initial_time < 0:
        raise ConfigError("E_CONFIG_INVALID", "initial_event_time_ms", "must be non-negative integer")
    tolerances = _parse_tolerances(root.get("tolerances", {"default": 0.10}))
    normalized = normalize_json(root)
    return TrainingConfig(
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        conversion_weight=conversion_weight,
        conversion_delay_cutoff_ms=cutoff,
        global_train_sample_budget=budget,
        per_batch_sample_quota=quota,
        thread_pool_workers=workers,
        snapshot_commit_interval_batches=interval,
        initial_event_time_ms=initial_time,
        tolerances=tolerances,
        raw=normalized,
    )


def load_training_config(path: str) -> TrainingConfig:
    try:
        raw = load_yaml_file(path)
    except OSError as exc:
        raise ConfigError("E_CONFIG_INVALID", "train_config", f"cannot load YAML: {exc}") from exc
    return _parse_training_config(raw)


def training_config_from_dict(raw: Any) -> TrainingConfig:
    return _parse_training_config(raw)
