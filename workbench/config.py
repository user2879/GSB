from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError
from .util import is_finite_number, stable_hash


DEFAULT_CONFIG: dict[str, Any] = {
    "timeline_alignment_ms": 100,
    "feature_dim": 4,
    "normalization": "minmax",
    "channel_weights": {},
    "fusion_temperature": 1.0,
    "fusion_bias": 0.0,
    "fusion_score_min": 0.0,
    "fusion_score_max": 1.0,
    "dedup_similarity_threshold": 0.6,
    "rerank_window_size": 4,
    "diversity_discount_factor": 0.8,
    "fixed_decimal_places": 4,
    "quantization_levels": 10001,
    "recompute_budget_per_run": 100,
    "thread_pool_workers": 4,
    "snapshot_commit_interval_shards": 1,
    "similarity_weights": {"category": 0.6, "price_band": 0.4},
}


@dataclass(frozen=True)
class RuntimeConfig:
    raw: dict[str, Any]
    config_hash: str
    scale: int
    quantization_levels: int
    places: int

    @property
    def alignment_ms(self) -> int:
        return int(self.raw["timeline_alignment_ms"])

    @property
    def feature_dim(self) -> int:
        return int(self.raw["feature_dim"])

    @property
    def normalization(self) -> str:
        return str(self.raw["normalization"])

    @property
    def channel_weights(self) -> dict[str, float]:
        return self.raw["channel_weights"]

    @property
    def weights_fingerprint(self) -> str:
        return stable_hash(
            {
                "channel_weights": self.raw["channel_weights"],
                "fusion_temperature": self.raw["fusion_temperature"],
                "fusion_bias": self.raw["fusion_bias"],
                "fusion_score_min": self.raw["fusion_score_min"],
                "fusion_score_max": self.raw["fusion_score_max"],
                "fixed_decimal_places": self.raw["fixed_decimal_places"],
                "quantization_levels": self.raw["quantization_levels"],
            }
        )

    @property
    def window_fingerprint(self) -> str:
        return stable_hash(
            {
                "dedup_similarity_threshold": self.raw["dedup_similarity_threshold"],
                "rerank_window_size": self.raw["rerank_window_size"],
                "diversity_discount_factor": self.raw["diversity_discount_factor"],
                "similarity_weights": self.raw["similarity_weights"],
                "fixed_decimal_places": self.raw["fixed_decimal_places"],
            }
        )

    @property
    def normalization_fingerprint(self) -> str:
        return stable_hash(
            {
                "normalization": self.raw["normalization"],
                "quantization_levels": self.raw["quantization_levels"],
                "fixed_decimal_places": self.raw["fixed_decimal_places"],
            }
        )


def validate_config(raw: dict[str, Any]) -> RuntimeConfig:
    if not isinstance(raw, dict):
        raise ConfigError("rerank config must be a JSON object")
    config = deepcopy(DEFAULT_CONFIG)
    config.update(deepcopy(raw))
    required_positive_ints = (
        "timeline_alignment_ms",
        "feature_dim",
        "rerank_window_size",
        "thread_pool_workers",
        "snapshot_commit_interval_shards",
    )
    for key in required_positive_ints:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{key} must be a positive integer")
    budget = config.get("recompute_budget_per_run")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ConfigError("recompute_budget_per_run must be a non-negative integer")
    if config["normalization"] not in {"minmax", "max", "none"}:
        raise ConfigError("normalization must be one of minmax, max, none")
    weights = config.get("channel_weights")
    if not isinstance(weights, dict) or not weights:
        raise ConfigError("channel_weights must be a non-empty object")
    for channel, weight in weights.items():
        if not isinstance(channel, str) or not channel:
            raise ConfigError("channel weight keys must be non-empty strings")
        if not is_finite_number(weight) or float(weight) < 0:
            raise ConfigError(f"channel weight for {channel} must be finite and non-negative")
    if sum(float(value) for value in weights.values()) <= 0:
        raise ConfigError("sum of channel weights must be greater than zero")
    value = config.get("fusion_temperature")
    if not is_finite_number(value) or float(value) <= 0:
        raise ConfigError("fusion_temperature must be finite and greater than zero")
    for key in ("fusion_bias", "fusion_score_min", "fusion_score_max"):
        value = config.get(key)
        if not is_finite_number(value):
            raise ConfigError(f"{key} must be finite")
    if float(config["fusion_score_min"]) >= float(config["fusion_score_max"]):
        raise ConfigError("fusion_score_min must be lower than fusion_score_max")
    threshold = config.get("dedup_similarity_threshold")
    if not is_finite_number(threshold) or not 0 <= float(threshold) <= 1:
        raise ConfigError("dedup_similarity_threshold must be in [0, 1]")
    discount = config.get("diversity_discount_factor")
    if not is_finite_number(discount) or not 0 <= float(discount) <= 1:
        raise ConfigError("diversity_discount_factor must be in [0, 1]")
    places = config.get("fixed_decimal_places")
    if not isinstance(places, int) or isinstance(places, bool) or not 0 <= places <= 9:
        raise ConfigError("fixed_decimal_places must be an integer in [0, 9]")
    levels = config.get("quantization_levels")
    if not isinstance(levels, int) or isinstance(levels, bool) or not 2 <= levels <= 10_000_001:
        raise ConfigError("quantization_levels must be an integer in [2, 10000001]")
    similarity = config.get("similarity_weights")
    if not isinstance(similarity, dict):
        raise ConfigError("similarity_weights must be an object")
    for key in ("category", "price_band"):
        value = similarity.get(key)
        if not is_finite_number(value) or not 0 <= float(value) <= 1:
            raise ConfigError(f"similarity_weights.{key} must be in [0, 1]")
    if float(similarity["category"]) + float(similarity["price_band"]) > 1.0000001:
        raise ConfigError("similarity weights must sum to at most one")
    return RuntimeConfig(
        raw=config,
        config_hash=stable_hash(config),
        scale=10**places,
        quantization_levels=levels,
        places=places,
    )


def patch_config(config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    from .util import deep_patch

    return deep_patch(config, patch)
