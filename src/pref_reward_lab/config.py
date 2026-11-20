from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import read_json
from .errors import ConfigError

DEFAULT_REWARD_LABELS = (
    "strong_preference",
    "preference",
    "neutral",
    "weak_preference",
)
DEFAULT_PRECISION_LABELS = ("milli_exact", "float32", "float64")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a finite number", field=field_name)
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field_name} must be finite", field=field_name)
    return result


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer", field=field_name)
    return value


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field_name} must be a non-empty string array", field=field_name)
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"{field_name}[{index}] must be a non-empty string",
                field=f"{field_name}[{index}]",
            )
        result.append(item.strip())
    if len(set(result)) != len(result):
        raise ConfigError(f"{field_name} contains duplicates", field=field_name)
    return tuple(result)


def _normalize_tokenizer_version(value: Any) -> str:
    aliases = {
        "v1": "subword_v1",
        "subword-v1": "subword_v1",
        "subword_1": "subword_v1",
        "v2": "subword_v2",
        "subword-v2": "subword_v2",
        "subword_2": "subword_v2",
    }
    if not isinstance(value, str):
        raise ConfigError("tokenizer.version must be a string", field="tokenizer.version")
    normalized = aliases.get(value.strip(), value.strip())
    if normalized not in {"subword_v1", "subword_v2"}:
        raise ConfigError(
            "tokenizer.version must be one of subword_v1, subword_v2",
            field="tokenizer.version",
        )
    return normalized


@dataclass(frozen=True)
class StrategyTier:
    name: str
    max_pairs: int
    bucket_multiplier: int
    threshold_correction: float
    precision_tier: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_pairs": self.max_pairs,
            "bucket_multiplier": self.bucket_multiplier,
            "threshold_correction": self.threshold_correction,
            "precision_tier": self.precision_tier,
            "description": self.description,
        }


@dataclass(frozen=True)
class LabConfig:
    raw: dict[str, Any]
    tokenizer_version: str
    feature_dim: int
    hash_buckets: int
    normalize_text: bool
    scoring_weights: dict[str, float]
    reward_delta_threshold: float
    concurrency: int
    snapshot_commit_interval: int
    reward_labels: tuple[str, ...]
    precision_labels: tuple[str, ...]
    v1_default_label: str
    source_version: str
    strategy_tiers: tuple[StrategyTier, ...]
    forced_strategy_tier: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LabConfig":
        if not isinstance(value, dict):
            raise ConfigError("config root must be an object")

        tokenizer = value.get("tokenizer", value.get("tokenizer_config", {}))
        if isinstance(tokenizer, str):
            tokenizer = {"version": tokenizer}
        if not isinstance(tokenizer, dict):
            raise ConfigError("tokenizer must be an object or version string", field="tokenizer")
        tokenizer_version = _normalize_tokenizer_version(
            tokenizer.get(
                "version",
                value.get(
                    "subword_rule_version",
                    value.get("tokenizer_rule_version", "subword_v1"),
                ),
            )
        )

        feature_dim = _positive_int(
            value.get("feature_dim", value.get("feature_dimension", 64)), "feature_dim"
        )
        hash_buckets = _positive_int(value.get("hash_buckets", 256), "hash_buckets")

        normalize_text = value.get("normalize_text", value.get("normalization", True))
        if not isinstance(normalize_text, bool):
            raise ConfigError("normalize_text must be boolean", field="normalize_text")

        raw_weights = value.get("scoring_weights", {})
        weight_keys = ("prompt", "chosen", "rejected", "length", "bias")
        if isinstance(raw_weights, list):
            if len(raw_weights) != len(weight_keys):
                raise ConfigError(
                    "scoring_weights array must contain prompt, chosen, rejected, length, bias",
                    field="scoring_weights",
                )
            raw_weights = dict(zip(weight_keys, raw_weights))
        if not isinstance(raw_weights, dict):
            raise ConfigError("scoring_weights must be an object", field="scoring_weights")
        weight_aliases = {
            "prompt_weight": "prompt",
            "chosen_weight": "chosen",
            "rejected_weight": "rejected",
            "length_weight": "length",
            "bias_weight": "bias",
            "answer_weight": "chosen",
        }
        normalized_weights: dict[str, Any] = {}
        for key, raw_value in raw_weights.items():
            normalized_key = weight_aliases.get(key, key)
            normalized_weights.setdefault(normalized_key, raw_value)
        raw_weights = normalized_weights
        weight_defaults = {"prompt": 0.15, "chosen": 1.0, "rejected": 1.0, "length": 0.02, "bias": 0.0}
        scoring_weights: dict[str, float] = {}
        for key, default in weight_defaults.items():
            scoring_weights[key] = _finite_number(
                raw_weights.get(key, default), f"scoring_weights.{key}"
            )
        extra_weights = sorted(set(raw_weights) - set(weight_defaults))
        if extra_weights:
            raise ConfigError(
                f"unsupported scoring weight(s): {', '.join(extra_weights)}",
                field="scoring_weights",
            )

        reward_delta_threshold = _finite_number(
            value.get(
                "reward_delta_threshold",
                value.get("delta_threshold", 0.05),
            ),
            "reward_delta_threshold",
        )
        concurrency = _positive_int(value.get("concurrency", 1), "concurrency")
        snapshot_commit_interval = _positive_int(
            value.get(
                "snapshot_commit_interval",
                value.get("snapshot_interval_rounds", 1),
            ),
            "snapshot_commit_interval",
        )

        labels_object = value.get("labels", {})
        if not isinstance(labels_object, dict):
            raise ConfigError("labels must be an object", field="labels")
        reward_labels = _string_list(
            value.get(
                "reward_labels",
                value.get("named_reward_labels", labels_object.get("reward", list(DEFAULT_REWARD_LABELS))),
            ),
            "reward_labels",
        )
        precision_labels = _string_list(
            value.get(
                "precision_labels",
                value.get("precision_tags", labels_object.get("precision", list(DEFAULT_PRECISION_LABELS))),
            ),
            "precision_labels",
        )
        v1_default_label = value.get("v1_default_label", "neutral")
        if not isinstance(v1_default_label, str) or v1_default_label not in reward_labels:
            raise ConfigError(
                "v1_default_label must be present in reward_labels",
                field="v1_default_label",
            )

        source_version = value.get("source_version", value.get("exchange_version", "v2"))
        if source_version not in {"v1", "v2"}:
            raise ConfigError("source_version must be v1 or v2", field="source_version")

        raw_tiers = value.get(
            "strategy_tiers",
            value.get("scale_tiers", _default_strategy_tiers()),
        )
        if not isinstance(raw_tiers, list) or not raw_tiers:
            raise ConfigError("strategy_tiers must be a non-empty array", field="strategy_tiers")
        tiers: list[StrategyTier] = []
        for index, raw_tier in enumerate(raw_tiers):
            prefix = f"strategy_tiers[{index}]"
            if not isinstance(raw_tier, dict):
                raise ConfigError("strategy tier must be an object", field=prefix)
            name = raw_tier.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ConfigError("strategy tier name is required", field=f"{prefix}.name")
            precision_tier = raw_tier.get(
                "precision_tier", raw_tier.get("precision", "compact")
            )
            if precision_tier not in {"compact", "balanced", "fine"}:
                raise ConfigError(
                    "precision_tier must be compact, balanced, or fine",
                    field=f"{prefix}.precision_tier",
                )
            tiers.append(
                StrategyTier(
                    name=name.strip(),
                    max_pairs=_positive_int(raw_tier.get("max_pairs"), f"{prefix}.max_pairs"),
                    bucket_multiplier=_positive_int(
                        raw_tier.get("bucket_multiplier", raw_tier.get("bucket_scale", 1)),
                        f"{prefix}.bucket_multiplier",
                    ),
                    threshold_correction=_finite_number(
                        raw_tier.get(
                            "threshold_correction",
                            raw_tier.get("threshold_delta", 0.0),
                        ),
                        f"{prefix}.threshold_correction",
                    ),
                    precision_tier=precision_tier,
                    description=str(raw_tier.get("description", "")),
                )
            )
        if len({tier.name for tier in tiers}) != len(tiers):
            raise ConfigError("strategy tier names must be unique", field="strategy_tiers")
        if len({tier.max_pairs for tier in tiers}) != len(tiers):
            raise ConfigError("strategy tier max_pairs must be unique", field="strategy_tiers")

        forced_strategy_tier = value.get("forced_strategy_tier")
        if forced_strategy_tier is not None:
            if not isinstance(forced_strategy_tier, str):
                raise ConfigError(
                    "forced_strategy_tier must be null or a tier name",
                    field="forced_strategy_tier",
                )
            if forced_strategy_tier not in {tier.name for tier in tiers}:
                raise ConfigError(
                    f"unknown forced_strategy_tier: {forced_strategy_tier}",
                    field="forced_strategy_tier",
                )

        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ConfigError("metadata must be an object", field="metadata")

        normalized_raw = dict(value)
        normalized_raw.update(
            {
                "tokenizer": {**tokenizer, "version": tokenizer_version},
                "feature_dim": feature_dim,
                "hash_buckets": hash_buckets,
                "normalize_text": normalize_text,
                "scoring_weights": scoring_weights,
                "reward_delta_threshold": reward_delta_threshold,
                "concurrency": concurrency,
                "snapshot_commit_interval": snapshot_commit_interval,
                "reward_labels": list(reward_labels),
                "precision_labels": list(precision_labels),
                "v1_default_label": v1_default_label,
                "source_version": source_version,
                "strategy_tiers": [tier.to_dict() for tier in tiers],
                "forced_strategy_tier": forced_strategy_tier,
                "metadata": metadata,
            }
        )
        return cls(
            raw=normalized_raw,
            tokenizer_version=tokenizer_version,
            feature_dim=feature_dim,
            hash_buckets=hash_buckets,
            normalize_text=normalize_text,
            scoring_weights=scoring_weights,
            reward_delta_threshold=reward_delta_threshold,
            concurrency=concurrency,
            snapshot_commit_interval=snapshot_commit_interval,
            reward_labels=reward_labels,
            precision_labels=precision_labels,
            v1_default_label=v1_default_label,
            source_version=source_version,
            strategy_tiers=tuple(tiers),
            forced_strategy_tier=forced_strategy_tier,
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabConfig":
        try:
            value = read_json(path)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config: {exc}") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def stage_params(self, stage: str) -> dict[str, Any]:
        if stage == "features":
            return {
                "tokenizer_version": self.tokenizer_version,
                "feature_dim": self.feature_dim,
                "normalize_text": self.normalize_text,
                "precision_labels": list(self.precision_labels),
            }
        if stage == "scoring":
            return {
                "scoring_weights": self.scoring_weights,
                "reward_delta_threshold": self.reward_delta_threshold,
                "reward_labels": list(self.reward_labels),
                "precision_labels": list(self.precision_labels),
            }
        if stage == "conversion":
            return {
                "reward_labels": list(self.reward_labels),
                "precision_labels": list(self.precision_labels),
                "v1_default_label": self.v1_default_label,
            }
        if stage == "metrics":
            return {
                "reward_delta_threshold": self.reward_delta_threshold,
                "metrics_schema": 1,
            }
        raise ConfigError(f"unknown stage: {stage}", field="stage")

    def with_updates(self, updates: dict[str, Any]) -> "LabConfig":
        merged = dict(self.raw)
        for key, value in updates.items():
            if key == "tokenizer" and isinstance(value, dict):
                merged["tokenizer"] = {**dict(merged.get("tokenizer", {})), **value}
            else:
                merged[key] = value
        return LabConfig.from_dict(merged)


def _default_strategy_tiers() -> list[dict[str, Any]]:
    return [
        {
            "name": "small",
            "max_pairs": 200,
            "bucket_multiplier": 1,
            "threshold_correction": 0.0,
            "precision_tier": "compact",
            "description": "Small samples use the configured base hash space.",
        },
        {
            "name": "medium",
            "max_pairs": 1000,
            "bucket_multiplier": 2,
            "threshold_correction": 0.01,
            "precision_tier": "balanced",
            "description": "Medium samples double buckets and add subword detail.",
        },
        {
            "name": "large",
            "max_pairs": 5000,
            "bucket_multiplier": 4,
            "threshold_correction": 0.02,
            "precision_tier": "fine",
            "description": "Large samples expand buckets and use fine subwords.",
        },
    ]
