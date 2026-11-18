"""Configuration, rule table and fault-injection loading + validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, E_RULE_WEIGHT, E_THRESHOLD_BOUNDARY, E_TIER_LIMIT_ZERO, PipelineError


@dataclass(frozen=True)
class Tier:
    name: str
    max_samples: int


@dataclass(frozen=True)
class HallucinationRule:
    rule_id: str
    name: str
    weight: float
    priority: int
    params: dict


@dataclass(frozen=True)
class Config:
    feature_dim: int
    hash_buckets: int
    precision_level: int
    tokenizer_seed: int
    preference_sample_ratio: float
    reward_boundary_threshold: float
    hallucination_threshold: float
    metrics: tuple
    concurrency: int
    snapshot_interval: int
    tiers: tuple
    raw: dict = field(hash=False, compare=False)

    def tier_for_size(self, total: int) -> tuple:
        """Return (tier, index) for a total sample count."""
        for index, tier in enumerate(self.tiers):
            if total <= tier.max_samples:
                return tier, index
        return self.tiers[-1], len(self.tiers) - 1


def _require(condition: bool, message: str, code: str = E_CONFIG_INVALID) -> None:
    if not condition:
        raise PipelineError(code, message)


def load_config(path: str) -> Config:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise PipelineError(E_CONFIG_INVALID, f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineError(E_CONFIG_INVALID, "pipeline config must be a mapping")

    try:
        feature_dim = int(raw["feature_dim"])
        hash_buckets = int(raw["hash_buckets"])
        precision_level = int(raw.get("precision_level", 2))
        tokenizer_seed = int(raw.get("tokenizer_seed", 0))
        preference_sample_ratio = float(raw["preference_sample_ratio"])
        reward_boundary_threshold = float(raw["reward_boundary_threshold"])
        hallucination_threshold = float(raw["hallucination_threshold"])
        metrics = tuple(str(item) for item in raw["metrics"])
        concurrency = int(raw["concurrency"])
        snapshot_interval = int(raw["snapshot_interval"])
        tier_raw = raw["tiers"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(E_CONFIG_INVALID, f"missing or mistyped config field: {exc}") from exc

    _require(feature_dim > 0, "feature_dim must be positive")
    _require(hash_buckets > 0, "hash_buckets must be positive")
    _require(hash_buckets % feature_dim == 0, "hash_buckets must be a multiple of feature_dim")
    _require(1 <= precision_level <= 3, "precision_level must be in [1, 3]")
    _require(0.0 < preference_sample_ratio <= 1.0, "preference_sample_ratio must be in (0, 1]")
    _require(reward_boundary_threshold >= 0.0, "reward_boundary_threshold must be >= 0")
    _require(0.0 <= hallucination_threshold <= 1.0, "hallucination_threshold must be in [0, 1]")
    _require(concurrency > 0, "concurrency must be a positive integer")
    _require(snapshot_interval > 0, "snapshot_interval must be a positive integer")
    _require(len(metrics) > 0, "at least one metric is required")

    tiers = []
    previous = 0
    for item in tier_raw:
        tier = Tier(name=str(item["name"]), max_samples=int(item["max_samples"]))
        if tier.max_samples <= 0:
            raise PipelineError(E_TIER_LIMIT_ZERO, f"tier {tier.name} has non-positive max_samples")
        if tier.max_samples <= previous:
            raise PipelineError(E_CONFIG_INVALID, "tier max_samples must be strictly increasing")
        previous = tier.max_samples
        tiers.append(tier)
    _require(len(tiers) > 0, "at least one scale tier is required")

    return Config(
        feature_dim=feature_dim,
        hash_buckets=hash_buckets,
        precision_level=precision_level,
        tokenizer_seed=tokenizer_seed,
        preference_sample_ratio=preference_sample_ratio,
        reward_boundary_threshold=reward_boundary_threshold,
        hallucination_threshold=hallucination_threshold,
        metrics=metrics,
        concurrency=concurrency,
        snapshot_interval=snapshot_interval,
        tiers=tuple(tiers),
        raw=raw,
    )


def load_rules(path: str, failures: list | None = None) -> tuple:
    """Load the hallucination rule table.

    Out-of-range weights raise E_RULE_WEIGHT. When a failure list is supplied
    the offending rule is dropped (recorded in the list) instead of aborting,
    which is how injected rule-weight faults stay isolated.
    """
    import json

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rules = []
    for item in raw["rules"]:
        rule = HallucinationRule(
            rule_id=str(item["rule_id"]),
            name=str(item["name"]),
            weight=float(item["weight"]),
            priority=int(item["priority"]),
            params=dict(item.get("params", {})),
        )
        if not (0.0 <= rule.weight <= 1.0):
            record = PipelineError(
                E_RULE_WEIGHT,
                f"rule {rule.rule_id} weight {rule.weight} outside [0, 1]",
                sample_id=rule.rule_id,
                stage="rule_validation",
            ).to_record()
            if failures is None:
                raise PipelineError(E_RULE_WEIGHT, record["message"])
            failures.append(record)
            continue
        rules.append(rule)
    priorities = [rule.priority for rule in rules]
    if len(set(priorities)) != len(priorities):
        raise PipelineError(E_CONFIG_INVALID, "rule priorities must be unique (fixed tie-breaking)")
    return tuple(rules)


@dataclass(frozen=True)
class Injection:
    item_no: int
    kind: str
    target_id: str
    detail: str


def load_injection(path: str | None) -> tuple:
    """Load the standalone injection JSON. Disabled items are ignored."""
    if not path:
        return ()
    raw: dict[str, Any]
    import json

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    injections = []
    for item in raw.get("items", []):
        if not item.get("enabled", False):
            continue
        injections.append(
            Injection(
                item_no=int(item["item_no"]),
                kind=str(item["kind"]),
                target_id=str(item.get("target_id", "")),
                detail=str(item.get("detail", "")),
            )
        )
    injections.sort(key=lambda injection: injection.item_no)
    return tuple(injections)


def boundary_injection_present(injections) -> bool:
    for injection in injections:
        if injection.kind == "threshold_boundary":
            raise PipelineError(
                E_THRESHOLD_BOUNDARY,
                f"injection {injection.item_no}: hallucination threshold is exactly on the boundary",
                sample_id=injection.target_id,
                stage="config_validation",
            )
    return False
