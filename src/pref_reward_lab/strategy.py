from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import LabConfig, StrategyTier
from .features import FeatureParams, extract_pair_features
from .models import RawPair
from .scoring import score_features


@dataclass(frozen=True)
class StrategyDecision:
    selected_tier: str
    effective_hash_buckets: int
    precision_tier: str
    threshold_correction: float
    effective_threshold: float
    metrics: dict[str, Any]
    candidates: list[dict[str, Any]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_tier": self.selected_tier,
            "effective_hash_buckets": self.effective_hash_buckets,
            "precision_tier": self.precision_tier,
            "threshold_correction": self.threshold_correction,
            "effective_threshold": self.effective_threshold,
            "metrics": self.metrics,
            "candidates": self.candidates,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategyDecision":
        return cls(
            selected_tier=str(value["selected_tier"]),
            effective_hash_buckets=int(value["effective_hash_buckets"]),
            precision_tier=str(value["precision_tier"]),
            threshold_correction=float(value["threshold_correction"]),
            effective_threshold=float(value["effective_threshold"]),
            metrics=dict(value["metrics"]),
            candidates=list(value["candidates"]),
            reason=str(value["reason"]),
        )


def _candidate_score(
    tier: StrategyTier,
    *,
    pair_count: int,
    unique_buckets: int,
    boundary_ratio: float,
    base_buckets: int,
) -> tuple[float, dict[str, float]]:
    effective_buckets = base_buckets * tier.bucket_multiplier
    if pair_count > tier.max_pairs:
        capacity_penalty = ((pair_count - tier.max_pairs) / pair_count) * 100.0
    else:
        unused_ratio = (tier.max_pairs - pair_count) / tier.max_pairs
        capacity_penalty = unused_ratio * 0.18
    collision_pressure = unique_buckets / effective_buckets
    precision_cost = {"compact": 0.0, "balanced": 0.08, "fine": 0.16}[tier.precision_tier]
    boundary_penalty = abs(boundary_ratio - 0.15) * 1.5
    score = capacity_penalty + collision_pressure * 0.35 + precision_cost + boundary_penalty
    components = {
        "capacity_penalty": round(capacity_penalty, 9),
        "collision_pressure": round(collision_pressure, 9),
        "precision_cost": precision_cost,
        "boundary_penalty": round(boundary_penalty, 9),
    }
    return round(score, 9), components


def select_strategy(
    pairs: Iterable[RawPair],
    config: LabConfig,
) -> StrategyDecision:
    ordered_pairs = sorted(pairs, key=lambda pair: pair.pair_id)
    pilot_params = FeatureParams.from_config(
        config,
        precision_tier="compact",
        hash_buckets=config.hash_buckets,
    )
    unique_buckets: set[int] = set()
    boundary_count = 0
    usable_count = 0
    for pair in ordered_pairs:
        features = extract_pair_features(pair, pilot_params)
        unique_buckets.update(features.buckets)
        score = score_features(
            features,
            config,
            effective_threshold=config.reward_delta_threshold,
        )
        usable_count += 1
        if score.boundary:
            boundary_count += 1
    pair_count = len(ordered_pairs)
    boundary_ratio = boundary_count / usable_count if usable_count else 0.0
    metrics = {
        "pair_count": pair_count,
        "usable_pair_count": usable_count,
        "unique_feature_buckets": len(unique_buckets),
        "boundary_sample_count": boundary_count,
        "boundary_sample_ratio": round(boundary_ratio, 9),
        "pilot_hash_buckets": config.hash_buckets,
        "pilot_precision_tier": "compact",
    }
    candidates: list[dict[str, Any]] = []
    for tier in config.strategy_tiers:
        score, components = _candidate_score(
            tier,
            pair_count=pair_count,
            unique_buckets=len(unique_buckets),
            boundary_ratio=boundary_ratio,
            base_buckets=config.hash_buckets,
        )
        candidates.append(
            {
                "tier": tier.name,
                "score": score,
                "max_pairs": tier.max_pairs,
                "bucket_multiplier": tier.bucket_multiplier,
                "effective_hash_buckets": config.hash_buckets * tier.bucket_multiplier,
                "precision_tier": tier.precision_tier,
                "threshold_correction": tier.threshold_correction,
                "score_components": components,
                "reason": tier.description,
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["tier"]))
    if config.forced_strategy_tier is not None:
        selected_name = config.forced_strategy_tier
        reason = f"forced_strategy_tier={selected_name}"
    else:
        selected_name = str(candidates[0]["tier"])
        reason = (
            "lowest deterministic weighted cost across capacity fit, bucket pressure, "
            "precision cost and boundary ratio"
        )
    selected_tier = next(tier for tier in config.strategy_tiers if tier.name == selected_name)
    effective_threshold = round(
        config.reward_delta_threshold + selected_tier.threshold_correction, 12
    )
    return StrategyDecision(
        selected_tier=selected_tier.name,
        effective_hash_buckets=config.hash_buckets * selected_tier.bucket_multiplier,
        precision_tier=selected_tier.precision_tier,
        threshold_correction=selected_tier.threshold_correction,
        effective_threshold=effective_threshold,
        metrics=metrics,
        candidates=candidates,
        reason=reason,
    )
