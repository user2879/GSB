"""Scale-adaptive strategy selector with auditable candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    scale_index: int
    precision_level: int
    bucket_multiplier: int
    threshold_adjustment: float
    capacity: int
    # candidate score weights (part of the fixed preset strategy table)
    fit_weight: float
    pressure_weight: float
    cardinality_penalty_weight: float


STRATEGY_TABLE = (
    Strategy("S0_small", 0, 3, 1, 0.00, 500, 1.0, 0.20, 0.10),
    Strategy("S1_medium", 1, 2, 2, -0.02, 2_500, 1.0, 0.20, 0.15),
    Strategy("S2_large", 2, 1, 4, -0.05, 10_000, 1.0, 0.25, 0.20),
    Strategy("S3_xlarge", 3, 1, 8, -0.08, 1_000_000, 1.0, 0.30, 0.25),
)


def scale_metrics(total_samples: int, unique_buckets: int, hash_buckets: int,
                  high_card_ratio: float) -> dict:
    return {
        "total_samples": int(total_samples),
        "unique_buckets": int(unique_buckets),
        "hash_buckets": int(hash_buckets),
        "unique_bucket_ratio": (unique_buckets / hash_buckets) if hash_buckets else 0.0,
        "high_cardinality_ratio": float(high_card_ratio),
    }


def select_strategy(config: Config, total_samples: int, unique_buckets: int,
                    high_card_ratio: float) -> dict:
    """Choose a strategy deterministically from the preset table.

    Only strategies whose scale_index does not exceed the current tier index
    are eligible; the highest deterministic candidate score wins, with the
    lowest scale_index breaking ties.
    """
    _tier, tier_index = config.tier_for_size(total_samples)
    metrics = scale_metrics(total_samples, unique_buckets, config.hash_buckets, high_card_ratio)
    candidates = []
    for strategy in STRATEGY_TABLE:
        if strategy.scale_index > tier_index:
            continue
        tier_cap = config.tiers[strategy.scale_index].max_samples
        fill = min(1.0, total_samples / tier_cap)
        pressure = min(1.0, total_samples / strategy.capacity)
        capacity_penalty = 1.0 if total_samples > strategy.capacity else 0.0
        score = (
            strategy.fit_weight * fill
            + strategy.pressure_weight * pressure
            + strategy.cardinality_penalty_weight * metrics["unique_bucket_ratio"]
            + 0.5 * high_card_ratio
            - 2.0 * capacity_penalty
        )
        candidates.append({
            "strategy_id": strategy.strategy_id,
            "scale_index": strategy.scale_index,
            "score": score,
            "precision_level": strategy.precision_level,
            "bucket_multiplier": strategy.bucket_multiplier,
            "threshold_adjustment": strategy.threshold_adjustment,
            "capacity_violation": bool(capacity_penalty),
        })

    best = max(
        candidates,
        key=lambda candidate: (candidate["score"], -candidate["scale_index"]),
    )
    chosen = next(item for item in STRATEGY_TABLE if item.strategy_id == best["strategy_id"])
    reasons = [
        f"tier_index={tier_index} makes {len(candidates)} strategies eligible",
        f"highest candidate score {best['score']:.6f}",
        f"fill={fill:.6f}, high_cardinality_ratio={high_card_ratio:.6f}",
        f"select precision={chosen.precision_level}, "
        f"bucket_multiplier={chosen.bucket_multiplier}, "
        f"threshold_adjustment={chosen.threshold_adjustment:+.2f}",
    ]
    return {
        "tier_name": config.tiers[tier_index].name,
        "tier_index": tier_index,
        "strategy_id": chosen.strategy_id,
        "precision_level": chosen.precision_level,
        "bucket_multiplier": chosen.bucket_multiplier,
        "threshold_adjustment": chosen.threshold_adjustment,
        "adjusted_hallucination_threshold": max(
            0.0, min(1.0, config.hallucination_threshold + chosen.threshold_adjustment)
        ),
        "effective_hash_buckets": config.hash_buckets * chosen.bucket_multiplier,
        "scale_metrics": metrics,
        "candidate_scores": candidates,
        "reasons": reasons,
    }
