"""Data-scale-aware strategy selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import RenderConfig
from .samples import SequenceSpec
from .stable import round_half_away


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    encoding_precision: int
    block_size: int
    prediction_path: str
    quantization_level: int
    quality_weight: float
    speed_weight: float
    cost_factor: float


@dataclass(frozen=True)
class ScaledRunInput:
    tier: int
    horizon: int
    trajectory_limit: int
    sequences: tuple[SequenceSpec, ...]
    trajectory_count: int
    decision: dict[str, Any]
    selected: StrategyCandidate


def _mid(values: list[int]) -> int:
    return values[len(values) // 2]


def _candidate_table(config: RenderConfig) -> list[StrategyCandidate]:
    precisions = sorted(set(min(level, config.fixed_point_bits) for level in config.precision_levels))
    quantizations = sorted(set(config.quantization_levels))
    return [
        StrategyCandidate(
            candidate_id="quality-linear",
            encoding_precision=precisions[-1],
            block_size=4,
            prediction_path="linear",
            quantization_level=quantizations[0],
            quality_weight=1.0,
            speed_weight=0.35,
            cost_factor=1.0,
        ),
        StrategyCandidate(
            candidate_id="balanced-weighted",
            encoding_precision=precisions[_mid_index(precisions)],
            block_size=8,
            prediction_path="weighted",
            quantization_level=quantizations[_mid_index(quantizations)],
            quality_weight=0.82,
            speed_weight=0.75,
            cost_factor=0.8,
        ),
        StrategyCandidate(
            candidate_id="throughput-weighted",
            encoding_precision=precisions[0],
            block_size=16,
            prediction_path="weighted",
            quantization_level=quantizations[-1],
            quality_weight=0.60,
            speed_weight=1.0,
            cost_factor=0.55,
        ),
    ]


def _mid_index(values: list[int]) -> int:
    return (len(values) - 1) // 2 if len(values) > 1 else 0


def select_scale_input(
    config: RenderConfig,
    sequences: list[SequenceSpec],
    tier: int,
) -> ScaledRunInput:
    if tier not in config.scale_tiers:
        raise ValueError(f"tier {tier} is not one of {list(config.scale_tiers)}")
    if not sequences:
        raise ValueError("at least one valid sequence is required")

    scale_tiers = list(config.scale_tiers)
    tier_index = scale_tiers.index(tier)
    min_tier = scale_tiers[0]
    max_tier = scale_tiers[-1]
    tier_ratio = tier / float(max_tier)

    ordered = sorted(sequences, key=lambda item: item.sequence_id)
    sequence_limit = max(1, round_half_away(len(ordered) * tier_ratio))
    selected = ordered[: min(len(ordered), sequence_limit)]

    maximum_trajectories = max(
        len(frame["trajectories"])
        for sequence in selected
        for frame in sequence.frames
    )
    trajectory_limit = max(1, round_half_away(maximum_trajectories * tier_ratio))
    horizon = max(1, round_half_away(config.prediction_steps * tier / float(min_tier)))

    trajectory_count = sum(
        min(len(frame["trajectories"]), trajectory_limit)
        for sequence in selected
        for frame in sequence.frames
    )
    frame_observations = sum(sequence.frame_count for sequence in selected)
    average_trajectories = trajectory_count / float(frame_observations)

    blocks = math.ceil(config.frame_width / 8.0) * math.ceil(config.frame_height / 8.0)
    estimated_per_sequence_cost = round(
        (
            config.frame_width * config.frame_height
            + average_trajectories * 24.0
            + horizon * blocks * 3.0
            + config.history_frames * 11.0
        )
        / 250.0,
        6,
    )

    candidates = _candidate_table(config)
    sequence_load = min(1.0, max(0.0, (len(selected) - 1) / 7.0))
    trajectory_load = min(1.0, max(0.0, (average_trajectories - 1.0) / 8.0))
    horizon_load = min(1.0, max(0.0, (horizon - 2) / 8.0))
    tier_load = tier_index / float(max(1, len(scale_tiers) - 1))
    latency_load = min(1.0, estimated_per_sequence_cost / float(config.latency_budget_ms))
    target_load = round(
        0.25 * sequence_load
        + 0.25 * trajectory_load
        + 0.25 * horizon_load
        + 0.15 * tier_load
        + 0.10 * latency_load,
        9,
    )

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        latency_ratio = round(
            estimated_per_sequence_cost
            * candidate.cost_factor
            * len(selected)
            / float(config.latency_budget_ms),
            9,
        )
        budget_penalty = max(0.0, latency_ratio - 1.0) * 30.0
        score = round(
            100.0
            * (
                candidate.quality_weight * (1.0 - target_load)
                + candidate.speed_weight * target_load
            )
            - budget_penalty,
            6,
        )
        scored.append(
            {
                "candidate_id": candidate.candidate_id,
                "encoding_precision": candidate.encoding_precision,
                "block_size": candidate.block_size,
                "prediction_path": candidate.prediction_path,
                "quantization_level": candidate.quantization_level,
                "estimated_latency_ratio": latency_ratio,
                "budget_penalty": round(budget_penalty, 6),
                "score": score,
            }
        )

    order = {
        row["candidate_id"]: index for index, row in enumerate(scored)
    }
    scored.sort(key=lambda row: (-row["score"], order[row["candidate_id"]]))
    winner_row = scored[0]
    winner = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == winner_row["candidate_id"]
    )
    metrics = {
        "sequence_count": len(selected),
        "available_sequence_count": len(ordered),
        "sequence_limit": sequence_limit,
        "average_trajectory_count": round(average_trajectories, 9),
        "trajectory_limit": trajectory_limit,
        "prediction_steps": horizon,
        "base_prediction_steps": config.prediction_steps,
        "scale_tier": tier,
        "tier_index": tier_index,
        "single_sequence_cost": estimated_per_sequence_cost,
        "aggregate_cost": round(estimated_per_sequence_cost * len(selected), 6),
        "latency_budget_ms": config.latency_budget_ms,
        "target_load": target_load,
        "load_components": {
            "sequence": round(sequence_load, 9),
            "trajectory": round(trajectory_load, 9),
            "horizon": round(horizon_load, 9),
            "tier": round(tier_load, 9),
            "latency": round(latency_load, 9),
        },
    }
    reason = (
        f"selected {winner.candidate_id} with score {winner_row['score']:.6f}; "
        f"sequences={metrics['sequence_count']}, "
        f"avg_trajectories={metrics['average_trajectory_count']:.3f}, "
        f"prediction_steps={horizon}, target_load={target_load:.6f}; "
        "higher target_load trades encoding precision and block size for throughput, "
        "while the budget penalty excludes candidates that exceed the latency budget"
    )
    decision = {
        "schema_version": 1,
        "metrics": metrics,
        "candidates": scored,
        "selected": {
            "candidate_id": winner.candidate_id,
            "encoding_precision": winner.encoding_precision,
            "block_size": winner.block_size,
            "prediction_path": winner.prediction_path,
            "quantization_level": winner.quantization_level,
            "score": winner_row["score"],
        },
        "reason": reason,
    }
    return ScaledRunInput(
        tier=tier,
        horizon=horizon,
        trajectory_limit=trajectory_limit,
        sequences=tuple(selected),
        trajectory_count=trajectory_count,
        decision=decision,
        selected=winner,
    )

