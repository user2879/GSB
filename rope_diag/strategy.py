from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .attention import build_chunks
from .config import DiagnosticConfig
from .errors import DiagnosticError, ErrorCode
from .pruning import PruningPolicy, find_policy
from .util import sha256_json


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    description: str
    scheme_priority: tuple[str, ...]
    pruning_policy_id: str
    max_chunk_size: int


STRATEGY_TABLE: tuple[StrategySpec, ...] = (
    StrategySpec(
        strategy_id="S_LEAN",
        description="Budget-guarded path: pruned pair prefix and small scoring chunks.",
        scheme_priority=("rope", "linear_interpolation", "ntk_aware"),
        pruning_policy_id="low_rank_50",
        max_chunk_size=8,
    ),
    StrategySpec(
        strategy_id="S_COMPACT",
        description="Small model/sample tier: retain 75% of RoPE pairs.",
        scheme_priority=("rope", "ntk_aware", "linear_interpolation"),
        pruning_policy_id="balanced_75",
        max_chunk_size=16,
    ),
    StrategySpec(
        strategy_id="S_BALANCED",
        description="General tier: retain 75% pairs and use moderate chunks.",
        scheme_priority=("rope", "linear_interpolation", "ntk_aware"),
        pruning_policy_id="balanced_75",
        max_chunk_size=32,
    ),
    StrategySpec(
        strategy_id="S_WIDE",
        description="High extrapolation/large-dimension tier: full dimensions and wide chunks.",
        scheme_priority=("rope", "linear_interpolation", "ntk_aware"),
        pruning_policy_id="full",
        max_chunk_size=64,
    ),
)


def _sample_tier(count: int) -> str:
    if count <= 4:
        return "tiny"
    if count <= 12:
        return "medium"
    return "large"


def _dimension_tier(dimension: int) -> str:
    if dimension <= 32:
        return "small"
    if dimension <= 64:
        return "medium"
    return "large"


def _extrapolation_tier(ratio: float) -> str:
    if ratio < 2.0:
        return "modest"
    if ratio < 4.0:
        return "high"
    return "extreme"


def _budget_tier(projected_points: int, budget: int) -> str:
    if budget < projected_points:
        return "constrained"
    if budget >= projected_points * 3:
        return "ample"
    return "adequate"


def select_strategy(
    config: DiagnosticConfig,
    *,
    sample_count: int,
    total_points: int,
    policies: Sequence[PruningPolicy],
) -> dict[str, Any]:
    if sample_count <= 0:
        raise DiagnosticError(
            ErrorCode.SAMPLE_INVALID,
            "无可用于策略选择的合法样本",
            stage="strategy",
        )
    projected_points = int(total_points * len(config.schemes))
    sample_tier = _sample_tier(sample_count)
    dimension_tier = _dimension_tier(config.dimension)
    extrapolation_tier = _extrapolation_tier(config.extrapolation_ratio)
    budget_tier = _budget_tier(projected_points, config.point_budget)

    reasons: list[str] = []
    if budget_tier == "constrained" or sample_tier == "tiny":
        selected = next(item for item in STRATEGY_TABLE if item.strategy_id == "S_LEAN")
        reasons.extend(
            [
                "budget_or_sample_tier_requires_lean_path"
                if budget_tier == "constrained"
                else "tiny_sample_tier_requires_lean_path"
            ]
        )
    elif extrapolation_tier == "extreme" or dimension_tier == "large":
        selected = next(item for item in STRATEGY_TABLE if item.strategy_id == "S_WIDE")
        reasons.append("extreme_extrapolation_or_large_dimension_uses_full_path")
    elif sample_tier == "large" and extrapolation_tier != "modest":
        selected = next(item for item in STRATEGY_TABLE if item.strategy_id == "S_BALANCED")
        reasons.append("large_sample_high_extrapolation_uses_balanced_path")
    else:
        selected = next(item for item in STRATEGY_TABLE if item.strategy_id == "S_COMPACT")
        reasons.append("small_or_general_workload_uses_compact_path")

    policy = find_policy(list(policies), selected.pruning_policy_id)
    if policy is None:
        raise DiagnosticError(
            ErrorCode.CONFIG_INVALID,
            "策略表引用的剪枝策略不存在",
            stage="strategy",
            details={
                "strategy_id": selected.strategy_id,
                "missing_policy_id": selected.pruning_policy_id,
            },
        )

    dimensions = tuple(policy.resolve_dimensions(config.dimension))
    chunk_size = min(config.dimension_chunk_size, selected.max_chunk_size)
    chunks = build_chunks(dimensions, chunk_size)
    enabled_schemes = [
        scheme for scheme in selected.scheme_priority if scheme in config.schemes
    ]
    for scheme in config.schemes:
        if scheme not in enabled_schemes:
            enabled_schemes.append(scheme)

    strategy_payload = {
        "strategy_id": selected.strategy_id,
        "strategy_description": selected.description,
        "scheme_priority": enabled_schemes,
        "dimension_policy_id": policy.policy_id,
        "dimension_start": dimensions[0],
        "dimension_end": dimensions[-1],
        "dimension_indices": list(dimensions),
        "dimension_count": len(dimensions),
        "scoring_chunk_size": chunk_size,
        "chunk_boundaries": [chunk.to_dict() for chunk in chunks],
        "tiers": {
            "sample_count": sample_tier,
            "dimension": dimension_tier,
            "extrapolation_ratio": extrapolation_tier,
            "budget": budget_tier,
        },
        "sample_statistics": {
            "sample_count": sample_count,
            "total_points": total_points,
            "projected_points_across_schemes": projected_points,
            "point_budget": config.point_budget,
        },
        "decision_basis": reasons
        + [
            f"sample_count={sample_count}",
            f"dimension={config.dimension}",
            f"extrapolation_ratio={config.extrapolation_ratio:.6f}",
            f"projected_points={projected_points}",
            f"point_budget={config.point_budget}",
            "strategy_order=budget_or_tiny>extreme_or_large_dim>large_sample>default",
        ],
    }
    strategy_payload["strategy_fingerprint"] = strategy_fingerprint(strategy_payload)
    strategy_payload["pruning_fingerprint"] = policy.fingerprint(config.dimension)
    strategy_payload["table"] = strategy_table_public()
    return strategy_payload


def strategy_fingerprint(strategy_payload: dict[str, Any]) -> str:
    return sha256_json(
        {
            "strategy_id": strategy_payload["strategy_id"],
            "scheme_priority": strategy_payload["scheme_priority"],
            "dimension_policy_id": strategy_payload["dimension_policy_id"],
            "dimension_indices": strategy_payload["dimension_indices"],
            "scoring_chunk_size": strategy_payload["scoring_chunk_size"],
            "chunk_boundaries": strategy_payload["chunk_boundaries"],
            "tiers": strategy_payload["tiers"],
            "schema": "rope-diag-strategy-v1",
        }
    )


def strategy_table_public() -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": item.strategy_id,
            "description": item.description,
            "scheme_priority": list(item.scheme_priority),
            "pruning_policy_id": item.pruning_policy_id,
            "max_chunk_size": item.max_chunk_size,
        }
        for item in STRATEGY_TABLE
    ]

