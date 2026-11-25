"""Scale-tier selection from dataset statistics and strategy cost estimates."""

from __future__ import annotations

from typing import Any

from .config import Config, EffectiveConfig, custom_effective_config, effective_config


TIER_SEQUENCE = ("small", "medium", "large")


def select_effective_config(config: Config, stats: dict[str, int]) -> tuple[EffectiveConfig, dict[str, Any]]:
    if config.tier == "custom":
        effective = custom_effective_config(config)
        basis = {
            "mode": "explicit_custom",
            "selected_tier": "custom",
            "stats": stats,
            "reason": "configuration requested the custom window/merge/precision tuple",
            "candidates": [],
        }
        return effective, basis
    if config.tier in TIER_SEQUENCE:
        effective = effective_config(config, config.tier)
        basis = {
            "mode": "explicit_tier",
            "selected_tier": config.tier,
            "stats": stats,
            "reason": f"configuration pinned tier {config.tier}",
            "candidates": [_candidate_evidence(config, name, stats) for name in TIER_SEQUENCE],
        }
        return effective, basis

    candidates: list[dict[str, Any]] = []
    selected = "large"
    for name in TIER_SEQUENCE:
        evidence = _candidate_evidence(config, name, stats)
        candidates.append(evidence)
        if evidence["fits"] and selected == "large":
            selected = name
    effective = effective_config(config, selected)
    basis = {
        "mode": "auto",
        "selected_tier": selected,
        "stats": stats,
        "reason": (
            f"smallest capacity-compatible tier is {selected}; "
            f"estimated image cost={effective.strategy.single_image_cost:.2f}"
        ),
        "candidates": candidates,
        "selection_rule": [
            "filter by total point, max column, and max chars-per-column capacity",
            "choose smallest sufficient tier in small, medium, large order",
            "fall back to large when no tier fits",
        ],
    }
    return effective, basis


def _candidate_evidence(config: Config, name: str, stats: dict[str, int]) -> dict[str, Any]:
    strategy = config.strategies[name]
    checks = {
        "points": stats["total_points"] <= strategy.capacity_points,
        "columns": stats["max_columns"] <= strategy.capacity_columns,
        "chars_per_column": stats["max_chars_per_column"] <= strategy.capacity_chars_per_column,
    }
    estimated_cost = stats["image_count"] * strategy.single_image_cost
    return {
        "tier": name,
        "fits": all(checks.values()),
        "checks": checks,
        "limits": {
            "capacity_points": strategy.capacity_points,
            "capacity_columns": strategy.capacity_columns,
            "capacity_chars_per_column": strategy.capacity_chars_per_column,
        },
        "single_image_cost": strategy.single_image_cost,
        "estimated_dataset_cost": round(estimated_cost, 6),
        "window": strategy.window.as_dict(),
        "merge_gap": strategy.merge_gap,
        "precision": strategy.precision.as_dict(),
    }
