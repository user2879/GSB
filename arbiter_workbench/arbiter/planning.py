"""Explicit full-vs-incremental arbitration path selection rules.

Cost model (all quantities are plain integers, so the decision is stable):

* full path work        = E * K * N_avg
    (E participating environments, K keys, N_avg average observations/env)
* incremental path work = A * K_dirty * N_avg + C_prop * (E + K)
    where A is the number of changed environments, K_dirty the keys touched by
    the change set, and C_prop a fixed propagation cost per relation edge.

A tolerance band of 10% (``TOLERANCE_RATIO``) is applied to the difference.
Inside the band the incremental path is selected by the stable rule
``incremental preferred`` and the tolerance effect is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TOLERANCE_RATIO_NUM = 1
TOLERANCE_RATIO_DEN = 10
PROPAGATION_COST = 2


@dataclass(frozen=True)
class ScaleParams:
    env_count: int
    key_count: int
    obs_total: int
    avg_obs_per_env: int
    changed_env_count: int
    dirty_key_count: int


@dataclass(frozen=True)
class PathDecision:
    selected: str
    reason: str
    scale: ScaleParams
    full_work: int
    incremental_work: int
    threshold_boundary: int
    within_tolerance: bool
    tolerance_effect: str
    not_triggered_boundaries: dict = field(default_factory=dict)


def estimate(scale: ScaleParams) -> tuple[int, int, int]:
    full = scale.env_count * scale.key_count * max(1, scale.avg_obs_per_env)
    incremental = (
        max(1, scale.changed_env_count)
        * max(1, scale.dirty_key_count)
        * max(1, scale.avg_obs_per_env)
        + PROPAGATION_COST * (scale.env_count + scale.key_count)
    )
    # tolerance boundary: 10% of full work, integer ceil to stay conservative
    boundary = -(-(full * TOLERANCE_RATIO_NUM) // TOLERANCE_RATIO_DEN)
    return full, incremental, boundary


def decide(
    env_count: int,
    key_count: int,
    obs_total: int,
    changed_env_count: int,
    dirty_key_count: int,
) -> PathDecision:
    avg_obs = obs_total // env_count if env_count else 0
    scale = ScaleParams(
        env_count=env_count,
        key_count=key_count,
        obs_total=obs_total,
        avg_obs_per_env=avg_obs,
        changed_env_count=changed_env_count,
        dirty_key_count=dirty_key_count,
    )
    full, incremental, boundary = estimate(scale)
    diff = full - incremental
    within = abs(diff) <= boundary
    if within:
        selected = "incremental"
        reason = (
            "work difference inside explicit 10% tolerance band; "
            "stable rule selects incremental"
        )
        tolerance_effect = (
            f"tolerance band: |full-incremental|={abs(diff)} <= boundary={boundary}; "
            "tie broken deterministically toward incremental"
        )
    elif incremental < full:
        selected = "incremental"
        reason = "incremental estimated work strictly below full path"
        tolerance_effect = "not in tolerance band; strict cost comparison"
    else:
        selected = "full"
        reason = "full path estimated work below incremental cost"
        tolerance_effect = "not in tolerance band; strict cost comparison"
    boundaries = {
        "tolerance_ratio": f"{TOLERANCE_RATIO_NUM}/{TOLERANCE_RATIO_DEN}",
        "boundary_work_units": boundary,
        "full_leq_incremental_threshold": full,
    }
    return PathDecision(
        selected=selected,
        reason=reason,
        scale=scale,
        full_work=full,
        incremental_work=incremental,
        threshold_boundary=boundary,
        within_tolerance=within,
        tolerance_effect=tolerance_effect,
        not_triggered_boundaries=boundaries,
    )
