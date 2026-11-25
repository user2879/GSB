"""Explicit scale rules for full vs incremental arbitration selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FULL = "full"
INCREMENTAL = "incremental"


@dataclass
class ScaleParameters:
    env_count: int
    key_count: int
    observation_count: int
    changed_observation_count: int
    affected_key_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "env_count": self.env_count,
            "key_count": self.key_count,
            "observation_count": self.observation_count,
            "changed_observation_count": self.changed_observation_count,
            "affected_key_count": self.affected_key_count,
        }


@dataclass
class PathDecision:
    path: str
    reason: str
    parameters: ScaleParameters
    full_effort: int
    incremental_effort: int
    effort_delta: int
    tolerance_band: int
    tolerance_influenced: bool
    boundary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "parameters": self.parameters.as_dict(),
            "full_effort": self.full_effort,
            "incremental_effort": self.incremental_effort,
            "effort_delta": self.effort_delta,
            "tolerance_band": self.tolerance_band,
            "tolerance_influenced": self.tolerance_influenced,
            "boundary": self.boundary,
        }


def estimate_full(params: ScaleParameters) -> int:
    """Full work touches every observation for every env and key."""
    return params.env_count * params.key_count * params.observation_count


def estimate_incremental(params: ScaleParameters) -> int:
    """Incremental work scans changes once and recomputes only affected keys."""
    scan = params.env_count * params.changed_observation_count
    recompute = params.env_count * params.affected_key_count * max(
        1, params.changed_observation_count
    )
    return scan + recompute


def choose_path(params: ScaleParameters, tolerance_ratio: float = 0.1) -> PathDecision:
    """Select a path under explicit, documented scale rules.

    Boundary rules (checked first):

    * no changes -> incremental no-op (zero affected keys);
    * every key affected -> full is always preferred (no unaffected subset);
    * tiny inputs under 256 observations -> full for simplest reproducibility.

    Otherwise estimate both efforts. When the absolute difference lies inside
    an explicit tolerance band (``tolerance_ratio`` of the full effort, with a
    floor of one unit), the stable tie-break prefers ``full`` and flags the
    decision as tolerance influenced.
    """
    full_effort = estimate_full(params)
    inc_effort = estimate_incremental(params)
    delta = abs(full_effort - inc_effort)
    band = max(1, int(round(full_effort * tolerance_ratio)))

    if params.changed_observation_count == 0 or params.affected_key_count == 0:
        return PathDecision(
            INCREMENTAL,
            "empty changeset: incremental no-op preserves all prior verdicts",
            params, full_effort, inc_effort, delta, band, False,
            boundary={"rule": "empty_changeset", "changed": 0, "affected_keys": 0},
        )
    if params.observation_count <= 256:
        return PathDecision(
            FULL,
            "observation count at or below the small-input boundary 256",
            params, full_effort, inc_effort, delta, band, False,
            boundary={"rule": "small_input", "threshold": 256,
                      "observation_count": params.observation_count},
        )
    if params.affected_key_count >= params.key_count:
        return PathDecision(
            FULL,
            "all keys affected: no unaffected subset exists to preserve",
            params, full_effort, inc_effort, delta, band, False,
            boundary={"rule": "all_keys_affected",
                      "affected": params.affected_key_count,
                      "key_count": params.key_count},
        )

    if delta <= band:
        return PathDecision(
            FULL,
            "effort difference inside tolerance band; stable tie-break -> full",
            params, full_effort, inc_effort, delta, band, True,
            boundary={"rule": "tolerance_tie", "delta": delta, "band": band},
        )
    if inc_effort < full_effort:
        return PathDecision(
            INCREMENTAL,
            "incremental effort strictly below full effort outside the band",
            params, full_effort, inc_effort, delta, band, False,
            boundary={"rule": "strict_delta", "delta": delta, "band": band},
        )
    return PathDecision(
        FULL,
        "full effort strictly below incremental effort outside the band",
        params, full_effort, inc_effort, delta, band, False,
        boundary={"rule": "strict_delta", "delta": delta, "band": band},
    )
