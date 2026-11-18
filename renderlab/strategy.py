"""Stable, data-scale-driven strategy selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import BUILTIN_STRATEGY_VERSION, RenderConfig, ResolutionTier
from .image import LATENT_CHANNELS
from .util import canonical_json, sha256_hex


SCALE_BUDGET_WORK_UNITS = 1_500_000.0
LATENCY_REFERENCE_WORK_UNITS = 250_000.0


@dataclass(frozen=True)
class StrategyPolicy:
    policy_id: str
    tier_mode: str
    batch_mode: str
    step_fraction: float
    chunk_latents: bool
    quality_bias: float
    latency_bias: float
    memory_bias: float
    rationale: str


STRATEGY_TABLE: tuple[StrategyPolicy, ...] = (
    StrategyPolicy("S0-quality-full", "all", "one", 1.00, False, 1.40, 0.15, 0.10, "maximise quality and all tiers"),
    StrategyPolicy("S1-balanced", "all", "requested", 1.00, True, 1.70, 0.70, 0.55, "balance quality with bounded cache residency"),
    StrategyPolicy("S2-throughput", "all", "max", 1.00, True, 0.95, 0.90, 0.35, "amortise prompt batches over all tiers"),
    StrategyPolicy("S3-low-latency", "lowest", "requested", 0.75, True, 0.40, 1.10, 0.95, "reduce tier and step work for large prompt sets"),
    StrategyPolicy("S4-cache-safe", "lowest", "one", 1.00, True, 0.30, 0.45, 1.00, "minimise resident blocks under a tight cache budget"),
)


@dataclass
class StrategyDecision:
    version: str
    metrics: dict[str, Any]
    candidates: list[dict[str, Any]]
    selected: dict[str, Any]
    reason: str

    @property
    def enabled_tiers(self) -> list[str]:
        return list(self.selected["enabled_tiers"])

    @property
    def batch_prompt_count(self) -> int:
        return int(self.selected["batch_prompt_count"])

    @property
    def denoise_steps(self) -> int:
        return int(self.selected["denoise_steps"])

    @property
    def chunk_latents(self) -> bool:
        return bool(self.selected["chunk_latents"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metrics": self.metrics,
            "candidates": self.candidates,
            "selected": self.selected,
            "reason": self.reason,
        }

    def fingerprint(self) -> str:
        return sha256_hex("strategy-decision-v1", self.to_dict())


def _round_steps(total_steps: int, fraction: float) -> int:
    return max(1, min(total_steps, int(math.floor(total_steps * fraction + 0.5))))


def select_strategy(
    render: RenderConfig,
    *,
    prompt_count: int,
    seed_count: int,
    requested_batch_size: int,
    tier_order: list[str] | None = None,
) -> StrategyDecision:
    """Select a policy from the versioned table using only scale metrics.

    Ties are resolved by policy id.  The score components and all input
    metrics are serialized into the decision record.
    """

    if prompt_count <= 0 or seed_count <= 0:
        raise ValueError("prompt_count and seed_count must be positive")
    if requested_batch_size not in render.batch_sizes:
        raise ValueError("requested_batch_size must be one of render.batch_sizes")

    tiers: list[ResolutionTier] = list(render.resolution_tiers)
    if tier_order:
        by_name = {tier.name: tier for tier in tiers}
        tiers = [by_name[name] for name in tier_order]
    max_area = max(tier.area for tier in tiers)
    frame_count = prompt_count * seed_count
    tier_costs = [
        {
            "tier": tier.name,
            "width": tier.width,
            "height": tier.height,
            "area": tier.area,
            "estimated_frame_cost_units": int(
                tier.area * LATENT_CHANNELS * render.denoise_steps + tier.area * 3
            ),
        }
        for tier in tiers
    ]
    cost_by_name = {item["tier"]: item["estimated_frame_cost_units"] for item in tier_costs}

    metrics = {
        "prompt_count": prompt_count,
        "seed_count": seed_count,
        "frame_count": frame_count,
        "requested_batch_size": requested_batch_size,
        "batch_size_options": list(render.batch_sizes),
        "resolution_tiers": tier_costs,
        "cache_budget_blocks": render.cache_budget_blocks,
        "denoise_steps": render.denoise_steps,
    }

    candidates: list[dict[str, Any]] = []
    for policy in STRATEGY_TABLE:
        selected_tiers = tiers if policy.tier_mode == "all" else tiers[:1]
        steps = _round_steps(render.denoise_steps, policy.step_fraction)
        if policy.batch_mode == "one":
            batch = 1
        elif policy.batch_mode == "max":
            batch = max(render.batch_sizes)
        else:
            batch = requested_batch_size
        batch = max(1, min(batch, prompt_count))

        area_ratio = (
            sum(tier.area for tier in selected_tiers) / len(selected_tiers) / max_area
        )
        step_ratio = steps / render.denoise_steps
        quality_score = 0.6 * area_ratio + 0.4 * step_ratio

        work_per_frame = 0.0
        blocks_per_frame = 0
        for tier in selected_tiers:
            area = tier.area
            work_per_frame += area * LATENT_CHANNELS * steps + area * 3
            blocks_per_frame += math.ceil(area * LATENT_CHANNELS / render.latent_block_size)
        batches = math.ceil(prompt_count / batch)
        work_units = batches * seed_count * work_per_frame
        pressure = (blocks_per_frame * steps * batch) / render.cache_budget_blocks

        latency_score = 1.0 / (1.0 + work_units / LATENCY_REFERENCE_WORK_UNITS)
        memory_score = 1.0 / (1.0 + pressure)
        overrun = max(0.0, work_units / SCALE_BUDGET_WORK_UNITS - 1.0)
        overrun_penalty = 0.37 * overrun
        total = (
            policy.quality_bias * quality_score
            + policy.latency_bias * latency_score
            + policy.memory_bias * memory_score
            - overrun_penalty
        )
        candidate = {
            "policy_id": policy.policy_id,
            "enabled_tiers": [tier.name for tier in selected_tiers],
            "batch_prompt_count": batch,
            "denoise_steps": steps,
            "chunk_latents": policy.chunk_latents,
            "rationale": policy.rationale,
            "metrics": {
                "estimated_work_units": round(work_units, 6),
                "estimated_cache_pressure": round(pressure, 6),
                "area_ratio": round(area_ratio, 12),
                "step_ratio": round(step_ratio, 12),
                "scale_overrun": round(overrun, 12),
            },
            "scores": {
                "quality": round(quality_score, 12),
                "latency": round(latency_score, 12),
                "memory": round(memory_score, 12),
                "overrun_penalty": round(overrun_penalty, 12),
                "total": round(total, 12),
            },
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item["scores"]["total"], item["policy_id"]))
    selected = candidates[0]
    reason = (
        f"selected {selected['policy_id']} with stable score {selected['scores']['total']:.12f}; "
        f"prompt_count={prompt_count}, seed_count={seed_count}, batch={selected['batch_prompt_count']}, "
        f"tiers={','.join(selected['enabled_tiers'])}, steps={selected['denoise_steps']}, "
        f"chunk_latents={str(selected['chunk_latents']).lower()}"
    )
    return StrategyDecision(
        version=BUILTIN_STRATEGY_VERSION,
        metrics=metrics,
        candidates=candidates,
        selected=selected,
        reason=reason,
    )
