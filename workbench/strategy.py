"""Scale-adaptive strategy selector.

The selector enumerates every legal path/quantization combination from the
fixed policy table, scores it against the latency budget and quality weight,
and emits an auditable decision record.  Replaying the record reproduces the
same combination because all inputs are content fingerprints / integers.
"""

from __future__ import annotations

from itertools import product

from .algorithms import (
    INPAINT_COST,
    MASK_COST,
    STYLE_COST,
    estimate_cost_units,
)

MASK_PATHS = ("threshold", "edge")
INPAINT_PATHS = ("neighbor_mean", "directional")
STYLE_PATHS = ("lut", "linear")

# Deterministic quality table; higher means better visual fidelity.
MASK_QUALITY = {"threshold": 0.85, "edge": 0.95}
INPAINT_QUALITY = {"neighbor_mean": 0.8, "directional": 0.95}
STYLE_QUALITY = {"lut": 0.85, "linear": 0.95}

# One cost unit equals this many milliseconds of estimated interaction latency.
MS_PER_UNIT = 0.01


def _unique_constraint(values):
    distinct = sorted({value for value in values if value})
    return distinct[0] if len(distinct) == 1 else None


def instruction_constraints(ops: list) -> dict:
    return {
        "mask": _unique_constraint(op.get("mask_hint") for op in ops),
        "inpaint": _unique_constraint(
            op.get("repair") for op in ops if op.get("repair") != "auto"
        ),
    }


def quality_score(mask_path, inpaint_path, style_path, quant_levels):
    quant_quality = quant_levels / 256.0
    if style_path == "linear":
        style_quality = STYLE_QUALITY["linear"]
    else:
        style_quality = 0.7 + 0.3 * quant_quality
    score = (
        MASK_QUALITY[mask_path]
        + INPAINT_QUALITY[inpaint_path]
        + style_quality
    ) / 3.0
    return round(score, 6)


def select_strategy(
    *,
    sample_count: int,
    primitives: int,
    total_regions: int,
    pixels: int,
    regions_per_sample: int,
    instruction_length: int,
    quant_levels: tuple,
    latency_budget_ms: float,
    quality_weight: float,
    constraints: dict,
) -> dict:
    budget_units = latency_budget_ms / MS_PER_UNIT
    candidates = []
    mask_paths = MASK_PATHS
    inpaint_paths = INPAINT_PATHS
    if constraints.get("mask"):
        mask_paths = (constraints["mask"],)
    if constraints.get("inpaint"):
        inpaint_paths = (constraints["inpaint"],)

    for mask_path, inpaint_path, style_path, quant in product(
        mask_paths, inpaint_paths, STYLE_PATHS, quant_levels
    ):
        cost = estimate_cost_units(
            pixels=pixels,
            regions=regions_per_sample,
            instruction_length=instruction_length,
            mask_path=mask_path,
            inpaint_path=inpaint_path,
            style_path=style_path,
            quant_levels=quant,
        )
        est_ms = round(cost * MS_PER_UNIT, 4)
        quality = quality_score(mask_path, inpaint_path, style_path, quant)
        overrun = max(0.0, (cost - budget_units) / budget_units)
        score = round(quality * quality_weight - overrun * 1000.0, 6)
        candidates.append(
            {
                "name": f"{mask_path}|{inpaint_path}|{style_path}|q{quant}",
                "mask_path": mask_path,
                "inpaint_path": inpaint_path,
                "style_path": style_path,
                "quant_levels": quant,
                "cost_units": cost,
                "estimated_ms": est_ms,
                "quality": quality,
                "over_budget": cost > budget_units,
                "score": score,
            }
        )

    feasible = [candidate for candidate in candidates if not candidate["over_budget"]]
    pool = feasible or candidates
    # Deterministic tie-break: highest score, then lowest cost, then name.
    chosen = sorted(pool, key=lambda c: (-c["score"], c["cost_units"], c["name"]))[0]
    reason = (
        "highest score within latency budget"
        if feasible
        else "all candidates exceed budget; cheapest candidate selected as fallback"
    )
    return {
        "metrics": {
            "sample_count": sample_count,
            "primitives": primitives,
            "total_regions": total_regions,
            "regions_per_sample": regions_per_sample,
            "instruction_length": instruction_length,
            "pixels_per_sample": pixels,
            "latency_budget_ms": latency_budget_ms,
            "budget_units": round(budget_units, 3),
            "quality_weight": quality_weight,
        },
        "constraints": constraints,
        "candidates": candidates,
        "selected": {key: chosen[key] for key in (
            "name",
            "mask_path",
            "inpaint_path",
            "style_path",
            "quant_levels",
            "cost_units",
            "estimated_ms",
            "quality",
            "score",
        )},
        "reason": reason,
        "table_fingerprint": "policy-table-v1",
        "path_costs": {
            "mask": dict(MASK_COST),
            "inpaint": dict(INPAINT_COST),
            "style": dict(STYLE_COST),
        },
    }
