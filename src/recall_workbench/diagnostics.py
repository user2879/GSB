"""Per-stage fixed-point error propagation diagnostics."""

from __future__ import annotations

from typing import Any

from .fixed import FixedFormat, fixed_difference
from .models import AppConfig, CHANNEL_ORDER


STAGE_ORDER = ("normalization", "channel_weighted_sum", "feature_add", "clipping")


def _candidate_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["item_id"]: candidate for candidate in result["candidates"]}


def _stage_values(
    primary: dict[str, Any],
    reference: dict[str, Any],
    item_id: str,
    primary_fmt: FixedFormat,
    reference_fmt: FixedFormat,
) -> dict[str, Any]:
    primary_candidate = _candidate_map(primary)[item_id]
    reference_candidate = _candidate_map(reference)[item_id]
    normalization_difference = 0.0
    normalization_by_channel: dict[str, float] = {}
    for name in CHANNEL_ORDER:
        primary_channel = primary["channels"][name]
        reference_channel = reference["channels"][name]
        primary_entries = {entry["item_id"]: entry for entry in primary_channel["normalized_candidates"]}
        reference_entries = {entry["item_id"]: entry for entry in reference_channel["normalized_candidates"]}
        channel_difference = 0.0
        if item_id not in primary_entries and item_id not in reference_entries:
            channel_difference = 0.0
        else:
            primary_fixed = int(primary_entries.get(item_id, {}).get("normalized_fixed", 0))
            reference_fixed = int(reference_entries.get(item_id, {}).get("normalized_fixed", 0))
            difference = fixed_difference(primary_fixed, primary_fmt, reference_fixed, reference_fmt)
            channel_difference = max(channel_difference, difference)
        normalization_by_channel[name] = channel_difference
        normalization_difference = max(normalization_difference, channel_difference)
    return {
        "normalization": normalization_difference,
        "_normalization_by_channel": normalization_by_channel,
        "channel_weighted_sum": fixed_difference(
            int(primary_candidate["fused_fixed"]), primary_fmt, int(reference_candidate["fused_fixed"]), reference_fmt
        ),
        "feature_add": fixed_difference(
            int(primary_candidate["preclip_fixed"]), primary_fmt, int(reference_candidate["preclip_fixed"]), reference_fmt
        ),
        "clipping": fixed_difference(
            int(primary_candidate["final_fixed"]), primary_fmt, int(reference_candidate["final_fixed"]), reference_fmt
        ),
    }


def compare_precisions(primary: dict[str, Any], reference: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    primary_fmt = FixedFormat(config.primary_bits, config.rounding)
    reference_fmt = FixedFormat(config.reference_bits, config.rounding)
    primary_candidates = _candidate_map(primary)
    reference_candidates = _candidate_map(reference)
    item_ids = sorted(set(primary_candidates) | set(reference_candidates))

    primary_final_rank = {item_id: rank for rank, item_id in enumerate(primary["final_order"], start=1)}
    reference_final_rank = {item_id: rank for rank, item_id in enumerate(reference["final_order"], start=1)}
    primary_weighted_rank = {item_id: rank for rank, item_id in enumerate(primary["weighted_order"], start=1)}
    reference_weighted_rank = {item_id: rank for rank, item_id in enumerate(reference["weighted_order"], start=1)}

    candidate_diagnostics: list[dict[str, Any]] = []
    for item_id in item_ids:
        primary_candidate = primary_candidates.get(item_id)
        reference_candidate = reference_candidates.get(item_id)
        if primary_candidate is None or reference_candidate is None:
            candidate_diagnostics.append(
                {
                    "item_id": item_id,
                    "primary_rank": primary_final_rank.get(item_id),
                    "reference_rank": reference_final_rank.get(item_id),
                    "rank_changed": primary_final_rank.get(item_id) != reference_final_rank.get(item_id),
                    "position_change_cause": "candidate_set_mismatch",
                    "stage_differences": {stage: None for stage in STAGE_ORDER},
                    "first_exceeds_tolerance": None,
                }
            )
            continue

        stages = _stage_values(primary, reference, item_id, primary_fmt, reference_fmt)
        first_exceeds = next((stage for stage in STAGE_ORDER if stages[stage] > config.tolerance), None)
        rank_changed = primary_final_rank[item_id] != reference_final_rank[item_id]
        if not rank_changed:
            cause = None
        elif primary_weighted_rank[item_id] != reference_weighted_rank[item_id]:
            cause = "normalization_residual"
        else:
            cause = "feature_rounding_accumulation"
        candidate_diagnostics.append(
            {
                "item_id": item_id,
                "rank_vector": primary_candidate["rank_vector"],
                "primary_rank": primary_final_rank[item_id],
                "reference_rank": reference_final_rank[item_id],
                "primary_weighted_rank": primary_weighted_rank[item_id],
                "reference_weighted_rank": reference_weighted_rank[item_id],
                "rank_changed": rank_changed,
                "position_change_cause": cause,
                "stage_differences": {stage: stages[stage] for stage in STAGE_ORDER},
                "normalization_difference_by_channel": stages["_normalization_by_channel"],
                "stage_values": {
                    "primary": {
                        "fused_score": primary_candidate["fused_score"],
                        "preclip_score": primary_candidate["preclip_score"],
                        "final_score": primary_candidate["final_score"],
                    },
                    "reference": {
                        "fused_score": reference_candidate["fused_score"],
                        "preclip_score": reference_candidate["preclip_score"],
                        "final_score": reference_candidate["final_score"],
                    },
                },
                "first_exceeds_tolerance": first_exceeds,
            }
        )

    primary_order = primary["final_order"]
    reference_order = reference["final_order"]
    rank_order = {item_id: index for index, item_id in enumerate(primary_order)}
    candidate_diagnostics.sort(key=lambda candidate: (rank_order.get(candidate["item_id"], 10**9), candidate["item_id"]))
    changed = [candidate for candidate in candidate_diagnostics if candidate["rank_changed"]]
    return {
        "user_id": primary["user_id"],
        "primary_bits": config.primary_bits,
        "reference_bits": config.reference_bits,
        "tolerance": config.tolerance,
        "candidate_count": len(candidate_diagnostics),
        "first_exceeding_count": sum(1 for candidate in candidate_diagnostics if candidate["first_exceeds_tolerance"]),
        "rank_changed_count": len(changed),
        "rank_changes": [
            {
                "item_id": candidate["item_id"],
                "primary_rank": candidate["primary_rank"],
                "reference_rank": candidate["reference_rank"],
                "cause": candidate["position_change_cause"],
            }
            for candidate in changed
        ],
        "candidates": candidate_diagnostics,
        "final_order_primary": primary_order,
        "final_order_reference": reference_order,
    }


def boundary_sensitive_candidates(
    primary: dict[str, Any], reference: dict[str, Any], comparison: dict[str, Any], config: AppConfig
) -> list[dict[str, Any]]:
    """Find near-boundary candidates and candidates whose rank changes by width."""

    changed_ids = {change["item_id"] for change in comparison["rank_changes"]}
    entries: list[dict[str, Any]] = []
    for result, label in ((primary, "primary"), (reference, "reference")):
        ordered = [next(c for c in result["candidates"] if c["item_id"] == item_id) for item_id in result["final_order"]]
        for index, candidate in enumerate(ordered):
            neighbor_margin = None
            if index + 1 < len(ordered):
                neighbor_margin = abs(candidate["final_score"] - ordered[index + 1]["final_score"])
                if candidate["final_fixed"] == ordered[index + 1]["final_fixed"]:
                    neighbor_margin = 0.0
            near_boundary = neighbor_margin is not None and neighbor_margin <= config.tolerance
            if candidate["item_id"] in changed_ids or near_boundary:
                reasons = []
                if candidate["item_id"] in changed_ids:
                    reasons.append("rank_changed_between_widths")
                if near_boundary:
                    reasons.append("adjacent_margin_within_tolerance")
                entries.append(
                    {
                        "user_id": result["user_id"],
                        "item_id": candidate["item_id"],
                        "precision_role": label,
                        "bits": result["bits"],
                        "rank": index + 1,
                        "final_score": candidate["final_score"],
                        "next_rank_margin": neighbor_margin,
                        "reasons": reasons,
                    }
                )
    return entries
