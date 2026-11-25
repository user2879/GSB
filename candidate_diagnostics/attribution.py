from __future__ import annotations

from typing import Any

from .config import Config, deep_merge
from .models import Request
from .pipeline import RequestResult, analyze_request


def make_reference_config(config: Config) -> Config:
    raw = deep_merge(config.raw, {})
    quant = raw["quantization"]
    quant["fixed_decimals"] = int(quant.get("reference_fixed_decimals", quant["fixed_decimals"] + 1))
    quant["gear_exponent"] = 0
    quant["bit_width"] = int(quant.get("reference_bit_width", quant["bit_width"] + 1))
    return Config(raw, source=f"{config.source}#reference-high-width")


def _position_map(result: RequestResult) -> dict[int, int]:
    return {cid: index for index, cid in enumerate(result.shuffle_order)}


def attribute_request(
    request: Request,
    coarse_config: Config,
    reference_config: Config | None = None,
) -> dict[str, Any]:
    reference_config = reference_config or make_reference_config(coarse_config)
    coarse = analyze_request(request, coarse_config)
    reference = analyze_request(request, reference_config)
    tolerance = coarse_config.raw["tolerance"]

    coarse_scores = {item["candidate_id"]: item for item in coarse.scores}
    ref_scores = {item["candidate_id"]: item for item in reference.scores}
    normalization_deltas = []
    max_norm_delta = 0.0
    for cid in sorted(coarse_scores):
        delta = abs(
            float(coarse_scores[cid]["dequant_score"])
            - float(ref_scores[cid]["dequant_score"])
        )
        max_norm_delta = max(max_norm_delta, delta)
        normalization_deltas.append(
            {
                "candidate_id": cid,
                "coarse_dequant": coarse_scores[cid]["dequant_score"],
                "reference_dequant": ref_scores[cid]["dequant_score"],
                "abs_delta": delta,
            }
        )

    coarse_top = set(coarse.top_k_ids)
    ref_top = set(reference.top_k_ids)
    topk_flips = sorted(coarse_top.symmetric_difference(ref_top))
    rank_displacements = []
    coarse_rank = {cid: i + 1 for i, cid in enumerate(coarse.ranked_ids)}
    ref_rank = {cid: i + 1 for i, cid in enumerate(reference.ranked_ids)}
    max_rank_delta = 0
    for cid in sorted(set(coarse_rank) & set(ref_rank)):
        displacement = abs(coarse_rank[cid] - ref_rank[cid])
        max_rank_delta = max(max_rank_delta, displacement)
        rank_displacements.append(
            {
                "candidate_id": cid,
                "coarse_rank": coarse_rank[cid],
                "reference_rank": ref_rank[cid],
                "abs_rank_delta": displacement,
            }
        )

    quota_diffs: list[dict[str, Any]] = []
    max_quota_delta = 0
    channels = sorted(set(coarse.layers) | set(reference.layers))
    for channel in channels:
        coarse_quota = int(coarse.layers.get(channel, {}).get("quota", 0))
        ref_quota = int(reference.layers.get(channel, {}).get("quota", 0))
        delta = abs(coarse_quota - ref_quota)
        max_quota_delta = max(max_quota_delta, delta)
        quota_diffs.append(
            {
                "channel": channel,
                "coarse_quota": coarse_quota,
                "reference_quota": ref_quota,
                "abs_delta": delta,
            }
        )

    coarse_sample = set(coarse.sampled_ids)
    ref_sample = set(reference.sampled_ids)
    sample_flips = sorted(coarse_sample.symmetric_difference(ref_sample))

    coarse_positions = _position_map(coarse)
    ref_positions = _position_map(reference)
    shuffle_deltas = []
    max_shuffle_delta = 0
    for cid in sorted(set(coarse_positions) | set(ref_positions)):
        coarse_pos = coarse_positions.get(cid)
        ref_pos = ref_positions.get(cid)
        delta = abs((coarse_pos or 10**9) - (ref_pos or 10**9))
        if coarse_pos is not None and ref_pos is not None:
            max_shuffle_delta = max(max_shuffle_delta, delta)
        shuffle_deltas.append(
            {
                "candidate_id": cid,
                "coarse_position": coarse_pos,
                "reference_position": ref_pos,
                "abs_delta": None if coarse_pos is None or ref_pos is None else delta,
            }
        )

    flip_attribution = _classify_flips(
        coarse=coarse,
        reference=reference,
        topk_flips=topk_flips,
        sample_flips=sample_flips,
        quota_diffs=quota_diffs,
        normalization_deltas=normalization_deltas,
        normalization_tolerance=float(tolerance["normalization_dequant"]),
    )

    stages = {
        "normalization": {
            "max_abs_dequant_delta": max_norm_delta,
            "tolerance": float(tolerance["normalization_dequant"]),
            "exceeded": max_norm_delta > float(tolerance["normalization_dequant"]),
            "candidate_deltas": normalization_deltas,
        },
        "truncation_boundary": {
            "topk_flips": topk_flips,
            "max_rank_delta": max_rank_delta,
            "tolerance_rank": int(tolerance["truncation_rank"]),
            "exceeded": len(topk_flips) > int(tolerance["truncation_rank"]),
            "cutoff": {
                "coarse": coarse.cutoff_score,
                "reference": reference.cutoff_score,
            },
            "boundary": {"coarse": coarse.boundary, "reference": reference.boundary},
            "rank_displacements": rank_displacements,
        },
        "stratified_sampling": {
            "sample_flips": sample_flips,
            "max_quota_delta": max_quota_delta,
            "tolerance_quota": int(tolerance["sampling_quota"]),
            "tolerance_flip_count": int(tolerance.get("sampling_flip_count", 0)),
            "exceeded": (
                max_quota_delta > int(tolerance["sampling_quota"])
                or len(sample_flips) > int(tolerance.get("sampling_flip_count", 0))
            ),
            "quota_diffs": quota_diffs,
        },
        "shuffle": {
            "max_position_delta": max_shuffle_delta,
            "tolerance_position": int(tolerance["shuffle_position"]),
            "exceeded": max_shuffle_delta > int(tolerance["shuffle_position"]),
            "position_deltas": shuffle_deltas,
            "coarse_order": coarse.shuffle_order,
            "reference_order": reference.shuffle_order,
        },
    }
    first_exceeded = next((name for name, stage in stages.items() if stage["exceeded"]), None)
    return {
        "request_id": request.request_id,
        "coarse_config_hash": coarse_config.config_hash,
        "reference_config_hash": reference_config.config_hash,
        "first_exceeded_stage": first_exceeded,
        "stages": stages,
        "flip_attribution": flip_attribution,
    }


def _classify_flips(
    *,
    coarse: RequestResult,
    reference: RequestResult,
    topk_flips: list[int],
    sample_flips: list[int],
    quota_diffs: list[dict[str, Any]],
    normalization_deltas: list[dict[str, Any]],
    normalization_tolerance: float,
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    coarse_scores = {item["candidate_id"]: item for item in coarse.scores}
    ref_scores = {item["candidate_id"]: item for item in reference.scores}
    quota_changed_channels = {item["channel"] for item in quota_diffs if item["abs_delta"]}
    delta_by_candidate = {
        item["candidate_id"]: item["abs_delta"] for item in normalization_deltas
    }

    for cid in topk_flips:
        coarse_score = coarse_scores[cid]
        ref_score = ref_scores[cid]
        coarse_cut = coarse.cutoff_score
        coarse_tied = cid in coarse.boundary.get("tied_ids", [])
        boundary_disagreement_ids = (
            set(coarse.boundary.get("boundary_selected_ids", [])) - set(reference.top_k_ids)
        ) | set(coarse.boundary.get("boundary_rejected_ids", []))
        boundary_tie_decisive = (
            coarse_tied
            and coarse.boundary.get("tie_at_cutoff")
            and cid in boundary_disagreement_ids
        )
        quant_delta = float(delta_by_candidate.get(cid, 0.0))
        if quant_delta > normalization_tolerance:
            cause = "normalization_quantization"
            evidence = (
                f"dequant delta {quant_delta} exceeds tolerance "
                f"{normalization_tolerance}; coarse grid moves candidate onto cutoff tie"
            )
        elif boundary_tie_decisive:
            cause = "boundary_comparison"
            evidence = (
                "normalization delta is within tolerance; coarse fixed-point equality at the "
                "cutoff makes ascending candidate_id boundary comparison decisive"
            )
        else:
            cause = "normalization_quantization"
            evidence = (
                "membership crosses cutoff because quantized/dequantized score moved; "
                f"coarse_cutoff={coarse_cut}"
            )
        explanations.append(
            {
                "candidate_id": cid,
                "stage": "truncation",
                "cause": cause,
                "coarse_in_topk": cid in set(coarse.top_k_ids),
                "reference_in_topk": cid in set(reference.top_k_ids),
                "coarse_dequant": coarse_score["dequant_score"],
                "reference_dequant": ref_score["dequant_score"],
                "normalization_abs_delta": quant_delta,
                "evidence": evidence,
            }
        )

    for cid in sample_flips:
        if cid in topk_flips:
            continue
        channel = coarse_scores.get(cid, {}).get("channel")
        if channel in quota_changed_channels:
            cause = "sampling_quota_rounding"
            evidence = "TopK membership is unchanged but layer quota allocation/rounding changed"
        else:
            cause = "seeded_draw_or_shuffle"
            evidence = "layer quota is unchanged; deterministic seed/draw ranking changed"
        explanations.append(
            {
                "candidate_id": cid,
                "stage": "sampling",
                "cause": cause,
                "coarse_sampled": cid in set(coarse.sampled_ids),
                "reference_sampled": cid in set(reference.sampled_ids),
                "channel": channel,
                "evidence": evidence,
            }
        )
    return explanations


def attribute_many(requests: dict[str, Request], config: Config) -> dict[str, Any]:
    reference_config = make_reference_config(config)
    reports = [
        attribute_request(requests[rid], config, reference_config) for rid in sorted(requests)
    ]
    first_stage_counts: dict[str, int] = {}
    for report in reports:
        stage = report["first_exceeded_stage"] or "none_within_tolerance"
        first_stage_counts[stage] = first_stage_counts.get(stage, 0) + 1
    return {
        "reference_config_hash": reference_config.config_hash,
        "reports": reports,
        "first_exceeded_stage_counts": dict(sorted(first_stage_counts.items())),
    }
