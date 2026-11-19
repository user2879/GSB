"""Truth-set evaluation for each fixed-point precision."""

from __future__ import annotations

from typing import Any

from .models import AppConfig, Dataset


def evaluate_user(fusion_result: dict[str, Any], dataset: Dataset, config: AppConfig) -> dict[str, Any]:
    user_id = fusion_result["user_id"]
    final_order = list(fusion_result["final_order"])
    truth = dataset.truth.get(user_id)
    if truth is None:
        return {
            "user_id": user_id,
            "available": False,
            "hit_rate": 0.0,
            "precision_at_k": 0.0,
            "mean_position_deviation": None,
            "top_k_overlap": 0,
            "truth_candidate_count": 0,
            "final_top_k": final_order[: config.top_k],
            "missed_truth_candidates": [],
        }

    final_top_k = final_order[: config.top_k]
    truth_candidates = list(truth.candidates)
    truth_ranking = list(truth.ranking)
    candidate_hits = sorted(set(final_top_k).intersection(truth_candidates))
    hit_rate = len(candidate_hits) / len(truth_candidates) if truth_candidates else 0.0
    precision_at_k = len(candidate_hits) / len(final_top_k) if final_top_k else 0.0

    final_position = {item_id: position for position, item_id in enumerate(final_order, start=1)}
    truth_position = {item_id: position for position, item_id in enumerate(truth_ranking, start=1)}
    matched = sorted(set(final_position).intersection(truth_position))
    deviations = [abs(final_position[item_id] - truth_position[item_id]) for item_id in matched]
    mean_position_deviation = sum(deviations) / len(deviations) if deviations else None

    truth_top_k = truth_ranking[: config.top_k]
    top_k_overlap = len(set(final_top_k).intersection(truth_top_k))
    fusion_candidates = {candidate["item_id"]: candidate for candidate in fusion_result["candidates"]}
    missed: list[dict[str, Any]] = []
    for item_id in truth_candidates:
        if item_id in final_top_k:
            continue
        candidate = fusion_candidates.get(item_id)
        missed.append(
            {
                "item_id": item_id,
                "truth_rank": truth_position.get(item_id),
                "final_rank": final_position.get(item_id),
                "recalled": candidate is not None,
                "channel_ranks": None
                if candidate is None
                else {entry["channel"]: entry["source_rank"] for entry in candidate["channel_contributions"]},
                "channel_contributions": None
                if candidate is None
                else {
                    entry["channel"]: entry["contribution"] for entry in candidate["channel_contributions"]
                },
                "final_score": None if candidate is None else candidate["final_score"],
            }
        )
    return {
        "user_id": user_id,
        "available": True,
        "hit_rate": hit_rate,
        "precision_at_k": precision_at_k,
        "mean_position_deviation": mean_position_deviation,
        "top_k_overlap": top_k_overlap,
        "truth_candidate_count": len(truth_candidates),
        "final_top_k": final_top_k,
        "truth_top_k": truth_top_k,
        "candidate_hits": candidate_hits,
        "missed_truth_candidates": missed,
    }


def summarize_evaluation(users: list[dict[str, Any]], config: AppConfig, bits: int | None = None) -> dict[str, Any]:
    available = [user for user in users if user["available"]]
    deviations = [user["mean_position_deviation"] for user in available if user["mean_position_deviation"] is not None]
    missed = sum(len(user["missed_truth_candidates"]) for user in available)
    return {
        "bits": config.primary_bits if bits is None else bits,
        "users_evaluated": len(available),
        "mean_hit_rate": sum(user["hit_rate"] for user in available) / len(available) if available else 0.0,
        "mean_precision_at_k": sum(user["precision_at_k"] for user in available) / len(available) if available else 0.0,
        "mean_position_deviation": sum(deviations) / len(deviations) if deviations else None,
        "mean_top_k_overlap": sum(user["top_k_overlap"] for user in available) / len(available) if available else 0.0,
        "missed_truth_candidate_count": missed,
        "top_k": config.top_k,
    }
