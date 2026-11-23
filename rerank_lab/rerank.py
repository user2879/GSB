"""Deterministic window diversity reranking with integer similarity tests."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from .canonical import decimal_text, digest, fraction_text, round_half_away
from .models import FeatureCatalog, LabConfig, RawCandidate


def _similarity(
    a: list[int],
    b: list[int],
    indices: tuple[int, ...],
    threshold: Fraction,
) -> dict[str, Any]:
    dot = 0
    norm_a = 0
    norm_b = 0
    for index in indices:
        left = a[index]
        right = b[index]
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if threshold == 0:
        meets = True
    elif dot <= 0 or norm_a == 0 or norm_b == 0:
        meets = False
    else:
        meets = dot * dot >= threshold * threshold * norm_a * norm_b
    if norm_a == 0 or norm_b == 0:
        cosine_decimal = "0"
    else:
        with localcontext() as ctx:
            ctx.prec = 50
            denominator = (Decimal(norm_a) * Decimal(norm_b)).sqrt()
            cosine_decimal = decimal_text(Decimal(dot) / denominator, places=18)
    return {
        "dot": dot,
        "norm_product": norm_a * norm_b,
        "threshold": fraction_text(threshold),
        "meets_threshold": meets,
        "cosine": cosine_decimal,
    }


def _sort_entries(entries: list[dict[str, Any]]) -> None:
    entries.sort(key=lambda item: (-item["score_fixed"], item["item_id"]))


def rerank_request(
    candidates: tuple[RawCandidate, ...],
    fusion: dict[str, Any],
    normalization_entry: dict[str, Any],
    catalog: FeatureCatalog,
    config: LabConfig,
) -> dict[str, Any]:
    """Apply the configured window and fixed integer diversity reductions."""

    normalized = normalization_entry["normalized_fixed"]
    initial_order = list(fusion["fusion_order"])
    window_ids = initial_order[: config.rerank_window_size]
    tail_ids = initial_order[config.rerank_window_size :]
    penalty_fixed = round_half_away(
        Fraction(config.diversity_penalty)
        * 10**config.fixed_decimal_places
        / config.quantization_step
    )
    lower_fixed = fusion["score_min_fixed"]
    upper_fixed = fusion["score_max_fixed"]
    threshold = config.similarity_threshold
    indices = catalog.similarity_indices

    similarity_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for left_index, left_id in enumerate(window_ids):
        for right_id in window_ids[left_index + 1 :]:
            similarity_pairs[(left_id, right_id)] = _similarity(
                list(normalized[left_id]),
                list(normalized[right_id]),
                indices,
                threshold,
            )

    def pair(left_id: str, right_id: str) -> dict[str, Any]:
        key = (left_id, right_id) if (left_id, right_id) in similarity_pairs else (right_id, left_id)
        return similarity_pairs[key]

    scores = {
        item_id: int(fusion["candidates"][item_id]["score_fixed"]) for item_id in initial_order
    }
    pending = [
        {"item_id": item_id, "score_fixed": scores[item_id]} for item_id in window_ids
    ]
    _sort_entries(pending)
    selected: list[str] = []
    rounds: list[dict[str, Any]] = []
    reductions: list[dict[str, Any]] = []
    round_no = 1
    while pending:
        current_order = [entry["item_id"] for entry in pending]
        round_reductions: list[dict[str, Any]] = []
        for entry in list(pending):
            item_id = entry["item_id"]
            close_to: list[dict[str, Any]] = []
            for selected_id in selected:
                relation = pair(item_id, selected_id)
                if relation["meets_threshold"]:
                    close_to.append(
                        {
                            "selected_item_id": selected_id,
                            "cosine": relation["cosine"],
                            "dot": relation["dot"],
                            "norm_product": relation["norm_product"],
                            "meets_threshold": True,
                        }
                    )
            if not close_to:
                continue
            old_score = int(entry["score_fixed"])
            new_score = max(lower_fixed, old_score - penalty_fixed)
            if new_score >= old_score:
                continue
            event = {
                "round": round_no,
                "item_id": item_id,
                "previous_score_fixed": old_score,
                "reduction_fixed": old_score - new_score,
                "new_score_fixed": new_score,
                "lower_bound_fixed": lower_fixed,
                "selected_too_close": close_to,
                "event_key": digest(
                    {
                        "round": round_no,
                        "item_id": item_id,
                        "previous_score_fixed": old_score,
                        "new_score_fixed": new_score,
                        "selected_too_close": [
                            item["selected_item_id"] for item in close_to
                        ],
                    }
                ),
            }
            entry["score_fixed"] = new_score
            scores[item_id] = new_score
            pending.remove(entry)
            pending.append(entry)
            _sort_entries(pending)
            reductions.append(event)
            round_reductions.append(event)
        chosen = pending.pop(0)
        selected.append(chosen["item_id"])
        rounds.append(
            {
                "round": round_no,
                "kind": "select",
                "pending_before": current_order,
                "selected_item_id": chosen["item_id"],
                "selected_score_fixed": chosen["score_fixed"],
                "reductions": round_reductions,
            }
        )
        round_no += 1

    final_order = selected + tail_ids
    scores_after = {
        item_id: int(scores[item_id]) for item_id in final_order
    }
    return {
        "window_size": config.rerank_window_size,
        "window_initial_order": window_ids,
        "window_final_order": selected,
        "tail_order": tail_ids,
        "order": final_order,
        "scores_after_rerank": scores_after,
        "rounds": rounds,
        "reductions": reductions,
        "diversity_penalty_fixed": penalty_fixed,
        "lower_bound_fixed": lower_fixed,
        "upper_bound_fixed": upper_fixed,
        "similarity_threshold": fraction_text(threshold),
        "similarity_features": list(catalog.similarity_features),
        "similarity_pairs": {
            f"{left}\u0000{right}": relation
            for (left, right), relation in sorted(
                similarity_pairs.items(), key=lambda item: (item[0][0], item[0][1])
            )
        },
        "pairwise_mac_units": (
            len(window_ids) * (len(window_ids) - 1) // 2 * catalog.dimension
        ),
    }


def attach_reduction_cache_keys(
    rerank_entry: dict[str, Any], rerank_key: str, config: LabConfig
) -> None:
    for event in rerank_entry["reductions"]:
        event["cache_key"] = digest(
            {
                "stage": "rerank_reduction",
                "upstream": rerank_key,
                "round": event["round"],
                "item_id": event["item_id"],
                "selected_too_close": [
                    item["selected_item_id"] for item in event["selected_too_close"]
                ],
                "config": {
                    "diversity_penalty": str(config.diversity_penalty),
                    "window": config.rerank_window_size,
                },
                "quantization": config.quantisation_dict(),
            }
        )


def rerank_summary(rerank_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "order": rerank_entry["order"],
        "window_initial_order": rerank_entry["window_initial_order"],
        "window_final_order": rerank_entry["window_final_order"],
        "reductions": rerank_entry["reductions"],
        "rounds": rerank_entry["rounds"],
        "pairwise_mac_units": rerank_entry["pairwise_mac_units"],
    }
