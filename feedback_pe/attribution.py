from __future__ import annotations

from typing import Any

import numpy as np

from .config import QUANT_TIER_CHOICES
from .pipeline import compute_encoding_and_scores, layer_indices


def _dot_encoded(query: list[int], key: list[int]) -> int:
    accumulator = 0
    for dimension_index in range(len(query)):
        accumulator += int(query[dimension_index]) * int(key[dimension_index])
    return accumulator


def _exact_score_margin(bundle: dict[str, Any], winner: int, loser: int) -> float:
    scale = int(bundle["score"]["scale"])
    rows = {row["key_index"]: row for row in bundle["score"]["rows"]}
    return float(rows[winner]["accumulator_int"] - rows[loser]["accumulator_int"]) / (scale * scale)


def _hybrid_margin(low_position: dict[str, Any], high_position: dict[str, Any],
                   winner: int, loser: int) -> float:
    scale = 1 << int(low_position["frac_bits"])
    query = [
        low_position["query"]["feature_int"][d] + high_position["query"]["pe_int"][d]
        for d in range(len(low_position["query"]["encoded_int"]))
    ]
    margins = []
    for key_index in (winner, loser):
        low_key = low_position["keys"][key_index]
        high_key = high_position["keys"][key_index]
        key = [
            low_key["feature_int"][d] + high_key["pe_int"][d]
            for d in range(len(low_key["encoded_int"]))
        ]
        margins.append(_dot_encoded(query, key))
    return float(margins[0] - margins[1]) / (scale * scale)


def _normalization_effect(bundle: dict[str, Any], winner: int, loser: int) -> float:
    scale = int(bundle["score"]["scale"])
    score_ints = [row["score_int"] for row in bundle["score"]["rows"]]
    max_score = max(score_ints)
    winner_float = np.float64(score_ints[winner]) / np.float64(scale)
    loser_float = np.float64(score_ints[loser]) / np.float64(scale)
    max_float = np.float64(max_score) / np.float64(scale)
    shifted_winner = winner_float - max_float
    shifted_loser = loser_float - max_float
    after_max = float(shifted_winner - shifted_loser)
    direct = float(winner_float - loser_float)
    return after_max - direct


def _rank_order(bundle: dict[str, Any], indexes: list[int]) -> list[int]:
    rows = {row["key_index"]: row for row in bundle["score"]["rows"]}
    return sorted(indexes, key=lambda index: (-rows[index]["score_float"], index))


def analyze_tier_flips(sample: dict[str, Any], scheme: dict[str, Any],
                       frac_bits: int) -> dict[str, Any]:
    groups = layer_indices(sample)
    extrap_indexes = groups["outside_extrapolation"]
    tier_bundles = {
        tier: compute_encoding_and_scores(sample, scheme, frac_bits, tier)
        for tier in QUANT_TIER_CHOICES
    }
    flips: list[dict[str, Any]] = []

    for tier_index in range(len(QUANT_TIER_CHOICES) - 1):
        tier_from = QUANT_TIER_CHOICES[tier_index]
        tier_to = QUANT_TIER_CHOICES[tier_index + 1]
        low = tier_bundles[tier_from]
        high = tier_bundles[tier_to]
        low_rows = {row["key_index"]: row for row in low["score"]["rows"]}
        high_rows = {row["key_index"]: row for row in high["score"]["rows"]}
        low_order = _rank_order(low, extrap_indexes)
        high_order = _rank_order(high, extrap_indexes)

        for winner in extrap_indexes:
            for loser in extrap_indexes:
                if winner == loser:
                    continue
                low_margin = low_rows[winner]["score_float"] - low_rows[loser]["score_float"]
                high_margin = high_rows[winner]["score_float"] - high_rows[loser]["score_float"]
                if not (low_margin > 0.0 and high_margin < 0.0):
                    continue

                exact_low = _exact_score_margin(low, winner, loser)
                exact_high = _exact_score_margin(high, winner, loser)
                position_delta = exact_high - exact_low
                hybrid_high = _hybrid_margin(low["position"], high["position"], winner, loser)
                position_only_delta = hybrid_high - exact_low
                feature_delta = exact_high - hybrid_high

                rounding_low = low_margin - exact_low
                rounding_high = high_margin - exact_high
                accumulation_delta = rounding_high - rounding_low

                normalization_low = _normalization_effect(low, winner, loser)
                normalization_high = _normalization_effect(high, winner, loser)
                normalization_delta = normalization_high - normalization_low
                total_delta = high_margin - low_margin
                explained = position_delta + accumulation_delta + normalization_delta
                unexplained = total_delta - explained

                after_position = low_margin + position_delta
                after_accumulation = after_position + accumulation_delta
                if after_position < 0.0:
                    source = "position_vector_quantization"
                elif after_accumulation < 0.0:
                    source = "accumulation_rounding"
                elif after_accumulation + normalization_delta < 0.0:
                    source = "normalization_max_subtraction"
                else:
                    source = "combined_or_unresolved"

                contributions = {
                    "position_vector_quantization": position_delta,
                    "position_only_hybrid": position_only_delta,
                    "input_feature_quantization": feature_delta,
                    "accumulation_rounding": accumulation_delta,
                    "normalization_max_subtraction": normalization_delta,
                    "unexplained_float64": unexplained,
                }
                dominant_source = max(
                    (
                        ("position_vector_quantization", abs(position_delta)),
                        ("accumulation_rounding", abs(accumulation_delta)),
                        ("normalization_max_subtraction", abs(normalization_delta)),
                    ),
                    key=lambda item: item[1],
                )[0]

                flips.append({
                    "sample_id": sample["id"],
                    "scheme": scheme["name"],
                    "frac_bits": frac_bits,
                    "tier_from": tier_from,
                    "tier_to": tier_to,
                    "winner_key_index": winner,
                    "loser_key_index": loser,
                    "winner_position": sample["key_positions"][winner],
                    "loser_position": sample["key_positions"][loser],
                    "old_rank": low_order.index(winner) + 1,
                    "old_loser_rank": low_order.index(loser) + 1,
                    "new_rank": high_order.index(winner) + 1,
                    "new_loser_rank": high_order.index(loser) + 1,
                    "old_score_margin": low_margin,
                    "new_score_margin": high_margin,
                    "attributed_source": source,
                    "dominant_source": dominant_source,
                    "contributions": contributions,
                })

    order_changed = low_order != high_order if extrap_indexes else False
    return {
        "sample_id": sample["id"],
        "scheme": scheme["name"],
        "frac_bits": frac_bits,
        "extrapolation_order_by_tier": {
            tier: _rank_order(bundle, extrap_indexes)
            for tier, bundle in tier_bundles.items()
        },
        "order_changed_in_adjacent_tiers": any(
            _rank_order(tier_bundles[a], extrap_indexes)
            != _rank_order(tier_bundles[b], extrap_indexes)
            for a, b in zip(QUANT_TIER_CHOICES, QUANT_TIER_CHOICES[1:])
        ),
        "flips": flips,
    }
