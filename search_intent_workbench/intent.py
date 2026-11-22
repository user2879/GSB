"""Vocabulary-weighted intent classification with stable softmax scoring."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

import numpy as np

from .canonical import digest
from .normalization import normalize_match_tokens


INTENT_ALGORITHM_VERSION = "intent-v1"


def _build_feature_patterns(
    config: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for intent in config.get("intent_categories", []):
        for feature_index, feature in enumerate(intent.get("features", [])):
            feature_tokens = normalize_match_tokens(feature["term"], rules, config)
            if not feature_tokens:
                continue
            patterns.append(
                {
                    "intent": intent["name"],
                    "feature": feature["term"],
                    "feature_weight": float(feature["weight"]),
                    "feature_tokens": feature_tokens,
                    "feature_index": feature_index,
                }
            )
    return patterns


def find_feature_matches(
    tokens: list[str], config: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for pattern in _build_feature_patterns(config, rules):
        width = len(pattern["feature_tokens"])
        if width > len(tokens):
            continue
        for start in range(0, len(tokens) - width + 1):
            if tokens[start : start + width] != pattern["feature_tokens"]:
                continue
            matches.append(
                {
                    "intent": pattern["intent"],
                    "feature": pattern["feature"],
                    "feature_weight": pattern["feature_weight"],
                    "feature_index": pattern["feature_index"],
                    "token_start": start,
                    "token_end": start + width,
                    "token_length": width,
                    "matched_text": " ".join(tokens[start : start + width]),
                }
            )
    matches.sort(
        key=lambda item: (
            item["intent"],
            item["token_start"],
            item["feature_index"],
            item["feature"],
        )
    )
    return matches


def intent_parameter_fingerprint(
    matches: list[dict[str, Any]], config: dict[str, Any], rules: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Fingerprint only parameter slices that can affect this query."""

    matched_intents = {match["intent"] for match in matches}
    intent_weights = [
        {"intent": intent["name"], "weight": float(intent["weight"])}
        for intent in config.get("intent_categories", [])
        if intent["name"] in matched_intents
    ]
    intent_weights.sort(key=lambda item: item["intent"])
    feature_definitions = sorted(
        [
            {
                "intent": intent["name"],
                "features": [
                    {"term": feature["term"], "weight": float(feature["weight"])}
                    for feature in sorted(
                        intent.get("features", []), key=lambda item: item["term"]
                    )
                ],
            }
            for intent in config.get("intent_categories", [])
        ],
        key=lambda item: item["intent"],
    )
    scoring = config.get("intent_scoring", {})
    fingerprint = digest(
        {
            "algorithm": INTENT_ALGORITHM_VERSION,
            "rule_table_version": rules.get("version"),
            "category_roster": sorted(
                intent["name"] for intent in config.get("intent_categories", [])
            ),
            "feature_definitions": feature_definitions,
            "position_decay": float(scoring.get("position_decay", 0.0)),
            "decimal_places": int(scoring.get("decimal_places", 6)),
            "quantum": str(scoring.get("quantum", "0.000001")),
            "intent_weights": intent_weights,
        }
    )
    return fingerprint, [
        {"kind": "intent", "id": item["intent"], "weight": item["weight"]}
        for item in intent_weights
    ]


def intent_cache_key(
    content_fingerprint: str,
    normalization_key: str,
    parameter_fingerprint: str,
    rules: dict[str, Any],
    dictionary_context_hash: str,
) -> str:
    return digest(
        {
            "algorithm": INTENT_ALGORITHM_VERSION,
            "content_fingerprint": content_fingerprint,
            "normalization_key": normalization_key,
            "parameter_fingerprint": parameter_fingerprint,
            "rule_table_version": rules.get("version"),
            "dictionary_context_hash": dictionary_context_hash,
        }
    )


def _quantize(value: float, decimal_places: int, quantum_value: Decimal) -> tuple[int, float, str]:
    quantized = Decimal(str(value)).quantize(quantum_value, rounding=ROUND_HALF_EVEN)
    units = int((quantized / quantum_value).to_integral_value(rounding=ROUND_HALF_EVEN))
    numeric = float(quantized)
    formatted = format(numeric, f".{decimal_places}f")
    return units, numeric, formatted


def classify_intent(
    matches: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    scoring = config.get("intent_scoring", {})
    position_decay = float(scoring.get("position_decay", 0.0))
    decimal_places = int(scoring.get("decimal_places", 6))
    quantum_value = Decimal(str(scoring.get("quantum", "0.000001")))
    intents = sorted(config.get("intent_categories", []), key=lambda item: item["name"])

    raw_scores = {intent["name"]: [] for intent in intents}
    contributions_by_intent: dict[str, list[dict[str, Any]]] = {
        intent["name"]: [] for intent in intents
    }
    intent_weight = {intent["name"]: float(intent["weight"]) for intent in intents}
    for match in matches:
        decay_factor = 1.0 / (1.0 + position_decay * match["token_start"])
        contribution = (
            intent_weight[match["intent"]]
            * match["feature_weight"]
            * match["token_length"]
            * decay_factor
        )
        raw_scores[match["intent"]].append(contribution)
        contribution_units, contribution_numeric, contribution_formatted = _quantize(
            contribution, decimal_places, quantum_value
        )
        contributions_by_intent[match["intent"]].append(
            {
                "feature": match["feature"],
                "matched_text": match["matched_text"],
                "token_start": match["token_start"],
                "token_length": match["token_length"],
                "feature_weight": match["feature_weight"],
                "intent_weight": intent_weight[match["intent"]],
                "position_decay_factor": round(decay_factor, 12),
                "contribution": contribution_numeric,
                "contribution_formatted": contribution_formatted,
                "contribution_units": contribution_units,
            }
        )

    combined_scores = {
        intent["name"]: math.fsum(sorted(raw_scores[intent["name"]]))
        for intent in intents
    }
    ordered_names = [intent["name"] for intent in intents]
    if ordered_names:
        score_array = np.asarray(
            [combined_scores[name] for name in ordered_names], dtype=np.float64
        )
        shifted = score_array - np.max(score_array)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials)
    else:
        probabilities = np.asarray([], dtype=np.float64)

    score_rows: list[dict[str, Any]] = []
    for index, name in enumerate(ordered_names):
        units, numeric, formatted = _quantize(
            float(probabilities[index]), decimal_places, quantum_value
        )
        score_rows.append(
            {
                "intent": name,
                "raw_score": round(combined_scores[name], 12),
                "score": numeric,
                "score_formatted": formatted,
                "score_units": units,
                "contributions": contributions_by_intent[name],
            }
        )
    score_rows.sort(key=lambda item: (-item["score_units"], item["intent"]))
    primary = score_rows[0]["intent"] if score_rows else None
    return {
        "primary_intent": primary,
        "intent_scores": score_rows,
        "contributions": [
            {
                "intent": row["intent"],
                **contribution,
            }
            for row in score_rows
            for contribution in row["contributions"]
        ],
        "normalization": {
            "method": "stable_softmax(max_subtraction)",
            "decimal_places": decimal_places,
            "quantum": str(quantum_value),
            "rounding": "ROUND_HALF_EVEN",
            "normalization_note": "scores are independently quantized after softmax",
        },
        "scoring_formula": {
            "raw_score": "sum(intent_weight * feature_weight * token_length * (1 / (1 + position_decay * token_start)))",
            "probability": "exp(raw_score - max_raw_score) / sum(exp(raw_score - max_raw_score))",
            "tie_break": "descending quantized score, then intent name ascending",
        },
    }


def intent_output_fingerprint(result: dict[str, Any]) -> str:
    return digest({"algorithm": INTENT_ALGORITHM_VERSION, "result": result})
