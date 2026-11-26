from __future__ import annotations

from typing import Any

import numpy as np

from .composition import RISK_ORDER


STAGE_ORDER = ("scope_assignment", "cancellation", "length_penalty", "composition")


def reference_ticket(
    ticket: dict[str, Any],
    scan: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    config = bundle["config"]
    tokens = ticket["tokens"]
    strength = np.float64(config["cancellation_strength_points"])
    lower = np.float64(config["score_lower_bound_points"])
    per_token = np.float64(config["length_penalty_bp_per_token"]) / np.float64(10000.0)
    min_retention = np.float64(config["length_min_retention_bp"]) / np.float64(10000.0)
    weight = np.float64(bundle["channels"][ticket["channel_id"]]) / np.float64(1000.0)
    scope_by_id = {scope["scope_id"]: scope for scope in scan["scopes"]}

    scope_value = np.float64(0.0)
    cancellation_value = np.float64(0.0)
    penalty_value = np.float64(0.0)
    signed_value = np.float64(0.0)

    for hit in scan["hits"]:
        initial = np.float64(hit["base_evidence_points"])
        effective_id = hit["effective_scope_id"]
        events: list[dict[str, Any]] = []
        retention = np.float64(1.0)
        if effective_id is not None:
            scope = scope_by_id[effective_id]
            scope_value += initial
            events = sorted(
                (
                    {
                        "order": int(bundle["conflicts"][tokens[pos]]["order"]),
                        "position": pos,
                    }
                    for pos in scope["token_positions"]
                    if tokens[pos] in bundle["conflicts"]
                ),
                key=lambda event: (event["order"], event["position"]),
            )
            retention = np.maximum(
                min_retention,
                np.float64(1.0) - np.float64(scope["length"] - 1) * per_token,
            )

        current = initial
        for _ in events:
            current = np.maximum(lower, current - strength)
        after_penalty = current * retention
        cancellation_value += current
        penalty_value += after_penalty
        sign = np.float64(1.0 if hit["polarity"] == "positive" else -1.0)
        signed_value += sign * after_penalty

    weighted = signed_value * weight
    risk_score = np.maximum(np.float64(0.0), -weighted)
    pos_threshold = np.float64(config["positive_threshold_points"])
    neg_threshold = np.float64(config["negative_threshold_points"])
    if weighted >= pos_threshold:
        polarity = "positive"
    elif weighted <= neg_threshold:
        polarity = "negative"
    else:
        polarity = "neutral"

    thresholds = config["risk_thresholds_points"]
    risk_level = "low"
    if risk_score >= np.float64(thresholds["critical"]):
        risk_level = "critical"
    elif risk_score >= np.float64(thresholds["high"]):
        risk_level = "high"
    elif risk_score >= np.float64(thresholds["medium"]):
        risk_level = "medium"

    return {
        "ticket_id": ticket["ticket_id"],
        "values_float64": {
            "scope_assignment": float(scope_value),
            "cancellation": float(cancellation_value),
            "length_penalty": float(penalty_value),
            "composition": float(weighted),
            "risk_score": float(risk_score),
        },
        "polarity": polarity,
        "risk_level": risk_level,
        "risk_rank": RISK_ORDER[risk_level],
    }


def fixed_stage_values(settlement: dict[str, Any], conclusion: dict[str, Any]) -> dict[str, float]:
    scale = int(settlement["scale"])
    scope_value_q = sum(
        hit["initial_evidence_q"] for hit in settlement["hit_results"] if hit["effective_scope_id"]
    )
    return {
        "scope_assignment": scope_value_q / scale,
        "cancellation": settlement["evidence_after_cancel_q"] / scale,
        "length_penalty": settlement["evidence_after_penalty_q"] / scale,
        "composition": conclusion["weighted_sentiment_q"] / scale,
        "risk_score": conclusion["risk_score_q"] / scale,
    }


def compare_ticket_error(
    ticket_id: str,
    reference: dict[str, Any],
    settlement: dict[str, Any],
    conclusion: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    fixed_values = fixed_stage_values(settlement, conclusion)
    reference_values = reference["values_float64"]
    diffs = {
        stage: abs(np.float64(fixed_values[stage]) - np.float64(reference_values[stage]))
        for stage in STAGE_ORDER
    }
    exceeded = [
        stage for stage in STAGE_ORDER
        if diffs[stage] > np.float64(tolerances[stage])
    ]
    first = exceeded[0] if exceeded else None
    return {
        "ticket_id": ticket_id,
        "fixed_values": fixed_values,
        "reference_values": {key: float(value) for key, value in reference_values.items()},
        "stage_diffs": {key: float(value) for key, value in diffs.items()},
        "tolerances": {key: float(tolerances[key]) for key in STAGE_ORDER},
        "exceeded_stages": exceeded,
        "first_exceeded_stage": first,
        "fixed_risk_level": conclusion["risk_level"],
        "reference_risk_level": reference["risk_level"],
        "risk_matches_float64": conclusion["risk_level"] == reference["risk_level"],
        "polarity_matches_float64": conclusion["polarity"] == reference["polarity"],
    }


def attribution_for_risk_flip(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_assignment_fingerprint: str | None = None,
    right_assignment_fingerprint: str | None = None,
) -> str:
    """Classify one risk flip between two deterministic result profiles/snapshots."""
    if left_assignment_fingerprint != right_assignment_fingerprint:
        return "scope_attribution"

    left_sat = bool(left.get("saturation_fingerprint"))
    right_sat = bool(right.get("saturation_fingerprint"))
    if left_sat != right_sat or left.get("saturation_event_count") != right.get("saturation_event_count"):
        return "cancellation_saturation"

    left_boundary = bool(left.get("boundary"))
    right_boundary = bool(right.get("boundary"))
    if left_boundary or right_boundary:
        return "threshold_comparison"

    # If assignments and saturation are unchanged but the discrete level differs,
    # the final threshold comparison is the observable decision boundary.
    return "threshold_comparison"


def summarize_flip_sources(attributions: list[str]) -> dict[str, int]:
    counts = {
        "scope_attribution": 0,
        "cancellation_saturation": 0,
        "threshold_comparison": 0,
    }
    for item in attributions:
        if item in counts:
            counts[item] += 1
    counts["total"] = len(attributions)
    return counts
