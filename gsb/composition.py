from __future__ import annotations

from typing import Any

from .quant import quantize_number, round_ratio


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _boundary_distance(
    num_q: int, denominator: int, threshold_q: int, half_step_q: float = 0.5
) -> tuple[bool, bool]:
    """Return exact threshold equality and within half a quantized step."""
    delta_num = abs(num_q - threshold_q * denominator)
    exact = delta_num == 0
    within_half_step = delta_num <= half_step_q * denominator
    return exact, within_half_step


def classify_ticket(
    weighted_q: int,
    exact_num_q: int,
    denominator: int,
    config: dict[str, Any],
    scale: int,
    rounding: str,
    width: int,
) -> dict[str, Any]:
    pos_q = quantize_number(config["positive_threshold_points"], scale, rounding)
    neg_q = quantize_number(config["negative_threshold_points"], scale, rounding)
    risk_thresholds = config["risk_thresholds_points"]
    medium_q = quantize_number(risk_thresholds["medium"], scale, rounding)
    high_q = quantize_number(risk_thresholds["high"], scale, rounding)
    critical_q = quantize_number(risk_thresholds["critical"], scale, rounding)
    risk_q = max(0, -weighted_q)
    boundary_width = int(config.get("boundary_half_step_decimal_width", 2))
    report_step_q = float(10 ** max(0, width - boundary_width)) if boundary_width < width else 1.0
    half_step_q = max(0.5, report_step_q / 2.0)

    if weighted_q >= pos_q:
        polarity = "positive"
    elif weighted_q <= neg_q:
        polarity = "negative"
    else:
        polarity = "neutral"

    risk_level = "low"
    if risk_q >= critical_q:
        risk_level = "critical"
    elif risk_q >= high_q:
        risk_level = "high"
    elif risk_q >= medium_q:
        risk_level = "medium"

    checks: list[dict[str, Any]] = []
    for label, threshold_q, observed_q in (
        ("polarity_positive", pos_q, weighted_q),
        ("polarity_negative", neg_q, weighted_q),
        ("risk_medium", medium_q, risk_q),
        ("risk_high", high_q, risk_q),
        ("risk_critical", critical_q, risk_q),
    ):
        if label == "polarity_negative":
            exact_num = exact_num_q
        elif label.startswith("risk"):
            # risk_score = max(0, -weighted); for the negative branch this is -num.
            exact_num = max(0, -exact_num_q)
        else:
            exact_num = exact_num_q
        exact, half = _boundary_distance(exact_num, denominator, threshold_q, half_step_q)
        if exact or half:
            checks.append({
                "threshold": label,
                "threshold_q": threshold_q,
                "observed_q": observed_q,
                "exact_equal": exact,
                "within_half_step": half,
                "decision_rule": "inclusive equality crosses; half-step-only remains on conservative side and is flagged",
            })

    return {
        "polarity": polarity,
        "risk_level": risk_level,
        "risk_rank": RISK_ORDER[risk_level],
        "risk_score_q": risk_q,
        "boundary": bool(checks),
        "boundary_checks": checks,
    }


def compose_ticket(
    ticket: dict[str, Any],
    scan: dict[str, Any],
    settlement: dict[str, Any],
    bundle: dict[str, Any],
    decimal_width: int | None = None,
) -> dict[str, Any]:
    config = bundle["config"]
    fixed = config["fixed_point"]
    width = int(fixed["decimal_width"] if decimal_width is None else decimal_width)
    scale = 10 ** width
    rounding = fixed["rounding_mode"]
    weight = int(bundle["channels"][ticket["channel_id"]])

    signed_q = 0
    contributions: list[dict[str, Any]] = []
    for result in settlement["hit_results"]:
        sign = 1 if result["polarity"] == "positive" else -1
        value_q = sign * int(result["after_length_penalty_q"])
        signed_q += value_q
        contributions.append({
            "hit_id": result["hit_id"],
            "polarity": result["polarity"],
            "signed_value_q": value_q,
            "scope_id": result["effective_scope_id"],
        })

    weighted_q = round_ratio(signed_q, weight, 1000, rounding)
    classification = classify_ticket(weighted_q, signed_q * weight, 1000, config, scale, rounding, width)
    referenced_scope_ids = sorted(
        {result["effective_scope_id"] for result in settlement["hit_results"] if result["effective_scope_id"]},
        key=lambda scope_id: next(scope["start"] for scope in scan["scopes"] if scope["scope_id"] == scope_id),
    )

    return {
        "ticket_id": ticket["ticket_id"],
        "channel_id": ticket["channel_id"],
        "channel_weight_permille": weight,
        "annotator_version": ticket["annotator_version"],
        "completion_status": ticket["completion_status"],
        "decimal_width": width,
        "scale": scale,
        "raw_sentiment_q": signed_q,
        "weighted_sentiment_q": weighted_q,
        "weighted_sentiment_points": weighted_q / scale,
        "contributions": contributions,
        "referenced_scope_ids": referenced_scope_ids,
        "cancellation_trails": settlement["hit_results"],
        "saturation_event_count": settlement["saturation_event_count"],
        "scope_results_summary": [
            {key: scope[key] for key in (
                "scope_id", "start", "end", "length", "attributed_hit_count", "point_cost",
                "evidence_after_cancel_q", "evidence_after_penalty_q", "saturation_event_count",
            )}
            for scope in settlement["scope_results"]
        ],
        **classification,
    }


def order_conclusions(conclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        conclusions,
        key=lambda item: (-item["risk_rank"], -item["risk_score_q"], item["weighted_sentiment_q"], item["ticket_id"]),
    )
    for rank, conclusion in enumerate(ordered, 1):
        conclusion["tie_break_order"] = rank
    return ordered
