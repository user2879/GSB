from __future__ import annotations

from typing import Any

from .quant import q_scale, quantize_number, round_ratio


def profile_settings(bundle: dict[str, Any], decimal_width: int | None = None) -> tuple[int, str]:
    fixed = bundle["config"]["fixed_point"]
    width = int(fixed["decimal_width"] if decimal_width is None else decimal_width)
    return q_scale(width), str(fixed["rounding_mode"])


def _conflict_events(scope: dict[str, Any], ticket: dict[str, Any], bundle: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = ticket["tokens"]
    events: list[dict[str, Any]] = []
    for position in scope["token_positions"]:
        term = tokens[position]
        conflict = bundle["conflicts"].get(term)
        if conflict is None:
            continue
        events.append({
            "conflict_id": conflict["conflict_id"],
            "term": term,
            "order": int(conflict["order"]),
            "position": position,
        })
    return sorted(events, key=lambda event: (event["order"], event["position"]))


def settle_hit(
    hit: dict[str, Any],
    events: list[dict[str, Any]],
    scope: dict[str, Any] | None,
    bundle: dict[str, Any],
    scale: int,
    rounding: str,
) -> dict[str, Any]:
    config = bundle["config"]
    initial_q = quantize_number(hit["base_evidence_points"], scale, rounding)
    strength_q = quantize_number(config["cancellation_strength_points"], scale, rounding)
    lower_q = quantize_number(config["score_lower_bound_points"], scale, rounding)
    current_q = initial_q
    steps: list[dict[str, Any]] = []
    saturation_positions: list[int] = []

    for step_index, event in enumerate(events):
        requested_q = current_q - strength_q
        saturated = requested_q < lower_q
        before_q = current_q
        current_q = max(lower_q, requested_q)
        if saturated:
            saturation_positions.append(event["position"])
        steps.append({
            "step_index": step_index,
            "conflict_id": event["conflict_id"],
            "conflict_term": event["term"],
            "conflict_order": event["order"],
            "saturation_position": event["position"] if saturated else None,
            "before_q": before_q,
            "requested_after_q": requested_q,
            "after_cancellation_q": current_q,
            "saturated": saturated,
        })

    length = int(scope["length"]) if scope else 0
    if scope is None:
        retention_bp = 10000
    else:
        per_token = int(config["length_penalty_bp_per_token"])
        minimum = int(config["length_min_retention_bp"])
        retention_bp = max(minimum, 10000 - max(0, length - 1) * per_token)
    after_penalty_q = round_ratio(current_q, retention_bp, 10000, rounding)

    return {
        "hit_id": hit["hit_id"],
        "ticket_id": hit["ticket_id"],
        "position": hit["position"],
        "term": hit["term"],
        "polarity": hit["polarity"],
        "effective_scope_id": hit["effective_scope_id"],
        "initial_evidence_q": initial_q,
        "cancellation_steps": steps,
        "final_cancellation_q": current_q,
        "canceled_to_zero": current_q == lower_q == 0,
        "saturation_positions": saturation_positions,
        "saturation_event_count": len(saturation_positions),
        "length_retention_bp": retention_bp,
        "after_length_penalty_q": after_penalty_q,
        "outside_scope": scope is None,
    }


def settle_scope(
    scan: dict[str, Any],
    scope: dict[str, Any],
    ticket: dict[str, Any],
    bundle: dict[str, Any],
    decimal_width: int | None = None,
) -> dict[str, Any]:
    scale, rounding = profile_settings(bundle, decimal_width)
    events = _conflict_events(scope, ticket, bundle)
    scope_hit_results: list[dict[str, Any]] = []
    for hit_id in scope["attributed_hit_ids"]:
        hit = next(item for item in scan["hits"] if item["hit_id"] == hit_id)
        scope_hit_results.append(settle_hit(hit, events, scope, bundle, scale, rounding))
    cost = scope["length"] * len(scope["attributed_hit_ids"])
    retention_bp = scope_hit_results[0]["length_retention_bp"] if scope_hit_results else max(
        int(bundle["config"]["length_min_retention_bp"]),
        10000 - max(0, scope["length"] - 1) * int(bundle["config"]["length_penalty_bp_per_token"]),
    )
    return {
        "scope_id": scope["scope_id"],
        "ticket_id": scan["ticket_id"],
        "trigger_id": scope["trigger_id"],
        "start": scope["start"],
        "end": scope["end"],
        "length": scope["length"],
        "attributed_hit_count": len(scope["attributed_hit_ids"]),
        "point_cost": cost,
        "decimal_width": int(bundle["config"]["fixed_point"]["decimal_width"] if decimal_width is None else decimal_width),
        "scale": scale,
        "conflict_events": events,
        "retention_bp": retention_bp,
        "hit_results": scope_hit_results,
        "evidence_before_cancel_q": sum(item["initial_evidence_q"] for item in scope_hit_results),
        "evidence_after_cancel_q": sum(item["final_cancellation_q"] for item in scope_hit_results),
        "evidence_after_penalty_q": sum(item["after_length_penalty_q"] for item in scope_hit_results),
        "saturation_event_count": sum(item["saturation_event_count"] for item in scope_hit_results),
        "canceled_hit_ids": [item["hit_id"] for item in scope_hit_results if item["canceled_to_zero"]],
    }


def settle_outside_hits(
    scan: dict[str, Any], bundle: dict[str, Any], decimal_width: int | None = None
) -> list[dict[str, Any]]:
    scale, rounding = profile_settings(bundle, decimal_width)
    results: list[dict[str, Any]] = []
    for hit in scan["hits"]:
        if hit["effective_scope_id"] is not None:
            continue
        results.append(settle_hit(hit, [], None, bundle, scale, rounding))
    return results


def assemble_settlement(
    scan: dict[str, Any],
    scope_results: list[dict[str, Any]],
    bundle: dict[str, Any],
    decimal_width: int | None = None,
) -> dict[str, Any]:
    width = int(bundle["config"]["fixed_point"]["decimal_width"] if decimal_width is None else decimal_width)
    scale, _ = profile_settings(bundle, decimal_width)
    outside_results = settle_outside_hits(scan, bundle, decimal_width)
    hit_results = outside_results + [
        hit_result for scope_result in scope_results for hit_result in scope_result["hit_results"]
    ]
    hit_results.sort(key=lambda item: item["position"])
    ordered_scope_results = sorted(scope_results, key=lambda item: item["start"])
    return {
        "ticket_id": scan["ticket_id"],
        "decimal_width": width,
        "scale": scale,
        "scope_results": ordered_scope_results,
        "hit_results": hit_results,
        "evidence_after_cancel_q": sum(item["final_cancellation_q"] for item in hit_results),
        "evidence_after_penalty_q": sum(item["after_length_penalty_q"] for item in hit_results),
        "saturation_event_count": sum(item["saturation_event_count"] for item in hit_results),
    }


def settle_ticket(
    scan: dict[str, Any],
    ticket: dict[str, Any],
    bundle: dict[str, Any],
    decimal_width: int | None = None,
) -> dict[str, Any]:
    scope_results = [
        settle_scope(scan, scope, ticket, bundle, decimal_width)
        for scope in sorted(scan["scopes"], key=lambda item: (item["start"], item["scope_id"]))
    ]
    return assemble_settlement(scan, scope_results, bundle, decimal_width)
