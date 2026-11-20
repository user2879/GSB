from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

from .canonical import as_fraction, fraction_json, sha256_json

TIE_BREAK_RULE = (
    "service rank: estimated p95, service id, strategy version, policy version, policy id; "
    "policy rank: estimated mean, policy version, policy id"
)


def _policy_order_key(result: dict[str, Any]) -> tuple[Fraction, int, str]:
    return (
        as_fraction(result["estimated_mean_us"]),
        int(result["policy_version"]),
        str(result["policy_id"]),
    )


def _select_policy(service_analysis: dict[str, Any]) -> dict[str, Any]:
    results = list(service_analysis["policies"].values())
    return sorted(results, key=_policy_order_key)[0]


def _service_order_key(service_analysis: dict[str, Any]) -> tuple[int, str, int, int, str]:
    selected = _select_policy(service_analysis)
    p95 = int(selected["quantiles"]["p95"]["estimated_fixed_us"])
    return (
        p95,
        str(service_analysis["service_id"]),
        int(service_analysis["strategy_version"]),
        int(selected["policy_version"]),
        str(selected["policy_id"]),
    )


def _p95_bound(result: dict[str, Any]) -> Fraction:
    return as_fraction(result["quantiles"]["p95"]["total_error_bound_us"])


def _mean_bound(result: dict[str, Any]) -> Fraction:
    return as_fraction(result["mean_error_bound_us"])


def _comparison(
    *,
    kind: str,
    left: str,
    right: str,
    left_value: Any,
    right_value: Any,
    left_bound: Fraction,
    right_bound: Fraction,
) -> dict[str, Any]:
    difference = abs(as_fraction(left_value) - as_fraction(right_value))
    combined = left_bound + right_bound
    return {
        "kind": kind,
        "left": left,
        "right": right,
        "left_value": fraction_json(as_fraction(left_value)),
        "right_value": fraction_json(as_fraction(right_value)),
        "absolute_difference_us": fraction_json(difference),
        "left_error_bound_us": fraction_json(left_bound),
        "right_error_bound_us": fraction_json(right_bound),
        "combined_error_bound_us": fraction_json(combined),
        "affected_by_error_bound": difference <= combined,
    }


def summarize_analyses(service_analyses: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered_services = sorted(service_analyses, key=lambda item: str(item["service_id"]))
    service_by_id = {str(item["service_id"]): item for item in ordered_services}
    selected = {
        service_id: _select_policy(analysis)
        for service_id, analysis in service_by_id.items()
    }

    ranking_rows = []
    for analysis in sorted(ordered_services, key=_service_order_key):
        service_id = str(analysis["service_id"])
        chosen = selected[service_id]
        ranking_rows.append(
            {
                "service_id": service_id,
                "strategy_version": int(analysis["strategy_version"]),
                "selected_policy": str(chosen["policy_id"]),
                "selected_policy_version": int(chosen["policy_version"]),
                "estimated_p95_fixed_us": int(
                    chosen["quantiles"]["p95"]["estimated_fixed_us"]
                ),
                "p95_error_bound_us": fraction_json(_p95_bound(chosen)),
                "estimated_mean_us": chosen["estimated_mean_us"],
                "mean_error_bound_us": chosen["mean_error_bound_us"],
            }
        )

    policy_choices: dict[str, Any] = {}
    policy_comparisons: list[dict[str, Any]] = []
    service_ids = sorted(service_by_id)
    for service_id in service_ids:
        analysis = service_by_id[service_id]
        results = sorted(
            analysis["policies"].values(),
            key=lambda item: str(item["policy_id"]),
        )
        ranked_policies = sorted(results, key=_policy_order_key)
        winner = ranked_policies[0]
        runner_up = ranked_policies[1] if len(ranked_policies) > 1 else None
        choice_comparisons = []
        for left_index, left in enumerate(results):
            for right in results[left_index + 1 :]:
                comparison = _comparison(
                    kind="policy_mean",
                    left=f"{service_id}/{left['policy_id']}",
                    right=f"{service_id}/{right['policy_id']}",
                    left_value=left["estimated_mean_us"],
                    right_value=right["estimated_mean_us"],
                    left_bound=_mean_bound(left),
                    right_bound=_mean_bound(right),
                )
                choice_comparisons.append(comparison)
                policy_comparisons.append(comparison)
        winner_margin = None
        if runner_up is not None:
            winner_margin = _comparison(
                kind="policy_winner_margin",
                left=f"{service_id}/{winner['policy_id']}",
                right=f"{service_id}/{runner_up['policy_id']}",
                left_value=winner["estimated_mean_us"],
                right_value=runner_up["estimated_mean_us"],
                left_bound=_mean_bound(winner),
                right_bound=_mean_bound(runner_up),
            )
        policy_choices[service_id] = {
            "winner": str(winner["policy_id"]),
            "winner_version": int(winner["policy_version"]),
            "winner_estimated_mean_us": winner["estimated_mean_us"],
            "winner_interval_low_us": fraction_json(
                as_fraction(winner["estimated_mean_us"]) - _mean_bound(winner)
            ),
            "winner_interval_high_us": fraction_json(
                as_fraction(winner["estimated_mean_us"]) + _mean_bound(winner)
            ),
            "runner_up": str(runner_up["policy_id"]) if runner_up else None,
            "winner_margin": winner_margin,
            "affected_by_error_bound": bool(
                winner_margin and winner_margin["affected_by_error_bound"]
            ),
            "ranked_policies": [str(item["policy_id"]) for item in ranked_policies],
            "comparisons": choice_comparisons,
        }

    service_comparisons: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(service_ids):
        for right_id in service_ids[left_index + 1 :]:
            left = selected[left_id]
            right = selected[right_id]
            service_comparisons.append(
                _comparison(
                    kind="service_p95",
                    left=left_id,
                    right=right_id,
                    left_value=left["quantiles"]["p95"]["estimated_fixed_us"],
                    right_value=right["quantiles"]["p95"]["estimated_fixed_us"],
                    left_bound=_p95_bound(left),
                    right_bound=_p95_bound(right),
                )
            )

    boundary_decisions = [
        comparison
        for comparison in service_comparisons + policy_comparisons
        if comparison["affected_by_error_bound"]
    ]
    return {
        "tie_break_rule": TIE_BREAK_RULE,
        "affected_rule": "absolute difference <= sum of both total error bounds",
        "service_order": [row["service_id"] for row in ranking_rows],
        "service_ranking": ranking_rows,
        "policy_choices": policy_choices,
        "service_comparisons": sorted(
            service_comparisons, key=lambda item: (item["left"], item["right"])
        ),
        "policy_comparisons": sorted(
            policy_comparisons, key=lambda item: (item["left"], item["right"])
        ),
        "boundary_decisions": sorted(
            boundary_decisions,
            key=lambda item: (item["kind"], item["left"], item["right"]),
        ),
        "has_error_bound_ambiguity": bool(boundary_decisions),
    }


def _scenario_state(service_analyses: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_analyses(service_analyses)
    choices = {
        service_id: row["winner"] for service_id, row in summary["policy_choices"].items()
    }
    p95_bounds = {
        row["service_id"]: row["p95_error_bound_us"]
        for row in summary["service_ranking"]
    }
    policy_bounds = {}
    for service_id, row in summary["policy_choices"].items():
        margin = row["winner_margin"]
        policy_bounds[service_id] = {
            "winner": row["winner"],
            "winner_estimated_mean_us": row["winner_estimated_mean_us"],
            "winner_margin_bound_us": (
                margin["combined_error_bound_us"] if margin else None
            ),
        }
    return {
        "service_order": summary["service_order"],
        "selected_policies": choices,
        "service_p95_error_bounds_us": p95_bounds,
        "policy_choice_error_bounds_us": policy_bounds,
    }


def detect_parameter_flips(
    base_service_analyses: Iterable[dict[str, Any]],
    scenarios: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_state = _scenario_state(base_service_analyses)
    events: list[dict[str, Any]] = []
    for scenario in scenarios:
        after_state = _scenario_state(scenario["service_analyses"])
        order_changed = base_state["service_order"] != after_state["service_order"]
        policies_changed = (
            base_state["selected_policies"] != after_state["selected_policies"]
        )
        if not order_changed and not policies_changed:
            continue
        changed_policy_services = sorted(
            service_id
            for service_id in base_state["selected_policies"]
            if base_state["selected_policies"].get(service_id)
            != after_state["selected_policies"].get(service_id)
        )
        event_core = {
            "kind": "parameter_sensitivity",
            "policy_id": scenario["policy_id"],
            "parameter": scenario["parameter"],
            "old_value": scenario["old_value"],
            "new_value": scenario["new_value"],
            "before": base_state,
            "after": after_state,
            "changed_service_order": order_changed,
            "changed_selected_policies": policies_changed,
            "changed_policy_services": changed_policy_services,
        }
        event = dict(event_core)
        event["flip_key"] = sha256_json(event_core)
        events.append(event)
    return sorted(
        events,
        key=lambda item: (
            item["policy_id"],
            item["parameter"],
            int(item["old_value"]),
            int(item["new_value"]),
            item["flip_key"],
        ),
    )


def diagnostics_digest(summary: dict[str, Any], flip_events: list[dict[str, Any]]) -> str:
    return sha256_json({"summary": summary, "flip_events": flip_events})


def flatten_flips_for_ledger(
    summary: dict[str, Any],
    flip_events: list[dict[str, Any]],
    service_order: list[str],
) -> list[dict[str, Any]]:
    owner_default = service_order[-1] if service_order else "<ledger>"
    records: list[dict[str, Any]] = []
    for comparison in summary["boundary_decisions"]:
        if comparison["kind"] == "service_p95":
            owner = max(comparison["left"], comparison["right"])
        else:
            owner = comparison["left"].split("/", 1)[0]
        item = {
            "flip_key": sha256_json({"kind": "boundary", "comparison": comparison}),
            "owner_service": owner,
            "kind": "error_bound_boundary",
            "payload": comparison,
        }
        records.append(item)
    for event in flip_events:
        item = {
            "flip_key": event["flip_key"],
            "owner_service": owner_default,
            "kind": "parameter_sensitivity",
            "payload": event,
        }
        records.append(item)
    return sorted(records, key=lambda item: (item["owner_service"], item["flip_key"]))
