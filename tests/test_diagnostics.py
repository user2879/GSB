from __future__ import annotations

from trace_diag.canonical import as_fraction, canonical_json
from trace_diag.engine import analyze_documents
from trace_diag.samplegen import build_default_policies, build_default_services
from trace_diag.spec import parse_policies_document, parse_services_document


def _bundle(records: int = 300):
    services = parse_services_document(build_default_services(records))
    policies = parse_policies_document(build_default_policies())
    return analyze_documents(services, policies)


def test_near_ties_are_marked_and_evidence_is_stable():
    first = _bundle(300)
    second = _bundle(300)
    assert first.bundle_digest == second.bundle_digest
    assert canonical_json(first.diagnostics) == canonical_json(second.diagnostics)
    assert first.diagnostics["boundary_decisions"]
    assert all(
        row["affected_by_error_bound"] for row in first.diagnostics["boundary_decisions"]
    )
    assert "service id" in first.diagnostics["tie_break_rule"]
    for service in first.service_analyses:
        for result in service["policies"].values():
            for quantile in result["quantiles"].values():
                assert abs(as_fraction(quantile["difference_us"])) <= as_fraction(
                    quantile["total_error_bound_us"]
                )


def test_parameter_sensitivity_flip_events_are_sorted_and_have_before_after():
    bundle = _bundle(300)
    assert bundle.flip_events
    keys = [
        (
            event["policy_id"],
            event["parameter"],
            int(event["old_value"]),
            int(event["new_value"]),
            event["flip_key"],
        )
        for event in bundle.flip_events
    ]
    assert keys == sorted(keys)
    for event in bundle.flip_events:
        assert event["before"]["service_order"]
        assert event["after"]["service_order"]
        assert event["before"]["selected_policies"]
        assert event["after"]["selected_policies"]
        assert event["before"]["service_p95_error_bounds_us"]
        assert event["parameter"] in {"scale_power", "bucket_count"}


def test_policy_choice_ties_use_error_bounds():
    bundle = _bundle(300)
    for service_id, choice in bundle.diagnostics["policy_choices"].items():
        assert choice["ranked_policies"]
        if choice["winner_margin"]:
            assert "combined_error_bound_us" in choice["winner_margin"]
        assert service_id in bundle.diagnostics["service_order"]


def test_parameter_scan_detects_service_order_flip():
    from fractions import Fraction

    from trace_diag.canonical import fraction_json
    from trace_diag.diagnostics import detect_parameter_flips

    def analysis(service_id: str, p95: int, mean: int):
        return {
            "service_id": service_id,
            "strategy_version": 1,
            "policies": {
                "only": {
                    "policy_id": "only",
                    "policy_version": 1,
                    "strategy_version": 1,
                    "estimated_mean_us": fraction_json(Fraction(mean)),
                    "mean_error_bound_us": fraction_json(Fraction(1)),
                    "quantiles": {
                        "p95": {
                            "estimated_fixed_us": p95,
                            "total_error_bound_us": fraction_json(Fraction(10)),
                        }
                    },
                }
            },
        }

    base = [analysis("svc-a", 100, 100), analysis("svc-b", 200, 200)]
    changed = [analysis("svc-a", 100, 100), analysis("svc-b", 80, 200)]
    events = detect_parameter_flips(
        base,
        [
            {
                "scenario_key": "only:scale_power:0->1",
                "policy_id": "only",
                "parameter": "scale_power",
                "old_value": 0,
                "new_value": 1,
                "service_analyses": changed,
            }
        ],
    )
    assert len(events) == 1
    assert events[0]["changed_service_order"] is True
    assert events[0]["before"]["service_order"] == ["svc-a", "svc-b"]
    assert events[0]["after"]["service_order"] == ["svc-b", "svc-a"]
