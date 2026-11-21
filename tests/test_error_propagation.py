from __future__ import annotations

from dataclasses import replace

from qosfdiag.engine import CascadeEngine
from qosfdiag.error_propagation import compare_configurations, propagate_fixed_errors
from qosfdiag.generator import DeterministicUdpGenerator
from qosfdiag.policy import load_policy
from qosfdiag.samples import (
    three_stage_generator,
    three_stage_near_equal_policy,
    three_stage_policy,
)
from qosfdiag.transport import InMemoryTransport


def run(policy_dict: dict, generator_dict: dict):
    policy = load_policy(policy_dict)
    generator = DeterministicUdpGenerator(generator_dict)
    with CascadeEngine(
        policy,
        generator,
        scale=3,
        transport=InMemoryTransport([s.shaper_id for s in policy.shapers]),
    ) as engine:
        result = engine.run()
    errors = propagate_fixed_errors(
        policy,
        result.summary,
        result.events,
        scale=3,
        generator_digest=generator.sequence_digest,
    )
    return result, errors


def test_error_bound_terms_and_dominant_stage() -> None:
    result, errors = run(three_stage_policy(), three_stage_generator())
    assert errors.final_released_bytes == result.summary.final_released_bytes
    assert errors.bytes_error_bound == sum(term.bytes_contribution for term in errors.terms)
    assert errors.average_delay_error_bound_ns == sum(
        term.delay_contribution_ns for term in errors.terms
    )
    assert errors.tick_ns == 1_000_000
    assert errors.dominant_stage == "edge"
    assert abs(sum(term.bytes_share for term in errors.terms) - 1.0) < 1e-12
    assert all(term.rate_quantization_error_bytes_per_s == 0 for term in errors.terms)
    assert all(
        term.derivation["refill_cumulative_error_units"]
        == term.derivation["refill_boundary_count"]
        * term.derivation["refill_error_units_per_boundary"]
        for term in errors.terms
    )


def test_near_equal_configuration_is_order_independent() -> None:
    _, first = run(three_stage_policy(), three_stage_generator())
    _, second = run(three_stage_near_equal_policy(), three_stage_generator())
    left = compare_configurations(first, second)
    right = compare_configurations(second, first)
    assert left.indistinguishable is True
    assert left.error_affected is True
    assert left.winner == "INDISTINGUISHABLE"
    assert left == right
    assert left.released_bytes_difference <= left.combined_bytes_error_bound
    assert left.stable_order == tuple(sorted(left.stable_order))


def test_distinct_configuration_uses_larger_release_winner() -> None:
    _, first = run(three_stage_policy(), three_stage_generator())
    second = replace(
        first,
        final_released_bytes=first.final_released_bytes + first.bytes_error_bound + 1,
        bytes_error_bound=0,
        config_fingerprint="0" * 64,
        average_delay_ns=first.average_delay_ns + 1,
    )
    decision = compare_configurations(first, second)
    assert decision.indistinguishable is False
    assert decision.error_affected is False
    assert decision.winner == second.config_fingerprint
