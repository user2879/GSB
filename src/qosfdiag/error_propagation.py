"""Integer error-bound composition and stable indistinguishability decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import digest
from .fixed import FixedPointSpec, ceil_div
from .models import (
    DecisionEvent,
    ErrorPropagationResult,
    ErrorTerm,
    IndistinguishabilityResult,
    PolicySet,
    RunSummary,
)


DEFAULT_DOMINANCE_THRESHOLD = 0.50


def _summary_stats(summary: RunSummary) -> dict[str, Any]:
    return {stats.stage_id: stats for stats in summary.stage_stats}


def propagate_fixed_errors(
    policy: PolicySet,
    summary: RunSummary,
    events: Iterable[DecisionEvent],
    *,
    scale: int,
    generator_digest: str = "",
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> ErrorPropagationResult:
    """Compose a conservative per-stage error bound along the cascade.

    Derivation per stage i:
      rate error = |floor(kbps*125*S)/S - kbps*125| = 0 bytes/s;
      one arrival/release boundary loses < 1 tick, i.e. at most
        rate_units_per_tick token units, so R boundaries contribute
        ceil(R*rate_units_per_tick/S) bytes;
      queue delay uses floor arrival time and ceiling release time, so each
        delayed packet has delay error < 2*tick_ns.

    The local byte contribution is added to the upstream cumulative bound in
    topological order. Delay error bounds are additive across stages because a
    packet's end-to-end queue delay is the sum of per-stage queue delays.
    """

    if not 0.0 < dominance_threshold <= 1.0:
        raise ValueError("dominance_threshold must be in (0, 1]")
    spec = FixedPointSpec(scale)
    stats_by_id = _summary_stats(summary)
    refill_counts: dict[str, int] = {shaper.shaper_id: 0 for shaper in policy.shapers}
    for event in events:
        if event.stage_id in refill_counts and event.kind in {"arrival", "release"}:
            refill_counts[event.stage_id] += 1

    local_bytes: dict[str, int] = {}
    local_delay: dict[str, int] = {}
    derivation: dict[str, dict[str, int]] = {}
    for shaper in policy.shapers:
        stage_id = shaper.shaper_id
        stats = stats_by_id[stage_id]
        rate = spec.rate_quantization_analysis(shaper.committed_rate_kbps)
        refills = refill_counts[stage_id]
        refill_units = refills * rate["refill_error_units_per_boundary"]
        refill_bytes = ceil_div(refill_units, spec.token_units_per_byte)
        rate_bytes = rate["rate_quantization_error_bytes_per_s"]
        local_bytes[stage_id] = refill_bytes + rate_bytes
        local_delay[stage_id] = (
            spec.delay_quantization_analysis()["queue_delay_error_bound_ns"]
            if stats.released_packets > 0
            else 0
        )
        derivation[stage_id] = {
            "scale": scale,
            "tick_ns": spec.tick_ns,
            "committed_rate_kbps": shaper.committed_rate_kbps,
            "exact_bytes_per_s": rate["exact_bytes_per_s"],
            "represented_units_per_tick": rate["represented_units_per_tick"],
            "rate_quantization_error_bytes_per_s": rate_bytes,
            "refill_boundary_count": refills,
            "refill_error_units_per_boundary": rate["refill_error_units_per_boundary"],
            "refill_cumulative_error_units": refill_units,
            "refill_cumulative_error_bytes_ceiling": refill_bytes,
            "delay_arrival_floor_error_ns_bound_exclusive": spec.tick_ns,
            "delay_release_ceiling_error_ns_bound_exclusive": spec.tick_ns,
            "delay_quantization_error_ns": local_delay[stage_id],
            "released_packets": stats.released_packets,
            "forwarded_bytes": stats.forwarded_bytes,
        }

    total_bytes_bound = sum(local_bytes.values())
    total_delay_bound = sum(local_delay.values())
    terms: list[ErrorTerm] = []
    cumulative_bytes = 0
    cumulative_delay = 0
    for stage_id in policy.topological_order():
        cumulative_bytes += local_bytes[stage_id]
        cumulative_delay += local_delay[stage_id]
        stage_derivation = dict(derivation[stage_id])
        stage_derivation["cumulative_bytes_error_bound_to_stage"] = cumulative_bytes
        stage_derivation["cumulative_delay_error_bound_ns_to_stage"] = cumulative_delay
        bytes_share = 0.0 if total_bytes_bound == 0 else local_bytes[stage_id] / total_bytes_bound
        delay_share = 0.0 if total_delay_bound == 0 else local_delay[stage_id] / total_delay_bound
        terms.append(
            ErrorTerm(
                stage_id=stage_id,
                rate_quantization_error_bytes_per_s=derivation[stage_id][
                    "rate_quantization_error_bytes_per_s"
                ],
                refill_cumulative_error_units=derivation[stage_id][
                    "refill_cumulative_error_units"
                ],
                refill_cumulative_error_bytes_ceiling=derivation[stage_id][
                    "refill_cumulative_error_bytes_ceiling"
                ],
                delay_quantization_error_ns=derivation[stage_id][
                    "delay_quantization_error_ns"
                ],
                bytes_contribution=local_bytes[stage_id],
                delay_contribution_ns=local_delay[stage_id],
                bytes_share=bytes_share,
                delay_share=delay_share,
                derivation=stage_derivation,
            )
        )

    max_bytes_share = max((term.bytes_share for term in terms), default=0.0)
    candidates = [
        term for term in terms if abs(term.bytes_share - max_bytes_share) <= 1e-12
    ]
    if max_bytes_share >= dominance_threshold and candidates:
        dominant_stage = sorted(
            candidates, key=lambda term: (-term.delay_share, term.stage_id)
        )[0].stage_id
    else:
        dominant_stage = "NONE"

    fingerprint = digest(
        {
            "scale": scale,
            "policy_version": policy.policy_version,
            "shapers": [shaper.as_dict() for shaper in policy.shapers],
            "edges": [edge.as_dict() for edge in policy.edges],
            "generator_digest": generator_digest,
        }
    )
    intermediate = {
        "fixed_point": spec.as_dict(),
        "rate_analyses": {
            shaper.shaper_id: spec.rate_quantization_analysis(shaper.committed_rate_kbps)
            for shaper in policy.shapers
        },
        "delay_analysis": spec.delay_quantization_analysis(),
        "refill_boundary_counts": dict(sorted(refill_counts.items())),
        "composition_order": list(policy.topological_order()),
    }
    return ErrorPropagationResult(
        scale=scale,
        tick_ns=spec.tick_ns,
        total_released_bytes=sum(stats.forwarded_bytes for stats in summary.stage_stats),
        released_packet_count=sum(stats.released_packets for stats in summary.stage_stats),
        final_released_bytes=summary.final_released_bytes,
        bytes_error_bound=total_bytes_bound,
        average_delay_ns=summary.average_delay_ns,
        average_delay_error_bound_ns=total_delay_bound,
        terms=tuple(terms),
        dominant_stage=dominant_stage,
        dominance_threshold=dominance_threshold,
        dominance_rule=(
            "choose the stage with the largest byte-error share when that share "
            f">= {dominance_threshold:.2f}; ties break by delay share then ascending id; "
            "otherwise report NONE"
        ),
        intermediate_terms=intermediate,
        config_fingerprint=fingerprint,
    )


def compare_configurations(
    first: ErrorPropagationResult,
    second: ErrorPropagationResult,
    *,
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> IndistinguishabilityResult:
    """Compare two fixed-point configurations independent of argument order.

    Stable canonical order is (config_fingerprint, released_bytes,
    average_delay_ns). The indistinguishability predicate is exactly:

        abs(bytes_a - bytes_b) <= bound_a + bound_b

    Delay bounds are reported as corroborating information but do not silently
    broaden the requested byte-based predicate.
    """

    del dominance_threshold
    ordered = sorted(
        (first, second),
        key=lambda item: (
            item.config_fingerprint,
            item.final_released_bytes,
            item.average_delay_ns,
        ),
    )
    a, b = ordered
    bytes_diff = abs(a.final_released_bytes - b.final_released_bytes)
    combined_bytes_bound = a.bytes_error_bound + b.bytes_error_bound
    delay_diff = abs(a.average_delay_ns - b.average_delay_ns)
    combined_delay_bound = (
        a.average_delay_error_bound_ns + b.average_delay_error_bound_ns
    )
    indistinguishable = bytes_diff <= combined_bytes_bound
    if indistinguishable:
        winner = "INDISTINGUISHABLE"
        basis = (
            "|released_bytes_a-released_bytes_b| <= bytes_error_bound_a+"
            "bytes_error_bound_b; canonical fingerprint order used for labels"
        )
    else:
        winner = a.config_fingerprint
        basis = (
            "released-byte difference exceeds combined error bound; larger released "
            "bytes wins, canonical fingerprint order used only for labels"
        )
    return IndistinguishabilityResult(
        indistinguishable=indistinguishable,
        error_affected=indistinguishable,
        released_bytes_a=a.final_released_bytes,
        released_bytes_b=b.final_released_bytes,
        released_bytes_difference=bytes_diff,
        combined_bytes_error_bound=combined_bytes_bound,
        average_delay_ns_a=a.average_delay_ns,
        average_delay_ns_b=b.average_delay_ns,
        average_delay_difference_ns=delay_diff,
        combined_delay_error_bound_ns=combined_delay_bound,
        winner=winner,
        stable_order=(a.config_fingerprint, b.config_fingerprint),
        decision_basis=basis,
    )


@dataclass(frozen=True)
class AnalysisResult:
    run: Any
    errors: ErrorPropagationResult

    def as_dict(self) -> dict[str, Any]:
        return {"run": self.run.as_dict(), "errors": self.errors.as_dict()}
