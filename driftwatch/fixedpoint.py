from __future__ import annotations

from .models import Criteria, DriftResult, ErrorBound, HostSpec, MergedResult

MAX_SIGNED_63 = (1 << 63) - 1
MIN_SIGNED_63 = -(1 << 63)


def saturate(value: int) -> int:
    if value > MAX_SIGNED_63:
        return MAX_SIGNED_63
    if value < MIN_SIGNED_63:
        return MIN_SIGNED_63
    return int(value)


def sat_mul(left: int, right: int) -> int:
    return saturate(int(left) * int(right))


def sat_add(left: int, right: int) -> int:
    return saturate(int(left) + int(right))


def sat_sub(left: int, right: int) -> int:
    return saturate(int(left) - int(right))


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point denominator must not be zero")
    quotient = abs(int(numerator)) // abs(int(denominator))
    return quotient if (numerator >= 0) == (denominator > 0) else -quotient


def judge_drift(
    snapshot_seq: int,
    host_id: str,
    metric_id: str,
    host_spec: HostSpec,
    merged: MergedResult,
    criteria: Criteria,
) -> DriftResult:
    expected = int(host_spec.expected[metric_id])
    tolerance_bps = int(host_spec.tolerance_bps[metric_id])
    if merged.sparse or merged.authority_value is None:
        return DriftResult(
            snapshot_seq=snapshot_seq,
            host_id=host_id,
            metric_id=metric_id,
            status="SPARSE",
            decision="UNKNOWN",
            expected=expected,
            observed=None,
            tolerance_bps=tolerance_bps,
            drift_signed_fixed=None,
            drift_excess_fixed=None,
            drift_ratio_bps=None,
            allowed_fixed=None,
            error_bound=None,
            error_affected=False,
            boundary_rule=criteria.boundary_rule,
            saturated=False,
            error_code="E_SPARSE",
            reason="effective observation agents are below min_sources; no authority is asserted",
        )

    observed = int(merged.authority_value)
    scale = 10 ** criteria.scale_exp
    raw_expected_fixed = expected * scale
    raw_observed_fixed = observed * scale
    raw_allowed = (expected * scale * tolerance_bps) // 10000
    raw_drift = (observed - expected) * scale
    raw_merge_error = int(merged.max_abs_deviation) * scale
    raw_composite = scale + 1 + raw_merge_error
    positive_excess_raw = max(0, raw_drift)
    raw_ratio_bps = trunc_div_toward_zero(positive_excess_raw * 10000, expected * scale)
    saturated = any(
        value > MAX_SIGNED_63
        for value in (
            raw_expected_fixed,
            raw_observed_fixed,
            raw_allowed,
            abs(raw_drift),
            raw_merge_error,
            raw_composite,
            raw_ratio_bps,
        )
    )

    expected_fixed = saturate(raw_expected_fixed)
    observed_fixed = saturate(raw_observed_fixed)
    allowed_fixed = saturate(raw_allowed)
    drift_signed_fixed = saturate(raw_drift)
    drift_excess_fixed = max(0, drift_signed_fixed)
    drift_ratio_bps = saturate(raw_ratio_bps)
    error_bound = ErrorBound(
        observation_quantization=scale,
        tolerance_comparison=1,
        merge_deviation=saturate(raw_merge_error),
        composite=saturate(raw_composite),
        saturated=saturated,
    )
    upper_boundary = sat_add(allowed_fixed, error_bound.composite)
    lower_boundary = sat_add(drift_excess_fixed, error_bound.composite)

    if drift_excess_fixed > upper_boundary:
        decision = "REJECT"
        error_affected = False
        reason = "excess exceeds tolerance plus composite error bound"
    elif lower_boundary <= allowed_fixed:
        decision = "ACCEPT"
        error_affected = False
        reason = "excess plus composite error bound remains within tolerance"
    else:
        error_affected = True
        if criteria.boundary_rule == "conservative_reject":
            decision = "REJECT"
            reason = "ambiguous boundary; conservative rule rejects"
        else:
            decision = "ACCEPT" if drift_excess_fixed <= allowed_fixed else "REJECT"
            reason = "ambiguous boundary; stable tolerance-floor tie rule applied"

    return DriftResult(
        snapshot_seq=snapshot_seq,
        host_id=host_id,
        metric_id=metric_id,
        status="DRIFT" if decision == "REJECT" else "STABLE",
        decision=decision,
        expected=expected,
        observed=observed,
        tolerance_bps=tolerance_bps,
        drift_signed_fixed=drift_signed_fixed,
        drift_excess_fixed=drift_excess_fixed,
        drift_ratio_bps=drift_ratio_bps,
        allowed_fixed=allowed_fixed,
        error_bound=error_bound,
        error_affected=error_affected,
        boundary_rule=criteria.boundary_rule,
        saturated=saturated,
        reason=reason,
    )


def judge_snapshot(merged_results: list[MergedResult], snapshot_hosts: dict[str, HostSpec], criteria: Criteria) -> list[DriftResult]:
    results: list[DriftResult] = []
    for merged in merged_results:
        host_spec = snapshot_hosts[merged.host_id]
        results.append(
            judge_drift(
                merged.snapshot_seq,
                merged.host_id,
                merged.metric_id,
                host_spec,
                merged,
                criteria,
            )
        )
    return results
