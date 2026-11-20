from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from typing import Any, Sequence

from .canonical import as_fraction, fraction_json, sha256_json
from .errors import BucketError
from .sampling import SamplingResult
from .spec import SamplingPolicy

QUANTILE_PERCENTS = (50, 90, 95, 99)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def bucket_cut_values(durations: Sequence[int], bucket_count: int) -> list[int]:
    if not durations:
        raise BucketError("$.durations", "cannot build buckets from an empty record set")
    ordered = sorted(int(value) for value in durations)
    total = len(ordered)
    cuts: list[int] = []
    for index in range(1, bucket_count):
        rank = _ceil_div(index * total, bucket_count)
        cut_index = min(total - 1, max(0, rank - 1))
        cuts.append(ordered[cut_index])
    return cuts


def bucket_bounds(
    durations: Sequence[int], cuts: Sequence[int]
) -> list[tuple[int, int]]:
    if not durations:
        raise BucketError("$.durations", "cannot build bucket bounds from an empty record set")
    minimum = min(durations)
    maximum = max(durations)
    bounds: list[tuple[int, int]] = []
    for index in range(len(cuts) + 1):
        lower = minimum if index == 0 else cuts[index - 1]
        upper = cuts[index] if index < len(cuts) else maximum + 1
        bounds.append((lower, upper))
    return bounds


def fixed_quantile_ticks(sorted_ticks: Sequence[int], percent: int) -> int:
    if not sorted_ticks:
        raise BucketError("$.durations", "cannot compute a quantile from an empty record set")
    count = len(sorted_ticks)
    position = Fraction((count - 1) * percent, 100)
    lower_index = position.numerator // position.denominator
    upper_index = min(count - 1, lower_index + 1)
    remainder = position - lower_index
    interpolated = Fraction(sorted_ticks[lower_index]) * (1 - remainder) + Fraction(
        sorted_ticks[upper_index]
    ) * remainder
    return interpolated.numerator // interpolated.denominator


def _midpoint(lower: int, upper_exclusive: int) -> Fraction:
    return Fraction(lower + upper_exclusive - 1, 2)


def _half_width_bound(lower: int, upper_exclusive: int) -> Fraction:
    width = max(0, upper_exclusive - lower)
    return Fraction(max(0, width - 1), 2)


def _fraction_share(value: Fraction, total: Fraction) -> dict[str, int]:
    if total == 0:
        return fraction_json(Fraction(0))
    return fraction_json(value / total)


def dominant_segment(
    values: dict[str, Fraction], *, threshold: Fraction = Fraction(1, 2)
) -> str:
    total = sum(values.values(), Fraction(0))
    if total == 0:
        return "none"
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    top_name, top_value = ordered[0]
    if top_value / total >= threshold and all(
        top_value > other_value for _, other_value in ordered[1:]
    ):
        return top_name
    return "balanced"


def analyze_policy(
    *,
    service_id: str,
    records: Sequence[dict[str, Any]],
    sampling: SamplingResult,
    policy: SamplingPolicy,
    strategy_version: int,
) -> dict[str, Any]:
    if not records:
        raise BucketError(
            f"$.services[{service_id}].records",
            "at least one record is required",
        )
    total_count = len(records)
    quantum = policy.quantum_us
    durations = [int(record["duration_us"]) for record in records]
    cuts = bucket_cut_values(durations, policy.bucket_count)
    bounds = bucket_bounds(durations, cuts)

    bucket_records: list[list[int]] = [[] for _ in bounds]
    for index, record in enumerate(records):
        bucket_index = bisect_right(cuts, int(record["duration_us"]))
        bucket_records[bucket_index].append(index)

    sample_hits_total = len(sampling.hits)
    if sample_hits_total == 0:
        raise BucketError(
            f"$.policies[{policy.id}].sample_hits",
            "policy selects no records; estimated distribution is undefined",
        )
    reweight = Fraction(total_count, sample_hits_total)

    bucket_stats: list[dict[str, Any]] = []
    attribution_buckets: list[dict[str, Any]] = []
    estimate_total = Fraction(0)
    true_total = Fraction(sum(durations))
    for index, ((lower, upper), members) in enumerate(zip(bounds, bucket_records)):
        member_durations = [durations[member] for member in members]
        hit_members = [member for member in members if member in sampling.hits]
        head_hits = sum(
            1
            for member in members
            if sampling.decisions[member].reason == "head"
        )
        tail_hits = sum(
            1
            for member in members
            if sampling.decisions[member].reason == "tail"
        )
        sampled_true_sum = sum(durations[member] for member in hit_members)
        sampled_fixed_sum = sum(
            (durations[member] // quantum) * quantum for member in hit_members
        )
        true_sum = sum(member_durations)
        midpoint = _midpoint(lower, upper)
        estimated_bucket_total = Fraction(len(hit_members)) * midpoint * reweight
        estimate_total += estimated_bucket_total

        bucket_stat = {
            "bucket_index": index,
            "lower_us": lower,
            "upper_us_exclusive": upper,
            "boundary_rule": "[lower, upper); final bucket includes max_us",
            "record_count": len(members),
            "sample_hits": len(hit_members),
            "head_hits": head_hits,
            "tail_hits": tail_hits,
            "duration_min_us": min(member_durations) if member_durations else None,
            "duration_max_us": max(member_durations) if member_durations else None,
            "duration_sum_us": true_sum,
            "sampled_duration_sum_us": sampled_true_sum,
            "sampled_fixed_sum_us": sampled_fixed_sum,
            "record_trace_digest": sha256_json(
                sorted(records[member]["trace_id"] for member in members)
            ),
            "sample_trace_digest": sha256_json(
                sorted(records[member]["trace_id"] for member in hit_members)
            ),
        }
        bucket_stat["bucket_digest"] = sha256_json(bucket_stat)
        bucket_stats.append(bucket_stat)

        sampling_signed = Fraction(sampled_true_sum) * reweight - Fraction(true_sum)
        fixed_signed = (
            Fraction(sampled_fixed_sum) * reweight
            - Fraction(sampled_true_sum) * reweight
        )
        boundary_signed = (
            estimated_bucket_total - Fraction(sampled_fixed_sum) * reweight
        )
        sampling_bound = abs(sampling_signed)
        fixed_bound = Fraction(len(hit_members) * (quantum - 1)) * reweight
        boundary_bound = (
            sum(
                abs(Fraction((durations[member] // quantum) * quantum) - midpoint)
                for member in hit_members
            )
            * reweight
        )
        total_bound = sampling_bound + fixed_bound + boundary_bound
        estimated_minus_true = estimated_bucket_total - Fraction(true_sum)
        decomposition_sum = sampling_signed + fixed_signed + boundary_signed
        if decomposition_sum != estimated_minus_true:
            raise AssertionError("bucket attribution does not add up")
        contributions = {
            "sampling_sparse": sampling_signed,
            "fixed_truncation": fixed_signed,
            "bucket_boundary": boundary_signed,
        }
        contribution_bounds = {
            "sampling_sparse": sampling_bound,
            "fixed_truncation": fixed_bound,
            "bucket_boundary": boundary_bound,
        }
        attribution_buckets.append(
            {
                "bucket_index": index,
                "lower_us": lower,
                "upper_us_exclusive": upper,
                "estimated_count": fraction_json(Fraction(len(hit_members)) * reweight),
                "estimated_total_us": fraction_json(estimated_bucket_total),
                "full_total_us": fraction_json(Fraction(true_sum)),
                "estimated_minus_true_us": fraction_json(estimated_minus_true),
                "contributions_us": {
                    name: fraction_json(value) for name, value in contributions.items()
                },
                "contribution_bounds_us": {
                    name: fraction_json(value)
                    for name, value in contribution_bounds.items()
                },
                "total_error_bound_us": fraction_json(total_bound),
                "error_abs_us": fraction_json(abs(estimated_minus_true)),
                "dominant_segment": dominant_segment(contribution_bounds),
                "shares": {
                    name: _fraction_share(value, total_bound)
                    for name, value in contribution_bounds.items()
                },
            }
        )

    ordered_durations = sorted(durations)
    ordered_ticks = [duration // quantum for duration in ordered_durations]
    bucket_weights = [
        Fraction(stat["sample_hits"] * total_count, sample_hits_total)
        for stat in bucket_stats
    ]
    cdf_error = Fraction(0)
    sampling_rank_deviation = Fraction(0)
    for stat, weight in zip(bucket_stats, bucket_weights):
        cdf_error += weight - int(stat["record_count"])
        sampling_rank_deviation = max(sampling_rank_deviation, abs(cdf_error))
    quantiles: dict[str, Any] = {}
    total_quantization_bound = Fraction(2 * (quantum - 1))
    for percent in QUANTILE_PERCENTS:
        full_ticks = fixed_quantile_ticks(ordered_ticks, percent)
        full_fixed_us = full_ticks * quantum
        target = Fraction(percent * total_count, 100)
        cumulative = Fraction(0)
        selected_bucket = 0
        for bucket_index, weight in enumerate(bucket_weights):
            cumulative += weight
            if cumulative >= target:
                selected_bucket = bucket_index
                break
        else:
            selected_bucket = len(bucket_weights) - 1
        lower, upper = bounds[selected_bucket]
        midpoint = _midpoint(lower, upper)
        estimated_fixed_us = (midpoint // quantum) * quantum
        bucket_bound = _half_width_bound(lower, upper)

        rank = Fraction((total_count - 1) * percent, 100)
        lower_rank = max(
            0,
            (rank - sampling_rank_deviation).numerator
            // (rank - sampling_rank_deviation).denominator,
        )
        upper_rank_value = rank + sampling_rank_deviation
        upper_rank = min(
            total_count - 1,
            -((-upper_rank_value.numerator) // upper_rank_value.denominator),
        )
        band_low_us = ordered_durations[lower_rank]
        band_high_us = ordered_durations[upper_rank]
        sampling_gap = max(
            abs(Fraction(estimated_fixed_us - band_low_us)),
            abs(Fraction(estimated_fixed_us - band_high_us)),
        )
        sampling_bound = max(Fraction(0), sampling_gap - bucket_bound)
        total_bound = sampling_bound + total_quantization_bound + bucket_bound
        difference = estimated_fixed_us - full_fixed_us
        quantiles[f"p{percent}"] = {
            "percent": percent,
            "full_fixed_us": full_fixed_us,
            "estimated_fixed_us": estimated_fixed_us,
            "difference_us": fraction_json(Fraction(difference)),
            "sampling_bound_us": fraction_json(sampling_bound),
            "quantization_bound_us": fraction_json(total_quantization_bound),
            "bucket_bound_us": fraction_json(bucket_bound),
            "total_error_bound_us": fraction_json(total_bound),
            "sampling_rank_deviation_records": fraction_json(sampling_rank_deviation),
            "full_rank_band": [lower_rank, upper_rank],
            "estimated_bucket_index": selected_bucket,
            "estimated_value_in_error_bound": abs(difference) <= total_bound,
            "full_rule": "fixed point rank interpolation, floor to quantum",
            "estimated_rule": "reweighted bucket occupancy, bucket midpoint, floor to quantum",
            "bound_rule": "sampling CDF rank band + two-sided fixed truncation + selected bucket half-width",
        }

    estimated_mean = estimate_total / total_count
    full_mean = Fraction(sum(durations), total_count)

    total_signed = {
        "sampling_sparse": sum(
            (as_fraction(item["contributions_us"]["sampling_sparse"]) for item in attribution_buckets),
            Fraction(0),
        ),
        "fixed_truncation": sum(
            (as_fraction(item["contributions_us"]["fixed_truncation"]) for item in attribution_buckets),
            Fraction(0),
        ),
        "bucket_boundary": sum(
            (as_fraction(item["contributions_us"]["bucket_boundary"]) for item in attribution_buckets),
            Fraction(0),
        ),
    }
    total_bounds = {
        "sampling_sparse": sum(
            (as_fraction(item["contribution_bounds_us"]["sampling_sparse"]) for item in attribution_buckets),
            Fraction(0),
        ),
        "fixed_truncation": sum(
            (as_fraction(item["contribution_bounds_us"]["fixed_truncation"]) for item in attribution_buckets),
            Fraction(0),
        ),
        "bucket_boundary": sum(
            (as_fraction(item["contribution_bounds_us"]["bucket_boundary"]) for item in attribution_buckets),
            Fraction(0),
        ),
    }
    total_bound = sum(total_bounds.values(), Fraction(0))
    estimated_minus_true_total = estimate_total - true_total
    if sum(total_signed.values(), Fraction(0)) != estimated_minus_true_total:
        raise AssertionError("global attribution does not add up")
    attribution_totals = {
        "estimated_total_us": fraction_json(estimate_total),
        "full_total_us": fraction_json(true_total),
        "estimated_minus_true_us": fraction_json(estimated_minus_true_total),
        "contributions_us": {
            name: fraction_json(value) for name, value in total_signed.items()
        },
        "contribution_bounds_us": {
            name: fraction_json(value) for name, value in total_bounds.items()
        },
        "total_error_bound_us": fraction_json(total_bound),
        "error_abs_us": fraction_json(abs(estimated_minus_true_total)),
        "dominant_segment": dominant_segment(total_bounds),
        "shares": {
            name: _fraction_share(value, total_bound)
            for name, value in total_bounds.items()
        },
        "dominant_rule": "share >= 1/2 and strictly greater than every other segment; otherwise balanced",
    }

    bucket_max_bound = max(
        (as_fraction(item["total_error_bound_us"]) for item in attribution_buckets),
        default=Fraction(0),
    )
    fixed_error = {
        "quantization_rule": "floor(duration_us / 10^scale_power)",
        "rounding_rule": "toward negative infinity; durations are nonnegative",
        "scale_power": policy.scale_power,
        "quantum_us": quantum,
        "single_record_max_us": quantum - 1,
        "bucket_max_us": fraction_json(bucket_max_bound),
        "quantile_max_us": fraction_json(
            max(
                (as_fraction(item["total_error_bound_us"]) for item in quantiles.values()),
                default=Fraction(0),
            )
        ),
        "mean_max_us": fraction_json(total_bound / total_count),
        "bucket_boundary_rule": "quantiles use value cuts; equality belongs to the right bucket",
    }

    mean_bound = total_bound / total_count
    result: dict[str, Any] = {
        "policy_id": policy.id,
        "policy_version": policy.version,
        "strategy_version": strategy_version,
        "policy": policy.to_dict(),
        "sample_summary": sampling.summary,
        "bucket_stats": bucket_stats,
        "quantiles": quantiles,
        "fixed_error": fixed_error,
        "attribution": {"buckets": attribution_buckets, "totals": attribution_totals},
        "full_mean_us": fraction_json(full_mean),
        "estimated_mean_us": fraction_json(estimated_mean),
        "estimated_mean_fixed_us": int((estimated_mean // quantum) * quantum),
        "mean_error_bound_us": fraction_json(mean_bound),
    }
    result["result_digest"] = sha256_json(result)
    return result
