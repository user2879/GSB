from __future__ import annotations

from .model import ErrorAttribution


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

SEGMENT_PRECEDENCE = {
    "QUANTIZATION": 0,
    "LATE_ADMISSION": 1,
    "WINDOW_BOUNDARY": 2,
}


def clamp_i64(value: int) -> int:
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return value


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = abs(numerator) // denominator
    return quotient if numerator >= 0 else -quotient


def ceil_div_positive(numerator: int, denominator: int) -> int:
    if numerator < 0:
        raise ValueError("numerator must be non-negative")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def quantize(measure: int, scale_factor: int) -> tuple[int, int]:
    exact = measure * scale_factor
    fixed_value = clamp_i64(exact)
    return fixed_value, abs(exact - fixed_value)


def _share_bp(segment: int, total: int) -> int:
    if total <= 0:
        return 0
    return (segment * 10_000 + total // 2) // total


def build_error_attribution(
    *,
    count: int,
    sum_exact: int,
    quantization_error_sum: int,
    max_single_quantization_error: int,
    late_accepted_abs_amount: int,
    boundary_abs_amount: int,
    threshold_bp: int,
) -> tuple[ErrorAttribution, int, int]:
    sum_fixed = clamp_i64(sum_exact)
    sum_saturation_error = abs(sum_exact - sum_fixed)
    mean_exact_toward_zero = trunc_div_toward_zero(sum_fixed, count) if count else 0
    mean_fixed = clamp_i64(mean_exact_toward_zero)
    mean_saturation_error = abs(mean_exact_toward_zero - mean_fixed)

    quantization_segment = clamp_i64(quantization_error_sum + sum_saturation_error)
    late_segment = clamp_i64(late_accepted_abs_amount)
    boundary_segment = clamp_i64(boundary_abs_amount)
    sum_bound = clamp_i64(quantization_segment + late_segment + boundary_segment)
    mean_bound = (
        clamp_i64(ceil_div_positive(sum_bound, count) + 1 + mean_saturation_error)
        if count
        else 0
    )

    segments = {
        "QUANTIZATION": quantization_segment,
        "LATE_ADMISSION": late_segment,
        "WINDOW_BOUNDARY": boundary_segment,
    }
    total = sum(segments.values())
    shares = {name: _share_bp(value, total) for name, value in segments.items()}
    if total == 0:
        dominant = "NONE"
        dominant_share = 0
    else:
        dominant = max(
            segments,
            key=lambda name: (shares[name], -SEGMENT_PRECEDENCE[name]),
        )
        dominant_share = shares[dominant]
        if dominant_share < threshold_bp:
            dominant = "NONE"
    attribution = ErrorAttribution(
        quantization_truncation=quantization_segment,
        late_admission=late_segment,
        window_boundary=boundary_segment,
        sum_bound=sum_bound,
        mean_bound=mean_bound,
        single_value_quantization=clamp_i64(max_single_quantization_error),
        dominant_segment=dominant,
        dominant_share_bp=dominant_share,
        threshold_bp=threshold_bp,
        quantization_share_bp=shares["QUANTIZATION"],
        late_share_bp=shares["LATE_ADMISSION"],
        boundary_share_bp=shares["WINDOW_BOUNDARY"],
    )
    return attribution, sum_fixed, mean_fixed
