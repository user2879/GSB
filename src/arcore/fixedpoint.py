"""Fixed-point statistics with explicit truncation and saturation rules.

Scale factors are positive powers of ten. Quantization truncates toward zero
(floor of the absolute value), and results saturate at +/- ``saturation_max``
instead of overflowing. numpy vectorizes batch quantization while scalar
helpers keep pure-Python call sites simple.
"""

from dataclasses import dataclass

import numpy as np


def _check_scale(scale: int) -> None:
    if scale <= 0 or (10 ** int(round(np.log10(scale))) != scale):
        raise ValueError("scale must be a positive power of 10")


def trunc_toward_zero(value: float) -> int:
    return int(np.trunc(value))


def quantize_scalar(value: float, scale: int,
                    saturation_max: int = 1_000_000) -> int:
    _check_scale(scale)
    if not np.isfinite(value):
        raise ValueError("only finite values can be quantized")
    scaled = trunc_toward_zero(float(value) * scale)
    if scaled > saturation_max:
        return saturation_max
    if scaled < -saturation_max:
        return -saturation_max
    return scaled


def quantize_array(values, scale: int,
                   saturation_max: int = 1_000_000) -> np.ndarray:
    _check_scale(scale)
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("only finite values can be quantized")
    scaled = np.trunc(array * scale).astype(np.int64)
    return np.clip(scaled, -saturation_max, saturation_max)


def quantization_error_bound(scale: int) -> float:
    """Worst-case absolute single-value error after truncation."""
    _check_scale(scale)
    return 1.0 / scale


def ratio_comparison_error_bound(scale: int) -> float:
    """Maximum error when comparing two independently quantized ratios."""
    return 2.0 / scale


@dataclass(frozen=True)
class FixedPointStats:
    scale: int
    saturation_max: int
    truncation: str
    suppressed_ratio_q: int
    correlation_hit_rate_q: int
    average_queue_ms_q: int
    single_value_error_bound: float
    ratio_comparison_error_bound: float

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "saturation_max": self.saturation_max,
            "truncation": self.truncation,
            "suppressed_ratio_q": self.suppressed_ratio_q,
            "correlation_hit_rate_q": self.correlation_hit_rate_q,
            "average_queue_ms_q": self.average_queue_ms_q,
            "single_value_error_bound": self.single_value_error_bound,
            "ratio_comparison_error_bound": self.ratio_comparison_error_bound,
        }


def compute_stats(total_alerts: int, suppressed_count: int,
                  correlated_alert_count: int, queue_wait_ms: list[int],
                  scale: int, saturation_max: int) -> FixedPointStats:
    suppressed_ratio = (suppressed_count / total_alerts
                        if total_alerts else 0.0)
    hit_rate = (correlated_alert_count / total_alerts
                if total_alerts else 0.0)
    average_queue = (sum(queue_wait_ms) / len(queue_wait_ms)
                     if queue_wait_ms else 0.0)
    return FixedPointStats(
        scale=scale,
        saturation_max=saturation_max,
        truncation="toward_zero",
        suppressed_ratio_q=quantize_scalar(
            suppressed_ratio, scale, saturation_max),
        correlation_hit_rate_q=quantize_scalar(
            hit_rate, scale, saturation_max),
        average_queue_ms_q=quantize_scalar(
            average_queue, scale, saturation_max),
        single_value_error_bound=quantization_error_bound(scale),
        ratio_comparison_error_bound=ratio_comparison_error_bound(scale),
    )


@dataclass(frozen=True)
class Misattribution:
    label: str
    measured: int
    bound: int
    explanation: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "measured": self.measured,
            "bound": self.bound,
            "explanation": self.explanation,
        }


def attribute_misjudgments(measured_false_positives: int,
                           measured_misses: int,
                           fp_upper_bound: int,
                           miss_upper_bound: int) -> list[Misattribution]:
    return [
        Misattribution(
            label="approximate_false_positive",
            measured=measured_false_positives,
            bound=fp_upper_bound,
            explanation=(
                "Same-bucket subset evaluation reuses the exact predicate; "
                "any non-zero measured value is an implementation defect "
                "reported as E_DIVERGE.")),
        Misattribution(
            label="approximate_miss",
            measured=measured_misses,
            bound=miss_upper_bound,
            explanation=(
                "Misses are cross-bucket pairs deliberately skipped by "
                "fingerprint bucketing; the structural bound is the "
                "cross-bucket pair count and the measured value is exact.")),
    ]
