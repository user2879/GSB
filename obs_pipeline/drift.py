"""Reference vs observation window statistics and drift verdicts (requirement 6).

Per metric, over each window we aggregate (aggregation order is explicit:
records sorted by obs_seq ascending):
  mean          - exact rational mean of non-null fixed-point values
  bucket counts - quantile-bucket histogram with the stage's bucket params
  missing rate  - exact rational null share

Drift is the absolute difference between observation and reference stats.
Classification combines drift with the composed error bound (in final-stage
ulp, converted to the statistic's units):
  drift - bound >  threshold  -> CERTAIN_EXCEEDED
  drift + bound <= threshold  -> CERTAIN_NOT_EXCEEDED
  otherwise                   -> ERROR_INDETERMINATE: the stable tie-break
                                 rule (drift >= threshold => EXCEEDED else
                                 NOT_EXCEEDED) decides, and the verdict is
                                 flagged error_affected=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

CERTAIN_EXCEEDED = "certain-exceeded"
CERTAIN_NOT_EXCEEDED = "certain-not-exceeded"
ERROR_INDETERMINATE = "error-indeterminate"

AGGREGATION_ORDER = "obs_seq-ascending"


@dataclass(frozen=True)
class WindowMetricStats:
    metric_key: str
    window_id: str
    count: int
    null_count: int
    mean: Fraction | None
    missing_rate: Fraction
    bucket_counts: tuple
    aggregation_order: str

    def canonical(self):
        mean = "-" if self.mean is None else str(self.mean)
        return "%s|%s|%d|%d|%s|%s|%s" % (
            self.metric_key,
            self.window_id,
            self.count,
            self.null_count,
            mean,
            str(self.missing_rate),
            ",".join(str(c) for c in self.bucket_counts),
        )


@dataclass(frozen=True)
class DriftVerdict:
    metric_key: str
    reference_window: str
    observation_window: str
    mean_drift: Fraction | None
    bucket_drift: int
    missing_rate_drift: Fraction
    drift_amount: Fraction  # combined drift measure compared to the threshold
    threshold: Fraction
    error_bound: Fraction
    cause: str
    exceeded: bool
    error_affected: bool
    basis: str


def compute_window_stats(records, window, metric_key, buckets, lo, hi):
    """Aggregate one metric over one window. records: seq-sorted slice."""
    values = [r.value_fixed for r in records if r.metric_key == metric_key]
    total = len(values)
    non_null = [v for v in values if v is not None]
    null_count = total - len(non_null)
    mean = Fraction(sum(non_null), len(non_null)) if non_null else None
    missing_rate = Fraction(null_count, total) if total else Fraction(0)
    counts = [0] * buckets
    span = hi - lo
    for value in non_null:
        index = (value - lo) * buckets // span
        counts[min(max(index, 0), buckets - 1)] += 1
    return WindowMetricStats(
        metric_key=metric_key,
        window_id=window.id,
        count=total,
        null_count=null_count,
        mean=mean,
        missing_rate=missing_rate,
        bucket_counts=tuple(counts),
        aggregation_order=AGGREGATION_ORDER,
    )


def judge_drift(reference, observation, threshold, error_bound):
    """Compare two WindowMetricStats for the same metric.

    threshold and error_bound are in fixed-point ulp of the record scale.
    """
    if reference.mean is None or observation.mean is None:
        mean_drift = None
        mean_term = Fraction(0)
    else:
        mean_drift = abs(observation.mean - reference.mean)
        mean_term = mean_drift
    bucket_drift = max(
        (abs(o - r) for o, r in zip(observation.bucket_counts, reference.bucket_counts)),
        default=0,
    )
    missing_drift = abs(observation.missing_rate - reference.missing_rate)
    # Combined drift measure: mean drift dominates; bucket/missing terms are
    # secondary and added in fixed-point units of the record scale.
    drift_amount = mean_term + missing_drift
    threshold = Fraction(threshold)
    error_bound = Fraction(error_bound)

    if drift_amount - error_bound > threshold:
        cause, exceeded, affected = CERTAIN_EXCEEDED, True, False
    elif drift_amount + error_bound <= threshold:
        cause, exceeded, affected = CERTAIN_NOT_EXCEEDED, False, False
    else:
        cause = ERROR_INDETERMINATE
        affected = True
        # Stable tie-break rule, declared: compare raw drift to threshold.
        exceeded = drift_amount > threshold
    basis = (
        "drift=%s threshold=%s bound=%s order=%s rule=%s"
        % (drift_amount, threshold, error_bound, AGGREGATION_ORDER,
           "drift-bound>threshold => exceeded; drift+bound<=threshold => not-exceeded; "
           "else tie-break drift>threshold, flagged error-affected")
    )
    return DriftVerdict(
        metric_key=observation.metric_key,
        reference_window=reference.window_id,
        observation_window=observation.window_id,
        mean_drift=mean_drift,
        bucket_drift=bucket_drift,
        missing_rate_drift=missing_drift,
        drift_amount=drift_amount,
        threshold=threshold,
        error_bound=error_bound,
        cause=cause,
        exceeded=exceeded,
        error_affected=affected,
        basis=basis,
    )
