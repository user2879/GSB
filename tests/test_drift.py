"""Requirement 6: window stats, drift verdicts, three-way cause split and the
stable tie-break rule."""

from fractions import Fraction

from obs_pipeline.drift import (
    CERTAIN_EXCEEDED,
    CERTAIN_NOT_EXCEEDED,
    ERROR_INDETERMINATE,
    compute_window_stats,
    judge_drift,
)
from obs_pipeline.records import NormalizedRecord
from obs_pipeline.spec import Window


def _record(seq, metric, value):
    return NormalizedRecord(3, seq, "src", seq, metric, 100, value, (), ())


def _window(wid="w0"):
    return Window(wid, 0, 4, 0)


def test_stats_exact_and_ordered():
    records = [
        _record(1, "m0", 100), _record(2, "m0", None),
        _record(3, "m0", 300), _record(4, "m1", 999),
    ]
    stats = compute_window_stats(records, _window(), "m0", 4, 0, 400)
    assert stats.count == 3
    assert stats.null_count == 1
    assert stats.mean == Fraction(400, 2)
    assert stats.missing_rate == Fraction(1, 3)
    assert sum(stats.bucket_counts) == 2
    assert stats.aggregation_order == "obs_seq-ascending"
    assert stats.canonical() == stats.canonical()


def _stats(window_id, mean_num, mean_den, null_num, null_den, buckets):
    from obs_pipeline.drift import WindowMetricStats

    return WindowMetricStats(
        metric_key="m0", window_id=window_id, count=4,
        null_count=null_num, mean=Fraction(mean_num, mean_den),
        missing_rate=Fraction(null_num, null_den), bucket_counts=tuple(buckets),
        aggregation_order="obs_seq-ascending",
    )


def test_certain_exceeded():
    verdict = judge_drift(_stats("w0", 0, 1, 0, 4, [4, 0, 0, 0]),
                          _stats("w1", 1000, 1, 0, 4, [4, 0, 0, 0]),
                          threshold=100, error_bound=Fraction(10))
    assert verdict.cause == CERTAIN_EXCEEDED
    assert verdict.exceeded and not verdict.error_affected


def test_certain_not_exceeded():
    verdict = judge_drift(_stats("w0", 0, 1, 0, 4, [4, 0, 0, 0]),
                          _stats("w1", 50, 1, 0, 4, [4, 0, 0, 0]),
                          threshold=100, error_bound=Fraction(10))
    assert verdict.cause == CERTAIN_NOT_EXCEEDED
    assert not verdict.exceeded and not verdict.error_affected


def test_error_indeterminate_uses_stable_tiebreak():
    # drift=105, bound=10, threshold=100: 95 <= 100 < 115 -> indeterminate
    verdict = judge_drift(_stats("w0", 0, 1, 0, 4, [4, 0, 0, 0]),
                          _stats("w1", 105, 1, 0, 4, [4, 0, 0, 0]),
                          threshold=100, error_bound=Fraction(10))
    assert verdict.cause == ERROR_INDETERMINATE
    assert verdict.error_affected
    assert verdict.exceeded  # tie-break: raw drift 105 > 100
    # Same band but raw drift below threshold -> not exceeded, still flagged
    verdict2 = judge_drift(_stats("w0", 0, 1, 0, 4, [4, 0, 0, 0]),
                           _stats("w1", 95, 1, 0, 4, [4, 0, 0, 0]),
                           threshold=100, error_bound=Fraction(10))
    assert verdict2.cause == ERROR_INDETERMINATE
    assert not verdict2.exceeded and verdict2.error_affected
    assert "tie-break" in verdict.basis
