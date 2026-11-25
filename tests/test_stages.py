"""Requirements 3+4: integer fixed-point transforms, declared truncation,
null propagation rules, byte-identical summaries across batch orders."""

import random

from obs_pipeline.spec import Stage
from obs_pipeline.spec import Window
from obs_pipeline.stages import NULL_RULES, SEMANTICS, run_stage, summarize_outputs


def _stage(kind, params, scale=100, sid="s"):
    return Stage(sid, kind, params, scale, (), 0)


def _window(wid="w0"):
    return Window(wid, 0, 8, 0)


def test_std_diff_null_propagation():
    result = run_stage(_stage("std_diff", {}), _window(), (10, 15, None, 7, 9, None, None, 3))
    assert result.outputs == (None, 5, None, None, 2, None, None, None)
    assert result.null_count == 6
    assert "propagate" in result.null_rule


def test_window_avg_skips_nulls_floor_division():
    result = run_stage(_stage("window_avg", {"window": 3}), _window(),
                       (1, 2, None, 2, -7, 10, None, None))
    # i0: [1] -> 1 ; i1: [1,2] -> 1 ; i2: [1,2] -> 1 ; i3: [2,2] -> 2
    # i4: [2,-7] -> -5//2 = -3 ; i5: [2,-7,10] -> 5//3 = 1
    # i6: [-7,10] -> 3//2 = 1 ; i7: [10] -> 10
    assert result.outputs == (1, 1, 1, 2, -3, 1, 1, 10)
    assert result.null_count == 0
    # floor toward negative infinity, not truncation toward zero
    neg = run_stage(_stage("window_avg", {"window": 2}), _window(), (-7, 2, 0, 0, 0, 0, 0, 0))
    assert neg.outputs[1] == -3  # -5 // 2 == -3 (floor), not -2


def test_window_avg_all_null_window_is_null():
    result = run_stage(_stage("window_avg", {"window": 2}), _window(),
                       (None, None, 5, 5, 5, 5, 5, 5))
    assert result.outputs[0] is None and result.outputs[1] is None
    assert "skip" in result.null_rule


def test_quantile_bucket_clamps_and_counts_nulls():
    result = run_stage(_stage("quantile_bucket", {"buckets": 4, "lo": 0, "hi": 100}),
                       _window(), (-5, 0, 25, 50, 99, 100, 250, None))
    assert result.outputs == (0, 0, 1, 2, 3, 3, 3, None)
    assert result.null_count == 1


def test_threshold_norm_saturates():
    result = run_stage(_stage("threshold_norm", {"threshold": 10, "span": 5}, scale=1000),
                       _window(), (10, 12, 20, -100, None, 15, 14, 13))
    assert result.outputs == (0, 400, 1000, 0, None, 1000, 800, 600)


def test_semantics_declared_for_all_kinds():
    for kind in ("std_diff", "window_avg", "quantile_bucket", "threshold_norm"):
        assert set(SEMANTICS[kind]) == {"division", "saturation", "negative"}
        assert kind in NULL_RULES


def test_summary_independent_of_batch_execution_order():
    stage = _stage("window_avg", {"window": 3})
    rng = random.Random(7)
    values = tuple(rng.randint(-50, 50) for _ in range(24))
    windows = [Window("w%d" % i, i * 8, (i + 1) * 8, i) for i in range(3)]
    slices = {w.id: values[w.start:w.end] for w in windows}
    forward = [run_stage(stage, w, slices[w.id]).summary for w in windows]
    reverse = [run_stage(stage, w, slices[w.id]).summary for w in reversed(windows)]
    assert sorted(forward) == sorted(reverse)
    assert forward[0] != forward[1]  # summaries actually differ per window
    assert summarize_outputs((1, None, 2)) == summarize_outputs((1, None, 2))
