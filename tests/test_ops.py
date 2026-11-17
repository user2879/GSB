from __future__ import annotations

import numpy as np

from softmax_layernorm.ops import layer_norm_lane, softmax_lane


def test_stable_softmax_is_deterministic_and_normalized() -> None:
    values = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
    first = softmax_lane(values, stabilize=True, lane=3, row=3)
    second = softmax_lane(values, stabilize=True, lane=3, row=3)
    assert first.failure is None
    assert first.stats == {"max_subtraction": True}
    assert first.values.tobytes() == second.values.tobytes()
    assert first.values.dtype == np.float32
    assert np.isclose(float(first.values.sum(dtype=np.float32)), 1.0, atol=2e-7)


def test_unstable_softmax_reports_overflow_and_lane() -> None:
    result = softmax_lane(
        np.array([1000.0, 0.0, 1.0], dtype=np.float32),
        stabilize=False,
        call_id="overflow-call",
        call_sequence=7,
        lane=2,
        row=2,
    )
    assert result.failure is not None
    assert result.failure.code == "E_EXP_OVERFLOW"
    assert result.failure.lane == 2
    assert result.failure.row == 2
    assert result.failure.call_sequence == 7
    assert result.stats == {"max_subtraction": False}


def test_layernorm_outputs_mean_variance_and_epsilon_marker() -> None:
    result = layer_norm_lane(
        np.array([3.0, 3.0, 3.0], dtype=np.float32), epsilon=1e-5, lane=0, row=0
    )
    assert result.failure is None
    assert result.stats["mean"] == np.float32(3.0)
    assert result.stats["variance"] == np.float32(0.0)
    assert result.stats["epsilon_applied"] is True
    assert np.array_equal(result.values, np.zeros(3, dtype=np.float32))


def test_layernorm_zero_variance_zero_epsilon_keeps_statistics() -> None:
    result = layer_norm_lane(
        np.array([2.0, 2.0], dtype=np.float32), epsilon=0.0, lane=1, row=1
    )
    assert result.failure is not None
    assert result.failure.code == "E_ZERO_VARIANCE"
    assert result.stats["mean"] == np.float32(2.0)
    assert result.stats["variance"] == np.float32(0.0)
    assert np.all(np.isnan(result.values))
