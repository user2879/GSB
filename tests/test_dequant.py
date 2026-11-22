from __future__ import annotations

import numpy as np

from cqdiag.dequant import (
    BRANCH_ZERO_TRUTH,
    SOURCE_DEQUANT_ROUND,
    SOURCE_QUANTILE,
    SOURCE_SATURATION,
    attribute_element,
    dequant_column,
)
from cqdiag.calibration import quant_range
from cqdiag.quantize import quantize_column


def _pipeline_column(values, scale, zp, bits, frac_bits, q_lo, q_hi,
                     abs_tol=0.01, rel_tol=0.01, saturation="clamp"):
    qmin, qmax = quant_range(bits)
    vals = np.asarray(values, dtype=np.float64)
    missing = np.zeros(len(values), dtype=bool)
    quant = quantize_column(vals, missing, scale, zp, qmin, qmax, frac_bits, saturation)
    return dequant_column(vals, missing, quant, scale, zp, q_lo, q_hi, abs_tol, rel_tol)


def test_saturation_attribution():
    # 分位区间覆盖真值, 误差只能来自饱和截断
    out = _pipeline_column([100.0], scale=1.0, zp=0, bits=4, frac_bits=8,
                           q_lo=-200.0, q_hi=200.0)
    assert out["oor"][0] is True
    att = out["attributions"][0]
    assert att["source"] == SOURCE_SATURATION
    assert att["contributions"][SOURCE_SATURATION] > 0
    assert "tie_priority" in att["basis"]


def test_dequant_rounding_attribution():
    # scale 可精确定点化 (0.5), x 在分位区间内, 误差纯来自取整
    out = _pipeline_column([0.3], scale=0.5, zp=0, bits=8, frac_bits=8,
                           q_lo=-10.0, q_hi=10.0, abs_tol=0.01, rel_tol=0.01)
    assert out["oor"][0] is True
    att = out["attributions"][0]
    assert att["source"] == SOURCE_DEQUANT_ROUND
    assert att["contributions"][SOURCE_DEQUANT_ROUND] == 0.2


def test_tie_break_priority():
    # 分位离散与反量化舍入贡献并列 (0.25), 优先级 quantile > rounding
    att = attribute_element(x=1.25, q=1, clip_dir=0, zero_point=0,
                            scale=1.0, scale_q=1.0, q_lo=0.0, q_hi=1.0,
                            qmin=-8, qmax=7)
    assert att["contributions"][SOURCE_QUANTILE] == 0.25
    assert att["contributions"][SOURCE_DEQUANT_ROUND] == 0.25
    assert att["source"] == SOURCE_QUANTILE


def test_zero_truth_branch():
    out = _pipeline_column([0.0, 1.0], scale=0.5, zp=0, bits=8, frac_bits=8,
                           q_lo=-2.0, q_hi=2.0)
    assert out["branch"][0] == BRANCH_ZERO_TRUTH
    assert out["rel_err"][0] is None  # 真值为零不计算相对误差
    assert out["oor"][0] is False


def test_mask_strategy_marks_saturated():
    out = _pipeline_column([100.0, 0.25], scale=1.0, zp=0, bits=4, frac_bits=8,
                           q_lo=-5.0, q_hi=5.0, saturation="mask",
                           abs_tol=0.5, rel_tol=0.5)
    assert out["oor"][0] is True
    assert out["branch"][0] == "masked_saturated"
    assert out["attributions"][0]["source"] == SOURCE_SATURATION
    assert out["oor"][1] is False


def test_within_tolerance_not_oor():
    out = _pipeline_column([2.0], scale=0.5, zp=0, bits=8, frac_bits=8,
                           q_lo=-5.0, q_hi=5.0, abs_tol=1.0, rel_tol=1.0)
    assert out["oor"][0] is False
    assert out["attributions"][0] is None
