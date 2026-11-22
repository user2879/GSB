from __future__ import annotations

import numpy as np
import pytest

from cqdiag.calibration import (
    calibrate_column,
    derive_scale_zero_point,
    quant_range,
    quantile_permille,
)


def test_quantile_linear_interp():
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert quantile_permille(vals, 0) == 0.0
    assert quantile_permille(vals, 1000) == 4.0
    assert quantile_permille(vals, 500) == 2.0
    assert quantile_permille(vals, 250) == 1.0
    assert quantile_permille(vals, 125) == 0.5  # 插值中点


def test_quantile_ties_stable():
    vals = np.array([1.0, 1.0, 1.0, 5.0])
    assert quantile_permille(vals, 500) == 1.0  # 并列值由稳定排序+插值确定


def test_calibrate_excludes_missing():
    values = np.array([1.0, 2.0, 3.0, 99.0])
    missing = np.array([False, False, False, True])
    calib = calibrate_column(values, missing, 0, 1000)
    assert calib["valid_count"] == 3
    assert calib["total_count"] == 4
    assert calib["col_max"] == 3.0  # 缺失的 99 不参与
    assert calib["element_refs"] == [0, 1, 2]
    assert calib["meta"]["interpolation"] == "linear"


def test_symmetric_scale():
    calib = {"q_lo": -3.0, "q_hi": 4.0}
    out = derive_scale_zero_point(calib, "symmetric", 8)
    qmin, qmax = quant_range(8)
    assert out["zero_point"] == 0
    assert out["scale"] == pytest.approx(4.0 / qmax)


def test_asymmetric_scale_zp():
    calib = {"q_lo": -1.0, "q_hi": 3.0}
    out = derive_scale_zero_point(calib, "asymmetric", 8)
    qmin, qmax = quant_range(8)
    assert out["scale"] == pytest.approx(4.0 / (qmax - qmin))
    # zp = qmin - q_lo/scale, 裁剪在 [qmin, qmax]
    assert qmin <= out["zero_point"] <= qmax
    # 反量化边界还原 (误差在一个量化步长内)
    lo = (qmin - out["zero_point"]) * out["scale"]
    hi = (qmax - out["zero_point"]) * out["scale"]
    assert lo <= -1.0 + out["scale"] and hi >= 3.0 - out["scale"]


def test_asymmetric_zp_clipped_to_range():
    calib = {"q_lo": 1000.0, "q_hi": 1001.0}  # 区间远离原点
    out = derive_scale_zero_point(calib, "asymmetric", 4)
    qmin, qmax = quant_range(4)
    assert out["zero_point"] == qmin  # 被裁剪到下界


def test_bits_change_range():
    calib = {"q_lo": -1.0, "q_hi": 1.0}
    s4 = derive_scale_zero_point(calib, "symmetric", 4)
    s8 = derive_scale_zero_point(calib, "symmetric", 8)
    assert s4["scale"] != s8["scale"]  # 位宽真实改变缩放因子
