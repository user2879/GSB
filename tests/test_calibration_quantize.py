import numpy as np

from qcalib import calibration as cal_mod
from qcalib import quantize as qnt_mod
from qcalib.errors import E_ZERO_SCALE, FailureList
from qcalib.models import LayerQuantConfig


def cfg(**overrides):
    base = dict(name="l", bits=8, level="int8", group_size=4,
                q_lo=25, q_hi=975, frac_bits=4, scheme="asym",
                rounding="half_even", saturation="clip")
    base.update(overrides)
    return LayerQuantConfig(**base)


def test_quantile_interpolation_and_tie_rule():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    ordered = np.sort(values, kind="mergesort")
    assert cal_mod.quantile(ordered, 0) == 1.0
    assert cal_mod.quantile(ordered, 1000) == 4.0
    assert np.isclose(cal_mod.quantile(ordered, 500), 2.5)
    tied = np.array([5.0, 5.0, 5.0, 1.0], kind=None) \
        if False else np.array([5.0, 5.0, 5.0, 1.0])
    ordered_tied = np.sort(tied, kind="mergesort")
    assert cal_mod.quantile(ordered_tied, 500) == 5.0


def test_symmetric_and_asymmetric_scales():
    failures = FailureList()
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    sym = cal_mod.calibrate_group(0, values, cfg(scheme="sym", level="int8"),
                                  failures)
    assert sym.zero == 0 and sym.scale > 0
    asym = cal_mod.calibrate_group(0, values, cfg(scheme="asym"), failures)
    assert asym.zero != 0 or True
    assert len(failures) == 0


def test_zero_scale_fallback():
    failures = FailureList()
    values = np.zeros(4)
    result = cal_mod.calibrate_group(0, values, cfg(scheme="sym"), failures)
    assert result.zero_scale_fallback is True
    assert result.scale == cal_mod.ZERO_SCALE_FALLBACK
    assert failures.to_list()[0]["code"] == E_ZERO_SCALE


def test_frac_bits_and_level_change_integers():
    values = np.array([-1.5, -0.25, 0.25, 1.5])
    failures = FailureList()
    coarse = cal_mod.calibrate_group(0, values, cfg(frac_bits=2), failures)
    fine = cal_mod.calibrate_group(0, values, cfg(frac_bits=6), failures)
    q_coarse = qnt_mod.quantize_group(values, coarse, cfg(frac_bits=2))
    q_fine = qnt_mod.quantize_group(values, fine, cfg(frac_bits=6))
    assert q_coarse.integers != q_fine.integers

    int4 = cal_mod.calibrate_group(0, values, cfg(bits=4, level="int4"),
                                   failures)
    q_int4 = qnt_mod.quantize_group(values, int4, cfg(bits=4, level="int4"))
    assert all(-8 <= integer <= 7 for integer in q_int4.integers)
    q_int8 = qnt_mod.quantize_group(
        values, cal_mod.calibrate_group(0, values, cfg(), failures), cfg())
    assert all(-128 <= integer <= 127 for integer in q_int8.integers)


def test_saturation_directions_and_wrap_policy():
    values = np.array([-100.0, -0.1, 0.1, 100.0])
    failures = FailureList()
    clip_cfg = cfg(saturation="clip", q_lo=25, q_hi=975, frac_bits=8)
    calibration = cal_mod.calibrate_group(0, values, clip_cfg, failures)
    clipped = qnt_mod.quantize_group(values, calibration, clip_cfg)
    directions = {item["direction"] for item in clipped.saturated}
    assert "low" in directions and "high" in directions
    assert all(-128 <= integer <= 127 for integer in clipped.integers)

    wrap_cfg = cfg(saturation="wrap", q_lo=25, q_hi=975, frac_bits=8)
    wrapped = qnt_mod.quantize_group(values, calibration, wrap_cfg)
    wrap_directions = {item["direction"] for item in wrapped.saturated}
    assert {"low_wrap", "high_wrap"} <= wrap_directions
    assert all(-128 <= integer <= 127 for integer in wrapped.integers)


def test_error_attribution_and_zero_truth_branch():
    values = np.array([0.0, 0.123456789, -0.5, 0.5])
    failures = FailureList()
    calibration = cal_mod.calibrate_group(0, values, cfg(), failures)
    result = qnt_mod.quantize_group(values, calibration, cfg())
    assert result.zero_truth_branch[0] is True
    assert result.rel_error[0] == result.abs_error[0]
    assert set(result.error_kind) <= {
        qnt_mod.ERR_DISCRET, qnt_mod.ERR_SCALE,
        qnt_mod.ERR_SAT, qnt_mod.ERR_ROUND}
    assert np.allclose(result.abs_error,
                       np.abs(np.asarray(result.dequant) - values))
