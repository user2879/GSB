from __future__ import annotations

import numpy as np

from cqdiag.calibration import quant_range
from cqdiag.quantize import quantize_column, scale_to_fixed


def _run(values, scale, zp, bits, frac_bits, saturation="clamp", missing=None):
    qmin, qmax = quant_range(bits)
    if missing is None:
        missing = np.zeros(len(values), dtype=bool)
    return quantize_column(np.asarray(values, dtype=np.float64), missing,
                           scale, zp, qmin, qmax, frac_bits, saturation)


def test_clip_direction_and_preclip_recorded():
    out = _run([100.0, -100.0, 0.5], scale=1.0, zp=0, bits=4, frac_bits=4)
    qmin, qmax = quant_range(4)
    assert out["q"][0] == qmax and out["clip_dir"][0] == 1
    assert out["q"][1] == qmin and out["clip_dir"][1] == -1
    assert out["clip_dir"][2] == 0
    assert out["pre_clip"][0] == 100  # 截断前定点值被记录
    assert out["pre_clip"][1] == -100
    assert out["clipped_count"] == 2


def test_missing_elements_null():
    missing = np.array([True, False])
    out = _run([1.0, 2.0], scale=1.0, zp=0, bits=8, frac_bits=4, missing=missing)
    assert out["q"][0] is None and out["clip_dir"][0] is None
    assert out["q"][1] == 2


def test_frac_bits_change_intermediates():
    values = [0.333, 1.777, -2.555]
    a = _run(values, scale=0.13, zp=1, bits=8, frac_bits=4)
    b = _run(values, scale=0.13, zp=1, bits=8, frac_bits=12)
    assert a["scale_fp"] != b["scale_fp"]
    assert a["q"] != b["q"] or a["pre_clip"] != b["pre_clip"]


def test_quant_bits_change_clip_set():
    values = [10.0, 5.0, 1.0, -9.0]
    wide = _run(values, scale=1.0, zp=0, bits=8, frac_bits=4)
    narrow = _run(values, scale=1.0, zp=0, bits=4, frac_bits=4)
    assert wide["clipped_count"] == 0
    assert narrow["clipped_count"] == 2  # 10 与 -9 超出 4bit 区间
    assert wide["q"] != narrow["q"]


def test_zero_point_applied():
    out = _run([0.0], scale=0.5, zp=3, bits=8, frac_bits=4)
    assert out["q"][0] == 3


def test_scale_to_fixed_minimum_one():
    assert scale_to_fixed(1e-9, 4) == 1
