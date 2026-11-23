import numpy as np
import pytest

from voicecalib.fixedpoint import (ALLOWED_FRAC_BITS, ALLOWED_QUANT_GEARS,
                                   FixedPoint, half_even_div)


def test_half_even_rounding():
    assert half_even_div(3, 2) == 2
    assert half_even_div(5, 2) == 2
    assert half_even_div(4, 2) == 2
    assert half_even_div(-3, 2) == -2
    assert half_even_div(-5, 2) == -2


def test_quantize_residual_and_clamp():
    fp = FixedPoint(8, 64)
    q, residual = fp.quantize(1.0 / 3.0)
    assert q == 85
    assert np.isclose(residual, 1 / 3 - 85 / 256, atol=1e-9)
    # 档位 64 headroom=1，超界饱和。
    q_sat, _ = fp.quantize(2.5)
    assert q_sat == fp.limit == 256


def test_allowed_enums():
    with pytest.raises(ValueError):
        FixedPoint(10, 256)
    with pytest.raises(ValueError):
        FixedPoint(12, 128)
    assert ALLOWED_FRAC_BITS == (8, 12, 16)
    assert ALLOWED_QUANT_GEARS == (64, 256, 1024)


def test_mul_scaled_half_even():
    fp = FixedPoint(12, 256)
    a = fp.quantize(0.8)[0]
    b = fp.quantize(0.3)[0]
    product = fp.mul_scaled(a, b)
    assert product == int(np.round(0.8 * 0.3 * fp.scale))
