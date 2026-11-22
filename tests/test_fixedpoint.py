import numpy as np
import pytest

from mlimgedit.errors import ConfigError
from mlimgedit.fixedpoint import (
    FixedPointQuantizer,
    quantize_probability,
    threshold_integer,
)


def test_quantize_round_trip_and_residual():
    quantizer = FixedPointQuantizer("Q1.15", 15, "HALF_UP")
    values = np.linspace(-1.5, 1.5, 31)
    integers, residual = quantizer.quantize(values)
    restored = quantizer.dequantize(integers) + residual
    np.testing.assert_allclose(restored, values, atol=1e-9)
    assert integers.dtype == np.int64


def test_half_up_is_deterministic_and_stable():
    quantizer = FixedPointQuantizer("Q0.8", 8)
    values = np.asarray([0.5 / 256.0, 1.5 / 256.0, -1.5 / 256.0])
    first_int, first_res = quantizer.quantize(values)
    second_int, second_res = quantizer.quantize(values)
    np.testing.assert_array_equal(first_int, second_int)
    np.testing.assert_array_equal(first_res, second_res)
    assert first_int.tolist() == [1, 2, -1]


def test_trunc_and_half_even_modes_differ_predictably():
    values = np.asarray([0.6 / 256.0])
    trunc = FixedPointQuantizer("Q0.8", 8, "TRUNC").quantize(values)[0]
    half_up = FixedPointQuantizer("Q0.8", 8, "HALF_UP").quantize(values)[0]
    assert int(trunc[0]) == 0
    assert int(half_up[0]) == 1


def test_invalid_enums_rejected():
    with pytest.raises(ConfigError):
        FixedPointQuantizer("Q9.9", 15)
    with pytest.raises(ConfigError):
        FixedPointQuantizer("Q0.8", 15)
    with pytest.raises(ConfigError):
        FixedPointQuantizer("Q1.15", 15, "ROUND_RANDOM")


def test_probability_threshold_integer_compare():
    quantizer = FixedPointQuantizer("Q1.15", 15)
    probability_int = quantize_probability(0.5, quantizer)
    threshold_int = threshold_integer(0.5, quantizer)
    assert probability_int == threshold_int
    assert probability_int >= threshold_int
