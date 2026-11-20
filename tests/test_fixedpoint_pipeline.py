from fractions import Fraction

from rerank_core.fixedpoint import isqrt_frac, quantize, round_half_away
from rerank_core.normalize import (
    NormalizationStats,
    normalize_value,
    quantize_value,
)
from rerank_core.rerank import isqrt_int, similarity_fixed


def test_round_half_away_is_symmetric():
    assert round_half_away(Fraction(5, 2)) == 3
    assert round_half_away(Fraction(-5, 2)) == -3
    assert round_half_away(Fraction(2, 3)) == 1


def test_quantize_changes_with_frac_bits():
    value = Fraction(1, 3)
    assert quantize(value, 2) == 33
    assert quantize(value, 4) == 3333
    assert quantize(value, 6) == 333333


def test_quant_levels_hard_cap():
    assert quantize_value(Fraction(200), 4, 256) == 1_280_000
    assert quantize_value(Fraction(-200), 4, 256) == -1_280_000


def test_minmax_constant_feature_is_zero():
    stat = NormalizationStats("r", "minmax", 0, "3/1", "3/1", None, None, True)
    assert normalize_value(Fraction(3), stat, "minmax") == Fraction(0)


def test_exact_sqrt():
    assert isqrt_frac(Fraction(9, 16)) == Fraction(3, 4)
    assert isqrt_int(15) == 3 and isqrt_int(16) == 4


def test_fixed_cosine_orthogonal_and_identical():
    assert similarity_fixed([10000, 0], [0, 10000], 10000, 256) == 0
    assert similarity_fixed([10000, 0], [10000, 0], 10000, 256) == 10000
