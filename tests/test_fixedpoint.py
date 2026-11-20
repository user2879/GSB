from chaos.fixedpoint import (TRUNCATION, quantile_fixed, rate_fixed,
                              summarize, trunc_div)


def test_trunc_toward_zero():
    assert trunc_div(7, 2) == 3
    assert trunc_div(-7, 2) == -3
    assert TRUNCATION == "toward_zero"


def test_rate_fixed_and_bound():
    value, bound = rate_fixed(1, 3, 10000)
    assert value == 3333
    assert bound == 1  # error <= 1/scale


def test_quantile_exact():
    values = sorted([10, 20, 30, 40, 50])
    p50, b50 = quantile_fixed(values, 1, 2, 10000)
    p95, b95 = quantile_fixed(values, 95, 100, 10000)
    assert p50 == 30 * 10000
    assert p95 == 50 * 10000
    assert b50 == 0 and b95 == 0


def test_undecidable_below_threshold():
    s = summarize([5, 6], [1, 1], scale=10000, min_samples=3)
    assert s["decidable"] == 0
    s2 = summarize([5, 6, 7], [1, 0, 1], scale=10000, min_samples=3)
    assert s2["decidable"] == 1
    assert s2["rate_fixed"] == (2 * 10000) // 3
