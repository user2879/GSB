from __future__ import annotations

import pytest

from faultisolation.errors import SpecError
from faultisolation.fixedpoint import (
    comparison_error_bound, diagnose, rank_ratios, scaled_ratio, single_error_bound,
)


def test_integer_scaled_ratio_truncation():
    # 1/3 在 K=100 下向零截断为 33
    assert scaled_ratio(1, 3, 2) == 33
    assert diagnose("x", 1, 3, 2).quantized == 33
    assert diagnose("x", 1, 3, 2).truncation == "toward_zero"


def test_error_bounds_are_explicit():
    assert single_error_bound(2) == 1
    assert comparison_error_bound(2) == 2
    assert diagnose("x", 1, 2, 2).single_error_human == "1/100"
    assert diagnose("x", 1, 2, 2).comparison_error_human == "2/100"


def test_power_validation():
    with pytest.raises(SpecError):
        scaled_ratio(1, 2, 4)


def test_ranking_flags_error_influence():
    # 两个量化值相同（差 0，落在 2 单位合成界内），稳定决胜并标注受误差影响
    ranked = rank_ratios([("a", 1, 4), ("b", 1, 4)], 2)
    assert [r.label for r in ranked] == ["a", "b"]
    assert ranked[1].error_influenced is True


def test_ranking_strict_separation_not_flagged():
    ranked = rank_ratios([("a", 4, 4), ("b", 1, 4)], 2)
    assert ranked[0].label == "a"
    assert ranked[0].error_influenced is False
