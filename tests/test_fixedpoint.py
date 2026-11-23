from __future__ import annotations

from leasebench.fixedpoint import quantize_ratio, rank_ratios


def test_floor_and_single_value_error() -> None:
    value = quantize_ratio(1, 3, 1000)
    assert value.fixed == 333
    assert value.quantization_error_fp == 1
    assert value.single_value_error_fp == 1
    assert value.comparison_error_fp == 2
    assert value.dominant_error_segment == "QUANTIZATION"


def test_comparison_bound_marks_tie_and_stable_sort() -> None:
    one_third = quantize_ratio(1, 3, 1000)
    two_sixths = quantize_ratio(2, 6, 1000)
    one_half = quantize_ratio(1, 2, 1000)
    ranked = rank_ratios(
        [("b", two_sixths), ("c", one_half), ("a", one_third)]
    )
    assert [item.key for item in ranked] == ["c", "a", "b"]
    by_key = {item.key: item for item in ranked}
    assert by_key["a"].affected_by_error
    assert by_key["b"].affected_by_error
    assert not by_key["c"].affected_by_error


def test_borrow_delay_error_can_dominate() -> None:
    value = quantize_ratio(2, 4, 1000, near_borrow_units=3)
    assert value.fixed == 500
    assert value.borrow_return_error_fp == 750
    assert value.total_comparison_error_fp == 752
    assert value.dominant_error_segment == "BORROW_RETURN_DELAY"
