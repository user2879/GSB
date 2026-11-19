"""权限口径转换、八进制舍入误差界与传播测试。"""

from __future__ import annotations

from fractions import Fraction

import pytest

from audit_engine.errors import AuditError
from audit_engine.models import PERM_READ, PERM_WRITE
from audit_engine.permission import (
    accumulate_chain,
    is_critical,
    parse_int_mask,
    parse_octal_string,
    parse_permission,
)


def test_int_mask_has_zero_error_bound():
    mask, bound = parse_int_mask(7)
    assert mask == 7
    assert bound == Fraction(0)


def test_int_mask_rejects_unknown_and_negative_bits():
    with pytest.raises(AuditError):
        parse_int_mask(16)
    with pytest.raises(AuditError):
        parse_int_mask(-1)


def test_octal_truncation_bound_equals_dropped_fraction():
    mask, residual, bound = parse_octal_string("12.4", "truncate")
    assert mask == 0o12
    assert residual == Fraction(4, 8)
    assert bound == Fraction(1, 2)


def test_octal_nearest_bound_is_half_unit():
    mask, residual, bound = parse_octal_string("10.5", "nearest")
    assert mask == 0o11
    assert bound == Fraction(1, 2)


def test_octal_multi_digit_fraction():
    mask, residual, bound = parse_octal_string("3.24", "truncate")
    assert mask == 3
    assert residual == Fraction(2, 8) + Fraction(4, 64)
    assert bound == Fraction(20, 64)


def test_octal_invalid_strings_rejected():
    for bad in ("", "8.1", "1.9", "abc", "-"):
        with pytest.raises(AuditError):
            parse_octal_string(bad)


def test_unified_parser_marks_representation():
    parsed_int = parse_permission("int_mask", 3, "truncate")
    parsed_oct = parse_permission("octal_string", "3.4", "truncate")
    assert parsed_int["mask"] == 3 and parsed_int["epsilon"] == 0
    assert parsed_oct["mask"] == 3 and parsed_oct["epsilon"] == Fraction(1, 2)


def test_chain_accumulation_is_linear_and_conservative():
    combined, bound = accumulate_chain(
        [(1, Fraction(1, 2)), (2, Fraction(1, 8)), (4, Fraction(0))]
    )
    assert combined == 7
    assert bound == Fraction(5, 8)


def test_critical_uncertainty_threshold():
    # 翻转一个位所需的最小偏差等于该位位值
    # 点估计写位为 1（mask=2/3），eps >= 2 才可翻转
    assert is_critical(2, Fraction(2), PERM_WRITE) is True
    assert is_critical(2, Fraction(1), PERM_WRITE) is False
    assert is_critical(3, Fraction(1), PERM_WRITE) is False
    # 点估计写位为 0（mask=1），eps >= 2 才可置位
    assert is_critical(1, Fraction(2), PERM_WRITE) is True
    assert is_critical(1, Fraction(1), PERM_WRITE) is False
    assert is_critical(1, Fraction(1, 4), PERM_WRITE) is False
    # 点估计读位为 1，eps 不足 1，读位稳定
    assert is_critical(1, Fraction(1, 2), PERM_READ) is False
    # 点估计读位为 0，eps=1 可翻转
    assert is_critical(0, Fraction(1), PERM_READ) is True


def test_bound_direction_is_non_negative():
    for value, rounding in [("12.4", "truncate"), ("10.5", "nearest"), ("7", "truncate")]:
        _, _, bound = parse_octal_string(value, rounding)
        assert bound >= 0
