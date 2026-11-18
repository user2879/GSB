"""Exact integer fixed-point arithmetic.

All numeric intermediates in the workbench are exact Python integers in a
declared fixed-point scale.  Input floats are parsed through
:class:`decimal.Decimal` so host binary-float rounding never enters the
pipeline.
Every truncation step names its direction explicitly:

* ``toward_zero`` - integer truncation, the rounding mode used by spec;
* ``floor``        - towards negative infinity, used by documented divisions.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Optional, Union

from .errors import WorkbenchError

ROUNDING_MODES = ("toward_zero", "floor")
JsonNumber = Union[int, float, str, None]


def to_fixed(value: JsonNumber, scale: int,
             rounding: str = "toward_zero") -> Optional[int]:
    """Convert a JSON number (or None) using an absolute integer scale."""
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is an int subclass but never a valid numeric cell here.
        raise WorkbenchError("E_DATA", "boolean cells are not supported",
                             field_path="$.rows")
    if isinstance(value, int):
        return value * scale
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise WorkbenchError(
                "E_DATA", f"non-numeric cell value {value!r}",
                field_path="$.rows")
    if not decimal_value.is_finite():
        raise WorkbenchError("E_DATA", "non-finite numeric cell",
                             field_path="$.rows")
    scaled = decimal_value * Decimal(scale)
    return _round_decimal(scaled, rounding)


def scale_int(value: int, decimal_scale: int, divisor: int,
              rounding: str = "toward_zero") -> int:
    """Compute ``value * decimal_scale / divisor`` with named truncation.

    ``decimal_scale`` is an explicit integer session multiplier; division by
    ``divisor`` truncates in the direction named by ``rounding``.
    """
    if divisor <= 0:
        raise WorkbenchError("E_PARAM", "divisor must be a positive integer",
                             field_path="$.params.divisor")
    numerator = value * decimal_scale
    if rounding == "toward_zero":
        # abs-based truncation keeps the operation sign-explicit.
        sign = -1 if numerator < 0 else 1
        return sign * (abs(numerator) // divisor)
    if rounding == "floor":
        return numerator // divisor
    raise WorkbenchError("E_PARAM", f"unknown rounding mode {rounding!r}",
                         field_path="$.params.rounding")

def _round_decimal(value: Decimal, rounding: str) -> int:
    if rounding == "toward_zero":
        return int(value)  # Decimal.__int__ truncates towards zero
    if rounding == "floor":
        return int(value.to_integral_value(rounding="ROUND_FLOOR"))
    raise WorkbenchError("E_PARAM", f"unknown rounding mode {rounding!r}",
                         field_path="$.params.rounding")
