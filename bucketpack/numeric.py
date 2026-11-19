"""Deterministic numeric primitives with an optional NumPy fast path."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

try:  # NumPy is optional in the execution sandbox, required by the document.
    import numpy as _np
except Exception:  # pragma: no cover - depends on the local interpreter
    _np = None


NP_VERSION = getattr(_np, "__version__", None)
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class Quantized:
    value: int
    error: Fraction
    saturated: bool


def quantize(value: int | float, scale: int) -> Quantized:
    """Quantize with truncation toward zero and int64 saturation.

    Floating point input is converted through ``Decimal(str(value))`` before
    scaling.  That makes the documented decimal interpretation explicit and
    avoids accidentally depending on binary floating point formatting.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("quantize only accepts int or float")
    if scale < 0 or scale > 9:
        raise ValueError("scale must be in 0..9")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        source = Decimal(str(value))
    else:
        source = Decimal(value)
    factor = 10**scale
    with localcontext() as ctx:
        ctx.prec = max(50, len(source.as_tuple().digits) + scale + 8)
        scaled = source * factor
        truncated = int(scaled)  # int(Decimal) truncates toward zero
    saturated = False
    if truncated < INT64_MIN:
        truncated = INT64_MIN
        saturated = True
    elif truncated > INT64_MAX:
        truncated = INT64_MAX
        saturated = True
    reconstructed = Decimal(truncated) / factor
    error = abs(source - reconstructed)
    return Quantized(value=truncated, error=Fraction(error), saturated=saturated)


def error_bound(scale: int, observed: Fraction, saturated: bool) -> Fraction:
    """Return a defensible upper bound for one column's quantization error.

    Without saturation the theoretical bound is ``10^-scale``.  Saturation is
    data dependent, so the observed maximum is the sharp, still verifiable
    bound for that expanded column.
    """

    theoretical = Fraction(1, 10**scale)
    return max(theoretical, observed) if saturated else theoretical


def bincount(values: Sequence[int], bucket_count: int) -> list[int]:
    """Count ordered bucket ids using NumPy when available.

    NumPy's ``bincount`` is the actual numerical routine on supported local
    installations.  The pure-Python path keeps the component runnable in the
    minimal CI sandbox without silently changing semantics.
    """

    if bucket_count < 1:
        return []
    if _np is not None:
        if values:
            counts = _np.bincount(_np.asarray(values, dtype=_np.int64), minlength=bucket_count)
            return [int(v) for v in counts[:bucket_count]]
        return [0] * bucket_count
    counts = [0] * bucket_count
    for value in values:
        counts[value] += 1
    return counts


def sum_counts(count_groups: Iterable[Sequence[int]], bucket_count: int) -> list[int]:
    total = [0] * bucket_count
    if _np is not None:
        for group in count_groups:
            if len(group) != bucket_count:
                raise ValueError("incompatible bucket count vectors")
            if group:
                total = _np.add(total, _np.asarray(group, dtype=_np.int64)).tolist()
    else:
        for group in count_groups:
            if len(group) != bucket_count:
                raise ValueError("incompatible bucket count vectors")
            for idx, value in enumerate(group):
                total[idx] += int(value)
    return [int(v) for v in total]
