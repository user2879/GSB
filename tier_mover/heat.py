"""Explicit, integer-only heat formula and deterministic ordering helpers.

Heat is computed with arbitrary precision Python integers so the result never
depends on floating point, hash seeds or dict iteration order::

    heat = access_count * HEAT_TIME_SCALE + (last_access_ms // 1000)

where ``HEAT_TIME_SCALE = 1 << 40``.  The time component is the access time
in whole seconds and is always smaller than the scale, so the access count is
the dominant term and the timestamp only breaks ties within one count.

Object ordering additionally breaks every remaining tie with the pair
``(key, digest)``, guaranteeing a total, reproducible order.
"""

from __future__ import annotations

from typing import List

from .models import Obj

#: 2**40 > 1_197_000_000_000, the maximum whole-second value of any
#: nonnegative millisecond timestamp currently representable.
HEAT_TIME_SCALE = 1 << 40


def heat_value(obj: Obj) -> int:
    return obj.access_count * HEAT_TIME_SCALE + (obj.last_access_ms // 1000)


def hottest_first(objects: List[Obj]) -> List[Obj]:
    """Return objects sorted from hottest to coldest (total order)."""
    return sorted(
        objects,
        key=lambda o: (-heat_value(o), o.key, o.digest),
    )


def coldest_first(objects: List[Obj]) -> List[Obj]:
    """Return objects sorted from coldest to hottest (total order)."""
    return sorted(
        objects,
        key=lambda o: (heat_value(o), o.key, o.digest),
    )
