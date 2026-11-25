"""双路径逐字段一致性校验与偏差界。"""

from __future__ import annotations

from fractions import Fraction

from .budget import burn_rate_bounds, deviation_bounds
from .errors import DivergenceError
from .models import (
    DivergenceRecord,
    MinutePoint,
    WindowConclusion,
)

_COMPARE_FIELDS = ("remaining_budget_scaled", "burn_rate_num", "burn_rate_den", "alert")


def _index(conclusions: list[WindowConclusion]) -> dict[int, WindowConclusion]:
    return {item.minute: item for item in conclusions}


def verify_paths(
    service_id: str,
    exact: list[WindowConclusion],
    incremental: list[WindowConclusion],
    chunk_capacity: int,
    points: list[MinutePoint],
) -> list[DivergenceRecord]:
    """逐服务逐分钟比较剩余预算、燃烧率与告警状态。

    数值字段在理论偏差界内视为一致；alert 差异仅在燃烧率贴近阈值且由近似
    边界扰动导致时容忍，否则以 E_DIVERGE 定位首个不一致。
    返回全部偏差记录（含未越界的可解释偏差），越界即抛错。
    """
    points_by_minute = {point.minute: point for point in points}
    exact_by_minute = _index(exact)
    inc_by_minute = _index(incremental)
    records: list[DivergenceRecord] = []
    first_exceeded: DivergenceRecord | None = None

    for minute in sorted(exact_by_minute):
        exact_item = exact_by_minute[minute]
        inc_item = inc_by_minute.get(minute)
        if inc_item is None:
            raise DivergenceError(
                "增量路径缺失该分钟结论",
                service_id=service_id,
                minute=minute,
                field="minute",
            )
        bounds = deviation_bounds(exact_item, chunk_capacity, points_by_minute)

        delta_remaining = inc_item.remaining_budget_scaled - exact_item.remaining_budget_scaled
        bound_remaining = bounds["bound_remaining_scaled"]
        if abs(delta_remaining) > bound_remaining:
            record = DivergenceRecord(
                service_id, minute, "remaining_budget_scaled",
                exact_item.remaining_budget_scaled,
                inc_item.remaining_budget_scaled,
                bound_remaining, True, bounds["basis"],
            )
            records.append(record)
            first_exceeded = first_exceeded or record

        exact_burn = exact_item.burn_rate
        inc_burn = inc_item.burn_rate
        min_burn, max_burn = burn_rate_bounds(
            exact_item,
            bounds["bound_total_requests"],
            bounds["bound_failures"],
        )
        burn_out = not (min_burn <= inc_burn <= max_burn)
        if burn_out:
            record = DivergenceRecord(
                service_id, minute, "burn_rate",
                f"{exact_burn.numerator}/{exact_burn.denominator}",
                f"{inc_burn.numerator}/{inc_burn.denominator}",
                f"[{min_burn.numerator}/{min_burn.denominator},"
                f"{max_burn.numerator}/{max_burn.denominator}]",
                True,
                bounds["basis"],
            )
            records.append(record)
            first_exceeded = first_exceeded or record

        if inc_item.alert != exact_item.alert:
            threshold = Fraction(0)
            # 告警翻转容忍：增量燃烧率落在包含精确值的偏差区间内即视为可解释
            tolerated = not burn_out
            record = DivergenceRecord(
                service_id, minute, "alert",
                exact_item.alert,
                inc_item.alert,
                "tolerated-within-burn-bound" if tolerated else "outside-bound",
                not tolerated,
                bounds["basis"],
            )
            records.append(record)
            if not tolerated:
                first_exceeded = first_exceeded or record

    if first_exceeded is not None:
        raise DivergenceError(
            "双路径结论超出偏差界",
            service_id=first_exceeded.service_id,
            minute=first_exceeded.minute,
            field=first_exceeded.field,
            exact=first_exceeded.exact_value,
            incremental=first_exceeded.incremental_value,
            bound=first_exceeded.bound,
            basis=first_exceeded.basis,
        )
    return records
