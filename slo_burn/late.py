"""迟到批的窗口归属、预算重算与受影响集合稳定输出。"""

from __future__ import annotations

from .budget import run_exact, run_incremental
from .models import (
    LateImpact,
    MinutePoint,
    PATH_EXACT,
    ServiceSpec,
    WindowConclusion,
)
from .stream import CoverageRecord, merge_streams
from .models import MinuteBatch


def windows_owning_minute(minute: int, window_minutes: int, horizon: int) -> list[int]:
    """分钟 minute 被哪些窗口终点引用：窗口终点 t 满足 t-W+1 <= minute <= t。"""
    start = minute
    end = min(minute + window_minutes - 1, horizon)
    if start > end:
        return []
    return list(range(start, end + 1))


def recompute_with_late(
    spec: ServiceSpec,
    regular: list[MinuteBatch],
    late: list[MinuteBatch],
    path: str,
    chunk_capacity: int | None = None,
) -> tuple[
    list[WindowConclusion],
    list[MinutePoint],
    list[CoverageRecord],
    list[LateImpact],
]:
    """并入迟到批后重算全部受影响窗口。

    窗口引用关系：窗口 W(t)=[t-W+1, t]，迟到分钟 m 影响所有
    t in [m, m+W-1] 的窗口。结论按服务标识序与分钟序稳定输出。
    """
    service_regular = [b for b in regular if b.service_id == spec.service_id]
    service_late = [b for b in late if b.service_id == spec.service_id]
    baseline_points, _ = merge_streams(service_regular, [])
    baseline_conclusions = _run_path(
        spec,
        baseline_points.get(spec.service_id, []),
        path,
        chunk_capacity,
    )
    merged_points, coverage = merge_streams(service_regular, service_late)
    points = merged_points.get(spec.service_id, [])
    new_conclusions = _run_path(spec, points, path, chunk_capacity)

    baseline_by_minute = {item.minute: item for item in baseline_conclusions}
    late_minutes = sorted({batch.minute for batch in service_late})
    horizon = max((point.minute for point in points), default=-1)

    impacted_minutes: set[int] = set()
    for minute in late_minutes:
        impacted_minutes.update(
            windows_owning_minute(minute, spec.window_minutes, horizon)
        )
    # 被覆盖分钟同样触发其所属窗口重算
    for record in coverage:
        impacted_minutes.update(
            windows_owning_minute(record.minute, spec.window_minutes, horizon)
        )

    impacts: list[LateImpact] = []
    for minute in sorted(impacted_minutes):
        new_item = next((item for item in new_conclusions if item.minute == minute), None)
        old_item = baseline_by_minute.get(minute)
        if new_item is None:
            continue
        changed: set[str] = set()
        if old_item is None:
            changed.update(("remaining_budget_scaled", "burn_rate", "alert"))
        else:
            if old_item.remaining_budget_scaled != new_item.remaining_budget_scaled:
                changed.add("remaining_budget_scaled")
            if old_item.burn_rate != new_item.burn_rate:
                changed.add("burn_rate")
            if old_item.alert != new_item.alert:
                changed.add("alert")
        if changed:
            impacts.append(
                LateImpact(
                    service_id=spec.service_id,
                    minute=minute,
                    changed_fields=tuple(sorted(changed)),
                    path=path,
                )
            )
    return new_conclusions, points, coverage, impacts


def _run_path(
    spec: ServiceSpec,
    points: list[MinutePoint],
    path: str,
    chunk_capacity: int | None,
) -> list[WindowConclusion]:
    if path == PATH_EXACT or chunk_capacity is None:
        return run_exact(spec, points)
    conclusions, _ = run_incremental(spec, points, chunk_capacity)
    return conclusions
