"""滚动窗口错误预算账本与双路径燃烧率判定。

精确路径（exact）
    逐分钟维护长度为 window_minutes 的环形分钟缓冲，每分钟 O(1) 滑动累计，
    窗口内成功/失败为精确整数，结论即权威值。

分块增量近似路径（incremental）
    以定长 chunk（分钟块）环形缓冲按批推进：完整块只保留块级合计并被窗口
    整体复用，仅窗口尾部的部分块按分钟展开。因此每步状态规模与“完整块数量 +
    一个部分块”成正比，并复用上一分钟窗口状态（增量 = 新入块 - 离窗块）。
    近似仅发生在尾部边界块，偏差可严格量化（见 bound_*）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from .models import (
    PATH_EXACT,
    PATH_INCREMENTAL,
    PERMYRIAD,
    SELECT_BATCH_COUNT,
    SELECT_CHUNK_CAPACITY,
    SELECT_INCREMENTAL_STATE_RATIO,
    SELECT_WINDOW_MINUTES,
    MinutePoint,
    PathDecision,
    ServiceSpec,
    WindowConclusion,
)


def _alert_reasons(
    spec: ServiceSpec,
    remaining_scaled: int,
    burn_rate: Fraction,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if remaining_scaled < spec.budget_floor:
        reasons.append("remaining_budget_below_floor")
    if burn_rate * PERMYRIAD >= spec.burn_threshold:
        reasons.append("burn_rate_at_or_above_threshold")
    return bool(reasons), tuple(reasons)


def _make_conclusion(
    spec: ServiceSpec,
    minute: int,
    path: str,
    total: int,
    failures: int,
    chunk_capacity: int | None,
) -> WindowConclusion:
    budget_total_scaled = total * (PERMYRIAD - spec.target_permyriad)
    remaining_scaled = budget_total_scaled - failures * PERMYRIAD
    if total == 0:
        burn_num, burn_den = 0, 1
        burn_rate = Fraction(0)
    else:
        burn_num = failures * PERMYRIAD * PERMYRIAD
        burn_den = total * (PERMYRIAD - spec.target_permyriad)
        burn_rate = Fraction(burn_num, burn_den)
    alert, reasons = _alert_reasons(spec, remaining_scaled, burn_rate)
    return WindowConclusion(
        service_id=spec.service_id,
        minute=minute,
        window_minutes=spec.window_minutes,
        param_version=spec.param_version,
        path=path,
        total=total,
        failures=failures,
        budget_total_scaled=budget_total_scaled,
        remaining_budget_scaled=remaining_scaled,
        burn_rate_num=burn_num,
        burn_rate_den=burn_den,
        alert=alert,
        alert_reasons=reasons,
        chunk_capacity=chunk_capacity,
    )


@dataclass
class _MinuteRing:
    capacity: int
    ring: deque[tuple[int, int, int]]
    total: int = 0
    failures: int = 0

    @classmethod
    def create(cls, capacity: int) -> "_MinuteRing":
        return cls(capacity=capacity, ring=deque())

    def push(self, minute: int, successes: int, failures: int) -> None:
        self.ring.append((minute, successes, failures))
        self.total += successes + failures
        self.failures += failures
        while len(self.ring) > self.capacity:
            _, old_s, old_f = self.ring.popleft()
            self.total -= old_s + old_f
            self.failures -= old_f


def run_exact(
    spec: ServiceSpec,
    points: list[MinutePoint],
) -> list[WindowConclusion]:
    """权威精确路径：逐分钟滑动窗口累计。"""
    ring = _MinuteRing.create(spec.window_minutes)
    conclusions: list[WindowConclusion] = []
    for point in points:
        ring.push(point.minute, point.successes, point.failures)
        conclusions.append(
            _make_conclusion(
                spec,
                point.minute,
                PATH_EXACT,
                ring.total,
                ring.failures,
                None,
            )
        )
    return conclusions


@dataclass
class _Chunk:
    """定长分钟块；sealed 后只保留合计，分钟明细被丢弃（近似来源）。"""

    base_minute: int
    capacity: int
    successes: int = 0
    failures: int = 0
    count: int = 0
    detail: dict[int, tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}

    def add(self, minute: int, successes: int, failures: int) -> None:
        self.detail[minute] = (successes, failures)
        self.successes += successes
        self.failures += failures
        self.count += 1

    def seal(self) -> None:
        self.detail = {}

    @property
    def sealed(self) -> bool:
        return not self.detail and self.count > 0

    @property
    def full(self) -> bool:
        return self.count >= self.capacity


@dataclass
class IncrementalState:
    """增量路径的可观测状态规模（用于路径选择的占比指标）。"""

    chunk_capacity: int
    ring_chunks: int
    active_detail_minutes: int

    @property
    def entries(self) -> int:
        return self.ring_chunks + self.active_detail_minutes


def run_incremental(
    spec: ServiceSpec,
    points: list[MinutePoint],
    chunk_capacity: int,
) -> tuple[list[WindowConclusion], list[IncrementalState]]:
    """分块增量近似路径。

    块边界对齐到绝对分钟（minute // chunk_capacity），因此迟到/重放确定性。
    每步顺序：先按新窗口起点回收离窗块（块级增量复用），再纳入当前分钟，
    最后汇总。密封块只保留合计；仅与窗口部分相交的密封边界块按整块计入，
    是唯一近似来源，其偏差由 deviation_bounds 严格量化。
    """
    if chunk_capacity < 1:
        raise ValueError("chunk_capacity 必须为正整数")
    ring_capacity = (spec.window_minutes + chunk_capacity - 1) // chunk_capacity + 2
    chunk_ring: deque[_Chunk] = deque()
    conclusions: list[WindowConclusion] = []
    states: list[IncrementalState] = []

    for point in points:
        window_start = point.minute - spec.window_minutes + 1
        # 1) 离窗块回收：整块落在窗口外则弹出（块级状态复用的关键增量步骤）
        while chunk_ring:
            head = chunk_ring[0]
            head_hi = head.base_minute * chunk_capacity + chunk_capacity - 1
            if head_hi < window_start:
                chunk_ring.popleft()
            else:
                break

        # 2) 当前分钟写入活动块；块满则密封并进入环形缓冲
        chunk_index = point.minute // chunk_capacity
        if not chunk_ring or chunk_ring[-1].base_minute != chunk_index:
            chunk_ring.append(_Chunk(base_minute=chunk_index, capacity=chunk_capacity))
        current = chunk_ring[-1]
        current.add(point.minute, point.successes, point.failures)
        if current.full:
            current.seal()
        while len(chunk_ring) > ring_capacity:
            chunk_ring.popleft()

        # 3) 窗口汇总：完整块复用合计；活动块按分钟明细精确展开
        total = 0
        failures = 0
        detail_minutes = 0
        for chunk in chunk_ring:
            chunk_lo = chunk.base_minute * chunk_capacity
            chunk_hi = chunk_lo + chunk_capacity - 1
            if chunk_hi < window_start:
                continue
            if chunk.sealed:
                if chunk_lo >= window_start:
                    total += chunk.successes + chunk.failures
                    failures += chunk.failures
                else:
                    # 密封边界块与窗口部分相交：明细已丢弃，按整块计入
                    total += chunk.successes + chunk.failures
                    failures += chunk.failures
                    detail_minutes += chunk.count
            else:
                for minute, (successes, fail_count) in chunk.detail.items():
                    if minute >= window_start:
                        total += successes + fail_count
                        failures += fail_count
                        detail_minutes += 1

        conclusions.append(
            _make_conclusion(
                spec,
                point.minute,
                PATH_INCREMENTAL,
                total,
                failures,
                chunk_capacity,
            )
        )
        states.append(
            IncrementalState(
                chunk_capacity=chunk_capacity,
                ring_chunks=len(chunk_ring),
                active_detail_minutes=detail_minutes,
            )
        )
    return conclusions, states


def choose_path(
    window_minutes: int,
    batch_count: int,
    chunk_capacity: int = SELECT_CHUNK_CAPACITY,
) -> PathDecision:
    """显式路径选择规则（可解释、可审计）。

    选择增量近似路径当且仅当同时满足：
      1. 窗口长度 >= SELECT_WINDOW_MINUTES（长窗口才有显著规模收益）；
      2. 批数 >= SELECT_BATCH_COUNT（短流直接走精确路径，零近似成本）；
      3. 增量状态占比 = chunk_capacity/window_minutes
         <= SELECT_INCREMENTAL_STATE_RATIO（状态压缩达到一半以上）。
    其余情形默认精确路径。
    """
    ratio = chunk_capacity / window_minutes
    conditions = [
        window_minutes >= SELECT_WINDOW_MINUTES,
        batch_count >= SELECT_BATCH_COUNT,
        ratio <= SELECT_INCREMENTAL_STATE_RATIO,
    ]
    if all(conditions):
        reason = (
            "长窗口且批数充足，分块环形缓冲状态占比 "
            f"{ratio:.3f} <= {SELECT_INCREMENTAL_STATE_RATIO}，默认采用增量近似路径"
        )
        path = PATH_INCREMENTAL
    else:
        failed = [
            name
            for name, ok in zip(
                (
                    f"window_minutes>={SELECT_WINDOW_MINUTES}",
                    f"batch_count>={SELECT_BATCH_COUNT}",
                    f"state_ratio<={SELECT_INCREMENTAL_STATE_RATIO}",
                ),
                conditions,
            )
            if not ok
        ]
        reason = "精确路径更优，未满足条件: " + ", ".join(failed)
        path = PATH_EXACT
    return PathDecision(
        path=path,
        reason=reason,
        window_minutes=window_minutes,
        batch_count=batch_count,
        incremental_state_ratio=ratio,
        chunk_capacity=chunk_capacity,
    )


def deviation_bounds(
    exact: WindowConclusion,
    chunk_capacity: int,
    points_by_minute: dict[int, MinutePoint],
) -> dict[str, int | str | list[int]]:
    """增量近似相对精确累计的偏差上界及其量化依据。

    近似只可能发生在窗口尾部边界块：密封块按整块计入，而精确路径会剔除
    落在窗口外的块内分钟。因此
        |total_inc - total_exact|       <= 尾部块内窗口外分钟的请求量
        |failures_inc - failures_exact| <= 同上（失败计数上界）
        |remaining_inc - remaining_exact|
            = 10000 * |total_exact - total_inc| （纯整数恒等式）
    """
    window_start = exact.minute - exact.window_minutes + 1
    if window_start <= 0:
        boundary_chunk_index = -1
        bound_total = 0
        bound_failures = 0
        covered_minutes: list[int] = []
    else:
        boundary_chunk_index = (window_start - 1) // chunk_capacity
        bound_total = 0
        bound_failures = 0
        covered_minutes = []
        for minute in range(
            boundary_chunk_index * chunk_capacity,
            boundary_chunk_index * chunk_capacity + chunk_capacity,
        ):
            if minute >= window_start:
                continue
            point = points_by_minute.get(minute)
            if point is not None:
                bound_total += point.successes + point.failures
                bound_failures += point.failures
                covered_minutes.append(minute)
    bound_remaining = PERMYRIAD * bound_total
    return {
        "chunk_capacity": chunk_capacity,
        "boundary_chunk_index": boundary_chunk_index,
        "bound_total_requests": bound_total,
        "bound_failures": bound_failures,
        "bound_remaining_scaled": bound_remaining,
        "boundary_minutes": covered_minutes,
        "basis": (
            "仅尾部边界块可能被整块计入；偏差上界=该块落在窗口外分钟的请求量，"
            "剩余预算偏差=10000*请求量偏差（整数恒等式）"
        ),
    }


def burn_rate_bounds(
    exact: WindowConclusion,
    bound_total: int,
    bound_failures: int,
) -> tuple[Fraction, Fraction]:
    """给定请求量/失败量绝对偏差上界，推导燃烧率分式扰动区间。

    燃烧率 = failures*1e8 / (total * (1-target))，分母中的错误率系数对同一
    服务为常量，因此区间端点由 total/failures 的整数扰动直接给出。
    """
    if exact.total == 0:
        return Fraction(0), Fraction(0)
    error_coefficient = exact.budget_total_scaled // exact.total
    if error_coefficient * exact.total != exact.budget_total_scaled:
        raise ValueError("错误率系数不是整数，预算恒等式不成立")
    low_total = max(exact.total - bound_total, 1)
    high_total = exact.total + bound_total
    low_failures = max(exact.failures - bound_failures, 0)
    high_failures = exact.failures + bound_failures
    low = Fraction(
        low_failures * PERMYRIAD * PERMYRIAD,
        high_total * error_coefficient,
    )
    high = Fraction(
        high_failures * PERMYRIAD * PERMYRIAD,
        low_total * error_coefficient,
    )
    return low, high
