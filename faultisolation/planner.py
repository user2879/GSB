"""仲裁路径选择与显式工作量估算。

三条路径共享同一状态机，仅事件组织方式不同，结论必然逐字节一致；
本模块只负责依据域数 D、事件数 N、槽位总量 S 选择默认路径并给出可复核理由。

显式工作量规则（整数运算，单位为“仲裁工作量点”）：

* 逐事件 ``event``:  W_event  = N + N/4
* 分批   ``batch``:  W_batch  = N + ceil(N/B) * D + N/4
* 归并   ``merge``:  W_merge  = N + N*D/ceil(N/D) + N/3

决胜规则：取最小工作量；当任意两路径工作量之差的绝对值
``|Wa - Wb| <= T``（显式阈值 ``T = ceil(N/10)``）时视为接近，
按固定优先级 ``event < batch < merge`` 稳定决胜，并在理由中标注“阈值内稳定决胜”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .spec import SpecBundle

PATH_EVENT = "event"
PATH_BATCH = "batch"
PATH_MERGE = "merge"
PATHS = (PATH_EVENT, PATH_BATCH, PATH_MERGE)
TIE_BREAK_ORDER = {PATH_EVENT: 0, PATH_BATCH: 1, PATH_MERGE: 2}
PATH_DESCRIPTIONS = {
    PATH_EVENT: "逐事件仲裁：按规范序逐条推进状态机",
    PATH_BATCH: "分批仲裁：按到达时刻分桶，逐桶仲裁",
    PATH_MERGE: "归并仲裁：先按域聚合实例需求，批量仲裁后按规范序合并",
}


@dataclass(frozen=True)
class ScaleParameters:
    domain_count: int
    event_count: int
    total_slots: int
    batch_size: int
    global_lease_cap: int

    def as_dict(self) -> dict[str, int]:
        return {
            "domain_count": self.domain_count,
            "event_count": self.event_count,
            "total_slots": self.total_slots,
            "batch_size": self.batch_size,
            "global_lease_cap": self.global_lease_cap,
        }


def scale_parameters(bundle: SpecBundle) -> ScaleParameters:
    return ScaleParameters(
        domain_count=len(bundle.domains),
        event_count=len(bundle.events),
        total_slots=sum(d.slots for d in bundle.domains),
        batch_size=bundle.session.batch_size,
        global_lease_cap=bundle.session.global_lease_cap,
    )


def estimate_work(bundle: SpecBundle) -> dict[str, int]:
    params = scale_parameters(bundle)
    n = params.event_count
    d = max(1, params.domain_count)
    b = max(1, params.batch_size)
    w_event = n + n // 4
    w_batch = n + math.ceil(n / b) * d + n // 4
    group = max(1, math.ceil(n / d))
    w_merge = n + (n * d) // group + n // 3
    return {PATH_EVENT: w_event, PATH_BATCH: w_batch, PATH_MERGE: w_merge}


def tie_threshold(bundle: SpecBundle) -> int:
    return max(1, math.ceil(len(bundle.events) / 10))


def choose_path(bundle: SpecBundle) -> dict[str, object]:
    work = estimate_work(bundle)
    threshold = tie_threshold(bundle)
    params = scale_parameters(bundle)

    ranked = sorted(work.items(), key=lambda kv: (kv[1], TIE_BREAK_ORDER[kv[0]]))
    winner, winner_work = ranked[0]
    near = [path for path in PATHS if path != winner and abs(work[path] - winner_work) <= threshold]
    if near:
        reason = (
            f"最低工作量路径 {winner}（{winner_work} 点）与 {','.join(near)} 的工作量差落在显式阈值 "
            f"{threshold} 点以内，按稳定决胜优先级 event < batch < merge 选定 {winner}"
        )
        tie_applied = True
    else:
        reason = f"{winner} 工作量 {winner_work} 点严格最低，且与次优差距超过阈值 {threshold} 点，直接选定"
        tie_applied = False
    return {
        "selected": winner,
        "reason": reason,
        "tie_applied": tie_applied,
        "threshold": threshold,
        "work_estimate": work,
        "scale_parameters": params.as_dict(),
        "descriptions": PATH_DESCRIPTIONS,
    }
