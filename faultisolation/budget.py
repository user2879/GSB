"""配额预算诊断的数据口径（纯整数计数）。

引擎逐事件推进时仲裁器维护权威的活跃租约集合，本模块在每个事件后采样
各域真实占用槽位数，从而得到：

* ``current_used``：流末仍活跃租约的槽位之和；
* ``peak_used``：会话内各域采样到的最大占用；
* 全局峰值为同一事件时刻跨域占用之和的最大值（不跨域求和峰值）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine, canonical_event_order


@dataclass(frozen=True)
class UsageSnapshot:
    domain_id: str
    slots: int
    current_used: int
    peak_used: int


@dataclass(frozen=True)
class GlobalUsage:
    total_slots: int
    current_used: int
    peak_used: int


def compute_usage(engine: Engine) -> tuple[list[UsageSnapshot], GlobalUsage]:
    peak = {did: 0 for did in engine.arbiters}
    global_peak = 0
    ordered = canonical_event_order(engine.bundle.events, engine.priority_of)
    for event in ordered:
        engine.process_event(event)
        total_now = 0
        for did, arb in engine.arbiters.items():
            used = sum(len(lease.slots) for lease in arb.active.values())
            peak[did] = max(peak[did], used)
            total_now += used
        global_peak = max(global_peak, total_now)
    snapshots = []
    for did in sorted(engine.arbiters):
        arb = engine.arbiters[did]
        current = sum(len(lease.slots) for lease in arb.active.values())
        snapshots.append(UsageSnapshot(did, arb.spec.slots, current, peak[did]))
    total_slots = sum(dom.slots for dom in engine.bundle.domains)
    global_current = sum(snap.current_used for snap in snapshots)
    return snapshots, GlobalUsage(total_slots, global_current, global_peak)
