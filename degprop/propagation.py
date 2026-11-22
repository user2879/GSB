"""降级沿依赖拓扑传播并在故障域边界收敛为唯一可复核方案。

传播语义（反向边，即下游 -> 上游）：
  * forced 边：无条件把上游纳入降级集合（强制传播）；
  * threshold 边：仅当下游当前可用比例（定点整数）低于边阈值时才传播，
    健康上游不会被并入故障域降级集合。

故障域收敛时同时构造两套候选方案：
  * local_priority：域内容量上限直接加总（整数）；
  * cross_domain_apportion：按各域并发配额占比做定点整数分摊，floor 截断，
    余数按稳定序再分配。
两方案以合成判定误差界 J=B_a+B_b 比较，落入界内标注 indistinguishable，
并按 (容量降序, 域标识序, 降级集合字典序) 稳定决胜，与枚举顺序无关。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .fixedpoint import (
    Interval,
    judgment_bound,
    quantized_interval,
)
from .models import PropagationMode, Topology


@dataclass
class PropagationHop:
    seq_context: str
    degraded_service: str
    upstream_service: str
    edge_mode: str
    threshold_fixed: int | None
    availability_fixed: int | None
    propagated: bool
    reason: str


@dataclass
class DomainPlan:
    domain_id: str
    degraded_services: list[str]
    total_capacity: int
    total_capacity_fixed: int
    availability_rate_fixed: int
    source_plan: str
    indistinguishable: bool
    judgment: dict


@dataclass
class PropagationResult:
    degraded_services: list[str]
    hops: list[PropagationHop]
    domain_plans: dict[str, DomainPlan]
    intermediates: list[dict] = field(default_factory=list)


class PropagationPlanner:
    def __init__(self, topology: Topology, window_states: dict,
                 triggers: list) -> None:
        self.topology = topology
        self.states = window_states
        self.triggers = triggers
        self.factor = topology.params.factor
        self.hops: list[PropagationHop] = []
        self.intermediates: list[dict] = []

    def run(self) -> PropagationResult:
        seeds = sorted({t.service_id for t in self.triggers})
        degraded = set(seeds)
        queue = list(seeds)
        while queue:
            current = queue.pop(0)
            for edge in self._reverse_edges(current):
                upstream = edge.upstream
                state = self.states.get(current)
                rate = state.success_rate_fixed if state else 0
                propagated, reason = self._edge_decision(edge, rate)
                self.hops.append(PropagationHop(
                    seq_context=f"from:{current}",
                    degraded_service=current, upstream_service=upstream,
                    edge_mode=edge.mode, threshold_fixed=edge.threshold,
                    availability_fixed=rate, propagated=propagated, reason=reason))
                if propagated and upstream not in degraded:
                    degraded.add(upstream)
                    queue.append(upstream)
            queue.sort()

        ordered = [sid for sid in self.topology.order_index if sid in degraded]
        domain_plans = self._converge(ordered)
        return PropagationResult(ordered, self.hops, domain_plans,
                                 self.intermediates)

    def _reverse_edges(self, downstream: str):
        return sorted((e for e in self.topology.edges
                       if e.downstream == downstream),
                      key=lambda e: e.upstream)

    def _edge_decision(self, edge, availability_fixed: int) -> tuple[bool, str]:
        if edge.mode == PropagationMode.FORCED.value:
            return True, "forced_edge_unconditional"
        if availability_fixed < (edge.threshold or 0):
            return True, (f"availability {availability_fixed} < threshold "
                          f"{edge.threshold}")
        return False, (f"availability {availability_fixed} >= threshold "
                       f"{edge.threshold}")

    def _converge(self, degraded_services: list[str]) -> dict[str, DomainPlan]:
        plans: dict[str, DomainPlan] = {}
        for domain_id in sorted(self.topology.domains):
            members = sorted(sid for sid in degraded_services
                             if self.topology.services[sid].domain_id == domain_id)
            if not members:
                continue
            local = self._local_scheme(domain_id, members)
            cross = self._cross_domain_scheme(domain_id, members)
            chosen = self._compare(domain_id, local, cross)
            plans[domain_id] = chosen
            self.intermediates.append(chosen.judgment)
        return plans

    def _local_scheme(self, domain_id: str, members: list[str]) -> DomainPlan:
        capacity = sum(self.topology.services[sid].capacity_limit for sid in members)
        return DomainPlan(domain_id, members, capacity,
                          capacity * self.factor,
                          self._domain_availability(domain_id, members),
                          "local_priority", False, {})

    def _cross_domain_scheme(self, domain_id: str, members: list[str]) -> DomainPlan:
        total_quota = sum(d.in_flight_quota
                          for d in self.topology.domains.values())
        own_quota = self.topology.domains[domain_id].in_flight_quota
        raw_capacity = sum(self.topology.services[sid].capacity_limit
                           for sid in members)
        # 定点整数分摊：floor 截断，余数按域标识稳定序分配，全程整数运算。
        shares: dict[str, int] = {}
        remainders: list[tuple[int, str]] = []
        for did in sorted(self.topology.domains):
            quota = self.topology.domains[did].in_flight_quota
            numerator = raw_capacity * quota * self.factor
            share, remainder = divmod(numerator, total_quota)
            shares[did] = share
            remainders.append((remainder, did))
        allocated_fixed = shares[domain_id]
        leftover = raw_capacity * self.factor - sum(shares.values())
        unit = self.factor
        cursor = 0
        while leftover >= unit and cursor < len(remainders):
            _, did = sorted(remainders, key=lambda x: (-x[0], x[1]))[cursor]
            if did == domain_id:
                allocated_fixed += unit
            leftover -= unit
            cursor += 1
        allocated = allocated_fixed // self.factor
        return DomainPlan(domain_id, members, allocated, allocated_fixed,
                          self._domain_availability(domain_id, members),
                          "cross_domain_apportion", False, {})

    def _domain_availability(self, domain_id: str, members: list[str]) -> int:
        observed = [sid for sid in members
                    if sid in self.states and self.states[sid].samples > 0]
        if not observed:
            return 0
        return sum(self.states[sid].success_rate_fixed
                   for sid in observed) // len(observed)

    def _compare(self, domain_id: str, local: DomainPlan,
                 cross: DomainPlan) -> DomainPlan:
        local_iv = quantized_interval(
            Fraction(local.total_capacity_fixed, self.factor),
            self._scaling(), f"{domain_id}.local.capacity")
        cross_iv = quantized_interval(
            Fraction(cross.total_capacity_fixed, self.factor),
            self._scaling(), f"{domain_id}.cross.capacity")
        capacity_judgment = judgment_bound(
            local_iv, cross_iv, self.factor, f"{domain_id}.capacity")

        rate_local = Interval(local.availability_rate_fixed, 1)
        rate_cross = Interval(cross.availability_rate_fixed, 1)
        rate_judgment = judgment_bound(
            rate_local, rate_cross, self.factor, f"{domain_id}.availability")

        indistinguishable = capacity_judgment["indistinguishable"]
        candidates = sorted(
            [local, cross],
            key=lambda plan: (-plan.total_capacity_fixed,
                              plan.domain_id,
                              ",".join(plan.degraded_services)),
        )
        winner = candidates[0]
        return DomainPlan(
            domain_id=domain_id,
            degraded_services=list(winner.degraded_services),
            total_capacity=winner.total_capacity,
            total_capacity_fixed=winner.total_capacity_fixed,
            availability_rate_fixed=winner.availability_rate_fixed,
            source_plan=winner.source_plan,
            indistinguishable=indistinguishable,
            judgment={
                "capacity": capacity_judgment,
                "availability": rate_judgment,
                "tie_break_keys": [
                    {"domain_id": p.domain_id,
                     "degraded_services": list(p.degraded_services),
                     "capacity_fixed": p.total_capacity_fixed}
                    for p in candidates
                ],
            },
        )

    def _scaling(self):
        from .fixedpoint import Scaling

        return Scaling(self.topology.params.scale_exponent)
