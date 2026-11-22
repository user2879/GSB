"""观测窗口内服务可用状态判定与降级触发。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fixedpoint import numpy_window_rates
from .models import Outcome, ServiceState, Topology


@dataclass
class WindowState:
    service_id: str
    samples: int
    success_count: int
    refused_count: int
    timeout_count: int
    slow_count: int
    success_rate_fixed: int
    state: str
    limited_threshold: int
    unavailable_threshold: int
    outcome_seqs: list[int] = field(default_factory=list)


@dataclass
class DegradeTrigger:
    seq: int
    service_id: str
    domain_id: str
    source: str            # observed_limited / observed_unavailable / forced_propagation
    strategy: str
    capacity_used: int
    capacity_used_fixed: int


class AvailabilityEvaluator:
    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.factor = topology.params.factor
        self.window = topology.params.window_size
        self.states: dict[str, WindowState] = {}
        self.triggers: list[DegradeTrigger] = []
        self._records_by_service: dict[str, list] = {}

    def ingest(self, records: list) -> None:
        for record in records:
            self._records_by_service.setdefault(record.downstream, []).append(record)
        for service_id in self.topology.order_index:
            self._evaluate(service_id)

    def _evaluate(self, service_id: str) -> None:
        service = self.topology.services[service_id]
        window_records = self._records_by_service.get(service_id, [])[-self.window:]
        total = len(window_records)
        if total == 0:
            self.states[service_id] = WindowState(
                service_id, 0, 0, 0, 0, 0, 0, ServiceState.AVAILABLE.value,
                self.topology.params.limited_threshold,
                self.topology.params.unavailable_threshold)
            return
        outcomes = np.array([r.outcome for r in window_records])
        success = int(np.isin(outcomes, [
            Outcome.SUCCESS.value, Outcome.SLOW_SUCCESS.value]).sum())
        refused = int((outcomes == Outcome.REFUSED.value).sum())
        timeout = int((outcomes == Outcome.TIMEOUT.value).sum())
        slow = int((outcomes == Outcome.SLOW_SUCCESS.value).sum())
        rates = numpy_window_rates(np.array([success]), np.array([total]),
                                   self.factor)
        rate = int(rates[0])
        if rate < self.topology.params.unavailable_threshold:
            state = ServiceState.UNAVAILABLE.value
        elif rate < self.topology.params.limited_threshold:
            state = ServiceState.DEGRADED_LIMITED.value
        else:
            state = ServiceState.AVAILABLE.value
        self.states[service_id] = WindowState(
            service_id, total, success, refused, timeout, slow, rate, state,
            self.topology.params.limited_threshold,
            self.topology.params.unavailable_threshold,
            [r.seq for r in window_records])

        # 同一次调用不得既计入成功又计入降级：仅对窗口内失败结局触发。
        if state != ServiceState.AVAILABLE.value:
            source = ("observed_unavailable"
                      if state == ServiceState.UNAVAILABLE.value
                      else "observed_limited")
            failed_seqs = [r.seq for r in window_records if not r.is_success]
            for seq in failed_seqs:
                self.triggers.append(DegradeTrigger(
                    seq=seq, service_id=service_id, domain_id=service.domain_id,
                    source=source, strategy=service.strategy,
                    capacity_used=1,
                    capacity_used_fixed=self.factor))

    def add_forced_trigger(self, seq: int, service_id: str) -> None:
        service = self.topology.services[service_id]
        self.triggers.append(DegradeTrigger(
            seq=seq, service_id=service_id, domain_id=service.domain_id,
            source="forced_propagation", strategy=service.strategy,
            capacity_used=1, capacity_used_fixed=self.factor))
