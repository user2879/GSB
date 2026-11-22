"""实例健康状态机与故障域级隔离状态机。

实例状态: unknown / healthy / degraded / unhealthy
域状态:   normal / isolating / isolated / recovering

域隔离阈值由隔离等级决定(确定性推导):
  0 从不隔离; 1 任意不健康权重即触发; 2 不健康权重过半; 3 全部权重不健康。
冷却窗口(域声明的 recovery_cooldown_ms, 以事件流时间戳度量)未结束,
域不得从 isolated 转入 recovering。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import DiagError
from .spec import FAULT_TYPES, SUPPORTED_ISOLATION_LEVELS, DomainSpec, Event, InstanceSpec
from .util import monotonic_ms

HEALTH_STATES = ("unknown", "healthy", "degraded", "unhealthy")
DOMAIN_STATES = ("normal", "isolating", "isolated", "recovering")


def apply_fault(health: str, fault: str) -> str:
    """确定性的实例健康迁移规则。"""
    if fault == "crash":
        return "unhealthy"
    if fault == "degrade":
        return "unhealthy" if health == "unhealthy" else "degraded"
    if fault == "flap":
        return "degraded" if health in ("unknown", "healthy") else health
    if fault == "recover":
        return "healthy"
    raise DiagError("E_EVENT", f"unsupported fault type {fault!r}",
                    {"fault": fault})


@dataclass(frozen=True)
class CommandIntent:
    instance: str
    domain: str
    action: str  # "isolate" | "recover"
    event_seq: int


class Engine:
    """按事件流推进的确定性状态引擎。"""

    def __init__(self, instances: dict[str, InstanceSpec], domains: dict[str, DomainSpec],
                 mono_clock=monotonic_ms):
        self.instances = instances
        self.domains = domains
        self.health = {iid: "unknown" for iid in instances}
        self.domain_state = {did: "normal" for did in domains}
        self.isolated_since_ts: dict[str, int] = {}
        self.was_isolated: dict[str, set[str]] = {did: set() for did in domains}
        self.transitions: list[dict] = []
        self._mono = mono_clock
        self.current_seq = 0
        self.current_ts = 0

    # ---- 查询 ----
    def domain_instances(self, domain: str) -> list[str]:
        return sorted(iid for iid, spec in self.instances.items() if spec.domain == domain)

    def unhealthy_weight(self, domain: str) -> int:
        return sum(self.instances[iid].weight for iid in self.domain_instances(domain)
                   if self.health[iid] == "unhealthy")

    def total_weight(self, domain: str) -> int:
        return sum(self.instances[iid].weight for iid in self.domain_instances(domain))

    def threshold_met(self, domain: str) -> bool:
        level = self.domains[domain].isolation_level
        if level not in SUPPORTED_ISOLATION_LEVELS:
            raise DiagError("E_LEVEL",
                            f"domain {domain!r} isolation level {level} out of range",
                            {"domain": domain, "isolation_level": level,
                             "supported": list(SUPPORTED_ISOLATION_LEVELS)})
        uw, tw = self.unhealthy_weight(domain), self.total_weight(domain)
        if level == 0:
            return False
        if level == 1:
            return uw >= 1
        if level == 2:
            return uw * 2 > tw
        return tw > 0 and uw >= tw

    def expected_isolated(self, iid: str) -> bool:
        domain = self.instances[iid].domain
        return self.domain_state[domain] in ("isolating", "isolated") \
            and self.health[iid] == "unhealthy"

    def expected_view(self, domain: str) -> dict:
        """域级期望隔离结论(规范化后可逐字节比对)。"""
        return {"domain": domain,
                "state": self.domain_state[domain],
                "instances": {iid: {"isolated": self.expected_isolated(iid)}
                              for iid in self.domain_instances(domain)}}

    def cooldown_remaining_ms(self, domain: str) -> int:
        if domain not in self.isolated_since_ts:
            return 0
        elapsed = self.current_ts - self.isolated_since_ts[domain]
        return max(0, self.domains[domain].recovery_cooldown_ms - elapsed)

    # ---- 迁移 ----
    def _record(self, kind: str, entity: str, old: str, new: str, reason: str) -> dict:
        transition = {"kind": kind, "entity": entity, "from": old, "to": new,
                      "reason": reason, "event_seq": self.current_seq,
                      "mono_ms": self._mono()}
        self.transitions.append(transition)
        return transition

    def apply_event(self, event: Event) -> list[CommandIntent]:
        if event.fault not in FAULT_TYPES:
            raise DiagError("E_EVENT",
                            f"event {event.seq}: unsupported fault type {event.fault!r}",
                            {"event_seq": event.seq, "instance": event.instance,
                             "fault": event.fault})
        self.current_seq = event.seq
        self.current_ts = event.ts_ms
        spec = self.instances[event.instance]
        domain = spec.domain
        old = self.health[event.instance]
        new = apply_fault(old, event.fault)
        if new != old:
            self.health[event.instance] = new
            self._record("instance", event.instance, old, new, f"fault:{event.fault}")
        return self._eval_domain(domain)

    def _eval_domain(self, domain: str) -> list[CommandIntent]:
        intents: list[CommandIntent] = []
        state = self.domain_state[domain]
        if state == "normal" and self.threshold_met(domain):
            self.domain_state[domain] = "isolating"
            self._record("domain", domain, "normal", "isolating",
                         "unhealthy weight threshold met")
            for iid in self.domain_instances(domain):
                if self.health[iid] == "unhealthy":
                    intents.append(CommandIntent(iid, domain, "isolate", self.current_seq))
                    self.was_isolated[domain].add(iid)
        elif state == "isolated":
            no_unhealthy = all(self.health[iid] != "unhealthy"
                               for iid in self.domain_instances(domain))
            cooled = (self.current_ts - self.isolated_since_ts[domain]
                      >= self.domains[domain].recovery_cooldown_ms)
            if no_unhealthy and cooled:
                self.domain_state[domain] = "recovering"
                self._record("domain", domain, "isolated", "recovering",
                             "cooldown elapsed and no unhealthy instance")
                for iid in sorted(self.was_isolated[domain]):
                    intents.append(CommandIntent(iid, domain, "recover", self.current_seq))
        return intents

    # ---- 收敛确认(由运行器在回执逐字节一致后调用) ----
    def confirm_isolated(self, domain: str) -> None:
        if self.domain_state[domain] == "isolating":
            self.domain_state[domain] = "isolated"
            self.isolated_since_ts[domain] = self.current_ts
            self._record("domain", domain, "isolating", "isolated",
                         "isolate receipts converged")

    def confirm_normal(self, domain: str) -> None:
        if self.domain_state[domain] == "recovering":
            self.domain_state[domain] = "normal"
            self.was_isolated[domain].clear()
            self._record("domain", domain, "recovering", "normal",
                         "recover receipts converged")

    # ---- 快照/账本恢复 ----
    def restore(self, health: dict, domain_state: dict, isolated_since_ts: dict,
                was_isolated: dict, current_seq: int, current_ts: int) -> None:
        self.health.update(health)
        self.domain_state.update(domain_state)
        self.isolated_since_ts.update(isolated_since_ts)
        for did, members in was_isolated.items():
            self.was_isolated[did] = set(members)
        self.current_seq = current_seq
        self.current_ts = current_ts
