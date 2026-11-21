"""Deterministic dual-budget arbitration scheduler (rule version 1).

Judgment order per evaluation (rule v1):
  1. tenant remaining quota   -> reject R_QUOTA_EXHAUSTED
  2. tenant in-flight count   -> queue  R_TENANT_SLOTS_FULL
  3. tenant burst allowance   -> grant flag burst_used (not a blocker)
  4. global in-flight count   -> queue  R_GLOBAL_SLOTS_FULL
  5. target instance health   -> queue R_INSTANCE_NOT_HEALTHY /
                                 reject R_INSTANCE_EJECTED
  otherwise                   -> grant  R_GRANT

The scheduler is a discrete-event simulation over a virtual millisecond
clock: request arrivals, request completions and probe-driven health
transitions are processed in (time, priority, sequence) order with
priority completion < transition < arrival. Given the same requests,
the same probe transition list and the same rule version, the outcome
and basis sequence of every request is byte-identical across runs.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .probe import HealthTimeline, Transition
from .spec import Request, Spec

PRIO_COMPLETION = 0
PRIO_TRANSITION = 1
PRIO_ARRIVAL = 2

OUTCOME_GRANTED = "granted"
OUTCOME_REJECTED = "rejected"


@dataclass(frozen=True)
class BasisEntry:
    at_ms: int
    step: str
    code: str
    detail: str

    def encode(self) -> str:
        return f"{self.at_ms}|{self.step}|{self.code}|{self.detail}"


@dataclass
class RequestOutcome:
    request_id: str
    tenant_id: str
    demand: int
    arrival_ms: int
    outcome: str = ""
    basis: list = field(default_factory=list)
    wait_start_ms: int = -1
    wait_end_ms: int = -1
    grant_ms: int = -1
    done_ms: int = -1

    def basis_blob(self) -> str:
        return ";".join(entry.encode() for entry in self.basis)


@dataclass
class RunResult:
    outcomes: dict
    finalize_order: list
    tenant_consumed: dict
    transitions: list


@dataclass
class _TenantState:
    consumed: int = 0
    inflight: int = 0


class _Decision:
    def __init__(self, verdict: str, entry: BasisEntry) -> None:
        self.verdict = verdict
        self.entry = entry


def service_duration_ms(spec: Spec, tenant_id: str, demand: int) -> int:
    inst = spec.instance_of(tenant_id)
    return max(1, (demand * spec.duration_factor_ms) // inst.weight)


def _evaluate(spec: Spec, req: Request, now_ms: int, tstate: _TenantState,
              global_inflight: int, health: HealthTimeline) -> _Decision:
    tenant = spec.tenants[req.tenant_id]
    remaining = tenant.share - tstate.consumed
    if req.demand > remaining:
        return _Decision("reject", BasisEntry(
            now_ms, "quota", "R_QUOTA_EXHAUSTED",
            f"remaining={remaining},demand={req.demand}"))
    slot_cap = tenant.max_concurrency + tenant.burst
    if tstate.inflight >= slot_cap:
        return _Decision("queue", BasisEntry(
            now_ms, "tenant_inflight", "R_TENANT_SLOTS_FULL",
            f"inflight={tstate.inflight},cap={slot_cap}"))
    burst_used = 1 if tstate.inflight >= tenant.max_concurrency else 0
    if global_inflight >= spec.global_concurrency:
        return _Decision("queue", BasisEntry(
            now_ms, "global_inflight", "R_GLOBAL_SLOTS_FULL",
            f"global={global_inflight},cap={spec.global_concurrency}"))
    inst = spec.instance_of(req.tenant_id)
    state = health.state_at(inst.instance_id, now_ms)
    if state == "ejected":
        return _Decision("reject", BasisEntry(
            now_ms, "instance_health", "R_INSTANCE_EJECTED",
            f"instance={inst.instance_id}"))
    if state != "healthy":
        return _Decision("queue", BasisEntry(
            now_ms, "instance_health", "R_INSTANCE_NOT_HEALTHY",
            f"instance={inst.instance_id},state={state}"))
    return _Decision("grant", BasisEntry(
        now_ms, "grant", "R_GRANT",
        f"instance={inst.instance_id},burst_used={burst_used}"))


def run_arbitration(spec: Spec, requests: list, transitions: list, sink=None) -> RunResult:
    """Run the deterministic arbitration. `sink`, when given, receives
    on_grant(outcome, now_ms) and on_finalize(outcome) callbacks in
    processing order (used by the ledger persistence layer)."""
    health = HealthTimeline(spec.instances.keys())
    tstates = {tid: _TenantState() for tid in spec.tenants}
    outcomes = {r.request_id: RequestOutcome(r.request_id, r.tenant_id, r.demand, r.arrival_ms)
                for r in requests}
    finalize_order: list = []
    global_inflight = 0
    queue: list = []

    events: list = []
    seq = 0
    for t in transitions:
        heapq.heappush(events, (t.virt_ms, PRIO_TRANSITION, seq, t))
        seq += 1
    for r in requests:
        heapq.heappush(events, (r.arrival_ms, PRIO_ARRIVAL, seq, r))
        seq += 1

    def grant(req: Request, now_ms: int) -> None:
        nonlocal global_inflight
        oc = outcomes[req.request_id]
        oc.grant_ms = now_ms
        if oc.wait_start_ms >= 0:
            oc.wait_end_ms = now_ms
        ts = tstates[req.tenant_id]
        ts.consumed += req.demand
        ts.inflight += 1
        global_inflight += 1
        if sink is not None:
            sink.on_grant(oc, now_ms)
        duration = service_duration_ms(spec, req.tenant_id, req.demand)
        nonlocal_seq[0] += 1
        heapq.heappush(events, (now_ms + duration, PRIO_COMPLETION, nonlocal_seq[0], req))

    nonlocal_seq = [10_000_000]

    def finalize(req: Request, outcome: str, now_ms: int) -> None:
        oc = outcomes[req.request_id]
        oc.outcome = outcome
        oc.done_ms = now_ms
        finalize_order.append(req.request_id)
        if sink is not None:
            sink.on_finalize(oc)

    def drain_queue(now_ms: int) -> None:
        nonlocal global_inflight
        still_waiting = []
        for req in queue:
            oc = outcomes[req.request_id]
            decision = _evaluate(spec, req, now_ms, tstates[req.tenant_id], global_inflight, health)
            oc.basis.append(decision.entry)
            if decision.verdict == "grant":
                grant(req, now_ms)
            elif decision.verdict == "reject":
                finalize(req, OUTCOME_REJECTED, now_ms)
            else:
                still_waiting.append(req)
        queue[:] = still_waiting

    while events:
        now_ms, prio, _seq, payload = heapq.heappop(events)
        if prio == PRIO_COMPLETION:
            req = payload
            tstates[req.tenant_id].inflight -= 1
            global_inflight -= 1
            finalize(req, OUTCOME_GRANTED, now_ms)
            drain_queue(now_ms)
        elif prio == PRIO_TRANSITION:
            health.apply(payload)
            drain_queue(now_ms)
        else:
            req = payload
            oc = outcomes[req.request_id]
            decision = _evaluate(spec, req, now_ms, tstates[req.tenant_id], global_inflight, health)
            oc.basis.append(decision.entry)
            if decision.verdict == "grant":
                grant(req, now_ms)
            elif decision.verdict == "reject":
                finalize(req, OUTCOME_REJECTED, now_ms)
            else:
                oc.wait_start_ms = now_ms
                queue.append(req)

    for req in queue:
        oc = outcomes[req.request_id]
        oc.basis.append(BasisEntry(oc.wait_start_ms, "end_of_stream",
                                   "R_END_OF_STREAM", "request still queued at end of event stream"))
        finalize(req, OUTCOME_REJECTED, oc.wait_start_ms)

    return RunResult(
        outcomes=outcomes,
        finalize_order=finalize_order,
        tenant_consumed={tid: st.consumed for tid, st in tstates.items()},
        transitions=list(transitions),
    )
