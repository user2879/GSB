"""确定性调度引擎。

引擎只消费事件流携带的单调毫秒（:class:`DeterministicClock`），因此同一声明
在任意批次切分、任意真实等待、在线补算或重启重放下，逐节点余量、令牌归属
与聚合输入都逐字节一致。真实 TCP/NumPy 聚合由 :mod:`dispatch_tokens.runner`
在引擎裁决之后执行，引擎本身不做网络 IO。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .bucket import TokenBucket
from .clock import DeterministicClock
from .errors import LeaseError
from .fixedpoint import quantize
from .invalidation import (InvalidationResult, diff_revision,
                           propagate_invalidation, stable_topo_order)
from .lease import DEFAULT_TTL_MS, LeaseRegistry
from .spec import Event, Revision, Spec
from .watermark import (comparison_error_bound, demand_gap,
                        global_quota_error_bound, node_water_level)


@dataclass
class ShardOutcome:
    shard_id: str
    node_id: str
    status: str
    param_version: int
    token_id: str | None
    token_version: int | None
    summary: dict[str, Any] | None
    ownership: dict[str, Any]
    bucket: dict[str, Any]
    cursor_seq: int
    updated_ms: int


@dataclass
class EngineResult:
    outcomes: dict[str, ShardOutcome]
    refill_audit: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    invalidations: list[dict[str, Any]]
    water_levels: list[dict[str, Any]]
    revision_carryovers: list[dict[str, Any]]
    error_bounds: dict[str, Any]
    dispatch_plans: list[dict[str, Any]]
    tokens: list[dict[str, Any]]
    shard_nodes: dict[str, str]
    current_versions: dict[str, int]


class SchedulerEngine:
    def __init__(self, spec: Spec, ttl_ms: int = DEFAULT_TTL_MS) -> None:
        self.spec = spec
        self.ttl_ms = ttl_ms
        self.clock = DeterministicClock(spec.events[0].ts_ms
                                        if spec.events else 0)
        self.registry = LeaseRegistry()
        self.topo_order = stable_topo_order(spec.shards)
        self.topo_index = {sid: idx for idx, sid in enumerate(self.topo_order)}
        self.current_scale = self._initial_scale()
        self.global_rate, self.global_burst = self._initial_globals()
        self.current_version = spec.revisions[0].version
        self.buckets: dict[str, TokenBucket] = {
            node.node_id: TokenBucket(
                node_id=node.node_id,
                rate=self._effective_rate(node.node_id, spec.revisions[0]),
                burst=self._effective_burst(node.node_id, spec.revisions[0]),
                scale=self.current_scale, bucket_key=node.bucket_key,
                start_ms=self.clock.now_ms(),
                param_version=spec.revisions[0].version,
            )
            for node in spec.nodes
        }
        self.refill_audit: list[dict[str, Any]] = []
        self.invalidations: list[InvalidationResult] = []
        self.carryovers: list[dict[str, Any]] = []
        self.dispatch_plans: list[dict[str, Any]] = []
        self.outcomes: dict[str, ShardOutcome] = {}
        # 记录每个分片最后一次下发的节点，用于节点额度失效传播。
        self.shard_nodes: dict[str, str] = {}
        self._token_counter = 0

    def _initial_scale(self) -> int:
        for revision in self.spec.revisions:
            if revision.scale is not None:
                return revision.scale
        return 1

    def _initial_globals(self) -> tuple[int, int]:
        rate = 0
        burst = 0
        for revision in self.spec.revisions:
            if revision.global_rate is not None:
                rate = revision.global_rate
            if revision.global_burst is not None:
                burst = revision.global_burst
        return rate, burst

    def _effective_rate(self, node_id: str, revision: Revision) -> int:
        node = self.spec.node(node_id)
        if node_id in revision.node_rates:
            return revision.node_rates[node_id]
        if revision.global_rate is not None:
            return revision.global_rate
        return node.rate

    def _effective_burst(self, node_id: str, revision: Revision) -> int:
        node = self.spec.node(node_id)
        if node_id in revision.node_bursts:
            return revision.node_bursts[node_id]
        if revision.global_burst is not None:
            return revision.global_burst
        return node.burst

    def run(self, events: list[Event] | None = None) -> EngineResult:
        events = events if events is not None else list(self.spec.events)
        # 同到达序号稳定重排：分片标识序 + 事件类型序。
        for event in self.registry.stable_event_order(events):
            self._handle_event(event)
        return self.snapshot()

    def _handle_event(self, event: Event) -> None:
        self.clock.advance_to(event.ts_ms)
        if event.kind == "revise":
            self._handle_revise(event)
            return
        # 非修订事件先做真实到期判定。
        self.registry.expire_if_due(self.clock.now_ms(), event.seq,
                                    self.current_version)
        if event.kind == "dispatch":
            self._handle_dispatch(event)
        elif event.kind == "renew":
            self._handle_renew(event)
        elif event.kind == "cancel":
            self._handle_cancel(event)
        elif event.kind == "exhausted":
            self._handle_exhausted(event)

    def _handle_renew(self, event: Event) -> None:
        assert event.shard_id
        token = self.registry.active_token_for_shard(event.shard_id)
        if token is None:
            self.registry.decisions.append(
                self._bare_decision(event, "RENEW_NO_ACTIVE_LEASE",
                                    "renew_without_active_lease"))
            return
        bucket = self.buckets[token.node_id]
        refill = bucket.refill(self.clock.now_ms(), self.current_version)
        self.refill_audit.append(asdict(refill))
        # 引擎侧乐观版本判定；节点侧 RPC 校验由 runner 执行。
        # 拒绝（E_LEASE）已写入裁决表且不改变到期时刻，这里吞掉异常以便
        # 事件流整体可重放；测试可直接调用 registry.renew 断言异常路径。
        try:
            self.registry.renew(
                token_id=token.token_id,
                expected_version=event.expected_version,
                now_ms=self.clock.now_ms(), ttl_ms=self.ttl_ms,
                event_seq=event.seq, param_version=self.current_version,
                node_check_ok=True)
        except LeaseError:
            return
        if event.shard_id in self.outcomes:
            outcome = self.outcomes[event.shard_id]
            outcome.token_version = token.token_version
            outcome.ownership = self._last_decision()
            outcome.cursor_seq = event.seq
            outcome.updated_ms = event.ts_ms
            outcome.bucket = bucket.state.to_dict()

    def _revise_by_version(self, version: int) -> Revision:
        for revision in self.spec.revisions:
            if revision.version == version:
                return revision
        raise KeyError(version)

    def _handle_revise(self, event: Event) -> None:
        revision = self._revise_by_version(event.version)
        previous = self._previous_revision(revision)
        # 先逐桶补到提交时刻并结转，绝不重算历史。
        for node_id in sorted(self.buckets):
            record = self.buckets[node_id].apply_revision(
                new_rate=self._effective_rate(node_id, revision),
                new_burst=self._effective_burst(node_id, revision),
                new_scale=revision.scale,
                committed_ms=event.ts_ms, new_version=revision.version,
            )
            self.refill_audit.append(record["refill"])
            self.carryovers.append(record)
        if revision.scale is not None:
            self.current_scale = revision.scale
        if revision.global_rate is not None:
            self.global_rate = revision.global_rate
        if revision.global_burst is not None:
            self.global_burst = revision.global_burst
        self.current_version = revision.version
        result = propagate_invalidation(
            self.spec, previous, revision, dict(self.shard_nodes))
        self.invalidations.append(result)
        self.registry.decisions.append(
            self._revision_decision(event, revision, result))

    def _previous_revision(self, revision: Revision) -> Revision | None:
        if revision.parent is None:
            return None
        return self._revise_by_version(revision.parent)

    def _revision_decision(self, event: Event, revision: Revision,
                           result: InvalidationResult):
        from .lease import OwnershipDecision
        changes = diff_revision(self._previous_revision(revision), revision)
        basis = "revision_commit:" + ",".join(
            f"{key}={','.join(values)}" for key, values in changes.items()
            if values)
        return OwnershipDecision(
            event_seq=event.seq, event_kind="revise", shard_id="",
            node_id="", basis=basis, param_version=revision.version,
            token_version=0, token_id=None, demand=0,
            decision=f"INVALIDATE:{len(result.invalidated)}")

    def _handle_dispatch(self, event: Event) -> None:
        assert event.shard_id and event.node_id
        node_id = event.node_id
        shard_id = event.shard_id
        bucket = self.buckets[node_id]
        refill = bucket.refill(self.clock.now_ms(), self.current_version)
        self.refill_audit.append(asdict(refill))
        shard = self.spec.shard(shard_id)
        demand_scaled = quantize(shard.weight, self.current_scale)
        balance = bucket.state.balance
        cap = bucket.state.cap()
        water = node_water_level(node_id, balance, cap, bucket.state.rate,
                                 self.current_scale)
        gap = demand_gap(balance, demand_scaled)
        bound = comparison_error_bound(self.current_scale)
        # 误差界内仍按稳定决胜规则放行，并标注受误差影响。
        affordable = gap >= 0
        error_influenced = abs(gap) <= bound
        plan = {
            "event_seq": event.seq,
            "shard_id": shard_id,
            "node_id": node_id,
            "demand": shard.weight,
            "demand_scaled": demand_scaled,
            "balance_before": balance,
            "gap": gap,
            "comparison_error_bound": bound,
            "affordable": affordable,
            "error_influenced": error_influenced,
            "topo_index": self.topo_index[shard_id],
            "priority": shard.priority,
            "water_level": water.to_dict(),
        }
        if not affordable:
            plan["decision"] = "REJECT_INSUFFICIENT_QUOTA"
            self.dispatch_plans.append(plan)
            self._record_outcome(event, shard_id, node_id,
                                 status="REJECTED", token=None, summary=None,
                                 ownership={"decision": "REJECT_NO_QUOTA",
                                            "event_seq": event.seq})
            return
        try:
            token = self.registry.issue(
                token_id=self._next_token_id(), shard_id=shard_id,
                node_id=node_id, now_ms=self.clock.now_ms(),
                demand=shard.weight, param_version=self.current_version,
                ttl_ms=self.ttl_ms, event_seq=event.seq,
                basis="dispatch_grant_affordable"
                      + ("|within_error_bound" if error_influenced else ""),
            )
        except LeaseError:
            plan["decision"] = "REJECT_LEASE_CONTENTION"
            self.dispatch_plans.append(plan)
            raise
        ok, remaining = bucket.try_consume(shard.weight)
        assert ok
        plan["decision"] = "GRANT"
        plan["balance_after"] = remaining
        self.dispatch_plans.append(plan)
        self.shard_nodes[shard_id] = node_id
        self._record_outcome(event, shard_id, node_id, status="DISPATCHED",
                             token=token, summary=None,
                             ownership=self._last_decision())

    def _handle_cancel(self, event: Event) -> None:
        assert event.shard_id
        shard_id = event.shard_id
        token = self.registry.active_token_for_shard(shard_id)
        if token is None:
            # 无活跃令牌：取消只记录裁决，不重复回收额度，避免双重归属。
            self.registry.decisions.append(
                self._bare_decision(event, "CANCEL_NO_ACTIVE_LEASE",
                                    "cancel_without_active_lease"))
            self._record_outcome(
                event, shard_id,
                node_id=self.shard_nodes.get(shard_id, ""),
                status="CANCELLED", token=None, summary=None,
                ownership=self._last_decision())
            return
        node_id = token.node_id
        bucket = self.buckets[node_id]
        refill = bucket.refill(self.clock.now_ms(), self.current_version)
        self.refill_audit.append(asdict(refill))
        cancelled = self.registry.cancel(
            shard_id, event.seq, self.current_version,
            basis="cancel_reclaims_held_quota_once")
        bucket.refund(cancelled.demand)
        self._record_outcome(event, shard_id, node_id, status="CANCELLED",
                             token=cancelled, summary=None,
                             ownership=self._last_decision())

    def _handle_exhausted(self, event: Event) -> None:
        assert event.shard_id
        shard_id = event.shard_id
        token = self.registry.active_token_for_shard(shard_id)
        if token is None:
            self.registry.decisions.append(
                self._bare_decision(event, "EXHAUSTED_NO_ACTIVE_LEASE",
                                    "exhausted_without_active_lease"))
            self._record_outcome(
                event, shard_id,
                node_id=self.shard_nodes.get(shard_id, ""),
                status="EXHAUSTED", token=None, summary=None,
                ownership=self._last_decision())
            return
        node_id = token.node_id
        bucket = self.buckets[node_id]
        refill = bucket.refill(self.clock.now_ms(), self.current_version)
        self.refill_audit.append(asdict(refill))
        exhausted = self.registry.mark_exhausted(
            shard_id, event.seq, self.current_version)
        # 耗尽不回收未使用额度之外的部分：仅释放持有关系，已消耗量保持消耗。
        self._record_outcome(event, shard_id, node_id, status="EXHAUSTED",
                             token=exhausted, summary=None,
                             ownership=self._last_decision())

    def renew_lease(self, token_id: str, expected_version: int,
                    event_seq: int, node_check_ok: bool):
        """供 runner 在真实节点乐观校验后驱动续约。"""
        return self.registry.renew(
            token_id=token_id, expected_version=expected_version,
            now_ms=self.clock.now_ms(), ttl_ms=self.ttl_ms,
            event_seq=event_seq, param_version=self.current_version,
            node_check_ok=node_check_ok)

    def _next_token_id(self) -> str:
        self._token_counter += 1
        return f"tok-{self._token_counter:06d}"

    def _bare_decision(self, event: Event, decision: str, basis: str):
        from .lease import OwnershipDecision
        return OwnershipDecision(
            event_seq=event.seq, event_kind=event.kind,
            shard_id=event.shard_id or "", node_id=event.node_id or "",
            basis=basis, param_version=self.current_version,
            token_version=0, token_id=None, demand=0, decision=decision)

    def _last_decision(self) -> dict[str, Any]:
        return asdict(self.registry.decisions[-1])

    def _record_outcome(self, event: Event, shard_id: str, node_id: str,
                        *, status: str, token, summary,
                        ownership: dict[str, Any]) -> None:
        bucket_state = (self.buckets[node_id].state.to_dict()
                        if node_id and node_id in self.buckets
                        else {})
        self.outcomes[shard_id] = ShardOutcome(
            shard_id=shard_id, node_id=node_id, status=status,
            param_version=self.current_version,
            token_id=token.token_id if token else None,
            token_version=token.token_version if token else None,
            summary=summary, ownership=ownership, bucket=bucket_state,
            cursor_seq=event.seq, updated_ms=event.ts_ms)

    def snapshot(self) -> EngineResult:
        water_levels = [
            node_water_level(
                node_id, bucket.state.balance, bucket.state.cap(),
                bucket.state.rate, self.current_scale).to_dict()
            for node_id, bucket in sorted(self.buckets.items())
        ]
        scales = [self.current_scale for _ in self.buckets]
        error_bounds = {
            "single_node_quantization_error": 1,
            "comparison_error_bound": comparison_error_bound(
                self.current_scale),
            "global_quota_error_upper_bound":
                global_quota_error_bound(scales),
            "truncation": "TRUNC_TOWARD_ZERO",
            "scale": self.current_scale,
        }
        return EngineResult(
            outcomes=dict(self.outcomes),
            refill_audit=list(self.refill_audit),
            decisions=[asdict(d) for d in self.registry.decisions],
            invalidations=[item.to_dict() for item in self.invalidations],
            water_levels=water_levels,
            revision_carryovers=list(self.carryovers),
            error_bounds=error_bounds,
            dispatch_plans=list(self.dispatch_plans),
            tokens=[token.to_dict() for token in self.registry.all_tokens()],
            shard_nodes=dict(self.shard_nodes),
            current_versions={
                node_id: bucket.state.param_version
                for node_id, bucket in self.buckets.items()},
        )
