"""下发许可令牌与唯一归属裁决。

令牌是节点令牌额度的唯一占用凭证：同一节点同一时刻，一份额度只能被一个
分片的一个令牌持有。续约采用乐观版本号——续约请求必须携带当前令牌版本，
节点侧校验通过才递增版本并延长到期时刻；版本不符抛 E_LEASE，到期时刻
保持不变。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .errors import LeaseError

# 默认令牌有效期（单调毫秒）；可由调度器参数覆盖。
DEFAULT_TTL_MS = 30_000
# 同到达序号事件重排时的稳定类型次序。
EVENT_TYPE_ORDER = {"dispatch": 0, "cancel": 1, "exhausted": 2, "revise": 3}


@dataclass
class LeaseToken:
    token_id: str
    shard_id: str
    node_id: str
    granted_ms: int
    expires_ms: int
    renew_count: int
    token_version: int
    demand: int
    active: bool = True

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OwnershipDecision:
    """一次归属变化的唯一裁决记录。"""

    event_seq: int
    event_kind: str
    shard_id: str
    node_id: str
    basis: str
    param_version: int
    token_version: int
    token_id: str | None
    demand: int
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseRegistry:
    """维护每节点的活跃令牌与全部归属裁决。"""

    def __init__(self) -> None:
        self._tokens: dict[str, LeaseToken] = {}
        self._active_by_shard: dict[str, str] = {}
        self._active_by_node: dict[str, str] = {}
        self.decisions: list[OwnershipDecision] = []

    @staticmethod
    def stable_event_order(events: list[Any]) -> list[Any]:
        """同到达序号下按分片标识序、再按事件类型序的稳定排序。"""
        return sorted(
            events,
            key=lambda e: (
                e.seq,
                e.shard_id or "",
                EVENT_TYPE_ORDER.get(e.kind, 99),
                e.node_id or "",
            ),
        )

    def issue(self, token_id: str, shard_id: str, node_id: str, now_ms: int,
              demand: int, param_version: int, ttl_ms: int,
              event_seq: int, basis: str) -> LeaseToken:
        if shard_id in self._active_by_shard:
            raise LeaseError(
                f"shard {shard_id!r} already holds an active lease",
                location=f"events[{event_seq}]", field="shard_id")
        if node_id in self._active_by_node:
            holder = self._tokens[self._active_by_node[node_id]]
            raise LeaseError(
                f"node {node_id!r} quota already held by shard "
                f"{holder.shard_id!r}",
                location=f"events[{event_seq}]", field="node_id")
        token = LeaseToken(
            token_id=token_id, shard_id=shard_id, node_id=node_id,
            granted_ms=now_ms, expires_ms=now_ms + ttl_ms,
            renew_count=0, token_version=1, demand=demand, active=True,
        )
        self._tokens[token_id] = token
        self._active_by_shard[shard_id] = token_id
        self._active_by_node[node_id] = token_id
        self.decisions.append(OwnershipDecision(
            event_seq=event_seq, event_kind="dispatch", shard_id=shard_id,
            node_id=node_id, basis=basis, param_version=param_version,
            token_version=token.token_version, token_id=token_id,
            demand=demand, decision="GRANT",
        ))
        return token

    def active_token_for_node(self, node_id: str) -> LeaseToken | None:
        token_id = self._active_by_node.get(node_id)
        return self._tokens.get(token_id) if token_id else None

    def active_token_for_shard(self, shard_id: str) -> LeaseToken | None:
        token_id = self._active_by_shard.get(shard_id)
        return self._tokens.get(token_id) if token_id else None

    def expire_if_due(self, now_ms: int, event_seq: int,
                      param_version: int) -> list[LeaseToken]:
        """基于单调时钟的真实到期判定：到期令牌释放占用并留裁决。"""
        expired: list[LeaseToken] = []
        for token in list(self._tokens.values()):
            if not token.active:
                continue
            if token.is_expired(now_ms):
                self._release(token, event_seq, "expired", param_version,
                              decision="EXPIRE")
                expired.append(token)
        return expired

    def cancel(self, shard_id: str, event_seq: int, param_version: int,
               basis: str) -> LeaseToken:
        token_id = self._active_by_shard.get(shard_id)
        if token_id is None:
            raise LeaseError(
                f"no active lease for shard {shard_id!r}",
                location=f"events[{event_seq}]", field="shard_id")
        token = self._tokens[token_id]
        self._release(token, event_seq, basis, param_version,
                      decision="CANCEL")
        return token

    def mark_exhausted(self, shard_id: str, event_seq: int,
                       param_version: int) -> LeaseToken:
        token_id = self._active_by_shard.get(shard_id)
        if token_id is None:
            raise LeaseError(
                f"no active lease for shard {shard_id!r}",
                location=f"events[{event_seq}]", field="shard_id")
        token = self._tokens[token_id]
        self._release(token, event_seq, "token_exhausted_event",
                      param_version, decision="EXHAUSTED")
        return token

    def renew(self, token_id: str, expected_version: int, now_ms: int,
              ttl_ms: int, event_seq: int, param_version: int,
              node_check_ok: bool) -> LeaseToken:
        """乐观续约：版本不符或节点校验失败时拒绝且不延长到期。"""
        token = self._tokens.get(token_id)
        if token is None or not token.active:
            raise LeaseError(f"unknown or inactive token {token_id!r}",
                             location=f"events[{event_seq}]",
                             field="token_id")
        if token.token_version != expected_version:
            self.decisions.append(OwnershipDecision(
                event_seq=event_seq, event_kind="renew",
                shard_id=token.shard_id, node_id=token.node_id,
                basis="optimistic_version_mismatch",
                param_version=param_version,
                token_version=token.token_version, token_id=token_id,
                demand=token.demand, decision="RENEW_REJECTED",
            ))
            raise LeaseError(
                f"token version conflict: expected {token.token_version}, "
                f"got {expected_version}; expiry unchanged",
                location=f"events[{event_seq}]", field="token_version")
        if not node_check_ok:
            self.decisions.append(OwnershipDecision(
                event_seq=event_seq, event_kind="renew",
                shard_id=token.shard_id, node_id=token.node_id,
                basis="node_optimistic_check_failed",
                param_version=param_version,
                token_version=token.token_version, token_id=token_id,
                demand=token.demand, decision="RENEW_REJECTED",
            ))
            raise LeaseError(
                "node-side optimistic lease check rejected renewal",
                location=f"events[{event_seq}]", field="token_version")
        token.renew_count += 1
        token.token_version += 1
        token.expires_ms = now_ms + ttl_ms
        self.decisions.append(OwnershipDecision(
            event_seq=event_seq, event_kind="renew",
            shard_id=token.shard_id, node_id=token.node_id,
            basis="optimistic_version_match", param_version=param_version,
            token_version=token.token_version, token_id=token_id,
            demand=token.demand, decision="RENEW_GRANT",
        ))
        return token

    def _release(self, token: LeaseToken, event_seq: int, basis: str,
                 param_version: int, decision: str) -> None:
        token.active = False
        self._active_by_shard.pop(token.shard_id, None)
        self._active_by_node.pop(token.node_id, None)
        self.decisions.append(OwnershipDecision(
            event_seq=event_seq, event_kind=decision.lower(),
            shard_id=token.shard_id, node_id=token.node_id, basis=basis,
            param_version=param_version, token_version=token.token_version,
            token_id=token.token_id, demand=token.demand, decision=decision,
        ))

    def all_tokens(self) -> list[LeaseToken]:
        return list(self._tokens.values())
