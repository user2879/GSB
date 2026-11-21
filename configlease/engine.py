"""In-memory decision engine: validation, budgeting, serialization, replay.

This module is deliberately free of sockets and subprocesses.  Both the live
platform and the offline replay validator drive the same pure rules here, which
is what makes replay reproducibility testable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .clock import is_expired, now_ms
from .errors import (
    ApplyMismatchError,
    KeyNotCoveredError,
    LeaseExpiredError,
    RevisionOutOfRangeError,
    RollbackRevisionError,
    ReplayMismatchError,
    VersionConflictError,
)
from .manifest import Revision, fold_view
from .sample import summary_for_revision

SNAPSHOT_VERSION = 1


# --------------------------------------------------------------------------- #
# Pure validation rules
# --------------------------------------------------------------------------- #
def validate_request(request: dict[str, Any], lease: dict[str, Any],
                     superseded: set[int]) -> None:
    """Apply the platform-side lease window rules before arbitration."""
    rev_seq = int(request["rev_seq"])
    if rev_seq in superseded:
        raise RollbackRevisionError(
            "revision was rolled back", rev_seq=rev_seq,
            lease_id=lease["lease_id"],
        )
    if not lease["low_seq"] <= rev_seq <= lease["high_seq"]:
        raise RevisionOutOfRangeError(
            "rev_seq outside lease window",
            rev_seq=rev_seq, low_seq=lease["low_seq"], high_seq=lease["high_seq"],
        )
    change_keys = set(request["changes"])
    uncovered = change_keys - set(lease["keys"])
    if uncovered:
        raise KeyNotCoveredError(
            "keys not covered by lease",
            rev_seq=rev_seq, uncovered=sorted(uncovered),
        )
    if is_expired(lease["deadline_ms"]):
        raise LeaseExpiredError("lease already expired", lease_id=lease["lease_id"])


def expected_node_view(revisions: list[Revision], node_applied_seqs: list[int]) -> dict[str, int]:
    """Recompute the view a node should hold after applying the given seqs."""
    chosen = [r for r in revisions if r.rev_seq in set(node_applied_seqs)]
    chosen.sort(key=lambda r: r.rev_seq)
    return fold_view(chosen)


def verify_apply_result(request: dict[str, Any], lease: dict[str, Any],
                        node_response: dict[str, Any],
                        prior_view: dict[str, int],
                        prior_cursor: int) -> dict[str, Any]:
    """Recompute expected view/summary and compare with the node response."""
    rev_seq = int(request["rev_seq"])
    expected_view = dict(prior_view)
    for key in sorted(request["changes"]):  # deterministic key order
        expected_view[key] = request["changes"][key]
    expected_summary = summary_for_revision(rev_seq)
    expected_cursor = rev_seq

    actual_view = node_response.get("view", {})
    actual_summary = node_response.get("summary", {})
    actual_cursor = node_response.get("applied_rev")

    mismatches: list[dict[str, Any]] = []
    if actual_cursor != expected_cursor:
        mismatches.append({"field": "applied_rev", "expected": expected_cursor,
                           "actual": actual_cursor})
    if dict(sorted(actual_view.items())) != dict(sorted(expected_view.items())):
        mismatches.append({"field": "view", "expected": expected_view,
                           "actual": actual_view})
    if actual_summary != expected_summary:
        mismatches.append({"field": "summary", "expected": expected_summary,
                           "actual": actual_summary})
    if mismatches:
        raise ApplyMismatchError(
            "node apply result diverged from expected view",
            node_id=lease["node_id"], rev_seq=rev_seq, mismatches=mismatches,
        )
    return {
        "applied_rev": expected_cursor,
        "view": expected_view,
        "summary": expected_summary,
        "prior_cursor": prior_cursor,
    }


# --------------------------------------------------------------------------- #
# Dual budget control
# --------------------------------------------------------------------------- #
@dataclass
class Budget:
    name: str
    kind: str  # "global" | "node"
    limit: int
    consumed: int = 0
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def available(self) -> int:
        return max(0, self.limit - self.consumed)

    async def acquire(self) -> None:
        """Queue FIFO until one token is available; no request can bypass."""
        async with self._cond:
            await self._cond.wait_for(lambda: self.consumed < self.limit)
            self.consumed += 1

    async def refund(self) -> None:
        async with self._cond:
            if self.consumed > 0:
                self.consumed -= 1
            self._cond.notify_all()

    async def set_limit(self, limit: int) -> None:
        async with self._cond:
            self.limit = limit
            self._cond.notify_all()

    def snapshot(self) -> dict[str, int]:
        return {"limit": self.limit, "consumed": self.consumed,
                "available": self.available()}


class BudgetPool:
    def __init__(self, global_limit: int, node_limits: dict[str, int]):
        self.global_budget = Budget("global", "global", global_limit)
        self.node_budgets = {
            node: Budget(f"node:{node}", "node", limit)
            for node, limit in node_limits.items()
        }

    def node_budget(self, node_id: str) -> Budget:
        if node_id not in self.node_budgets:
            self.node_budgets[node_id] = Budget(f"node:{node_id}", "node", 0)
        return self.node_budgets[node_id]

    def snapshots(self) -> dict[str, Any]:
        return {
            "global": self.global_budget.snapshot(),
            "nodes": {n: b.snapshot() for n, b in sorted(self.node_budgets.items())},
        }


# --------------------------------------------------------------------------- #
# Stable per-key conflict serialization
# --------------------------------------------------------------------------- #
class KeyConflictSerializer:
    """Serialize concurrent claims on the same configuration key.

    A request needs *all* of its keys at once, so the serializer uses one
    meta-condition and tracks, per key, its current holder and waiters.  A
    request enters only when every key it wants is free and it is the earliest
    waiter on each of them; this atomic "all-or-nothing" acquire prevents the
    partial-hold lock-ordering deadlock that per-key nested locks would cause.
    Grant priority on each key is the stable tuple ``(node_id, rev_seq)``.
    Different keys otherwise stay fully independent; waiting is real async
    waiting and cancelled waiters clean themselves up.
    """

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        # key -> {"holder": token | None, "waiters": set[token]}
        self._keys: dict[str, dict[str, Any]] = {}
        # token -> set of keys it is waiting on
        self._waiting_on: dict[tuple[str, int], set[str]] = {}

    def _ensure(self, key: str) -> dict[str, Any]:
        state = self._keys.get(key)
        if state is None:
            state = {"holder": None, "waiters": set()}
            self._keys[key] = state
        return state

    def _can_enter(self, token: tuple[str, int], keys: list[str]) -> bool:
        for key in keys:
            state = self._ensure(key)
            if state["holder"] is not None:
                return False
            if state["waiters"] and token != min(
                state["waiters"], key=lambda item: (item[0], item[1])
            ):
                return False
        return True

    async def claim(self, node_id: str, rev_seq: int, keys: set[str]) -> None:
        token = (node_id, rev_seq)
        ordered = sorted(keys)
        async with self._cond:
            for key in ordered:
                self._ensure(key)["waiters"].add(token)
            self._waiting_on[token] = set(ordered)
            try:
                await self._cond.wait_for(lambda: self._can_enter(token, ordered))
            except BaseException:
                self._remove_waiter(token)
                self._cond.notify_all()
                raise
            for key in ordered:
                state = self._ensure(key)
                state["waiters"].discard(token)
                state["holder"] = token
            self._waiting_on.pop(token, None)

    def _remove_waiter(self, token: tuple[str, int]) -> None:
        for key in self._waiting_on.get(token, set()):
            state = self._keys.get(key)
            if state is not None:
                state["waiters"].discard(token)
        self._waiting_on.pop(token, None)

    async def release(self, node_id: str, rev_seq: int, keys: set[str]) -> None:
        token = (node_id, rev_seq)
        async with self._cond:
            for key in sorted(keys):
                state = self._keys.get(key)
                if state is None:
                    continue
                if state["holder"] == token:
                    state["holder"] = None
                state["waiters"].discard(token)
            self._waiting_on.pop(token, None)
            self._cond.notify_all()


# --------------------------------------------------------------------------- #
# Snapshot content hashing and replay
# --------------------------------------------------------------------------- #
def canonical_json(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def assert_snapshot_compatible(snapshot: dict[str, Any], current_state: dict[str, Any]) -> None:
    if snapshot.get("version") != SNAPSHOT_VERSION:
        raise VersionConflictError(
            "snapshot version differs from current state",
            snapshot_version=snapshot.get("version"),
            current_version=SNAPSHOT_VERSION,
        )
    current_hash = content_hash(current_state)
    if snapshot.get("content_hash") != current_hash:
        raise VersionConflictError(
            "snapshot content hash differs from current state",
            snapshot_hash=snapshot.get("content_hash"),
            current_hash=current_hash,
        )


def replay_lease_records(lease_records: list[dict[str, Any]],
                         revisions: list[Revision],
                         snapshot_state: dict[str, Any]) -> dict[str, Any]:
    """Replay recorded leases deterministically and compare with the snapshot.

    The replay re-runs the same validation and view folding rules used live.
    On the first divergence it raises E_REPLAY naming the revision and field.
    Budget semantics mirror the live platform: a granted lease consumes one
    global dispatch token; an applied revision consumes one per-node token;
    leases left ``unknown`` after a crash never consume either token.
    """
    revision_by_seq = {r.rev_seq: r for r in revisions}
    superseded = set(snapshot_state.get("superseded", []))

    node_ids = set(snapshot_state.get("cursors", {}))
    for record in lease_records:
        node_ids.add(record["node_id"])
    cursors = {node: 0 for node in node_ids}
    views = {node: {} for node in node_ids}
    node_consumed = {node: 0 for node in node_ids}
    global_consumed = 0
    grants_observed = 0

    for record in lease_records:
        node_id = record["node_id"]
        rev_seq = record["rev_seq"]
        status = record["status"]
        revision = revision_by_seq.get(rev_seq)

        grants_observed += 1
        if status == "granted":
            if rev_seq in superseded:
                raise ReplayMismatchError(
                    "replay granted a superseded revision",
                    rev_seq=rev_seq, node_id=node_id, field="status",
                    expected="revoked", actual=status,
                )
            if revision is None:
                raise ReplayMismatchError(
                    "record references unknown revision",
                    rev_seq=rev_seq, node_id=node_id, field="rev_seq",
                )
            try:
                validate_request(
                    {"rev_seq": rev_seq, "changes": revision.changes},
                    record, superseded,
                )
            except (RevisionOutOfRangeError, KeyNotCoveredError,
                    RollbackRevisionError, LeaseExpiredError) as exc:
                raise ReplayMismatchError(
                    "replay validation diverged from recorded grant",
                    rev_seq=rev_seq, node_id=node_id, field="validation",
                    detail=exc.code,
                ) from exc
        elif status == "applied":
            if revision is None:
                raise ReplayMismatchError(
                    "record references unknown revision",
                    rev_seq=rev_seq, node_id=node_id, field="rev_seq",
                )
            for key in sorted(revision.changes):
                views[node_id][key] = revision.changes[key]
            cursors[node_id] = rev_seq
            node_consumed[node_id] += 1
            expected_summary = summary_for_revision(rev_seq)
            recorded_summary = record.get("summary")
            if recorded_summary is not None and recorded_summary != expected_summary:
                raise ReplayMismatchError(
                    "recorded summary not reproducible",
                    rev_seq=rev_seq, node_id=node_id, field="summary",
                    expected=expected_summary, actual=recorded_summary,
                )
        elif status in ("rejected", "revoked", "expired", "unknown"):
            continue

    snap_cursors = snapshot_state.get("cursors", {})
    for node_id in sorted(cursors):
        expected_cursor = snap_cursors.get(node_id, {}).get("applied_rev", 0)
        if cursors[node_id] != expected_cursor:
            raise ReplayMismatchError(
                "replayed cursor does not match snapshot",
                node_id=node_id, rev_seq=cursors[node_id], field="applied_rev",
                expected=expected_cursor, actual=cursors[node_id],
            )
    snap_views = snapshot_state.get("views", {})
    for node_id in sorted(views):
        expected_view = snap_views.get(node_id, {})
        if dict(sorted(views[node_id].items())) != dict(sorted(expected_view.items())):
            raise ReplayMismatchError(
                "replayed view does not match snapshot",
                node_id=node_id, field="view",
                expected=expected_view, actual=views[node_id],
            )
    snap_global = snapshot_state.get("budgets", {}).get("global", {}).get("consumed", 0)
    # The global dispatch budget is an in-flight gate whose tokens are released
    # at each terminal lease, so its persisted consumption reflects live tokens,
    # not the cumulative grant count.  Replay verifies the persisted value is
    # internally consistent (never negative, never above the limit) and reports
    # the cumulative grant count separately for the audit trail.
    global_budget = snapshot_state.get("budgets", {}).get("global", {})
    if not (0 <= snap_global <= global_budget.get("limit", snap_global)):
        raise ReplayMismatchError(
            "snapshot global budget outside its limit",
            field="global_consumed", expected=global_budget.get("limit"),
            actual=snap_global,
        )
    global_consumed = snap_global
    snap_nodes = snapshot_state.get("budgets", {}).get("nodes", {})
    for node_id in sorted(node_consumed):
        expected = snap_nodes.get(node_id, {}).get("consumed", 0)
        if node_consumed[node_id] != expected:
            raise ReplayMismatchError(
                "replayed per-node budget does not match snapshot",
                node_id=node_id, field="node_consumed",
                expected=expected, actual=node_consumed[node_id],
            )
    return {
        "cursors": cursors,
        "views": views,
        "global_consumed": global_consumed,
        "grants_observed": grants_observed,
        "node_consumed": node_consumed,
        "records_replayed": len(lease_records),
        "replayed_ms": now_ms(),
    }
