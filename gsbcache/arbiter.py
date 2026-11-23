"""Deterministic lease arbitration, queueing, custody transitions and timing."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import ArbiterError, ReplyError
from .ledger import Ledger
from .protocol import NodeClient
from .spec import ArtifactSpec, NodeSpec, Spec
from .util import monotonic_ms, monotonic_ns, now_ms


@dataclass
class Candidate:
    node_id: str
    actual_digest: str | None
    expected_digest: str
    digest_match: bool
    refs_reusable: bool
    bad_references: list[str]
    capacity_remaining: int
    capacity_slots: int
    capacity_used: int
    quota_remaining: int
    lease_limit: int
    active_leases: int
    eligible: bool
    veto_reason: str
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "actual_digest": self.actual_digest,
            "expected_digest": self.expected_digest,
            "digest_match": self.digest_match,
            "refs_reusable": self.refs_reusable,
            "bad_references": list(self.bad_references),
            "capacity_remaining": self.capacity_remaining,
            "capacity_slots": self.capacity_slots,
            "capacity_used": self.capacity_used,
            "quota_remaining": self.quota_remaining,
            "lease_limit": self.lease_limit,
            "active_leases": self.active_leases,
            "eligible": self.eligible,
            "veto_reason": self.veto_reason,
            "rank": self.rank,
        }


@dataclass
class RequestOutcome:
    ok: bool
    request_id: str
    artifact_key: str
    node_id: str | None
    lease_id: str | None
    expires_at_ms: int | None
    digests: dict[str, str]
    reusable_keys: list[str]
    candidates: list[dict[str, Any]]
    arbitration_index: int | None
    queue_position: int | None
    queue_reason: str | None
    queue_started_monotonic_ms: int
    queue_ended_monotonic_ms: int
    queue_ms: int
    capacity_wait_ms: int
    gate_wait_ms: int
    execution_started_monotonic_ms: int
    execution_ended_monotonic_ms: int
    execution_ms: int
    global_inflight_at_start: int
    node_inflight_at_start: int
    event_seq: int | None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.ok,
            "request_id": self.request_id,
            "artifact_key": self.artifact_key,
            "node_id": self.node_id,
            "lease_id": self.lease_id,
            "expires_at_ms": self.expires_at_ms,
            "digests": dict(self.digests),
            "reusable_keys": list(self.reusable_keys),
            "candidates": list(self.candidates),
            "arbitration_index": self.arbitration_index,
            "queue_position": self.queue_position,
            "queue_reason": self.queue_reason,
            "queue_started_monotonic_ms": self.queue_started_monotonic_ms,
            "queue_ended_monotonic_ms": self.queue_ended_monotonic_ms,
            "queue_ms": self.queue_ms,
            "capacity_wait_ms": self.capacity_wait_ms,
            "gate_wait_ms": self.gate_wait_ms,
            "execution_started_monotonic_ms": self.execution_started_monotonic_ms,
            "execution_ended_monotonic_ms": self.execution_ended_monotonic_ms,
            "execution_ms": self.execution_ms,
            "global_inflight_at_start": self.global_inflight_at_start,
            "node_inflight_at_start": self.node_inflight_at_start,
            "event_seq": self.event_seq,
        }
        if self.error is not None:
            result["error"] = self.error
            result["reason"] = self.error.get("message")
        return result


class InflightMeter:
    def __init__(self, node_ids: Iterable[str]) -> None:
        self.lock = threading.Lock()
        self.global_current = 0
        self.global_max = 0
        self.node_current = {node_id: 0 for node_id in node_ids}
        self.node_max = {node_id: 0 for node_id in node_ids}

    def enter(self, node_id: str) -> tuple[int, int]:
        with self.lock:
            self.global_current += 1
            self.node_current[node_id] = self.node_current.get(node_id, 0) + 1
            self.global_max = max(self.global_max, self.global_current)
            self.node_max[node_id] = max(self.node_max.get(node_id, 0), self.node_current[node_id])
            return self.global_current, self.node_current[node_id]

    def leave(self, node_id: str) -> None:
        with self.lock:
            self.global_current = max(0, self.global_current - 1)
            self.node_current[node_id] = max(0, self.node_current.get(node_id, 0) - 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "global_inflight_max": self.global_max,
                "global_inflight_current": self.global_current,
                "node_inflight_max": dict(sorted(self.node_max.items())),
                "node_inflight_current": dict(sorted(self.node_current.items())),
            }


class LeaseArbiter:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        clients: dict[str, NodeClient],
        session_id: str,
        *,
        lease_ms: int = 500,
        queue_timeout_ms: int = 15_000,
        request_timeout_s: float = 3.0,
    ) -> None:
        if lease_ms <= 0:
            raise ArbiterError("lease_ms must be positive", field="lease_ms")
        self.spec = spec
        self.ledger = ledger
        self.clients = clients
        self.session_id = session_id
        self.lease_ms = int(lease_ms)
        self.queue_timeout_ms = int(queue_timeout_ms)
        self.request_timeout_s = float(request_timeout_s)
        self.cond = threading.Condition(threading.RLock())
        self.snapshots: dict[str, dict[str, str]] = {}
        self.reusable: dict[str, set[str]] = {}
        self.pending_slots = {node.node_id: 0 for node in spec.nodes}
        self.pending_leases = {node.node_id: 0 for node in spec.nodes}
        self.reserved_keys: dict[str, str] = {}
        self.active_keys: dict[str, str] = {}
        self.active_leases: dict[str, dict[str, Any]] = {}
        self.waiting_count = 0
        existing_indices = [
            int(event["arbitration_index"])
            for event in ledger.get_events("lease_granted")
            if event.get("arbitration_index") is not None
        ]
        next_index = (max(existing_indices) + 1) if existing_indices else 1
        self.next_arbitration_index = next_index
        self.next_commit_index = next_index
        self.canceled_commit_indices: set[int] = set()
        self.meter = InflightMeter(node.node_id for node in spec.nodes)
        self.global_gate = threading.BoundedSemaphore(spec.pipeline.global_concurrency)
        self.node_gates = {
            node.node_id: threading.BoundedSemaphore(node.lease_limit) for node in spec.nodes
        }
        self.stop_event = threading.Event()
        self.reaper = threading.Thread(target=self._reap_loop, name="lease-reaper", daemon=True)
        self._load_active_leases()
        self.refresh_inventory()

    def _load_active_leases(self) -> None:
        for lease in self.ledger.get_active_leases():
            self.active_leases[lease["lease_id"]] = lease
            for key in lease["artifact_keys"]:
                self.active_keys[key] = lease["lease_id"]

    def refresh_inventory(self) -> dict[str, dict[str, Any]]:
        snapshots: dict[str, dict[str, str]] = {}
        reusable: dict[str, set[str]] = {}
        details: dict[str, dict[str, Any]] = {}
        for node in self.spec.nodes:
            response = self.clients[node.node_id].inventory(node.node_id)
            digests = {str(key): str(value) for key, value in response["digests"].items()}
            expected_keys = set(self.spec.artifacts)
            if set(digests) != expected_keys:
                raise ReplyError(
                    "node inventory does not cover the manifest artifact set",
                    node_id=node.node_id,
                    field="digests",
                    missing=sorted(expected_keys - set(digests)),
                    unexpected=sorted(set(digests) - expected_keys),
                )
            snapshots[node.node_id] = digests
            reusable[node.node_id] = self._reusable_keys_for_node(node, digests)
            details[node.node_id] = {
                "digests": digests,
                "reusable_keys": sorted(reusable[node.node_id]),
            }
        with self.cond:
            self.snapshots = snapshots
            self.reusable = reusable
            self.cond.notify_all()
        return details

    def _reusable_keys_for_node(self, node: NodeSpec, digests: dict[str, str]) -> set[str]:
        result: set[str] = set()
        for artifact in self.spec.ordered_artifacts:
            if digests.get(artifact.key) != artifact.expected_digest:
                continue
            if all(digests.get(ref) == self.spec.artifacts[ref].expected_digest for ref in artifact.refs):
                result.add(artifact.key)
        return result

    def start(self) -> "LeaseArbiter":
        self.reaper.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        if self.reaper.is_alive():
            self.reaper.join(timeout=2.0)

    def _active_key_owner(self, key: str) -> str | None:
        lease_id = self.active_keys.get(key)
        if lease_id:
            return self.active_leases.get(lease_id, {}).get("node_id")
        return None

    def _candidates_locked(self, artifact: ArtifactSpec) -> list[Candidate]:
        result: list[Candidate] = []
        ledger_nodes = {node["node_id"]: node for node in self.ledger.get_nodes()}
        key_owner = self._active_key_owner(artifact.key)
        reserved_by = self.reserved_keys.get(artifact.key)
        for node in self.spec.nodes:
            actual = self.snapshots.get(node.node_id, {}).get(artifact.key)
            digest_match = actual == artifact.expected_digest
            bad_refs = [
                ref
                for ref in artifact.refs
                if self.snapshots.get(node.node_id, {}).get(ref) != self.spec.artifacts[ref].expected_digest
            ]
            refs_ok = not bad_refs
            row = ledger_nodes.get(node.node_id, {})
            cap_slots = int(row.get("capacity_slots", node.capacity_slots))
            cap_used = int(row.get("capacity_used", 0))
            active_leases = int(row.get("active_leases", 0))
            cap_remaining = cap_slots - cap_used - self.pending_slots[node.node_id]
            quota_remaining = node.lease_limit - active_leases - self.pending_leases[node.node_id]
            eligible = True
            reason = "selected:digest_match_refs_reusable"
            if key_owner is not None:
                eligible = False
                reason = f"key_already_leased:node={key_owner}"
            elif reserved_by is not None:
                eligible = False
                reason = f"key_already_reserved:request={reserved_by}"
            elif not digest_match:
                eligible = False
                reason = f"digest_mismatch:expected={artifact.expected_digest}:actual={actual}"
            elif not refs_ok:
                eligible = False
                reason = "reference_digest_mismatch:keys=" + ",".join(sorted(bad_refs))
            elif quota_remaining <= 0:
                eligible = False
                reason = (
                    f"lease_limit_exhausted:limit={node.lease_limit}:active={active_leases}:"
                    f"pending={self.pending_leases[node.node_id]}"
                )
            elif cap_remaining <= 0:
                eligible = False
                reason = (
                    f"capacity_slots_exhausted:slots={cap_slots}:used={cap_used}:"
                    f"pending={self.pending_slots[node.node_id]}"
                )
            result.append(
                Candidate(
                    node_id=node.node_id,
                    actual_digest=actual,
                    expected_digest=artifact.expected_digest,
                    digest_match=digest_match,
                    refs_reusable=refs_ok,
                    bad_references=sorted(bad_refs),
                    capacity_remaining=cap_remaining,
                    capacity_slots=cap_slots,
                    capacity_used=cap_used,
                    quota_remaining=quota_remaining,
                    lease_limit=node.lease_limit,
                    active_leases=active_leases,
                    eligible=eligible,
                    veto_reason=reason,
                )
            )
        eligible = [candidate for candidate in result if candidate.eligible]
        eligible.sort(key=lambda item: (-item.capacity_remaining, item.node_id))
        selected_id = eligible[0].node_id if eligible else None
        for rank, candidate in enumerate(eligible, start=1):
            candidate.rank = rank
            if candidate.node_id != selected_id:
                candidate.veto_reason = f"lower_rank:winner={selected_id}"
        return result

    def request_lease(
        self,
        artifact_key: str,
        request_id: str,
        *,
        stage_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RequestOutcome:
        if artifact_key not in self.spec.artifacts:
            raise ArbiterError("unknown artifact key", artifact_key=artifact_key, field="artifact_key")
        artifact = self.spec.artifacts[artifact_key]
        request_started = monotonic_ms()
        queue_position: int | None = None
        queue_reason: str | None = None
        selected: Candidate | None = None
        arbitration_index: int | None = None
        deadline = time.monotonic() + self.queue_timeout_ms / 1000.0
        with self.cond:
            while True:
                candidates = self._candidates_locked(artifact)
                available = sorted(
                    (candidate for candidate in candidates if candidate.eligible),
                    key=lambda candidate: (candidate.rank or 10**9, candidate.node_id),
                )
                key_owner = self._active_key_owner(artifact.key)
                if key_owner is not None:
                    raise ArbiterError(
                        "artifact key is already held by an active lease",
                        artifact_key=artifact.key,
                        field="artifact_key",
                        reason="key_already_leased",
                        node_id=key_owner,
                        candidates=[candidate.to_dict() for candidate in candidates],
                    )
                if artifact.key in self.reserved_keys:
                    raise ArbiterError(
                        "artifact key is already reserved by another request",
                        artifact_key=artifact.key,
                        field="artifact_key",
                        reason="key_already_reserved",
                        request_id=self.reserved_keys[artifact.key],
                        candidates=[candidate.to_dict() for candidate in candidates],
                    )
                if available:
                    selected = available[0]
                    self.reserved_keys[artifact.key] = request_id
                    self.pending_slots[selected.node_id] += 1
                    self.pending_leases[selected.node_id] += 1
                    arbitration_index = self.next_arbitration_index
                    self.next_arbitration_index += 1
                    break
                if all(not c.digest_match for c in candidates):
                    raise ArbiterError(
                        "no node can serve the artifact digest",
                        artifact_key=artifact.key,
                        field="digest",
                        reason="all_candidates_digest_mismatch",
                        candidates=[candidate.to_dict() for candidate in candidates],
                    )
                if all((not c.digest_match) or (not c.refs_reusable) for c in candidates):
                    raise ArbiterError(
                        "no node has a reusable digest-and-reference closure",
                        artifact_key=artifact.key,
                        field="reusable_keys",
                        reason="all_candidates_not_reusable",
                        candidates=[candidate.to_dict() for candidate in candidates],
                    )
                if queue_position is None:
                    self.waiting_count += 1
                    queue_position = self.waiting_count
                    queue_reason = ";".join(f"{c.node_id}:{c.veto_reason}" for c in candidates)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if queue_position is not None:
                        self.waiting_count = max(0, self.waiting_count - 1)
                    raise ArbiterError(
                        "lease request queue timed out",
                        artifact_key=artifact.key,
                        field="queue_timeout_ms",
                        reason="queue_timeout",
                        queue_position=queue_position,
                        queue_reason=queue_reason,
                        timeout_ms=self.queue_timeout_ms,
                        candidates=[candidate.to_dict() for candidate in candidates],
                    )
                self.cond.wait(timeout=min(remaining, 0.05))
            reservation_ended = monotonic_ms()
            if queue_position is not None:
                self.waiting_count = max(0, self.waiting_count - 1)
            candidates_payload = [candidate.to_dict() for candidate in candidates]
            selected_id = selected.node_id
            arbitration = arbitration_index
            assert selected_id is not None and arbitration is not None
            expected_digest = artifact.expected_digest
            queue_position_value = queue_position
            queue_reason_value = queue_reason

        # From here on the capacity/quota and key reservation are exclusively
        # owned by this request.  Node and global gates only limit real sockets.
        attempt_id: str | None = None
        node_response: dict[str, Any] | None = None
        execution_started = monotonic_ms()
        global_now = 0
        node_now = 0
        try:
            with self.global_gate:
                with self.node_gates[selected_id]:
                    global_now, node_now = self.meter.enter(selected_id)
                    execution_started = monotonic_ms()
                    attempt_id = self.ledger.begin_attempt(
                        request_id,
                        artifact.key,
                        selected_id,
                        owner_session=self.session_id,
                        details={"stage_id": stage_id, "arbitration_index": arbitration},
                    )
                    try:
                        node_response = self.clients[selected_id].lease(
                            node_id=selected_id,
                            request_id=request_id,
                            artifact_keys=[artifact.key],
                            lease_ms=self.lease_ms,
                            sequence=arbitration,
                        )
                    finally:
                        self.meter.leave(selected_id)
            if node_response is None:
                raise ReplyError("node produced no lease response", node_id=selected_id)
            actual = node_response["digests"].get(artifact.key)
            if actual != expected_digest:
                try:
                    self.clients[selected_id].release(
                        node_id=selected_id,
                        lease_id=node_response["lease_id"],
                        reason="digest_changed_after_selection",
                    )
                except Exception:
                    pass
                raise ArbiterError(
                    "node digest changed after inventory selection",
                    artifact_key=artifact.key,
                    node_id=selected_id,
                    field="digest",
                    expected=expected_digest,
                    actual=actual,
                )
            pause_ms = int(os.environ.get("GSBCACHE_PAUSE_AFTER_NODE_GRANT_MS", "0") or "0")
            if pause_ms > 0:
                time.sleep(pause_ms / 1000.0)
            if not self._wait_commit_turn(arbitration):
                raise ArbiterError(
                    "arbitration commit was canceled",
                    artifact_key=artifact.key,
                    field="arbitration_index",
                    arbitration_index=arbitration,
                )
            # If a test hook or a busy thread delayed the ordered commit, make
            # sure the node-side lease is still alive before recording it.
            remaining_ms = int(node_response["expires_at_ms"]) - now_ms()
            if remaining_ms < max(25, self.lease_ms // 2):
                renewed = self.clients[selected_id].renew(
                    node_id=selected_id,
                    lease_id=node_response["lease_id"],
                    extend_ms=self.lease_ms,
                )
                node_response["expires_at_ms"] = renewed["expires_at_ms"]
                remaining_ms = int(node_response["expires_at_ms"]) - now_ms()
            remaining_ms = max(1, remaining_ms)
            deadline_mono_ns = monotonic_ns() + remaining_ms * 1_000_000
            execution_ended = monotonic_ms()
            capacity_wait_ms = max(0, reservation_ended - request_started)
            gate_wait_ms = max(0, execution_started - reservation_ended)
            total_queue_ms = max(0, execution_started - request_started)
            effective_queue_reason = queue_reason_value
            if effective_queue_reason is None and total_queue_ms > 0:
                effective_queue_reason = "global_or_node_concurrency_gate"
            commit_details = {
                "stage_id": stage_id,
                "queue_position": queue_position_value,
                "queue_reason": effective_queue_reason,
                "queue_started_monotonic_ms": request_started,
                "queue_ended_monotonic_ms": execution_started,
                "queue_ms": total_queue_ms,
                "capacity_wait_ms": capacity_wait_ms,
                "gate_wait_ms": gate_wait_ms,
                "execution_started_monotonic_ms": execution_started,
                "execution_ended_monotonic_ms": execution_ended,
                "execution_ms": max(0, execution_ended - execution_started),
                "global_inflight_at_start": global_now,
                "node_inflight_at_start": node_now,
                "candidates": candidates_payload,
                "selected_node_reason": "digest_match>capacity_remaining_desc>node_id",
                **(extra or {}),
            }
            event = self.ledger.commit_attempt(
                attempt_id,
                node_id=selected_id,
                lease_id=node_response["lease_id"],
                request_id=request_id,
                artifact_keys=[artifact.key],
                digests={artifact.key: actual},
                expected_digest=expected_digest,
                expires_at_ms=int(node_response["expires_at_ms"]),
                deadline_mono_ns=deadline_mono_ns,
                owner_session=self.session_id,
                arbitration_index=arbitration,
                details=commit_details,
            )
            lease_record = {
                "lease_id": node_response["lease_id"],
                "node_id": selected_id,
                "artifact_keys": [artifact.key],
                "digests": {artifact.key: actual},
                "expires_at_ms": int(node_response["expires_at_ms"]),
                "deadline_mono_ns": deadline_mono_ns,
                "state": "active",
                "owner_session": self.session_id,
            }
            with self.cond:
                self.active_leases[lease_record["lease_id"]] = lease_record
                self.active_keys[artifact.key] = lease_record["lease_id"]
                self.pending_slots[selected_id] = max(0, self.pending_slots[selected_id] - 1)
                self.pending_leases[selected_id] = max(0, self.pending_leases[selected_id] - 1)
                self.reserved_keys.pop(artifact.key, None)
                self._advance_commit_locked(arbitration)
                self.cond.notify_all()
            return RequestOutcome(
                ok=True,
                request_id=request_id,
                artifact_key=artifact.key,
                node_id=selected_id,
                lease_id=node_response["lease_id"],
                expires_at_ms=int(node_response["expires_at_ms"]),
                digests={artifact.key: actual},
                reusable_keys=list(node_response.get("reusable_keys", [artifact.key])),
                candidates=candidates_payload,
                arbitration_index=arbitration,
                queue_position=queue_position_value,
                queue_reason=effective_queue_reason,
                queue_started_monotonic_ms=request_started,
                queue_ended_monotonic_ms=execution_started,
                queue_ms=total_queue_ms,
                capacity_wait_ms=capacity_wait_ms,
                gate_wait_ms=gate_wait_ms,
                execution_started_monotonic_ms=execution_started,
                execution_ended_monotonic_ms=execution_ended,
                execution_ms=max(0, execution_ended - execution_started),
                global_inflight_at_start=global_now,
                node_inflight_at_start=node_now,
                event_seq=int(event["seq"]),
            )
        except Exception:
            with self.cond:
                self.pending_slots[selected_id] = max(0, self.pending_slots[selected_id] - 1)
                self.pending_leases[selected_id] = max(0, self.pending_leases[selected_id] - 1)
                self.reserved_keys.pop(artifact.key, None)
                self._cancel_commit_locked(arbitration)
                self.cond.notify_all()
            if attempt_id is not None:
                try:
                    self.ledger.abort_attempt(attempt_id, "request_failed")
                except Exception:
                    pass
            if node_response is not None:
                try:
                    self.clients[selected_id].release(
                        node_id=selected_id,
                        lease_id=node_response["lease_id"],
                        reason="controller_commit_failed",
                    )
                except Exception:
                    pass
            raise

    def _wait_commit_turn(self, index: int) -> bool:
        deadline = time.monotonic() + max(5.0, self.request_timeout_s * 2)
        with self.cond:
            while index != self.next_commit_index:
                if index in self.canceled_commit_indices:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.cond.wait(timeout=min(remaining, 0.05))
            return True

    def _advance_commit_locked(self, index: int) -> None:
        if index == self.next_commit_index:
            self.next_commit_index += 1
            while self.next_commit_index in self.canceled_commit_indices:
                self.canceled_commit_indices.remove(self.next_commit_index)
                self.next_commit_index += 1

    def _cancel_commit_locked(self, index: int) -> None:
        self.canceled_commit_indices.add(index)
        if index == self.next_commit_index:
            self._advance_commit_locked(index)

    def _reap_loop(self) -> None:
        while not self.stop_event.is_set():
            now_ns = monotonic_ns()
            expired: list[dict[str, Any]] = []
            next_deadline: int | None = None
            with self.cond:
                for lease_id, lease in list(self.active_leases.items()):
                    deadline = int(lease["deadline_mono_ns"])
                    if deadline <= now_ns:
                        expired.append(lease)
                    else:
                        next_deadline = deadline if next_deadline is None else min(next_deadline, deadline)
            for lease in expired:
                result = self.ledger.expire_lease(lease["lease_id"])
                if result.get("closed") or result.get("already_closed"):
                    with self.cond:
                        self.active_leases.pop(lease["lease_id"], None)
                        for key in lease["artifact_keys"]:
                            if self.active_keys.get(key) == lease["lease_id"]:
                                self.active_keys.pop(key, None)
                        self.cond.notify_all()
                    if result.get("closed"):
                        try:
                            self.clients[lease["node_id"]].release(
                                node_id=lease["node_id"],
                                lease_id=lease["lease_id"],
                                reason="monotonic_expiry",
                            )
                        except Exception:
                            pass
            if expired:
                continue
            if next_deadline is None:
                self.stop_event.wait(0.01)
                continue
            wait_s = max(0.0, min((next_deadline - monotonic_ns()) / 1_000_000_000, 0.05))
            self.stop_event.wait(wait_s)

    def recycle_quota(self, lease_id: str, reason: str = "quota_reclaim") -> dict[str, Any]:
        """Race-safe quota reclamation.  Only one close transition can win."""
        result = self.ledger.recycle_quota(lease_id, reason)
        if result.get("closed"):
            lease = self.active_leases.get(lease_id)
            with self.cond:
                self.active_leases.pop(lease_id, None)
                if lease:
                    for key in lease["artifact_keys"]:
                        if self.active_keys.get(key) == lease_id:
                            self.active_keys.pop(key, None)
                self.cond.notify_all()
            if lease:
                try:
                    self.clients[lease["node_id"]].release(
                        node_id=lease["node_id"],
                        lease_id=lease_id,
                        reason="quota_recycle",
                    )
                except Exception:
                    pass
        return result

    def renew(self, lease_id: str, extend_ms: int | None = None) -> dict[str, Any]:
        extend = int(extend_ms or self.lease_ms)
        lease = self.ledger.get_lease(lease_id)
        if lease is None or lease["state"] != "active":
            raise ArbiterError("lease is not active", lease_id=lease_id, field="state")
        node_response = self.clients[lease["node_id"]].renew(
            node_id=lease["node_id"], lease_id=lease_id, extend_ms=extend
        )
        deadline_mono_ns = monotonic_ns() + extend * 1_000_000
        event = self.ledger.renew_lease(
            lease_id,
            extend_ms=extend,
            expires_at_ms=int(node_response["expires_at_ms"]),
            deadline_mono_ns=deadline_mono_ns,
            details={"node_expires_at_ms": int(node_response["expires_at_ms"])},
        )
        with self.cond:
            if lease_id in self.active_leases:
                self.active_leases[lease_id]["expires_at_ms"] = int(node_response["expires_at_ms"])
                self.active_leases[lease_id]["deadline_mono_ns"] = deadline_mono_ns
            self.cond.notify_all()
        return {"event": event, "node_expires_at_ms": node_response["expires_at_ms"]}

    def reconcile_active_from_ledger(self) -> None:
        active_ids = {lease["lease_id"] for lease in self.ledger.get_active_leases()}
        with self.cond:
            changed = False
            for lease_id, lease in list(self.active_leases.items()):
                if lease_id not in active_ids:
                    self.active_leases.pop(lease_id, None)
                    for key in lease["artifact_keys"]:
                        if self.active_keys.get(key) == lease_id:
                            self.active_keys.pop(key, None)
                    changed = True
            if changed:
                self.cond.notify_all()

    def metrics(self) -> dict[str, Any]:
        self.reconcile_active_from_ledger()
        with self.cond:
            return {
                **self.meter.snapshot(),
                "configured": {
                    "global_concurrency": self.spec.pipeline.global_concurrency,
                    "node_lease_limits": {node.node_id: node.lease_limit for node in self.spec.nodes},
                    "node_capacity_slots": {node.node_id: node.capacity_slots for node in self.spec.nodes},
                },
                "active_leases": len(self.active_leases),
                "reserved_keys": sorted(self.reserved_keys),
                "pending_slots_by_node": dict(sorted(self.pending_slots.items())),
                "pending_leases_by_node": dict(sorted(self.pending_leases.items())),
                "queue_waiters": self.waiting_count,
            }
