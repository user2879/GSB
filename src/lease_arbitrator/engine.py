from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .cluster import NodeCluster
from .domain import DomainState
from .errors import GenerationChainError, LeaseArbitratorError
from .invalidation import propagate_invalidation
from .ledger import Ledger
from .model import monotonic_ms


class ArbitrationEngine:
    def __init__(
        self,
        spec: dict[str, Any],
        events: list[dict[str, Any]],
        cluster: NodeCluster,
        ledger: Ledger | None = None,
        *,
        clock=monotonic_ms,
        max_workers: int | None = None,
    ) -> None:
        self.spec = spec
        self.events = events
        self.cluster = cluster
        self.ledger = ledger
        self.clock = clock
        self.domain = DomainState(spec)
        self.cond = threading.Condition()
        self.lease_locks: dict[str, threading.RLock] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers or min(32, max(4, len(spec["leases"]) + 2)),
            thread_name_prefix="lease-scheduler",
        )
        self.futures: dict[int, Future[dict[str, Any]]] = {}
        self.results: list[dict[str, Any]] = []
        self.cached_winner: str | None = None
        self.winner_dirty = True
        self.closing = False
        self.admissions_frozen = False
        self.reaper_stop = threading.Event()
        self.reaper_thread: threading.Thread | None = None

    def lease_lock(self, lease_id: str) -> threading.RLock:
        with self.cond:
            if lease_id not in self.lease_locks:
                self.lease_locks[lease_id] = threading.RLock()
            return self.lease_locks[lease_id]

    def start_reaper(self, interval_seconds: float = 0.01) -> None:
        if self.reaper_thread is not None:
            return

        def loop() -> None:
            while not self.reaper_stop.wait(interval_seconds):
                try:
                    self.reap_once()
                except Exception as exc:  # reaper must stay alive
                    with self.cond:
                        self.results.append({"ok": False, "stage": "reaper", "error": str(exc)})

        self.reaper_thread = threading.Thread(target=loop, name="lease-reaper", daemon=True)
        self.reaper_thread.start()

    def reap_once(self) -> list[dict[str, Any]]:
        now = self.clock()
        with self.cond:
            due = [lease.lease_id for lease in self.domain.due_leases(now)]
        results = []
        for lease_id in due:
            result = self.terminal_cas(
                lease_id,
                action="release",
                reason="hold_timeout_monotonic_deadline",
                event_seq=None,
                ledger_kind="expire",
            )
            results.append(result)
        return results

    def _append(self, **kwargs: Any) -> dict[str, Any] | None:
        if self.ledger is None:
            return None
        kwargs.setdefault("ts_monotonic_ms", self.clock())
        return self.ledger.append(**kwargs)

    def register_apply(self, event: dict[str, Any]) -> None:
        now = self.clock()
        with self.cond:
            self.domain.register_apply(event["lease_id"], now, event["seq"])
            self.winner_dirty = True
            self.cond.notify_all()

    def launch_apply(self, event: dict[str, Any]) -> None:
        future = self.executor.submit(self.apply_worker, event["lease_id"], event["seq"])
        self.futures[event["seq"]] = future

    def apply_worker(self, lease_id: str, event_seq: int) -> dict[str, Any]:
        reservation: dict[str, Any] | None = None
        with self.cond:
            while True:
                lease = self.domain.leases[lease_id]
                if lease.state == "revoked" or self.closing:
                    return {"ok": False, "lease_id": lease_id, "reason": "canceled_or_closing"}
                if self.admissions_frozen:
                    self.cond.wait(timeout=0.02)
                    continue
                if self.winner_dirty:
                    self.cached_winner, _ = self.domain.choose_winner()
                    self.winner_dirty = False
                if self.cached_winner == lease_id:
                    reservation = self.domain.reserve(lease_id, self.clock(), event_seq)
                    if reservation is not None:
                        self.cached_winner = None
                        self.winner_dirty = True
                        self.cond.notify_all()
                        break
                    self.winner_dirty = True
                self.cond.wait(timeout=0.02)

        lease = self.domain.leases[lease_id]
        try:
            response = self.cluster.call(
                lease.node_id,
                {
                    "type": "apply",
                    "lease_id": lease_id,
                    "units": lease.units,
                    "generation": 1,
                    "event_seq": event_seq,
                },
                timeout=self.spec["node_timeout_seconds"],
            )
        except LeaseArbitratorError as exc:
            response = {"ok": False, "error": exc.code, "message": exc.message}

        now = self.clock()
        with self.cond:
            if not response.get("ok"):
                self.domain.rollback_reservation(lease_id, now, str(response.get("error", "E_NODE")))
                self.winner_dirty = True
                self.cond.notify_all()
                self._append(
                    kind="grant_failed",
                    event_seq=event_seq,
                    ok=False,
                    error_code=str(response.get("error")),
                    lease_id=lease_id,
                    node_id=lease.node_id,
                    state_before="queued",
                    state_after="failed",
                    payload={"response": response},
                    global_used=reservation["quota"]["global_used"],
                    global_remaining=reservation["quota"]["global_remaining"],
                    node_used=reservation["quota"]["node_used"],
                    node_remaining=reservation["quota"]["node_remaining"],
                    wait_start_ms=reservation["wait_start_ms"],
                    wait_end_ms=reservation["wait_end_ms"],
                )
                return {"ok": False, "lease_id": lease_id, "response": response}

            transition = self.domain.confirm_grant(lease_id, response, now, event_seq)
            snapshot = reservation["quota"]
            entry = self._append(
                kind="grant",
                event_seq=event_seq,
                lease_id=lease_id,
                node_id=lease.node_id,
                state_before=transition.from_state,
                state_after=transition.to_state,
                generation_used=transition.generation_used,
                generation_before=transition.generation_before,
                generation_after=transition.generation_after,
                global_used=snapshot["global_used"],
                global_remaining=snapshot["global_remaining"],
                node_used=snapshot["node_used"],
                node_remaining=snapshot["node_remaining"],
                wait_start_ms=reservation["wait_start_ms"],
                wait_end_ms=reservation["wait_end_ms"],
                payload={
                    "response": response,
                    "reservation": {
                        k: v for k, v in reservation.items() if k != "quota"
                    },
                    "quota_snapshot": snapshot,
                },
            )
            self.winner_dirty = True
            self.cond.notify_all()
            return {
                "ok": True,
                "lease_id": lease_id,
                "transition": transition.to_dict(),
                "ledger_seq": entry["seq"] if entry else None,
            }

    def wait_active(self, lease_id: str, timeout_seconds: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self.cond:
            while not self.domain.leases[lease_id].active:
                if self.domain.leases[lease_id].state in {"released", "expired", "revoked", "failed"}:
                    return False
                if time.monotonic() >= deadline:
                    return False
                self.cond.wait(timeout=0.02)
            return True

    def generation_cas(
        self,
        event: dict[str, Any],
        action: str,
        *,
        reason: str | None = None,
        ledger_kind: str | None = None,
    ) -> dict[str, Any]:
        lease_id = event["lease_id"]
        event_seq = event["seq"]
        lock = self.lease_lock(lease_id)
        with lock:
            if not self.wait_active(lease_id, timeout_seconds=self.spec["node_timeout_seconds"] + 1.0):
                result = {
                    "ok": False,
                    "expected_error": event.get("expect_error"),
                    "error_code": "E_STATE",
                    "lease_id": lease_id,
                }
                self.results.append(result)
                return result
            with self.cond:
                lease = self.domain.leases[lease_id]
                request_generation, classification = self.domain.classify_generation(
                    lease, event.get("generation")
                )
                node_id = lease.node_id

            if classification == "forward_jump":
                raise AssertionError("forward_jump should raise before this point")
            response = self.cluster.call(
                node_id,
                {
                    "type": action,
                    "lease_id": lease_id,
                    "generation": request_generation,
                    "event_seq": event_seq,
                },
                timeout=self.spec["node_timeout_seconds"],
            )
            now = self.clock()
            with self.cond:
                transition, quota = self.domain.apply_cas_response(
                    lease_id,
                    action,
                    reason or f"user_{action}",
                    request_generation,
                    response,
                    now,
                    event_seq,
                )
                lease = self.domain.leases[lease_id]
                kind = ledger_kind or action
                if transition is None:
                    entry = self._append(
                        kind=f"{kind}_rejected",
                        event_seq=event_seq,
                        ok=False,
                        error_code=response.get("error"),
                        lease_id=lease_id,
                        node_id=node_id,
                        state_before=lease.state,
                        state_after=lease.state,
                        generation_used=request_generation,
                        generation_before=lease.generation_current,
                        generation_after=lease.generation_current,
                        global_used=quota["global_used"],
                        global_remaining=quota["global_remaining"],
                        node_used=quota["node_used"],
                        node_remaining=quota["node_remaining"],
                        payload={"event": event, "response": response},
                    )
                    self.winner_dirty = True
                    self.cond.notify_all()
                    result = {
                        "ok": False,
                        "lease_id": lease_id,
                        "error_code": response.get("error"),
                        "response": response,
                        "ledger_seq": entry["seq"] if entry else None,
                    }
                else:
                    entry = self._append(
                        kind=kind,
                        event_seq=event_seq,
                        lease_id=lease_id,
                        node_id=node_id,
                        state_before=transition.from_state,
                        state_after=transition.to_state,
                        generation_used=transition.generation_used,
                        generation_before=transition.generation_before,
                        generation_after=transition.generation_after,
                        global_used=quota["global_used"],
                        global_remaining=quota["global_remaining"],
                        node_used=quota["node_used"],
                        node_remaining=quota["node_remaining"],
                        payload={"event": event, "response": response},
                    )
                    self.winner_dirty = True
                    self.cond.notify_all()
                    result = {
                        "ok": True,
                        "lease_id": lease_id,
                        "transition": transition.to_dict(),
                        "ledger_seq": entry["seq"] if entry else None,
                    }
                expected = event.get("expect_error")
                result["expectation_met"] = (
                    response.get("error") == expected if expected else response.get("ok") is True
                )
                self.results.append(result)
                return result

    def terminal_cas(
        self,
        lease_id: str,
        *,
        action: str = "release",
        event_seq: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        event = {
            "seq": event_seq,
            "lease_id": lease_id,
            "generation": self.domain.leases[lease_id].generation_current,
        }
        return self.generation_cas(event, action, **kwargs)

    def handle_generation_error(self, event: dict[str, Any], exc: GenerationChainError) -> dict[str, Any]:
        lease_id = event["lease_id"]
        with self.cond:
            lease = self.domain.leases[lease_id]
            entry = self._append(
                kind=f"{event['type']}_rejected",
                event_seq=event["seq"],
                ok=False,
                error_code="E_GEN",
                lease_id=lease_id,
                node_id=lease.node_id,
                state_before=lease.state,
                state_after=lease.state,
                generation_used=event.get("generation"),
                generation_before=lease.generation_current,
                generation_after=lease.generation_current,
                payload={"event": event, "details": exc.details},
            )
            result = {
                "ok": False,
                "lease_id": lease_id,
                "error_code": "E_GEN",
                "details": exc.details,
                "ledger_seq": entry["seq"] if entry else None,
                "expectation_met": event.get("expect_error") == "E_GEN",
            }
            self.results.append(result)
            return result

    def upgrade_quota(self, event: dict[str, Any]) -> dict[str, Any]:
        version = int(event["quota_version"])
        now = self.clock()
        with self.cond:
            plan = self.domain.upgrade_policy(version, now)
            self.admissions_frozen = True
            self.winner_dirty = True
            self.cond.notify_all()
        revoked: list[dict[str, Any]] = []
        for lease_id in plan["active_to_revoke"]:
            result = self.terminal_cas(
                lease_id,
                action="revoke",
                reason=f"quota_upgrade_whole_replacement:{plan['old_version']}->{version}",
                event_seq=event["seq"],
                ledger_kind="revoke",
            )
            revoked.append(result)

        canceled: list[str] = []
        now = self.clock()
        with self.cond:
            for lease_id in plan["pending_to_cancel"]:
                lease = self.domain.leases[lease_id]
                if lease_id in self.domain.pending:
                    self.domain.pending.remove(lease_id)
                if lease.state == "queued":
                    transition = self.domain.transition(
                        lease,
                        "revoked",
                        f"quota_upgrade_cancels_pending:{plan['old_version']}->{version}",
                        now,
                        event_seq=event["seq"],
                        generation_before=0,
                        generation_after=0,
                    )
                    if lease.charged_units or lease.active_slot:
                        self.domain._release_capacity(lease)
                    quota_snapshot = self.domain.quota_snapshot(lease)
                    self._append(
                        kind="revoke",
                        event_seq=event["seq"],
                        lease_id=lease_id,
                        node_id=lease.node_id,
                        state_before=transition.from_state,
                        state_after=transition.to_state,
                        generation_before=0,
                        generation_after=0,
                        global_used=quota_snapshot["global_used"],
                        global_remaining=quota_snapshot["global_remaining"],
                        node_used=quota_snapshot["node_used"],
                        node_remaining=quota_snapshot["node_remaining"],
                        payload={"pending_cancel": True, "event": event},
                    )
                    canceled.append(lease_id)
            self.domain.commit_policy_replacement(plan["policy"])
            graph = propagate_invalidation(
                self.spec,
                [f"version:{version}"],
                events=self.events,
            )
            entry = self._append(
                kind="quota_upgrade",
                event_seq=event["seq"],
                state_before=f"version:{plan['old_version']}",
                state_after=f"version:{version}",
                global_used=self.domain.global_used,
                global_remaining=self.domain.global_budget - self.domain.global_used,
                payload={"plan": plan, "invalidation": graph, "event": event},
            )
            self.winner_dirty = True
            self.admissions_frozen = False
            self.cond.notify_all()
        result = {
            "ok": all(item.get("ok", False) for item in revoked) if revoked else True,
            "type": "upgrade_quota",
            "version": version,
            "revoked": revoked,
            "canceled_pending": canceled,
            "invalidation": graph,
            "ledger_seq": entry["seq"] if entry else None,
        }
        self.results.append(result)
        return result

    def run_event_stream(self) -> None:
        origin = time.monotonic()

        def wait_arrival(arrival_ms: int) -> None:
            target = origin + arrival_ms / 1000.0
            while True:
                remaining = target - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(0.005, remaining))

        index = 0
        while index < len(self.events):
            arrival = self.events[index]["arrival_ms"]
            group: list[dict[str, Any]] = []
            while index < len(self.events) and self.events[index]["arrival_ms"] == arrival:
                group.append(self.events[index])
                index += 1
            wait_arrival(arrival)

            apply_events = [event for event in group if event["type"] == "apply"]
            non_apply = [event for event in group if event["type"] != "apply"]
            for event in apply_events:
                self.register_apply(event)
            for event in apply_events:
                self.launch_apply(event)
            for event in non_apply:
                self.dispatch_non_apply(event)

    def dispatch_non_apply(self, event: dict[str, Any]) -> dict[str, Any] | None:
        try:
            if event["type"] == "renew":
                return self.generation_cas(event, "renew")
            if event["type"] == "release":
                return self.generation_cas(event, "release")
            if event["type"] == "upgrade_quota":
                return self.upgrade_quota(event)
        except GenerationChainError as exc:
            return self.handle_generation_error(event, exc)
        return None

    def wait_settled(self, timeout_seconds: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self.cond:
                active = [lease for lease in self.domain.leases.values() if lease.active]
                if not active:
                    return True
            time.sleep(0.01)
        return False

    def shutdown(self) -> None:
        self.reaper_stop.set()
        if self.reaper_thread is not None:
            self.reaper_thread.join(timeout=1.0)
        with self.cond:
            self.closing = True
            self.cond.notify_all()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def report(self) -> dict[str, Any]:
        with self.cond:
            return {
                "ok": not any(
                    isinstance(item, dict) and item.get("ok") is False and not item.get("expectation_met")
                    for item in self.results
                )
                and not self.domain.invariant_violations,
                "current_view": self.domain.current_view(),
                "stale_view": self.domain.stale_view(),
                "results": self.results,
                "invariant_violations": self.domain.invariant_violations,
            }

    def run(self, settle_seconds: float = 0.6) -> dict[str, Any]:
        self.start_reaper()
        self.run_event_stream()
        self.wait_settled(settle_seconds)
        self.shutdown()
        return self.report()
