from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .protocol import monotonic_now_ns
from .specs import SchedulerConfig


NodeInvoke = Callable[[str, str, list[dict[str, Any]]], dict[str, Any]]


@dataclass
class DispatchOutcome:
    request_id: str
    ok: bool
    responses: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None


class DispatchScheduler:
    """Bounded pool with global request slots and per-node call slots."""

    def __init__(
        self,
        config: SchedulerConfig,
        node_ids: list[str],
        invoke_node: NodeInvoke,
    ) -> None:
        self.config = config
        self.node_ids = list(node_ids)
        self.invoke_node = invoke_node
        self._pool = ThreadPoolExecutor(
            max_workers=config.pool_workers,
            thread_name_prefix="hopledger-dispatch",
        )
        self._global_slots = threading.BoundedSemaphore(config.max_in_flight)
        self._node_slots = {
            node_id: threading.BoundedSemaphore(config.hop_quota)
            for node_id in self.node_ids
        }
        self._lock = threading.Lock()
        self._global_in_flight = 0
        self._node_in_flight = {node_id: 0 for node_id in self.node_ids}
        self.max_observed_global = 0
        self.max_observed_hop = {node_id: 0 for node_id in self.node_ids}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)

    def run_requests(self, request_ids: list[str]) -> list[DispatchOutcome]:
        futures = [
            self._pool.submit(self._run_request, request_id)
            for request_id in request_ids
        ]
        return [future.result() for future in futures]

    def _wait_event(
        self,
        *,
        scope: str,
        request_id: str,
        node_id: str | None,
        start_ns: int,
        end_ns: int,
    ) -> dict[str, Any]:
        return {
            "type": "queue_wait",
            "scope": scope,
            "request_id": request_id,
            "node_id": node_id,
            "wait_start_monotonic_ms": start_ns // 1_000_000,
            "wait_end_monotonic_ms": end_ns // 1_000_000,
            "wait_ms": max(0, (end_ns - start_ns) // 1_000_000),
        }

    def _run_request(self, request_id: str) -> DispatchOutcome:
        outcome = DispatchOutcome(request_id=request_id, ok=False)
        global_wait_start = monotonic_now_ns()
        self._global_slots.acquire()
        global_wait_end = monotonic_now_ns()
        outcome.events.append(self._wait_event(
            scope="global", request_id=request_id, node_id=None,
            start_ns=global_wait_start, end_ns=global_wait_end,
        ))
        with self._lock:
            self._global_in_flight += 1
            self.max_observed_global = max(
                self.max_observed_global, self._global_in_flight
            )
        try:
            prior_receipts: list[dict[str, Any]] = []
            for node_id in self.node_ids:
                slot = self._node_slots[node_id]
                wait_start = monotonic_now_ns()
                slot.acquire()
                wait_end = monotonic_now_ns()
                outcome.events.append(self._wait_event(
                    scope="hop", request_id=request_id, node_id=node_id,
                    start_ns=wait_start, end_ns=wait_end,
                ))
                with self._lock:
                    self._node_in_flight[node_id] += 1
                    self.max_observed_hop[node_id] = max(
                        self.max_observed_hop[node_id],
                        self._node_in_flight[node_id],
                    )
                try:
                    response = self.invoke_node(request_id, node_id, prior_receipts)
                    outcome.responses.append(response)
                    receipt = response.get("receipt")
                    if isinstance(receipt, dict):
                        outcome.receipts.append(receipt)
                        prior_receipts.append(receipt)
                    if not response.get("ok", False):
                        outcome.error = response.get("error") or {
                            "code": "E_FORWARD",
                            "message": "node reported failure",
                        }
                        return outcome
                except Exception as exc:
                    outcome.error = getattr(exc, "to_dict", lambda: {
                        "code": "E_DISPATCH",
                        "message": str(exc),
                    })()
                    return outcome
                finally:
                    with self._lock:
                        self._node_in_flight[node_id] -= 1
                    slot.release()
            outcome.ok = True
            return outcome
        finally:
            with self._lock:
                self._global_in_flight -= 1
            self._global_slots.release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_observed_global": self.max_observed_global,
                "max_observed_hop": dict(self.max_observed_hop),
                "config": {
                    "max_in_flight": self.config.max_in_flight,
                    "hop_quota": self.config.hop_quota,
                    "pool_workers": self.config.pool_workers,
                },
            }
