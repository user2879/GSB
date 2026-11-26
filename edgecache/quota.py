"""Atomic in-memory quota reservations for the three dispatch constraints."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .errors import EdgeError


@dataclass
class Reservation:
    reservation_id: str
    node_id: str
    units: int
    state: str = "active"


class QuotaPool:
    def __init__(self, node_quotas: dict[str, int], global_quota: int, block_weight: int):
        self._condition = threading.Condition()
        self._node_quotas = dict(node_quotas)
        self._global_quota = int(global_quota)
        self._block_weight = int(block_weight)
        self._sequence = 0
        self._reservations: dict[str, Reservation] = {}
        self._release_log: dict[str, dict[str, object]] = {}
        self._consumed_total = 0
        self._expired_total = 0

    def _next_id(self, node_id: str) -> str:
        self._sequence += 1
        return f"quota-{node_id}-{self._sequence}"

    def reserve(self, node_id: str, evict_blocks: int, timeout_s: float = 10.0) -> Reservation:
        units = evict_blocks * self._block_weight
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._node_quotas.get(node_id, -1) >= units and self._global_quota >= units,
                timeout=timeout_s,
            ):
                raise EdgeError(
                    "E_QUOTA",
                    f"quota reservation timed out for node {node_id}",
                    details={
                        "node_id": node_id,
                        "requested_units": units,
                        "node_available": self._node_quotas.get(node_id, 0),
                        "global_available": self._global_quota,
                    },
                    exit_code=10,
                )
            reservation_id = self._next_id(node_id)
            reservation = Reservation(
                reservation_id=reservation_id,
                node_id=node_id,
                units=units,
            )
            self._reservations[reservation_id] = reservation
            self._node_quotas[node_id] -= units
            self._global_quota -= units
            return reservation

    def _release_locked(self, reservation_id: str, reason: str) -> bool:
        reservation = self._reservations.get(reservation_id)
        if reservation is None or reservation.state != "active":
            return False
        reservation.state = reason
        if reason in {"expired", "cancelled"}:
            self._node_quotas[reservation.node_id] += reservation.units
            self._global_quota += reservation.units
        self._release_log[reservation_id] = {
            "reason": reason,
            "units": reservation.units,
            "node_id": reservation.node_id,
        }
        if reason == "expired":
            self._expired_total += reservation.units
        elif reason == "consumed":
            self._consumed_total += reservation.units
        self._condition.notify_all()
        return True

    def finalize(self, reservation: Reservation) -> bool:
        with self._condition:
            return self._release_locked(reservation.reservation_id, "consumed")

    def expire(self, reservation: Reservation) -> bool:
        with self._condition:
            return self._release_locked(reservation.reservation_id, "expired")

    def cancel(self, reservation: Reservation) -> bool:
        with self._condition:
            return self._release_locked(reservation.reservation_id, "cancelled")

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "node_available": dict(sorted(self._node_quotas.items())),
                "global_available": self._global_quota,
                "reservations": {
                    key: {"node_id": value.node_id, "units": value.units, "state": value.state}
                    for key, value in sorted(self._reservations.items())
                },
                "release_log": dict(sorted(self._release_log.items())),
                "consumed_total": self._consumed_total,
                "expired_total": self._expired_total,
            }
