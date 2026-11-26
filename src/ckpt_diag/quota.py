"""Requirement 5: unique ownership arbitration for a single pool of quota units.

One pool of ``global_budget`` resource units is arbitrated between:

* active leases held by a shard/batch;
* units held back by lease-expiry reclamation;
* units held back by checkpoint-rollback parameter-version void reclamation.

The arbiter is a *pure deterministic state machine* over the ordered event
stream. Replaying any prefix of persisted events therefore reproduces the exact
same ownership sequence and free-capacity numbers, regardless of real timing
or event batch boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import LeaseError, SpecError

OWNER_ACTIVE = "active"
OWNER_EXPIRED = "expired_reclaim"
OWNER_VOID = "void_reclaim"
OWNER_FREE = "free"


@dataclass(frozen=True)
class OwnershipRecord:
    unit_index: int
    owner_kind: str          # active | expired_reclaim | void_reclaim | free
    shard_id: str | None
    lease_id: str | None
    lease_version: int | None
    param_version: int | None
    event_seq: int
    basis: str


@dataclass
class Decision:
    event_seq: int
    kind: str
    basis: str
    param_version: int | None
    lease_version: int | None
    units: tuple[int, ...]
    before_owner: str
    after_owner: str
    shard_id: str | None
    lease_id: str | None

    def to_dict(self) -> dict:
        return {
            "event_seq": self.event_seq,
            "kind": self.kind,
            "basis": self.basis,
            "param_version": self.param_version,
            "lease_version": self.lease_version,
            "units": list(self.units),
            "before_owner": self.before_owner,
            "after_owner": self.after_owner,
            "shard_id": self.shard_id,
            "lease_id": self.lease_id,
        }


@dataclass
class ArbiterState:
    budget: int
    units: list[OwnershipRecord] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    shard_caps: dict[str, int] = field(default_factory=dict)
    # lease_id -> set(unit_index) for currently active leases
    active_lease_units: dict[str, set[int]] = field(default_factory=dict)
    # param_version -> set(unit_index) held in void reclaim
    void_units: dict[int, set[int]] = field(default_factory=dict)
    expired_units: set[int] = field(default_factory=set)

    def free_capacity(self) -> int:
        return sum(1 for record in self.units if record.owner_kind == OWNER_FREE)

    def active_for_shard(self, shard_id: str) -> int:
        return sum(1 for record in self.units
                   if record.owner_kind == OWNER_ACTIVE and record.shard_id == shard_id)


class QuotaArbiter:
    """Deterministic, replayable unique-ownership arbiter."""

    def __init__(self, budget: int, shard_lease_caps: dict[str, int]) -> None:
        if budget <= 0:
            raise SpecError("global budget must be positive", "global_budget", "global_budget")
        self.state = ArbiterState(
            budget=budget,
            units=[OwnershipRecord(index, OWNER_FREE, None, None, None, None, -1,
                                   "initial_free_pool")
                   for index in range(budget)],
            shard_caps=dict(shard_lease_caps),
        )

    def _free_units(self, count: int) -> list[int]:
        free = [record.unit_index for record in self.state.units
                if record.owner_kind == OWNER_FREE]
        if len(free) < count:
            raise LeaseError(
                f"insufficient free capacity: need {count}, have {len(free)}",
                "quota", "global_budget")
        return free[:count]

    def _reassign(self, unit_index: int, owner_kind: str, *, event_seq: int,
                  basis: str, shard_id: str | None, lease_id: str | None,
                  lease_version: int | None, param_version: int | None) -> None:
        before = self.state.units[unit_index].owner_kind
        self.state.units[unit_index] = OwnershipRecord(
            unit_index, owner_kind, shard_id, lease_id, lease_version,
            param_version, event_seq, basis)
        self.state.decisions.append(Decision(
            event_seq=event_seq, kind=owner_kind, basis=basis,
            param_version=param_version, lease_version=lease_version,
            units=(unit_index,), before_owner=before, after_owner=owner_kind,
            shard_id=shard_id, lease_id=lease_id))

    def grant(self, *, event_seq: int, shard_id: str, lease_id: str,
              lease_version: int, param_version: int, cost_units: int) -> tuple[int, ...]:
        cap = self.state.shard_caps.get(shard_id)
        if cap is None:
            raise SpecError(f"unknown shard {shard_id}", "quota", "shard_id")
        if self.state.active_for_shard(shard_id) + cost_units > cap:
            raise LeaseError(
                f"shard {shard_id} lease cap {cap} exceeded by {cost_units}",
                f"event:{event_seq}", "lease_cap")
        units = self._free_units(cost_units)
        for unit_index in units:
            # Unique ownership: a unit must currently be free to become active.
            self._reassign(unit_index, OWNER_ACTIVE, event_seq=event_seq,
                           basis="grant_from_free_pool", shard_id=shard_id,
                           lease_id=lease_id, lease_version=lease_version,
                           param_version=param_version)
        self.state.active_lease_units[lease_id] = set(units)
        return tuple(units)

    def expire_lease(self, *, event_seq: int, lease_id: str,
                     lease_version: int, param_version: int) -> tuple[int, ...]:
        units = self.state.active_lease_units.get(lease_id)
        if not units:
            raise LeaseError(f"lease {lease_id} has no active units to expire",
                             f"event:{event_seq}", "lease_id")
        for unit_index in sorted(units):
            record = self.state.units[unit_index]
            if record.owner_kind != OWNER_ACTIVE:
                # Hard invariant: never double-count a unit.
                raise LeaseError(
                    f"unit {unit_index} cannot be expired; already owned by "
                    f"{record.owner_kind}", f"event:{event_seq}", "ownership")
            self._reassign(unit_index, OWNER_EXPIRED, event_seq=event_seq,
                           basis="lease_ttl_expiry_reclaim", shard_id=record.shard_id,
                           lease_id=lease_id, lease_version=lease_version,
                           param_version=param_version)
            self.state.expired_units.add(unit_index)
        del self.state.active_lease_units[lease_id]
        return tuple(sorted(units))

    def void_param_version(self, *, event_seq: int, shard_id: str,
                           param_version: int, lease_id: str | None,
                           lease_version: int | None,
                           unit_indices: tuple[int, ...]) -> tuple[int, ...]:
        """Move units into void reclaim after rollback, never duplicating them."""
        moved: list[int] = []
        for unit_index in sorted(unit_indices):
            record = self.state.units[unit_index]
            if record.owner_kind == OWNER_VOID:
                continue  # idempotent replay
            if record.owner_kind == OWNER_EXPIRED:
                raise LeaseError(
                    f"unit {unit_index} already counted in expiry reclaim; cannot "
                    "also count as void reclaim (unique ownership)",
                    f"event:{event_seq}", "ownership")
            if record.owner_kind != OWNER_ACTIVE:
                raise LeaseError(
                    f"unit {unit_index} cannot be voided from state {record.owner_kind}",
                    f"event:{event_seq}", "ownership")
            self._reassign(unit_index, OWNER_VOID, event_seq=event_seq,
                           basis="rollback_param_version_void_reclaim",
                           shard_id=shard_id, lease_id=lease_id,
                           lease_version=lease_version, param_version=param_version)
            self.state.void_units.setdefault(param_version, set()).add(unit_index)
            moved.append(unit_index)
        return tuple(moved)

    def reclaim_to_free(self, *, event_seq: int,
                        unit_indices: tuple[int, ...], basis: str) -> tuple[int, ...]:
        """Return reclaimed (expired or void) units to the free pool."""
        freed: list[int] = []
        for unit_index in sorted(unit_indices):
            record = self.state.units[unit_index]
            if record.owner_kind not in (OWNER_EXPIRED, OWNER_VOID):
                raise LeaseError(
                    f"unit {unit_index} cannot be freed from state {record.owner_kind}",
                    f"event:{event_seq}", "ownership")
            self.state.expired_units.discard(unit_index)
            for units in self.state.void_units.values():
                units.discard(unit_index)
            self._reassign(unit_index, OWNER_FREE, event_seq=event_seq, basis=basis,
                           shard_id=None, lease_id=None, lease_version=None,
                           param_version=None)
            freed.append(unit_index)
        return tuple(freed)

    def release_lease(self, *, event_seq: int, lease_id: str,
                      lease_version: int, param_version: int) -> tuple[int, ...]:
        units = self.state.active_lease_units.pop(lease_id, set())
        freed: list[int] = []
        for unit_index in sorted(units):
            record = self.state.units[unit_index]
            if record.owner_kind != OWNER_ACTIVE:
                raise LeaseError(
                    f"unit {unit_index} release conflict; state={record.owner_kind}",
                    f"event:{event_seq}", "ownership")
            self._reassign(unit_index, OWNER_FREE, event_seq=event_seq,
                           basis="lease_release_return", shard_id=None,
                           lease_id=lease_id, lease_version=lease_version,
                           param_version=param_version)
            freed.append(unit_index)
        return tuple(freed)

    def change_quota(self, *, event_seq: int, shard_id: str,
                     new_lease_cap: int) -> None:
        old = self.state.shard_caps.get(shard_id)
        self.state.shard_caps[shard_id] = new_lease_cap
        self.state.decisions.append(Decision(
            event_seq=event_seq, kind="quota_change",
            basis=f"lease_cap {old} -> {new_lease_cap}",
            param_version=None, lease_version=None, units=(),
            before_owner=f"cap:{old}", after_owner=f"cap:{new_lease_cap}",
            shard_id=shard_id, lease_id=None))

    def snapshot(self) -> dict:
        return {
            "budget": self.state.budget,
            "free_capacity": self.state.free_capacity(),
            "shard_caps": dict(sorted(self.state.shard_caps.items())),
            "units": [record.__dict__ for record in self.state.units],
            "active_lease_units": {k: sorted(v) for k, v in
                                   sorted(self.state.active_lease_units.items())},
            "void_units": {k: sorted(v) for k, v in
                           sorted(self.state.void_units.items())},
            "expired_units": sorted(self.state.expired_units),
        }
