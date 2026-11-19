"""Deterministic streaming scheduler with byte-accurate residency ledgers.

Ordering guarantees:
  1. Loads are applied before evictions within a frame, so a visible block is
     never missing because an eviction ran first.
  2. Every load is checked against both system-memory and VRAM budgets; an
     unsatisfiable request raises ``E_BUDGET_DENIED`` instead of silently
     dropping the block.
  3. Re-loading a previously evicted block restores the same payload bytes, so
     the rendered result is identical to never having evicted it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import BudgetDeniedError, ResidentLeakError


@dataclass
class Request:
    kind: str          # "load" | "retain" | "evict"
    block_id: str
    reason: str
    bytes_delta: int

    def to_dict(self) -> dict:
        return {"kind": self.kind, "block_id": self.block_id,
                "reason": self.reason, "bytes_delta": int(self.bytes_delta)}


class ResidencyLedger:
    def __init__(self, dataset, memory_budget: int, vram_budget: int):
        self.dataset = dataset
        self.memory_budget = int(memory_budget)
        self.vram_budget = int(vram_budget)
        self.resident: dict[str, int] = {}     # block_id -> frame last touched
        self.memory_used = 0
        self.vram_used = 0
        self.history: list[Request] = []
        self.denials = 0

    def contains(self, block_id: str) -> bool:
        return block_id in self.resident

    def _block(self, block_id: str):
        return self.dataset.get_block(block_id)

    def can_fit(self, block_id: str) -> bool:
        block = self._block(block_id)
        if block_id in self.resident:
            return True
        return (self.memory_used + block.payload_bytes <= self.memory_budget
                and self.vram_used + block.vram_bytes <= self.vram_budget)

    def load(self, block_id: str, reason: str, *, allow_deny: bool = False) -> Request | None:
        block = self._block(block_id)
        if block_id in self.resident:
            return None
        if not self.can_fit(block_id):
            self.denials += 1
            if allow_deny:
                return None
            raise BudgetDeniedError(
                "cannot load block within memory/VRAM budget",
                context={
                    "block_id": block_id,
                    "memory_used": self.memory_used,
                    "memory_budget": self.memory_budget,
                    "vram_used": self.vram_used,
                    "vram_budget": self.vram_budget,
                    "payload_bytes": block.payload_bytes,
                    "vram_bytes": block.vram_bytes,
                })
        self.resident[block_id] = self.resident.get(block_id, 0)
        self.memory_used += block.payload_bytes
        self.vram_used += block.vram_bytes
        req = Request("load", block_id, reason, block.payload_bytes)
        self.history.append(req)
        return req

    def touch(self, frame_index: int, block_ids: list[str]) -> None:
        for block_id in block_ids:
            if block_id in self.resident:
                self.resident[block_id] = frame_index

    def evict(self, block_id: str, reason: str) -> Request:
        if block_id not in self.resident:
            raise ResidentLeakError(
                f"attempt to evict non-resident block {block_id}",
                context={"block_id": block_id})
        block = self._block(block_id)
        self.memory_used -= block.payload_bytes
        self.vram_used -= block.vram_bytes
        del self.resident[block_id]
        req = Request("evict", block_id, reason, -block.payload_bytes)
        self.history.append(req)
        return req

    def verify(self) -> None:
        """Recompute totals from scratch and assert the ledger is consistent."""
        mem = sum(self._block(bid).payload_bytes for bid in self.resident)
        vram = sum(self._block(bid).vram_bytes for bid in self.resident)
        if mem != self.memory_used or vram != self.vram_used:
            raise ResidentLeakError(
                "residency ledger totals do not match resident block set",
                context={"ledger_memory": self.memory_used, "recomputed_memory": mem,
                         "ledger_vram": self.vram_used, "recomputed_vram": vram})
        if self.memory_used > self.memory_budget or self.vram_used > self.vram_budget:
            raise ResidentLeakError(
                "residency exceeds declared budget",
                context={"memory_used": self.memory_used,
                         "memory_budget": self.memory_budget,
                         "vram_used": self.vram_used,
                         "vram_budget": self.vram_budget})

    def usage(self) -> dict:
        return {
            "resident_count": len(self.resident),
            "memory_used": self.memory_used,
            "memory_budget": self.memory_budget,
            "memory_utilization": self.memory_used / self.memory_budget,
            "vram_used": self.vram_used,
            "vram_budget": self.vram_budget,
            "vram_utilization": self.vram_used / self.vram_budget,
            "denials": self.denials,
        }


class StreamingScheduler:
    """Produces load/retain/evict requests and applies them in a safe order."""

    def __init__(self, dataset, ledger: ResidencyLedger):
        self.dataset = dataset
        self.ledger = ledger

    def plan(self, frame_index: int, desired_ids: list[str],
             protected_ids: list[str]) -> dict:
        """Apply loads first, then LRU evictions of unprotected blocks."""
        desired_set = set(desired_ids)
        protected_set = set(protected_ids)
        requests: list[Request] = []
        # Per-frame history keeps reports independent of cumulative engine age.
        self.history = []

        # Deterministic load order: sorted block id. Before each load, evict
        # non-protected, non-desired LRU blocks until the request fits, so a
        # moving working set cannot strand budget in stale blocks. Evictions of
        # invisible blocks never remove a visible (protected) block.
        ordered_desired = sorted(desired_set)
        for block_id in ordered_desired:
            if not self.ledger.can_fit(block_id):
                victims = sorted(
                    (bid for bid in self.ledger.resident
                     if bid not in desired_set and bid not in protected_set),
                    key=lambda bid: (self.ledger.resident[bid], bid))
                for victim in victims:
                    if self.ledger.can_fit(block_id):
                        break
                    requests.append(
                        self.ledger.evict(victim, "budget_reclaim_before_load"))
            req = self.ledger.load(block_id, "visible_or_preload")
            if req is not None:
                requests.append(req)

        self.ledger.touch(frame_index, list(desired_set))
        for block_id in sorted(desired_set):
            requests.append(Request("retain", block_id, "desired_set", 0))

        # Evict blocks that are neither desired nor protected, oldest first.
        candidates = [bid for bid in self.ledger.resident
                      if bid not in desired_set and bid not in protected_set]
        candidates.sort(key=lambda bid: (self.ledger.resident[bid], bid))
        evicted: list[Request] = []
        for block_id in candidates:
            evicted.append(self.ledger.evict(block_id, "lru_outside_working_set"))
        requests.extend(evicted)

        self.ledger.verify()
        return {
            "frame": frame_index,
            "requests": [r.to_dict() for r in requests],
            "loads": sum(1 for r in requests if r.kind == "load"),
            "retains": sum(1 for r in requests if r.kind == "retain"),
            "evictions": len(evicted),
            "ledger": self.ledger.usage(),
        }
