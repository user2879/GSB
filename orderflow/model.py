from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ORDER_TYPES = ("MOVE", "SUPPRESS", "SUPPLY", "RETREAT")
EXCLUSIVE_PAIRS = (frozenset({"MOVE", "RETREAT"}),)
EXCLUSIVE_TYPES = frozenset().union(*EXCLUSIVE_PAIRS)
ORDER_COST = {"MOVE": -10, "SUPPRESS": -25, "SUPPLY": 40, "RETREAT": -5}

E_SEQ_GAP = "E_SEQ_GAP"
E_ORDER_TYPE = "E_ORDER_TYPE"
E_CELL_BOUNDS = "E_CELL_BOUNDS"
E_PARSE = "E_PARSE"
E_LATE_ORDER = "E_LATE_ORDER"
E_DUPLICATE_ORDER = "E_DUPLICATE_ORDER"
E_TIE_UNRESOLVED = "E_TIE_UNRESOLVED"
E_ORDER_OVERRIDDEN = "E_ORDER_OVERRIDDEN"
E_SUPPLY_OVERFLOW = "E_SUPPLY_OVERFLOW"
E_BUDGET_DENIED = "E_BUDGET_DENIED"


@dataclass(frozen=True)
class Config:
    tick_ms: int = 100
    max_latency_ms: int = 250
    adjudication_window_ms: int = 150
    grid_w: int = 64
    grid_h: int = 64
    supply_cap: int = 200
    initial_supply: int = 100


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj) -> bytes:
    return (canonical_json(obj) + "\n").encode("utf-8")


def content_hash(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


@dataclass
class Order:
    line: int
    tick: int
    commander: str
    unit_id: str
    order_type: str
    target_cell: tuple
    priority: int
    latency_ms: int
    seq: int
    expired: bool = False

    @property
    def order_id(self) -> str:
        return f"{self.commander}#{self.seq}"

    def abs_time_ms(self, tick_ms: int) -> int:
        return self.tick * tick_ms + self.latency_ms

    def fingerprint(self) -> str:
        return content_hash({
            "tick": self.tick,
            "commander": self.commander,
            "unit_id": self.unit_id,
            "order_type": self.order_type,
            "target_cell": list(self.target_cell),
            "priority": self.priority,
            "latency_ms": self.latency_ms,
            "seq": self.seq,
        })
