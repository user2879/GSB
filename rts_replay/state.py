"""Authoritative match state, production queues and snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorCode, RtsError
from .hashing import canonical_hash
from .rules import RuleConfig

SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class QueueItem:
    """A unit order waiting in the production queue."""

    order_id: int
    target: str
    amount: int
    enqueue_tick: int
    ready_tick: int
    quality: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "target": self.target,
            "amount": self.amount,
            "enqueue_tick": self.enqueue_tick,
            "ready_tick": self.ready_tick,
            "quality": self.quality,
        }


@dataclass
class FactionState:
    resources: int = 0
    total_collected: int = 0
    total_spent: int = 0
    idle_units: int = 0
    deployed_units: int = 0
    active_commands: int = 0
    queue: list[QueueItem] = field(default_factory=list)
    next_order_id: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources,
            "total_collected": self.total_collected,
            "total_spent": self.total_spent,
            "idle_units": self.idle_units,
            "deployed_units": self.deployed_units,
            "active_commands": self.active_commands,
            "queue": [item.to_dict() for item in self.queue],
            "next_order_id": self.next_order_id,
        }


@dataclass
class GameState:
    tick: int
    factions: dict[str, FactionState]
    last_seq: dict[str, int]

    @classmethod
    def initial(cls, config: RuleConfig) -> "GameState":
        return cls(
            tick=0,
            factions={
                name: FactionState() for name in config.faction_names()
            },
            last_seq={name: -1 for name in config.faction_names()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "factions": {
                name: faction.to_dict()
                for name, faction in sorted(self.factions.items())
            },
            "last_seq": dict(sorted(self.last_seq.items())),
        }

    def state_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def clone(self) -> "GameState":
        return GameState(
            tick=self.tick,
            factions={
                name: FactionState(
                    resources=item.resources,
                    total_collected=item.total_collected,
                    total_spent=item.total_spent,
                    idle_units=item.idle_units,
                    deployed_units=item.deployed_units,
                    active_commands=item.active_commands,
                    queue=[QueueItem(**vars(queue_item)) for queue_item in item.queue],
                    next_order_id=item.next_order_id,
                )
                for name, item in self.factions.items()
            },
            last_seq=dict(self.last_seq),
        )


@dataclass(frozen=True)
class Snapshot:
    format_version: int
    tick: int
    state: GameState
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "tick": self.tick,
            "state_hash": self.state_hash,
            "state": self.state.to_dict(),
        }


def build_snapshot(state: GameState) -> Snapshot:
    return Snapshot(
        format_version=SNAPSHOT_FORMAT_VERSION,
        tick=state.tick,
        state=state.clone(),
        state_hash=state.state_hash(),
    )


def snapshot_from_dict(raw: Any) -> Snapshot:
    if not isinstance(raw, dict):
        raise RtsError(ErrorCode.E_INPUT_MALFORMED, "snapshot must be a JSON object")
    if raw.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise RtsError(
            ErrorCode.E_SNAPSHOT_STALE,
            "snapshot format version is stale or unsupported",
            tick=raw.get("tick"),
            context={
                "expected": SNAPSHOT_FORMAT_VERSION,
                "actual": raw.get("format_version"),
            },
        )
    state_raw = raw["state"]
    factions = {}
    for name, item in state_raw["factions"].items():
        queue = [QueueItem(**queue_item) for queue_item in item.pop("queue")]
        factions[name] = FactionState(queue=queue, **item)
    state = GameState(
        tick=state_raw["tick"],
        factions=factions,
        last_seq=dict(state_raw["last_seq"]),
    )
    recomputed = state.state_hash()
    if recomputed != raw["state_hash"]:
        raise RtsError(
            ErrorCode.E_SNAPSHOT_STALE,
            "snapshot content does not match its recorded state hash",
            tick=state.tick,
            context={"recorded": raw["state_hash"], "recomputed": recomputed},
        )
    return Snapshot(
        format_version=raw["format_version"],
        tick=raw["tick"],
        state=state,
        state_hash=recomputed,
    )
