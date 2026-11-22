"""Dataclasses shared by every pipeline stage.

All fields are plain JSON-serialisable types so that checkpoints can be
written directly and compared field-by-field between incremental and full
replay runs.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Event:
    """One raw line from the event JSONL stream."""

    session_id: str
    op: str                      # append | replace | delete
    turn: Optional[int]          # target/materialised turn number
    utterance: str = ""
    action: Optional[str] = None
    slot_updates: tuple = ()     # tuple[dict] - immutable, hashable context
    batch_id: str = ""
    source_turn: Optional[int] = None  # pre-delete sequence position
    injection_id: Optional[str] = None
    line_no: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransitionRule:
    rule_id: str
    from_state: str
    action: str
    to_state: str

    def to_dict(self):
        return asdict(self)


@dataclass
class SlotRecord:
    """Full per-slot state. ``history`` is an ordered provenance trail that
    records the strategy chosen and the source turn of every merge."""

    name: str
    type: str
    strategy: str
    value: Any = None
    confidence: int = 0
    weighted_sum: int = 0        # fixed-point accumulator (value * conf sum)
    weight_total: int = 0
    history: list = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "strategy": self.strategy,
            "value": self.value,
            "confidence": self.confidence,
            "weighted_sum": self.weighted_sum,
            "weight_total": self.weight_total,
            "history": list(self.history),
        }


@dataclass
class TurnCheckpoint:
    """Everything derivable up to and including one turn.

    Stored keyed by the per-turn hash-chain key.  A full prefix of checkpoints
    is the reusable cache; the longest matching prefix determines how many
    turns may be skipped after an edit.
    """

    session_id: str
    turn: int
    key: str
    state: str
    slots: dict                  # name -> SlotRecord
    transition_trace: list       # accepted transition entries
    rejected: list               # rejected transition entries
    failures: list               # FailureRecord dicts produced up to this turn
    consumed_entries: int        # state entries consumed by this session

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "key": self.key,
            "state": self.state,
            "slots": {name: rec.to_dict() for name, rec in sorted(self.slots.items())},
            "transition_trace": list(self.transition_trace),
            "rejected": list(self.rejected),
            "failures": list(self.failures),
            "consumed_entries": self.consumed_entries,
        }


@dataclass
class SessionResult:
    session_id: str
    status: str                  # complete | quota_exceeded | budget_pending
    final_state: str
    slots: dict
    transition_trace: list
    rejected: list
    failures: list
    checkpoints: list            # TurnCheckpoint list, turn order
    processed_turns: list        # turns advanced in the current run
    reused_turns: list           # turns served from cache in the current run
    consumed_entries: int


@dataclass(frozen=True)
class FailureRecord:
    session_id: str
    turn: Optional[int]
    stage: str
    code: str
    message: str
    batch_id: str = ""
    injection_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)

