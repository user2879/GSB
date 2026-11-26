from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import GenerationError


STATE_PENDING = "pending"
STATE_GRANTED = "granted"
STATE_RENEWED = "renewed"
STATE_RELEASED = "released"
STATE_EXPIRED = "expired"

ACTIVE_STATES = {STATE_GRANTED, STATE_RENEWED}
TERMINAL_STATES = {STATE_RELEASED, STATE_EXPIRED}

ALLOWED_TRANSITIONS: dict[tuple[str, str], str] = {
    (STATE_PENDING, "grant"): STATE_GRANTED,
    (STATE_GRANTED, "renew"): STATE_RENEWED,
    (STATE_RENEWED, "renew"): STATE_RENEWED,
    (STATE_GRANTED, "release"): STATE_RELEASED,
    (STATE_RENEWED, "release"): STATE_RELEASED,
    (STATE_GRANTED, "expire"): STATE_EXPIRED,
    (STATE_RENEWED, "expire"): STATE_EXPIRED,
    (STATE_GRANTED, "invalidate"): STATE_EXPIRED,
    (STATE_RENEWED, "invalidate"): STATE_EXPIRED,
}


def next_state(state_before: str, trigger: str) -> str:
    try:
        return ALLOWED_TRANSITIONS[(state_before, trigger)]
    except KeyError as exc:
        raise GenerationError(
            f"transition {state_before} --{trigger}--> is not permitted",
            field="state",
            details={"state_before": state_before, "trigger": trigger},
        ) from exc


@dataclass(frozen=True)
class TransitionRecord:
    event_seq: int
    ordinal: int
    lease_id: str
    node_id: str
    trigger: str
    reason: str
    state_before: str
    state_after: str
    generation_before: int
    generation_after: int
    carried_label: int | None
    quota_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationChain:
    """Validates exact +1 monotonic generation steps for one lease."""

    def __init__(self, lease_id: str, generation: int = 0) -> None:
        self.lease_id = lease_id
        self.generation = generation
        self.records: list[TransitionRecord] = []

    def append(
        self,
        *,
        event_seq: int,
        ordinal: int,
        node_id: str,
        trigger: str,
        reason: str,
        state_before: str,
        state_after: str,
        generation_before: int,
        generation_after: int,
        carried_label: int | None,
        quota_version: int,
    ) -> TransitionRecord:
        if generation_before != self.generation:
            raise GenerationError(
                "generation cross-step or rollback rejected",
                entry=self.lease_id,
                field="generation_before",
                details={
                    "current_generation": self.generation,
                    "generation_before": generation_before,
                    "event_seq": event_seq,
                },
            )
        if generation_after != generation_before + 1:
            raise GenerationError(
                "generation may advance by exactly one",
                entry=self.lease_id,
                field="generation_after",
                details={
                    "current_generation": self.generation,
                    "generation_before": generation_before,
                    "generation_after": generation_after,
                    "event_seq": event_seq,
                },
            )
        expected_after = next_state(state_before, trigger)
        if expected_after != state_after:
            raise GenerationError(
                "state transition violates explicit permission table",
                entry=self.lease_id,
                field="state_after",
                details={"expected": expected_after, "actual": state_after},
            )
        record = TransitionRecord(
            event_seq=event_seq,
            ordinal=ordinal,
            lease_id=self.lease_id,
            node_id=node_id,
            trigger=trigger,
            reason=reason,
            state_before=state_before,
            state_after=state_after,
            generation_before=generation_before,
            generation_after=generation_after,
            carried_label=carried_label,
            quota_version=quota_version,
        )
        self.generation = generation_after
        self.records.append(record)
        return record


def sorted_transitions(records: list[TransitionRecord | dict[str, Any]]) -> list[Any]:
    def key(record: TransitionRecord | dict[str, Any]) -> tuple[int, int, str]:
        if isinstance(record, TransitionRecord):
            return record.event_seq, record.ordinal, record.lease_id
        return (
            int(record.get("event_seq", 0)),
            int(record.get("ordinal", 0)),
            str(record.get("lease_id", "")),
        )

    return sorted(records, key=key)
