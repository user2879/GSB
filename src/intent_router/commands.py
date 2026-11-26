"""Deterministic edit-command stream applied to corpus and trigger chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import FailureCollector, E_BAD_COMMAND
from .models import TextItem


@dataclass(frozen=True)
class Command:
    seq: int
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Command":
        return cls(
            seq=int(data["seq"]),
            kind=str(data["kind"]),
            payload=dict(data.get("payload", {})),
        )


VALID_KINDS = {
    "add_text",
    "replace_text",
    "delete_text",
    "set_head_revision",
    "upsert_trigger",
    "delete_trigger",
    "set_weight",
    "set_dedup_threshold",
    "set_gear",
    "set_fraction_bits",
}


def parse_commands(
    raw_commands: list[dict[str, Any]], failures: FailureCollector
) -> list[Command]:
    commands: list[Command] = []
    for raw in raw_commands:
        kind = str(raw.get("kind", ""))
        if kind not in VALID_KINDS:
            failures.add(
                E_BAD_COMMAND,
                "unknown command kind",
                "command",
                str(raw.get("seq", "?")),
                kind=kind,
            )
            continue
        commands.append(Command.from_dict(raw))
    commands.sort(key=lambda c: c.seq)
    return commands


def apply_text_commands(
    texts: dict[str, TextItem],
    commands: list[Command],
    vocab: tuple[str, ...],
    failures: FailureCollector,
) -> dict[str, TextItem]:
    """Pure text-state transitions; returns a new dict, never mutates input."""
    vocab_set = set(vocab)
    state = dict(texts)
    for command in commands:
        p = command.payload
        if command.kind == "add_text" or command.kind == "replace_text":
            tokens = tuple(str(t) for t in p["tokens"])
            bad = sorted({t for t in tokens if t not in vocab_set})
            if bad:
                failures.add(
                    "E_TOKEN_OUT_OF_VOCAB",
                    "tokens outside controlled vocab",
                    "text",
                    str(p["text_id"]),
                    tokens=bad,
                )
                continue
            state[str(p["text_id"])] = TextItem(
                text_id=str(p["text_id"]),
                tokens=tokens,
                channel=str(p["channel"]),
                gold_label=str(p["gold_label"]),
            )
        elif command.kind == "delete_text":
            state.pop(str(p["text_id"]), None)
    return state
