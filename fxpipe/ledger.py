"""Content-fingerprint cache ledger with exact operator/token invalidation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fingerprint import cache_key, fingerprint


@dataclass
class LedgerEntry:
    entry_key: str                 # stable identity: node + stage + token set
    node_id: str
    chain_id: int
    seq: int
    op_type: str
    stage: str
    token_indices: list[int]
    cache_key: str
    output_fp: str
    output: object
    upstream_keys: list[str]
    param_fp: str
    gear: str
    exchange_version: str
    frac_bits: int
    referenced_tokens: list[int]
    referenced_ops: list[str]
    status: str = "ready"          # ready | pending
    cost_points: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerEntry":
        return cls(**data)


@dataclass
class LedgerEvent:
    kind: str                      # invalidated | reused
    entry_key: str
    reason: str
    cache_key: str


class Ledger:
    def __init__(self) -> None:
        self.entries: dict[str, LedgerEntry] = {}
        self.events: list[LedgerEvent] = []

    def upsert(self, entry: LedgerEntry) -> None:
        self.entries[entry.entry_key] = entry

    def get(self, entry_key: str) -> LedgerEntry | None:
        return self.entries.get(entry_key)

    def all_entries(self) -> list[LedgerEntry]:
        return [self.entries[key] for key in sorted(self.entries)]

    def invalidate(self, entry_key: str, reason: str) -> bool:
        entry = self.entries.pop(entry_key, None)
        if entry is None:
            return False
        self.events.append(LedgerEvent("invalidated", entry_key, reason, entry.cache_key))
        return True

    def record_reuse(self, entry: LedgerEntry, reason: str) -> None:
        self.events.append(LedgerEvent("reused", entry.entry_key, reason, entry.cache_key))

    def events_since(self, marker: int) -> list[LedgerEvent]:
        return self.events[marker:]

    def save(self, path: str | Path) -> None:
        payload = {
            "entries": [entry.to_dict() for entry in self.all_entries()],
            "events": [asdict(event) for event in self.events],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        ledger = cls()
        if not Path(path).exists():
            return ledger
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for data in payload.get("entries", []):
            ledger.upsert(LedgerEntry.from_dict(data))
        for event in payload.get("events", []):
            ledger.events.append(LedgerEvent(**event))
        return ledger


def make_entry_key(node_id: str, stage: str, token_indices: list[int]) -> str:
    return f"{node_id}:{stage}:t{','.join(map(str, token_indices))}"


def derive_cache_key(
    upstream_keys: list[str],
    param_fp: str,
    gear: str,
    frac_bits: int,
    exchange_version: str,
) -> str:
    return cache_key(
        [sorted(upstream_keys), param_fp, {"gear": gear, "frac_bits": frac_bits, "version": exchange_version}],
        prefix="cache",
    )


def param_fingerprint(op: object) -> str:
    payload = {
        "op_type": op.op_type,
        "params": op.params,
        "permille": op.quant.permille,
        "width": op.quant.width,
        "scheme": op.quant.scheme,
        "precision_label": op.quant.precision_label,
        "missing": list(op.quant.missing),
    }
    return fingerprint(payload, prefix="param")


def output_fingerprint(output: object) -> str:
    return fingerprint(output, prefix="out")


def token_row_fingerprint(token: int, row: list[int]) -> str:
    return fingerprint({"token": token, "row": row}, prefix="embrow")
