"""Append-only deterministic audit ledger plus canonical state snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class LedgerWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "ledger.jsonl"
        self.state_path = self.root / "state.json"

    def append_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")
            handle.flush()

    def write_state(self, state: dict[str, Any]) -> None:
        temporary = self.root / ".state.tmp"
        temporary.write_text(canonical_json(state) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        with self.events_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
