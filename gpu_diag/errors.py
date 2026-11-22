"""Business error codes and structured logging for the GPU diagnostic workbench.

All domain failures carry a stable ``E_*`` code.  Logging is deterministic:
records are appended as one JSON object per line, sorted by key, so repeated
runs produce byte-identical log output for identical inputs.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


E_DECL = "E_DECL"        # invalid resource/task/shard declaration
E_MEMORY = "E_MEMORY"    # per-task memory budget exceeded / pool memory too small
E_QUOTA = "E_QUOTA"      # quota share sum or global GPU quota constraint violated
E_CYCLE = "E_CYCLE"      # dependency graph contains a cycle
E_LEDGER = "E_LEDGER"    # ledger content digest mismatch / corruption
E_REPLAY = "E_REPLAY"    # snapshot replay diverges from ledger
E_PREVIEW = "E_PREVIEW"  # authoritative recompute diverges from cached ledger
E_STATE = "E_STATE"      # unknown / uncommitted state after forced termination


@dataclass
class DiagError(Exception):
    code: str
    message: str
    details: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


class StructuredLogger:
    """Append-only JSON-lines logger, safe for the worker thread pool."""

    def __init__(self, path: str | Path | None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Append-only: repeated demonstration runs accumulate an audit trail.
            self._path.touch(exist_ok=True)

    def log(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"code": code, "message": message}
        record.update(fields)
        record = {key: record[key] for key in sorted(record)}
        if self._path is not None:
            line = json.dumps(record, sort_keys=True, ensure_ascii=False)
            with self._lock:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record
