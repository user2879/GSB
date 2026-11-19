"""Domain error codes and structured logging for the block snapshot component.

All domain failures raise :class:`DomainError` carrying one of the documented
business codes (``E_INPUT`` ...).  Every raised error is also appended to the
structured JSONL log so that triage sessions can reconstruct the failure chain
with snapshot id, block number and manifest version attached.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

E_INPUT = "E_INPUT"
E_PATH = "E_PATH"
E_INTEGRITY = "E_INTEGRITY"
E_CHAIN = "E_CHAIN"
E_SHAPE = "E_SHAPE"
E_STATE = "E_STATE"

ALL_CODES = (E_INPUT, E_PATH, E_INTEGRITY, E_CHAIN, E_SHAPE, E_STATE)


class DomainError(Exception):
    """Business exception carrying a stable code and structured context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        block_no: Optional[int] = None,
        manifest_version: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if code not in ALL_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.snapshot_id = snapshot_id
        self.block_no = block_no
        self.manifest_version = manifest_version
        self.extra = dict(extra or {})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "snapshot_id": self.snapshot_id,
            "block_no": self.block_no,
            "manifest_version": self.manifest_version,
        }
        data.update(self.extra)
        return data


class StructuredLogger:
    """Append-only JSONL logger shared by the package.

    The logger is thread safe and never raises into business logic: disk
    failures degrade to in-memory entries instead of breaking a rollback.
    """

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.entries: list[dict[str, Any]] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if line:
                                self.entries.append(json.loads(line))
                except (OSError, json.JSONDecodeError):
                    pass

    def log(
        self,
        level: str,
        event: str,
        *,
        code: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        block_no: Optional[int] = None,
        manifest_version: Optional[int] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "level": level,
            "event": event,
            "code": code,
            "snapshot_id": snapshot_id,
            "block_no": block_no,
            "manifest_version": manifest_version,
        }
        for key, value in fields.items():
            if key not in entry:
                entry[key] = value
        with self._lock:
            self.entries.append(entry)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, sort_keys=True) + "\n")
                except OSError:
                    pass
        return entry

    def log_error(self, err: DomainError, event: str = "domain_error") -> dict[str, Any]:
        payload = err.to_dict()
        return self.log(
            "ERROR",
            event,
            code=err.code,
            snapshot_id=err.snapshot_id,
            block_no=err.block_no,
            manifest_version=err.manifest_version,
            field=err.field,
            **{k: v for k, v in err.extra.items()},
            message=err.message,
        )

    def query(
        self,
        *,
        code: Optional[str] = None,
        event: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            result = list(self.entries)
        if code is not None:
            result = [row for row in result if row.get("code") == code]
        if event is not None:
            result = [row for row in result if row.get("event") == event]
        if snapshot_id is not None:
            result = [row for row in result if row.get("snapshot_id") == snapshot_id]
        return result
