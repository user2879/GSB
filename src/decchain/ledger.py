"""Durable, atomic ledger with staged snapshot replacement.

Snapshots are written to a temp file and atomically renamed only after the
full payload is serialized.  Recovery therefore never observes a partial
snapshot: interrupted temp files are discarded on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from .config import DecodeConfig, canonical_json
from .errors import BizCode, BizError


@dataclass
class Snapshot:
    version: int
    config_fingerprint: str
    config: dict[str, Any]
    exchange_version: str
    conclusions: list[dict[str, Any]]
    boundaries: list[dict[str, Any]]
    cache_events: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    todos: list[dict[str, Any]]

    def payload_hash(self) -> str:
        body = canonical_json({
            "conclusions": self.conclusions,
            "boundaries": self.boundaries,
            "cache_events": self.cache_events,
            "failures": self.failures,
            "todos": self.todos,
        })
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.snapshot_path = os.path.join(directory, "snapshot.json")
        self.temp_path = os.path.join(directory, "snapshot.json.tmp")
        self.event_path = os.path.join(directory, "events.jsonl")

    def commit_snapshot(self, snapshot: Snapshot) -> str:
        payload = {
            "version": snapshot.version,
            "config_fingerprint": snapshot.config_fingerprint,
            "config": snapshot.config,
            "exchange_version": snapshot.exchange_version,
            "payload_hash": snapshot.payload_hash(),
            "conclusions": snapshot.conclusions,
            "boundaries": snapshot.boundaries,
            "cache_events": snapshot.cache_events,
            "failures": snapshot.failures,
            "todos": snapshot.todos,
        }
        # Write fully then atomically replace; readers never see partial data.
        fd, tmp = tempfile.mkstemp(prefix="snapshot.", suffix=".tmp",
                                   dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.snapshot_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._append_event({"type": "commit", "version": snapshot.version,
                            "config_fingerprint": snapshot.config_fingerprint})
        return payload["payload_hash"]

    def load_snapshot(self, expected_version: int | None = None,
                      expected_config_fp: str | None = None) -> Snapshot | None:
        if not os.path.exists(self.snapshot_path):
            return None
        with open(self.snapshot_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        snap = Snapshot(
            version=payload["version"],
            config_fingerprint=payload["config_fingerprint"],
            config=payload["config"],
            exchange_version=payload["exchange_version"],
            conclusions=payload["conclusions"],
            boundaries=payload["boundaries"],
            cache_events=payload["cache_events"],
            failures=payload["failures"],
            todos=payload["todos"],
        )
        if expected_version is not None and snap.version != expected_version:
            raise BizError(BizCode.STALE_SNAPSHOT_VERSION, "ledger.version",
                           f"snapshot v{snap.version} != expected "
                           f"v{expected_version}")
        if (expected_config_fp is not None
                and snap.config_fingerprint != expected_config_fp):
            raise BizError(BizCode.CONFIG_HASH_MISMATCH, "ledger.config",
                           "snapshot config hash differs from active config")
        return snap

    def _append_event(self, event: dict[str, Any]) -> None:
        with open(self.event_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True,
                                    ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.event_path):
            return []
        with open(self.event_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def discard_temp_files(self) -> list[str]:
        removed = []
        for name in os.listdir(self.directory):
            if name.endswith(".tmp"):
                os.unlink(os.path.join(self.directory, name))
                removed.append(name)
        return removed
