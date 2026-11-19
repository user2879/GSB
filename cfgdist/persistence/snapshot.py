"""Snapshot file + SQLite coordination with version/digest identification."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util.canonical import digest
from ..util.logging_util import StructuredLogger
from .database import Database


@dataclass
class SnapshotIdentity:
    version: int
    manifest_digest: str
    plan_digest: str
    same_version: bool
    same_digest: bool
    replaced: bool


class SnapshotStore:
    def __init__(self, db: Database, snapshot_dir: str | Path, logger: StructuredLogger) -> None:
        self.db = db
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _file_path(self, session_id: str, version: int) -> Path:
        return self.snapshot_dir / f"{session_id}-v{version}.json"

    def save(
        self,
        session_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> SnapshotIdentity:
        manifest_digest = digest(payload["manifest"])
        plan_digest = digest(payload["plan"])
        existing_row = self.db.get_snapshot(session_id, version)
        same_digest = (
            existing_row is not None
            and existing_row["manifest_digest"] == manifest_digest
            and existing_row["plan_digest"] == plan_digest
        )
        # Always execute whole-snapshot replacement, never an incremental merge.
        self.db.replace_snapshot(session_id, version, payload)
        with self._file_path(session_id, version).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, ensure_ascii=False, indent=2)
        replaced = existing_row is not None and not same_digest
        self.logger.log(
            "snapshot_saved",
            source="snapshot",
            key=str(version),
            replaced=replaced,
            manifest_digest=manifest_digest,
        )
        return SnapshotIdentity(
            version=version,
            manifest_digest=manifest_digest,
            plan_digest=plan_digest,
            same_version=existing_row is not None,
            same_digest=same_digest,
            replaced=replaced,
        )

    def identify(self, session_id: str, candidate_version: int, candidate_manifest: dict[str, Any]) -> SnapshotIdentity | None:
        row = self.db.get_snapshot(session_id, candidate_version)
        if row is None:
            return None
        candidate_digest = digest(candidate_manifest)
        return SnapshotIdentity(
            version=candidate_version,
            manifest_digest=row["manifest_digest"],
            plan_digest=row["plan_digest"],
            same_version=True,
            same_digest=row["manifest_digest"] == candidate_digest,
            replaced=False,
        )

    def load(self, session_id: str, version: int) -> dict[str, Any]:
        return self.db.load_snapshot_payload(session_id, version)
