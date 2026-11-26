from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, sha256_value
from .errors import CkptError
from .plan import BatchPlan, CheckpointPlan, RollbackPlan
from .spec import SpecBundle


FORMAT_VERSION = 1


def _json_load(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class SnapshotIdentity:
    format_version: int
    shard_set_version: str
    parameter_version_dialect: str
    content_digest: str

    @classmethod
    def from_bundle(cls, bundle: SpecBundle) -> "SnapshotIdentity":
        return cls(
            format_version=FORMAT_VERSION,
            shard_set_version=bundle.shard_set_version,
            parameter_version_dialect=bundle.parameter_dialect,
            content_digest=bundle.content_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "shard_set_version": self.shard_set_version,
            "parameter_version_dialect": self.parameter_version_dialect,
            "content_digest": self.content_digest,
        }


@dataclass
class SnapshotReport:
    action: str
    differences: list[dict[str, Any]] = field(default_factory=list)
    backup_path: str | None = None
    manifest_path: str | None = None

    @property
    def rebuilt(self) -> bool:
        return self.action == "replaced"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rebuilt": self.rebuilt,
            "differences": list(self.differences),
            "backup_path": self.backup_path,
            "manifest_path": self.manifest_path,
        }


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    sample_ids TEXT NOT NULL,
    samples_digest TEXT,
    parameter_digest TEXT,
    loss_json TEXT,
    loss_digest TEXT,
    duration_ms INTEGER,
    checkpoint_id TEXT,
    checkpoint_status TEXT NOT NULL,
    lease_id TEXT,
    lease_version INTEGER,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL,
    PRIMARY KEY (shard_id, batch_id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    parameter_digests TEXT NOT NULL,
    batch_keys TEXT NOT NULL,
    loss_baseline TEXT NOT NULL,
    baseline_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rollbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    checkpoint_id TEXT NOT NULL,
    target_version INTEGER NOT NULL,
    target_generation INTEGER NOT NULL,
    revoked_versions TEXT NOT NULL,
    revoked_checkpoints TEXT NOT NULL,
    affected_batches TEXT NOT NULL,
    generation_before INTEGER NOT NULL,
    generation_after INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inflight (
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    sample_ids TEXT NOT NULL,
    checkpoint_id TEXT,
    lease_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    started_ms INTEGER NOT NULL,
    attempt_id TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (shard_id, batch_id)
);
CREATE TABLE IF NOT EXISTS batch_attempts (
    attempt_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    finished_ms INTEGER,
    lease_id TEXT NOT NULL,
    detail TEXT,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    granted_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_renewals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    old_version INTEGER NOT NULL,
    new_version INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    shard_id TEXT PRIMARY KEY,
    cursor_seq INTEGER NOT NULL,
    cursor_batch_id TEXT,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS impact_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_json TEXT NOT NULL,
    report_digest TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    lease_id TEXT,
    detail TEXT,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS batches_order_idx ON batches (seq, shard_id, batch_id);
CREATE INDEX IF NOT EXISTS rollbacks_seq_idx ON rollbacks (seq);
CREATE UNIQUE INDEX IF NOT EXISTS rollbacks_unique_idx ON rollbacks (seq, checkpoint_id);
CREATE UNIQUE INDEX IF NOT EXISTS lease_renewals_unique_idx
    ON lease_renewals (lease_id, old_version, new_version);
CREATE UNIQUE INDEX IF NOT EXISTS recovery_advance_unique_idx
    ON recovery_log (content_digest) WHERE status = 'advance';
"""


def _batch_content(payload: dict[str, Any]) -> str:
    return sha256_value({"kind": "batch", **payload})


class Ledger:
    def __init__(self, path: str | Path, connection: sqlite3.Connection) -> None:
        self.path = Path(path)
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.snapshot_report: SnapshotReport | None = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        identity: SnapshotIdentity,
        initial_versions: dict[str, int],
        initial_generation: int = 0,
    ) -> "Ledger":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        report = SnapshotReport(action="created")
        if target.exists() and target.stat().st_size > 0:
            previous, differences = cls._inspect(target, identity)
            if differences:
                backup = cls._backup(target)
                report = SnapshotReport(
                    action="replaced",
                    differences=differences,
                    backup_path=str(backup),
                )
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(str(target) + suffix)
                    if sidecar.exists():
                        sidecar.unlink()
                target.unlink()
            else:
                report = SnapshotReport(action="opened")
        connection = sqlite3.connect(
            str(target),
            timeout=15,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        ledger = cls(target, connection)
        with ledger._transaction():
            meta = {
                "format_version": str(identity.format_version),
                "shard_set_version": identity.shard_set_version,
                "parameter_version_dialect": identity.parameter_version_dialect,
                "content_digest": identity.content_digest,
            }
            for key, value in meta.items():
                existing = connection.execute(
                    "SELECT value FROM meta WHERE key = ?", (key,)
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?)", (key, value)
                    )
            if connection.execute(
                "SELECT value FROM meta WHERE key = 'current_versions'"
            ).fetchone() is None:
                connection.execute(
                    "INSERT INTO meta (key, value) VALUES ('current_versions', ?)",
                    (canonical_json(dict(sorted(initial_versions.items()))),),
                )
                connection.execute(
                    "INSERT INTO meta (key, value) VALUES ('generation_counter', ?)",
                    (str(initial_generation),),
                )
        report.manifest_path = str(ledger._write_snapshot_manifest(identity, report))
        ledger.snapshot_report = report
        return ledger

    @staticmethod
    def _inspect(
        path: Path, identity: SnapshotIdentity
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            rows = connection.execute("SELECT key, value FROM meta").fetchall()
            previous = {str(key): str(value) for key, value in rows}
            connection.close()
        except sqlite3.Error:
            return {}, [
                {
                    "field": "database",
                    "previous": "unreadable",
                    "current": "readable snapshot",
                }
            ]
        current = identity.to_dict()
        differences: list[dict[str, Any]] = []
        for key in (
            "format_version",
            "shard_set_version",
            "parameter_version_dialect",
            "content_digest",
        ):
            old = previous.get(key)
            new = str(current[key])
            if old != new:
                differences.append({"field": key, "previous": old, "current": new})
        return previous, differences

    @staticmethod
    def _backup(path: Path) -> Path:
        try:
            connection = sqlite3.connect(str(path), timeout=5)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
        except sqlite3.Error:
            pass
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = path.with_name(f"{path.name}.backup-{timestamp}.sqlite")
        counter = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.backup-{timestamp}-{counter}.sqlite")
            counter += 1
        shutil.copy2(path, backup)
        os.chmod(backup, 0o444)
        return backup

    def _write_snapshot_manifest(
        self, identity: SnapshotIdentity, report: SnapshotReport
    ) -> Path:
        manifest = self.path.with_name(self.path.name + ".snapshot.json")
        payload = {
            "identity": identity.to_dict(),
            "snapshot": report.to_dict(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        temporary.replace(manifest)
        return manifest

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

    def set_meta(self, key: str, value: Any) -> None:
        with self._transaction():
            self.connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, canonical_json(value) if not isinstance(value, str) else value),
            )

    def current_state(self) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT key, value FROM meta WHERE key IN ('current_versions', 'generation_counter')"
            ).fetchall()
        values = {str(item["key"]): str(item["value"]) for item in row}
        return {
            "current_versions": json.loads(values.get("current_versions", "{}")),
            "generation_counter": int(values.get("generation_counter", "0")),
        }

    def begin_batch(
        self,
        plan: BatchPlan,
        *,
        lease_id: str,
        lease_version: int,
        granted_ms: int,
        expires_ms: int,
    ) -> str:
        started_ms = _clock_ms()
        attempt_id = f"attempt-{plan.event.seq}-{plan.shard_id}-{plan.batch_id}-{started_ms}"
        inflight_payload = {
            "shard_id": plan.shard_id,
            "batch_id": plan.batch_id,
            "seq": plan.event.seq,
            "parameter_version": plan.parameter_version,
            "generation": plan.generation,
            "sample_ids": list(plan.sample_ids),
            "checkpoint_id": plan.checkpoint_id,
            "lease_id": lease_id,
            "lease_version": lease_version,
            "started_ms": started_ms,
            "attempt_id": attempt_id,
        }
        digest = sha256_value({"kind": "inflight", **inflight_payload})
        attempt_digest = sha256_value(
            {
                "kind": "attempt-start",
                "attempt_id": attempt_id,
                "shard_id": plan.shard_id,
                "batch_id": plan.batch_id,
                "status": "running",
                "started_ms": started_ms,
                "lease_id": lease_id,
            }
        )
        lease_digest = sha256_value(
            {
                "kind": "lease",
                "lease_id": lease_id,
                "shard_id": plan.shard_id,
                "batch_id": plan.batch_id,
                "version": lease_version,
                "granted_ms": granted_ms,
                "expires_ms": expires_ms,
                "status": "active",
            }
        )
        with self._transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO inflight "
                "(shard_id,batch_id,seq,parameter_version,generation,sample_ids,"
                " checkpoint_id,lease_id,lease_version,started_ms,attempt_id,content_digest) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan.shard_id,
                    plan.batch_id,
                    plan.event.seq,
                    plan.parameter_version,
                    plan.generation,
                    canonical_json(list(plan.sample_ids)),
                    plan.checkpoint_id,
                    lease_id,
                    lease_version,
                    started_ms,
                    attempt_id,
                    digest,
                ),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO batch_attempts "
                "(attempt_id,shard_id,batch_id,status,started_ms,finished_ms,lease_id,detail,content_digest) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    plan.shard_id,
                    plan.batch_id,
                    "running",
                    started_ms,
                    None,
                    lease_id,
                    None,
                    attempt_digest,
                ),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO leases "
                "(lease_id,shard_id,batch_id,version,granted_ms,expires_ms,status,content_digest) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    lease_id,
                    plan.shard_id,
                    plan.batch_id,
                    lease_version,
                    granted_ms,
                    expires_ms,
                    "active",
                    lease_digest,
                ),
            )
        return attempt_id

    def complete_batch(
        self,
        plan: BatchPlan,
        response: dict[str, Any],
        *,
        checkpoint_id: str | None,
        checkpoint_status: str,
        lease_id: str,
        lease_version: int,
        attempt_id: str,
        status: str = "committed",
    ) -> dict[str, Any]:
        updated_ms = _clock_ms()
        sample_ids = sorted(plan.sample_ids)
        payload = {
            "shard_id": plan.shard_id,
            "batch_id": plan.batch_id,
            "seq": plan.event.seq,
            "parameter_version": plan.parameter_version,
            "generation": plan.generation,
            "sample_ids": sample_ids,
            "samples_digest": response["samples_digest"],
            "parameter_digest": response["parameter_digest"],
            "loss_json": response["loss"],
            "loss_digest": response["loss_digest"],
            "duration_ms": response["duration_ms"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_status": checkpoint_status,
            "lease_id": lease_id,
            "lease_version": lease_version,
            "status": status,
        }
        digest = _batch_content(payload)
        attempt_digest = sha256_value(
            {
                "kind": "attempt-finish",
                "attempt_id": attempt_id,
                "status": status,
                "finished_ms": updated_ms,
                "loss_digest": response["loss_digest"],
            }
        )
        with self._transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO batches "
                "(shard_id,batch_id,seq,parameter_version,generation,sample_ids,"
                " samples_digest,parameter_digest,loss_json,loss_digest,duration_ms,"
                " checkpoint_id,checkpoint_status,lease_id,lease_version,status,"
                " content_digest,updated_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan.shard_id,
                    plan.batch_id,
                    plan.event.seq,
                    plan.parameter_version,
                    plan.generation,
                    canonical_json(sample_ids),
                    response["samples_digest"],
                    response["parameter_digest"],
                    canonical_json(response["loss"]),
                    response["loss_digest"],
                    response["duration_ms"],
                    checkpoint_id,
                    checkpoint_status,
                    lease_id,
                    lease_version,
                    status,
                    digest,
                    updated_ms,
                ),
            )
            self.connection.execute(
                "UPDATE batch_attempts SET status=?, finished_ms=?, detail=?, content_digest=? "
                "WHERE attempt_id=?",
                (status, updated_ms, None, attempt_digest, attempt_id),
            )
            self.connection.execute(
                "UPDATE leases SET status='released', content_digest=? WHERE lease_id=?",
                (
                    sha256_value(
                        {
                            "kind": "lease",
                            "lease_id": lease_id,
                            "status": "released",
                            "version": lease_version,
                        }
                    ),
                    lease_id,
                ),
            )
            self.connection.execute(
                "DELETE FROM inflight WHERE shard_id=? AND batch_id=?",
                (plan.shard_id, plan.batch_id),
            )
            current = self.connection.execute(
                "SELECT cursor_seq FROM progress WHERE shard_id=?", (plan.shard_id,)
            ).fetchone()
            if current is None or int(current["cursor_seq"]) <= plan.event.seq:
                progress_digest = sha256_value(
                    {
                        "kind": "progress",
                        "shard_id": plan.shard_id,
                        "cursor_seq": plan.event.seq,
                        "cursor_batch_id": plan.batch_id,
                    }
                )
                self.connection.execute(
                    "INSERT INTO progress (shard_id,cursor_seq,cursor_batch_id,content_digest) "
                    "VALUES (?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET "
                    "cursor_seq=excluded.cursor_seq, cursor_batch_id=excluded.cursor_batch_id, "
                    "content_digest=excluded.content_digest",
                    (plan.shard_id, plan.event.seq, plan.batch_id, progress_digest),
                )
        return {**payload, "content_digest": digest, "updated_ms": updated_ms}

    def _mark_unknown_rows(
        self,
        rows: list[sqlite3.Row],
        *,
        detail: str,
        lease_status: str,
    ) -> list[dict[str, Any]]:
        updated_ms = _clock_ms()
        recovered: list[dict[str, Any]] = []
        for row in rows:
            payload = {
                "shard_id": row["shard_id"],
                "batch_id": row["batch_id"],
                "seq": row["seq"],
                "parameter_version": row["parameter_version"],
                "generation": row["generation"],
                "sample_ids": json.loads(row["sample_ids"]),
                "samples_digest": None,
                "parameter_digest": None,
                "loss_json": None,
                "loss_digest": None,
                "duration_ms": None,
                "checkpoint_id": row["checkpoint_id"],
                "checkpoint_status": "unknown",
                "lease_id": row["lease_id"],
                "lease_version": row["lease_version"],
                "status": "unknown",
            }
            digest = _batch_content(payload)
            self.connection.execute(
                "INSERT OR REPLACE INTO batches "
                "(shard_id,batch_id,seq,parameter_version,generation,sample_ids,"
                " samples_digest,parameter_digest,loss_json,loss_digest,duration_ms,"
                " checkpoint_id,checkpoint_status,lease_id,lease_version,status,"
                " content_digest,updated_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    payload["shard_id"],
                    payload["batch_id"],
                    payload["seq"],
                    payload["parameter_version"],
                    payload["generation"],
                    canonical_json(payload["sample_ids"]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    payload["checkpoint_id"],
                    "unknown",
                    payload["lease_id"],
                    payload["lease_version"],
                    "unknown",
                    digest,
                    updated_ms,
                ),
            )
            self.connection.execute(
                "UPDATE batch_attempts SET status='unknown', finished_ms=?, detail=?, content_digest=? "
                "WHERE attempt_id=?",
                (
                    updated_ms,
                    detail,
                    sha256_value(
                        {
                            "kind": "attempt-unknown",
                            "attempt_id": row["attempt_id"],
                            "detail": detail,
                            "finished_ms": updated_ms,
                        }
                    ),
                    row["attempt_id"],
                ),
            )
            self.connection.execute(
                "UPDATE leases SET status=?, content_digest=? WHERE lease_id=?",
                (
                    lease_status,
                    sha256_value(
                        {
                            "kind": "lease",
                            "lease_id": row["lease_id"],
                            "status": lease_status,
                        }
                    ),
                    row["lease_id"],
                ),
            )
            recovery_digest = sha256_value(
                {
                    "kind": "recovery-log",
                    "shard_id": row["shard_id"],
                    "batch_id": row["batch_id"],
                    "status": "unknown",
                    "lease_id": row["lease_id"],
                    "detail": detail,
                    "updated_ms": updated_ms,
                }
            )
            self.connection.execute(
                "INSERT INTO recovery_log "
                "(shard_id,batch_id,status,lease_id,detail,content_digest,updated_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    row["shard_id"],
                    row["batch_id"],
                    "unknown",
                    row["lease_id"],
                    detail,
                    recovery_digest,
                    updated_ms,
                ),
            )
            self.connection.execute(
                "DELETE FROM inflight WHERE shard_id=? AND batch_id=?",
                (row["shard_id"], row["batch_id"]),
            )
            recovered.append(
                {
                    "shard_id": row["shard_id"],
                    "batch_id": row["batch_id"],
                    "lease_id": row["lease_id"],
                    "status": "unknown",
                }
            )
        return recovered

    def recover_inflight(self, detail: str = "process terminated before commit") -> list[dict[str, Any]]:
        with self._transaction():
            rows = self.connection.execute(
                "SELECT * FROM inflight ORDER BY seq, shard_id, batch_id"
            ).fetchall()
            if not rows:
                return []
            return self._mark_unknown_rows(
                list(rows), detail=detail, lease_status="released_recovered"
            )

    def mark_batch_unknown(
        self, shard_id: str, batch_id: str, detail: str
    ) -> dict[str, Any] | None:
        with self._transaction():
            rows = self.connection.execute(
                "SELECT * FROM inflight WHERE shard_id=? AND batch_id=?",
                (shard_id, batch_id),
            ).fetchall()
            if not rows:
                return None
            recovered = self._mark_unknown_rows(
                list(rows), detail=detail, lease_status="released_failed"
            )
            return recovered[0] if recovered else None

    def record_checkpoint(
        self,
        plan: CheckpointPlan,
        *,
        parameter_digests: dict[str, str],
        loss_baseline: list[dict[str, Any]],
        status: str = "committed",
    ) -> dict[str, Any]:
        updated_ms = _clock_ms()
        baseline_digest = sha256_value({"loss_baseline": loss_baseline})
        batch_keys = sorted(plan.batch_keys)
        payload = {
            "checkpoint_id": plan.spec.checkpoint_id,
            "seq": plan.event.seq,
            "parameter_version": plan.spec.parameter_version,
            "generation": plan.spec.generation,
            "parameter_digests": dict(sorted(parameter_digests.items())),
            "batch_keys": [list(key) for key in batch_keys],
            "loss_baseline": loss_baseline,
            "baseline_digest": baseline_digest,
            "status": status,
        }
        digest = sha256_value({"kind": "checkpoint", **payload})
        with self._transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(checkpoint_id,seq,parameter_version,generation,parameter_digests,batch_keys,"
                " loss_baseline,baseline_digest,status,content_digest,updated_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    payload["checkpoint_id"],
                    payload["seq"],
                    payload["parameter_version"],
                    payload["generation"],
                    canonical_json(payload["parameter_digests"]),
                    canonical_json(payload["batch_keys"]),
                    canonical_json(loss_baseline),
                    baseline_digest,
                    status,
                    digest,
                    updated_ms,
                ),
            )
            for shard_id, batch_id in batch_keys:
                self.connection.execute(
                    "UPDATE batches SET checkpoint_status='committed', "
                    "content_digest=? WHERE shard_id=? AND batch_id=?",
                    (
                        sha256_value(
                            {
                                "kind": "batch-checkpoint",
                                "checkpoint_id": plan.spec.checkpoint_id,
                                "shard_id": shard_id,
                                "batch_id": batch_id,
                                "checkpoint_status": "committed",
                            }
                        ),
                        shard_id,
                        batch_id,
                    ),
                )
                self._refresh_batch_digest(shard_id, batch_id)
        return {**payload, "content_digest": digest, "updated_ms": updated_ms}

    def _refresh_batch_digest(self, shard_id: str, batch_id: str) -> None:
        row = self.connection.execute(
            "SELECT * FROM batches WHERE shard_id=? AND batch_id=?",
            (shard_id, batch_id),
        ).fetchone()
        if row is None:
            return
        payload = {
            "shard_id": row["shard_id"],
            "batch_id": row["batch_id"],
            "seq": row["seq"],
            "parameter_version": row["parameter_version"],
            "generation": row["generation"],
            "sample_ids": json.loads(row["sample_ids"]),
            "samples_digest": row["samples_digest"],
            "parameter_digest": row["parameter_digest"],
            "loss_json": _json_load(row["loss_json"], None),
            "loss_digest": row["loss_digest"],
            "duration_ms": row["duration_ms"],
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_status": row["checkpoint_status"],
            "lease_id": row["lease_id"],
            "lease_version": row["lease_version"],
            "status": row["status"],
        }
        self.connection.execute(
            "UPDATE batches SET content_digest=? WHERE shard_id=? AND batch_id=?",
            (_batch_content(payload), shard_id, batch_id),
        )

    def record_rollback(
        self,
        plan: RollbackPlan,
        *,
        current_versions: dict[str, int],
    ) -> dict[str, Any]:
        updated_ms = _clock_ms()
        payload = plan.to_dict()
        digest = sha256_value({"kind": "rollback", **payload})
        with self._transaction():
            self.connection.execute(
                "INSERT OR IGNORE INTO rollbacks "
                "(seq,checkpoint_id,target_version,target_generation,revoked_versions,"
                " revoked_checkpoints,affected_batches,generation_before,generation_after,"
                " content_digest,updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan.event.seq,
                    plan.checkpoint_id,
                    plan.target_version,
                    plan.target_generation,
                    canonical_json(payload["revoked_versions"]),
                    canonical_json(payload["revoked_checkpoints"]),
                    canonical_json(payload["affected_committed_batches"]),
                    plan.generation_before,
                    plan.generation_after,
                    digest,
                    updated_ms,
                ),
            )
            for checkpoint_id in plan.revoked_checkpoints:
                self.connection.execute(
                    "UPDATE checkpoints SET status='rolled_back', content_digest=? "
                    "WHERE checkpoint_id=?",
                    (
                        sha256_value(
                            {
                                "kind": "checkpoint-status",
                                "checkpoint_id": checkpoint_id,
                                "status": "rolled_back",
                            }
                        ),
                        checkpoint_id,
                    ),
                )
            for shard_id, batch_id in plan.affected_committed_batches:
                self.connection.execute(
                    "UPDATE batches SET checkpoint_status='rolled_back' "
                    "WHERE shard_id=? AND batch_id=?",
                    (shard_id, batch_id),
                )
                self._refresh_batch_digest(shard_id, batch_id)
            self.connection.execute(
                "INSERT INTO meta (key,value) VALUES ('current_versions',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (canonical_json(dict(sorted(current_versions.items()))),),
            )
            self.connection.execute(
                "INSERT INTO meta (key,value) VALUES ('generation_counter',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(plan.generation_after),),
            )
        return {**payload, "content_digest": digest, "updated_ms": updated_ms}

    def record_lease_renewal(
        self,
        *,
        lease_id: str,
        shard_id: str,
        batch_id: str,
        old_version: int,
        new_version: int,
        expires_ms: int,
    ) -> str:
        updated_ms = _clock_ms()
        payload = {
            "lease_id": lease_id,
            "shard_id": shard_id,
            "batch_id": batch_id,
            "old_version": old_version,
            "new_version": new_version,
            "expires_ms": expires_ms,
            "updated_ms": updated_ms,
        }
        digest = sha256_value({"kind": "lease-renewal", **payload})
        with self._transaction():
            self.connection.execute(
                "INSERT OR IGNORE INTO lease_renewals "
                "(lease_id,shard_id,batch_id,old_version,new_version,expires_ms,"
                " content_digest,updated_ms) VALUES (?,?,?,?,?,?,?,?)",
                (
                    lease_id,
                    shard_id,
                    batch_id,
                    old_version,
                    new_version,
                    expires_ms,
                    digest,
                    updated_ms,
                ),
            )
            self.connection.execute(
                "UPDATE leases SET version=?, expires_ms=?, content_digest=? WHERE lease_id=?",
                (
                    new_version,
                    expires_ms,
                    sha256_value(
                        {
                            "kind": "lease-renewed",
                            "lease_id": lease_id,
                            "version": new_version,
                            "expires_ms": expires_ms,
                        }
                    ),
                    lease_id,
                ),
            )
        return digest

    def lease_renewal_exists(
        self,
        *,
        lease_id: str,
        old_version: int,
        new_version: int,
    ) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM lease_renewals "
                "WHERE lease_id=? AND old_version=? AND new_version=? LIMIT 1",
                (lease_id, old_version, new_version),
            ).fetchone()
        return row is not None

    def record_advance(
        self,
        *,
        seq: int,
        parameter_version: int,
        generation: int,
        current_versions: dict[str, int],
    ) -> None:
        payload = {
            "kind": "advance",
            "seq": seq,
            "parameter_version": parameter_version,
            "generation": generation,
            "current_versions": dict(sorted(current_versions.items())),
        }
        with self._transaction():
            self.connection.execute(
                "INSERT OR IGNORE INTO recovery_log "
                "(shard_id,batch_id,status,lease_id,detail,content_digest,updated_ms) "
                "VALUES ('*','*','advance',NULL,?,?,?)",
                (canonical_json(payload), sha256_value(payload), _clock_ms()),
            )
            self.connection.execute(
                "INSERT INTO meta (key,value) VALUES ('current_versions',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (canonical_json(dict(sorted(current_versions.items()))),),
            )
            current_generation = self.connection.execute(
                "SELECT value FROM meta WHERE key='generation_counter'"
            ).fetchone()
            if current_generation is None or int(current_generation["value"]) < generation:
                self.connection.execute(
                    "INSERT INTO meta (key,value) VALUES ('generation_counter',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(generation),),
                )

    def record_impact(self, report: dict[str, Any]) -> str:
        digest = sha256_value(report)
        with self._transaction():
            self.connection.execute(
                "INSERT INTO impact_runs (report_json,report_digest,created_ms) VALUES (?,?,?)",
                (canonical_json(report), digest, _clock_ms()),
            )
        return digest

    def latest_impact(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT report_json, report_digest FROM impact_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "report": json.loads(row["report_json"]),
            "report_digest": row["report_digest"],
        }

    def get_batch(self, shard_id: str, batch_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM batches WHERE shard_id=? AND batch_id=?",
                (shard_id, batch_id),
            ).fetchone()
        return self._decode_batch(row) if row is not None else None

    def _decode_batch(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "shard_id": row["shard_id"],
            "batch_id": row["batch_id"],
            "seq": row["seq"],
            "parameter_version": row["parameter_version"],
            "generation": row["generation"],
            "sample_ids": json.loads(row["sample_ids"]),
            "samples_digest": row["samples_digest"],
            "parameter_digest": row["parameter_digest"],
            "loss": _json_load(row["loss_json"], None),
            "loss_digest": row["loss_digest"],
            "duration_ms": row["duration_ms"],
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_status": row["checkpoint_status"],
            "lease_id": row["lease_id"],
            "lease_version": row["lease_version"],
            "status": row["status"],
            "content_digest": row["content_digest"],
            "updated_ms": row["updated_ms"],
        }

    def list_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batches ORDER BY shard_id, seq, batch_id"
            ).fetchall()
        return [self._decode_batch(row) for row in rows]

    def list_batches_by_seq(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batches ORDER BY seq, shard_id, batch_id"
            ).fetchall()
        return [self._decode_batch(row) for row in rows]

    def list_checkpoints(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM checkpoints ORDER BY generation, checkpoint_id"
            ).fetchall()
        return [
            {
                "checkpoint_id": row["checkpoint_id"],
                "seq": row["seq"],
                "parameter_version": row["parameter_version"],
                "generation": row["generation"],
                "parameter_digests": json.loads(row["parameter_digests"]),
                "batch_keys": [tuple(item) for item in json.loads(row["batch_keys"])],
                "loss_baseline": json.loads(row["loss_baseline"]),
                "baseline_digest": row["baseline_digest"],
                "status": row["status"],
                "content_digest": row["content_digest"],
                "updated_ms": row["updated_ms"],
            }
            for row in rows
        ]

    def list_rollbacks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM rollbacks ORDER BY seq, id"
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "checkpoint_id": row["checkpoint_id"],
                "target_version": row["target_version"],
                "target_generation": row["target_generation"],
                "revoked_versions": json.loads(row["revoked_versions"]),
                "revoked_checkpoints": json.loads(row["revoked_checkpoints"]),
                "affected_committed_batches": json.loads(row["affected_batches"]),
                "generation_before": row["generation_before"],
                "generation_after": row["generation_after"],
                "content_digest": row["content_digest"],
            }
            for row in rows
        ]

    def list_attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batch_attempts ORDER BY started_ms, attempt_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_recovery(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM recovery_log ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_inflight(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute("SELECT * FROM inflight ORDER BY seq").fetchall()
        return [dict(row) for row in rows]

    def read_meta(self) -> dict[str, str]:
        with self._lock:
            rows = self.connection.execute("SELECT key,value FROM meta ORDER BY key").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def open_readonly(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    if not target.exists():
        raise CkptError("E_VERIFY", f"ledger {target} does not exist", field="ledger")
    connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection
