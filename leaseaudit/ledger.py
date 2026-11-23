from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .errors import BusinessError, ErrorCode
from .scenario import Scenario
from .util import (
    atomic_copy,
    canonical_digest,
    canonical_json,
    read_json_strict,
    sha256_bytes,
    sha256_text,
    write_json_atomic,
)


LEDGER_FORMAT_VERSION = 1
GENESIS_DIGEST = sha256_text("leaseaudit-ledger:v1:genesis")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _snapshot_error(message: str, field_path: str | None = None, **details: Any) -> BusinessError:
    return BusinessError(ErrorCode.E_SNAPSHOT, message, field_path=field_path, details=details)


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event_id: str
    event_type: str
    source_id: str | None
    lease_id: str | None
    amount: int | None
    mono_ms: int
    params_version: int
    params_digest: str
    progress_cursor: int
    previous_digest: str
    event_hash: str
    payload: dict
    committed_utc: str

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_id": self.source_id,
            "lease_id": self.lease_id,
            "amount": self.amount,
            "mono_ms": self.mono_ms,
            "params_version": self.params_version,
            "params_digest": self.params_digest,
            "progress_cursor": self.progress_cursor,
            "previous_digest": self.previous_digest,
            "event_hash": self.event_hash,
            "payload": self.payload,
            "committed_utc": self.committed_utc,
        }


class AuditLedger:
    """Append-only SQLite audit ledger with a hash-chained content digest."""

    def __init__(
        self,
        path: str | Path,
        *,
        scenario: Scenario | None = None,
        readonly: bool = False,
        verify: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        self.readonly = readonly
        self._closed = False
        if readonly:
            if not self.path.exists():
                raise _snapshot_error("ledger file does not exist", "$.ledger", path=str(self.path))
            uri = f"{self.path.as_uri()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA query_only=ON")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self._require_schema()
            if verify:
                self.verify_integrity()
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")

        exists = self._schema_exists()
        if not exists:
            if scenario is None:
                raise _snapshot_error("a scenario is required to initialize a ledger", "$.scenario")
            self._initialize_schema(scenario)
        else:
            self._require_schema()
            if verify:
                self.verify_integrity()
            if scenario is not None:
                meta = self.meta()
                if meta["scenario_digest"] != scenario.scenario_digest:
                    raise _snapshot_error(
                        "existing ledger belongs to a different scenario/version",
                        "$.ledger_meta.scenario_digest",
                        expected=meta["scenario_digest"],
                        actual=scenario.scenario_digest,
                    )

    def _schema_exists(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_meta'"
        ).fetchone()
        return row is not None

    def _require_schema(self) -> None:
        if not self._schema_exists():
            raise _snapshot_error("ledger schema is missing", "$.ledger_schema")
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version != LEDGER_FORMAT_VERSION:
            raise _snapshot_error(
                "unsupported SQLite ledger schema version",
                "$.user_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=version,
            )

    def _initialize_schema(self, scenario: Scenario) -> None:
        now = _utc_now()
        params_json = canonical_json(scenario.params_dict())
        scenario_json = canonical_json(scenario.scenario_digest_document())
        statements = [
            """
            CREATE TABLE ledger_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                params_version INTEGER NOT NULL,
                params_digest TEXT NOT NULL,
                scenario_digest TEXT NOT NULL,
                external_digest TEXT NOT NULL,
                scenario_version TEXT NOT NULL,
                params_json TEXT NOT NULL,
                scenario_json TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE events (
                seq INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                source_id TEXT,
                lease_id TEXT,
                amount INTEGER,
                mono_ms INTEGER NOT NULL,
                params_version INTEGER NOT NULL,
                params_digest TEXT NOT NULL,
                progress_cursor INTEGER NOT NULL,
                previous_digest TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                committed_utc TEXT NOT NULL
            )
            """,
            "CREATE INDEX events_mono_idx ON events(mono_ms, seq)",
            "CREATE INDEX events_source_idx ON events(source_id, seq)",
            """
            CREATE TABLE issuance_intents (
                intent_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                mono_ms INTEGER NOT NULL,
                params_version INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING','COMMITTED','UNKNOWN')),
                created_utc TEXT NOT NULL,
                resolved_utc TEXT
            )
            """,
            """
            CREATE TABLE unknown_leases (
                unknown_id TEXT PRIMARY KEY,
                intent_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                mono_ms INTEGER NOT NULL,
                basis TEXT NOT NULL,
                event_seq INTEGER,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(intent_id) REFERENCES issuance_intents(intent_id)
            )
            """,
        ]
        with self._transaction():
            for statement in statements:
                self.conn.execute(statement)
            self.conn.execute(
                """
                INSERT INTO ledger_meta(
                    id, format_version, content_digest, event_count, last_seq,
                    params_version, params_digest, scenario_digest, external_digest,
                    scenario_version, params_json, scenario_json, created_utc, updated_utc
                ) VALUES (1, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    LEDGER_FORMAT_VERSION,
                    GENESIS_DIGEST,
                    scenario.lease_params_version,
                    scenario.params_digest,
                    scenario.scenario_digest,
                    scenario.external_digest,
                    scenario.scenario_version,
                    params_json,
                    scenario_json,
                    now,
                    now,
                ),
            )
            self.conn.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._assert_writable()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def _assert_writable(self) -> None:
        if self.readonly:
            raise BusinessError(ErrorCode.E_LEDGER, "ledger is open read-only")
        if self._closed:
            raise BusinessError(ErrorCode.E_LEDGER, "ledger is closed")

    def meta(self) -> dict:
        row = self.conn.execute("SELECT * FROM ledger_meta WHERE id=1").fetchone()
        if row is None:
            raise _snapshot_error("ledger metadata row is missing", "$.ledger_meta")
        meta = dict(row)
        for field in ("params_json", "scenario_json"):
            try:
                meta[field.replace("_json", "")] = json.loads(meta[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise _snapshot_error(
                    "ledger metadata contains invalid JSON", f"$.ledger_meta.{field}", error=str(exc)
                ) from exc
        return meta

    def next_seq(self) -> int:
        return int(self.meta()["last_seq"]) + 1

    def events(self) -> list[EventRecord]:
        records: list[EventRecord] = []
        for row in self.conn.execute("SELECT * FROM events ORDER BY seq ASC"):
            try:
                payload = json.loads(row["payload_json"])
            except ValueError as exc:
                raise _snapshot_error(
                    "event payload is not valid JSON",
                    f"$.events[{row['seq']}].payload_json",
                    event_seq=row["seq"],
                    error=str(exc),
                ) from exc
            records.append(
                EventRecord(
                    seq=row["seq"],
                    event_id=row["event_id"],
                    event_type=row["event_type"],
                    source_id=row["source_id"],
                    lease_id=row["lease_id"],
                    amount=row["amount"],
                    mono_ms=row["mono_ms"],
                    params_version=row["params_version"],
                    params_digest=row["params_digest"],
                    progress_cursor=row["progress_cursor"],
                    previous_digest=row["previous_digest"],
                    event_hash=row["event_hash"],
                    payload=payload,
                    committed_utc=row["committed_utc"],
                )
            )
        return records

    @staticmethod
    def _event_hash(sequence: int, event_id: str, **fields: Any) -> str:
        return canonical_digest(
            {
                "seq": sequence,
                "event_id": event_id,
                **fields,
            }
        )

    def verify_integrity(self) -> dict:
        meta = self.meta()
        if not isinstance(meta.get("format_version"), int):
            raise _snapshot_error(
                "ledger format_version must be an integer",
                "$.ledger_meta.format_version",
                actual=meta.get("format_version"),
            )
        if meta["format_version"] != LEDGER_FORMAT_VERSION:
            raise _snapshot_error(
                "unsupported ledger format version",
                "$.ledger_meta.format_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=meta["format_version"],
            )
        if canonical_digest(meta["params"]) != meta["params_digest"]:
            raise _snapshot_error(
                "lease parameter digest mismatch",
                "$.ledger_meta.params_digest",
                expected=meta["params_digest"],
                actual=canonical_digest(meta["params"]),
            )
        if canonical_digest(meta["scenario"]) != meta["scenario_digest"]:
            raise _snapshot_error(
                "scenario digest mismatch",
                "$.ledger_meta.scenario_digest",
                expected=meta["scenario_digest"],
                actual=canonical_digest(meta["scenario"]),
            )

        digest = GENESIS_DIGEST
        expected_count = 0
        for row in self.conn.execute("SELECT * FROM events ORDER BY seq ASC"):
            seq = int(row["seq"])
            expected_seq = expected_count + 1
            if seq != expected_seq:
                raise _snapshot_error(
                    "event sequence is not contiguous",
                    f"$.events[{expected_seq}].seq",
                    expected=expected_seq,
                    actual=seq,
                )
            try:
                payload = json.loads(row["payload_json"])
            except ValueError as exc:
                raise _snapshot_error(
                    "event payload is not valid JSON",
                    f"$.events[{seq}].payload_json",
                    event_seq=seq,
                    error=str(exc),
                ) from exc
            expected_hash = self._event_hash(
                seq,
                row["event_id"],
                event_type=row["event_type"],
                source_id=row["source_id"],
                lease_id=row["lease_id"],
                amount=row["amount"],
                mono_ms=row["mono_ms"],
                params_version=row["params_version"],
                params_digest=row["params_digest"],
                progress_cursor=row["progress_cursor"],
                payload=payload,
            )
            if row["event_hash"] != expected_hash:
                raise _snapshot_error(
                    "event content hash mismatch",
                    f"$.events[{seq}].event_hash",
                    event_seq=seq,
                    expected=row["event_hash"],
                    actual=expected_hash,
                )
            if row["previous_digest"] != digest:
                raise _snapshot_error(
                    "event previous digest does not match the ledger chain",
                    f"$.events[{seq}].previous_digest",
                    event_seq=seq,
                    expected=digest,
                    actual=row["previous_digest"],
                )
            if row["progress_cursor"] != seq:
                raise _snapshot_error(
                    "event progress cursor mismatch",
                    f"$.events[{seq}].progress_cursor",
                    event_seq=seq,
                    expected=seq,
                    actual=row["progress_cursor"],
                )
            if row["params_version"] != meta["params_version"]:
                raise _snapshot_error(
                    "event lease parameter version mismatch",
                    f"$.events[{seq}].params_version",
                    event_seq=seq,
                    expected=meta["params_version"],
                    actual=row["params_version"],
                )
            if row["params_digest"] != meta["params_digest"]:
                raise _snapshot_error(
                    "event lease parameter digest mismatch",
                    f"$.events[{seq}].params_digest",
                    event_seq=seq,
                    expected=meta["params_digest"],
                    actual=row["params_digest"],
                )
            digest = sha256_text(digest + expected_hash)
            expected_count += 1

        if meta["event_count"] != expected_count:
            raise _snapshot_error(
                "ledger event_count mismatch",
                "$.ledger_meta.event_count",
                expected=meta["event_count"],
                actual=expected_count,
            )
        if meta["last_seq"] != expected_count:
            raise _snapshot_error(
                "ledger last_seq mismatch",
                "$.ledger_meta.last_seq",
                expected=meta["last_seq"],
                actual=expected_count,
            )
        if meta["content_digest"] != digest:
            raise _snapshot_error(
                "ledger content digest mismatch",
                "$.ledger_meta.content_digest",
                expected=meta["content_digest"],
                actual=digest,
            )
        return {
            "format_version": meta["format_version"],
            "content_digest": digest,
            "event_count": expected_count,
            "last_seq": expected_count,
            "params_version": meta["params_version"],
            "params_digest": meta["params_digest"],
            "scenario_digest": meta["scenario_digest"],
            "external_digest": meta["external_digest"],
        }

    def _append_in_transaction(
        self,
        *,
        event_type: str,
        mono_ms: int,
        source_id: str | None,
        lease_id: str | None,
        amount: int | None,
        payload: Mapping[str, Any],
        intent_id: str | None = None,
    ) -> EventRecord:
        self._assert_writable()
        meta = self.meta()
        seq = int(meta["last_seq"]) + 1
        event_id = f"evt-{seq:08d}"
        progress_cursor = seq
        payload_dict = json.loads(canonical_json(dict(payload)))
        event_hash = self._event_hash(
            seq,
            event_id,
            event_type=event_type,
            source_id=source_id,
            lease_id=lease_id,
            amount=amount,
            mono_ms=mono_ms,
            params_version=meta["params_version"],
            params_digest=meta["params_digest"],
            progress_cursor=progress_cursor,
            payload=payload_dict,
        )
        content_digest = sha256_text(meta["content_digest"] + event_hash)
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO events(
                seq, event_id, event_type, source_id, lease_id, amount, mono_ms,
                params_version, params_digest, progress_cursor, previous_digest,
                event_hash, payload_json, committed_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seq,
                event_id,
                event_type,
                source_id,
                lease_id,
                amount,
                mono_ms,
                meta["params_version"],
                meta["params_digest"],
                progress_cursor,
                meta["content_digest"],
                event_hash,
                canonical_json(payload_dict),
                now,
            ),
        )
        self.conn.execute(
            """
            UPDATE ledger_meta SET content_digest=?, event_count=?, last_seq=?, updated_utc=?
            WHERE id=1
            """,
            (content_digest, seq, seq, now),
        )
        if intent_id is not None:
            cursor = self.conn.execute(
                "UPDATE issuance_intents SET status='COMMITTED', resolved_utc=? "
                "WHERE intent_id=? AND status='PENDING'",
                (now, intent_id),
            )
            if cursor.rowcount != 1:
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "issuance intent is absent or no longer pending",
                    field_path="$.issuance_intents",
                    details={"intent_id": intent_id},
                    event_seq=seq,
                )
        return EventRecord(
            seq=seq,
            event_id=event_id,
            event_type=event_type,
            source_id=source_id,
            lease_id=lease_id,
            amount=amount,
            mono_ms=mono_ms,
            params_version=meta["params_version"],
            params_digest=meta["params_digest"],
            progress_cursor=progress_cursor,
            previous_digest=meta["content_digest"],
            event_hash=event_hash,
            payload=payload_dict,
            committed_utc=now,
        )

    def append_event(
        self,
        *,
        event_type: str,
        mono_ms: int,
        source_id: str | None,
        lease_id: str | None = None,
        amount: int | None = None,
        payload: Mapping[str, Any] | None = None,
        intent_id: str | None = None,
    ) -> EventRecord:
        with self._transaction():
            return self._append_in_transaction(
                event_type=event_type,
                mono_ms=mono_ms,
                source_id=source_id,
                lease_id=lease_id,
                amount=amount,
                payload=payload or {},
                intent_id=intent_id,
            )

    def begin_intent(self, *, source_id: str, amount: int, mono_ms: int) -> str:
        with self._transaction():
            meta = self.meta()
            intent_id = f"intent-{uuid.uuid4().hex}"
            self.conn.execute(
                """
                INSERT INTO issuance_intents(
                    intent_id, source_id, amount, mono_ms, params_version, status, created_utc
                ) VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (intent_id, source_id, amount, mono_ms, meta["params_version"], _utc_now()),
            )
            return intent_id

    def pending_intents(self) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM issuance_intents WHERE status='PENDING' ORDER BY created_utc, intent_id"
            )
        ]

    def resolve_unknown_intent(
        self,
        intent: Mapping[str, Any],
        *,
        mono_ms: int,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        intent_id = str(intent["intent_id"])
        basis = "RECOVERY:committed_intent_without_committed_lease_event"
        with self._transaction():
            row = self.conn.execute(
                "SELECT status FROM issuance_intents WHERE intent_id=?", (intent_id,)
            ).fetchone()
            if row is None or row["status"] != "PENDING":
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "pending intent changed concurrently",
                    field_path="$.issuance_intents",
                    details={"intent_id": intent_id},
                )
            event = self._append_in_transaction(
                event_type="LEASE_UNKNOWN",
                mono_ms=mono_ms,
                source_id=intent["source_id"],
                lease_id=None,
                amount=None,
                payload=payload,
            )
            now = _utc_now()
            self.conn.execute(
                "UPDATE issuance_intents SET status='UNKNOWN', resolved_utc=? WHERE intent_id=?",
                (now, intent_id),
            )
            self.conn.execute(
                """
                INSERT INTO unknown_leases(
                    unknown_id, intent_id, source_id, amount, mono_ms, basis,
                    event_seq, created_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"unknown-{intent_id}",
                    intent_id,
                    intent["source_id"],
                    intent["amount"],
                    intent["mono_ms"],
                    basis,
                    event.seq,
                    now,
                ),
            )
            return event

    def unknown_leases(self) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM unknown_leases ORDER BY event_seq ASC, source_id ASC, unknown_id ASC"
            )
        ]

    def create_snapshot(self, out_dir: str | Path) -> dict:
        self._assert_writable()
        integrity = self.verify_integrity()
        out_root = Path(out_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        snapshot_id = f"snap-{integrity['last_seq']:08d}-{integrity['content_digest'][:12]}"
        snapshot_dir = out_root / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json_strict(manifest_path)
            if manifest.get("content_digest") == integrity["content_digest"]:
                existing_db = snapshot_dir / str(manifest.get("ledger_file", "ledger.sqlite"))
                if not existing_db.is_file():
                    raise _snapshot_error(
                        "existing snapshot database is missing",
                        "$.snapshot.ledger_file",
                        snapshot_id=snapshot_id,
                    )
                actual_file_digest = sha256_bytes(existing_db.read_bytes())
                if actual_file_digest != manifest.get("file_sha256"):
                    raise _snapshot_error(
                        "existing snapshot file SHA-256 mismatch",
                        "$.snapshot.manifest.file_sha256",
                        expected=manifest.get("file_sha256"),
                        actual=actual_file_digest,
                    )
                existing = AuditLedger(existing_db, readonly=True, verify=True)
                try:
                    existing_integrity = existing.verify_integrity()
                finally:
                    existing.close()
                if existing_integrity["content_digest"] != integrity["content_digest"]:
                    raise _snapshot_error(
                        "existing snapshot content digest mismatch",
                        "$.snapshot.manifest.content_digest",
                        expected=integrity["content_digest"],
                        actual=existing_integrity["content_digest"],
                    )
                return manifest

        temp_dir = out_root / f".{snapshot_id}.tmp-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        db_name = "ledger.sqlite"
        temp_db = temp_dir / db_name
        try:
            dest = sqlite3.connect(temp_db, isolation_level=None)
            try:
                self.conn.backup(dest)
                dest.execute("PRAGMA journal_mode=DELETE")
                dest.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                dest.close()
            copied = AuditLedger(temp_db, readonly=True, verify=True)
            try:
                copied_integrity = copied.verify_integrity()
            finally:
                copied.close()
            if copied_integrity["content_digest"] != integrity["content_digest"]:
                raise _snapshot_error(
                    "snapshot content digest differs from live ledger",
                    "$.snapshot.content_digest",
                    expected=integrity["content_digest"],
                    actual=copied_integrity["content_digest"],
                )
            manifest = {
                "tool": "leaseaudit",
                "manifest_version": 1,
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "snapshot_id": snapshot_id,
                "ledger_file": db_name,
                "file_sha256": sha256_bytes(temp_db.read_bytes()),
                **integrity,
                "created_utc": _utc_now(),
            }
            write_json_atomic(temp_dir / "manifest.json", manifest)
            if snapshot_dir.exists():
                # The content-identical snapshot is authoritative; preserve immutable contents.
                import shutil

                shutil.rmtree(temp_dir)
            else:
                os.replace(temp_dir, snapshot_dir)
            return manifest
        except BaseException:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    @classmethod
    def restore_snapshot(
        cls,
        snapshot_dir: str | Path,
        target_ledger: str | Path,
    ) -> dict:
        snapshot_dir = Path(snapshot_dir).resolve()
        target = Path(target_ledger).resolve()
        manifest_path = snapshot_dir / "manifest.json"
        try:
            manifest = read_json_strict(manifest_path)
        except FileNotFoundError as exc:
            raise _snapshot_error("snapshot manifest is missing", "$.snapshot.manifest") from exc
        except Exception as exc:
            raise _snapshot_error("snapshot manifest is unreadable", "$.snapshot.manifest", error=str(exc)) from exc
        if not isinstance(manifest, dict):
            raise _snapshot_error("snapshot manifest must be an object", "$.snapshot.manifest")
        version = manifest.get("ledger_format_version")
        if not isinstance(version, int) or version != LEDGER_FORMAT_VERSION:
            raise _snapshot_error(
                "snapshot ledger format version is unsupported",
                "$.snapshot.manifest.ledger_format_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=version,
            )
        snapshot_file = snapshot_dir / str(manifest.get("ledger_file", "ledger.sqlite"))
        if not snapshot_file.is_file():
            raise _snapshot_error("snapshot database file is missing", "$.snapshot.ledger_file")
        actual_file_digest = sha256_bytes(snapshot_file.read_bytes())
        if actual_file_digest != manifest.get("file_sha256"):
            raise _snapshot_error(
                "snapshot file SHA-256 mismatch",
                "$.snapshot.manifest.file_sha256",
                expected=manifest.get("file_sha256"),
                actual=actual_file_digest,
            )
        verified = cls(snapshot_file, readonly=True, verify=True)
        try:
            snapshot_integrity = verified.verify_integrity()
        finally:
            verified.close()
        if snapshot_integrity["content_digest"] != manifest.get("content_digest"):
            raise _snapshot_error(
                "snapshot content digest mismatch",
                "$.snapshot.manifest.content_digest",
                expected=manifest.get("content_digest"),
                actual=snapshot_integrity["content_digest"],
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_name(target.name + ".pre-restore")
        target_existed = target.exists()
        if target_existed:
            atomic_copy(target, backup)
        temp_target = target.with_name(target.name + f".restore-{uuid.uuid4().hex}.tmp")
        try:
            atomic_copy(snapshot_file, temp_target)
            for suffix in ("-wal", "-shm"):
                stale = Path(str(target) + suffix)
                if stale.exists():
                    stale.unlink()
            os.replace(temp_target, target)
            check = cls(target, readonly=True, verify=True)
            try:
                restored = check.verify_integrity()
            finally:
                check.close()
            if restored["content_digest"] != snapshot_integrity["content_digest"]:
                raise _snapshot_error(
                    "restored ledger content digest mismatch",
                    "$.restore.content_digest",
                    expected=snapshot_integrity["content_digest"],
                    actual=restored["content_digest"],
                )
            return {
                "status": "RESTORED",
                "target": str(target),
                "snapshot_id": manifest.get("snapshot_id"),
                "content_digest": restored["content_digest"],
                "event_count": restored["event_count"],
                "backup": str(backup) if backup.exists() else None,
            }
        except BaseException:
            if temp_target.exists():
                temp_target.unlink()
            if backup.exists():
                os.replace(backup, target)
            elif not target_existed and target.exists():
                target.unlink()
            raise

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __enter__(self) -> "AuditLedger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
