"""SQLite experiment ledger: persistence, recovery and whole-snapshot replace.

Commit boundary is one SQLite transaction per completed injection containing
the injection row, every request record and the progress cursor, so a hard
kill can only leave fully committed work or nothing at all. An injection left
in ``running`` state is reclassified to ``unknown`` on reopen and must rerun.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import LedgerError
from .spec import ExperimentSpec, canonical_json_bytes
from .timeline import TimelineEntry, timeline_digest

LEDGER_FORMAT_VERSION = 1
LEDGER_FILENAME = "ledger.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._create_tables()

    # ------------------------------------------------------------------ setup
    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                ledger_format_version INTEGER NOT NULL,
                spec_version INTEGER NOT NULL,
                experiment_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                spec_digest TEXT NOT NULL,
                timeline_digest TEXT NOT NULL,
                canonical_spec TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS injections (
                idx INTEGER PRIMARY KEY,
                start_offset_ms INTEGER NOT NULL,
                target_id TEXT NOT NULL,
                injection_id TEXT NOT NULL,
                injection_type TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                status TEXT NOT NULL,
                observed_state TEXT,
                commited_at TEXT
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                injection_idx INTEGER NOT NULL,
                phase TEXT NOT NULL,
                sample_interval_ms INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                FOREIGN KEY (injection_idx) REFERENCES injections(idx)
            );
            CREATE TABLE IF NOT EXISTS request_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                injection_idx INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                phase TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS replacements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                replaced_at TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                old_format_version INTEGER,
                new_format_version INTEGER NOT NULL,
                old_spec_digest TEXT,
                new_spec_digest TEXT NOT NULL,
                skipped_injections INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_committed_idx INTEGER,
                total_injections INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------- experiment
    def initialize_experiment(self, spec: ExperimentSpec, entries: List[TimelineEntry]) -> None:
        existing = self.conn.execute("SELECT 1 FROM experiment WHERE id = 1").fetchone()
        if existing is not None:
            raise LedgerError("ledger already initialized")
        with self.conn:
            self.conn.execute(
                "INSERT INTO experiment VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    1,
                    LEDGER_FORMAT_VERSION,
                    spec.spec_version,
                    spec.experiment_id,
                    spec.seed,
                    spec.spec_digest,
                    timeline_digest(entries),
                    spec.canonical_bytes.decode("utf-8"),
                    _utc_now(),
                ),
            )
            self.conn.execute(
                "INSERT INTO progress (id, last_committed_idx, total_injections, updated_at) "
                "VALUES (1, NULL, ?, ?)",
                (len(entries), _utc_now()),
            )
            for entry in entries:
                self.conn.execute(
                    "INSERT INTO injections (idx, start_offset_ms, target_id, injection_id, "
                    "injection_type, duration_ms, params_json, status) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        entry.index,
                        entry.start_offset_ms,
                        entry.target_id,
                        entry.injection_id,
                        entry.injection_type,
                        entry.duration_ms,
                        canonical_json_bytes(entry.params).decode("utf-8"),
                        "pending",
                    ),
                )

    def experiment_row(self) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM experiment WHERE id = 1").fetchone()

    # -------------------------------------------------------------- digesting
    def content_digest(self) -> str:
        hasher = hashlib.sha256()
        for table in ("injections", "observations", "request_records", "progress", "replacements"):
            hasher.update(table.encode("utf-8"))
            rows = self.conn.execute(
                f"SELECT * FROM {table} ORDER BY 1"
            ).fetchall()
            for row in rows:
                hasher.update(canonical_json_bytes(dict(row)))
        return hasher.hexdigest()

    def verify_content_digest(self, expected: str) -> None:
        actual = self.content_digest()
        if actual != expected:
            raise LedgerError(
                "ledger content digest mismatch; file is corrupt or externally modified",
                details={"expected": expected, "actual": actual},
            )

    def stored_content_digest(self) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'content_digest'").fetchone()
        return row["value"] if row else None

    def refresh_content_digest(self) -> str:
        digest = self.content_digest()
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES ('content_digest', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (digest,),
        )
        return digest

    # ---------------------------------------------------------------- commits
    def mark_running(self, idx: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE injections SET status = 'running' WHERE idx = ?", (idx,))
            # Every state-changing transaction must refresh the stored digest,
            # otherwise a hard kill between mark_running and the commit would
            # leave a durable ledger whose digest does not match its content.
            self.refresh_content_digest()

    def commit_injection(self, idx: int, observed_state: str,
                         records_by_phase: Dict[str, List[Any]],
                         summaries: List[Dict[str, Any]],
                         sample_interval_ms: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE injections SET status = 'committed', observed_state = ?, commited_at = ? "
                "WHERE idx = ?",
                (observed_state, _utc_now(), idx),
            )
            for summary in summaries:
                self.conn.execute(
                    "INSERT INTO observations (injection_idx, phase, sample_interval_ms, "
                    "sample_count, summary_json) VALUES (?,?,?,?,?)",
                    (
                        idx,
                        summary["phase"],
                        sample_interval_ms,
                        summary["sample_count"],
                        canonical_json_bytes(summary).decode("utf-8"),
                    ),
                )
            for phase, records in records_by_phase.items():
                self.conn.executemany(
                    "INSERT INTO request_records (injection_idx, timestamp_ms, phase, success, "
                    "latency_ms, error) VALUES (?,?,?,?,?,?)",
                    [
                        (idx, record.timestamp_ms, phase, 1 if record.success else 0,
                         record.latency_ms, record.error)
                        for record in records
                    ],
                )
            self.conn.execute(
                "UPDATE progress SET last_committed_idx = ?, updated_at = ? WHERE id = 1",
                (idx, _utc_now()),
            )
            self.refresh_content_digest()

    # --------------------------------------------------------------- recovery
    def mark_unknown_in_flight(self) -> List[int]:
        rows = self.conn.execute(
            "SELECT idx FROM injections WHERE status = 'running' ORDER BY idx"
        ).fetchall()
        unknown_indexes = [row["idx"] for row in rows]
        if unknown_indexes:
            with self.conn:
                self.conn.execute(
                    "UPDATE injections SET status = 'unknown', observed_state = 'unknown' "
                    "WHERE status = 'running'"
                )
        return unknown_indexes

    def recovery_report(self, spec: ExperimentSpec) -> Dict[str, Any]:
        committed = self.conn.execute(
            "SELECT COUNT(*) AS c FROM injections WHERE status = 'committed'"
        ).fetchone()["c"]
        unknown = [
            row["idx"]
            for row in self.conn.execute(
                "SELECT idx FROM injections WHERE status = 'unknown' ORDER BY idx"
            ).fetchall()
        ]
        pending = [
            row["idx"]
            for row in self.conn.execute(
                "SELECT idx FROM injections WHERE status IN ('pending','unknown') ORDER BY idx"
            ).fetchall()
        ]
        total = self.conn.execute("SELECT total_injections FROM progress WHERE id = 1").fetchone()[
            "total_injections"
        ]
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "spec_version": spec.spec_version,
            "experiment_id": spec.experiment_id,
            "recovered_committed": committed,
            "unknown_indexes": unknown,
            "pending_indexes": pending,
            "remaining_budget": spec.budget.max_injections - committed,
            "total_injections": total,
            "replacement_events": self.list_replacements(),
            "exit_report_promised": False,
        }

    def list_replacements(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM replacements ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def record_replacement(self, backup_path: Path, old_row: Optional[sqlite3.Row],
                           spec: ExperimentSpec, skipped: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO replacements (replaced_at, backup_path, old_format_version, "
                "new_format_version, old_spec_digest, new_spec_digest, skipped_injections) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    _utc_now(),
                    str(backup_path),
                    old_row["ledger_format_version"] if old_row is not None else None,
                    LEDGER_FORMAT_VERSION,
                    old_row["spec_digest"] if old_row is not None else None,
                    spec.spec_digest,
                    skipped,
                ),
            )

    # ------------------------------------------------------------- snapshots
    def committed_indexes(self) -> List[int]:
        return [
            row["idx"]
            for row in self.conn.execute(
                "SELECT idx FROM injections WHERE status = 'committed' ORDER BY idx"
            ).fetchall()
        ]

    def injection_row(self, idx: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM injections WHERE idx = ?", (idx,)).fetchone()

    def observation_rows(self, idx: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM observations WHERE injection_idx = ? ORDER BY id", (idx,)
        ).fetchall()

    def request_count(self, idx: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS c FROM request_records WHERE injection_idx = ?", (idx,)
        ).fetchone()["c"]

    def close(self) -> None:
        self.conn.close()


def _make_readonly_backup(db_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"ledger.backup-{timestamp}.sqlite3")
    shutil.copy2(db_path, backup)
    os.chmod(backup, 0o444)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar_backup = Path(str(backup) + suffix)
            shutil.copy2(sidecar, sidecar_backup)
            os.chmod(sidecar_backup, 0o444)
    return backup


def open_or_replace_ledger(db_path: str | Path, spec: ExperimentSpec,
                           entries: List[TimelineEntry],
                           expected_content_digest: Optional[str] = None) -> tuple[Ledger, Dict[str, Any]]:
    """Open an existing ledger or replace it wholesale when identity differs.

    Whole-snapshot replacement (never incremental merge) is used whenever the
    spec version, seed or spec digest recorded in the snapshot does not match
    the current session. The old ledger is kept as a read-only backup and the
    replacement is recorded with old/new versions and skipped injection count.
    """
    db_path = Path(db_path)
    event: Dict[str, Any] = {"action": "opened", "replaced": False}

    if db_path.exists():
        probe = sqlite3.connect(str(db_path))
        probe.row_factory = sqlite3.Row
        try:
            tables = {
                row[0]
                for row in probe.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "experiment" not in tables:
                raise LedgerError("ledger file is missing the experiment table")
            old_row = probe.execute("SELECT * FROM experiment WHERE id = 1").fetchone()
            if old_row is None:
                raise LedgerError("ledger has no experiment row")
            if old_row["ledger_format_version"] != LEDGER_FORMAT_VERSION:
                identity_mismatch = True
                mismatch_basis = "ledger_format_version"
            elif old_row["spec_version"] != spec.spec_version or old_row["seed"] != spec.seed \
                    or old_row["spec_digest"] != spec.spec_digest:
                identity_mismatch = True
                mismatch_basis = "spec_version|seed|spec_digest"
            else:
                identity_mismatch = False
                mismatch_basis = None

            skipped = probe.execute(
                "SELECT COUNT(*) AS c FROM injections WHERE status = 'committed'"
            ).fetchone()["c"]
        finally:
            probe.close()

        if identity_mismatch:
            backup = _make_readonly_backup(db_path)
            for sidecar in ("-wal", "-shm"):
                Path(str(db_path) + sidecar).unlink(missing_ok=True)
            db_path.unlink()
            ledger = Ledger(db_path)
            ledger.initialize_experiment(spec, entries)
            ledger.record_replacement(backup, old_row, spec, skipped)
            event = {
                "action": "replaced",
                "replaced": True,
                "basis": mismatch_basis,
                "backup_path": str(backup),
                "old_format_version": old_row["ledger_format_version"],
                "new_format_version": LEDGER_FORMAT_VERSION,
                "skipped_injections": skipped,
            }
        else:
            ledger = Ledger(db_path)
            stored = ledger.stored_content_digest()
            if stored is not None:
                ledger.verify_content_digest(stored)
            # Reclassify in-flight work only after the on-disk content has been
            # verified; marking unknown itself changes content, so it must not
            # happen before the digest check.
            unknown = ledger.mark_unknown_in_flight()
            event = {"action": "opened", "replaced": False, "unknown_indexes": unknown}
    else:
        ledger = Ledger(db_path)
        ledger.initialize_experiment(spec, entries)

    return ledger, event
