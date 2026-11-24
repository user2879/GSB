"""SQLite diagnostic ledger with snapshot recovery semantics.

Every source-batch write and every replay write happens inside a single
SQLite transaction, so a hard kill can never leave a half-written record.
The ledger carries an integer format version and a running SHA-256
content digest; reopening a snapshot with an incompatible version or a
mismatched digest replaces the database wholesale and keeps a read-only
backup.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from .errors import ReplayMismatchError
from .models import canonical_json

LEDGER_FORMAT_VERSION = 1
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_batches (
    scheme_version INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS merge_conclusions (
    scheme_version INTEGER NOT NULL,
    query_id INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (scheme_version, query_id)
);
CREATE TABLE IF NOT EXISTS error_bounds (
    scheme_version INTEGER NOT NULL,
    query_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (scheme_version, query_id, metric)
);
CREATE TABLE IF NOT EXISTS replay_jobs (
    scheme_version INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    rewritten_sql TEXT NOT NULL,
    status TEXT NOT NULL,
    ack_json TEXT,
    digest TEXT,
    queued_at_ms INTEGER,
    committed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (scheme_version, candidate_id, source_id)
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    scheme_version INTEGER PRIMARY KEY,
    last_source_id TEXT,
    last_candidate_id TEXT,
    updated_at_ms INTEGER NOT NULL
);
"""


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _plain(payload: Any) -> Any:
    """Recursively convert dataclasses/tuples into JSON-friendly objects."""
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {key: _plain(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_plain(value) for value in payload]
    return payload


class DiagnosticLedger:
    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        self.conn: sqlite3.Connection | None = None
        self.scheme_version = 0

    # ------------------------------------------------------------------
    def open(self) -> "DiagnosticLedger":
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        fresh = not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0
        self.conn = sqlite3.connect(self.db_path)
        # Use rollback-journal (DELETE) mode: a hard kill simply discards
        # the uncommitted transaction, while committed rows live in the
        # main database file.  This keeps the reopen digest recomputation
        # independent of leftover -wal/-shm sidecars.
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if fresh:
            self._initialize_schema()
        else:
            self._verify_or_replace()
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _initialize_schema(self) -> None:
        assert self.conn is not None
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT INTO ledger_meta(key, value) VALUES (?, ?)",
            ("format_version", str(LEDGER_FORMAT_VERSION)),
        )
        self.conn.execute(
            "INSERT INTO ledger_meta(key, value) VALUES (?, ?)",
            ("content_digest", hashlib.sha256(b"").hexdigest()),
        )
        self.conn.commit()

    def _verify_or_replace(self) -> None:
        assert self.conn is not None
        try:
            version_row = self.conn.execute(
                "SELECT value FROM ledger_meta WHERE key='format_version'"
            ).fetchone()
            digest_row = self.conn.execute(
                "SELECT value FROM ledger_meta WHERE key='content_digest'"
            ).fetchone()
        except sqlite3.DatabaseError:
            version_row = digest_row = None

        if version_row is None or digest_row is None:
            self._replace_untrusted_snapshot("missing ledger_meta")
            return
        try:
            on_disk_version = int(version_row[0])
        except (TypeError, ValueError):
            self._replace_untrusted_snapshot("non-integer format_version")
            return
        if on_disk_version != LEDGER_FORMAT_VERSION:
            self._replace_untrusted_snapshot(
                f"format_version {on_disk_version} != {LEDGER_FORMAT_VERSION}"
            )
            return
        recomputed = self._recompute_digest()
        if recomputed != digest_row[0]:
            self._replace_untrusted_snapshot("content digest mismatch")

    def _replace_untrusted_snapshot(self, reason: str) -> None:
        assert self.conn is not None
        self.conn.close()
        backup_path = self.db_path + ".readonly.bak"
        shutil.copy2(self.db_path, backup_path)
        try:
            os.chmod(backup_path, 0o444)
        except OSError:
            pass
        os.remove(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self._initialize_schema()
        self.replacement_reason = reason
        self.backup_path = backup_path

    def _recompute_digest(self) -> str:
        assert self.conn is not None
        hasher = hashlib.sha256()
        for table in (
            "source_batches", "merge_conclusions", "error_bounds",
            "replay_jobs", "progress_cursor",
        ):
            for row in self.conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            ):
                hasher.update(canonical_json(list(row)))
        return hasher.hexdigest()

    def _chain_digest(self, payload: Any) -> str:
        """Audit-only per-write hash over previous digest + payload.

        The authoritative ``content_digest`` in ``ledger_meta`` is
        recomputed from the full table set inside the same transaction via
        :meth:`_commit_content_digest`, so write-time and reopen-time use
        one identical calculation.
        """
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key='content_digest'"
        ).fetchone()
        previous = row[0].encode("ascii")
        return hashlib.sha256(
            previous + canonical_json(_plain(payload))
        ).hexdigest()

    def _commit_content_digest(self) -> None:
        assert self.conn is not None
        digest = self._recompute_digest()
        self.conn.execute(
            "UPDATE ledger_meta SET value=? WHERE key='content_digest'",
            (digest,),
        )

    # ------------------------------------------------------------------
    def begin_scheme(self, scheme_version: int) -> None:
        self.scheme_version = scheme_version

    def write_source_batch(self, batch, evidence: dict) -> None:
        """Atomic: normalized summary + progress cursor in one transaction."""
        assert self.conn is not None
        payload = {
            "records": [record.to_dict() for record in batch.records],
            "evidence": {
                str(query_id): _plain(evidence[query_id])
                for query_id in sorted(evidence)
            },
        }
        now = _now_ms()
        with self.conn:
            digest = self._chain_digest(_plain(payload))
            self.conn.execute(
                """INSERT OR REPLACE INTO source_batches
                   (scheme_version, source_id, source_type, format_version,
                    record_count, summary_json, digest, committed_at_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.scheme_version, batch.source.source_id,
                    batch.source.source_type, batch.source.format_version,
                    len(batch.records),
                    canonical_json(_plain(payload)).decode("utf-8"),
                    digest, now,
                ),
            )
            self._upsert_cursor(last_source=batch.source.source_id)
            self._commit_content_digest()

    def write_conclusions(self, verdicts: dict, bounds: list) -> None:
        assert self.conn is not None
        now = _now_ms()
        with self.conn:
            for query_id in sorted(verdicts):
                verdict = verdicts[query_id]
                payload = {
                    "query_id": query_id,
                    "latency": _plain(verdict.latency),
                    "rows": _plain(verdict.rows),
                    "participating_sources": list(verdict.participating_sources),
                }
                digest = self._chain_digest(_plain(payload))
                self.conn.execute(
                    """INSERT OR REPLACE INTO merge_conclusions
                       (scheme_version, query_id, payload_json, digest,
                        committed_at_ms) VALUES (?, ?, ?, ?, ?)""",
                    (self.scheme_version, query_id,
                     canonical_json(_plain(payload)).decode("utf-8"),
                     digest, now),
                )
            for bound in sorted(bounds, key=lambda b: (b.metric, b.query_id)):
                bound_payload = _plain(bound)
                digest = self._chain_digest(bound_payload)
                self.conn.execute(
                    """INSERT OR REPLACE INTO error_bounds
                       (scheme_version, query_id, metric, payload_json,
                        digest, committed_at_ms) VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.scheme_version, bound.query_id, bound.metric,
                     canonical_json(bound_payload).decode("utf-8"),
                     digest, now),
                )
            self._upsert_cursor()
            self._commit_content_digest()

    def write_replay_intent(self, job) -> None:
        """Persist the dispatch intention before the socket call.

        Because this row is committed separately from the receipt, a hard
        kill during the in-flight replay leaves status ``unknown`` and the
        recovery path reruns that candidate.
        """
        assert self.conn is not None
        now = _now_ms()
        with self.conn:
            existing = self.conn.execute(
                """SELECT status FROM replay_jobs
                   WHERE scheme_version=? AND candidate_id=? AND source_id=?""",
                (self.scheme_version, job.candidate_id, job.source_id),
            ).fetchone()
            if existing is not None and existing[0] == "committed":
                # A committed receipt survived a hard kill; never downgrade
                # it back to unknown because of a re-dispatched intent.
                return
            self.conn.execute(
                """INSERT OR REPLACE INTO replay_jobs
                   (scheme_version, candidate_id, source_id, rewritten_sql,
                    status, ack_json, digest, queued_at_ms, committed_at_ms)
                   VALUES (?, ?, ?, ?, 'unknown', NULL, NULL, ?, ?)""",
                (self.scheme_version, job.candidate_id, job.source_id,
                 job.rewritten_sql, job.queued_at_ms, now),
            )
            self._chain_digest({"intent": [job.candidate_id, job.source_id]})
            self._upsert_cursor(last_candidate=job.candidate_id)
            self._commit_content_digest()

    def write_replay_receipt(self, result) -> None:
        """Atomic receipt commit; failed sockets persist coded status too."""
        assert self.conn is not None
        job = result.job
        now = _now_ms()
        status = "committed" if result.ok else (result.error_code or "failed")
        ack = result.ack
        payload = {
            "candidate_id": job.candidate_id,
            "source_id": job.source_id,
            "status": status,
            "ack": ack,
        }
        with self.conn:
            digest = self._chain_digest(payload)
            self.conn.execute(
                """UPDATE replay_jobs SET status=?, ack_json=?, digest=?,
                       committed_at_ms=?
                   WHERE scheme_version=? AND candidate_id=? AND source_id=?""",
                (status,
                 canonical_json(ack).decode("utf-8") if ack is not None else None,
                 digest, now, self.scheme_version,
                 job.candidate_id, job.source_id),
            )
            self._upsert_cursor(last_candidate=job.candidate_id)
            self._commit_content_digest()

    def _upsert_cursor(self, last_source: str | None = None,
                       last_candidate: str | None = None) -> None:
        assert self.conn is not None
        self.conn.execute(
            """INSERT INTO progress_cursor(scheme_version, last_source_id,
                   last_candidate_id, updated_at_ms)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(scheme_version) DO UPDATE SET
                   last_source_id=COALESCE(excluded.last_source_id,
                       progress_cursor.last_source_id),
                   last_candidate_id=COALESCE(excluded.last_candidate_id,
                       progress_cursor.last_candidate_id),
                   updated_at_ms=excluded.updated_at_ms""",
            (self.scheme_version, last_source, last_candidate, _now_ms()),
        )

    # ------------------------------------------------------------------
    def load_replay_jobs(self, scheme_version: int) -> list[dict]:
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT candidate_id, source_id, rewritten_sql, status, ack_json
               FROM replay_jobs WHERE scheme_version=?
               ORDER BY candidate_id, source_id""",
            (scheme_version,),
        ).fetchall()
        return [
            {
                "candidate_id": row[0],
                "source_id": row[1],
                "rewritten_sql": row[2],
                "status": row[3],
                "ack_json": row[4],
            }
            for row in rows
        ]

    def unknown_replay_jobs(self, scheme_version: int) -> list[dict]:
        return [
            row for row in self.load_replay_jobs(scheme_version)
            if row["status"] == "unknown"
        ]

    def latest_scheme_with_unknown_jobs(self) -> int | None:
        assert self.conn is not None
        row = self.conn.execute(
            """SELECT scheme_version FROM replay_jobs
               WHERE status='unknown'
               ORDER BY scheme_version DESC LIMIT 1"""
        ).fetchone()
        return None if row is None else int(row[0])

    def verify_replay_against_ledger(self, scheme_version: int,
                                     receipts: list[dict]) -> None:
        """Compare fresh replay receipts with committed ledger rows.

        Raises :class:`ReplayMismatchError` (``E_REPLAY``) at the first
        disagreeing query/candidate and field.
        """
        ledger_rows = {
            (row["candidate_id"], row["source_id"]): row
            for row in self.load_replay_jobs(scheme_version)
            if row["status"] == "committed" or row["ack_json"] is not None
        }
        for receipt in sorted(receipts, key=lambda item: (
            item["candidate_id"], item["source_id"]
        )):
            key = (receipt["candidate_id"], receipt["source_id"])
            stored = ledger_rows.get(key)
            if stored is None:
                raise ReplayMismatchError(
                    "replay candidate absent from ledger",
                    query=receipt["candidate_id"], field="existence",
                )
            import json as _json
            stored_ack = _json.loads(stored["ack_json"])
            fresh = receipt["ack"]
            for field in ("est_latency_us", "est_rows_scanned", "source_id",
                          "candidate_id"):
                if stored_ack.get(field) != fresh.get(field):
                    raise ReplayMismatchError(
                        f"field {field!r} diverges: ledger="
                        f"{stored_ack.get(field)!r} replay={fresh.get(field)!r}",
                        query=receipt["candidate_id"], field=field,
                    )

    def current_digest(self) -> str:
        assert self.conn is not None
        return self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key='content_digest'"
        ).fetchone()[0]
