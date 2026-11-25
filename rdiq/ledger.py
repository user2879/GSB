"""SQLite ledger with versioned atomic batches and whole-snapshot recovery.

Round arbitration, error attribution and progress cursors are inserted in one
SQLite transaction per batch. A process killed before ``COMMIT`` leaves no
partial batch because SQLite rolls the transaction back automatically; such
rounds are reported as unknown and must be recomputed.
"""

import os
import shutil
import sqlite3

from .canonical import canonical_text, digest
from .errors import LedgerError, ReplayError
from .fixedpoint import FORMAT_VERSION


SCHEMA = [
    """
    CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE batches (
        batch_seq INTEGER PRIMARY KEY,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        committed_at_order INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE round_records (
        session_id TEXT NOT NULL,
        round_index INTEGER NOT NULL,
        batch_seq INTEGER NOT NULL,
        round_json TEXT NOT NULL,
        error_json TEXT NOT NULL,
        round_digest TEXT NOT NULL,
        PRIMARY KEY (session_id, round_index)
    )
    """,
    """
    CREATE TABLE cursors (
        session_id TEXT PRIMARY KEY,
        last_committed_round INTEGER NOT NULL,
        total_rounds INTEGER NOT NULL
    )
    """,
]


class Ledger:
    def __init__(self, db_path, backup_dir, expected):
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.expected = expected
        self.replacement = None
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        replaced = self._open_or_replace()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        if replaced:
            self._initialize_meta()
        self._assert_meta()

    def _open_or_replace(self):
        if not os.path.exists(self.db_path):
            return True
        try:
            probe = sqlite3.connect(db_path_safe(self.db_path))
            probe.row_factory = sqlite3.Row
            rows = probe.execute("SELECT key, value FROM meta").fetchall()
            probe.close()
        except sqlite3.DatabaseError as exc:
            return self._replace(f"ledger is unreadable: {exc}")
        meta = {row["key"]: row["value"] for row in rows}
        differences = self._meta_differences(meta)
        if differences:
            return self._replace("version or content digest mismatch", differences)
        return False

    def _meta_differences(self, meta):
        wanted = self._expected_meta()
        return [
            {
                "key": key,
                "on_disk": meta.get(key),
                "expected": wanted[key],
            }
            for key in sorted(wanted)
            if meta.get(key) != wanted[key]
        ]

    def _replace(self, reason, differences=None):
        reason_token = digest({"reason": reason, "path": self.db_path})[:12]
        sequence = 1
        while True:
            backup_path = os.path.join(
                self.backup_dir,
                f"ledger-backup-{reason_token}-{sequence:03d}.sqlite",
            )
            if not os.path.exists(backup_path):
                break
            sequence += 1
        shutil.copy2(self.db_path, backup_path)
        os.chmod(backup_path, 0o444)
        for suffix in ("-wal", "-shm"):
            sidecar = self.db_path + suffix
            if os.path.exists(sidecar):
                side_backup = backup_path + suffix
                shutil.copy2(sidecar, side_backup)
                os.chmod(side_backup, 0o444)
        os.remove(self.db_path)
        for suffix in ("-wal", "-shm"):
            sidecar = self.db_path + suffix
            if os.path.exists(sidecar):
                os.remove(sidecar)
        self.replacement = {
            "reason": reason,
            "differences": differences or [],
            "read_only_backup": backup_path,
            "mode": "whole-snapshot-replace",
        }
        return True

    def _expected_meta(self):
        return {
            "format_version": str(FORMAT_VERSION),
            "rule_version": str(self.expected["rule_version"]),
            "weight_version": str(self.expected["weight_version"]),
            "precision_version": str(self.expected["precision_version"]),
            "content_digest": self.expected["content_digest"],
        }

    def _initialize_meta(self):
        conn = sqlite3.connect(self.db_path)
        try:
            for statement in SCHEMA:
                conn.execute(statement)
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                sorted(self._expected_meta().items()),
            )
            conn.commit()
        finally:
            conn.close()

    def _assert_meta(self):
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {row["key"]: row["value"] for row in rows}
        differences = self._meta_differences(meta)
        if differences:
            raise LedgerError(
                "ledger metadata failed after initialization",
                field="meta",
                differences=differences,
            )

    def next_batch_seq(self):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(batch_seq), 0) AS seq FROM batches"
        ).fetchone()
        return int(row["seq"]) + 1

    def write_batch(self, entries, cursors, batch_content):
        batch_seq = self.next_batch_seq()
        content_digest = digest(batch_content)
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?)",
                (batch_seq, FORMAT_VERSION, content_digest, batch_seq),
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO round_records
                (session_id, round_index, batch_seq, round_json, error_json,
                 round_digest)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry["session_id"],
                        entry["round"],
                        batch_seq,
                        canonical_text(entry["round_record"]),
                        canonical_text(entry["error_record"]),
                        entry["round_record"]["round_digest"],
                    )
                    for entry in entries
                ],
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO cursors
                (session_id, last_committed_round, total_rounds)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        cursor["session_id"],
                        cursor["last_committed_round"],
                        cursor["total_rounds"],
                    )
                    for cursor in cursors
                ],
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return {"batch_seq": batch_seq, "content_digest": content_digest}

    def begin_interrupted_batch(self, entries, cursors, batch_content):
        """Open a transaction without committing, used by crash tests."""
        batch_seq = self.next_batch_seq()
        content_digest = digest(batch_content)
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute(
            "INSERT INTO batches VALUES (?, ?, ?, ?)",
            (batch_seq, FORMAT_VERSION, content_digest, batch_seq),
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO round_records
            (session_id, round_index, batch_seq, round_json, error_json,
             round_digest)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry["session_id"],
                    entry["round"],
                    batch_seq,
                    canonical_text(entry["round_record"]),
                    canonical_text(entry["error_record"]),
                    entry["round_record"]["round_digest"],
                )
                for entry in entries
            ],
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO cursors
            (session_id, last_committed_round, total_rounds)
            VALUES (?, ?, ?)
            """,
            [
                (
                    cursor["session_id"],
                    cursor["last_committed_round"],
                    cursor["total_rounds"],
                )
                for cursor in cursors
            ],
        )
        return batch_seq

    def committed_state(self):
        batches = self.conn.execute(
            "SELECT batch_seq, format_version, content_digest FROM batches "
            "ORDER BY batch_seq"
        ).fetchall()
        rows = self.conn.execute(
            "SELECT session_id, round_index, round_json, error_json, round_digest "
            "FROM round_records ORDER BY session_id, round_index"
        ).fetchall()
        cursor_rows = self.conn.execute(
            "SELECT session_id, last_committed_round, total_rounds "
            "FROM cursors ORDER BY session_id"
        ).fetchall()
        return {
            "batches": [dict(row) for row in batches],
            "rounds": [dict(row) for row in rows],
            "cursors": [dict(row) for row in cursor_rows],
        }

    def close(self):
        self.conn.close()


def db_path_safe(path):
    return path


def expected_from_config(config):
    return {
        "rule_version": config.rule["version"],
        "weight_version": config.weights["version"],
        "precision_version": config.precision["version"],
        "content_digest": config.digest,
    }


def recover_unknown_rounds(dataset, state):
    known = {
        (row["session_id"], row["round_index"]) for row in state["rounds"]
    }
    unknown = []
    for session in sorted(dataset.sessions, key=lambda item: item["session_id"]):
        for request in sorted(session["rounds"], key=lambda item: item["round"]):
            key = (session["session_id"], request["round"])
            if key not in known:
                unknown.append(
                    {
                        "session_id": session["session_id"],
                        "round": request["round"],
                        "status": "unknown_rerun_required",
                    }
                )
    return unknown


def verify_replay_round(session_id, request, stored_round, stored_error, actual_round,
                        actual_error):
    field_checks = [
        ("confidence", stored_round["confidence"], actual_round["confidence"]),
        (
            "round_digest",
            stored_round["round_digest"],
            actual_round["round_digest"],
        ),
        (
            "evidence_chain_digest",
            digest(stored_round["evidence_digests"]),
            digest(actual_round["evidence_digests"]),
        ),
        (
            "adopted_canonical",
            canonical_text(stored_round["adopted"]),
            canonical_text(actual_round["adopted"]),
        ),
        (
            "error_attribution",
            canonical_text(stored_error),
            canonical_text(actual_error),
        ),
    ]
    for field, stored_value, actual_value in field_checks:
        if stored_value != actual_value:
            raise ReplayError(
                "recomputed result differs from committed ledger record",
                entry=session_id,
                field=["round", request["round"], field],
                ledger_value=stored_value,
                recomputed_value=actual_value,
            )
