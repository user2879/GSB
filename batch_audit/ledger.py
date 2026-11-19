"""SQLite execution ledger with transactional slot commits and integrity digest."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical import array_payload, canonical_json_bytes, sha256_hex
from .constants import ARRAY_DIGEST_DOMAIN, LEDGER_FORMAT_VERSION
from .timeutil import utc_now

LEDGER_CONTENT_DOMAIN = b"LEDGER-CONTENT-V1\n"


class LedgerError(Exception):
    code = "E_STATE"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details}


@dataclass(frozen=True)
class RecoveryInfo:
    interrupted_attempts: tuple[int, ...]
    unknown_slots: tuple[int, ...]
    redone_slots: tuple[int, ...] = ()


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ledger_header (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY,
    feed_digest TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    param_digest TEXT NOT NULL,
    numbering_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    sample_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    replacement_reason TEXT,
    supersedes_generation INTEGER,
    affected_samples INTEGER NOT NULL DEFAULT 0,
    affected_slots_json TEXT NOT NULL DEFAULT '[]',
    progress_cursor INTEGER NOT NULL DEFAULT -1,
    completed_slot_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS arrays (
    generation_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    dtype TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (generation_id, name),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS slots (
    generation_id INTEGER NOT NULL,
    batch_slot INTEGER NOT NULL,
    status TEXT NOT NULL,
    cursor INTEGER NOT NULL DEFAULT 0,
    members_json TEXT NOT NULL DEFAULT '[]',
    overwritten_json TEXT NOT NULL DEFAULT '[]',
    failed_json TEXT NOT NULL DEFAULT '[]',
    output_digest TEXT,
    assignment_count INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (generation_id, batch_slot),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS assignments (
    generation_id INTEGER NOT NULL,
    batch_slot INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    sample_id TEXT NOT NULL,
    arrival_seq INTEGER NOT NULL,
    assignment_id TEXT NOT NULL,
    assignment_digest TEXT NOT NULL,
    sample_output_digest TEXT NOT NULL,
    batch_output_digest TEXT NOT NULL,
    overwritten_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (generation_id, batch_slot, rank),
    UNIQUE (generation_id, sample_id),
    FOREIGN KEY (generation_id, batch_slot)
        REFERENCES slots(generation_id, batch_slot) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    last_committed_slot INTEGER,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_generation_status
    ON attempts(generation_id, status);
CREATE INDEX IF NOT EXISTS idx_assignments_generation_slot
    ON assignments(generation_id, batch_slot, rank);
"""


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._ensure_header()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _ensure_header(self) -> None:
        row = self.conn.execute("SELECT * FROM ledger_header WHERE singleton = 1").fetchone()
        if row is None:
            empty_digest = sha256_hex(LEDGER_CONTENT_DOMAIN + b"empty\n")
            with self.conn:
                self.conn.execute(
                    "INSERT INTO ledger_header(singleton, format_version, content_digest, updated_at) "
                    "VALUES (1, ?, ?, ?)",
                    (LEDGER_FORMAT_VERSION, empty_digest, utc_now()),
                )
            return
        if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                f"unsupported ledger format version {row['format_version']}; "
                f"expected {LEDGER_FORMAT_VERSION}"
            )

    @property
    def format_version(self) -> int:
        row = self.conn.execute(
            "SELECT format_version FROM ledger_header WHERE singleton = 1"
        ).fetchone()
        return int(row["format_version"])

    @property
    def content_digest(self) -> str:
        row = self.conn.execute(
            "SELECT content_digest FROM ledger_header WHERE singleton = 1"
        ).fetchone()
        return str(row["content_digest"])

    def latest_generation(self) -> Mapping[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM generations ORDER BY generation_id DESC LIMIT 1"
        ).fetchone()
        return _generation_row(row) if row is not None else None

    def generation(self, generation_id: int) -> Mapping[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        return _generation_row(row) if row is not None else None

    def next_generation_id(self, snapshot_version: int = 0) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(generation_id), 0) AS n FROM generations").fetchone()
        return max(int(row["n"]), int(snapshot_version)) + 1

    def create_generation(
        self,
        *,
        generation_id: int,
        feed_digest: str,
        input_digest: str,
        param_digest: str,
        numbering_version: str,
        config: Mapping[str, Any],
        sample_ids: Sequence[str],
        supported_slots: Sequence[int],
        arrays: Mapping[str, np.ndarray],
        replacement_reason: str | None,
        supersedes_generation: int | None,
        affected_samples: int,
        affected_slots: Sequence[int],
    ) -> None:
        now = utc_now()
        with self.conn:
            if supersedes_generation is not None:
                self.conn.execute(
                    "UPDATE generations SET status = 'superseded' WHERE generation_id = ?",
                    (int(supersedes_generation),),
                )
            self.conn.execute(
                """
                INSERT INTO generations(
                    generation_id, feed_digest, input_digest, param_digest,
                    numbering_version, config_json, sample_ids_json, status,
                    replacement_reason, supersedes_generation, affected_samples,
                    affected_slots_json, progress_cursor, completed_slot_count,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, -1, 0, ?, NULL)
                """,
                (
                    int(generation_id),
                    feed_digest,
                    input_digest,
                    param_digest,
                    numbering_version,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    json.dumps(list(sample_ids), ensure_ascii=False),
                    replacement_reason,
                    supersedes_generation,
                    int(affected_samples),
                    json.dumps(sorted(int(slot) for slot in affected_slots)),
                    now,
                ),
            )
            for name, array in arrays.items():
                arr = np.ascontiguousarray(array, dtype="<f8")
                digest = sha256_hex(ARRAY_DIGEST_DOMAIN + array_payload(name, arr) + b"\n")
                self.conn.execute(
                    "INSERT INTO arrays(generation_id, name, dtype, shape_json, digest, data) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        int(generation_id),
                        name,
                        "<f8",
                        json.dumps(list(arr.shape)),
                        digest,
                        arr.tobytes(order="C"),
                    ),
                )
            for slot in sorted(int(item) for item in supported_slots):
                self.conn.execute(
                    "INSERT INTO slots(generation_id, batch_slot, status, cursor, updated_at) "
                    "VALUES (?, ?, 'pending', 0, ?)",
                    (int(generation_id), slot, now),
                )
            self._refresh_content_digest()

    def recover_interrupted(self, generation_id: int) -> RecoveryInfo:
        running = self.conn.execute(
            "SELECT attempt_id FROM attempts WHERE generation_id = ? AND status = 'running' "
            "ORDER BY attempt_id",
            (int(generation_id),),
        ).fetchall()
        attempt_ids = tuple(int(row["attempt_id"]) for row in running)
        unknown_slots = tuple(
            sorted(
                int(row["batch_slot"])
                for row in self.conn.execute(
                    "SELECT batch_slot FROM slots WHERE generation_id = ? "
                    "AND status NOT IN ('committed', 'empty') ORDER BY batch_slot",
                    (int(generation_id),),
                ).fetchall()
            )
        )
        if not attempt_ids:
            if not unknown_slots:
                return RecoveryInfo(interrupted_attempts=(), unknown_slots=())
        now = utc_now()
        with self.conn:
            for attempt_id in attempt_ids:
                self.conn.execute(
                    "UPDATE attempts SET status = 'interrupted', finished_at = ?, reason = ? "
                    "WHERE attempt_id = ?",
                    (now, "process_terminated_without_exit_report", attempt_id),
                )
            if unknown_slots:
                placeholders = ",".join("?" for _ in unknown_slots)
                self.conn.execute(
                    f"UPDATE slots SET status = 'unknown', error_code = 'E_UNKNOWN_REQUIRES_REDO', "
                    f"updated_at = ? WHERE generation_id = ? AND batch_slot IN ({placeholders})",
                    (now, int(generation_id), *unknown_slots),
                )
            self._refresh_content_digest()
        return RecoveryInfo(interrupted_attempts=attempt_ids, unknown_slots=unknown_slots)

    def start_attempt(self, generation_id: int) -> int:
        now = utc_now()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO attempts(generation_id, started_at, status, last_committed_slot) "
                "VALUES (?, ?, 'running', NULL)",
                (int(generation_id), now),
            )
            self._refresh_content_digest()
            return int(cursor.lastrowid)

    def finish_attempt(self, attempt_id: int, *, status: str, reason: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE attempts SET status = ?, finished_at = ?, reason = ? WHERE attempt_id = ?",
                (status, utc_now(), reason, int(attempt_id)),
            )
            self._refresh_content_digest()

    def commit_slot(
        self,
        *,
        generation_id: int,
        attempt_id: int,
        batch_slot: int,
        status: str,
        members: Sequence[Mapping[str, Any]],
        overwritten: Sequence[Mapping[str, Any]],
        failed: Sequence[Mapping[str, Any]],
        output_digest: str | None,
        assignments: Sequence[Mapping[str, Any]],
    ) -> None:
        if status not in {"committed", "empty"}:
            raise LedgerError(f"invalid slot commit status {status!r}")
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "DELETE FROM assignments WHERE generation_id = ? AND batch_slot = ?",
                (int(generation_id), int(batch_slot)),
            )
            for assignment in assignments:
                self.conn.execute(
                    """
                    INSERT INTO assignments(
                        generation_id, batch_slot, rank, sample_id, arrival_seq,
                        assignment_id, assignment_digest, sample_output_digest,
                        batch_output_digest, overwritten_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(generation_id),
                        int(batch_slot),
                        int(assignment["rank"]),
                        str(assignment["sample_id"]),
                        int(assignment["arrival_seq"]),
                        str(assignment["assignment_id"]),
                        str(assignment["assignment_digest"]),
                        str(assignment["sample_output_digest"]),
                        str(assignment["batch_output_digest"]),
                        int(assignment.get("overwritten_count", 0)),
                    ),
                )
            self.conn.execute(
                """
                UPDATE slots SET status = ?, cursor = ?, members_json = ?,
                    overwritten_json = ?, failed_json = ?, output_digest = ?,
                    assignment_count = ?, error_code = ?, updated_at = ?
                WHERE generation_id = ? AND batch_slot = ?
                """,
                (
                    status,
                    int(batch_slot),
                    json.dumps(list(members), ensure_ascii=False, sort_keys=True),
                    json.dumps(list(overwritten), ensure_ascii=False, sort_keys=True),
                    json.dumps(list(failed), ensure_ascii=False, sort_keys=True),
                    output_digest,
                    len(assignments),
                    "E_EMPTY" if status == "empty" else None,
                    now,
                    int(generation_id),
                    int(batch_slot),
                ),
            )
            self.conn.execute(
                "UPDATE generations SET progress_cursor = MAX(progress_cursor, ?) "
                "WHERE generation_id = ?",
                (int(batch_slot), int(generation_id)),
            )
            completed = self.conn.execute(
                "SELECT COUNT(*) AS n FROM slots WHERE generation_id = ? "
                "AND status IN ('committed', 'empty')",
                (int(generation_id),),
            ).fetchone()["n"]
            self.conn.execute(
                "UPDATE generations SET completed_slot_count = ? WHERE generation_id = ?",
                (int(completed), int(generation_id)),
            )
            self.conn.execute(
                "UPDATE attempts SET last_committed_slot = ? WHERE attempt_id = ?",
                (int(batch_slot), int(attempt_id)),
            )
            self._refresh_content_digest()

    def complete_generation(self, generation_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE generations SET status = 'complete', completed_at = ? WHERE generation_id = ?",
                (utc_now(), int(generation_id)),
            )
            self._refresh_content_digest()

    def slots(self, generation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM slots WHERE generation_id = ? ORDER BY batch_slot",
            (int(generation_id),),
        ).fetchall()
        return [_slot_row(row) for row in rows]

    def assignments(self, generation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM assignments WHERE generation_id = ? ORDER BY batch_slot, rank",
            (int(generation_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def attempts(self, generation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE generation_id = ? ORDER BY attempt_id",
            (int(generation_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def load_arrays(self, generation_id: int) -> dict[str, np.ndarray]:
        rows = self.conn.execute(
            "SELECT * FROM arrays WHERE generation_id = ? ORDER BY name",
            (int(generation_id),),
        ).fetchall()
        result: dict[str, np.ndarray] = {}
        for row in rows:
            dtype = np.dtype(str(row["dtype"]))
            shape = tuple(int(item) for item in json.loads(row["shape_json"]))
            array = np.frombuffer(row["data"], dtype=dtype).reshape(shape).copy()
            expected = sha256_hex(
                ARRAY_DIGEST_DOMAIN + array_payload(str(row["name"]), array) + b"\n"
            )
            if expected != str(row["digest"]):
                raise LedgerError(
                    "ledger array digest mismatch",
                    generation_id=int(generation_id),
                    name=str(row["name"]),
                    expected=str(row["digest"]),
                    actual=expected,
                )
            result[str(row["name"])] = array
        return result

    def array_rows(self, generation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT generation_id, name, dtype, shape_json, digest, data "
            "FROM arrays WHERE generation_id = ? ORDER BY name",
            (int(generation_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def content_digest_matches(self) -> bool:
        return compute_content_digest(self.conn) == self.content_digest

    def _refresh_content_digest(self) -> None:
        digest = compute_content_digest(self.conn)
        self.conn.execute(
            "UPDATE ledger_header SET content_digest = ?, updated_at = ? WHERE singleton = 1",
            (digest, utc_now()),
        )

    def bump_unknown_for_unfinished(self, generation_id: int) -> None:
        """Compatibility helper used when an attempt is known to have died."""

        self.recover_interrupted(generation_id)


def compute_content_digest(conn: sqlite3.Connection) -> str:
    generations = [
        _generation_row(row)
        for row in conn.execute("SELECT * FROM generations ORDER BY generation_id").fetchall()
    ]
    arrays = [
        {
            "generation_id": int(row["generation_id"]),
            "name": str(row["name"]),
            "dtype": str(row["dtype"]),
            "shape": json.loads(row["shape_json"]),
            "digest": str(row["digest"]),
        }
        for row in conn.execute(
            "SELECT generation_id, name, dtype, shape_json, digest FROM arrays "
            "ORDER BY generation_id, name"
        ).fetchall()
    ]
    slots = [
        _slot_row(row)
        for row in conn.execute(
            "SELECT * FROM slots ORDER BY generation_id, batch_slot"
        ).fetchall()
    ]
    assignments = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM assignments ORDER BY generation_id, batch_slot, rank"
        ).fetchall()
    ]
    attempts = [
        dict(row)
        for row in conn.execute("SELECT * FROM attempts ORDER BY attempt_id").fetchall()
    ]
    payload = {
        "format_version": LEDGER_FORMAT_VERSION,
        "generations": generations,
        "arrays": arrays,
        "slots": slots,
        "assignments": assignments,
        "attempts": attempts,
    }
    return hashlib.sha256(LEDGER_CONTENT_DOMAIN + canonical_json_bytes(payload) + b"\n").hexdigest()


def _generation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "generation_id": int(row["generation_id"]),
        "feed_digest": str(row["feed_digest"]),
        "input_digest": str(row["input_digest"]),
        "param_digest": str(row["param_digest"]),
        "numbering_version": str(row["numbering_version"]),
        "config": json.loads(row["config_json"]),
        "sample_ids": json.loads(row["sample_ids_json"]),
        "status": str(row["status"]),
        "replacement_reason": row["replacement_reason"],
        "supersedes_generation": row["supersedes_generation"],
        "affected_samples": int(row["affected_samples"]),
        "affected_slots": json.loads(row["affected_slots_json"]),
        "progress_cursor": int(row["progress_cursor"]),
        "completed_slot_count": int(row["completed_slot_count"]),
        "created_at": str(row["created_at"]),
        "completed_at": row["completed_at"],
    }


def _slot_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "generation_id": int(row["generation_id"]),
        "batch_slot": int(row["batch_slot"]),
        "status": str(row["status"]),
        "cursor": int(row["cursor"]),
        "members": json.loads(row["members_json"]),
        "overwritten": json.loads(row["overwritten_json"]),
        "failed": json.loads(row["failed_json"]),
        "output_digest": row["output_digest"],
        "assignment_count": int(row["assignment_count"]),
        "error_code": row["error_code"],
        "updated_at": str(row["updated_at"]),
    }
