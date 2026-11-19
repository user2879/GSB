"""Transactional SQLite pack ledger with replay and crash recovery."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from .canonical import digest, dumps
from .engine import Pack, PackPlan, PipelineState
from .errors import E_LEDGER, E_REPLAY, fail
from .scheduler import ScheduleEntry

LEDGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path, *, fail_on_write: bool = False) -> None:
        self.path = str(path)
        self.fail_on_write = fail_on_write
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=FULL")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        except sqlite3.Error as exc:
            fail(
                E_LEDGER,
                "$.ledger",
                "cannot open or initialise SQLite ledger",
                ledger_path=self.path,
                error=str(exc),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    snapshot_digest TEXT,
                    expected_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_epoch_ms REAL NOT NULL,
                    finished_epoch_ms REAL
                );
                CREATE TABLE IF NOT EXISTS expected_packs (
                    batch_id TEXT NOT NULL,
                    pack_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    source_id TEXT NOT NULL,
                    expected_digest TEXT NOT NULL,
                    PRIMARY KEY (batch_id, pack_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS packs (
                    batch_id TEXT NOT NULL,
                    pack_id TEXT NOT NULL,
                    source_id TEXT,
                    summary_json TEXT,
                    width_used INTEGER,
                    cost_micros INTEGER,
                    queue_start_ms REAL,
                    start_ms REAL,
                    end_ms REAL,
                    wait_ms REAL,
                    digest TEXT,
                    status TEXT NOT NULL,
                    committed_epoch_ms REAL,
                    PRIMARY KEY (batch_id, pack_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_packs_batch_status
                    ON packs(batch_id, status, pack_id);
                """
            )
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT value FROM ledger_meta WHERE key='format_version'"
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO ledger_meta(key,value) VALUES('format_version',?)",
                        (str(LEDGER_FORMAT_VERSION),),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _now_epoch_ms() -> float:
        return round(time.time() * 1000.0, 3)

    def format_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM ledger_meta WHERE key='format_version'"
            ).fetchone()
            if row is None:
                return LEDGER_FORMAT_VERSION
            try:
                return int(row["value"])
            except (TypeError, ValueError):
                return -1

    def _check_injection(self, operation: str) -> None:
        if self.fail_on_write:
            fail(
                E_LEDGER,
                "$.ledger.write",
                "injected ledger write failure",
                operation=operation,
                ledger_path=self.path,
            )

    def begin_batch(
        self,
        batch_id: str,
        packs: Sequence[Pack],
        *,
        spec_digest: str,
        snapshot_digest: str,
    ) -> None:
        expected = sorted(packs, key=lambda item: item.pack_id)
        self._begin_manifest(
            batch_id,
            [(pack.pack_id, pack.source_id, pack.digest, pack) for pack in expected],
            spec_digest=spec_digest,
            snapshot_digest=snapshot_digest,
        )

    def begin_batch_plans(
        self,
        batch_id: str,
        plans: Sequence[PackPlan],
        *,
        spec_digest: str,
        snapshot_digest: str | None = None,
    ) -> None:
        """Create a manifest before packs finish; commit_pack fills each row."""

        ordered = sorted(plans, key=lambda item: item.pack_id)
        self._begin_manifest(
            batch_id,
            [(plan.pack_id, plan.source_id, "", None) for plan in ordered],
            spec_digest=spec_digest,
            snapshot_digest=snapshot_digest,
        )

    def _begin_manifest(
        self,
        batch_id: str,
        entries: Sequence[tuple[str, str, str, Pack | None]],
        *,
        spec_digest: str,
        snapshot_digest: str | None,
    ) -> None:
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if existing is not None:
                raise_failure = existing["spec_digest"] != spec_digest or existing[
                    "expected_count"
                ] != len(entries)
                if raise_failure:
                    fail(
                        E_LEDGER,
                        "$.ledger.batches",
                        "batch id already exists with incompatible content",
                        batch_id=batch_id,
                )
                return
            conn.execute(
                """
                INSERT INTO batches(
                    batch_id, format_version, spec_digest, snapshot_digest,
                    expected_count, status, created_epoch_ms
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    batch_id,
                    LEDGER_FORMAT_VERSION,
                    spec_digest,
                    snapshot_digest,
                    len(entries),
                    "RUNNING",
                    self._now_epoch_ms(),
                ),
            )
            for ordinal, (pack_id, source_id, expected_digest, _pack) in enumerate(entries):
                conn.execute(
                    """
                    INSERT INTO expected_packs(
                        batch_id, pack_id, ordinal, source_id, expected_digest
                    ) VALUES(?,?,?,?,?)
                    """,
                    (batch_id, pack_id, ordinal, source_id, expected_digest),
                )
            self._update_digests(conn, batch_id)

    def commit_pack(
        self,
        batch_id: str,
        pack: Pack,
        schedule: ScheduleEntry | Mapping[str, Any] | None = None,
    ) -> str:
        self._check_injection("commit_pack")
        summary_json = dumps(pack.to_dict())
        if isinstance(schedule, ScheduleEntry):
            queue_start = schedule.queue_start_ms
            start_ms = schedule.start_ms
            end_ms = schedule.end_ms
            wait_ms = schedule.wait_ms
        elif schedule is not None:
            queue_start = float(schedule.get("queue_start_ms", 0.0))
            start_ms = float(schedule.get("start_ms", 0.0))
            end_ms = float(schedule.get("end_ms", 0.0))
            wait_ms = float(schedule.get("wait_ms", 0.0))
        else:
            queue_start = start_ms = end_ms = wait_ms = 0.0
        try:
            with self._transaction() as conn:
                batch = conn.execute(
                    "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
                if batch is None:
                    fail(
                        E_LEDGER,
                        "$.ledger.batches",
                        "cannot commit a pack into a missing batch",
                        batch_id=batch_id,
                        pack_id=pack.pack_id,
                    )
                expected = conn.execute(
                    "SELECT * FROM expected_packs WHERE batch_id=? AND pack_id=?",
                    (batch_id, pack.pack_id),
                ).fetchone()
                if expected is None:
                    fail(
                        E_LEDGER,
                        f"$.ledger.batches[{batch_id}].packs[{pack.pack_id}]",
                        "pack is not part of the batch manifest",
                        batch_id=batch_id,
                        pack_id=pack.pack_id,
                    )
                conn.execute(
                    """
                    INSERT INTO packs(
                        batch_id, pack_id, source_id, summary_json, width_used,
                        cost_micros, queue_start_ms, start_ms, end_ms, wait_ms,
                        digest, status, committed_epoch_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(batch_id, pack_id) DO UPDATE SET
                        source_id=excluded.source_id,
                        summary_json=excluded.summary_json,
                        width_used=excluded.width_used,
                        cost_micros=excluded.cost_micros,
                        queue_start_ms=excluded.queue_start_ms,
                        start_ms=excluded.start_ms,
                        end_ms=excluded.end_ms,
                        wait_ms=excluded.wait_ms,
                        digest=excluded.digest,
                        status=excluded.status,
                        committed_epoch_ms=excluded.committed_epoch_ms
                    """,
                    (
                        batch_id,
                        pack.pack_id,
                        pack.source_id,
                        summary_json,
                        pack.width_used,
                        pack.cost_micros,
                        queue_start,
                        start_ms,
                        end_ms,
                        wait_ms,
                        pack.digest,
                        "COMMITTED",
                        self._now_epoch_ms(),
                    ),
                )
                digest_value = self._update_digests(conn, batch_id)
                return digest_value
        except sqlite3.Error as exc:
            fail(
                E_LEDGER,
                f"$.ledger.packs[{pack.pack_id}]",
                "SQLite pack commit failed",
                batch_id=batch_id,
                pack_id=pack.pack_id,
                error=str(exc),
            )

    def finish_batch(self, batch_id: str, *, snapshot_digest: str | None = None) -> str:
        self._check_injection("finish_batch")
        try:
            with self._transaction() as conn:
                batch = conn.execute(
                    "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
                ).fetchone()
                if batch is None:
                    fail(
                        E_LEDGER,
                        "$.ledger.batches",
                        "cannot finish a missing batch",
                        batch_id=batch_id,
                    )
                committed = conn.execute(
                    "SELECT COUNT(*) AS n FROM packs WHERE batch_id=? AND status='COMMITTED'",
                    (batch_id,),
                ).fetchone()["n"]
                if committed != batch["expected_count"]:
                    fail(
                        E_LEDGER,
                        f"$.ledger.batches[{batch_id}]",
                        "cannot finish a batch with uncommitted packs",
                        batch_id=batch_id,
                        expected_count=batch["expected_count"],
                        committed_count=committed,
                    )
                conn.execute(
                    "UPDATE batches SET status='COMPLETE', finished_epoch_ms=?, snapshot_digest=? WHERE batch_id=?",
                    (
                        self._now_epoch_ms(),
                        snapshot_digest if snapshot_digest is not None else batch["snapshot_digest"],
                        batch_id,
                    ),
                )
                return self._update_digests(conn, batch_id)
        except sqlite3.Error as exc:
            fail(
                E_LEDGER,
                f"$.ledger.batches[{batch_id}]",
                "SQLite batch finalisation failed",
                batch_id=batch_id,
                error=str(exc),
            )

    def _update_digests(self, conn: sqlite3.Connection, batch_id: str) -> str:
        batch = conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch is None:
            return ""
        rows = conn.execute(
            """
            SELECT pack_id, source_id, summary_json, width_used, cost_micros, digest, status
            FROM packs WHERE batch_id=? AND status='COMMITTED' ORDER BY pack_id
            """,
            (batch_id,),
        ).fetchall()
        payload = {
            "format_version": int(batch["format_version"]),
            "batch_id": batch_id,
            "packs": [
                {
                    "pack_id": row["pack_id"],
                    "source_id": row["source_id"],
                    "summary_json": row["summary_json"],
                    "width_used": row["width_used"],
                    "cost_micros": row["cost_micros"],
                    "digest": row["digest"],
                    "status": row["status"],
                }
                for row in rows
            ],
        }
        batch_digest = digest(payload)
        conn.execute(
            "INSERT INTO ledger_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (f"batch:{batch_id}:content_digest", batch_digest),
        )
        batches = conn.execute(
            "SELECT batch_id FROM batches ORDER BY batch_id"
        ).fetchall()
        ledger_payload = {
            "format_version": self._meta_int_no_lock(conn, "format_version"),
            "batches": [
                {
                    "batch_id": item["batch_id"],
                    "content_digest": self._meta_value_no_lock(
                        conn, f"batch:{item['batch_id']}:content_digest"
                    ),
                }
                for item in batches
            ],
        }
        ledger_digest = digest(ledger_payload)
        conn.execute(
            "INSERT INTO ledger_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("ledger_content_digest", ledger_digest),
        )
        return batch_digest

    @staticmethod
    def _meta_value_no_lock(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    @classmethod
    def _meta_int_no_lock(cls, conn: sqlite3.Connection, key: str) -> int:
        value = cls._meta_value_no_lock(conn, key)
        try:
            return int(value) if value is not None else LEDGER_FORMAT_VERSION
        except ValueError:
            return -1

    def batch_content_digest(self, batch_id: str) -> str | None:
        with self._lock:
            return self._meta_value_no_lock(self._conn, f"batch:{batch_id}:content_digest")

    def ledger_content_digest(self) -> str | None:
        with self._lock:
            return self._meta_value_no_lock(self._conn, "ledger_content_digest")

    def recover_batch(
        self,
        batch_id: str,
        packs: Sequence[Pack],
        *,
        spec_digest: str,
        snapshot_digest: str,
        schedule: Mapping[str, ScheduleEntry | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Verify the committed prefix and mark uncommitted packs UNKNOWN.

        A version/content mismatch is repaired by replacing the whole ledger
        with the supplied verified snapshot and then raising ``E_REPLAY``.
        """

        expected = {pack.pack_id: pack for pack in sorted(packs, key=lambda item: item.pack_id)}
        schedule = schedule or {}
        with self._lock:
            current_version = self.format_version()
            batch = self._conn.execute(
                "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if current_version != LEDGER_FORMAT_VERSION or (
            batch is not None and int(batch["format_version"]) != LEDGER_FORMAT_VERSION
        ):
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.batches[{batch_id}].format_version",
                "ledger format version mismatch; ledger state was replaced",
                first_differing_pack_id=next(iter(expected), ""),
                expected_version=LEDGER_FORMAT_VERSION,
                actual_version=current_version,
            )
        if batch is None:
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            return {
                "batch_id": batch_id,
                "status": "created",
                "unknown_packs": [],
                "verified_packs": sorted(expected),
                "first_replay_difference": None,
            }

        if batch["spec_digest"] != spec_digest:
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.batches[{batch_id}].spec_digest",
                "ledger specification digest differs from the supplied snapshot; ledger state was replaced",
                first_differing_pack_id=next(iter(expected), ""),
            )

        expected_rows = self._expected_rows(batch_id)
        supplied_manifest = {
            pack_id: (pack.source_id, pack.digest) for pack_id, pack in expected.items()
        }
        if not self._manifest_matches(expected_rows, supplied_manifest):
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.batches[{batch_id}].expected_packs",
                "ledger manifest differs from the supplied snapshot; ledger state was replaced",
                first_differing_pack_id=self._first_manifest_difference(expected_rows, supplied_manifest),
            )

        committed_rows = {
            row["pack_id"]: row
            for row in self._conn.execute(
                "SELECT * FROM packs WHERE batch_id=? ORDER BY pack_id", (batch_id,)
            ).fetchall()
        }
        extra = sorted(set(committed_rows) - set(expected))
        if extra:
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.packs[{extra[0]}]",
                "ledger contains an unexpected pack; ledger state was replaced",
                first_differing_pack_id=extra[0],
            )

        unknown: list[str] = []
        verified: list[str] = []
        for pack_id in sorted(expected):
            pack = expected[pack_id]
            row = committed_rows.get(pack_id)
            if row is None or row["status"] != "COMMITTED":
                unknown.append(pack_id)
                continue
            if self._pack_row_matches(row, pack) is False:
                self._replace_with_snapshot(
                    batch_id,
                    expected,
                    spec_digest=spec_digest,
                    snapshot_digest=snapshot_digest,
                    schedule=schedule,
                )
                fail(
                    E_REPLAY,
                    f"$.ledger.packs[{pack_id}]",
                    "pack replay digest/content mismatch; ledger state was replaced",
                    first_differing_pack_id=pack_id,
                    first_differing_bucket_id=self._first_bucket_difference(row, pack),
                )
            verified.append(pack_id)

        stored_digest = self.batch_content_digest(batch_id)
        computed_digest = self._batch_digest_from_rows(batch_id, committed_rows)
        if stored_digest != computed_digest:
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.batches[{batch_id}].content_digest",
                "ledger content digest mismatch; ledger state was replaced",
                first_differing_pack_id=next(iter(sorted(expected)), ""),
                stored_digest=stored_digest,
                computed_digest=computed_digest,
            )

        if unknown and batch["status"] == "COMPLETE":
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.packs[{unknown[0]}]",
                "a complete ledger batch is missing a pack; ledger state was replaced",
                first_differing_pack_id=unknown[0],
            )

        if unknown:
            with self._transaction() as conn:
                for pack_id in unknown:
                    conn.execute(
                        """
                        INSERT INTO packs(
                            batch_id, pack_id, source_id, summary_json, width_used,
                            cost_micros, queue_start_ms, start_ms, end_ms, wait_ms,
                            digest, status, committed_epoch_ms
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(batch_id, pack_id) DO UPDATE SET status='UNKNOWN'
                        """,
                        (batch_id, pack_id, None, None, None, None, None, None, None, None, None, "UNKNOWN", None),
                    )
                self._update_digests(conn, batch_id)
            return {
                "batch_id": batch_id,
                "status": "partial",
                "unknown_packs": unknown,
                "verified_packs": verified,
                "first_replay_difference": None,
                "requires_rerun": True,
            }

        if batch["status"] != "COMPLETE":
            self.finish_batch(batch_id, snapshot_digest=snapshot_digest)
        elif batch["snapshot_digest"] and batch["snapshot_digest"] != snapshot_digest:
            self._replace_with_snapshot(
                batch_id,
                expected,
                spec_digest=spec_digest,
                snapshot_digest=snapshot_digest,
                schedule=schedule,
            )
            fail(
                E_REPLAY,
                f"$.ledger.batches[{batch_id}].snapshot_digest",
                "ledger snapshot digest differs from the supplied snapshot; ledger state was replaced",
                first_differing_pack_id=next(iter(sorted(expected)), ""),
            )
        return {
            "batch_id": batch_id,
            "status": "complete",
            "unknown_packs": [],
            "verified_packs": verified,
            "first_replay_difference": None,
            "requires_rerun": False,
        }

    def _batch_digest_from_rows(
        self, batch_id: str, rows: Mapping[str, sqlite3.Row]
    ) -> str:
        with self._lock:
            batch = self._conn.execute(
                "SELECT format_version FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        payload = {
            "format_version": int(batch["format_version"]) if batch is not None else LEDGER_FORMAT_VERSION,
            "batch_id": batch_id,
            "packs": [
                {
                    "pack_id": rows[pack_id]["pack_id"],
                    "source_id": rows[pack_id]["source_id"],
                    "summary_json": rows[pack_id]["summary_json"],
                    "width_used": rows[pack_id]["width_used"],
                    "cost_micros": rows[pack_id]["cost_micros"],
                    "digest": rows[pack_id]["digest"],
                    "status": rows[pack_id]["status"],
                }
                for pack_id in sorted(rows)
                if rows[pack_id]["status"] == "COMMITTED"
            ],
        }
        return digest(payload)

    def _expected_rows(self, batch_id: str) -> dict[str, tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT pack_id, source_id, expected_digest FROM expected_packs WHERE batch_id=? ORDER BY pack_id",
                (batch_id,),
            ).fetchall()
        return {row["pack_id"]: (row["source_id"], row["expected_digest"]) for row in rows}

    @staticmethod
    def _first_manifest_difference(
        old: Mapping[str, tuple[str, str]], new: Mapping[str, tuple[str, str]]
    ) -> str:
        for pack_id in sorted(set(old) | set(new)):
            if old.get(pack_id) != new.get(pack_id):
                return pack_id
        return ""

    @staticmethod
    def _manifest_matches(
        old: Mapping[str, tuple[str, str]], new: Mapping[str, tuple[str, str]]
    ) -> bool:
        if set(old) != set(new):
            return False
        for pack_id in old:
            old_source, old_digest = old[pack_id]
            new_source, new_digest = new[pack_id]
            if old_source != new_source:
                return False
            if old_digest and old_digest != new_digest:
                return False
        return True

    @staticmethod
    def _pack_row_matches(row: sqlite3.Row, pack: Pack) -> bool:
        return (
            row["source_id"] == pack.source_id
            and row["summary_json"] == dumps(pack.to_dict())
            and int(row["width_used"]) == pack.width_used
            and int(row["cost_micros"]) == pack.cost_micros
            and row["digest"] == pack.digest
        )

    @staticmethod
    def _first_bucket_difference(row: sqlite3.Row, pack: Pack) -> str | None:
        try:
            old = json_loads(row["summary_json"]) if row["summary_json"] else {}
        except Exception:
            return next(iter(sorted(pack.bucket_counts)), None)
        old_counts = old.get("bucket_counts", {})
        for bucket_id in sorted(pack.bucket_counts):
            if old_counts.get(bucket_id) != list(pack.bucket_counts[bucket_id]):
                return bucket_id
        return None

    def _replace_with_snapshot(
        self,
        batch_id: str,
        packs: Mapping[str, Pack],
        *,
        spec_digest: str,
        snapshot_digest: str,
        schedule: Mapping[str, ScheduleEntry | Mapping[str, Any]],
    ) -> None:
        try:
            with self._transaction() as conn:
                conn.execute("DELETE FROM packs")
                conn.execute("DELETE FROM expected_packs")
                conn.execute("DELETE FROM batches")
                conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('format_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                conn.execute(
                    "INSERT INTO batches(batch_id,format_version,spec_digest,snapshot_digest,expected_count,status,created_epoch_ms,finished_epoch_ms) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        batch_id,
                        LEDGER_FORMAT_VERSION,
                        spec_digest,
                        snapshot_digest,
                        len(packs),
                        "COMPLETE",
                        self._now_epoch_ms(),
                        self._now_epoch_ms(),
                    ),
                )
                for ordinal, pack_id in enumerate(sorted(packs)):
                    pack = packs[pack_id]
                    conn.execute(
                        "INSERT INTO expected_packs(batch_id,pack_id,ordinal,source_id,expected_digest) VALUES(?,?,?,?,?)",
                        (batch_id, pack_id, ordinal, pack.source_id, pack.digest),
                    )
                    self._insert_pack_no_lock(conn, batch_id, pack, schedule.get(pack_id))
                self._update_digests(conn, batch_id)
        except sqlite3.Error as exc:
            fail(
                E_LEDGER,
                "$.ledger",
                "failed to replace ledger state",
                error=str(exc),
            )

    def _insert_pack_no_lock(
        self,
        conn: sqlite3.Connection,
        batch_id: str,
        pack: Pack,
        schedule: ScheduleEntry | Mapping[str, Any] | None,
    ) -> None:
        if isinstance(schedule, ScheduleEntry):
            values = (
                schedule.queue_start_ms,
                schedule.start_ms,
                schedule.end_ms,
                schedule.wait_ms,
            )
        elif schedule is not None:
            values = (
                float(schedule.get("queue_start_ms", 0.0)),
                float(schedule.get("start_ms", 0.0)),
                float(schedule.get("end_ms", 0.0)),
                float(schedule.get("wait_ms", 0.0)),
            )
        else:
            values = (0.0, 0.0, 0.0, 0.0)
        conn.execute(
            """
            INSERT INTO packs(
                batch_id,pack_id,source_id,summary_json,width_used,cost_micros,
                queue_start_ms,start_ms,end_ms,wait_ms,digest,status,committed_epoch_ms
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                batch_id,
                pack.pack_id,
                pack.source_id,
                dumps(pack.to_dict()),
                pack.width_used,
                pack.cost_micros,
                values[0],
                values[1],
                values[2],
                values[3],
                pack.digest,
                "COMMITTED",
                self._now_epoch_ms(),
            ),
        )

    def persist_state(
        self,
        state: PipelineState,
        batch_id: str,
        *,
        crash_after: int | None = None,
        on_commit: Any = None,
    ) -> dict[str, Any]:
        packs = sorted(state.packs, key=lambda item: item.pack_id)
        schedule: dict[str, Any] = {
            entry["pack_id"]: entry for entry in state.result["schedule"]["entries"]
        }
        self.begin_batch(
            batch_id,
            packs,
            spec_digest=str(state.result["spec_digest"]),
            snapshot_digest=str(state.result["content_digest"]),
        )
        if crash_after is not None and crash_after <= 0:
            os._exit(17)
        for idx, pack in enumerate(packs, start=1):
            self.commit_pack(batch_id, pack, schedule.get(pack.pack_id))
            if on_commit is not None:
                on_commit(pack)
            if crash_after is not None and idx >= crash_after:
                os._exit(17)
        digest_value = self.finish_batch(
            batch_id, snapshot_digest=str(state.result["content_digest"])
        )
        return {
            "batch_id": batch_id,
            "status": "complete",
            "content_digest": digest_value,
            "packs": len(packs),
        }

    def read_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM batches ORDER BY batch_id"
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                committed = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM packs WHERE batch_id=? AND status='COMMITTED'",
                    (row["batch_id"],),
                ).fetchone()["n"]
                unknown = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM packs WHERE batch_id=? AND status='UNKNOWN'",
                    (row["batch_id"],),
                ).fetchone()["n"]
                out.append(
                    {
                        "batch_id": row["batch_id"],
                        "format_version": row["format_version"],
                        "spec_digest": row["spec_digest"],
                        "snapshot_digest": row["snapshot_digest"],
                        "expected_count": row["expected_count"],
                        "committed_count": committed,
                        "unknown_count": unknown,
                        "status": row["status"],
                        "content_digest": self._meta_value_no_lock(
                            self._conn, f"batch:{row['batch_id']}:content_digest"
                        ),
                    }
                )
            return out

    def read_packs(self, batch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT pack_id,source_id,width_used,cost_micros,queue_start_ms,start_ms,end_ms,wait_ms,digest,status FROM packs WHERE batch_id=? ORDER BY pack_id",
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def expected_pack_ids(self, batch_id: str) -> list[str]:
        return sorted(self._expected_rows(batch_id))

    def schema_digest(self) -> str:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return digest([(row["name"], row["sql"]) for row in rows])


def json_loads(text: str) -> Any:
    import json

    return json.loads(text)


def schedule_map_from_state(state: PipelineState) -> dict[str, dict[str, Any]]:
    return {entry["pack_id"]: entry for entry in state.result["schedule"]["entries"]}


def ledger_summary(ledger: Ledger) -> dict[str, Any]:
    batches = ledger.read_batches()
    return {
        "format_version": ledger.format_version(),
        "content_digest": ledger.ledger_content_digest(),
        "batches": batches,
    }
