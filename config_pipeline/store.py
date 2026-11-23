"""SQLite persistence for immutable ledger entries and whole-view snapshots."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import E_REPLAY, PipelineError
from .util import canonical_json, digest_json, monotonic_ms


DB_SCHEMA_VERSION = 1
LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


class Store:
    """A small, thread-safe SQLite store.

    The connection is shared, so all writes are serialised by a re-entrant
    lock.  Each ledger completion is nevertheless one real SQLite transaction;
    a forced process death can only leave a committed prefix or roll it back.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.path, timeout=30.0, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        with self.lock:
            try:
                self.connection.execute("PRAGMA foreign_keys = ON")
                self.connection.execute("PRAGMA synchronous = FULL")
                if self.path != ":memory:":
                    self.connection.execute("PRAGMA journal_mode = WAL")
                self._create_schema()
                self._initialise_and_verify_ledger()
            except Exception:
                self.connection.close()
                raise

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                resource_units INTEGER NOT NULL,
                aggregate_digest TEXT,
                aggregate_json TEXT,
                revision_version INTEGER NOT NULL,
                compile_digest TEXT NOT NULL,
                progress_cursor INTEGER NOT NULL UNIQUE,
                payload_digest TEXT NOT NULL,
                committed_at_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_shard
                ON ledger_entries(shard_id, entry_id);
            CREATE INDEX IF NOT EXISTS idx_ledger_generation
                ON ledger_entries(generation, entry_id);

            CREATE TABLE IF NOT EXISTS inflight (
                generation TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                state TEXT NOT NULL,
                started_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (generation, shard_id)
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                generation TEXT NOT NULL UNIQUE,
                format_version INTEGER NOT NULL,
                is_current INTEGER NOT NULL,
                is_stale INTEGER NOT NULL,
                cluster_version TEXT NOT NULL,
                revision_chain_version TEXT NOT NULL,
                budget_version TEXT NOT NULL,
                ledger_cursor INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshot_current
                ON snapshots(is_current, is_stale);

            CREATE TABLE IF NOT EXISTS view_shards (
                generation TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                revision_version INTEGER NOT NULL,
                compile_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                ledger_entry_id INTEGER,
                is_stale INTEGER NOT NULL,
                PRIMARY KEY (generation, shard_id)
            );
            CREATE INDEX IF NOT EXISTS idx_view_generation
                ON view_shards(generation, is_stale);

            CREATE TABLE IF NOT EXISTS invalidations (
                invalidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation TEXT NOT NULL,
                revision_version INTEGER NOT NULL,
                shard_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                basis_json TEXT NOT NULL,
                is_stale INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_invalidations_generation
                ON invalidations(generation, is_stale);

            CREATE TABLE IF NOT EXISTS generation_stats (
                generation TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                is_stale INTEGER NOT NULL
            );
            """
        )
        self.connection.commit()

    def _meta_get(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _meta_set(self, key: str, value: str | int) -> None:
        self.connection.execute(
            """
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, str(value)),
        )

    def _ledger_rows_locked(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM ledger_entries ORDER BY entry_id"
        ).fetchall()
        return [self._row_payload(row) for row in rows]

    @staticmethod
    def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
        aggregate_summary = None
        if row["aggregate_json"] is not None:
            aggregate_summary = json.loads(row["aggregate_json"])
        return {
            "entry_id": int(row["entry_id"]),
            "generation": str(row["generation"]),
            "shard_id": str(row["shard_id"]),
            "outcome": str(row["outcome"]),
            "resource_units": int(row["resource_units"]),
            "aggregate_digest": row["aggregate_digest"],
            "aggregate_summary": aggregate_summary,
            "revision_version": int(row["revision_version"]),
            "compile_digest": str(row["compile_digest"]),
            "progress_cursor": int(row["progress_cursor"]),
            "committed_at_ms": int(row["committed_at_ms"]),
        }

    def _ledger_content_digest_locked(self) -> str:
        rows = self._ledger_rows_locked()
        with_payload_digests = []
        raw = self.connection.execute(
            "SELECT entry_id, payload_digest FROM ledger_entries ORDER BY entry_id"
        ).fetchall()
        digests = {int(row[0]): str(row[1]) for row in raw}
        for row in rows:
            item = dict(row)
            item["payload_digest"] = digests[int(item["entry_id"])]
            with_payload_digests.append(item)
        return digest_json(with_payload_digests)

    def _initialise_and_verify_ledger(self) -> None:
        schema = self._meta_get("schema_version")
        if schema is None:
            self._meta_set("schema_version", DB_SCHEMA_VERSION)
        else:
            try:
                schema_int = int(schema)
            except ValueError as exc:
                raise PipelineError(
                    E_REPLAY,
                    "database schema version is not an integer",
                    field="meta.schema_version",
                ) from exc
            if schema_int != DB_SCHEMA_VERSION:
                raise PipelineError(
                    E_REPLAY,
                    "unsupported database schema version",
                    field="meta.schema_version",
                    details={"expected": DB_SCHEMA_VERSION, "actual": schema_int},
                )

        ledger_format = self._meta_get("ledger_format_version")
        if ledger_format is None:
            self._meta_set("ledger_format_version", LEDGER_FORMAT_VERSION)
        else:
            try:
                format_int = int(ledger_format)
            except ValueError as exc:
                raise PipelineError(
                    E_REPLAY,
                    "ledger format version is not an integer",
                    field="meta.ledger_format_version",
                ) from exc
            if format_int != LEDGER_FORMAT_VERSION:
                raise PipelineError(
                    E_REPLAY,
                    "unsupported ledger format version",
                    field="meta.ledger_format_version",
                    details={"expected": LEDGER_FORMAT_VERSION, "actual": format_int},
                )

        stored_cursor = self._meta_get("ledger_cursor")
        max_cursor_row = self.connection.execute(
            "SELECT COALESCE(MAX(progress_cursor), 0) FROM ledger_entries"
        ).fetchone()
        max_cursor = int(max_cursor_row[0])
        cursor = stored_cursor
        if cursor is None:
            self._meta_set("ledger_cursor", max_cursor)
        else:
            try:
                cursor_value = int(cursor)
            except ValueError as exc:
                raise PipelineError(
                    E_REPLAY,
                    "ledger cursor is not an integer",
                    field="meta.ledger_cursor",
                ) from exc
            if cursor_value != max_cursor:
                raise PipelineError(
                    E_REPLAY,
                    "ledger cursor does not match committed entries",
                    field="meta.ledger_cursor",
                    details={"stored": cursor_value, "actual": max_cursor},
                )

        for row in self.connection.execute(
            "SELECT * FROM ledger_entries ORDER BY entry_id"
        ).fetchall():
            payload = self._row_payload(row)
            expected_row_digest = digest_json(payload)
            if str(row["payload_digest"]) != expected_row_digest:
                raise PipelineError(
                    E_REPLAY,
                    "ledger row digest does not match its committed content",
                    field=f"ledger_entries[{int(row['entry_id'])}].payload_digest",
                    shard_id=str(row["shard_id"]),
                    details={
                        "expected": row["payload_digest"],
                        "actual": expected_row_digest,
                    },
                )

        actual_digest = self._ledger_content_digest_locked()
        expected_digest = self._meta_get("ledger_content_digest")
        if expected_digest is None:
            self._meta_set("ledger_content_digest", actual_digest)
        elif expected_digest != actual_digest:
            raise PipelineError(
                E_REPLAY,
                "ledger content digest does not match committed rows",
                field="meta.ledger_content_digest",
                details={"expected": expected_digest, "actual": actual_digest},
            )
        self.connection.commit()

    @property
    def ledger_cursor(self) -> int:
        with self.lock:
            value = self._meta_get("ledger_cursor")
            return int(value or 0)

    @property
    def ledger_format_version(self) -> int:
        with self.lock:
            return int(self._meta_get("ledger_format_version") or 0)

    @property
    def ledger_content_digest(self) -> str:
        with self.lock:
            return str(self._meta_get("ledger_content_digest") or "")

    def next_generation_id(self) -> str:
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                current = int(self._meta_get("generation_seq") or 0) + 1
                self._meta_set("generation_seq", current)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
            return f"g{current:08d}"

    def mark_inflight(self, generation: str, shard_id: str) -> None:
        now = monotonic_ms()
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    """
                    INSERT INTO inflight(generation, shard_id, state, started_at_ms, updated_at_ms)
                    VALUES(?, ?, 'running', ?, ?)
                    ON CONFLICT(generation, shard_id) DO UPDATE SET
                        state = 'running',
                        started_at_ms = excluded.started_at_ms,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (generation, shard_id, now, now),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def commit_shard(
        self,
        *,
        generation: str,
        shard_id: str,
        outcome: str,
        resource_units: int,
        aggregate_summary: Mapping[str, Any] | None,
        aggregate_digest: str | None,
        revision_version: int,
        compile_digest: str,
    ) -> dict[str, Any]:
        """Atomically append one outcome, update the cursor and in-flight state."""

        aggregate_json = (
            None if aggregate_summary is None else canonical_json(aggregate_summary)
        )
        now = monotonic_ms()
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = int(self._meta_get("ledger_cursor") or 0) + 1
                cursor_row = self.connection.execute(
                    "SELECT COALESCE(MAX(progress_cursor), 0) FROM ledger_entries"
                ).fetchone()
                cursor = max(cursor, int(cursor_row[0]) + 1)
                inserted = self.connection.execute(
                    """
                    INSERT INTO ledger_entries(
                        generation, shard_id, outcome, resource_units,
                        aggregate_digest, aggregate_json, revision_version,
                        compile_digest, progress_cursor, payload_digest, committed_at_ms
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                    """,
                    (
                        generation,
                        shard_id,
                        outcome,
                        int(resource_units),
                        aggregate_digest,
                        aggregate_json,
                        int(revision_version),
                        compile_digest,
                        cursor,
                        now,
                    ),
                )
                entry_id = int(inserted.lastrowid)
                payload = {
                    "entry_id": entry_id,
                    "generation": generation,
                    "shard_id": shard_id,
                    "outcome": outcome,
                    "resource_units": int(resource_units),
                    "aggregate_digest": aggregate_digest,
                    "aggregate_summary": dict(aggregate_summary)
                    if aggregate_summary is not None
                    else None,
                    "revision_version": int(revision_version),
                    "compile_digest": compile_digest,
                    "progress_cursor": cursor,
                    "committed_at_ms": now,
                }
                payload_digest = digest_json(payload)
                self.connection.execute(
                    "UPDATE ledger_entries SET payload_digest = ? WHERE entry_id = ?",
                    (payload_digest, entry_id),
                )
                self.connection.execute(
                    """
                    UPDATE inflight SET state = 'committed', updated_at_ms = ?
                    WHERE generation = ? AND shard_id = ?
                    """,
                    (now, generation, shard_id),
                )
                self._meta_set("ledger_cursor", cursor)
                content_digest = self._ledger_content_digest_locked()
                self._meta_set("ledger_content_digest", content_digest)
                self._meta_set("ledger_format_version", LEDGER_FORMAT_VERSION)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        payload["payload_digest"] = payload_digest
        return payload

    def mark_recovery_unknown(self) -> tuple[dict[str, Any], ...]:
        """Turn every previously running attempt into an explicit unknown row."""

        now = monotonic_ms()
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self.connection.execute(
                    "SELECT generation, shard_id, started_at_ms FROM inflight WHERE state = 'running'"
                ).fetchall()
                if rows:
                    self.connection.execute(
                        """
                        UPDATE inflight SET state = 'unknown', updated_at_ms = ?
                        WHERE state = 'running'
                        """,
                        (now,),
                    )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return tuple(
            {
                "generation": str(row["generation"]),
                "shard_id": str(row["shard_id"]),
                "started_at_ms": int(row["started_at_ms"]),
                "state": "unknown",
            }
            for row in rows
        )

    def ledger_entries(self) -> tuple[dict[str, Any], ...]:
        with self.lock:
            return tuple(self._ledger_rows_locked())

    def latest_committed_entries(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT * FROM ledger_entries
                WHERE entry_id IN (
                    SELECT MAX(entry_id) FROM ledger_entries GROUP BY shard_id
                )
                ORDER BY shard_id
                """
            ).fetchall()
            return {
                str(row["shard_id"]): self._row_payload(row) for row in rows
            }

    def committed_entry(self, entry_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM ledger_entries WHERE entry_id = ?", (int(entry_id),)
            ).fetchone()
            return None if row is None else self._row_payload(row)

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                """
                SELECT * FROM snapshots
                WHERE is_current = 1 AND is_stale = 0
                ORDER BY seq DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload_json"])
            return {
                "seq": int(row["seq"]),
                "generation": str(row["generation"]),
                "format_version": int(row["format_version"]),
                "cluster_version": str(row["cluster_version"]),
                "revision_chain_version": str(row["revision_chain_version"]),
                "budget_version": str(row["budget_version"]),
                "ledger_cursor": int(row["ledger_cursor"]),
                "content_digest": str(row["content_digest"]),
                "created_at_ms": int(row["created_at_ms"]),
                "payload": payload,
            }

    def current_view(self) -> tuple[dict[str, Any], ...]:
        with self.lock:
            row = self.connection.execute(
                "SELECT generation FROM snapshots WHERE is_current = 1 AND is_stale = 0"
            ).fetchone()
            if row is None:
                return ()
            rows = self.connection.execute(
                """
                SELECT * FROM view_shards
                WHERE generation = ? AND is_stale = 0 ORDER BY shard_id
                """,
                (str(row["generation"]),),
            ).fetchall()
            result = []
            for item in rows:
                payload = json.loads(item["payload_json"])
                payload["_ledger_entry_id"] = item["ledger_entry_id"]
                payload["_generation"] = str(item["generation"])
                result.append(payload)
            return tuple(result)

    def current_invalidations(self) -> tuple[dict[str, Any], ...]:
        with self.lock:
            row = self.connection.execute(
                "SELECT generation FROM snapshots WHERE is_current = 1 AND is_stale = 0"
            ).fetchone()
            if row is None:
                return ()
            rows = self.connection.execute(
                """
                SELECT * FROM invalidations
                WHERE generation = ? AND is_stale = 0
                ORDER BY revision_version, shard_id, invalidation_id
                """,
                (str(row["generation"]),),
            ).fetchall()
            return tuple(
                {
                    "revision_version": int(item["revision_version"]),
                    "shard_id": str(item["shard_id"]),
                    "stable_id": str(item["shard_id"]),
                    "kind": str(item["kind"]),
                    "basis": json.loads(item["basis_json"]),
                }
                for item in rows
            )

    def current_stats(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT generation FROM snapshots WHERE is_current = 1 AND is_stale = 0"
            ).fetchone()
            if row is None:
                return None
            stats = self.connection.execute(
                "SELECT payload_json FROM generation_stats WHERE generation = ?",
                (str(row["generation"]),),
            ).fetchone()
            return None if stats is None else json.loads(stats["payload_json"])

    def invalidate_current_view(self) -> None:
        """Mark every current conclusion stale without deleting history."""

        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "UPDATE snapshots SET is_current = 0, is_stale = 1 WHERE is_current = 1"
                )
                self.connection.execute("UPDATE view_shards SET is_stale = 1 WHERE is_stale = 0")
                self.connection.execute("UPDATE invalidations SET is_stale = 1 WHERE is_stale = 0")
                self.connection.execute("UPDATE generation_stats SET is_stale = 1 WHERE is_stale = 0")
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def save_snapshot(
        self,
        *,
        generation: str,
        cluster_version: str,
        revision_chain_version: str,
        budget_version: str,
        ledger_cursor: int,
        payload: Mapping[str, Any],
        shards: Iterable[Mapping[str, Any]],
        invalidations: Iterable[Mapping[str, Any]],
        stats: Mapping[str, Any],
        invalidated_reason: str | None = None,
    ) -> dict[str, Any]:
        """Replace the current snapshot and all current-view tables atomically."""

        normalized_payload = json.loads(canonical_json(payload))
        content_digest = digest_json(normalized_payload)
        now = monotonic_ms()
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "UPDATE snapshots SET is_current = 0, is_stale = 1 WHERE is_current = 1"
                )
                self.connection.execute("UPDATE view_shards SET is_stale = 1 WHERE is_stale = 0")
                self.connection.execute("UPDATE invalidations SET is_stale = 1 WHERE is_stale = 0")
                self.connection.execute("UPDATE generation_stats SET is_stale = 1 WHERE is_stale = 0")
                existing = self.connection.execute(
                    "SELECT seq FROM snapshots WHERE generation = ?", (generation,)
                ).fetchone()
                if existing is None:
                    self.connection.execute(
                        """
                        INSERT INTO snapshots(
                            generation, format_version, is_current, is_stale,
                            cluster_version, revision_chain_version, budget_version,
                            ledger_cursor, payload_json, content_digest, created_at_ms
                        ) VALUES(?, ?, 1, 0, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            generation,
                            SNAPSHOT_FORMAT_VERSION,
                            cluster_version,
                            revision_chain_version,
                            budget_version,
                            int(ledger_cursor),
                            canonical_json(normalized_payload),
                            content_digest,
                            now,
                        ),
                    )
                else:
                    self.connection.execute(
                        """
                        UPDATE snapshots SET
                            format_version = ?, is_current = 1, is_stale = 0,
                            cluster_version = ?, revision_chain_version = ?,
                            budget_version = ?, ledger_cursor = ?,
                            payload_json = ?, content_digest = ?, created_at_ms = ?
                        WHERE generation = ?
                        """,
                        (
                            SNAPSHOT_FORMAT_VERSION,
                            cluster_version,
                            revision_chain_version,
                            budget_version,
                            int(ledger_cursor),
                            canonical_json(normalized_payload),
                            content_digest,
                            now,
                            generation,
                        ),
                    )
                    self.connection.execute(
                        "DELETE FROM view_shards WHERE generation = ?", (generation,)
                    )
                    self.connection.execute(
                        "DELETE FROM invalidations WHERE generation = ?", (generation,)
                    )
                    self.connection.execute(
                        "DELETE FROM generation_stats WHERE generation = ?", (generation,)
                    )
                for shard in shards:
                    payload_json = canonical_json(shard)
                    self.connection.execute(
                        """
                        INSERT INTO view_shards(
                            generation, shard_id, cluster_id, task_id, shard_index,
                            revision_version, compile_digest, status, payload_json,
                            ledger_entry_id, is_stale
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            generation,
                            str(shard["stable_id"]),
                            str(shard["cluster_id"]),
                            str(shard["task_id"]),
                            int(shard["shard_index"]),
                            int(shard["revision_version"]),
                            str(shard["compile_digest"]),
                            str(shard["status"]),
                            payload_json,
                            shard.get("ledger_entry_id"),
                        ),
                    )
                for invalidation in invalidations:
                    self.connection.execute(
                        """
                        INSERT INTO invalidations(
                            generation, revision_version, shard_id, kind,
                            basis_json, is_stale
                        ) VALUES(?, ?, ?, ?, ?, 0)
                        """,
                        (
                            generation,
                            int(invalidation.get("revision_version", 0)),
                            str(invalidation["shard_id"]),
                            str(invalidation.get("kind", "rebuild")),
                            canonical_json(invalidation.get("basis", [])),
                        ),
                    )
                self.connection.execute(
                    """
                    INSERT INTO generation_stats(generation, payload_json, is_stale)
                    VALUES(?, ?, 0)
                    ON CONFLICT(generation) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        is_stale = 0
                    """,
                    (generation, canonical_json(stats)),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return {
            "generation": generation,
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "content_digest": content_digest,
            "payload": normalized_payload,
            "invalidated_reason": invalidated_reason,
        }

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
