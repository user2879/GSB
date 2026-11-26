"""SQLite ledger with atomic per-shard commits and snapshot replacement."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .errors import LedgerError
from .sample import canonical_json, digest_payload

LEDGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shard_outcomes (
                shard_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                capacity_version INTEGER NOT NULL,
                lease_id TEXT,
                lease_version INTEGER,
                summary_json TEXT,
                resource_units INTEGER NOT NULL,
                summary_digest TEXT NOT NULL,
                committed_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS water_marks (
                node_id TEXT NOT NULL,
                capacity_version INTEGER NOT NULL,
                high_water_ms INTEGER NOT NULL,
                shard_count INTEGER NOT NULL,
                resource_total INTEGER NOT NULL,
                PRIMARY KEY (node_id, capacity_version)
            );
            CREATE TABLE IF NOT EXISTS event_cursor (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_seq INTEGER NOT NULL,
                capacity_version INTEGER NOT NULL
            );
            """
        )

    def initialize_snapshot(
        self,
        *,
        spec_digest: str,
        node_set_version: str,
        capacity_version: int,
    ) -> tuple[bool, dict | None]:
        """Identify an existing snapshot; replace it whole when it diverges."""
        existing = self.meta_map()
        if not existing:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self._put_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
                self._put_meta("spec_digest", spec_digest)
                self._put_meta("node_set_version", node_set_version)
                self._put_meta("capacity_version", str(capacity_version))
                self.conn.execute(
                    "INSERT INTO event_cursor(id,last_seq,capacity_version) VALUES(1,-1,?)",
                    (capacity_version,),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return True, None

        mismatch = {
            "spec_digest": (existing.get("spec_digest"), spec_digest),
            "node_set_version": (
                existing.get("node_set_version"),
                node_set_version,
            ),
            "capacity_version": (
                int(existing.get("capacity_version", -1)),
                capacity_version,
            ),
        }
        incompatible = (
            existing.get("spec_digest") != spec_digest
            or existing.get("node_set_version") != node_set_version
        )
        if incompatible:
            backup = self.export_snapshot()
            self.replace_snapshot(spec_digest, node_set_version, capacity_version)
            return True, {"mismatch": mismatch, "backup": backup}
        return False, None

    def current_capacity_version(self) -> int:
        return int(self.meta_map().get("capacity_version", 0))

    def _put_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def meta_map(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key,value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def replace_snapshot(
        self, spec_digest: str, node_set_version: str, capacity_version: int
    ) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM shard_outcomes")
            self.conn.execute("DELETE FROM water_marks")
            self.conn.execute("DELETE FROM event_cursor")
            self.conn.execute("DELETE FROM meta")
            self._put_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
            self._put_meta("spec_digest", spec_digest)
            self._put_meta("node_set_version", node_set_version)
            self._put_meta("capacity_version", str(capacity_version))
            self.conn.execute(
                "INSERT INTO event_cursor(id,last_seq,capacity_version) VALUES(1,-1,?)",
                (capacity_version,),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def export_snapshot(self) -> dict:
        outcomes = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM shard_outcomes ORDER BY shard_id"
            ).fetchall()
        ]
        waters = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM water_marks ORDER BY node_id, capacity_version"
            ).fetchall()
        ]
        return {"meta": self.meta_map(), "outcomes": outcomes, "water_marks": waters}

    def commit_outcome(
        self,
        *,
        outcome: dict,
        water_marks: dict[str, dict],
        last_seq: int,
        capacity_version: int,
    ) -> None:
        """Atomically write one shard outcome, water marks, cursor and digest."""
        summary = outcome.get("summary")
        summary_digest = digest_payload(summary) if summary is not None else ""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                INSERT INTO shard_outcomes(
                    shard_id,node_id,status,capacity_version,lease_id,lease_version,
                    summary_json,resource_units,summary_digest,committed_ms
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    node_id=excluded.node_id,
                    status=excluded.status,
                    capacity_version=excluded.capacity_version,
                    lease_id=excluded.lease_id,
                    lease_version=excluded.lease_version,
                    summary_json=excluded.summary_json,
                    resource_units=excluded.resource_units,
                    summary_digest=excluded.summary_digest,
                    committed_ms=excluded.committed_ms
                """,
                (
                    outcome["shard_id"],
                    outcome["node_id"],
                    outcome["status"],
                    outcome["capacity_version"],
                    outcome.get("lease_id"),
                    outcome.get("lease_version"),
                    json.dumps(summary, sort_keys=True, separators=(",", ":")),
                    int(outcome["resource_units"]),
                    summary_digest,
                    int(outcome["committed_ms"]),
                ),
            )
            for node_id, mark in sorted(water_marks.items()):
                self.conn.execute(
                    """
                    INSERT INTO water_marks(node_id,capacity_version,high_water_ms,
                        shard_count,resource_total)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(node_id,capacity_version) DO UPDATE SET
                        high_water_ms=excluded.high_water_ms,
                        shard_count=excluded.shard_count,
                        resource_total=excluded.resource_total
                    """,
                    (
                        node_id,
                        int(mark["capacity_version"]),
                        int(mark["high_water_ms"]),
                        int(mark["shard_count"]),
                        int(mark["resource_total"]),
                    ),
                )
            self.conn.execute(
                "UPDATE event_cursor SET last_seq=?, capacity_version=? WHERE id=1",
                (int(last_seq), int(capacity_version)),
            )
            self._put_meta("capacity_version", str(capacity_version))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def get_outcome(self, shard_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM shard_outcomes WHERE shard_id=?", (shard_id,)
        ).fetchone()
        return None if row is None else self._decode_row(row)

    def all_outcomes(self) -> dict[str, dict]:
        rows = self.conn.execute("SELECT * FROM shard_outcomes").fetchall()
        return {row["shard_id"]: self._decode_row(row) for row in rows}

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        payload = dict(row)
        payload["summary"] = (
            json.loads(payload.pop("summary_json")) if payload.get("summary_json") else None
        )
        return payload

    def cursor(self) -> dict:
        row = self.conn.execute("SELECT * FROM event_cursor WHERE id=1").fetchone()
        if row is None:
            raise LedgerError("event cursor missing; snapshot not initialized", path="event_cursor")
        return dict(row)

    def water_marks(self) -> dict[tuple[str, int], dict]:
        rows = self.conn.execute("SELECT * FROM water_marks").fetchall()
        return {(row["node_id"], row["capacity_version"]): dict(row) for row in rows}

    def verify_digest(self, shard_id: str, summary: dict) -> None:
        row = self.get_outcome(shard_id)
        if row is None:
            raise LedgerError(f"no ledger record for {shard_id}", path=f"outcomes.{shard_id}")
        actual = digest_payload(summary)
        if row["summary_digest"] != actual:
            raise LedgerError(
                "summary digest mismatch against ledger",
                path=f"outcomes.{shard_id}.summary_digest",
            )

    def close(self) -> None:
        self.conn.close()

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.export_snapshot())
