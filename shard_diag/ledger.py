from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_json, content_digest
from .clock import monotonic_ms
from .errors import ReplayError
from .models import Spec


def first_difference(expected: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in actual:
                return child, expected[key], None
            if key not in expected:
                return child, None, actual[key]
            difference = first_difference(expected[key], actual[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}.length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return path, expected, actual
    return None


class Ledger:
    def __init__(self, path: str | Path, spec: Spec) -> None:
        self.spec = spec
        self.spec_digest = content_digest(spec.to_dict())
        self.format_version = LEDGER_FORMAT_VERSION
        self.path_text = str(path)
        self.lock = threading.RLock()

        if self.path_text != ":memory:":
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and not self._existing_snapshot_matches():
                self.replace_snapshot_files()
        else:
            self.path = Path(":memory:")

        self.conn = sqlite3.connect(self.path_text, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _existing_snapshot_matches(self) -> bool:
        check = sqlite3.connect(self.path_text)
        try:
            version = check.execute("PRAGMA user_version").fetchone()[0]
            if int(version) != self.format_version:
                return False
            row = check.execute(
                "SELECT value FROM meta WHERE key = 'spec_digest'"
            ).fetchone()
            return row is not None and row[0] == self.spec_digest
        except sqlite3.Error:
            return False
        finally:
            check.close()

    def replace_snapshot_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(self.path_text + suffix)
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def _create_schema(self) -> None:
        with self.lock:
            self.conn.execute(f"PRAGMA user_version = {self.format_version}")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    shard_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    committed_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS occupancy (
                    lease_id TEXT PRIMARY KEY,
                    shard_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resource_units INTEGER NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS progress_cursor (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_shard_id TEXT,
                    topo_index INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                """
            )
            with self.conn:
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(self.format_version),),
                )
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('spec_digest', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (self.spec_digest,),
                )
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (self.spec.session_id,),
                )

    def record_occupancy(self, lease: Any, resource_units: int) -> None:
        payload = {
            "lease": lease.to_dict(),
            "resource_units": int(resource_units),
            "recorded_at_ms": monotonic_ms(),
        }
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO occupancy(
                    lease_id, shard_id, node_id, status, resource_units,
                    capacity_version, payload_json, updated_at_ms
                ) VALUES(?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    status='active',
                    resource_units=excluded.resource_units,
                    payload_json=excluded.payload_json,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    lease.lease_id,
                    lease.shard_id,
                    lease.node_id,
                    int(resource_units),
                    int(lease.capacity_version),
                    canonical_json(payload),
                    monotonic_ms(),
                ),
            )

    def set_occupancy_status(self, lease_id: str, status: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE occupancy SET status=?, updated_at_ms=? WHERE lease_id=?",
                (status, monotonic_ms(), lease_id),
            )

    def active_occupancy(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM occupancy WHERE status='active' ORDER BY lease_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def active_resource_units(self) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(resource_units), 0) AS total "
                "FROM occupancy WHERE status='active'"
            ).fetchone()
        return int(row["total"])

    def revoke_active_occupancy_for_node(
        self, node_id: str, status: str = "capacity_revoked"
    ) -> list[dict[str, Any]]:
        active = [row for row in self.active_occupancy() if row["node_id"] == node_id]
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE occupancy SET status=?, updated_at_ms=? "
                "WHERE node_id=? AND status='active'",
                (status, monotonic_ms(), node_id),
            )
        return active

    def recover_orphaned_occupancy(self) -> list[dict[str, Any]]:
        orphaned = self.active_occupancy()
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE occupancy SET status='reclaimed', updated_at_ms=? WHERE status='active'",
                (monotonic_ms(),),
            )
        return orphaned

    def commit_outcome(self, payload: dict[str, Any]) -> dict[str, Any]:
        shard_id = payload["shard_id"]
        envelope = dict(payload)
        envelope["format_version"] = self.format_version
        envelope["spec_digest"] = self.spec_digest
        digest = content_digest(envelope)
        envelope["content_digest"] = digest
        encoded = canonical_json(envelope)
        cursor = envelope.get("cursor") or {}

        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO outcomes(
                    shard_id, format_version, spec_digest, content_digest,
                    status, payload_json, committed_at_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    format_version=excluded.format_version,
                    spec_digest=excluded.spec_digest,
                    content_digest=excluded.content_digest,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    committed_at_ms=excluded.committed_at_ms
                """,
                (
                    shard_id,
                    self.format_version,
                    self.spec_digest,
                    digest,
                    envelope.get("status", "COMMITTED"),
                    encoded,
                    monotonic_ms(),
                ),
            )
            lease = envelope.get("lease") or {}
            lease_id = lease.get("lease_id") if isinstance(lease, dict) else None
            if lease_id:
                self.conn.execute(
                    "UPDATE occupancy SET status='completed_released', updated_at_ms=? "
                    "WHERE lease_id=?",
                    (monotonic_ms(), lease_id),
                )
            self.conn.execute(
                """
                INSERT INTO progress_cursor(
                    id, last_shard_id, topo_index, position, updated_at_ms
                ) VALUES(1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_shard_id=excluded.last_shard_id,
                    topo_index=excluded.topo_index,
                    position=excluded.position,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    cursor.get("shard_id", shard_id),
                    int(cursor.get("topo_index", 0)),
                    int(cursor.get("position", 0)),
                    monotonic_ms(),
                ),
            )
        return envelope

    def get_outcome(self, shard_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT payload_json FROM outcomes WHERE shard_id=?", (shard_id,)
            ).fetchone()
        if row is None:
            return None
        import json

        return json.loads(row["payload_json"])

    def committed_ids(self) -> set[str]:
        with self.lock:
            rows = self.conn.execute("SELECT shard_id FROM outcomes").fetchall()
        return {row["shard_id"] for row in rows}

    def outcomes(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT payload_json FROM outcomes ORDER BY shard_id"
            ).fetchall()
        import json

        return [json.loads(row["payload_json"]) for row in rows]

    def latest_cursor(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM progress_cursor WHERE id=1").fetchone()
        return dict(row) if row else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "spec_digest": self.spec_digest,
            "outcomes": self.outcomes(),
            "active_occupancy": self.active_occupancy(),
            "cursor": self.latest_cursor(),
        }

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def verify_replay_view(
    expected_by_shard: dict[str, dict[str, Any]],
    actual_by_shard: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    for shard_id in order:
        expected = expected_by_shard.get(shard_id)
        actual = actual_by_shard.get(shard_id)
        if expected is None:
            raise ReplayError("replay produced no expected view", object=shard_id, field="$")
        if actual is None:
            raise ReplayError(
                "ledger is missing committed shard", object=shard_id, field="committed"
            )
        difference = first_difference(expected, actual, "$")
        if difference is not None:
            field, expected_value, actual_value = difference
            raise ReplayError(
                "ledger replay differs from recomputation",
                object=shard_id,
                field=field,
                details={"expected": expected_value, "actual": actual_value},
            )
