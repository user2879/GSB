from __future__ import annotations

import json
import sqlite3
import urllib.parse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .content import canonical_json, content_digest_from_versions, sha256_text
from .errors import PipelineError
from .fixed import FixedGrid
from .spec import Spec
from .stats import BatchRequestRecord, error_summary_from_metrics

SCHEMA_VERSION = "1"
NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    source_updates_json TEXT NOT NULL,
    initial_content_versions_json TEXT NOT NULL,
    initial_content_digest TEXT NOT NULL,
    grid_power INTEGER NOT NULL,
    grid_scale INTEGER NOT NULL,
    grid_truncation TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    batch_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    content_versions_json TEXT NOT NULL,
    grid_power INTEGER NOT NULL,
    grid_scale INTEGER NOT NULL,
    grid_truncation TEXT NOT NULL,
    cursor_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    readonly INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS batches (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    batch_id INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    request_count INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    grid_power INTEGER NOT NULL,
    grid_scale INTEGER NOT NULL,
    cursor_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    error_summary_json TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (session_id, batch_id)
);
CREATE TABLE IF NOT EXISTS node_metrics (
    session_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value_units INTEGER NOT NULL,
    error_bound_units INTEGER NOT NULL,
    dominant_error_source TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    PRIMARY KEY (session_id, batch_id, node_id, metric_name),
    FOREIGN KEY (session_id, batch_id) REFERENCES batches(session_id, batch_id)
);
CREATE TABLE IF NOT EXISTS batch_requests (
    session_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    path TEXT NOT NULL,
    query TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    hit_source TEXT NOT NULL,
    returned_etag TEXT NOT NULL,
    authoritative_etag TEXT NOT NULL,
    version_consistent INTEGER NOT NULL,
    remaining_ttl_ms INTEGER NOT NULL,
    entry_version INTEGER NOT NULL,
    stale INTEGER NOT NULL,
    error_code TEXT,
    record_json TEXT NOT NULL,
    PRIMARY KEY (session_id, batch_id, seq),
    FOREIGN KEY (session_id, batch_id) REFERENCES batches(session_id, batch_id)
);
CREATE TABLE IF NOT EXISTS active_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    snapshot_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    grid_power INTEGER NOT NULL,
    grid_scale INTEGER NOT NULL,
    conclusions_json TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS active_cache (
    node_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    PRIMARY KEY (node_id, cache_key)
);
CREATE TRIGGER IF NOT EXISTS snapshots_no_update
BEFORE UPDATE ON snapshots
BEGIN
    SELECT RAISE(ABORT, 'snapshot_is_read_only');
END;
CREATE TRIGGER IF NOT EXISTS snapshots_no_delete
BEFORE DELETE ON snapshots
BEGIN
    SELECT RAISE(ABORT, 'snapshot_is_read_only');
END;
"""


def snapshot_identity(
    *,
    session_id: str,
    batch_id: int,
    kind: str,
    cursor: dict[str, Any],
    content_digest: str,
    grid: FixedGrid,
) -> str:
    material = canonical_json(
        {
            "session_id": session_id,
            "batch_id": batch_id,
            "kind": kind,
            "cursor": cursor,
            "content_digest": content_digest,
            "grid": grid.as_dict(),
        }
    )
    return "snap-" + sha256_text(material)[:20]


class Ledger:
    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()
        else:
            if not self.path.exists():
                raise PipelineError("E_STATE", "ledger does not exist", path=str(self.path), field="file")
            quoted = urllib.parse.quote(str(self.path.resolve()), safe="/:\\")
            self.conn = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True, timeout=30.0)
            self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield self.conn
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def schema_version(self) -> str | None:
        row = self.conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
        return None if row is None else str(row["value"])

    def session_exists(self, session_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone() is not None

    def begin_session(
        self,
        spec: Spec,
        grid: FixedGrid,
        source_updates: list[dict[str, Any]],
        initial_content: dict[str, Any],
    ) -> None:
        if self.session_exists(spec.session.session_id):
            raise PipelineError(
                "E_STATE",
                "session_id already exists in the diagnostic ledger",
                path=spec.session.session_id,
                field="session_id",
                details={"ledger": str(self.path)},
            )
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions("
                "session_id,spec_json,spec_digest,source_updates_json,initial_content_versions_json,"
                "initial_content_digest,grid_power,grid_scale,grid_truncation,created_utc"
                ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    spec.session.session_id,
                    canonical_json(spec.raw),
                    sha256_text(canonical_json(spec.raw)),
                    canonical_json(source_updates),
                    canonical_json(initial_content["content_versions"]),
                    initial_content["content_digest"],
                    grid.power,
                    grid.scale,
                    grid.truncation,
                    NOW(),
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise PipelineError("E_STATE", "session not found", path=session_id, field="session_id")
        result = dict(row)
        result["spec"] = json.loads(result.pop("spec_json"))
        result["source_updates"] = json.loads(result.pop("source_updates_json"))
        result["initial_content_versions"] = json.loads(result.pop("initial_content_versions_json"))
        return result

    def _insert_snapshot_and_active(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        batch_id: int,
        kind: str,
        cursor: dict[str, Any],
        content_snapshot: dict[str, Any],
        grid: FixedGrid,
        node_states: dict[str, Any],
        conclusions: dict[str, Any],
        spec: Spec,
        source_updates: list[dict[str, Any]],
    ) -> str:
        content_digest = str(content_snapshot["content_digest"])
        snapshot_id = snapshot_identity(
            session_id=session_id,
            batch_id=batch_id,
            kind=kind,
            cursor=cursor,
            content_digest=content_digest,
            grid=grid,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "batch_id": batch_id,
            "kind": kind,
            "cursor": cursor,
            "content_snapshot": content_snapshot,
            "grid": grid.as_dict(),
            "node_states": node_states,
            "conclusions": conclusions,
            "spec": spec.raw,
            "source_updates": source_updates,
        }
        conn.execute(
            "INSERT INTO snapshots("
            "snapshot_id,session_id,batch_id,kind,content_digest,content_versions_json,"
            "grid_power,grid_scale,grid_truncation,cursor_json,payload_json,created_utc,readonly"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1)",
            (
                snapshot_id,
                session_id,
                batch_id,
                kind,
                content_digest,
                canonical_json(content_snapshot["content_versions"]),
                grid.power,
                grid.scale,
                grid.truncation,
                canonical_json(cursor),
                canonical_json(payload),
                NOW(),
            ),
        )
        conn.execute("DELETE FROM active_cache")
        for node_id in sorted(node_states):
            for entry in sorted(node_states[node_id].get("entries", []), key=lambda item: item["cache_key"]):
                conn.execute(
                    "INSERT INTO active_cache(node_id,cache_key,entry_json) VALUES(?,?,?)",
                    (node_id, entry["cache_key"], canonical_json(entry)),
                )
        conn.execute(
            "INSERT INTO active_state("
            "id,snapshot_id,session_id,batch_id,content_digest,grid_power,grid_scale,conclusions_json,updated_utc"
            ") VALUES(1,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET snapshot_id=excluded.snapshot_id,"
            "session_id=excluded.session_id,batch_id=excluded.batch_id,"
            "content_digest=excluded.content_digest,grid_power=excluded.grid_power,"
            "grid_scale=excluded.grid_scale,conclusions_json=excluded.conclusions_json,"
            "updated_utc=excluded.updated_utc",
            (
                snapshot_id,
                session_id,
                batch_id,
                content_digest,
                grid.power,
                grid.scale,
                canonical_json(conclusions),
                NOW(),
            ),
        )
        return snapshot_id

    def commit_genesis(
        self,
        *,
        spec: Spec,
        grid: FixedGrid,
        source_updates: list[dict[str, Any]],
        content_snapshot: dict[str, Any],
        node_states: dict[str, Any],
        cursor: dict[str, Any],
    ) -> str:
        conclusions = {
            "kind": "genesis",
            "request_count": 0,
            "message": "empty edge state before the first request batch",
        }
        with self.transaction() as conn:
            return self._insert_snapshot_and_active(
                conn,
                session_id=spec.session.session_id,
                batch_id=0,
                kind="genesis",
                cursor=cursor,
                content_snapshot=content_snapshot,
                grid=grid,
                node_states=node_states,
                conclusions=conclusions,
                spec=spec,
                source_updates=source_updates,
            )

    def commit_batch(
        self,
        *,
        spec: Spec,
        grid: FixedGrid,
        source_updates: list[dict[str, Any]],
        batch_id: int,
        records: list[BatchRequestRecord],
        metrics: dict[str, Any],
        content_snapshot: dict[str, Any],
        node_states: dict[str, Any],
        cursor: dict[str, Any],
        start_seq: int,
        end_seq: int,
    ) -> str:
        ordered = sorted(records, key=lambda item: item.seq)
        cursor_json = canonical_json(cursor)
        metrics_json = canonical_json(metrics)
        error_summary = error_summary_from_metrics(metrics)
        conclusions = {
            "kind": "batch",
            "metrics": metrics,
            "error_summary": error_summary,
        }
        with self.transaction() as conn:
            snapshot_id = self._insert_snapshot_and_active(
                conn,
                session_id=spec.session.session_id,
                batch_id=batch_id,
                kind="batch",
                cursor=cursor,
                content_snapshot=content_snapshot,
                grid=grid,
                node_states=node_states,
                conclusions=conclusions,
                spec=spec,
                source_updates=source_updates,
            )
            conn.execute(
                "INSERT INTO batches("
                "session_id,batch_id,snapshot_id,start_seq,end_seq,request_count,content_digest,"
                "grid_power,grid_scale,cursor_json,metrics_json,error_summary_json,created_utc"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    spec.session.session_id,
                    batch_id,
                    snapshot_id,
                    start_seq,
                    end_seq,
                    len(ordered),
                    content_snapshot["content_digest"],
                    grid.power,
                    grid.scale,
                    cursor_json,
                    metrics_json,
                    canonical_json(error_summary),
                    NOW(),
                ),
            )
            for record in ordered:
                conn.execute(
                    "INSERT INTO batch_requests("
                    "session_id,batch_id,seq,request_id,node_id,path,query,at_ms,hit_source,"
                    "returned_etag,authoritative_etag,version_consistent,remaining_ttl_ms,"
                    "entry_version,stale,error_code,record_json"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        spec.session.session_id,
                        batch_id,
                        record.seq,
                        record.request_id,
                        record.node_id,
                        record.path,
                        record.query,
                        record.at_ms,
                        record.hit_source,
                        record.returned_etag,
                        record.authoritative_etag,
                        1 if record.version_consistent else 0,
                        record.remaining_ttl_ms,
                        record.entry_version,
                        1 if record.stale else 0,
                        record.error_code,
                        canonical_json(record.as_dict()),
                    ),
                )
            node_metric_payloads: list[tuple[str, str, dict[str, Any]]] = []
            for node_id in sorted(metrics["nodes"]):
                for metric_name in sorted(metrics["nodes"][node_id]["metrics"]):
                    node_metric_payloads.append(
                        (node_id, metric_name, metrics["nodes"][node_id]["metrics"][metric_name])
                    )
            for metric_name in sorted(metrics["aggregate"]["metrics"]):
                node_metric_payloads.append(
                    ("__aggregate__", metric_name, metrics["aggregate"]["metrics"][metric_name])
                )
            for node_id, metric_name, metric in node_metric_payloads:
                conn.execute(
                    "INSERT INTO node_metrics("
                    "session_id,batch_id,node_id,metric_name,value_units,error_bound_units,"
                    "dominant_error_source,segments_json"
                    ") VALUES(?,?,?,?,?,?,?,?)",
                    (
                        spec.session.session_id,
                        batch_id,
                        node_id,
                        metric_name,
                        int(metric["value_units"]),
                        int(metric["error_bound_units"]),
                        str(metric["dominant_error_source"]),
                        canonical_json(metric["segments"]),
                    ),
                )
        return snapshot_id

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT session_id,initial_content_digest,grid_power,grid_scale,created_utc "
                "FROM sessions ORDER BY created_utc, session_id"
            ).fetchall()
        ]

    def list_snapshots(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self.conn.execute(
                "SELECT snapshot_id,session_id,batch_id,kind,content_digest,grid_power,grid_scale,"
                "cursor_json,created_utc FROM snapshots ORDER BY session_id,batch_id,snapshot_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT snapshot_id,session_id,batch_id,kind,content_digest,grid_power,grid_scale,"
                "cursor_json,created_utc FROM snapshots WHERE session_id=? ORDER BY batch_id,snapshot_id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if row is None:
            raise PipelineError("E_SNAPSHOT", "snapshot not found", path=snapshot_id, field="snapshot_id")
        result = dict(row)
        result["content_versions"] = json.loads(result.pop("content_versions_json"))
        result["cursor"] = json.loads(result.pop("cursor_json"))
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def verify_snapshot_integrity(self, snapshot: dict[str, Any]) -> None:
        recomputed = content_digest_from_versions(snapshot["content_versions"])
        if recomputed != snapshot["content_digest"]:
            raise PipelineError(
                "E_SNAPSHOT",
                "snapshot content digest does not match its content version set",
                path=snapshot["snapshot_id"],
                field="content_digest",
                details={"stored": snapshot["content_digest"], "recomputed": recomputed},
            )
        payload = snapshot["payload"]
        expected_id = snapshot_identity(
            session_id=snapshot["session_id"],
            batch_id=int(snapshot["batch_id"]),
            kind=snapshot["kind"],
            cursor=snapshot["cursor"],
            content_digest=snapshot["content_digest"],
            grid=FixedGrid(int(snapshot["grid_power"])),
        )
        if expected_id != snapshot["snapshot_id"]:
            raise PipelineError(
                "E_SNAPSHOT",
                "snapshot identity does not match its version/grid/cursor coordinates",
                path=snapshot["snapshot_id"],
                field="snapshot_id",
                details={"stored": snapshot["snapshot_id"], "recomputed": expected_id},
            )
        if payload.get("content_snapshot", {}).get("content_digest") != snapshot["content_digest"]:
            raise PipelineError(
                "E_SNAPSHOT",
                "snapshot payload content digest is inconsistent",
                path=snapshot["snapshot_id"],
                field="payload.content_snapshot.content_digest",
            )

    def active_state(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM active_state WHERE id=1").fetchone()
        if row is None:
            return None
        result = dict(row)
        result["conclusions"] = json.loads(result.pop("conclusions_json"))
        return result

    def active_cache_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM active_cache").fetchone()["n"])

    def restore_snapshot(
        self,
        snapshot_id: str,
        *,
        current_content_digest: str | None,
        current_grid: FixedGrid | None,
        expected_content_digest: str | None = None,
        mode: str = "replace",
    ) -> dict[str, Any]:
        snapshot = self.get_snapshot(snapshot_id)
        self.verify_snapshot_integrity(snapshot)
        if expected_content_digest is not None and snapshot["content_digest"] != expected_content_digest:
            raise PipelineError(
                "E_SNAPSHOT",
                "expected content digest does not match the snapshot",
                path=snapshot_id,
                field="content_digest",
                details={
                    "stored": snapshot["content_digest"],
                    "expected": expected_content_digest,
                    "basis": "caller-declared snapshot content digest",
                },
            )
        snapshot_grid = FixedGrid(int(snapshot["grid_power"]))
        mismatches: list[str] = []
        if current_content_digest is not None and current_content_digest != snapshot["content_digest"]:
            mismatches.append("content_digest")
        if current_grid is not None and current_grid != snapshot_grid:
            mismatches.append("grid")
        if mismatches and mode == "strict":
            raise PipelineError(
                "E_SNAPSHOT",
                "snapshot cannot be resumed against the current content/precision state",
                path=snapshot_id,
                field=mismatches[0],
                details={
                    "snapshot_content_digest": snapshot["content_digest"],
                    "current_content_digest": current_content_digest,
                    "snapshot_grid": snapshot_grid.as_dict(),
                    "current_grid": None if current_grid is None else current_grid.as_dict(),
                    "mismatches": mismatches,
                },
            )
        payload = snapshot["payload"]
        node_states = payload["node_states"]
        with self.transaction() as conn:
            conn.execute("DELETE FROM active_cache")
            for node_id in sorted(node_states):
                for entry in sorted(
                    node_states[node_id].get("entries", []), key=lambda item: item["cache_key"]
                ):
                    conn.execute(
                        "INSERT INTO active_cache(node_id,cache_key,entry_json) VALUES(?,?,?)",
                        (node_id, entry["cache_key"], canonical_json(entry)),
                    )
            conn.execute(
                "INSERT INTO active_state("
                "id,snapshot_id,session_id,batch_id,content_digest,grid_power,grid_scale,"
                "conclusions_json,updated_utc"
                ") VALUES(1,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "snapshot_id=excluded.snapshot_id,session_id=excluded.session_id,"
                "batch_id=excluded.batch_id,content_digest=excluded.content_digest,"
                "grid_power=excluded.grid_power,grid_scale=excluded.grid_scale,"
                "conclusions_json=excluded.conclusions_json,updated_utc=excluded.updated_utc",
                (
                    snapshot_id,
                    snapshot["session_id"],
                    int(snapshot["batch_id"]),
                    snapshot["content_digest"],
                    snapshot_grid.power,
                    snapshot_grid.scale,
                    canonical_json(payload.get("conclusions", {})),
                    NOW(),
                ),
            )
        return {
            "ok": True,
            "snapshot_id": snapshot_id,
            "mode": mode,
            "whole_replacement": True,
            "incremental_merge": False,
            "mismatches": mismatches,
            "content_digest": snapshot["content_digest"],
            "grid": snapshot_grid.as_dict(),
            "active_cache_entries": self.active_cache_count(),
            "old_snapshot_preserved_readonly": True,
            "message": (
                "content/grid mismatch: active cache, statistics and error conclusions were wholly replaced"
                if mismatches
                else "snapshot restored by whole replacement"
            ),
        }

    def get_batch_rows(self, session_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM batches WHERE session_id=? ORDER BY batch_id", (session_id,)
            ).fetchall()
        ]

    def get_batch(self, session_id: str, batch_id: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM batches WHERE session_id=? AND batch_id=?", (session_id, batch_id)
        ).fetchone()
        if row is None:
            raise PipelineError(
                "E_REPLAY",
                "batch not found in ledger",
                path=f"{session_id}/{batch_id}",
                field="batch_id",
            )
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json"))
        result["error_summary"] = json.loads(result.pop("error_summary_json"))
        result["cursor"] = json.loads(result.pop("cursor_json"))
        return result

    def get_batch_requests(self, session_id: str, batch_id: int) -> list[BatchRequestRecord]:
        rows = self.conn.execute(
            "SELECT record_json FROM batch_requests WHERE session_id=? AND batch_id=? ORDER BY seq",
            (session_id, batch_id),
        ).fetchall()
        return [BatchRequestRecord.from_dict(json.loads(row["record_json"])) for row in rows]
