"""SQLite persistence for decisions, active leases, watermarks and recovery."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    LEDGER_FORMAT_VERSION,
    WATERMARK_TRUNCATION,
    canonical_json,
    sha256_value,
)
from .errors import LeaseError
from .models import ClusterSpec, LeaseEvent, SpecBundle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_versions (
    cluster_id TEXT PRIMARY KEY,
    parameter_version INTEGER NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    lease_duration_ms INTEGER NOT NULL,
    renewal_interval_ms INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    priority_digest TEXT NOT NULL,
    updated_commit_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    cluster_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    holder_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    used_capacity INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    watermark_fixed INTEGER NOT NULL,
    watermark_scale INTEGER NOT NULL,
    watermark_truncation TEXT NOT NULL,
    ledger_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    commit_id TEXT NOT NULL,
    PRIMARY KEY (cluster_id, event_seq, parameter_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_decisions_current
    ON decisions(cluster_id, event_seq) WHERE is_stale = 0;
CREATE INDEX IF NOT EXISTS ix_decisions_cluster_seq
    ON decisions(cluster_id, event_seq);

CREATE TABLE IF NOT EXISTS active_leases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    parameter_version INTEGER NOT NULL,
    state TEXT NOT NULL,
    priority INTEGER NOT NULL,
    started_ms INTEGER NOT NULL,
    last_renewed_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    renewal_count INTEGER NOT NULL,
    last_event_seq INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    commit_id TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_current
    ON active_leases(cluster_id, holder_id) WHERE is_stale = 0;

CREATE TABLE IF NOT EXISTS ownership_intervals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    parameter_version INTEGER NOT NULL,
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER,
    start_event_seq INTEGER NOT NULL,
    end_event_seq INTEGER,
    end_reason TEXT,
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    commit_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ownership_cluster_holder
    ON ownership_intervals(cluster_id, holder_id, parameter_version, started_ms);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ownership_open
    ON ownership_intervals(cluster_id, holder_id, parameter_version)
    WHERE ended_ms IS NULL AND is_stale = 0;

CREATE TABLE IF NOT EXISTS watermarks (
    cluster_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    used_capacity INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    watermark_fixed INTEGER NOT NULL,
    watermark_scale INTEGER NOT NULL,
    watermark_truncation TEXT NOT NULL,
    watermark_digest TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    commit_id TEXT NOT NULL,
    PRIMARY KEY (cluster_id, event_seq, parameter_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_watermarks_current
    ON watermarks(cluster_id, event_seq) WHERE is_stale = 0;

CREATE TABLE IF NOT EXISTS progress (
    cluster_id TEXT PRIMARY KEY,
    parameter_version INTEGER NOT NULL,
    last_event_seq INTEGER NOT NULL,
    last_timestamp_ms INTEGER NOT NULL,
    commit_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_runs (
    batch_id TEXT PRIMARY KEY,
    from_seq INTEGER NOT NULL,
    to_seq INTEGER NOT NULL,
    cluster_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    reason TEXT,
    decision_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_batch_runs_status ON batch_runs(status, from_seq);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    event_stream_digest TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    parameter_versions_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str, *, readonly: bool = False) -> None:
        self.path = str(Path(path))
        self.readonly = readonly
        if not readonly:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            uri = f"file:{Path(self.path).resolve().as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        else:
            self.conn = sqlite3.connect(self.path, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        if not readonly:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
            with self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version',?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('stream_cursor','0')"
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_content_digest','')"
                )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def stream_cursor(self) -> int:
        return int(self.get_meta("stream_cursor", "0") or 0)

    def cluster_versions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM cluster_versions ORDER BY cluster_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def cluster_version(self, cluster_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM cluster_versions WHERE cluster_id=?", (cluster_id,)
        ).fetchone()
        return dict(row) if row else None

    def cluster_specs(self) -> list[ClusterSpec]:
        return [
            ClusterSpec(
                cluster_id=row["cluster_id"],
                host=row["host"],
                port=int(row["port"]),
                lease_duration_ms=int(row["lease_duration_ms"]),
                renewal_interval_ms=int(row["renewal_interval_ms"]),
                capacity=int(row["capacity"]),
            )
            for row in self.cluster_versions()
        ]

    def versions(self) -> dict[str, int]:
        return {
            row["cluster_id"]: int(row["parameter_version"])
            for row in self.cluster_versions()
        }

    def register_bundle(self, bundle: SpecBundle) -> None:
        """Register initial parameter version or verify an existing one."""

        existing = {row["cluster_id"]: row for row in self.cluster_versions()}
        expected_ids = {cluster.cluster_id for cluster in bundle.clusters}
        extra_ids = sorted(set(existing) - expected_ids)
        if extra_ids:
            raise LeaseError(
                "E_STATE",
                f"database contains clusters absent from the declaration: {extra_ids}",
                field="clusters",
            )
        with self.conn:
            for spec in sorted(bundle.clusters, key=lambda item: item.cluster_id):
                priority_digest = bundle.priority_digest(spec.cluster_id)
                row = existing.get(spec.cluster_id)
                if row is None:
                    self.conn.execute(
                        "INSERT INTO cluster_versions("
                        "cluster_id,parameter_version,host,port,lease_duration_ms,"
                        "renewal_interval_ms,capacity,priority_digest,updated_commit_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            spec.cluster_id,
                            1,
                            spec.host,
                            spec.port,
                            spec.lease_duration_ms,
                            spec.renewal_interval_ms,
                            spec.capacity,
                            priority_digest,
                            "bootstrap",
                        ),
                    )
                    self.conn.execute(
                        "INSERT INTO progress(cluster_id,parameter_version,last_event_seq,"
                        "last_timestamp_ms,commit_id) VALUES(?,?,?,?,?)",
                        (spec.cluster_id, 1, 0, 0, "bootstrap"),
                    )
                    continue
                differences = self._spec_differences(row, spec, priority_digest)
                if differences:
                    raise LeaseError(
                        "E_STATE",
                        "cluster parameters changed; run the evolve command instead of run",
                        field="clusters",
                        details={
                            "cluster_id": spec.cluster_id,
                            "parameter_version": row["parameter_version"],
                            "differences": differences,
                        },
                    )
            self._update_content_digest()

    @staticmethod
    def _spec_differences(
        row: dict[str, Any], spec: ClusterSpec, priority_digest: str
    ) -> list[dict[str, Any]]:
        differences: list[dict[str, Any]] = []
        for field, new_value in (
            ("host", spec.host),
            ("port", spec.port),
            ("lease_duration_ms", spec.lease_duration_ms),
            ("renewal_interval_ms", spec.renewal_interval_ms),
            ("capacity", spec.capacity),
            ("priority_digest", priority_digest),
        ):
            old_value = row[field]
            if old_value != new_value:
                differences.append({"field": field, "old": old_value, "new": new_value})
        return differences

    def progress_for(self, cluster_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM progress WHERE cluster_id=?", (cluster_id,)
        ).fetchone()
        if row is None:
            return {
                "cluster_id": cluster_id,
                "parameter_version": 0,
                "last_event_seq": 0,
                "last_timestamp_ms": 0,
            }
        return dict(row)

    def current_active(self, cluster_id: str | None = None) -> list[dict[str, Any]]:
        if cluster_id is None:
            rows = self.conn.execute(
                "SELECT * FROM active_leases WHERE is_stale=0 "
                "ORDER BY cluster_id,holder_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM active_leases WHERE is_stale=0 AND cluster_id=? "
                "ORDER BY holder_id",
                (cluster_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def current_decisions(
        self, cluster_id: str | None = None, parameter_version: int | None = None
    ) -> list[dict[str, Any]]:
        conditions = ["is_stale=0"]
        parameters: list[Any] = []
        if cluster_id is not None:
            conditions.append("cluster_id=?")
            parameters.append(cluster_id)
        if parameter_version is not None:
            conditions.append("parameter_version=?")
            parameters.append(parameter_version)
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE " + " AND ".join(conditions) + " "
            "ORDER BY cluster_id,event_seq",
            parameters,
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def decision_for(self, cluster_id: str, event_seq: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE is_stale=0 AND cluster_id=? AND event_seq=?",
            (cluster_id, event_seq),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return item

    def current_watermarks(
        self, cluster_id: str | None = None
    ) -> list[dict[str, Any]]:
        if cluster_id is None:
            rows = self.conn.execute(
                "SELECT * FROM watermarks WHERE is_stale=0 "
                "ORDER BY cluster_id,event_seq"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM watermarks WHERE is_stale=0 AND cluster_id=? "
                "ORDER BY event_seq",
                (cluster_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def current_ownership_intervals(
        self, cluster_id: str | None = None
    ) -> list[dict[str, Any]]:
        if cluster_id is None:
            rows = self.conn.execute(
                "SELECT * FROM ownership_intervals WHERE is_stale=0 "
                "ORDER BY cluster_id,started_ms,holder_id,id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM ownership_intervals WHERE is_stale=0 AND cluster_id=? "
                "ORDER BY started_ms,holder_id,id",
                (cluster_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def batch_runs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM batch_runs ORDER BY from_seq,batch_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM batch_runs WHERE status=? ORDER BY from_seq,batch_id",
                (status,),
            ).fetchall()
        return [dict(row) for row in rows]

    def snapshot_catalog(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM snapshots ORDER BY created_at_utc,snapshot_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def stale_records(self, cluster_ids: Iterable[str]) -> dict[str, Any]:
        ids = sorted(set(cluster_ids))
        if not ids:
            return {
                "active_leases": [],
                "decisions": [],
                "ownership_intervals": [],
                "watermarks": [],
            }
        placeholders = ",".join("?" for _ in ids)
        decisions = self.conn.execute(
            "SELECT cluster_id,event_seq,parameter_version,ledger_digest FROM decisions "
            f"WHERE is_stale=1 AND cluster_id IN ({placeholders}) "
            "ORDER BY cluster_id,parameter_version,event_seq",
            ids,
        ).fetchall()
        active = self.conn.execute(
            "SELECT cluster_id,holder_id,parameter_version,expires_ms FROM active_leases "
            f"WHERE is_stale=1 AND cluster_id IN ({placeholders}) "
            "ORDER BY cluster_id,parameter_version,holder_id",
            ids,
        ).fetchall()
        watermarks = self.conn.execute(
            "SELECT cluster_id,event_seq,parameter_version,watermark_fixed FROM watermarks "
            f"WHERE is_stale=1 AND cluster_id IN ({placeholders}) "
            "ORDER BY cluster_id,parameter_version,event_seq",
            ids,
        ).fetchall()
        intervals = self.conn.execute(
            "SELECT cluster_id,holder_id,parameter_version,started_ms,ended_ms,"
            "start_event_seq,end_event_seq,end_reason FROM ownership_intervals "
            f"WHERE is_stale=1 AND cluster_id IN ({placeholders}) "
            "ORDER BY cluster_id,parameter_version,started_ms,holder_id,id",
            ids,
        ).fetchall()
        return {
            "active_leases": [dict(row) for row in active],
            "decisions": [dict(row) for row in decisions],
            "ownership_intervals": [dict(row) for row in intervals],
            "watermarks": [dict(row) for row in watermarks],
        }

    def record_snapshot(self, snapshot: dict[str, Any], path: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO snapshots(snapshot_id,path,format_version,content_digest,"
                "event_stream_digest,created_at_utc,parameter_versions_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(snapshot_id) DO UPDATE SET "
                "path=excluded.path,format_version=excluded.format_version,"
                "content_digest=excluded.content_digest,event_stream_digest=excluded.event_stream_digest,"
                "created_at_utc=excluded.created_at_utc,parameter_versions_json=excluded.parameter_versions_json",
                (
                    snapshot["snapshot_id"],
                    path,
                    snapshot["format_version"],
                    snapshot["content_digest"],
                    snapshot["event_stream_digest"],
                    snapshot["created_at_utc"],
                    canonical_json(snapshot["parameter_versions"]),
                ),
            )

    @staticmethod
    def decision_payload(event: LeaseEvent, response: dict[str, Any]) -> dict[str, Any]:
        active = [
            {
                "expires_ms": int(holder["expires_ms"]),
                "holder_id": holder["holder_id"],
                "state": holder["state"],
                "started_ms": int(holder["started_ms"]),
            }
            for holder in sorted(response["active_holders"], key=lambda item: item["holder_id"])
        ]
        return {
            "active_leases": active,
            "capacity": int(response["capacity"]),
            "cluster_id": event.cluster_id,
            "conclusion": str(response["decision"]),
            "event_seq": event.event_seq,
            "reason": str(response["reason"]),
            "used_capacity": int(response["used_capacity"]),
            "watermark_fixed": int(response["watermark_fixed"]),
            "watermark_scale": int(response["watermark_scale"]),
        }

    def _decision_row(
        self, event: LeaseEvent, response: dict[str, Any], commit_id: str
    ) -> tuple[Any, ...]:
        payload = self.decision_payload(event, response)
        digest = sha256_value(payload)
        stored_payload = dict(response)
        stored_payload["active_leases"] = payload["active_leases"]
        stored_payload["conclusion"] = payload["conclusion"]
        return (
            event.cluster_id,
            event.event_seq,
            int(response["parameter_version"]),
            event.holder_id,
            event.event_type,
            event.timestamp_ms,
            str(response["decision"]),
            str(response["reason"]),
            int(response["used_capacity"]),
            int(response["capacity"]),
            int(response["watermark_fixed"]),
            int(response["watermark_scale"]),
            WATERMARK_TRUNCATION,
            digest,
            canonical_json(stored_payload),
            0,
            None,
            commit_id,
        )

    def _insert_decision(
        self, event: LeaseEvent, response: dict[str, Any], commit_id: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO decisions("
            "cluster_id,event_seq,parameter_version,holder_id,event_type,timestamp_ms,"
            "decision,reason,used_capacity,capacity,watermark_fixed,watermark_scale,"
            "watermark_truncation,ledger_digest,payload_json,is_stale,stale_reason,commit_id"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self._decision_row(event, response, commit_id),
        )
        watermark_payload = {
            "capacity": int(response["capacity"]),
            "cluster_id": event.cluster_id,
            "event_seq": event.event_seq,
            "parameter_version": int(response["parameter_version"]),
            "used_capacity": int(response["used_capacity"]),
            "watermark_fixed": int(response["watermark_fixed"]),
            "watermark_scale": int(response["watermark_scale"]),
            "watermark_truncation": WATERMARK_TRUNCATION,
        }
        self.conn.execute(
            "INSERT INTO watermarks("
            "cluster_id,event_seq,parameter_version,used_capacity,capacity,watermark_fixed,"
            "watermark_scale,watermark_truncation,watermark_digest,is_stale,stale_reason,commit_id"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.cluster_id,
                event.event_seq,
                int(response["parameter_version"]),
                int(response["used_capacity"]),
                int(response["capacity"]),
                int(response["watermark_fixed"]),
                int(response["watermark_scale"]),
                WATERMARK_TRUNCATION,
                sha256_value(watermark_payload),
                0,
                None,
                commit_id,
            ),
        )
        self._apply_ownership_transitions(event, response, commit_id)

    def _apply_ownership_transitions(
        self,
        event: LeaseEvent,
        response: dict[str, Any],
        commit_id: str,
    ) -> None:
        parameter_version = int(response["parameter_version"])
        for transition in response.get("transitions", []):
            holder_id = transition["holder_id"]
            after_state = transition["after_state"]
            transition_ms = int(transition["timestamp_ms"])
            if after_state == "granted":
                self.conn.execute(
                    "INSERT INTO ownership_intervals("
                    "cluster_id,holder_id,parameter_version,started_ms,ended_ms,"
                    "start_event_seq,end_event_seq,end_reason,is_stale,stale_reason,commit_id"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event.cluster_id,
                        holder_id,
                        parameter_version,
                        transition_ms,
                        None,
                        int(transition["event_seq"]),
                        None,
                        None,
                        0,
                        None,
                        commit_id,
                    ),
                )
            elif after_state in ("released", "expired"):
                self.conn.execute(
                    "UPDATE ownership_intervals SET ended_ms=?,end_event_seq=?,end_reason=? "
                    "WHERE cluster_id=? AND holder_id=? AND parameter_version=? "
                    "AND ended_ms IS NULL AND is_stale=0",
                    (
                        transition_ms,
                        int(transition["event_seq"]),
                        transition["reason"],
                        event.cluster_id,
                        holder_id,
                        parameter_version,
                    ),
                )

    def _replace_active_for_cluster(
        self,
        cluster_id: str,
        parameter_version: int,
        active_holders: Iterable[dict[str, Any]],
        commit_id: str,
    ) -> None:
        self.conn.execute(
            "UPDATE active_leases SET is_stale=1, stale_reason='current_view_replaced' "
            "WHERE cluster_id=? AND is_stale=0",
            (cluster_id,),
        )
        for holder in sorted(active_holders, key=lambda item: item["holder_id"]):
            self.conn.execute(
                "INSERT INTO active_leases("
                "cluster_id,holder_id,parameter_version,state,priority,started_ms,"
                "last_renewed_ms,expires_ms,renewal_count,last_event_seq,provenance,"
                "is_stale,stale_reason,commit_id"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cluster_id,
                    holder["holder_id"],
                    parameter_version,
                    holder["state"],
                    int(holder.get("priority", 0)),
                    int(holder["started_ms"]),
                    int(holder["last_renewed_ms"]),
                    int(holder["expires_ms"]),
                    int(holder["renewal_count"]),
                    int(holder["last_event_seq"]),
                    holder.get("provenance", "arbitration_service"),
                    0,
                    None,
                    commit_id,
                ),
            )

    def record_inflight(
        self,
        batch_id: str,
        events: list[LeaseEvent],
        cluster_ids: list[str],
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO batch_runs(batch_id,from_seq,to_seq,cluster_ids_json,status,"
                "started_at_utc,completed_at_utc,reason,decision_count) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    min(event.event_seq for event in events),
                    max(event.event_seq for event in events),
                    canonical_json(sorted(cluster_ids)),
                    "inflight",
                    utc_now(),
                    None,
                    None,
                    0,
                ),
            )

    def commit_batch(
        self,
        batch_id: str,
        events: list[LeaseEvent],
        responses: dict[int, dict[str, Any]],
        commit_id: str,
    ) -> None:
        if not events:
            raise LeaseError("E_STATE", "cannot commit an empty batch", field="events")
        last_seq = max(event.event_seq for event in events)
        versions = self.versions()
        with self.conn:
            final_active_by_cluster: dict[str, dict[str, Any]] = {}
            final_time_by_cluster: dict[str, int] = {}
            for event in sorted(events, key=lambda item: item.event_seq):
                response = responses[event.event_seq]
                expected_version = versions.get(event.cluster_id)
                if expected_version is None:
                    raise LeaseError("E_STATE", f"unknown cluster {event.cluster_id}", field="cluster_id")
                if int(response["parameter_version"]) != expected_version:
                    raise LeaseError(
                        "E_STATE",
                        "decision parameter version does not match the current ledger version",
                        field="parameter_version",
                    )
                existing = self.decision_for(event.cluster_id, event.event_seq)
                if existing is not None:
                    expected_digest = sha256_value(self.decision_payload(event, response))
                    if existing["ledger_digest"] != expected_digest:
                        raise LeaseError(
                            "E_STATE",
                            "event sequence is already committed with a different decision",
                            field="event_seq",
                            details={"cluster_id": event.cluster_id, "event_seq": event.event_seq},
                        )
                    continue
                self._insert_decision(event, response, commit_id)
                final_active_by_cluster[event.cluster_id] = response
                final_time_by_cluster[event.cluster_id] = event.timestamp_ms

            for cluster_id in sorted(final_active_by_cluster):
                final_response = final_active_by_cluster[cluster_id]
                parameter_version = int(final_response["parameter_version"])
                self._replace_active_for_cluster(
                    cluster_id, parameter_version, final_response["active_holders"], commit_id
                )
                self.conn.execute(
                    "INSERT INTO progress(cluster_id,parameter_version,last_event_seq,"
                    "last_timestamp_ms,commit_id) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(cluster_id) DO UPDATE SET "
                    "parameter_version=excluded.parameter_version,"
                    "last_event_seq=MAX(progress.last_event_seq,excluded.last_event_seq),"
                    "last_timestamp_ms=MAX(progress.last_timestamp_ms,excluded.last_timestamp_ms),"
                    "commit_id=excluded.commit_id",
                    (
                        cluster_id,
                        parameter_version,
                        max(event.event_seq for event in events if event.cluster_id == cluster_id),
                        final_time_by_cluster[cluster_id],
                        commit_id,
                    ),
                )
            self.conn.execute(
                "UPDATE batch_runs SET status='committed',completed_at_utc=?,"
                "decision_count=? WHERE batch_id=?",
                (utc_now(), len(events), batch_id),
            )
            current_cursor = self.stream_cursor()
            self.set_meta("stream_cursor", str(max(current_cursor, last_seq)))
            self._update_content_digest()

    def mark_inflight_unknown(self) -> list[dict[str, Any]]:
        with self.conn:
            rows = self.conn.execute(
                "SELECT * FROM batch_runs WHERE status='inflight' ORDER BY from_seq,batch_id"
            ).fetchall()
            unknown = []
            for row in rows:
                reason = (
                    "uncommitted transaction observed after forced termination; "
                    "events must be rerun and no exit report is available"
                )
                self.conn.execute(
                    "UPDATE batch_runs SET status='unknown',reason=?,completed_at_utc=? "
                    "WHERE batch_id=?",
                    (reason, utc_now(), row["batch_id"]),
                )
                item = dict(row)
                item["status"] = "unknown"
                item["reason"] = reason
                item["rerun_required"] = True
                unknown.append(item)
            return unknown

    def commit_evolution(
        self,
        plans: dict[str, dict[str, Any]],
        all_specs: dict[str, ClusterSpec],
        bundle: SpecBundle,
        commit_id: str,
    ) -> None:
        """Atomically replace affected clusters after parameter evolution.

        ``plans`` contains the fully replayed decisions for each affected
        cluster.  Old current rows are retained as stale read-only audit rows;
        no old row participates in the new current view.
        """

        with self.conn:
            for cluster_id in sorted(plans):
                plan = plans[cluster_id]
                old_version = plan.get("old_version")
                if old_version is not None:
                    for table in (
                        "decisions",
                        "active_leases",
                        "ownership_intervals",
                        "watermarks",
                    ):
                        self.conn.execute(
                            f"UPDATE {table} SET is_stale=1, stale_reason='parameter_version_changed' "
                            "WHERE cluster_id=? AND is_stale=0",
                            (cluster_id,),
                        )
                spec = all_specs[cluster_id]
                version = int(plan["new_version"])
                self.conn.execute(
                    "INSERT INTO cluster_versions("
                    "cluster_id,parameter_version,host,port,lease_duration_ms,"
                    "renewal_interval_ms,capacity,priority_digest,updated_commit_id"
                    ") VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(cluster_id) DO UPDATE SET "
                    "parameter_version=excluded.parameter_version,host=excluded.host,"
                    "port=excluded.port,lease_duration_ms=excluded.lease_duration_ms,"
                    "renewal_interval_ms=excluded.renewal_interval_ms,capacity=excluded.capacity,"
                    "priority_digest=excluded.priority_digest,updated_commit_id=excluded.updated_commit_id",
                    (
                        cluster_id,
                        version,
                        spec.host,
                        spec.port,
                        spec.lease_duration_ms,
                        spec.renewal_interval_ms,
                        spec.capacity,
                        bundle.priority_digest(cluster_id),
                        commit_id,
                    ),
                )
                events: list[LeaseEvent] = plan["events"]
                responses: dict[int, dict[str, Any]] = plan["responses"]
                for event in sorted(events, key=lambda item: item.event_seq):
                    response = responses[event.event_seq]
                    self._insert_decision(event, response, commit_id)
                final_response: dict[str, Any] | None = None
                for event in sorted(events, key=lambda item: item.event_seq, reverse=True):
                    final_response = responses[event.event_seq]
                    break
                if final_response is None:
                    final_response = {
                        "active_holders": [],
                        "parameter_version": version,
                    }
                self._replace_active_for_cluster(
                    cluster_id, version, final_response["active_holders"], commit_id
                )
                last_seq = int(plan.get("last_event_seq", 0))
                last_timestamp = int(plan.get("last_timestamp_ms", 0))
                self.conn.execute(
                    "INSERT INTO progress(cluster_id,parameter_version,last_event_seq,"
                    "last_timestamp_ms,commit_id) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(cluster_id) DO UPDATE SET "
                    "parameter_version=excluded.parameter_version,"
                    "last_event_seq=excluded.last_event_seq,"
                    "last_timestamp_ms=excluded.last_timestamp_ms,commit_id=excluded.commit_id",
                    (cluster_id, version, last_seq, last_timestamp, commit_id),
                )
            self._update_content_digest()

    def ledger_content(self) -> dict[str, Any]:
        versions = {row["cluster_id"]: row for row in self.cluster_versions()}
        clusters: list[dict[str, Any]] = []
        for cluster_id in sorted(versions):
            version = versions[cluster_id]
            decisions = self.current_decisions(cluster_id)
            active = self.current_active(cluster_id)
            intervals = self.current_ownership_intervals(cluster_id)
            watermarks = self.current_watermarks(cluster_id)
            progress = self.progress_for(cluster_id)
            clusters.append(
                {
                    "active_leases": [
                        {
                            "expires_ms": row["expires_ms"],
                            "holder_id": row["holder_id"],
                            "last_renewed_ms": row["last_renewed_ms"],
                            "renewal_count": row["renewal_count"],
                            "state": row["state"],
                        }
                        for row in active
                    ],
                    "cluster_id": cluster_id,
                    "decisions": [
                        {
                            "decision": row["decision"],
                            "event_seq": row["event_seq"],
                            "ledger_digest": row["ledger_digest"],
                            "reason": row["reason"],
                        }
                        for row in decisions
                    ],
                    "parameter_version": int(version["parameter_version"]),
                    "parameters": {
                        "capacity": version["capacity"],
                        "lease_duration_ms": version["lease_duration_ms"],
                        "priority_digest": version["priority_digest"],
                        "renewal_interval_ms": version["renewal_interval_ms"],
                    },
                    "ownership_intervals": [
                        {
                            "ended_ms": row["ended_ms"],
                            "end_event_seq": row["end_event_seq"],
                            "end_reason": row["end_reason"],
                            "holder_id": row["holder_id"],
                            "start_event_seq": row["start_event_seq"],
                            "started_ms": row["started_ms"],
                        }
                        for row in intervals
                    ],
                    "progress": {
                        "last_event_seq": progress["last_event_seq"],
                        "last_timestamp_ms": progress["last_timestamp_ms"],
                    },
                    "watermarks": [
                        {
                            "event_seq": row["event_seq"],
                            "watermark_digest": row["watermark_digest"],
                            "watermark_fixed": row["watermark_fixed"],
                        }
                        for row in watermarks
                    ],
                }
            )
        return {"clusters": clusters, "format_version": LEDGER_FORMAT_VERSION}

    def content_digest(self) -> str:
        return sha256_value(self.ledger_content())

    def _update_content_digest(self) -> None:
        self.set_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
        self.set_meta("ledger_content_digest", self.content_digest())

    def export_state(self) -> dict[str, Any]:
        return {
            "clusters": self.ledger_content()["clusters"],
            "decision_rows": self.current_decisions(),
            "format_version": LEDGER_FORMAT_VERSION,
            "ledger_digest": self.content_digest(),
            "stream_cursor": self.stream_cursor(),
        }

    def restore_snapshot_state(self, snapshot: dict[str, Any], *, commit_id: str) -> None:
        """Wholesale replace the current view from a validated snapshot."""

        clusters = snapshot["clusters"]
        with self.conn:
            self.conn.execute("DELETE FROM decisions")
            self.conn.execute("DELETE FROM active_leases")
            self.conn.execute("DELETE FROM ownership_intervals")
            self.conn.execute("DELETE FROM watermarks")
            self.conn.execute("DELETE FROM progress")
            self.conn.execute("DELETE FROM cluster_versions")
            for cluster in sorted(clusters, key=lambda item: item["cluster_id"]):
                spec = cluster["spec"]
                self.conn.execute(
                    "INSERT INTO cluster_versions("
                    "cluster_id,parameter_version,host,port,lease_duration_ms,"
                    "renewal_interval_ms,capacity,priority_digest,updated_commit_id"
                    ") VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        cluster["cluster_id"],
                        int(cluster["parameter_version"]),
                        spec["host"],
                        int(spec["port"]),
                        int(spec["lease_duration_ms"]),
                        int(spec["renewal_interval_ms"]),
                        int(spec["capacity"]),
                        cluster["priority_digest"],
                        commit_id,
                    ),
                )
                for decision in sorted(
                    cluster.get("decisions", []), key=lambda item: item["event_seq"]
                ):
                    payload = decision["payload"]
                    self.conn.execute(
                        "INSERT INTO decisions("
                        "cluster_id,event_seq,parameter_version,holder_id,event_type,timestamp_ms,"
                        "decision,reason,used_capacity,capacity,watermark_fixed,watermark_scale,"
                        "watermark_truncation,ledger_digest,payload_json,is_stale,stale_reason,commit_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cluster["cluster_id"],
                            int(decision["event_seq"]),
                            int(cluster["parameter_version"]),
                            decision["holder_id"],
                            decision["event_type"],
                            int(decision["timestamp_ms"]),
                            decision["decision"],
                            decision["reason"],
                            int(decision["used_capacity"]),
                            int(decision["capacity"]),
                            int(decision["watermark_fixed"]),
                            int(decision["watermark_scale"]),
                            decision["watermark_truncation"],
                            decision["ledger_digest"],
                            canonical_json(payload),
                            0,
                            None,
                            commit_id,
                        ),
                    )
                    self.conn.execute(
                        "INSERT INTO watermarks("
                        "cluster_id,event_seq,parameter_version,used_capacity,capacity,watermark_fixed,"
                        "watermark_scale,watermark_truncation,watermark_digest,is_stale,stale_reason,commit_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cluster["cluster_id"],
                            int(decision["event_seq"]),
                            int(cluster["parameter_version"]),
                            int(decision["used_capacity"]),
                            int(decision["capacity"]),
                            int(decision["watermark_fixed"]),
                            int(decision["watermark_scale"]),
                            decision["watermark_truncation"],
                            decision.get(
                                "watermark_digest",
                                sha256_value(
                                    {
                                        "capacity": int(decision["capacity"]),
                                        "cluster_id": cluster["cluster_id"],
                                        "event_seq": int(decision["event_seq"]),
                                        "parameter_version": int(cluster["parameter_version"]),
                                        "used_capacity": int(decision["used_capacity"]),
                                        "watermark_fixed": int(decision["watermark_fixed"]),
                                        "watermark_scale": int(decision["watermark_scale"]),
                                        "watermark_truncation": decision["watermark_truncation"],
                                    }
                                ),
                            ),
                            0,
                            None,
                            commit_id,
                        ),
                    )
                for lease in sorted(cluster.get("active_leases", []), key=lambda item: item["holder_id"]):
                    self.conn.execute(
                        "INSERT INTO active_leases("
                        "cluster_id,holder_id,parameter_version,state,priority,started_ms,"
                        "last_renewed_ms,expires_ms,renewal_count,last_event_seq,provenance,"
                        "is_stale,stale_reason,commit_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cluster["cluster_id"],
                            lease["holder_id"],
                            int(cluster["parameter_version"]),
                            lease["state"],
                            int(lease.get("priority", 0)),
                            int(lease["started_ms"]),
                            int(lease["last_renewed_ms"]),
                            int(lease["expires_ms"]),
                            int(lease["renewal_count"]),
                            int(lease["last_event_seq"]),
                            lease.get("provenance", "snapshot"),
                            0,
                            None,
                            commit_id,
                        ),
                    )
                for interval in sorted(
                    cluster.get("ownership_intervals", []),
                    key=lambda item: (
                        int(item["started_ms"]),
                        item["holder_id"],
                        int(item["start_event_seq"]),
                    ),
                ):
                    self.conn.execute(
                        "INSERT INTO ownership_intervals("
                        "cluster_id,holder_id,parameter_version,started_ms,ended_ms,"
                        "start_event_seq,end_event_seq,end_reason,is_stale,stale_reason,commit_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cluster["cluster_id"],
                            interval["holder_id"],
                            int(cluster["parameter_version"]),
                            int(interval["started_ms"]),
                            (
                                None
                                if interval.get("ended_ms") is None
                                else int(interval["ended_ms"])
                            ),
                            int(interval["start_event_seq"]),
                            (
                                None
                                if interval.get("end_event_seq") is None
                                else int(interval["end_event_seq"])
                            ),
                            interval.get("end_reason"),
                            0,
                            None,
                            commit_id,
                        ),
                    )
                progress = cluster["progress"]
                self.conn.execute(
                    "INSERT INTO progress(cluster_id,parameter_version,last_event_seq,"
                    "last_timestamp_ms,commit_id) VALUES(?,?,?,?,?)",
                    (
                        cluster["cluster_id"],
                        int(cluster["parameter_version"]),
                        int(progress["last_event_seq"]),
                        int(progress["last_timestamp_ms"]),
                        commit_id,
                    ),
                )
            self.set_meta("stream_cursor", str(int(snapshot.get("stream_cursor", 0))))
            self._update_content_digest()
