from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .db import connect, connect_readonly, ensure_schema
from .errors import SnapshotError
from .model import PLUGIN_NAMES
from .scheduler import stable_conclusion
from .util import canonical_json, digest_value, monotonic_ms


def _plugin_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {name: 0 for name in PLUGIN_NAMES}
    return {name: int(value.get(name, 0)) for name in PLUGIN_NAMES}


def entry_from_result(
    result: Mapping[str, Any],
    *,
    format_versions: Iterable[int | str],
    status: str = "committed",
    cursor: int = 0,
    session_id: str | None = None,
) -> dict[str, Any]:
    versions = sorted({str(value) for value in format_versions})
    summary = {
        "pod_id": str(result["pod_id"]),
        "node_name": str(result.get("node_name", "-")),
        "total_score": int(result.get("total_score", 0)),
        "veto_code": int(result.get("veto_code", 0)),
        "plugin_scores": _plugin_map(result.get("plugin_scores", {})),
        "plugin_contributions": _plugin_map(result.get("plugin_contributions", {})),
        "error_bound": int(result.get("error_bound", 0)),
        "parameter_version": int(result.get("parameter_version", 1)),
        "scale_factor": int(result.get("scale_factor", 1)),
        "format_versions": versions,
        "status": status,
        "conclusion": stable_conclusion(result) if status == "committed" else None,
        "session_id": session_id,
        "cursor": cursor,
    }
    summary["conclusion_digest"] = (
        digest_value(summary["conclusion"]) if summary["conclusion"] is not None else digest_value({"status": status})
    )
    return summary


def entry_content_digest(entry: Mapping[str, Any]) -> str:
    return digest_value(
        {
            "pod_id": entry["pod_id"],
            "node_name": entry["node_name"],
            "total_score": entry["total_score"],
            "veto_code": entry["veto_code"],
            "plugin_scores": entry["plugin_scores"],
            "plugin_contributions": entry["plugin_contributions"],
            "error_bound": entry["error_bound"],
            "parameter_version": entry["parameter_version"],
            "scale_factor": entry["scale_factor"],
            "format_versions": entry["format_versions"],
            "status": entry["status"],
            "conclusion_digest": entry["conclusion_digest"],
        }
    )


def snapshot_content_digest(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    format_versions: Iterable[int | str],
    parameter_version: int,
    scale_factor: int,
) -> str:
    ordered = {
        pod_id: entry_content_digest(entries[pod_id]) for pod_id in sorted(entries)
    }
    return digest_value(
        {
            "schema": "affinity-ledger-snapshot-v1",
            "format_version_set": sorted({str(value) for value in format_versions}),
            "parameter_version": parameter_version,
            "scale_factor": scale_factor,
            "entries": ordered,
        }
    )


@dataclass(frozen=True)
class SyncReport:
    snapshot_id: str
    mismatch: bool
    reason: str | None
    added: tuple[str, ...]
    deleted: tuple[str, ...]
    replaced: tuple[str, ...]
    content_digest: str
    format_version_set: tuple[str, ...]
    injected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "mismatch": self.mismatch,
            "reason": self.reason,
            "replacement": {
                "added": list(self.added),
                "deleted": list(self.deleted),
                "replaced": list(self.replaced),
            },
            "content_digest": self.content_digest,
            "format_version_set": list(self.format_version_set),
            "code": "E_SNAPSHOT" if self.mismatch else None,
            "injected": self.injected,
        }


def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    conclusion = None
    try:
        conclusion = json.loads(row["conclusion_json"])
    except (TypeError, json.JSONDecodeError):
        conclusion = None
    return {
        "pod_id": row["pod_id"],
        "node_name": row["node_name"],
        "total_score": int(row["total_score"]),
        "veto_code": int(row["veto_code"]),
        "plugin_scores": json.loads(row["plugin_scores_json"]),
        "plugin_contributions": json.loads(row["plugin_contributions_json"]),
        "error_bound": int(row["error_bound"]),
        "parameter_version": int(row["parameter_version"]),
        "scale_factor": int(row["scale_factor"]),
        "format_versions": json.loads(row["format_versions_json"]),
        "status": row["status"],
        "conclusion": conclusion,
        "conclusion_digest": row["conclusion_digest"],
        "session_id": row["session_id"],
        "cursor": int(row["cursor"]),
    }


def _meta_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def _meta_set(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO ledger_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _insert_entry(
    connection: sqlite3.Connection,
    entry: Mapping[str, Any],
    *,
    session_id: str | None,
    now: int,
) -> None:
    connection.execute(
        """
        INSERT INTO ledger_entries
        (pod_id, node_name, total_score, veto_code, plugin_scores_json,
         plugin_contributions_json, error_bound, parameter_version, scale_factor,
         format_versions_json, status, conclusion_json, conclusion_digest,
         cursor, session_id, updated_mono_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pod_id) DO UPDATE SET
          node_name=excluded.node_name,
          total_score=excluded.total_score,
          veto_code=excluded.veto_code,
          plugin_scores_json=excluded.plugin_scores_json,
          plugin_contributions_json=excluded.plugin_contributions_json,
          error_bound=excluded.error_bound,
          parameter_version=excluded.parameter_version,
          scale_factor=excluded.scale_factor,
          format_versions_json=excluded.format_versions_json,
          status=excluded.status,
          conclusion_json=excluded.conclusion_json,
          conclusion_digest=excluded.conclusion_digest,
          cursor=excluded.cursor,
          session_id=excluded.session_id,
          updated_mono_ms=excluded.updated_mono_ms
        """,
        (
            entry["pod_id"],
            entry["node_name"],
            entry["total_score"],
            entry["veto_code"],
            canonical_json(entry["plugin_scores"]),
            canonical_json(entry["plugin_contributions"]),
            entry["error_bound"],
            entry["parameter_version"],
            entry["scale_factor"],
            canonical_json(entry["format_versions"]),
            entry["status"],
            canonical_json(entry["conclusion"]) if entry["conclusion"] is not None else "null",
            entry["conclusion_digest"],
            int(entry.get("cursor", 0)),
            session_id,
            now,
        ),
    )


def sync_snapshot(
    db_path: str | Path,
    entries: Mapping[str, Mapping[str, Any]],
    *,
    format_versions: Iterable[int | str],
    parameter_version: int,
    scale_factor: int,
    inject_digest_mismatch: bool = False,
) -> SyncReport:
    connection = connect(db_path)
    ensure_schema(connection)
    version_set = tuple(sorted({str(value) for value in format_versions}))
    new_digest = snapshot_content_digest(
        entries,
        format_versions=version_set,
        parameter_version=parameter_version,
        scale_factor=scale_factor,
    )
    try:
        old_meta_digest = _meta_get(connection, "snapshot.content_digest")
        old_versions_raw = _meta_get(connection, "snapshot.format_version_set")
        old_versions = None
        if old_versions_raw is not None:
            try:
                old_versions = tuple(json.loads(old_versions_raw))
            except json.JSONDecodeError:
                old_versions = None
        old_rows = connection.execute("SELECT * FROM ledger_entries").fetchall()
        old_entries = {str(row["pod_id"]): _row_to_entry(row) for row in old_rows}
        initial = old_meta_digest is None and old_versions_raw is None
        old_snapshot_id = _meta_get(connection, "snapshot.active_session_id")
        archived_digests: dict[str, str] = {}
        if old_snapshot_id is not None:
            archived_digests = {
                str(row["pod_id"]): str(row["content_digest"])
                for row in connection.execute(
                    "SELECT pod_id, content_digest FROM ledger_snapshot_entries WHERE session_id=?",
                    (old_snapshot_id,),
                )
            }
        old_parameter_version = int(
            _meta_get(connection, "snapshot.parameter_version") or parameter_version
        )
        old_scale_factor = int(
            _meta_get(connection, "snapshot.scale_factor") or scale_factor
        )
        actual_old_digest = snapshot_content_digest(
            old_entries,
            format_versions=old_versions if old_versions is not None else version_set,
            parameter_version=old_parameter_version,
            scale_factor=old_scale_factor,
        )
        old_digests = {
            pod_id: entry_content_digest(entry)
            for pod_id, entry in old_entries.items()
        }
        comparison_old_digests = archived_digests or old_digests
        new_digests = {
            pod_id: entry_content_digest(entries[pod_id]) for pod_id in entries
        }
        mismatch = inject_digest_mismatch or (
            not initial
            and (
                old_meta_digest != new_digest
                or old_versions != version_set
                or actual_old_digest != old_meta_digest
                or old_digests != new_digests
            )
        )
        if not mismatch and not initial:
            session_row = connection.execute(
                "SELECT session_id FROM ledger_meta WHERE key='snapshot.active_session_id'"
            ).fetchone()
            session_id = (
                str(session_row["value"])
                if session_row is not None
                else f"snapshot-{new_digest[:16]}"
            )
            return SyncReport(
                snapshot_id=session_id,
                mismatch=False,
                reason=None,
                added=(),
                deleted=(),
                replaced=(),
                content_digest=new_digest,
                format_version_set=version_set,
            )

        added = (
            ()
            if initial and not inject_digest_mismatch
            else tuple(sorted(set(new_digests) - set(comparison_old_digests)))
        )
        deleted = tuple(sorted(set(comparison_old_digests) - set(new_digests)))
        replaced = tuple(
            sorted(
                pod_id
                for pod_id in set(comparison_old_digests) & set(new_digests)
                if comparison_old_digests[pod_id] != new_digests[pod_id]
            )
        )
        reasons: list[str] = []
        if inject_digest_mismatch:
            reasons.append("injected_digest_mismatch")
        if old_meta_digest != new_digest and not inject_digest_mismatch:
            reasons.append("content_digest_mismatch")
        if actual_old_digest != old_meta_digest and old_meta_digest is not None:
            reasons.append("ledger_entries_diverge_from_snapshot_digest")
        if old_versions != version_set:
            reasons.append("format_version_set_mismatch")
        reason = "initial_snapshot" if initial and not inject_digest_mismatch else ("+".join(reasons) or "snapshot_replace")
        session_id = f"snapshot-{uuid.uuid4().hex}"
        now = monotonic_ms()
        cursor = len(entries)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM ledger_entries")
            for pod_id in sorted(entries):
                _insert_entry(
                    connection,
                    entries[pod_id],
                    session_id=session_id,
                    now=now,
                )
            events: list[tuple[str, str, str | None, str]] = []
            for pod_id in added:
                events.append(("added", pod_id, None, new_digests[pod_id]))
            for pod_id in deleted:
                events.append(("deleted", pod_id, comparison_old_digests[pod_id], None))
            for pod_id in replaced:
                events.append(
                    ("replaced", pod_id, comparison_old_digests[pod_id], new_digests[pod_id])
                )
            events.sort(key=lambda item: (item[1], item[0]))
            for sequence, (action, pod_id, old_value, new_value) in enumerate(
                events, start=1
            ):
                connection.execute(
                    """
                    INSERT INTO ledger_replacement_events
                    (session_id, sequence, action, pod_id, old_digest, new_digest)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, sequence, action, pod_id, old_value, new_value),
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO ledger_snapshots
                (session_id, content_digest, format_version_set_json,
                 parameter_version, scale_factor, cursor, created_mono_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    new_digest,
                    canonical_json(list(version_set)),
                    parameter_version,
                    scale_factor,
                    cursor,
                    now,
                ),
            )
            for pod_id in sorted(new_digests):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO ledger_snapshot_entries
                    (session_id, pod_id, content_digest, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        pod_id,
                        new_digests[pod_id],
                        entries[pod_id]["status"],
                    ),
                )
            _meta_set(connection, "snapshot.content_digest", new_digest)
            _meta_set(
                connection, "snapshot.format_version_set", canonical_json(list(version_set))
            )
            _meta_set(connection, "snapshot.parameter_version", str(parameter_version))
            _meta_set(connection, "snapshot.scale_factor", str(scale_factor))
            _meta_set(connection, "snapshot.cursor", str(cursor))
            _meta_set(connection, "snapshot.active_session_id", session_id)
            _meta_set(connection, "snapshot.pending_session_id", session_id)
            _meta_set(connection, "pending.cursor", str(cursor))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        report = SyncReport(
            snapshot_id=session_id,
            mismatch=not initial or inject_digest_mismatch,
            reason=reason,
            added=added,
            deleted=deleted,
            replaced=replaced,
            content_digest=new_digest,
            format_version_set=version_set,
            injected=inject_digest_mismatch,
        )
        if inject_digest_mismatch:
            raise SnapshotError(
                "ledger content digest mismatch was injected and the snapshot was replaced",
                details=report.to_dict(),
            )
        return report
    finally:
        connection.close()


def inject_digest_mismatch(db_path: str | Path) -> None:
    connection = connect(db_path)
    ensure_schema(connection)
    try:
        _meta_set(connection, "snapshot.content_digest", "injected-corruption")
    finally:
        connection.close()


def begin_session(
    db_path: str | Path,
    pod_ids: Iterable[str],
    *,
    format_versions: Iterable[int | str],
    parameter_version: int,
    scale_factor: int,
) -> str:
    session_id = f"session-{uuid.uuid4().hex}"
    planned = sorted(set(pod_ids))
    connection = connect(db_path)
    ensure_schema(connection)
    now = monotonic_ms()
    version_set = sorted({str(value) for value in format_versions})
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                INSERT INTO ledger_snapshots
                (session_id, content_digest, format_version_set_json,
                 parameter_version, scale_factor, cursor, created_mono_ms)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    session_id,
                    "pending-session",
                    canonical_json(version_set),
                    parameter_version,
                    scale_factor,
                    now,
                ),
            )
            for index, pod_id in enumerate(planned, start=1):
                entry = {
                    "pod_id": pod_id,
                    "node_name": "-",
                    "total_score": 0,
                    "veto_code": 0,
                    "plugin_scores": {name: 0 for name in PLUGIN_NAMES},
                    "plugin_contributions": {name: 0 for name in PLUGIN_NAMES},
                    "error_bound": 0,
                    "parameter_version": parameter_version,
                    "scale_factor": scale_factor,
                    "format_versions": version_set,
                    "status": "unknown",
                    "conclusion": None,
                    "conclusion_digest": digest_value({"status": "unknown", "pod_id": pod_id}),
                    "cursor": 0,
                }
                _insert_entry(connection, entry, session_id=session_id, now=now)
            _meta_set(connection, "snapshot.pending_session_id", session_id)
            _meta_set(connection, "pending.cursor", "0")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return session_id
    finally:
        connection.close()


def commit_pod_result(
    db_path: str | Path,
    session_id: str,
    result: Mapping[str, Any],
    *,
    format_versions: Iterable[int | str],
    cursor: int,
) -> dict[str, Any]:
    entry = entry_from_result(
        result,
        format_versions=format_versions,
        status="committed",
        cursor=cursor,
        session_id=session_id,
    )
    connection = connect(db_path)
    ensure_schema(connection)
    now = monotonic_ms()
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _insert_entry(connection, entry, session_id=session_id, now=now)
            connection.execute(
                "UPDATE ledger_snapshots SET cursor=? WHERE session_id=?",
                (cursor, session_id),
            )
            _meta_set(connection, "pending.cursor", str(cursor))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return entry
    finally:
        connection.close()


def recover_ledger(db_path: str | Path) -> dict[str, Any]:
    connection = connect_readonly(db_path)
    try:
        session_row = connection.execute(
            "SELECT * FROM ledger_snapshots ORDER BY created_mono_ms DESC LIMIT 1"
        ).fetchone()
        if session_row is None:
            return {
                "session_id": None,
                "committed": [],
                "unknown": [],
                "cursor": 0,
                "recovery_basis": "committed_only",
                "exit_report_available": False,
                "requires_rerun": False,
                "message": "no ledger session exists",
            }
        session_id = str(session_row["session_id"])
        rows = connection.execute(
            "SELECT pod_id, status FROM ledger_entries WHERE session_id=? ORDER BY pod_id",
            (session_id,),
        ).fetchall()
        committed = [str(row["pod_id"]) for row in rows if row["status"] == "committed"]
        unknown = [str(row["pod_id"]) for row in rows if row["status"] != "committed"]
        return {
            "session_id": session_id,
            "committed": committed,
            "unknown": unknown,
            "cursor": int(session_row["cursor"]),
            "recovery_basis": "committed_only",
            "exit_report_available": False,
            "requires_rerun": bool(unknown),
            "message": (
                "uncommitted pods are unknown and must be rerun; no forced-exit report is claimed"
                if unknown
                else "all planned pods reached a committed state"
            ),
        }
    finally:
        connection.close()


def load_entries_readonly(db_path: str | Path) -> dict[str, dict[str, Any]]:
    connection = connect_readonly(db_path)
    try:
        rows = connection.execute("SELECT * FROM ledger_entries ORDER BY pod_id").fetchall()
        return {str(row["pod_id"]): _row_to_entry(row) for row in rows}
    finally:
        connection.close()


def ledger_status(db_path: str | Path) -> dict[str, Any]:
    connection = connect_readonly(db_path)
    try:
        meta = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM ledger_meta ORDER BY key")
        }
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM ledger_entries GROUP BY status ORDER BY status"
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        events = [
            dict(row)
            for row in connection.execute(
                "SELECT session_id,sequence,action,pod_id,old_digest,new_digest FROM ledger_replacement_events ORDER BY session_id,sequence"
            )
        ]
        return {
            "meta": meta,
            "entry_counts": counts,
            "replacement_events": events,
            "recovery": recover_ledger(db_path),
        }
    finally:
        connection.close()
