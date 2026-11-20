"""SQLite session persistence, structured logs and immutable snapshot files."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_IO, E_REPLAY, E_SNAPSHOT
from .scheduler import WorkbenchSession
from .util import atomic_write_text, canonical_json, pretty_json, safe_name, sha256_digest, utc_now


SNAPSHOT_FORMAT = "k8s-sched-diag-snapshot"
SNAPSHOT_FORMAT_VERSION = 1


class StructuredLogger:
    """Append-only JSON-lines diagnostic log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "timestamp": utc_now(),
            "level": level,
            "event": event,
            **fields,
        }
        try:
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(record) + "\n")
        except OSError:
            # Diagnostics must not mask the business failure they describe.
            pass
        return record

    def error(self, error: BusinessError, event: str = "business_error", **fields: Any) -> dict[str, Any]:
        return self.write("ERROR", event, **error.to_dict(), **fields)


def verify_snapshot_payload(payload: Any, *, source: str) -> dict[str, Any]:
    """Validate envelope identity, versions and both digests."""
    if not isinstance(payload, dict):
        raise BusinessError(E_SNAPSHOT, "snapshot payload is not a JSON object", source)
    if payload.get("format") != SNAPSHOT_FORMAT:
        raise BusinessError(E_SNAPSHOT, "unrecognized snapshot format", source)
    if payload.get("formatVersion") != SNAPSHOT_FORMAT_VERSION:
        raise BusinessError(E_SNAPSHOT, "unsupported snapshot formatVersion", source)
    versions = payload.get("versions")
    if not isinstance(versions, dict):
        raise BusinessError(E_REPLAY, "snapshot has no version block", source)
    missing_versions = [name for name in ("node", "pod", "plugin", "graph") if name not in versions]
    if missing_versions:
        raise BusinessError(
            E_REPLAY,
            "snapshot version block is incomplete",
            source,
            {"missingVersions": missing_versions},
        )
    for name in ("node", "pod", "plugin", "graph"):
        value = versions[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BusinessError(E_REPLAY, f"snapshot version '{name}' is invalid", source)
    if not isinstance(versions.get("factRevisions", {}), dict):
        raise BusinessError(E_REPLAY, "snapshot factRevisions are missing or invalid", source)
    if not isinstance(versions.get("conclusionRevisions", {}), dict):
        raise BusinessError(E_REPLAY, "snapshot conclusionRevisions are missing or invalid", source)

    content = payload.get("content")
    if not isinstance(content, dict):
        raise BusinessError(E_SNAPSHOT, "snapshot content is missing", source)
    expected_content_digest = payload.get("contentDigest")
    actual_content_digest = sha256_digest(content)
    if expected_content_digest != actual_content_digest:
        raise BusinessError(
            E_SNAPSHOT,
            "snapshot content digest mismatch",
            source + "#/contentDigest",
            {"expected": expected_content_digest, "actual": actual_content_digest},
        )
    envelope = {key: value for key, value in payload.items() if key != "envelopeDigest"}
    expected_envelope_digest = payload.get("envelopeDigest")
    actual_envelope_digest = sha256_digest(envelope)
    if expected_envelope_digest != actual_envelope_digest:
        raise BusinessError(
            E_SNAPSHOT,
            "snapshot envelope digest mismatch",
            source + "#/envelopeDigest",
            {"expected": expected_envelope_digest, "actual": actual_envelope_digest},
        )

    state = payload.get("state")
    if not isinstance(state, dict):
        raise BusinessError(E_SNAPSHOT, "snapshot state is missing", source)
    for key in ("results", "stats", "lastChange", "lastInvalidation", "graphSignature"):
        if key not in state:
            raise BusinessError(E_REPLAY, f"snapshot state field '{key}' is missing", source + f"#/state/{key}")
    if not isinstance(state["results"], list):
        raise BusinessError(E_REPLAY, "snapshot results must be an array", source + "#/state/results")
    if not isinstance(state["stats"], dict):
        raise BusinessError(E_REPLAY, "snapshot stats must be an object", source + "#/state/stats")
    return payload


class Persistence:
    def __init__(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir)
        try:
            self.workdir.mkdir(parents=True, exist_ok=True)
            probe = self.workdir / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise BusinessError(E_IO, f"work directory is not writable: {exc}", str(self.workdir)) from None
        self.db_path = self.workdir / "state.sqlite3"
        self.snapshot_dir = self.workdir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = StructuredLogger(self.workdir / "logs" / "workbench.jsonl")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    name TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    node_version INTEGER NOT NULL,
                    pod_version INTEGER NOT NULL,
                    plugin_version INTEGER NOT NULL,
                    graph_version INTEGER NOT NULL,
                    backup_path TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (name, revision)
                );
                CREATE TABLE IF NOT EXISTS current_session (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    updated_at TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def log(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        return self.logger.write(level, event, **fields)

    def log_error(self, error: BusinessError, event: str = "business_error", **fields: Any) -> dict[str, Any]:
        return self.logger.error(error, event, **fields)

    def _build_payload(self, session: WorkbenchSession, name: str, revision: int) -> dict[str, Any]:
        if len(session.results) != len(session.pods):
            raise BusinessError(
                E_SNAPSHOT,
                "all pod conclusions must be materialized before saving",
                details={"expected": len(session.pods), "actual": len(session.results)},
            )
        versions = session.versions()
        conclusion_revisions = {
            pod.pod_id: int(session.results[pod.pod_id].get("version", 0)) for pod in session.pods
        }
        payload: dict[str, Any] = {
            "format": SNAPSHOT_FORMAT,
            "formatVersion": SNAPSHOT_FORMAT_VERSION,
            "name": name,
            "revision": revision,
            "createdAt": utc_now(),
            "versions": {
                **versions,
                "factRevisions": dict(sorted(session.fact_revisions.items())),
                "conclusionRevisions": conclusion_revisions,
            },
            "content": session.content_payload(),
            "state": {
                "results": session.results_payload(),
                "stats": session.stats_payload(),
                "lastChange": session.last_change,
                "lastInvalidation": session.last_invalidation,
                "graphSignature": session.graph.signature(),
            },
        }
        payload["contentDigest"] = sha256_digest(payload["content"])
        payload["envelopeDigest"] = sha256_digest(payload)
        return payload

    def save_snapshot(self, session: WorkbenchSession, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            raise BusinessError(E_SNAPSHOT, "snapshot name must not be empty")
        name = name.strip()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision FROM snapshots WHERE name = ?",
                (name,),
            ).fetchone()
            revision = int(row["next_revision"])
        payload = self._build_payload(session, name, revision)
        backup_path = self.snapshot_dir / safe_name(name) / f"r{revision:04d}.json"
        text = pretty_json(payload) + "\n"
        atomic_write_text(backup_path, text, readonly=True)
        versions = payload["versions"]
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO snapshots
                    (name, revision, created_at, digest, node_version, pod_version,
                     plugin_version, graph_version, backup_path, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        revision,
                        payload["createdAt"],
                        payload["contentDigest"],
                        versions["node"],
                        versions["pod"],
                        versions["plugin"],
                        versions["graph"],
                        str(backup_path),
                        canonical_json(payload),
                    ),
                )
        except Exception:
            try:
                backup_path.chmod(0o600)
                backup_path.unlink()
            except OSError:
                pass
            raise
        self.log(
            "INFO",
            "snapshot_saved",
            name=name,
            revision=revision,
            digest=payload["contentDigest"],
            backupPath=str(backup_path),
        )
        return {
            "name": name,
            "revision": revision,
            "createdAt": payload["createdAt"],
            "digest": payload["contentDigest"],
            "versions": {key: versions[key] for key in ("node", "pod", "plugin", "graph")},
            "backupPath": str(backup_path),
            "readOnly": True,
        }

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT name, revision, created_at, digest, node_version, pod_version,
                       plugin_version, graph_version, backup_path
                FROM snapshots ORDER BY name ASC, revision ASC
                """
            ).fetchall()
        return [
            {
                "name": row["name"],
                "revision": row["revision"],
                "createdAt": row["created_at"],
                "digest": row["digest"],
                "versions": {
                    "node": row["node_version"],
                    "pod": row["pod_version"],
                    "plugin": row["plugin_version"],
                    "graph": row["graph_version"],
                },
                "backupPath": row["backup_path"],
                "readOnly": True,
            }
            for row in rows
        ]

    def _load_row_payload(self, row: sqlite3.Row, source: str) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise BusinessError(E_SNAPSHOT, f"stored snapshot JSON is invalid: {exc}", source) from None
        verified = verify_snapshot_payload(payload, source=source)
        if verified["contentDigest"] != row["digest"]:
            raise BusinessError(
                E_SNAPSHOT,
                "stored snapshot digest does not match the indexed digest",
                source + "#/contentDigest",
                {"expected": row["digest"], "actual": verified["contentDigest"]},
            )
        return verified

    def load_snapshot(self, name: str, revision: int | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            if revision is None:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE name = ? ORDER BY revision DESC LIMIT 1",
                    (name,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE name = ? AND revision = ?",
                    (name, revision),
                ).fetchone()
        if row is None:
            suffix = f" revision {revision}" if revision is not None else ""
            raise BusinessError(E_SNAPSHOT, f"snapshot '{name}'{suffix} was not found")
        return self._load_row_payload(row, row["backup_path"])

    def load_snapshot_file(self, path: str | Path) -> dict[str, Any]:
        source = str(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise BusinessError(E_SNAPSHOT, "snapshot file does not exist", source) from None
        except json.JSONDecodeError as exc:
            raise BusinessError(
                E_SNAPSHOT,
                f"snapshot file JSON is invalid: {exc.msg}",
                source,
                {"line": exc.lineno, "column": exc.colno},
            ) from None
        return verify_snapshot_payload(payload, source=source)

    def save_current(self, session: WorkbenchSession) -> dict[str, Any]:
        payload = self._build_payload(session, "<current-session>", 0)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO current_session (id, updated_at, digest, payload)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    digest = excluded.digest,
                    payload = excluded.payload
                """,
                (payload["createdAt"], payload["contentDigest"], canonical_json(payload)),
            )
        return {"digest": payload["contentDigest"], "versions": session.versions()}

    def load_current(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM current_session WHERE id = 1").fetchone()
        if row is None:
            return None
        return self._load_row_payload(row, str(self.db_path) + "#/current_session")
