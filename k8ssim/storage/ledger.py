"""SQLite scheduling ledger with format versioning and content digests.

The ledger is the single durable truth. Each pod result is inserted in its own
transaction only after the full decision is computed, so a forced termination
can never leave a half-written pod: an in-flight pod is represented by an
``UNKNOWN`` marker row that recovery explicitly asks to rerun.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..constants import LEDGER_FORMAT_VERSION, STATUS_COMMITTED, STATUS_UNKNOWN
from ..errors import SnapshotError
from ..models import Decision, Node, Pod
from ..util import digest_of
from ..scheduler.engine import SessionParams

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS specs (
    kind TEXT NOT NULL,
    content_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (kind)
);
CREATE TABLE IF NOT EXISTS parameter_versions (
    version INTEGER PRIMARY KEY,
    weights_json TEXT NOT NULL,
    scale_factor INTEGER NOT NULL,
    digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pod_results (
    seq INTEGER NOT NULL,
    pod_id TEXT NOT NULL,
    status TEXT NOT NULL,
    parameter_version INTEGER NOT NULL,
    decision_json TEXT,
    digest TEXT,
    PRIMARY KEY (seq)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    params_digest TEXT NOT NULL,
    parameter_version INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._ensure_meta()

    def close(self) -> None:
        self.conn.close()

    def _ensure_meta(self) -> None:
        existing = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.commit()
        elif int(existing["value"]) != LEDGER_FORMAT_VERSION:
            raise SnapshotError(
                f"ledger format version {existing['value']} is not supported "
                f"(expected {LEDGER_FORMAT_VERSION})",
                mismatches=[f"meta.format_version={existing['value']}"],
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Spec / parameter persistence
    # ------------------------------------------------------------------ #

    def replace_specs(self, nodes: Sequence[Node], pods: Sequence[Pod]) -> Dict[str, str]:
        nodes_json = json.dumps([n.to_dict() for n in nodes], sort_keys=True)
        pods_json = json.dumps([p.to_dict() for p in pods], sort_keys=True)
        node_digest = digest_of(json.loads(nodes_json))
        pod_digest = digest_of(json.loads(pods_json))
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO specs(kind, content_json, digest) VALUES (?, ?, ?)",
                ("nodes", nodes_json, node_digest),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO specs(kind, content_json, digest) VALUES (?, ?, ?)",
                ("pods", pods_json, pod_digest),
            )
        return {"nodes": node_digest, "pods": pod_digest}

    def load_specs(self) -> tuple[List[Node], List[Pod], Dict[str, str]]:
        rows = {
            row["kind"]: row
            for row in self.conn.execute("SELECT kind, content_json, digest FROM specs")
        }
        if "nodes" not in rows or "pods" not in rows:
            raise SnapshotError("ledger does not contain a complete specification")
        nodes = [Node.from_dict(item) for item in json.loads(rows["nodes"]["content_json"])]
        pods = [Pod.from_dict(item) for item in json.loads(rows["pods"]["content_json"])]
        return nodes, pods, {kind: rows[kind]["digest"] for kind in rows}

    def record_parameter_version(self, params: SessionParams) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO parameter_versions"
                "(version, weights_json, scale_factor, digest) VALUES (?, ?, ?, ?)",
                (
                    params.version,
                    json.dumps(dict(params.weights), sort_keys=True),
                    params.scale_factor,
                    params.content_digest(),
                ),
            )

    def latest_parameter_version(self) -> Optional[SessionParams]:
        row = self.conn.execute(
            "SELECT * FROM parameter_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return SessionParams(
            weights=json.loads(row["weights_json"]),
            scale_factor=row["scale_factor"],
            version=row["version"],
        )

    # ------------------------------------------------------------------ #
    # Per-pod commit protocol
    # ------------------------------------------------------------------ #

    def mark_unknown(self, seq: int, pod_id: str, parameter_version: int) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO pod_results"
                "(seq, pod_id, status, parameter_version, decision_json, digest)"
                " VALUES (?, ?, ?, ?, NULL, NULL)",
                (seq, pod_id, STATUS_UNKNOWN, parameter_version),
            )

    def commit_decision(self, seq: int, pod_id: str, decision: Decision) -> str:
        payload = decision.to_dict()
        decision_json = json.dumps(payload, sort_keys=True)
        decision_digest = digest_of(json.loads(decision_json))
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO pod_results"
                "(seq, pod_id, status, parameter_version, decision_json, digest)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    pod_id,
                    STATUS_COMMITTED,
                    decision.parameter_version,
                    decision_json,
                    decision_digest,
                ),
            )
        return decision_digest

    def load_pod_results(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM pod_results ORDER BY seq"
        ).fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            if entry["decision_json"]:
                entry["decision"] = json.loads(entry["decision_json"])
            results.append(entry)
        return results

    def reset_pod_results(self) -> None:
        with self.transaction():
            self.conn.execute("DELETE FROM pod_results")

    # ------------------------------------------------------------------ #
    # Snapshots and verification
    # ------------------------------------------------------------------ #

    def save_snapshot(self, snapshot_id: str, params: SessionParams,
                      spec_digest: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT OR REPLACE INTO snapshots"
                "(snapshot_id, format_version, spec_digest, params_digest, parameter_version)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    LEDGER_FORMAT_VERSION,
                    spec_digest,
                    params.content_digest(),
                    params.version,
                ),
            )

    def list_snapshots(self) -> List[Dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM snapshots ORDER BY created_at, snapshot_id"
            )
        ]

    def verify_snapshot(
        self,
        snapshot_id: str,
        current_spec_digest: str,
        current_params: SessionParams,
    ) -> None:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise SnapshotError(
                f"snapshot {snapshot_id!r} does not exist",
                mismatches=[f"snapshots.{snapshot_id}"],
            )
        mismatches: List[str] = []
        if row["format_version"] != LEDGER_FORMAT_VERSION:
            mismatches.append(
                f"format_version: snapshot={row['format_version']} "
                f"current={LEDGER_FORMAT_VERSION}"
            )
        if row["spec_digest"] != current_spec_digest:
            mismatches.append(
                f"spec_digest: snapshot={row['spec_digest'][:12]} "
                f"current={current_spec_digest[:12]}"
            )
        if row["params_digest"] != current_params.content_digest():
            mismatches.append(
                f"params_digest: snapshot={row['params_digest'][:12]} "
                f"current={current_params.content_digest()[:12]}"
            )
        if row["parameter_version"] != current_params.version:
            mismatches.append(
                f"parameter_version: snapshot={row['parameter_version']} "
                f"current={current_params.version}"
            )
        if mismatches:
            raise SnapshotError(
                "snapshot does not match the current state; full replacement required",
                mismatches=mismatches,
            )
