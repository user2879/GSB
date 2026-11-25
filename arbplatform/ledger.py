"""Arbitration ledger: versioned SQLite persistence and replay validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .db import connect, sqlite_version
from .determinism import canonical, digest_of
from .engine import ViewResult
from .errors import ReplayError
from .spec import Manifest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    scale_power INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS views (
    view_id TEXT NOT NULL,
    view_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    record_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (view_id, view_version)
);
CREATE TABLE IF NOT EXISTS pending (
    view_id TEXT PRIMARY KEY,
    view_version INTEGER NOT NULL,
    started_at_ms INTEGER NOT NULL,
    note TEXT NOT NULL
);
"""


@dataclass
class ReplayOutcome:
    matched: int
    unknown_views: list[str]
    replaced: bool
    reason: str | None


class Ledger:
    def __init__(self, db_path: str | Path, manifest: Manifest) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.spec_digest = digest_of(self._spec_payload())
        self.conn = connect(str(self.db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def runtime_sqlite_version() -> str:
        return sqlite_version()

    def _spec_payload(self) -> dict[str, Any]:
        return {
            "environments": [
                {"env_id": e.env_id, "host": e.host, "port": e.port,
                 "lease_capacity": e.lease_capacity, "seed": e.seed,
                 "weight": e.weight}
                for e in self.manifest.environments
            ],
            "instances": [
                {"instance_id": i.instance_id, "env_ref": i.env_ref,
                 "seq_low": i.seq_low, "seq_high": i.seq_high,
                 "time_base_ms": i.time_base_ms}
                for i in self.manifest.instances
            ],
            "views": [
                {"view_id": v.view_id, "env_refs": list(v.env_refs),
                 "policies": dict(sorted(v.policies.items())),
                 "tolerance": v.tolerance}
                for v in self.manifest.views
            ],
            "scale_power": self.manifest.scale_power,
            "global_concurrency": self.manifest.global_concurrency,
        }

    #-- lifecycle --------------------------------------------------------
    def open_or_replace(self) -> ReplayOutcome:
        row = self.conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        pending = [r["view_id"] for r in self.conn.execute(
            "SELECT view_id FROM pending ORDER BY view_id")]
        if row is None:
            self._write_meta()
            self.conn.commit()
            return ReplayOutcome(0, pending, False, "fresh ledger initialized")
        mismatch = (
            row["format_version"] != LEDGER_FORMAT_VERSION
            or row["spec_digest"] != self.spec_digest
        )
        if mismatch:
            # Whole-snapshot replacement, never incremental merging.
            self.conn.executescript(
                "DELETE FROM views; DELETE FROM pending; DELETE FROM meta;"
            )
            self._write_meta()
            self.conn.commit()
            reason = (
                "format version mismatch"
                if row["format_version"] != LEDGER_FORMAT_VERSION
                else "spec content digest mismatch"
            )
            return ReplayOutcome(0, [], True, reason)
        return ReplayOutcome(
            self.conn.execute("SELECT COUNT(*) AS c FROM views").fetchone()["c"],
            pending, False, "snapshot matches current declaration",
        )

    def _write_meta(self) -> None:
        payload_digest = digest_of({"spec": self._spec_payload()})
        self.conn.execute(
            "INSERT INTO meta (id, format_version, spec_digest, scale_power, "
            "content_digest) VALUES (1, ?, ?, ?, ?)",
            (LEDGER_FORMAT_VERSION, self.spec_digest,
             self.manifest.scale_power, payload_digest),
        )

    #-- transactional writes --------------------------------------------
    def begin_view(self, view_id: str, view_version: int,
                   started_at_ms: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO pending (view_id, view_version, "
            "started_at_ms, note) VALUES (?, ?, ?, ?)",
            (view_id, view_version, started_at_ms, "uncommitted arbitration"),
        )
        self.conn.commit()

    def commit_view(self, result: ViewResult, committed_at_ms: int) -> None:
        record = result.to_record()
        # Single SQLite transaction: pending removal + authoritative write.
        with self.conn:
            self.conn.execute(
                "INSERT INTO views (view_id, view_version, status, record_json, "
                "content_digest, committed_at_ms) VALUES (?, ?, 'committed', ?, ?, ?) "
                "ON CONFLICT(view_id, view_version) DO UPDATE SET "
                "status=excluded.status, record_json=excluded.record_json, "
                "content_digest=excluded.content_digest, "
                "committed_at_ms=excluded.committed_at_ms",
                (result.view_id, result.view_version,
                 canonical(record), result.content_digest, committed_at_ms),
            )
            self.conn.execute("DELETE FROM pending WHERE view_id = ?",
                              (result.view_id,))

    def mark_unknown_after_kill(self) -> list[str]:
        """Any view left in pending is unknown and must be rerun."""
        rows = self.conn.execute(
            "SELECT view_id, view_version FROM pending ORDER BY view_id"
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT OR IGNORE INTO views (view_id, view_version, status, "
                "record_json, content_digest, committed_at_ms) "
                "VALUES (?, ?, 'unknown', ?, '', 0)",
                (row["view_id"], row["view_version"],
                 canonical({"reason": "uncommitted at forced termination"})),
            )
        self.conn.commit()
        return [r["view_id"] for r in rows]

    #-- reads / replay ---------------------------------------------------
    def latest_committed(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        rows = self.conn.execute(
            "SELECT * FROM views v WHERE view_version = ("
            "SELECT MAX(view_version) FROM views WHERE view_id = v.view_id) "
            "AND status = 'committed' ORDER BY view_id"
        ).fetchall()
        for row in rows:
            out[row["view_id"]] = json.loads(row["record_json"])
        return out

    def replay(self, recomputed: dict[str, ViewResult]) -> ReplayOutcome:
        """Replay ledger against fresh results; locate first mismatch (E_REPLAY)."""
        stored = self.latest_committed()
        matched = 0
        for view_id in sorted(set(stored) | set(recomputed)):
            if view_id not in stored:
                raise ReplayError(
                    f"view {view_id} absent from ledger",
                    field=f"views.{view_id}",
                )
            if view_id not in recomputed:
                raise ReplayError(
                    f"view {view_id} absent from recomputation",
                    field=f"views.{view_id}",
                )
            fresh = recomputed[view_id].to_record()
            old = stored[view_id]
            envelope_keys = (set(fresh) | set(old)) - {"verdicts"}
            for key in sorted(envelope_keys):
                if canonical(old.get(key)) != canonical(fresh.get(key)):
                    raise ReplayError(
                        f"first mismatch in view {view_id} field {key}",
                        field=f"views.{view_id}.{key}",
                        detail={"ledger": old.get(key), "fresh": fresh.get(key)},
                    )
            for fresh_v, old_v in self._zip_verdicts(fresh, old, view_id):
                for field_name in sorted(set(fresh_v) | set(old_v)):
                    if canonical(old_v.get(field_name)) != canonical(
                        fresh_v.get(field_name)
                    ):
                        raise ReplayError(
                            f"first mismatch in view {view_id} key "
                            f"{fresh_v.get('key')} field {field_name}",
                            field=(
                                f"views.{view_id}.verdicts."
                                f"{fresh_v.get('key')}.{field_name}"
                            ),
                            detail={
                                "ledger": old_v.get(field_name),
                                "fresh": fresh_v.get(field_name),
                            },
                        )
                matched += 1
        return ReplayOutcome(matched, [], False, "replay matched ledger exactly")

    @staticmethod
    def _zip_verdicts(fresh: dict[str, Any], old: dict[str, Any],
                      view_id: str):
        fresh_by_key = {v["key"]: v for v in fresh["verdicts"]}
        old_by_key = {v["key"]: v for v in old["verdicts"]}
        for key in sorted(set(fresh_by_key) | set(old_by_key)):
            if key not in old_by_key or key not in fresh_by_key:
                raise ReplayError(
                    f"verdict key set mismatch for {key}",
                    field=f"views.{view_id}.verdicts.{key}",
                )
            yield fresh_by_key[key], old_by_key[key]

    def close(self) -> None:
        self.conn.close()
