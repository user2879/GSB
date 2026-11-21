"""SQLite-backed trust versions, revision ledgers, stale isolation and replay."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .canonical import canonical_json, digest_obj, digest_without_field, read_json, write_canonical_json
from .certfamily import Family
from .errors import E_REPLAY, E_ROLLBACK, E_SNAPSHOT, WorkbenchError
from .trust import TrustDocument, document_from_revision_contents
from .verify import (
    VALIDATOR_VERSION,
    compare_results,
    compute_flips,
    snapshot_chain_records,
    validate_snapshot,
)


SCHEMA_VERSION = 1
SESSION_FORMAT_VERSION = 1


def _now_ms() -> int:
    return int(time.time() * 1000)


class Ledger:
    def __init__(self, path: str | Path, *, auto_recover: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self._create_schema()
        self.startup_recovery: dict[str, Any] = {
            "incomplete_runs": [],
            "session_snapshots_rebuilt": [],
        }
        if auto_recover:
            self.startup_recovery = self.recover_incomplete_runs()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self._lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trust_versions (
                    version INTEGER PRIMARY KEY,
                    revision_chain_digest TEXT NOT NULL,
                    revisions_json TEXT NOT NULL,
                    active_revisions_json TEXT NOT NULL,
                    current INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    operation TEXT NOT NULL,
                    parent_version INTEGER,
                    rollback_target INTEGER,
                    source_path TEXT,
                    source_digest TEXT,
                    created_ms INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS trust_one_current
                    ON trust_versions(current) WHERE current = 1;
                CREATE TABLE IF NOT EXISTS runs (
                    run_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    trust_version INTEGER NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    chain_digest TEXT NOT NULL,
                    revision_numbers_json TEXT NOT NULL,
                    revision_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    format_version INTEGER NOT NULL,
                    validator_version INTEGER NOT NULL,
                    recovery_required INTEGER NOT NULL DEFAULT 0,
                    created_ms INTEGER NOT NULL,
                    completed_ms INTEGER
                );
                CREATE TABLE IF NOT EXISTS revision_results (
                    run_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    conclusion TEXT NOT NULL,
                    failure_category TEXT,
                    first_failure_layer INTEGER,
                    result_index TEXT,
                    result_json TEXT,
                    committed INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, revision)
                );
                CREATE TABLE IF NOT EXISTS flips (
                    run_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    from_revision INTEGER NOT NULL,
                    to_revision INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    flip_json TEXT NOT NULL,
                    committed INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS input_snapshots (
                    snapshot_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_digest TEXT NOT NULL UNIQUE,
                    chain_digest TEXT NOT NULL,
                    chain_fingerprints_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    created_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_snapshots (
                    snapshot_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_digest TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    trust_version INTEGER NOT NULL,
                    revision_chain_digest TEXT NOT NULL,
                    input_snapshot_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    complete INTEGER NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    created_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replacement_audit (
                    audit_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    old_version INTEGER,
                    new_version INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ui_cache (
                    version INTEGER PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    updated_ms INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('workbench_version', ?)",
                (__version__,),
            )

    # ------------------------------------------------------------------
    # Trust versions and stale replacement
    # ------------------------------------------------------------------
    def current_trust_version(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM trust_versions WHERE current = 1 AND stale = 0 ORDER BY version DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None

    def current_trust_document(self) -> TrustDocument | None:
        row = self.current_trust_version()
        if row is None:
            return None
        return document_from_revision_contents(json.loads(row["revisions_json"]))

    def _require_current_version(self) -> sqlite3.Row:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM trust_versions WHERE current = 1 AND stale = 0 ORDER BY version DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise WorkbenchError(E_ROLLBACK, "no current trust version exists", field_path="trust_version")
        return row

    def _replacement_summary(self, old_version: int | None, operation: str) -> dict[str, Any]:
        if old_version is None:
            return {
                "operation": operation,
                "old_trust_version": None,
                "affected_chains": [],
                "flip_points": [],
                "old_failure_categories": [],
            }
        with self._lock:
            chain_rows = self.conn.execute(
                """
                SELECT DISTINCT s.snapshot_digest, s.chain_digest, s.chain_fingerprints_json
                FROM runs r JOIN input_snapshots s ON s.snapshot_digest = r.snapshot_digest
                WHERE r.trust_version = ? AND r.status = 'COMMITTED'
                ORDER BY s.chain_digest, s.snapshot_digest
                """,
                (old_version,),
            ).fetchall()
            flip_rows = self.conn.execute(
                """
                SELECT f.flip_json, f.from_revision, f.to_revision, f.kind, r.snapshot_digest
                FROM flips f JOIN runs r ON r.run_id = f.run_id
                WHERE r.trust_version = ? AND f.committed = 1
                ORDER BY r.snapshot_digest, f.from_revision, f.to_revision, f.kind
                """,
                (old_version,),
            ).fetchall()
            category_rows = self.conn.execute(
                """
                SELECT DISTINCT rr.failure_category
                FROM revision_results rr JOIN runs r ON r.run_id = rr.run_id
                WHERE r.trust_version = ? AND rr.committed = 1 AND rr.failure_category IS NOT NULL
                ORDER BY rr.failure_category
                """,
                (old_version,),
            ).fetchall()
        affected = [
            {
                "snapshot_digest": row["snapshot_digest"],
                "chain_digest": row["chain_digest"],
                "chain_fingerprints": json.loads(row["chain_fingerprints_json"]),
            }
            for row in chain_rows
        ]
        affected.sort(key=lambda item: (item["chain_digest"], item["snapshot_digest"]))
        flip_points = []
        for row in flip_rows:
            flip = json.loads(row["flip_json"])
            flip_points.append(
                {
                    "snapshot_digest": row["snapshot_digest"],
                    "from_revision": row["from_revision"],
                    "to_revision": row["to_revision"],
                    "kind": row["kind"],
                    "from_failure_category": flip.get("from_failure_category"),
                    "to_failure_category": flip.get("to_failure_category"),
                }
            )
        flip_points.sort(
            key=lambda item: (
                item["snapshot_digest"],
                item["from_revision"],
                item["to_revision"],
                item["kind"],
            )
        )
        return {
            "operation": operation,
            "old_trust_version": old_version,
            "affected_chains": affected,
            "flip_points": flip_points,
            "old_failure_categories": [row["failure_category"] for row in category_rows],
        }

    def create_trust_version(
        self,
        document: TrustDocument,
        *,
        operation: str = "REBUILD",
        source_path: str | None = None,
        rollback_target: int | None = None,
    ) -> dict[str, Any]:
        revisions = [revision.content() for revision in document.revisions]
        revisions_json = canonical_json(revisions)
        chain_digest = document.chain_digest
        created = _now_ms()
        with self._lock:
            old = self.conn.execute(
                "SELECT * FROM trust_versions WHERE current = 1 ORDER BY version DESC LIMIT 1"
            ).fetchone()
            old_version = int(old["version"]) if old is not None else None
            summary = self._replacement_summary(old_version, operation)
            next_row = self.conn.execute("SELECT COALESCE(MAX(version), 0) + 1 AS next FROM trust_versions").fetchone()
            version = int(next_row["next"])
            with self.conn:
                self.conn.execute("UPDATE trust_versions SET current = 0, stale = 1")
                self.conn.execute("UPDATE runs SET stale = 1")
                self.conn.execute("UPDATE revision_results SET stale = 1")
                self.conn.execute("UPDATE flips SET stale = 1")
                self.conn.execute("UPDATE input_snapshots SET stale = 1")
                self.conn.execute("UPDATE session_snapshots SET stale = 1")
                self.conn.execute("UPDATE ui_cache SET stale = 1")
                self.conn.execute(
                    """
                    INSERT INTO trust_versions(
                        version, revision_chain_digest, revisions_json, active_revisions_json,
                        current, stale, operation, parent_version, rollback_target,
                        source_path, source_digest, created_ms
                    ) VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version,
                        chain_digest,
                        revisions_json,
                        canonical_json(document.revision_numbers),
                        operation,
                        old_version,
                        rollback_target,
                        source_path or document.source_path,
                        digest_obj(document.to_json()),
                        created,
                    ),
                )
                summary["new_trust_version"] = version
                self.conn.execute(
                    """
                    INSERT INTO replacement_audit(old_version, new_version, operation, summary_json, created_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (old_version, version, operation, canonical_json(summary), created),
                )
        return {"version": version, "revision_chain_digest": chain_digest, "summary": summary}

    def rollback_to_revision(self, target_revision: int) -> dict[str, Any]:
        current = self._require_current_version()
        revisions = json.loads(current["revisions_json"])
        available = [int(item["revision"]) for item in revisions]
        if target_revision not in available:
            raise WorkbenchError(
                E_ROLLBACK,
                "rollback target is not in the current revision chain; old isolated judgments remain read-only",
                field_path="target_revision",
                details={"target_revision": target_revision, "available_revisions": available},
            )
        prefix = revisions[: available.index(target_revision) + 1]
        document = document_from_revision_contents(prefix, source_path=current["source_path"])
        return self.create_trust_version(
            document,
            operation="ROLLBACK",
            source_path=current["source_path"],
            rollback_target=target_revision,
        )

    def get_trust_version(self, version: int | None = None) -> dict[str, Any]:
        with self._lock:
            if version is None:
                row = self._require_current_version()
            else:
                row = self.conn.execute(
                    "SELECT * FROM trust_versions WHERE version = ?",
                    (version,),
                ).fetchone()
        if row is None:
            raise WorkbenchError(E_REPLAY, "trust version referenced by snapshot is missing", field_path="trust_version")
        data = dict(row)
        data["revisions"] = json.loads(data.pop("revisions_json"))
        data["active_revisions"] = json.loads(data.pop("active_revisions_json"))
        return data

    def list_trust_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT version, revision_chain_digest, current, stale, operation, parent_version, rollback_target, created_ms FROM trust_versions ORDER BY version"
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_replacement_audit(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM replacement_audit ORDER BY audit_seq DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        return result

    # ------------------------------------------------------------------
    # Runs and per-revision commits
    # ------------------------------------------------------------------
    def put_input_snapshot(self, snapshot: dict[str, Any]) -> None:
        snapshot_digest = snapshot.get("snapshot_digest")
        if not isinstance(snapshot_digest, str):
            raise WorkbenchError(E_SNAPSHOT, "input snapshot has no digest", field_path="snapshot.snapshot_digest")
        payload = canonical_json(snapshot)
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO input_snapshots(
                    snapshot_digest, chain_digest, chain_fingerprints_json,
                    payload_json, format_version, stale, created_ms
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(snapshot_digest) DO UPDATE SET stale = 0
                """,
                (
                    snapshot_digest,
                    snapshot.get("chain_digest"),
                    canonical_json(snapshot.get("chain_fingerprints", [])),
                    payload,
                    int(snapshot.get("format_version", 1)),
                    _now_ms(),
                ),
            )

    def start_run(
        self,
        trust_version: int,
        snapshot: dict[str, Any],
        revision_numbers: Iterable[int],
    ) -> str:
        self.get_trust_version(trust_version)
        self.put_input_snapshot(snapshot)
        numbers = sorted(int(number) for number in revision_numbers)
        if not numbers:
            raise ValueError("a run needs at least one revision")
        run_id = f"run-{trust_version:04d}-{uuid.uuid4().hex[:12]}"
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO runs(
                    run_id, trust_version, snapshot_digest, chain_digest,
                    revision_numbers_json, revision_count, status, stale,
                    format_version, validator_version, recovery_required, created_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', 0, ?, ?, 0, ?)
                """,
                (
                    run_id,
                    trust_version,
                    snapshot["snapshot_digest"],
                    snapshot["chain_digest"],
                    canonical_json(numbers),
                    len(numbers),
                    SESSION_FORMAT_VERSION,
                    VALIDATOR_VERSION,
                    _now_ms(),
                ),
            )
        return run_id

    def record_revision_result(self, run_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            run = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise WorkbenchError(E_SNAPSHOT, "run does not exist", field_path="run_id")
        if run["status"] != "RUNNING":
            raise WorkbenchError(E_SNAPSHOT, "run is not accepting revision results", field_path="run.status")
        if result.get("snapshot_digest") != run["snapshot_digest"]:
            raise WorkbenchError(
                E_SNAPSHOT,
                "revision result references a different input snapshot",
                revision=result.get("revision"),
                field_path="result.snapshot_digest",
            )
        if result.get("validator_version") != run["validator_version"]:
            raise WorkbenchError(
                E_SNAPSHOT,
                "revision result validator version does not match the run",
                revision=result.get("revision"),
                field_path="result.validator_version",
            )
        expected = json.loads(run["revision_numbers_json"])
        if result["revision"] not in expected:
            raise WorkbenchError(E_SNAPSHOT, "result revision is not part of the run", revision=result.get("revision"))
        first_failure = result.get("first_failure") or {}
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO revision_results(
                    run_id, revision, conclusion, failure_category, first_failure_layer,
                    result_index, result_json, committed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    run_id,
                    result["revision"],
                    result["conclusion"],
                    result.get("failure_category"),
                    first_failure.get("layer"),
                    result.get("index"),
                    canonical_json(result),
                ),
            )

    def finalize_run(self, run_id: str) -> dict[str, Any]:
        results = self.committed_results(run_id)
        run = self.get_run(run_id)
        expected = run["revision_numbers"]
        got = [item["revision"] for item in results]
        if got != expected:
            raise WorkbenchError(
                E_SNAPSHOT,
                "run cannot be finalized before every revision result is committed",
                field_path="run.revision_results",
                details={"expected": expected, "committed": got},
            )
        flips = compute_flips(results)
        with self._lock, self.conn:
            for ordinal, flip in enumerate(flips):
                self.conn.execute(
                    """
                    INSERT INTO flips(run_id, ordinal, from_revision, to_revision, kind, flip_json, committed)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        run_id,
                        ordinal,
                        flip["from_revision"],
                        flip["to_revision"],
                        flip["kind"],
                        canonical_json(flip),
                    ),
                )
            self.conn.execute(
                "UPDATE runs SET status = 'COMMITTED', completed_ms = ?, recovery_required = 0 WHERE run_id = ?",
                (_now_ms(), run_id),
            )
        return self.create_session_snapshot(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise WorkbenchError(E_SNAPSHOT, "run does not exist", field_path="run_id")
            result_rows = self.conn.execute(
                "SELECT * FROM revision_results WHERE run_id = ? ORDER BY revision",
                (run_id,),
            ).fetchall()
        data = dict(row)
        data["revision_numbers"] = json.loads(data.pop("revision_numbers_json"))
        data["results"] = []
        for result_row in result_rows:
            item = dict(result_row)
            item["result"] = json.loads(item["result_json"]) if item["result_json"] else None
            data["results"].append(item)
        return data

    def committed_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT result_json FROM revision_results WHERE run_id = ? AND committed = 1 ORDER BY revision",
                (run_id,),
            ).fetchall()
        return [json.loads(row["result_json"]) for row in rows if row["result_json"]]

    def find_results_by_index(self, index_value: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT run_id, revision, result_json, stale FROM revision_results WHERE result_index = ? AND committed = 1 AND result_json IS NOT NULL ORDER BY run_id, revision",
                (index_value,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "revision": row["revision"],
                "result": json.loads(row["result_json"]),
                "stale": bool(row["stale"]),
            }
            for row in rows
        ]

    def compare_indexes(self, left_index: str, right_index: str) -> dict[str, Any]:
        left_matches = self.find_results_by_index(left_index)
        right_matches = self.find_results_by_index(right_index)
        if not left_matches or not right_matches:
            raise WorkbenchError(
                E_SNAPSHOT,
                "one or both result indexes were not found in committed ledger results",
                field_path="index",
                details={
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_found": len(left_matches),
                    "right_found": len(right_matches),
                },
            )
        left = left_matches[0]
        right = right_matches[0]
        comparison = compare_results(left["result"], right["result"])
        return {
            "left": {
                "index": left_index,
                "run_id": left["run_id"],
                "revision": left["revision"],
                "stale": left["stale"],
            },
            "right": {
                "index": right_index,
                "run_id": right["run_id"],
                "revision": right["revision"],
                "stale": right["stale"],
            },
            "comparison": comparison,
            "matches": {
                "left": len(left_matches),
                "right": len(right_matches),
            },
        }

    # ------------------------------------------------------------------
    # Recovery after forced termination
    # ------------------------------------------------------------------
    def recover_incomplete_runs(self) -> dict[str, Any]:
        marked: list[dict[str, Any]] = []
        rebuilt: list[str] = []
        with self._lock:
            running = self.conn.execute(
                "SELECT * FROM runs WHERE status = 'RUNNING' ORDER BY run_seq"
            ).fetchall()
        for row in running:
            expected = [int(item) for item in json.loads(row["revision_numbers_json"])]
            with self._lock, self.conn:
                self.conn.execute(
                    "UPDATE runs SET status = 'INCOMPLETE', recovery_required = 1 WHERE run_id = ?",
                    (row["run_id"],),
                )
                committed_rows = self.conn.execute(
                    "SELECT revision FROM revision_results WHERE run_id = ? AND committed = 1",
                    (row["run_id"],),
                ).fetchall()
                committed = {int(item["revision"]) for item in committed_rows}
                for revision in expected:
                    if revision not in committed:
                        self.conn.execute(
                            """
                            INSERT OR IGNORE INTO revision_results(
                                run_id, revision, conclusion, failure_category, first_failure_layer,
                                result_index, result_json, committed
                            ) VALUES (?, ?, 'UNKNOWN', NULL, NULL, NULL, NULL, 0)
                            """,
                            (row["run_id"], revision),
                        )
            marked.append(
                {
                    "run_id": row["run_id"],
                    "trust_version": row["trust_version"],
                    "snapshot_digest": row["snapshot_digest"],
                    "unknown_revisions": [revision for revision in expected if revision not in committed],
                    "committed_revisions": sorted(committed),
                    "status": "INCOMPLETE",
                }
            )
        with self._lock:
            committed_runs = self.conn.execute(
                "SELECT run_id FROM runs WHERE status = 'COMMITTED' ORDER BY run_seq"
            ).fetchall()
        for row in committed_runs:
            run_id = row["run_id"]
            with self._lock:
                exists = self.conn.execute(
                    "SELECT 1 FROM session_snapshots WHERE run_id = ? LIMIT 1",
                    (run_id,),
                ).fetchone()
            if exists is None:
                self.create_session_snapshot(run_id)
                rebuilt.append(run_id)
        return {"incomplete_runs": marked, "session_snapshots_rebuilt": rebuilt}

    # ------------------------------------------------------------------
    # Session snapshots and replay
    # ------------------------------------------------------------------
    def _build_session_payload(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] != "COMMITTED":
            raise WorkbenchError(
                E_SNAPSHOT,
                "only committed runs can produce a session snapshot",
                field_path="run.status",
                details={"run_id": run_id, "status": run["status"]},
            )
        results = self.committed_results(run_id)
        if [item["revision"] for item in results] != run["revision_numbers"]:
            raise WorkbenchError(E_SNAPSHOT, "committed revision results are incomplete", field_path="run.results")
        version = self.get_trust_version(run["trust_version"])
        with self._lock:
            snapshot_row = self.conn.execute(
                "SELECT payload_json FROM input_snapshots WHERE snapshot_digest = ?",
                (run["snapshot_digest"],),
            ).fetchone()
            flip_rows = self.conn.execute(
                "SELECT flip_json FROM flips WHERE run_id = ? AND committed = 1 ORDER BY ordinal",
                (run_id,),
            ).fetchall()
        if snapshot_row is None:
            raise WorkbenchError(E_SNAPSHOT, "input snapshot for run is missing", field_path="input_snapshot")
        input_snapshot = json.loads(snapshot_row["payload_json"])
        flips = [json.loads(row["flip_json"]) for row in flip_rows]
        revisions = version["revisions"]
        revision_chain_digest = digest_obj(revisions)
        payload: dict[str, Any] = {
            "format_version": SESSION_FORMAT_VERSION,
            "kind": "tlstrust.session_snapshot",
            "workbench_version": __version__,
            "validator_version": VALIDATOR_VERSION,
            "run_id": run_id,
            "trust_version": run["trust_version"],
            "revision_chain_digest": revision_chain_digest,
            "input_snapshot_digest": input_snapshot["snapshot_digest"],
            "chain_digest": input_snapshot["chain_digest"],
            "revisions": revisions,
            "input_snapshot": input_snapshot,
            "results": results,
            "flips": flips,
            "complete": True,
            "recovery_disposition": "committed_revision_results_only",
        }
        payload["content_digest"] = digest_without_field(payload, "content_digest")
        return payload

    def create_session_snapshot(self, run_id: str, output_path: str | Path | None = None) -> dict[str, Any]:
        payload = self._build_session_payload(run_id)
        encoded = canonical_json(payload)
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO session_snapshots(
                    content_digest, run_id, format_version, trust_version, revision_chain_digest,
                    input_snapshot_digest, payload_json, complete, stale, created_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
                ON CONFLICT(content_digest) DO UPDATE SET stale = 0
                """,
                (
                    payload["content_digest"],
                    payload["run_id"],
                    payload["format_version"],
                    payload["trust_version"],
                    payload["revision_chain_digest"],
                    payload["input_snapshot_digest"],
                    encoded,
                    _now_ms(),
                ),
            )
        path: Path | None = None
        if output_path is not None:
            target = Path(output_path)
            if target.suffix.lower() == ".json":
                path = target
            else:
                path = target / f"session-{payload['content_digest'][7:23]}.json"
            write_canonical_json(path, payload)
        return {"payload": payload, "path": str(path) if path is not None else None}

    def validate_session_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WorkbenchError(E_SNAPSHOT, "session snapshot must be an object", field_path="session")
        if payload.get("kind") != "tlstrust.session_snapshot":
            raise WorkbenchError(E_SNAPSHOT, "unsupported session snapshot kind", field_path="session.kind")
        version = payload.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkbenchError(E_SNAPSHOT, "snapshot format_version must be an integer", field_path="session.format_version")
        if version != SESSION_FORMAT_VERSION:
            raise WorkbenchError(
                E_SNAPSHOT,
                "unsupported session snapshot format version",
                field_path="session.format_version",
                details={"supported": SESSION_FORMAT_VERSION, "actual": version},
            )
        if payload.get("complete") is not True:
            raise WorkbenchError(E_SNAPSHOT, "session snapshot is not complete", field_path="session.complete")
        expected_digest = digest_without_field(payload, "content_digest")
        if payload.get("content_digest") != expected_digest:
            raise WorkbenchError(
                E_SNAPSHOT,
                "session snapshot content digest mismatch",
                field_path="session.content_digest",
                details={"expected": expected_digest, "actual": payload.get("content_digest")},
            )
        input_snapshot = payload.get("input_snapshot")
        if not isinstance(input_snapshot, dict):
            raise WorkbenchError(E_SNAPSHOT, "session snapshot has no input snapshot", field_path="session.input_snapshot")
        snapshot_chain_records(input_snapshot)
        if payload.get("input_snapshot_digest") != input_snapshot.get("snapshot_digest"):
            raise WorkbenchError(
                E_SNAPSHOT,
                "input snapshot digest does not match embedded snapshot",
                field_path="session.input_snapshot_digest",
            )
        if payload.get("chain_digest") != input_snapshot.get("chain_digest"):
            raise WorkbenchError(E_SNAPSHOT, "session chain digest mismatch", field_path="session.chain_digest")
        revisions = payload.get("revisions")
        if not isinstance(revisions, list) or not revisions:
            raise WorkbenchError(E_SNAPSHOT, "session snapshot has no revisions", field_path="session.revisions")
        if payload.get("revision_chain_digest") != digest_obj(revisions):
            raise WorkbenchError(
                E_SNAPSHOT,
                "session revision-chain digest mismatch",
                field_path="session.revision_chain_digest",
            )
        results = payload.get("results")
        if not isinstance(results, list) or sorted(item.get("revision") for item in results) != [
            item.get("revision") for item in revisions
        ]:
            raise WorkbenchError(E_SNAPSHOT, "session results do not cover the revision chain", field_path="session.results")
        if not isinstance(payload.get("flips"), list):
            raise WorkbenchError(E_SNAPSHOT, "session flips must be a list", field_path="session.flips")
        return payload

    def load_session_snapshot(self, source: str | Path) -> dict[str, Any]:
        target = Path(source)
        if target.exists():
            try:
                payload = read_json(target)
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkbenchError(E_SNAPSHOT, f"cannot load session snapshot: {exc}", field_path="session") from exc
        else:
            with self._lock:
                row = self.conn.execute(
                    "SELECT payload_json FROM session_snapshots WHERE content_digest = ?",
                    (str(source),),
                ).fetchone()
            if row is None:
                raise WorkbenchError(E_SNAPSHOT, "session snapshot was not found", field_path="session")
            payload = json.loads(row["payload_json"])
        return self.validate_session_payload(payload)

    def replay_session(self, payload: dict[str, Any], family: Family | None = None) -> dict[str, Any]:
        session = self.validate_session_payload(payload)
        results = sorted(session["results"], key=lambda item: item["revision"])
        first_revision = results[0]["revision"] if results else None
        if session.get("validator_version") != VALIDATOR_VERSION:
            raise WorkbenchError(
                E_REPLAY,
                "validation semantics changed; session cannot be replayed byte-for-byte",
                revision=first_revision,
                field_path="session.validator_version",
                details={"recorded": session.get("validator_version"), "current": VALIDATOR_VERSION},
            )
        try:
            version = self.get_trust_version(int(session["trust_version"]))
        except (WorkbenchError, ValueError, TypeError) as exc:
            raise WorkbenchError(
                E_REPLAY,
                "trust version recorded by snapshot is unavailable",
                revision=first_revision,
                field_path="session.trust_version",
            ) from exc
        if version["revision_chain_digest"] != session["revision_chain_digest"]:
            raise WorkbenchError(
                E_REPLAY,
                "trust revision chain changed since snapshot creation",
                revision=first_revision,
                field_path="session.revision_chain_digest",
                details={"recorded": session["revision_chain_digest"], "current": version["revision_chain_digest"]},
            )
        try:
            document = document_from_revision_contents(session["revisions"])
        except Exception as exc:
            raise WorkbenchError(
                E_REPLAY,
                "session revisions cannot be reconstructed",
                revision=first_revision,
                field_path="session.revisions",
            ) from exc
        input_snapshot = session["input_snapshot"]
        chain_records = snapshot_chain_records(input_snapshot)
        if family is not None:
            family_fingerprints = set(family.certificate_fingerprints)
            for layer, record in enumerate(chain_records):
                if record.fingerprint not in family_fingerprints:
                    raise WorkbenchError(
                        E_REPLAY,
                        "certificate family changed; snapshot chain member is no longer present",
                        revision=first_revision,
                        layer=layer,
                        field_path=f"input_snapshot.chain[{layer}].fingerprint",
                        details={"fingerprint": record.fingerprint},
                    )
        replayed: list[dict[str, Any]] = []
        recomputed_results: list[dict[str, Any]] = []
        for recorded in results:
            revision = document.by_number(int(recorded["revision"]))
            recomputed = validate_snapshot(input_snapshot, revision)
            recomputed_results.append(recomputed)
            comparison = compare_results(recorded, recomputed)
            if not comparison.get("equal"):
                raise WorkbenchError(
                    E_REPLAY,
                    "replayed revision result differs from the snapshot",
                    revision=revision.revision,
                    layer=comparison.get("layer"),
                    field_path=comparison.get("field"),
                    details={"expected": comparison.get("expected"), "actual": comparison.get("actual")},
                )
            replayed.append(
                {
                    "revision": revision.revision,
                    "conclusion": recomputed["conclusion"],
                    "failure_category": recomputed["failure_category"],
                    "index": recomputed["index"],
                    "result_digest": recomputed["result_digest"],
                }
            )
        recomputed_flips = compute_flips(recomputed_results)
        recorded_flips = session.get("flips", [])
        if canonical_json(recomputed_flips) != canonical_json(recorded_flips):
            maximum = max(len(recomputed_flips), len(recorded_flips))
            difference_index = next(
                (
                    index
                    for index in range(maximum)
                    if (recomputed_flips[index] if index < len(recomputed_flips) else None)
                    != (recorded_flips[index] if index < len(recorded_flips) else None)
                ),
                0,
            )
            expected_flip = recorded_flips[difference_index] if difference_index < len(recorded_flips) else None
            actual_flip = recomputed_flips[difference_index] if difference_index < len(recomputed_flips) else None
            flip_revision = None
            if isinstance(expected_flip, dict):
                flip_revision = expected_flip.get("from_revision")
            if flip_revision is None and isinstance(actual_flip, dict):
                flip_revision = actual_flip.get("from_revision")
            if flip_revision is None and results:
                flip_revision = results[0]["revision"]
            raise WorkbenchError(
                E_REPLAY,
                "replayed flip transcript differs from the snapshot",
                revision=flip_revision,
                layer=None,
                field_path=f"session.flips[{difference_index}]",
                details={"expected": expected_flip, "actual": actual_flip},
            )
        with self._lock:
            run_row = self.conn.execute(
                "SELECT status FROM runs WHERE run_id = ?",
                (session.get("run_id"),),
            ).fetchone()
        if run_row is not None and run_row["status"] == "COMMITTED":
            ledger_results = self.committed_results(session["run_id"])
            by_revision = {item["revision"]: item for item in ledger_results}
            for recorded in results:
                ledger_result = by_revision.get(int(recorded["revision"]))
                if ledger_result is None:
                    raise WorkbenchError(
                        E_REPLAY,
                        "committed ledger result is missing during replay",
                        revision=int(recorded["revision"]),
                        field_path="ledger.results",
                    )
                comparison = compare_results(ledger_result, recorded)
                if not comparison.get("equal"):
                    raise WorkbenchError(
                        E_REPLAY,
                        "ledger and session snapshot disagree",
                        revision=int(recorded["revision"]),
                        layer=comparison.get("layer"),
                        field_path=comparison.get("field"),
                        details={"expected": comparison.get("expected"), "actual": comparison.get("actual")},
                    )
        return {
            "ok": True,
            "run_id": session.get("run_id"),
            "trust_version": session["trust_version"],
            "revision_chain_digest": session["revision_chain_digest"],
            "input_snapshot_digest": session["input_snapshot_digest"],
            "replayed_revisions": sorted(replayed, key=lambda item: item["revision"]),
            "flips_digest": digest_obj(recomputed_flips),
        }

    def list_session_snapshots(self, *, current_only: bool = True) -> list[dict[str, Any]]:
        where = "WHERE stale = 0" if current_only else ""
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT snapshot_seq, content_digest, format_version, trust_version,
                       revision_chain_digest, input_snapshot_digest, complete, stale, created_ms
                FROM session_snapshots {where} ORDER BY snapshot_seq
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_input_snapshot(self, snapshot_digest: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT payload_json FROM input_snapshots WHERE snapshot_digest = ?",
                (snapshot_digest,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row is not None else None

    # ------------------------------------------------------------------
    # Current view and UI cache
    # ------------------------------------------------------------------
    def build_current_view(self, *, write_cache: bool = True) -> dict[str, Any]:
        current = self._require_current_version()
        version = int(current["version"])
        with self._lock:
            run_rows = self.conn.execute(
                "SELECT * FROM runs WHERE trust_version = ? AND stale = 0 ORDER BY run_seq",
                (version,),
            ).fetchall()
            snapshot_rows = self.conn.execute(
                "SELECT snapshot_seq, content_digest, format_version, trust_version, revision_chain_digest, input_snapshot_digest, complete, stale, created_ms FROM session_snapshots WHERE trust_version = ? AND stale = 0 ORDER BY snapshot_seq",
                (version,),
            ).fetchall()
        runs: list[dict[str, Any]] = []
        for run_row in run_rows:
            run = dict(run_row)
            run["revision_numbers"] = json.loads(run.pop("revision_numbers_json"))
            input_snapshot = self.get_input_snapshot(run["snapshot_digest"])
            if input_snapshot is not None:
                run["handshake"] = {
                    "leaf_id": input_snapshot.get("leaf_id"),
                    "snapshot_digest": input_snapshot.get("snapshot_digest"),
                    "chain_digest": input_snapshot.get("chain_digest"),
                    "chain_fingerprints": input_snapshot.get("chain_fingerprints", []),
                    "chain_serial_numbers": input_snapshot.get("chain_serial_numbers", []),
                    "negotiated": input_snapshot.get("negotiated", {}),
                    "captured_monotonic_ms": input_snapshot.get("captured_monotonic_ms"),
                    "endpoints": input_snapshot.get("endpoints", {}),
                }
            else:
                run["handshake"] = None
            with self._lock:
                result_rows = self.conn.execute(
                    "SELECT revision, conclusion, failure_category, first_failure_layer, result_index, result_json, committed, stale FROM revision_results WHERE run_id = ? AND stale = 0 ORDER BY revision",
                    (run["run_id"],),
                ).fetchall()
                flip_rows = self.conn.execute(
                    "SELECT ordinal, from_revision, to_revision, kind, flip_json FROM flips WHERE run_id = ? AND committed = 1 AND stale = 0 ORDER BY ordinal",
                    (run["run_id"],),
                ).fetchall()
            run["revisions"] = []
            for row in result_rows:
                item = dict(row)
                item["result"] = json.loads(item["result_json"]) if item["result_json"] else None
                run["revisions"].append(item)
            run["flips"] = [json.loads(row["flip_json"]) for row in flip_rows]
            runs.append(run)
        payload: dict[str, Any] = {
            "format_version": 1,
            "kind": "tlstrust.current_view",
            "trust_version": version,
            "revision_chain_digest": current["revision_chain_digest"],
            "revisions": json.loads(current["revisions_json"]),
            "runs": runs,
            "session_snapshots": [dict(row) for row in snapshot_rows],
            "replacement_audit": self.latest_replacement_audit(),
        }
        payload["view_digest"] = digest_obj(payload)
        if write_cache:
            with self._lock, self.conn:
                self.conn.execute("DELETE FROM ui_cache WHERE version = ?", (version,))
                self.conn.execute(
                    "INSERT INTO ui_cache(version, payload_json, payload_digest, stale, updated_ms) VALUES (?, ?, ?, 0, ?)",
                    (version, canonical_json(payload), payload["view_digest"], _now_ms()),
                )
        return payload
