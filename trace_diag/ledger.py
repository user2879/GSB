from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, sha256_json
from .errors import SnapshotError
from .persistence_model import batch_content_digest, split_service_analysis
from .spec import PoliciesDoc, ServicesDoc

LEDGER_FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def manifest_digest(services_doc: ServicesDoc, policies_doc: PoliciesDoc) -> str:
    return sha256_json(
        {
            "service_version": services_doc.version,
            "strategy_version": policies_doc.version,
            "services_digest": services_doc.content_digest,
            "policies_digest": policies_doc.content_digest,
        }
    )


class Ledger:
    def __init__(self, path: str | Path, *, mark_pending_unknown: bool = True) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.initialize()
        if mark_pending_unknown:
            self.mark_incomplete_batches_unknown()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def initialize(self) -> None:
        self.connection.executescript(
            """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    service_version INTEGER NOT NULL,
                    strategy_version INTEGER NOT NULL,
                    services_digest TEXT NOT NULL,
                    policies_digest TEXT NOT NULL,
                    manifest_digest TEXT NOT NULL,
                    services_json TEXT NOT NULL,
                    policies_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    expected_batches INTEGER NOT NULL,
                    committed_batches INTEGER NOT NULL DEFAULT 0,
                    progress_cursor INTEGER NOT NULL DEFAULT 0,
                    snapshot_digest TEXT,
                    created_utc TEXT NOT NULL,
                    completed_utc TEXT
                );
                CREATE TABLE IF NOT EXISTS batches (
                    run_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    service_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_cursor INTEGER NOT NULL,
                    content_digest TEXT,
                    updated_utc TEXT NOT NULL,
                    PRIMARY KEY (run_id, ordinal),
                    UNIQUE (run_id, service_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sample_summaries (
                    run_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (run_id, service_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS bucket_stats (
                    run_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    bucket_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (run_id, service_id, policy_id, bucket_index),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS error_bounds (
                    run_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (run_id, service_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS attribution_chain (
                    run_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    bucket_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (run_id, service_id, policy_id, scope, bucket_index),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS flips (
                    run_id TEXT NOT NULL,
                    owner_service TEXT NOT NULL,
                    flip_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (run_id, owner_service, flip_key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_runs_active ON runs(active, stale, status);
                CREATE INDEX IF NOT EXISTS idx_batches_run ON batches(run_id, ordinal);
            """
        )
        with self.transaction() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('created_utc', ?)",
                    (_utc_now(),),
                )
            elif int(row["value"]) != LEDGER_FORMAT_VERSION:
                raise SnapshotError(
                    "$.ledger.format_version",
                    f"unsupported ledger format version {row['value']}",
                )

    def mark_incomplete_batches_unknown(self) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE batches
                SET status='unknown', updated_utc=?
                WHERE status='pending'
                  AND run_id IN (SELECT run_id FROM runs WHERE status='building')
                """,
                (_utc_now(),),
            )
            conn.execute(
                """
                UPDATE runs
                SET status='recovering'
                WHERE status='building'
                  AND EXISTS (
                    SELECT 1 FROM batches
                    WHERE batches.run_id=runs.run_id AND batches.status='unknown'
                  )
                """
            )

    def latest_any_run(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def allocate_versions(
        self, services_doc: ServicesDoc, policies_doc: PoliciesDoc
    ) -> tuple[ServicesDoc, PoliciesDoc]:
        latest = self.latest_any_run()
        if latest is None:
            return services_doc, policies_doc
        if services_doc.content_digest == latest["services_digest"]:
            service_version = max(services_doc.version, int(latest["service_version"]))
        else:
            service_version = max(
                services_doc.version, int(latest["service_version"]) + 1
            )
        if policies_doc.content_digest == latest["policies_digest"]:
            strategy_version = max(
                policies_doc.version, int(latest["strategy_version"])
            )
        else:
            strategy_version = max(
                policies_doc.version, int(latest["strategy_version"]) + 1
            )
        return (
            services_doc.with_version(service_version),
            policies_doc.with_version(strategy_version),
        )

    def _run_by_manifest(self, digest: str, statuses: tuple[str, ...]) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in statuses)
        row = self.connection.execute(
            f"""
            SELECT * FROM runs
            WHERE manifest_digest=? AND status IN ({placeholders})
            ORDER BY rowid DESC LIMIT 1
            """,
            (digest, *statuses),
        ).fetchone()
        return dict(row) if row else None

    def create_or_resume_run(
        self,
        services_doc: ServicesDoc,
        policies_doc: PoliciesDoc,
    ) -> dict[str, Any]:
        digest = manifest_digest(services_doc, policies_doc)
        existing_active = self._run_by_manifest(digest, ("complete",))
        if existing_active and existing_active["active"] and not existing_active["stale"]:
            return {"run_id": existing_active["run_id"], "mode": "existing"}
        existing_building = self._run_by_manifest(digest, ("building", "recovering"))
        if existing_building:
            return {"run_id": existing_building["run_id"], "mode": "resume"}
        with self.transaction() as conn:
            conn.execute("UPDATE runs SET stale=1, active=0")
            next_sequence = conn.execute(
                "SELECT COALESCE(MAX(rowid), 0) + 1 AS next_id FROM runs"
            ).fetchone()["next_id"]
            run_id = f"run-{int(next_sequence):06d}-{digest[:12]}"
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, format_version, service_version, strategy_version,
                    services_digest, policies_digest, manifest_digest,
                    services_json, policies_json, status, active, stale,
                    expected_batches, committed_batches, progress_cursor,
                    snapshot_digest, created_utc, completed_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    LEDGER_FORMAT_VERSION,
                    services_doc.version,
                    policies_doc.version,
                    services_doc.content_digest,
                    policies_doc.content_digest,
                    digest,
                    canonical_json(services_doc.to_dict()),
                    canonical_json(policies_doc.to_dict()),
                    "building",
                    0,
                    0,
                    len(services_doc.services),
                    0,
                    0,
                    None,
                    _utc_now(),
                    None,
                ),
            )
            for ordinal, service in enumerate(services_doc.services):
                conn.execute(
                    """
                    INSERT INTO batches(
                        run_id, ordinal, service_id, status, progress_cursor,
                        content_digest, updated_utc
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (run_id, ordinal, service.id, "pending", ordinal, None, _utc_now()),
                )
        return {"run_id": run_id, "mode": "created"}

    def batch_map(self, run_id: str) -> dict[int, dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? ORDER BY ordinal", (run_id,)
        ).fetchall()
        return {int(row["ordinal"]): dict(row) for row in rows}

    def get_run_row(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise SnapshotError(f"$.runs[{run_id}]", "run does not exist")
        return dict(row)

    def commit_service(
        self,
        run_id: str,
        ordinal: int,
        service_analysis: dict[str, Any],
        flips: list[dict[str, Any]],
    ) -> bool:
        batch_rows = self.batch_map(run_id)
        if ordinal not in batch_rows:
            raise SnapshotError(
                f"$.runs[{run_id}].batches[{ordinal}]", "batch does not exist"
            )
        if batch_rows[ordinal]["status"] == "committed":
            return False
        run = self.get_run_row(run_id)
        if run["status"] not in {"building", "recovering"}:
            raise SnapshotError(
                f"$.runs[{run_id}].status",
                f"cannot commit to run in status {run['status']}",
            )
        service_id = service_analysis["service_id"]
        if service_id != batch_rows[ordinal]["service_id"]:
            raise SnapshotError(
                f"$.runs[{run_id}].batches[{ordinal}].service_id",
                "batch service does not match analysis",
            )
        split = split_service_analysis(service_analysis)
        owner_flips = sorted(
            [flip for flip in flips if flip["owner_service"] == service_id],
            key=lambda item: item["flip_key"],
        )
        batch_digest = batch_content_digest(
            run_id, ordinal, service_id, split, owner_flips
        )
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM sample_summaries WHERE run_id=? AND service_id=?",
                (run_id, service_id),
            )
            conn.execute(
                "DELETE FROM bucket_stats WHERE run_id=? AND service_id=?",
                (run_id, service_id),
            )
            conn.execute(
                "DELETE FROM error_bounds WHERE run_id=? AND service_id=?",
                (run_id, service_id),
            )
            conn.execute(
                "DELETE FROM attribution_chain WHERE run_id=? AND service_id=?",
                (run_id, service_id),
            )
            conn.execute(
                "DELETE FROM flips WHERE run_id=? AND owner_service=?",
                (run_id, service_id),
            )
            summary_json = canonical_json(split["summary"])
            conn.execute(
                """
                INSERT INTO sample_summaries(run_id, service_id, payload_json, payload_digest)
                VALUES(?,?,?,?)
                """,
                (run_id, service_id, summary_json, sha256_json(split["summary"])),
            )
            error_json = canonical_json(split["error_bounds"])
            conn.execute(
                """
                INSERT INTO error_bounds(run_id, service_id, payload_json, payload_digest)
                VALUES(?,?,?,?)
                """,
                (run_id, service_id, error_json, sha256_json(split["error_bounds"])),
            )
            for bucket in split["buckets"]:
                payload_json = canonical_json(bucket)
                conn.execute(
                    """
                    INSERT INTO bucket_stats(
                        run_id, service_id, policy_id, bucket_index,
                        payload_json, payload_digest
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        service_id,
                        bucket["policy_id"],
                        int(bucket["bucket_index"]),
                        payload_json,
                        sha256_json(bucket),
                    ),
                )
            for chain in split["attribution"]:
                bucket_index = -1 if chain["scope"] == "total" else int(chain["bucket_index"])
                payload_json = canonical_json(chain)
                conn.execute(
                    """
                    INSERT INTO attribution_chain(
                        run_id, service_id, policy_id, scope, bucket_index,
                        payload_json, payload_digest
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        service_id,
                        chain["policy_id"],
                        chain["scope"],
                        bucket_index,
                        payload_json,
                        sha256_json(chain),
                    ),
                )
            for flip in owner_flips:
                payload_json = canonical_json(flip["payload"])
                conn.execute(
                    """
                    INSERT INTO flips(
                        run_id, owner_service, flip_key, kind,
                        payload_json, payload_digest
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        service_id,
                        flip["flip_key"],
                        flip["kind"],
                        payload_json,
                        sha256_json(flip["payload"]),
                    ),
                )
            conn.execute(
                """
                UPDATE batches
                SET status='committed', progress_cursor=?, content_digest=?, updated_utc=?
                WHERE run_id=? AND ordinal=?
                """,
                (ordinal + 1, batch_digest, _utc_now(), run_id, ordinal),
            )
            rows = conn.execute(
                "SELECT ordinal, status FROM batches WHERE run_id=? ORDER BY ordinal",
                (run_id,),
            ).fetchall()
            prefix = 0
            for row in rows:
                if row["status"] != "committed":
                    break
                prefix = int(row["ordinal"]) + 1
            committed = sum(1 for row in rows if row["status"] == "committed")
            conn.execute(
                """
                UPDATE runs
                SET committed_batches=?, progress_cursor=?, status='building'
                WHERE run_id=?
                """,
                (committed, prefix, run_id),
            )
        return True

    def _load_json(self, text: str, path: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SnapshotError(path, "ledger JSON payload is invalid") from exc

    def _snapshot_object(self, run_id: str) -> dict[str, Any]:
        run = self.get_run_row(run_id)
        services = self._load_json(run["services_json"], f"$.runs[{run_id}].services")
        policies = self._load_json(run["policies_json"], f"$.runs[{run_id}].policies")
        batch_rows = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? ORDER BY ordinal", (run_id,)
        ).fetchall()
        batches: list[dict[str, Any]] = []
        for row in batch_rows:
            summary_row = self.connection.execute(
                "SELECT * FROM sample_summaries WHERE run_id=? AND service_id=?",
                (run_id, row["service_id"]),
            ).fetchone()
            error_row = self.connection.execute(
                "SELECT * FROM error_bounds WHERE run_id=? AND service_id=?",
                (run_id, row["service_id"]),
            ).fetchone()
            bucket_rows = self.connection.execute(
                """
                SELECT * FROM bucket_stats
                WHERE run_id=? AND service_id=?
                ORDER BY policy_id, bucket_index
                """,
                (run_id, row["service_id"]),
            ).fetchall()
            attribution_rows = self.connection.execute(
                """
                SELECT * FROM attribution_chain
                WHERE run_id=? AND service_id=?
                ORDER BY policy_id, CASE scope WHEN 'total' THEN 0 ELSE 1 END, bucket_index
                """,
                (run_id, row["service_id"]),
            ).fetchall()
            flip_rows = self.connection.execute(
                """
                SELECT * FROM flips
                WHERE run_id=? AND owner_service=?
                ORDER BY flip_key
                """,
                (run_id, row["service_id"]),
            ).fetchall()
            if row["status"] == "committed":
                if summary_row is None or error_row is None:
                    raise SnapshotError(
                        f"$.runs[{run_id}].batches[{row['ordinal']}]",
                        "committed batch is missing required payloads",
                    )
            batches.append(
                {
                    "ordinal": int(row["ordinal"]),
                    "service_id": row["service_id"],
                    "status": row["status"],
                    "progress_cursor": int(row["progress_cursor"]),
                    "content_digest": row["content_digest"],
                    "summary": (
                        self._load_json(
                            summary_row["payload_json"],
                            f"$.runs[{run_id}].services[{row['service_id']}].summary",
                        )
                        if summary_row
                        else None
                    ),
                    "error_bounds": (
                        self._load_json(
                            error_row["payload_json"],
                            f"$.runs[{run_id}].services[{row['service_id']}].error_bounds",
                        )
                        if error_row
                        else None
                    ),
                    "buckets": [
                        self._load_json(
                            bucket["payload_json"],
                            f"$.runs[{run_id}].services[{row['service_id']}].buckets[{bucket['bucket_index']}]",
                        )
                        for bucket in bucket_rows
                    ],
                    "attribution": [
                        self._load_json(
                            chain["payload_json"],
                            f"$.runs[{run_id}].services[{row['service_id']}].attribution[{chain['bucket_index']}]",
                        )
                        for chain in attribution_rows
                    ],
                    "flips": [
                        {
                            "flip_key": flip["flip_key"],
                            "kind": flip["kind"],
                            "payload": self._load_json(
                                flip["payload_json"],
                                f"$.runs[{run_id}].services[{row['service_id']}].flips[{flip['flip_key']}]",
                            ),
                        }
                        for flip in flip_rows
                    ],
                }
            )
        all_flips = self.connection.execute(
            "SELECT * FROM flips WHERE run_id=? ORDER BY owner_service, flip_key", (run_id,)
        ).fetchall()
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "run_id": run_id,
            "service_version": int(run["service_version"]),
            "strategy_version": int(run["strategy_version"]),
            "services_digest": run["services_digest"],
            "policies_digest": run["policies_digest"],
            "manifest_digest": run["manifest_digest"],
            "expected_batches": int(run["expected_batches"]),
            "progress_cursor": int(run["progress_cursor"]),
            "services": services,
            "policies": policies,
            "batches": batches,
            "flips": [
                {
                    "owner_service": row["owner_service"],
                    "flip_key": row["flip_key"],
                    "kind": row["kind"],
                    "payload": self._load_json(
                        row["payload_json"],
                        f"$.runs[{run_id}].flips[{row['flip_key']}]",
                    ),
                }
                for row in all_flips
            ],
        }

    def finalize_run(self, run_id: str) -> str:
        batches = self.batch_map(run_id)
        unfinished = [
            row["service_id"] for row in batches.values() if row["status"] != "committed"
        ]
        if unfinished:
            raise SnapshotError(
                f"$.runs[{run_id}].batches",
                f"cannot finalize with unfinished services: {', '.join(unfinished)}",
            )
        with self.transaction() as conn:
            snapshot = self._snapshot_object(run_id)
            digest = sha256_json(snapshot)
            conn.execute("UPDATE runs SET active=0, stale=1 WHERE run_id<>?", (run_id,))
            conn.execute(
                """
                UPDATE runs
                SET status='complete', active=1, stale=0, snapshot_digest=?,
                    progress_cursor=?, completed_utc=?
                WHERE run_id=?
                """,
                (digest, len(batches), _utc_now(), run_id),
            )
        return digest

    def get_current_run_row(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM runs
            WHERE active=1 AND stale=0 AND status='complete'
            ORDER BY rowid DESC LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None

    def verify_run(self, run_id: str) -> str:
        run = self.get_run_row(run_id)
        expected = run.get("snapshot_digest")
        if expected is None:
            raise SnapshotError(
                f"$.runs[{run_id}].snapshot_digest",
                "run has no committed snapshot digest",
            )
        actual = sha256_json(self._snapshot_object(run_id))
        if actual != expected:
            raise SnapshotError(
                f"$.runs[{run_id}].snapshot_digest",
                "ledger content digest does not match",
                {"expected": expected, "actual": actual},
            )
        return actual

    def get_run_view(self, run_id: str | None = None, *, verify: bool = True) -> dict[str, Any] | None:
        if run_id is None:
            row = self.get_current_run_row()
            if row is None:
                return None
            run_id = row["run_id"]
        run = self.get_run_row(run_id)
        if verify:
            self.verify_run(run_id)
        batch_rows = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? ORDER BY ordinal", (run_id,)
        ).fetchall()
        services: list[dict[str, Any]] = []
        for batch in batch_rows:
            summary_row = self.connection.execute(
                "SELECT payload_json FROM sample_summaries WHERE run_id=? AND service_id=?",
                (run_id, batch["service_id"]),
            ).fetchone()
            error_row = self.connection.execute(
                "SELECT payload_json FROM error_bounds WHERE run_id=? AND service_id=?",
                (run_id, batch["service_id"]),
            ).fetchone()
            bucket_rows = self.connection.execute(
                """
                SELECT payload_json FROM bucket_stats
                WHERE run_id=? AND service_id=?
                ORDER BY policy_id, bucket_index
                """,
                (run_id, batch["service_id"]),
            ).fetchall()
            attribution_rows = self.connection.execute(
                """
                SELECT payload_json FROM attribution_chain
                WHERE run_id=? AND service_id=?
                ORDER BY policy_id, CASE scope WHEN 'total' THEN 0 ELSE 1 END, bucket_index
                """,
                (run_id, batch["service_id"]),
            ).fetchall()
            flip_rows = self.connection.execute(
                """
                SELECT flip_key, kind, payload_json FROM flips
                WHERE run_id=? AND owner_service=? ORDER BY flip_key
                """,
                (run_id, batch["service_id"]),
            ).fetchall()
            services.append(
                {
                    "ordinal": int(batch["ordinal"]),
                    "service_id": batch["service_id"],
                    "status": batch["status"],
                    "progress_cursor": int(batch["progress_cursor"]),
                    "content_digest": batch["content_digest"],
                    "summary": (
                        self._load_json(summary_row["payload_json"], "$.summary")
                        if summary_row
                        else None
                    ),
                    "error_bounds": (
                        self._load_json(error_row["payload_json"], "$.error_bounds")
                        if error_row
                        else None
                    ),
                    "buckets": [
                        self._load_json(row["payload_json"], "$.buckets")
                        for row in bucket_rows
                    ],
                    "attribution": [
                        self._load_json(row["payload_json"], "$.attribution")
                        for row in attribution_rows
                    ],
                    "flips": [
                        {
                            "flip_key": row["flip_key"],
                            "kind": row["kind"],
                            "payload": self._load_json(row["payload_json"], "$.flips"),
                        }
                        for row in flip_rows
                    ],
                }
            )
        flips = self.connection.execute(
            "SELECT owner_service, flip_key, kind, payload_json FROM flips "
            "WHERE run_id=? ORDER BY owner_service, flip_key",
            (run_id,),
        ).fetchall()
        return {
            "run": {
                "run_id": run["run_id"],
                "format_version": int(run["format_version"]),
                "service_version": int(run["service_version"]),
                "strategy_version": int(run["strategy_version"]),
                "services_digest": run["services_digest"],
                "policies_digest": run["policies_digest"],
                "manifest_digest": run["manifest_digest"],
                "status": run["status"],
                "active": bool(run["active"]),
                "stale": bool(run["stale"]),
                "expected_batches": int(run["expected_batches"]),
                "committed_batches": int(run["committed_batches"]),
                "progress_cursor": int(run["progress_cursor"]),
                "snapshot_digest": run["snapshot_digest"],
                "created_utc": run["created_utc"],
                "completed_utc": run["completed_utc"],
                "services": self._load_json(run["services_json"], "$.services"),
                "policies": self._load_json(run["policies_json"], "$.policies"),
            },
            "services": services,
            "flips": [
                {
                    "owner_service": row["owner_service"],
                    "flip_key": row["flip_key"],
                    "kind": row["kind"],
                    "payload": self._load_json(row["payload_json"], "$.flips"),
                }
                for row in flips
            ],
        }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT run_id, format_version, service_version, strategy_version,
                   status, active, stale, expected_batches, committed_batches,
                   progress_cursor, manifest_digest, snapshot_digest,
                   created_utc, completed_utc
            FROM runs ORDER BY rowid DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_batches(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT ordinal, service_id, status, progress_cursor,
                   content_digest, updated_utc
            FROM batches WHERE run_id=? ORDER BY ordinal
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recoverable_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM runs
            WHERE status IN ('building', 'recovering')
            ORDER BY rowid DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def unknown_batches(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? AND status IN ('pending','unknown') "
            "ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def run_documents(self, run_id: str) -> tuple[ServicesDoc, PoliciesDoc]:
        run = self.get_run_row(run_id)
        from .spec import parse_policies_document, parse_services_document

        services = parse_services_document(
            self._load_json(run["services_json"], "$.services")
        )
        policies = parse_policies_document(
            self._load_json(run["policies_json"], "$.policies")
        )
        return services, policies

    def corrupt_snapshot_digest(self, run_id: str) -> str | None:
        run = self.get_run_row(run_id)
        original = run["snapshot_digest"]
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET snapshot_digest=? WHERE run_id=?",
                ("0" * 64, run_id),
            )
        return original

    def restore_snapshot_digest(self, run_id: str, digest: str | None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET snapshot_digest=? WHERE run_id=?",
                (digest, run_id),
            )
