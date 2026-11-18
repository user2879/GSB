"""SQLite window ledger: atomic commits, rule-version lineage and recovery view."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from .errors import BusinessError
from .engine import WindowResult
from .spec import RuleSpec, Spec
from .util import canonical_json, utc_now_iso
from .errors import LEDGER_FORMAT_VERSION


class Ledger:
    def __init__(self, path: str, *, expected_format_version: int = LEDGER_FORMAT_VERSION):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema(expected_format_version)

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self, expected_format_version: int) -> None:
        # An unsupported requested version is rejected up front, even for a
        # brand-new ledger, so an injected bad version can never be initialised.
        if expected_format_version != LEDGER_FORMAT_VERSION:
            raise BusinessError(
                "E_LEDGER",
                f"unsupported ledger format version {expected_format_version}",
                "ledger.format_version",
                {
                    "ledger_path": os.path.abspath(self.path),
                    "observed_version": expected_format_version,
                    "supported_versions": [LEDGER_FORMAT_VERSION],
                },
            )
        self.conn.execute("CREATE TABLE IF NOT EXISTS ledger_meta (key TEXT PRIMARY KEY, value)")
        row = self.conn.execute("SELECT value FROM ledger_meta WHERE key='format_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO ledger_meta(key, value) VALUES ('format_version', ?)",
                (LEDGER_FORMAT_VERSION,),
            )
            self.conn.commit()
        else:
            observed = row[0]
            if not isinstance(observed, int) or observed != LEDGER_FORMAT_VERSION:
                raise BusinessError(
                    "E_LEDGER",
                    f"unsupported ledger format version {observed}",
                    "ledger.format_version",
                    {
                        "ledger_path": os.path.abspath(self.path),
                        "observed_version": observed,
                        "supported_versions": [LEDGER_FORMAT_VERSION],
                    },
                )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rule_versions (
                rule_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                signature TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(rule_id, version)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                spec_fingerprint TEXT NOT NULL,
                rule_versions_json TEXT NOT NULL,
                requires_rerun INTEGER NOT NULL DEFAULT 0,
                note TEXT
            );
            CREATE TABLE IF NOT EXISTS window_results (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                rule_version INTEGER NOT NULL,
                window_start_ms INTEGER NOT NULL,
                window_end_ms INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('committed', 'stale')),
                committed_at TEXT NOT NULL,
                UNIQUE(rule_id, rule_version, window_start_ms),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_window_current
                ON window_results(rule_id, window_start_ms)
                WHERE status = 'committed';
            CREATE INDEX IF NOT EXISTS ix_window_rule_version
                ON window_results(rule_id, rule_version, status);
            """
        )
        self.conn.commit()

    def start_run(self, run_id: str, spec: Spec) -> None:
        rule_versions = {
            rule.rule_id: {"version": rule.version, "signature": rule.signature()}
            for rule in spec.rules
        }
        self.conn.execute(
            "INSERT INTO runs(run_id, started_at, status, spec_fingerprint, rule_versions_json) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, utc_now_iso(), spec.fingerprint(), canonical_json(rule_versions)),
        )
        self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        requires_rerun: bool = False,
        note: str | None = None,
    ) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, requires_rerun=?, note=? WHERE run_id=?",
            (utc_now_iso(), status, 1 if requires_rerun else 0, note, run_id),
        )
        self.conn.commit()

    def register_rule_versions(self, rules: tuple[RuleSpec, ...]) -> dict[str, list[int]]:
        """Record rule lineage; reject lower versions and mark superseded rows stale."""
        replaced: dict[str, list[int]] = {}
        for rule in rules:
            existing = self.conn.execute(
                "SELECT version, signature FROM rule_versions WHERE rule_id=? ORDER BY version",
                (rule.rule_id,),
            ).fetchall()
            max_version = max((row["version"] for row in existing), default=0)
            if rule.version < max_version:
                raise BusinessError(
                    "E_SPEC",
                    f"rule '{rule.rule_id}' version {rule.version} is lower than recorded {max_version}",
                    f"rule_versions[{rule.rule_id}:{rule.version}]",
                    {"rule_id": rule.rule_id, "recorded_version": max_version, "attempted_version": rule.version},
                )
            same = next((row for row in existing if row["version"] == rule.version), None)
            if same is not None and same["signature"] != rule.signature():
                raise BusinessError(
                    "E_REPLAY",
                    "rule parameters are not reproducible for the recorded version",
                    f"rule_versions[{rule.rule_id}:{rule.version}]",
                    {
                        "rule_id": rule.rule_id,
                        "rule_version": rule.version,
                        "ledger_signature": same["signature"],
                        "current_signature": rule.signature(),
                        "basis": "rule signature mismatch under the same rule version",
                    },
                )
            if same is None:
                self.conn.execute(
                    "INSERT INTO rule_versions(rule_id, version, signature, definition_json, is_current, created_at) "
                    "VALUES (?, ?, ?, ?, 1, ?)",
                    (
                        rule.rule_id,
                        rule.version,
                        rule.signature(),
                        canonical_json(rule.definition_dict()),
                        utc_now_iso(),
                    ),
                )
                if existing:
                    self.conn.execute(
                        "UPDATE rule_versions SET is_current=0 WHERE rule_id=? AND version<>?",
                        (rule.rule_id, rule.version),
                    )
                    stale_rows = self.conn.execute(
                        "UPDATE window_results SET status='stale' WHERE rule_id=? AND rule_version<>? "
                        "AND status='committed'",
                        (rule.rule_id, rule.version),
                    )
                    replaced[rule.rule_id] = [
                        row["window_start_ms"]
                        for row in self.conn.execute(
                            "SELECT window_start_ms FROM window_results WHERE rule_id=? AND status='stale' "
                            "AND rule_version<>? ORDER BY window_start_ms",
                            (rule.rule_id, rule.version),
                        )
                    ]
                    if stale_rows.rowcount == 0:
                        replaced.pop(rule.rule_id, None)
        self.conn.commit()
        return replaced

    def current_committed(self) -> dict[tuple[str, int], sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM window_results WHERE status='committed'"
        ).fetchall()
        return {(row["rule_id"], row["window_start_ms"]): row for row in rows}

    def commit_window(self, run_id: str, result: WindowResult) -> str:
        """Idempotent per-window transaction; a window can never be committed twice."""
        existing = self.conn.execute(
            "SELECT content_digest, status FROM window_results WHERE rule_id=? AND window_start_ms=?",
            (result.rule_id, result.window_start_ms),
        ).fetchone()
        if existing is not None and existing["status"] == "committed":
            if existing["content_digest"] != result.content_digest:
                raise BusinessError(
                    "E_REPLAY",
                    "recomputed window diverges from the committed ledger record",
                    f"window_results[{result.window_id}]",
                    {
                        "window_id": result.window_id,
                        "ledger_digest": existing["content_digest"],
                        "current_digest": result.content_digest,
                    },
                )
            return "reused"
        with self.conn:
            self.conn.execute(
                "INSERT INTO window_results(window_id, run_id, rule_id, rule_version, window_start_ms, "
                "window_end_ms, result_json, event_count, content_digest, status, committed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'committed', ?)",
                (
                    result.window_id,
                    run_id,
                    result.rule_id,
                    result.rule_version,
                    result.window_start_ms,
                    result.window_end_ms,
                    canonical_json(result.result_dict()),
                    result.event_count,
                    result.content_digest,
                    utc_now_iso(),
                ),
            )
        return "inserted"

    def current_view(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM window_results WHERE status='committed' ORDER BY window_start_ms, rule_id"
        ).fetchall()
        return [
            {
                "window_id": row["window_id"],
                "rule_id": row["rule_id"],
                "rule_version": row["rule_version"],
                "window_start_ms": row["window_start_ms"],
                "window_end_ms": row["window_end_ms"],
                "event_count": row["event_count"],
                "content_digest": row["content_digest"],
                "groups": __import__("json").loads(row["result_json"])["groups"],
            }
            for row in rows
        ]

    def stale_view(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT window_id, rule_id, rule_version, window_start_ms FROM window_results "
            "WHERE status='stale' ORDER BY rule_id, window_start_ms"
        ).fetchall()
        return [dict(row) for row in rows]

    def running_runs(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE status='running'").fetchall()
