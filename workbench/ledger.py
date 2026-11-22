"""SQLite ledger: transactional minute commits, versions, snapshots, recovery."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from .errors import SnapshotError
from .models import MinuteDecision, canonical_bytes, digest_of
from .models import AllocationResult, MinuteEstimate

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS versions (
        version INTEGER PRIMARY KEY,
        declaration_digest TEXT NOT NULL,
        declaration_json TEXT NOT NULL,
        created_ns INTEGER NOT NULL,
        active INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stale_versions (
        version INTEGER PRIMARY KEY,
        reason TEXT NOT NULL,
        marked_ns INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        version INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        digest TEXT NOT NULL,
        PRIMARY KEY (version, minute)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS progress (
        version INTEGER PRIMARY KEY,
        committed_minutes INTEGER NOT NULL,
        last_minute INTEGER NOT NULL,
        status TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS latency_samples (
        seq INTEGER NOT NULL,
        version INTEGER NOT NULL,
        kind TEXT NOT NULL,
        service_id TEXT NOT NULL,
        elapsed_ms INTEGER NOT NULL,
        budget_ms INTEGER NOT NULL,
        over_budget INTEGER NOT NULL,
        PRIMARY KEY (version, seq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        version INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        digest TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (version, minute)
    )
    """,
)


def _row_to_estimate(row: dict) -> MinuteEstimate:
    return MinuteEstimate(**row)


def _row_to_allocation(row: dict) -> AllocationResult:
    return AllocationResult(**row)


def decision_from_payload(payload: dict, digest: str) -> MinuteDecision:
    estimates = tuple(_row_to_estimate(row) for row in payload["estimates"])
    allocations = tuple(_row_to_allocation(row) for row in payload["allocations"])
    return MinuteDecision(
        minute=payload["minute"],
        version=payload["version"],
        estimates=estimates,
        allocations=allocations,
        total_allocated=payload["total_allocated"],
        global_budget=payload["global_budget"],
        selection=payload["selection"],
        selection_reason=payload["selection_reason"],
        threshold_boundary=payload["threshold_boundary"],
        estimates_digest=payload["estimates_digest"],
        digest=digest,
    )


class Ledger:
    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._apply_schema()

    def _apply_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self.conn.execute(statement)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def latest_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) AS v FROM versions").fetchone()
        return int(row["v"] or 0)

    def active_version(self) -> int | None:
        row = self.conn.execute(
            "SELECT version FROM versions WHERE active = 1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return None if row is None else int(row["version"])

    def register_version(self, version: int, declaration: dict) -> str:
        declaration_digest = digest_of(declaration)
        previous = self.active_version()
        with self.conn:
            if previous is not None:
                self.conn.execute("UPDATE versions SET active = 0 WHERE version = ?", (previous,))
                self.conn.execute(
                    "INSERT OR REPLACE INTO stale_versions(version, reason, marked_ns) "
                    "VALUES (?, ?, ?)",
                    (previous, "superseded_by_declaration_change", 0),
                )
            self.conn.execute(
                "INSERT INTO versions(version, declaration_digest, declaration_json, "
                "created_ns, active) VALUES (?, ?, ?, ?, 1)",
                (version, declaration_digest,
                 json.dumps(declaration, sort_keys=True, separators=(",", ":")),
                 time.time_ns()),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO progress(version, committed_minutes, last_minute, status) "
                "VALUES (?, 0, -1, 'running')",
                (version,),
            )
        return declaration_digest

    def mark_stale(self, version: int, reason: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE versions SET active = 0 WHERE version = ?", (version,))
            self.conn.execute(
                "INSERT OR REPLACE INTO stale_versions(version, reason, marked_ns) "
                "VALUES (?, ?, ?)",
                (version, reason, time.time_ns()),
            )

    def stale_versions(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT version FROM stale_versions ORDER BY version"
        ).fetchall()
        return [int(row["version"]) for row in rows]

    def commit_minute(self, decision: MinuteDecision) -> str:
        """Atomic single-transaction commit of decision, progress, snapshot."""

        payload = decision.ledger_payload()
        payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = digest_of(payload)
        decision_name = decision.__class__.__name__
        assert decision_name == "MinuteDecision"
        with self.conn:
            self.conn.execute(
                "INSERT INTO decisions(version, minute, payload_json, digest) "
                "VALUES (?, ?, ?, ?)",
                (decision.version, decision.minute, payload_text, digest),
            )
            self.conn.execute(
                "INSERT INTO snapshots(version, minute, digest, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (decision.version, decision.minute, digest, payload_text),
            )
            self.conn.execute(
                "UPDATE progress SET committed_minutes = ?, last_minute = ?, status = ? "
                "WHERE version = ?",
                (decision.minute + 1, decision.minute, "running", decision.version),
            )
        return digest

    def complete_version(self, version: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE progress SET status = 'complete' WHERE version = ?", (version,)
            )

    def progress(self, version: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM progress WHERE version = ?", (version,)
        ).fetchone()
        return None if row is None else dict(row)

    def load_decisions(self, version: int) -> list[MinuteDecision]:
        rows = self.conn.execute(
            "SELECT payload_json, digest FROM decisions WHERE version = ? ORDER BY minute",
            (version,),
        ).fetchall()
        return [
            decision_from_payload(json.loads(row["payload_json"]), row["digest"])
            for row in rows
        ]

    def load_snapshots(self, version: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT version, minute, digest, payload_json FROM snapshots "
            "WHERE version = ? ORDER BY minute",
            (version,),
        ).fetchall()
        return [dict(row) for row in rows]

    def verify_snapshots(self, version: int, *, inject_corrupt: bool = False) -> None:
        """Recompute every snapshot digest; mismatch raises E_SNAPSHOT."""

        for position, row in enumerate(self.load_snapshots(version)):
            payload = json.loads(row["payload_json"])
            effective_payload = payload
            if inject_corrupt and position == 0:
                # Deterministic fault injection: flip one committed field.
                effective_payload = dict(payload)
                effective_payload["total_allocated"] = payload["total_allocated"] + 1
            actual = digest_of(effective_payload)
            if actual != row["digest"]:
                raise SnapshotError(
                    "snapshot digest mismatch",
                    details={
                        "version": version,
                        "minute": row["minute"],
                        "expected_digest": row["digest"],
                        "actual_digest": actual,
                    },
                )

    def next_latency_seq(self, version: int) -> int:
        row = self.conn.execute(
            "SELECT MAX(seq) AS s FROM latency_samples WHERE version = ?", (version,)
        ).fetchone()
        return int(row["s"] or 0) + 1

    def insert_latency_sample(self, version: int, sample: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO latency_samples(seq, version, kind, service_id, elapsed_ms, "
                "budget_ms, over_budget) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sample["seq"],
                    version,
                    sample["kind"],
                    sample["service_id"],
                    sample["elapsed_ms"],
                    sample["budget_ms"],
                    1 if sample["over_budget"] else 0,
                ),
            )

    def latency_samples(self, version: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM latency_samples WHERE version = ? ORDER BY seq", (version,)
        ).fetchall()
        return [dict(row) for row in rows]

    def unknown_minutes(self, version: int, total_minutes: int) -> list[int]:
        info = self.progress(version)
        committed = 0 if info is None else int(info["committed_minutes"])
        return list(range(committed, total_minutes))
