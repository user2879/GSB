"""SQLite regression ledger and session snapshot store.

Durability model
----------------
* One collection batch (per node) is committed in a single SQLite transaction
  containing the diff conclusions, error bounds, findings, attributions and
  the progress cursor.  A forced kill can only leave fully committed batches
  or nothing at all for the in-flight batch.
* Every persisted payload carries a SHA-256 content digest and the integer
  ledger format version.  Reopening a snapshot whose declared version or
  digest mismatches triggers a *whole-store replacement* (never an incremental
  merge) and the replaced rows are listed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_digest
from .errors import ReplayError

LEDGER_FORMAT_VERSION = 1
PRECISION_SCHEME_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    baseline_version INTEGER NOT NULL,
    candidate_version INTEGER NOT NULL,
    committed_at_monotonic_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    batch_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    item_key TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (batch_id, node_id, item_key)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    state TEXT NOT NULL,
    started_monotonic_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot (
    snapshot_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    ledger_format_version INTEGER NOT NULL,
    precision_scheme_version INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("ledger_format_version", str(LEDGER_FORMAT_VERSION)))
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("precision_scheme_version", str(PRECISION_SCHEME_VERSION)))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Leases: mark in-flight batches so forced kills leave detectable gaps
    # ------------------------------------------------------------------ #
    def acquire_lease(self, lease_id: str, node_id: str,
                      started_monotonic_ms: int, payload: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO leases VALUES (?, ?, ?, ?, ?)",
                (lease_id, node_id, "in_flight", started_monotonic_ms,
                 json.dumps(payload, sort_keys=True)))

    def release_lease(self, lease_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM leases WHERE lease_id=?",
                              (lease_id,))

    def open_leases(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM leases WHERE state='in_flight' ORDER BY lease_id"
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Batch commit (single transaction)
    # ------------------------------------------------------------------ #
    def commit_batch(self, batch_id: str, node_id: str,
                     baseline_version: int, candidate_version: int,
                     committed_at_monotonic_ms: int,
                     batch_payload: dict,
                     findings: list[dict]) -> None:
        payload_json = json.dumps(batch_payload, sort_keys=True,
                                  separators=(",", ":"))
        finding_rows = []
        for finding in findings:
            fj = json.dumps(finding, sort_keys=True, separators=(",", ":"))
            finding_rows.append(
                (batch_id, finding["node_id"], finding["key"],
                 finding["status"], fj, canonical_digest(finding)))
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_id, node_id, baseline_version, candidate_version,
                 committed_at_monotonic_ms, payload_json,
                 canonical_digest(batch_payload)))
            self.conn.executemany(
                "INSERT OR REPLACE INTO findings VALUES (?, ?, ?, ?, ?, ?)",
                finding_rows)
            self.conn.execute("DELETE FROM leases WHERE lease_id=?",
                              (batch_id,))

    def committed_batches(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM batches ORDER BY batch_id").fetchall()
        return [dict(row) for row in rows]

    def committed_nodes(self) -> set[str]:
        return {row["node_id"] for row in self.committed_batches()}

    def load_findings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload_json FROM findings "
            "ORDER BY node_id, item_key").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    # ------------------------------------------------------------------ #
    # Snapshot with version/digest identification and whole replacement
    # ------------------------------------------------------------------ #
    def save_snapshot(self, snapshot_id: str, payload: dict,
                      ledger_format_version: int,
                      precision_scheme_version: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM snapshot")
            self.conn.execute(
                "INSERT INTO snapshot VALUES (?, ?, ?, ?, ?)",
                (snapshot_id,
                 json.dumps(payload, sort_keys=True, separators=(",", ":")),
                 canonical_digest(payload), ledger_format_version,
                 precision_scheme_version))

    def load_snapshot(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM snapshot ORDER BY snapshot_id LIMIT 1").fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        return {
            "snapshot_id": row["snapshot_id"],
            "payload": payload,
            "content_digest": row["content_digest"],
            "ledger_format_version": row["ledger_format_version"],
            "precision_scheme_version": row["precision_scheme_version"],
        }

    def verify_or_replace_snapshot(self, expected: dict,
                                   ledger_format_version: int,
                                   precision_scheme_version: int) -> dict:
        """Return a report; replace the whole store on any mismatch."""

        stored = self.load_snapshot()
        replaced: list[dict[str, str]] = []
        if stored is None:
            self.save_snapshot("session", expected, ledger_format_version,
                               precision_scheme_version)
            return {"action": "initialized", "replaced": replaced}
        mismatches = []
        if stored["ledger_format_version"] != ledger_format_version:
            mismatches.append(("ledger_format_version",
                               f"stored={stored['ledger_format_version']} "
                               f"declared={ledger_format_version}"))
        if stored["precision_scheme_version"] != precision_scheme_version:
            mismatches.append(("precision_scheme_version",
                               f"stored={stored['precision_scheme_version']} "
                               f"declared={precision_scheme_version}"))
        if stored["content_digest"] != canonical_digest(expected):
            mismatches.append(("content_digest",
                               f"stored={stored['content_digest'][:16]}... "
                               f"declared={canonical_digest(expected)[:16]}..."))
        if not mismatches:
            return {"action": "reused", "replaced": replaced}
        replaced = [{"subject": subject, "reason": reason}
                    for subject, reason in mismatches]
        self._whole_replace(expected, ledger_format_version,
                            precision_scheme_version)
        return {"action": "replaced", "replaced": replaced}

    def _whole_replace(self, expected: dict, ledger_format_version: int,
                       precision_scheme_version: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM findings")
            self.conn.execute("DELETE FROM batches")
            self.conn.execute("DELETE FROM leases")
            self.conn.execute("DELETE FROM snapshot")
        self.save_snapshot("session", expected, ledger_format_version,
                           precision_scheme_version)

    # ------------------------------------------------------------------ #
    # Replay verification
    # ------------------------------------------------------------------ #
    @staticmethod
    def replay_check(stored_findings: list[dict],
                     replayed_findings: list[dict],
                     stored_attributions: list[dict],
                     replayed_attributions: list[dict]) -> None:
        """Compare ledgered conclusions with replayed ones, first mismatch wins."""

        def keyed(rows: list[dict], *fields: str) -> dict[tuple, dict]:
            return {tuple(row[field] for field in fields): row for row in rows}

        stored_f = keyed(stored_findings, "node_id", "key")
        replay_f = keyed(replayed_findings, "node_id", "key")
        for identity in sorted(set(stored_f) | set(replay_f)):
            if identity not in stored_f:
                raise ReplayError(
                    "replay produced a finding absent from the ledger",
                    [(".".join(identity), "missing in ledger")])
            if identity not in replay_f:
                raise ReplayError(
                    "ledger finding missing from replay",
                    [(".".join(identity), "missing in replay")])
            old, new = stored_f[identity], replay_f[identity]
            for field_name in sorted(set(old) | set(new)):
                if old.get(field_name) != new.get(field_name):
                    raise ReplayError(
                        "replay finding mismatch",
                        [(f"{'.'.join(identity)}.{field_name}",
                          f"ledger={old.get(field_name)!r} "
                          f"replay={new.get(field_name)!r}")],
                    )
        stored_a = keyed(stored_attributions, "node_id", "key")
        replay_a = keyed(replayed_attributions, "node_id", "key")
        for identity in sorted(set(stored_a) | set(replay_a)):
            if identity not in stored_a:
                raise ReplayError(
                    "replay produced an attribution absent from the ledger",
                    [(".".join(identity), "missing in ledger")])
            if identity not in replay_a:
                raise ReplayError(
                    "ledger attribution missing from replay",
                    [(".".join(identity), "missing in replay")])
            old, new = stored_a[identity], replay_a[identity]
            for field_name in sorted(set(old) | set(new)):
                if old.get(field_name) != new.get(field_name):
                    raise ReplayError(
                        "replay attribution mismatch",
                        [(f"{'.'.join(identity)}.{field_name}",
                          f"ledger={old.get(field_name)!r} "
                          f"replay={new.get(field_name)!r}")],
                    )
