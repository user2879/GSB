from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from . import RECORD_FORMAT_VERSION
from .errors import DivergenceError, LedgerError
from .hashing import canonical_bytes, canonical_dumps, content_digest, sha256_hex
from .spec import SpecBundle


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spec_snapshot (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    spec_version TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    backup_path TEXT,
    created_at_wall_ms INTEGER NOT NULL,
    replaced_at_wall_ms INTEGER
);
CREATE TABLE IF NOT EXISTS snapshot_replacements (
    replacement_id TEXT PRIMARY KEY,
    old_format_version INTEGER NOT NULL,
    old_spec_digest TEXT NOT NULL,
    new_format_version INTEGER NOT NULL,
    new_spec_digest TEXT NOT NULL,
    backup_path TEXT NOT NULL,
    diff_json TEXT NOT NULL,
    created_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS commits (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    committed_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_status (
    shard_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    current_outcome_id TEXT,
    updated_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    resource_units INTEGER NOT NULL,
    resource_units_q INTEGER NOT NULL,
    status TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_wall_ms INTEGER NOT NULL,
    reclaimed_at_wall_ms INTEGER,
    reclaim_reason TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    outcome_revision INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    committed_at_wall_ms INTEGER NOT NULL,
    UNIQUE(shard_id, outcome_revision)
);
CREATE TABLE IF NOT EXISTS current_outcomes (
    shard_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_records (
    lease_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    outcome_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (lease_id, outcome_id)
);
CREATE TABLE IF NOT EXISTS resource_records (
    shard_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS water_records (
    shard_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS error_records (
    shard_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_shard_id TEXT,
    safe_cursor_shard_id TEXT,
    committed_count INTEGER NOT NULL,
    topo_order_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_events (
    recovery_id TEXT PRIMARY KEY,
    started_at_wall_ms INTEGER NOT NULL,
    reclaimed_json TEXT NOT NULL,
    unknown_shards_json TEXT NOT NULL,
    replay_json TEXT NOT NULL
);
"""


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


def json_diff(old: Any, new: Any, path: str = "$") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if type(old) is not type(new) and not (
        isinstance(old, (int, float)) and isinstance(new, (int, float))
    ):
        return [{"path": path, "op": "type_changed", "old": old, "new": new}]
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}"
            if key not in old:
                differences.append({"path": child, "op": "added", "new": new[key]})
            elif key not in new:
                differences.append({"path": child, "op": "removed", "old": old[key]})
            else:
                differences.extend(json_diff(old[key], new[key], child))
    elif isinstance(old, list) and isinstance(new, list):
        for index in range(max(len(old), len(new))):
            child = f"{path}[{index}]"
            if index >= len(old):
                differences.append({"path": child, "op": "added", "new": new[index]})
            elif index >= len(new):
                differences.append({"path": child, "op": "removed", "old": old[index]})
            else:
                differences.extend(json_diff(old[index], new[index], child))
    elif old != new:
        differences.append({"path": path, "op": "changed", "old": old, "new": new})
    return differences


def first_field_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                return {"path": child, "reason": "unexpected_actual_field", "expected": None, "actual": actual[key]}
            if key not in actual:
                return {"path": child, "reason": "missing_actual_field", "expected": expected[key], "actual": None}
            difference = first_field_difference(expected[key], actual[key], child)
            if difference:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return {
                "path": f"{path}.length",
                "reason": "list_length",
                "expected": len(expected),
                "actual": len(actual),
            }
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_field_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return {"path": path, "reason": "value_mismatch", "expected": expected, "actual": actual}
    return None


class Ledger:
    def __init__(self, path: str | Path, spec: SpecBundle) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.path.parent / "snapshot_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self.snapshot_report = self._ensure_snapshot()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('format_version', ?)",
            (str(RECORD_FORMAT_VERSION),),
        )
        self.conn.commit()

    def _ensure_snapshot(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM spec_snapshot WHERE id = 1").fetchone()
        now = wall_ms()
        payload = self.spec.raw
        payload_json = canonical_dumps(payload)
        digest = sha256_hex(payload)
        if row is None:
            content_digest_value = sha256_hex(
                {"format_version": RECORD_FORMAT_VERSION, "spec_digest": self.spec.spec_digest, "payload": payload}
            )
            self.conn.execute(
                """INSERT INTO spec_snapshot(id, format_version, spec_version, spec_digest,
                   payload_json, content_digest, created_at_wall_ms)
                   VALUES(1, ?, ?, ?, ?, ?, ?)""",
                (
                    RECORD_FORMAT_VERSION,
                    self.spec.session.spec_version,
                    self.spec.spec_digest,
                    payload_json,
                    content_digest_value,
                    now,
                ),
            )
            self.conn.commit()
            return {"action": "created", "spec_digest": self.spec.spec_digest, "differences": []}

        same_version = int(row["format_version"]) == RECORD_FORMAT_VERSION
        same_digest = row["spec_digest"] == self.spec.spec_digest
        if same_version and same_digest:
            return {
                "action": "unchanged",
                "spec_digest": self.spec.spec_digest,
                "differences": [],
                "backup_path": row["backup_path"],
            }

        old_payload = json.loads(row["payload_json"])
        differences = json_diff(old_payload, payload)
        backup_stem = f"snapshot-{row['spec_digest'][:12]}-{now}-{uuid.uuid4().hex[:8]}"
        backup_path = self.backup_dir / f"{backup_stem}.json"
        diff_path = self.backup_dir / f"{backup_stem}.diff.json"
        backup_document = {
            "read_only": True,
            "format_version": row["format_version"],
            "spec_version": row["spec_version"],
            "spec_digest": row["spec_digest"],
            "content_digest": row["content_digest"],
            "payload": old_payload,
        }
        backup_path.write_text(json.dumps(backup_document, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        diff_path.write_text(json.dumps(differences, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.chmod(backup_path, 0o444)
        os.chmod(diff_path, 0o444)

        replacement_id = uuid.uuid4().hex
        new_content_digest = sha256_hex(
            {"format_version": RECORD_FORMAT_VERSION, "spec_digest": self.spec.spec_digest, "payload": payload}
        )
        with self.conn:
            self.conn.execute(
                """INSERT INTO snapshot_replacements(replacement_id, old_format_version,
                   old_spec_digest, new_format_version, new_spec_digest, backup_path,
                   diff_json, created_at_wall_ms) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    replacement_id,
                    row["format_version"],
                    row["spec_digest"],
                    RECORD_FORMAT_VERSION,
                    self.spec.spec_digest,
                    str(backup_path),
                    canonical_dumps(differences),
                    now,
                ),
            )
            self.conn.execute(
                """UPDATE spec_snapshot SET format_version=?, spec_version=?, spec_digest=?,
                   payload_json=?, content_digest=?, backup_path=?, replaced_at_wall_ms=?
                   WHERE id=1""",
                (
                    RECORD_FORMAT_VERSION,
                    self.spec.session.spec_version,
                    self.spec.spec_digest,
                    payload_json,
                    new_content_digest,
                    str(backup_path),
                    now,
                ),
            )
        return {
            "action": "replaced",
            "replacement_id": replacement_id,
            "old_spec_digest": row["spec_digest"],
            "new_spec_digest": self.spec.spec_digest,
            "backup_path": str(backup_path),
            "diff_path": str(diff_path),
            "differences": differences,
        }

    def commit_reservation(self, reservation: dict[str, Any]) -> dict[str, Any]:
        payload = dict(reservation)
        payload["format_version"] = RECORD_FORMAT_VERSION
        payload["spec_digest"] = self.spec.spec_digest
        digest = content_digest(payload)
        payload["content_digest"] = digest
        reservation_id = str(payload["reservation_id"])
        now = wall_ms()
        with self.conn:
            self.conn.execute(
                """INSERT INTO reservations(reservation_id, shard_id, node_id, lease_id,
                   resource_units, resource_units_q, status, format_version, spec_digest,
                   content_digest, payload_json, created_at_wall_ms)
                   VALUES(?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
                (
                    reservation_id,
                    payload["shard_id"],
                    payload["node_id"],
                    payload["lease_id"],
                    int(payload["resource_units"]),
                    int(payload["resource_units_q"]),
                    RECORD_FORMAT_VERSION,
                    self.spec.spec_digest,
                    digest,
                    canonical_dumps(payload),
                    now,
                ),
            )
            self.conn.execute(
                """INSERT INTO shard_status(shard_id, status, updated_at_wall_ms)
                   VALUES(?, 'dispatched', ?)
                   ON CONFLICT(shard_id) DO UPDATE SET status='dispatched',
                     current_outcome_id=NULL, updated_at_wall_ms=excluded.updated_at_wall_ms""",
                (payload["shard_id"], now),
            )
        return payload

    def active_reservations(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM reservations WHERE status='active' ORDER BY created_at_wall_ms, reservation_id"
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def recover(self) -> dict[str, Any]:
        active_rows = self.conn.execute(
            "SELECT reservation_id, payload_json FROM reservations WHERE status='active'"
        ).fetchall()
        reclaimed: list[dict[str, Any]] = []
        now = wall_ms()
        with self.conn:
            for row in active_rows:
                payload = json.loads(row["payload_json"])
                payload["recovery"] = {"status": "reclaimed", "reclaimed_at_wall_ms": now}
                reclaimed.append(payload)
                self.conn.execute(
                    """UPDATE reservations SET status='reclaimed',
                       reclaimed_at_wall_ms=?, reclaim_reason='process_terminated_before_commit'
                       WHERE reservation_id=?""",
                    (now, row["reservation_id"]),
                )
                self.conn.execute(
                    """UPDATE shard_status SET status='unknown', current_outcome_id=NULL,
                       updated_at_wall_ms=? WHERE shard_id=? AND status != 'committed'""",
                    (now, payload["shard_id"]),
                )

        unknown_rows = self.conn.execute(
            "SELECT shard_id FROM shard_status WHERE status='unknown' ORDER BY shard_id"
        ).fetchall()
        unknown_shards = [row["shard_id"] for row in unknown_rows]
        replay = self.replay_verify()
        recovery_id = uuid.uuid4().hex
        with self.conn:
            self.conn.execute(
                "INSERT INTO recovery_events VALUES(?, ?, ?, ?, ?)",
                (
                    recovery_id,
                    now,
                    canonical_dumps(reclaimed),
                    canonical_dumps(unknown_shards),
                    canonical_dumps(replay),
                ),
            )
        return {
            "recovery_id": recovery_id,
            "reclaimed_reservations": reclaimed,
            "reclaimed_resource_units_total": sum(int(item["resource_units"]) for item in reclaimed),
            "unknown_shards": unknown_shards,
            "replay": replay,
            "note": "SIGKILL-time exit report cannot be reconstructed; only committed transactions are authoritative.",
        }

    def committed_shard_ids(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT shard_id FROM shard_status WHERE status='committed'"
        ).fetchall()
        return {row["shard_id"] for row in rows}

    def current_outcome(self, shard_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM current_outcomes WHERE shard_id=?", (shard_id,)
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def current_payload_bytes(self) -> dict[str, bytes]:
        rows = self.conn.execute("SELECT shard_id, payload_json FROM current_outcomes").fetchall()
        return {row["shard_id"]: row["payload_json"].encode("utf-8") for row in rows}

    def all_current_outcomes(self) -> dict[str, dict[str, Any]]:
        return {shard_id: json.loads(payload.decode("utf-8")) for shard_id, payload in self.current_payload_bytes().items()}

    def commit_shard_outcome(self, record: dict[str, Any], *, topo_order: list[str]) -> dict[str, Any]:
        payload = dict(record)
        shard_id = str(payload["shard_id"])
        payload["format_version"] = RECORD_FORMAT_VERSION
        payload["spec_digest"] = self.spec.spec_digest
        payload["topo_rank"] = topo_order.index(shard_id)
        digest = content_digest(payload)
        payload["content_digest"] = digest
        payload_json = canonical_dumps(payload)
        outcome_id = f"{shard_id}:{digest[:20]}"
        revision_row = self.conn.execute(
            "SELECT COALESCE(MAX(outcome_revision), 0) + 1 AS next_revision FROM outcomes WHERE shard_id=?",
            (shard_id,),
        ).fetchone()
        revision = int(revision_row["next_revision"])
        now = wall_ms()
        commit_id = f"commit-{outcome_id}-r{revision}"

        committed = set(self.committed_shard_ids())
        committed.add(shard_id)
        safe_cursor = None
        for candidate in topo_order:
            if candidate in committed:
                safe_cursor = candidate
            else:
                break
        cursor = {
            "last_committed_shard_id": shard_id,
            "safe_cursor_shard_id": safe_cursor,
            "committed_count": len(committed),
            "topo_order": topo_order,
            "spec_digest": self.spec.spec_digest,
        }
        cursor_digest = content_digest(cursor)
        cursor["content_digest"] = cursor_digest
        cursor_json = canonical_dumps(cursor)

        with self.conn:
            self.conn.execute(
                """INSERT INTO commits(commit_id, kind, format_version, spec_digest,
                   content_digest, payload_json, committed_at_wall_ms)
                   VALUES(?, 'shard_outcome', ?, ?, ?, ?, ?)""",
                (commit_id, RECORD_FORMAT_VERSION, self.spec.spec_digest, digest, payload_json, now),
            )
            self.conn.execute(
                """INSERT INTO outcomes(outcome_id, shard_id, outcome_revision, format_version,
                   spec_digest, content_digest, payload_json, committed_at_wall_ms)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (outcome_id, shard_id, revision, RECORD_FORMAT_VERSION, self.spec.spec_digest, digest, payload_json, now),
            )
            self.conn.execute(
                """INSERT INTO current_outcomes(shard_id, outcome_id, payload_json, updated_at_wall_ms)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(shard_id) DO UPDATE SET outcome_id=excluded.outcome_id,
                     payload_json=excluded.payload_json, updated_at_wall_ms=excluded.updated_at_wall_ms""",
                (shard_id, outcome_id, payload_json, now),
            )
            lease = payload["lease"]
            self.conn.execute(
                "INSERT INTO lease_records VALUES(?, ?, ?, ?, ?, ?)",
                (
                    lease["lease_id"],
                    shard_id,
                    outcome_id,
                    int(lease["version"]),
                    str(lease.get("status", "released")),
                    canonical_dumps(lease),
                ),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO resource_records VALUES(?, ?, ?)",
                (shard_id, outcome_id, canonical_dumps(payload["resource"])),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO water_records VALUES(?, ?, ?)",
                (shard_id, outcome_id, canonical_dumps(payload["water"])),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO error_records VALUES(?, ?, ?)",
                (shard_id, outcome_id, canonical_dumps(payload["error_attribution"])),
            )
            reservation_id = payload.get("reservation_id")
            if reservation_id:
                self.conn.execute(
                    "UPDATE reservations SET status='committed' WHERE reservation_id=?",
                    (reservation_id,),
                )
            self.conn.execute(
                """INSERT INTO shard_status(shard_id, status, current_outcome_id, updated_at_wall_ms)
                   VALUES(?, 'committed', ?, ?)
                   ON CONFLICT(shard_id) DO UPDATE SET status='committed',
                     current_outcome_id=excluded.current_outcome_id,
                     updated_at_wall_ms=excluded.updated_at_wall_ms""",
                (shard_id, outcome_id, now),
            )
            self.conn.execute(
                """INSERT INTO progress_cursor(id, last_committed_shard_id, safe_cursor_shard_id,
                   committed_count, topo_order_json, format_version, content_digest,
                   payload_json, updated_at_wall_ms)
                   VALUES(1, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     last_committed_shard_id=excluded.last_committed_shard_id,
                     safe_cursor_shard_id=excluded.safe_cursor_shard_id,
                     committed_count=excluded.committed_count,
                     topo_order_json=excluded.topo_order_json,
                     format_version=excluded.format_version,
                     content_digest=excluded.content_digest,
                     payload_json=excluded.payload_json,
                     updated_at_wall_ms=excluded.updated_at_wall_ms""",
                (
                    shard_id,
                    safe_cursor,
                    len(committed),
                    canonical_dumps(topo_order),
                    RECORD_FORMAT_VERSION,
                    cursor_digest,
                    cursor_json,
                    now,
                ),
            )
        return payload

    def replay_verify(self) -> dict[str, Any]:
        commit_rows = self.conn.execute("SELECT * FROM commits ORDER BY seq").fetchall()
        replayed_current_json: dict[str, str] = {}
        first_mismatch: dict[str, Any] | None = None
        for row in commit_rows:
            payload = json.loads(row["payload_json"])
            expected_digest = content_digest(payload)
            if expected_digest != row["content_digest"] or expected_digest != payload.get("content_digest"):
                first_mismatch = {
                    "path": f"$.commits[{row['seq']}].content_digest",
                    "expected": row["content_digest"],
                    "actual": expected_digest,
                }
                break
            if row["kind"] == "shard_outcome":
                replayed_current_json[payload["shard_id"]] = row["payload_json"]

        rows = self.conn.execute("SELECT * FROM outcomes ORDER BY outcome_revision, shard_id").fetchall()
        verified: list[str] = []
        if first_mismatch is None:
            for row in rows:
                payload = json.loads(row["payload_json"])
                expected_digest = content_digest(payload)
                if expected_digest != row["content_digest"] or expected_digest != payload.get("content_digest"):
                    first_mismatch = {
                        "path": f"$.outcomes[{row['shard_id']}].content_digest",
                        "expected": row["content_digest"],
                        "actual": expected_digest,
                    }
                    break
                verified.append(row["outcome_id"])

        current_rows = self.conn.execute("SELECT * FROM current_outcomes").fetchall()
        current_mismatches = 0
        if first_mismatch is None:
            for row in current_rows:
                outcome = self.conn.execute(
                    "SELECT payload_json, content_digest FROM outcomes WHERE outcome_id=?",
                    (row["outcome_id"],),
                ).fetchone()
                replayed_payload = replayed_current_json.get(row["shard_id"])
                if (
                    outcome is None
                    or outcome["payload_json"] != row["payload_json"]
                    or replayed_payload != row["payload_json"]
                ):
                    current_mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "path": f"$.current_outcomes[{row['shard_id']}]",
                            "expected": outcome["payload_json"] if outcome else replayed_payload,
                            "actual": row["payload_json"],
                        }
        if len(replayed_current_json) != len(current_rows) and first_mismatch is None:
            current_mismatches += 1
        ok = first_mismatch is None and current_mismatches == 0
        return {
            "ok": ok,
            "replayed_commit_count": len(commit_rows),
            "replayed_current_count": len(replayed_current_json),
            "verified_outcome_count": len(verified),
            "current_count": len(current_rows),
            "current_mismatches": current_mismatches,
            "first_mismatch": first_mismatch,
        }

    def assert_replay_ok(self) -> dict[str, Any]:
        report = self.replay_verify()
        if not report["ok"]:
            raise DivergenceError(
                "ledger replay verification failed",
                path=str((report["first_mismatch"] or {}).get("path", "$")),
                details=report,
            )
        return report

    def export_deterministic_outcomes(self) -> dict[str, Any]:
        """Export fields that a blank deterministic rebuild must reproduce."""

        result: dict[str, Any] = {}
        for shard_id, record in self.all_current_outcomes().items():
            result[shard_id] = deterministic_projection(record)
        return result


IGNORED_RUNTIME_PATHS = {
    "$.wall",
    "$.timing",
    "$.rpc_receipts",
    "$.reservation_id",
    "$.outcome_revision",
}


def deterministic_projection(record: dict[str, Any]) -> dict[str, Any]:
    projected = {
        key: record[key]
        for key in (
            "format_version",
            "shard_id",
            "node_id",
            "topo_rank",
            "sample_set_digest",
            "summary",
            "lease",
            "water",
            "error_attribution",
            "dependency_digests",
        )
        if key in record
    }
    lease = dict(projected.get("lease") or {})
    for key in (
        "granted_at_monotonic_ms",
        "expires_at_monotonic_ms",
        "granted_at_wall_ms",
        "remaining_ttl_ms",
    ):
        lease.pop(key, None)
    projected["lease"] = lease
    resource = dict(record.get("resource") or {})
    projected["resource"] = {
        key: resource[key]
        for key in (
            "resource_units",
            "resource_units_q",
            "global_budget_units",
            "global_budget_q",
            "node_concurrency_quota",
            "node_concurrency_quota_q",
            "node_max_leases",
            "node_max_leases_q",
        )
        if key in resource
    }
    timing = record.get("timing") or {}
    projected["timing_invariants"] = {
        "wait_duration_nonnegative": int(timing.get("wait_duration_ms", 0)) >= 0,
        "rpc_elapsed_nonnegative": int(timing.get("rpc_elapsed_ms", 0)) >= 0,
    }
    return projected


def compare_deterministic_exports(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    difference = first_field_difference(expected, actual, "$.outcomes")
    if difference:
        raise DivergenceError(
            "incremental ledger differs from blank full rebuild",
            path=difference["path"],
            details={"difference": difference},
        )
    return {"ok": True, "shard_count": len(actual), "first_difference": None}
