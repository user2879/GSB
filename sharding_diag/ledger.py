"""SQLite atomic ledger with content digests and snapshot replacement.

Every shard outcome and every freeze migration is written in one SQLite
transaction that spans the outcome row, aggregate summary, resource usage,
lease ownership, freeze/revocation records and the progress cursor.  Events
are additionally chained with SHA-256 so the ledger carries a tamper-evident
integer-versioned content digest.

Reopening a snapshot compares the stored node-set version, capacity version
map and content digest against the current declaration.  Any mismatch causes
a whole-file replacement (never an incremental merge) and keeps the old file
as a read-only backup.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from typing import Dict, List, Optional

from .errors import ReplayError, SpecError

LEDGER_FORMAT_VERSION = 1
_BUSY_TIMEOUT_MS = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    shard_id TEXT,
    payload_json TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    chain_hash TEXT NOT NULL,
    at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_outcomes (
    shard_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    node_id TEXT,
    lease_id TEXT,
    lease_version INTEGER,
    capacity_version INTEGER,
    summary_json TEXT,
    summary_digest TEXT,
    fixed_total INTEGER,
    entry_hash TEXT NOT NULL,
    updated_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_usage (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    node_id TEXT,
    lease_id TEXT,
    demand INTEGER NOT NULL,
    delta INTEGER NOT NULL,
    reason TEXT NOT NULL,
    at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_records (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    grant_ms INTEGER NOT NULL,
    expiry_ms INTEGER NOT NULL,
    version INTEGER NOT NULL,
    demand INTEGER NOT NULL,
    released INTEGER NOT NULL,
    release_reason TEXT,
    updated_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS freeze_records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    missing_upstreams_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS revoke_records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    cause TEXT NOT NULL,
    winner_json TEXT,
    decision_json TEXT NOT NULL,
    at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    shard_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_seq INTEGER NOT NULL
);
"""


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _entry_hash(event_type, shard_id, payload) -> str:
    material = _canonical({"event_type": event_type, "shard_id": shard_id, "payload": payload})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str, declaration=None, backup_dir: Optional[str] = None):
        self.path = path
        self.declaration = declaration
        self.backup_dir = backup_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "backups")
        self.replaced = False
        self.backup_path: Optional[str] = None
        self.replace_reason: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ open
    def open(self, declaration=None) -> "Ledger":
        if declaration is not None:
            self.declaration = declaration
        existed = os.path.exists(self.path)
        if existed:
            self._verify_or_replace()
        self._conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        if not existed or self.replaced:
            self._initialize_meta()
        self._conn.commit()
        return self

    def checkpoint(self):
        if self._conn is not None:
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self):
        if self._conn is not None:
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _initialize_meta(self):
        decl = self.declaration
        initial = {
            "format_version": str(LEDGER_FORMAT_VERSION),
            "node_set_version": decl.node_set_version(),
            "capacity_versions": _canonical(decl.capacity_version_map()),
            "head_chain_hash": "GENESIS",
            "last_seq": "0",
        }
        for key, value in initial.items():
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _meta(self, key) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def _set_meta(self, key, value):
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    # ------------------------------------------------- snapshot verification
    def expected_identity(self, declaration) -> dict:
        return {
            "node_set_version": declaration.node_set_version(),
            "capacity_versions": _canonical(declaration.capacity_version_map()),
        }

    def _verify_or_replace(self):
        probe = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        probe.row_factory = sqlite3.Row
        try:
            mismatch = self._identity_mismatch(probe)
            if mismatch is None:
                mismatch = self._digest_mismatch(probe)
        finally:
            probe.close()
        if mismatch is not None:
            self._replace_file(mismatch)

    def _identity_mismatch(self, probe) -> Optional[dict]:
        decl = self.declaration
        rows = {row["key"]: row["value"] for row in probe.execute("SELECT key, value FROM meta")}
        fmt = rows.get("format_version")
        if fmt is not None and int(fmt) != LEDGER_FORMAT_VERSION:
            return {"reason": "format_version", "stored": fmt, "expected": LEDGER_FORMAT_VERSION}
        if rows.get("node_set_version") != decl.node_set_version():
            return {"reason": "node_set_version",
                    "stored": rows.get("node_set_version"),
                    "expected": decl.node_set_version()}
        if rows.get("capacity_versions") != _canonical(decl.capacity_version_map()):
            return {"reason": "capacity_versions",
                    "stored": rows.get("capacity_versions"),
                    "expected": _canonical(decl.capacity_version_map())}
        return None

    def _digest_mismatch(self, probe) -> Optional[dict]:
        stored_head = None
        row = probe.execute("SELECT value FROM meta WHERE key='head_chain_hash'").fetchone()
        if row is not None:
            stored_head = row["value"]
        recomputed = self._recompute_chain(probe)
        if stored_head != recomputed:
            return {"reason": "content_digest", "stored": stored_head, "expected": recomputed}
        return None

    @staticmethod
    def _recompute_chain(probe) -> str:
        head = "GENESIS"
        for row in probe.execute(
            "SELECT event_type, shard_id, payload_json, entry_hash FROM events ORDER BY seq"
        ):
            if hashlib.sha256(
                _canonical({
                    "event_type": row["event_type"],
                    "shard_id": row["shard_id"],
                    "payload": json.loads(row["payload_json"]),
                }).encode("utf-8")
            ).hexdigest() != row["entry_hash"]:
                return "CORRUPT"
            head = hashlib.sha256((head + row["entry_hash"]).encode("utf-8")).hexdigest()
        return head

    def _replace_file(self, mismatch: dict):
        os.makedirs(self.backup_dir, exist_ok=True)
        stamp = str(len(os.listdir(self.backup_dir)))
        backup_path = os.path.join(self.backup_dir, f"snapshot-{stamp}.db")
        shutil.copy2(self.path, backup_path)
        os.chmod(backup_path, 0o444)  # read-only preserved backup
        for suffix in ("-wal", "-shm"):
            sidecar = self.path + suffix
            if os.path.exists(sidecar):
                os.remove(sidecar)
        os.remove(self.path)
        self.replaced = True
        self.backup_path = backup_path
        self.replace_reason = mismatch

    # ------------------------------------------------------------- writing
    def _append_event_locked(self, event_type, shard_id, payload, at_ms) -> int:
        entry = _entry_hash(event_type, shard_id, payload)
        prev_head = self._meta("head_chain_hash") or "GENESIS"
        chain = hashlib.sha256((prev_head + entry).encode("utf-8")).hexdigest()
        cur = self._conn.execute(
            "INSERT INTO events(event_type, shard_id, payload_json, entry_hash, chain_hash, at_ms) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (event_type, shard_id, _canonical(payload), entry, chain, at_ms),
        )
        seq = cur.lastrowid
        self._set_meta("head_chain_hash", chain)
        self._set_meta("last_seq", seq)
        return seq

    def record_lease_grant(self, lease, at_ms) -> int:
        payload = lease.to_dict()
        with self._conn:
            seq = self._append_event_locked("lease_grant", lease.shard_id, payload, at_ms)
            self._conn.execute(
                "INSERT INTO lease_records(lease_id, shard_id, node_id, grant_ms, expiry_ms, "
                "version, demand, released, release_reason, updated_seq) "
                "VALUES(?,?,?,?,?,?,?,0,NULL,?) "
                "ON CONFLICT(lease_id) DO UPDATE SET grant_ms=excluded.grant_ms, "
                "expiry_ms=excluded.expiry_ms, version=excluded.version, demand=excluded.demand, "
                "released=0, release_reason=NULL, updated_seq=excluded.updated_seq",
                (lease.lease_id, lease.shard_id, lease.node_id, lease.grant_ms,
                 lease.expiry_ms, lease.version, lease.demand, seq),
            )
            self._conn.execute(
                "INSERT INTO resource_usage(shard_id, node_id, lease_id, demand, delta, reason, at_ms) "
                "VALUES(?,?,?,?,?,?,?)",
                (lease.shard_id, lease.node_id, lease.lease_id, lease.demand,
                 lease.demand, "grant", at_ms),
            )
            self._upsert_progress(lease.shard_id, "DISPATCHED", seq)
        return seq

    def record_lease_release(self, event: dict) -> int:
        with self._conn:
            seq = self._append_event_locked("lease_release", event["shard_id"], event, event["at_ms"])
            self._conn.execute(
                "UPDATE lease_records SET released=1, release_reason=?, updated_seq=? WHERE lease_id=?",
                (event["reason"], seq, event["lease_id"]),
            )
            self._conn.execute(
                "INSERT INTO resource_usage(shard_id, node_id, lease_id, demand, delta, reason, at_ms) "
                "VALUES(?,?,?,?,?,?,?)",
                (event["shard_id"], event["node_id"], event["lease_id"], event["demand"],
                 -event["demand"], event["reason"], event["at_ms"]),
            )
        return seq

    def record_freeze_migration(self, record: dict) -> int:
        shard_id = record["shard_id"]
        payload = dict(record)
        with self._conn:
            seq = self._append_event_locked("freeze_migration", shard_id, payload, record["at_ms"])
            self._conn.execute(
                "INSERT INTO freeze_records(shard_id, from_state, to_state, missing_upstreams_json, reason, at_ms) "
                "VALUES(?,?,?,?,?,?)",
                (shard_id, record["from_state"], record["to_state"],
                 _canonical(record["missing_upstreams"]), record["reason"], record["at_ms"]),
            )
            self._upsert_progress(shard_id, record["to_state"], seq)
        return seq

    def record_revocation(self, decision: dict, at_ms) -> int:
        with self._conn:
            seq = self._append_event_locked("revocation", decision["shard_id"], decision, at_ms)
            self._conn.execute(
                "INSERT INTO revoke_records(shard_id, cause, winner_json, decision_json, at_ms) "
                "VALUES(?,?,?,?,?)",
                (decision["shard_id"], decision["cause"],
                 _canonical(decision["winner"]), _canonical(decision), at_ms),
            )
        return seq

    def record_outcome(self, outcome: dict) -> int:
        """Atomic shard outcome: summary, ownership, usage, freeze view, cursor."""
        shard_id = outcome["shard_id"]
        at_ms = outcome["at_ms"]
        with self._conn:
            seq = self._append_event_locked("shard_outcome", shard_id, outcome, at_ms)
            entry = _entry_hash("shard_outcome", shard_id, outcome)
            self._conn.execute(
                "INSERT INTO shard_outcomes(shard_id, status, node_id, lease_id, lease_version, "
                "capacity_version, summary_json, summary_digest, fixed_total, entry_hash, updated_seq) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(shard_id) DO UPDATE SET status=excluded.status, node_id=excluded.node_id, "
                "lease_id=excluded.lease_id, lease_version=excluded.lease_version, "
                "capacity_version=excluded.capacity_version, summary_json=excluded.summary_json, "
                "summary_digest=excluded.summary_digest, fixed_total=excluded.fixed_total, "
                "entry_hash=excluded.entry_hash, updated_seq=excluded.updated_seq",
                (shard_id, outcome["status"], outcome.get("node_id"), outcome.get("lease_id"),
                 outcome.get("lease_version"), outcome.get("capacity_version"),
                 _canonical(outcome.get("summary")), outcome.get("summary_digest"),
                 outcome.get("fixed_total"), entry, seq),
            )
            self._conn.execute(
                "INSERT INTO resource_usage(shard_id, node_id, lease_id, demand, delta, reason, at_ms) "
                "VALUES(?,?,?,?,?,?,?)",
                (shard_id, outcome.get("node_id"), outcome.get("lease_id"),
                 outcome.get("demand", 0), -int(outcome.get("demand", 0)), "outcome_complete", at_ms),
            )
            self._upsert_progress(shard_id, outcome["status"], seq)
        return seq

    def record_queue_wait(self, wait_record: dict) -> int:
        with self._conn:
            return self._append_event_locked("queue_wait", wait_record["shard_id"], wait_record,
                                             wait_record["enqueue_ms"])

    def _upsert_progress(self, shard_id, state, seq):
        self._conn.execute(
            "INSERT INTO progress(shard_id, state, updated_seq) VALUES(?,?,?) "
            "ON CONFLICT(shard_id) DO UPDATE SET state=excluded.state, updated_seq=excluded.updated_seq",
            (shard_id, state, seq),
        )

    # ------------------------------------------------------------- reading
    def head_chain_hash(self) -> str:
        return self._meta("head_chain_hash")

    def last_seq(self) -> int:
        value = self._meta("last_seq")
        return 0 if value is None else int(value)

    def identity(self) -> dict:
        return {
            "format_version": int(self._meta("format_version")),
            "node_set_version": self._meta("node_set_version"),
            "capacity_versions": json.loads(self._meta("capacity_versions")),
            "head_chain_hash": self.head_chain_hash(),
        }

    def outcomes(self) -> Dict[str, dict]:
        result = {}
        for row in self._conn.execute("SELECT * FROM shard_outcomes"):
            result[row["shard_id"]] = {
                "shard_id": row["shard_id"],
                "status": row["status"],
                "node_id": row["node_id"],
                "lease_id": row["lease_id"],
                "lease_version": row["lease_version"],
                "capacity_version": row["capacity_version"],
                "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
                "summary_digest": row["summary_digest"],
                "fixed_total": row["fixed_total"],
                "entry_hash": row["entry_hash"],
                "updated_seq": row["updated_seq"],
            }
        return result

    def lease_snapshot(self) -> dict:
        leases, versions = [], {}
        for row in self._conn.execute("SELECT * FROM lease_records ORDER BY lease_id"):
            leases.append({
                "lease_id": row["lease_id"], "shard_id": row["shard_id"],
                "node_id": row["node_id"], "grant_ms": row["grant_ms"],
                "expiry_ms": row["expiry_ms"], "version": row["version"],
                "demand": row["demand"], "released": bool(row["released"]),
                "release_reason": row["release_reason"],
            })
            versions[f"{row['node_id']}|{row['shard_id']}"] = row["version"]
        usage = self._conn.execute(
            "SELECT COALESCE(SUM(delta), 0) AS used FROM resource_usage"
        ).fetchone()["used"]
        releases = [
            {"lease_id": r["lease_id"], "shard_id": r["shard_id"], "node_id": r["node_id"],
             "version": r["version"], "demand": r["demand"], "reason": r["release_reason"],
             "at_ms": r["grant_ms"]}
            for r in self._conn.execute(
                "SELECT * FROM lease_records WHERE released=1 ORDER BY updated_seq")
        ]
        return {"leases": leases, "versions": versions,
                "budget_used": int(usage), "release_log": releases}

    def freeze_snapshot(self) -> dict:
        state, since, missing = {}, {}, {}
        history = []
        for row in self._conn.execute(
            "SELECT * FROM freeze_records ORDER BY seq"
        ):
            state[row["shard_id"]] = row["to_state"]
            since[row["shard_id"]] = row["at_ms"]
            missing[row["shard_id"]] = json.loads(row["missing_upstreams_json"])
            history.append({
                "shard_id": row["shard_id"], "from_state": row["from_state"],
                "to_state": row["to_state"],
                "missing_upstreams": json.loads(row["missing_upstreams_json"]),
                "reason": row["reason"], "at_ms": row["at_ms"],
            })
        return {"state": state, "since": since, "missing": missing, "history": history}

    def revoke_records(self) -> List[dict]:
        return [json.loads(r["decision_json"])
                for r in self._conn.execute("SELECT decision_json FROM revoke_records ORDER BY seq")]

    def progress(self) -> Dict[str, dict]:
        return {row["shard_id"]: {"shard_id": row["shard_id"], "state": row["state"],
                                  "updated_seq": row["updated_seq"]}
                for row in self._conn.execute("SELECT * FROM progress")}

    def verify_chain(self) -> bool:
        return self._recompute_chain(self._conn) == self.head_chain_hash()

    def entry_hashes(self) -> Dict[str, str]:
        return {row["shard_id"]: row["entry_hash"]
                for row in self._conn.execute("SELECT shard_id, entry_hash FROM shard_outcomes")}
