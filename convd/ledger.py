"""SQLite convergence ledger with snapshot versioning and wholesale replace.

Each completed round is committed in a single SQLite transaction covering
trajectory entries, conclusions, anomalies and the progress cursor.
The ledger digest is computed over logical fields only (versions, state
digests, conclusions, anomalies); raw wall-clock timings are persisted
alongside but excluded so identical inputs always yield byte-identical
digests regardless of executor startup order or host timing.
"""

import json
import os
import sqlite3

from . import LEDGER_FORMAT_VERSION
from .errors import ConvError
from .scene import canonical_json, content_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_no INTEGER NOT NULL,
    executor_id TEXT NOT NULL,
    logical TEXT NOT NULL,
    timing TEXT NOT NULL,
    PRIMARY KEY (round_no, executor_id)
);
CREATE TABLE IF NOT EXISTS anomalies (
    round_no INTEGER NOT NULL,
    executor_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retries (
    round_no INTEGER NOT NULL,
    executor_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    payload TEXT NOT NULL
);
"""


class SnapshotMismatch(ConvError):
    code = "E_SNAPSHOT"

    def __init__(self, diffs):
        super().__init__("snapshot version/digest mismatch; wholesale "
                         "replace applied", diffs=diffs)
        self.diffs = diffs


class Ledger:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- meta ------------------------------------------------------------
    def set_meta(self, **pairs):
        with self.conn:
            for key, value in pairs.items():
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)))

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def cursor(self):
        return self.get_meta("cursor", 0)

    # -- round commit ------------------------------------------------------
    def commit_round(self, round_no, entries, anomalies, retries):
        """Atomically persist one full round and advance the cursor."""
        with self.conn:
            for entry in entries:
                self.conn.execute(
                    "INSERT OR REPLACE INTO rounds"
                    "(round_no, executor_id, logical, timing)"
                    " VALUES(?, ?, ?, ?)",
                    (round_no, entry["executor_id"],
                     canonical_json(entry["logical"]),
                     canonical_json(entry["timing"])))
            for anomaly in anomalies:
                self.conn.execute(
                    "INSERT INTO anomalies(round_no, executor_id, kind,"
                    " payload) VALUES(?, ?, ?, ?)",
                    (round_no, anomaly["executor_id"], anomaly["kind"],
                     canonical_json(anomaly["payload"])))
            for retry in retries:
                self.conn.execute(
                    "INSERT INTO retries(round_no, executor_id, attempt,"
                    " payload) VALUES(?, ?, ?, ?)",
                    (round_no, retry["executor_id"], retry["attempt"],
                     canonical_json(retry["payload"])))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('cursor', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(round_no),))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('ledger_digest', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(self.content_digest()),))

    # -- reads -------------------------------------------------------------
    def round_entries(self):
        rows = self.conn.execute(
            "SELECT round_no, executor_id, logical FROM rounds"
            " ORDER BY round_no, executor_id").fetchall()
        return [(r, e, json.loads(payload)) for r, e, payload in rows]

    def anomaly_entries(self):
        rows = self.conn.execute(
            "SELECT round_no, executor_id, kind, payload FROM anomalies"
            " ORDER BY round_no, executor_id, kind").fetchall()
        return [(r, e, k, json.loads(p)) for r, e, k, p in rows]

    def retry_entries(self):
        rows = self.conn.execute(
            "SELECT round_no, executor_id, attempt, payload FROM retries"
            " ORDER BY round_no, executor_id, attempt").fetchall()
        return [(r, e, a, json.loads(p)) for r, e, a, p in rows]

    def content_digest(self):
        """Digest over committed logical content (stable across runs)."""
        material = {
            "format_version": LEDGER_FORMAT_VERSION,
            "rounds": [entry for _, _, entry in self.round_entries()],
            "anomalies": [
                {"round": r, "executor_id": e, "kind": k, "payload": p}
                for r, e, k, p in self.anomaly_entries()
            ],
        }
        return content_digest(material)

    def summary(self):
        """Per-executor convergence counts and per-round trajectories."""
        per_executor = {}
        for round_no, executor_id, entry in self.round_entries():
            slot = per_executor.setdefault(executor_id, {
                "converged_rounds": 0,
                "not_converged": [],
                "trajectory": [],
            })
            if entry["conclusion"] == "converged":
                slot["converged_rounds"] += 1
            else:
                slot["not_converged"].append(
                    {"round": round_no, "reason": entry["reason"]})
            slot["trajectory"].append({
                "round": round_no,
                "version_before": entry["version_before"],
                "version_after": entry["version_after"],
                "digest_after": entry["digest_after"],
                "in_lease": entry["in_lease"],
                "conclusion": entry["conclusion"],
            })
        return per_executor


# -- session snapshot --------------------------------------------------------

SNAPSHOT_FIELDS = ("scenario_version", "budget_version",
                   "executor_set_version", "ledger_digest")


def snapshot_path_for(db_path):
    return db_path + ".snapshot.json"


def backup_path_for(db_path):
    return db_path + ".snapshot.backup.json"


def write_snapshot(db_path, scenario, conclusions):
    payload = {
        "format_version": LEDGER_FORMAT_VERSION,
        "scenario_version": scenario.digest,
        "budget_version": scenario.budget_version,
        "executor_set_version": scenario.executor_set_version,
        "round_conclusions": conclusions,
        "ledger_digest": None,  # filled by caller via update_snapshot_digest
    }
    with open(snapshot_path_for(db_path), "w", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))
    return payload


def load_snapshot(db_path):
    path = snapshot_path_for(db_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def reconcile_snapshot(db_path, scenario, conclusions, ledger_digest):
    """Load the snapshot; on version/digest mismatch replace it wholesale.

    Returns (snapshot, diffs). On mismatch the old snapshot is preserved
    as a read-only backup and the diff items are listed in stable order.
    """
    desired = {
        "format_version": LEDGER_FORMAT_VERSION,
        "scenario_version": scenario.digest,
        "budget_version": scenario.budget_version,
        "executor_set_version": scenario.executor_set_version,
        "round_conclusions": conclusions,
        "ledger_digest": ledger_digest,
    }
    old = load_snapshot(db_path)
    if old is None:
        with open(snapshot_path_for(db_path), "w", encoding="utf-8") as fh:
            fh.write(canonical_json(desired))
        return desired, []
    diffs = []
    for field in sorted(desired):
        if old.get(field) != desired[field]:
            diffs.append({"field": field, "old": old.get(field),
                          "new": desired[field]})
    if not diffs:
        return old, []
    backup = backup_path_for(db_path)
    if os.path.exists(backup):
        os.chmod(backup, 0o644)
        os.remove(backup)
    os.replace(snapshot_path_for(db_path), backup)
    os.chmod(backup, 0o444)  # old snapshot kept read-only
    with open(snapshot_path_for(db_path), "w", encoding="utf-8") as fh:
        fh.write(canonical_json(desired))
    return desired, diffs
