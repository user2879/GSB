"""SQLite persistence for the hit ledger, rebuilds and session snapshots.

Each committed row carries a content digest over its logical payload. A batch
of hits, its merge inputs and the progress cursor are committed in one
transaction, so crash recovery can only observe fully committed batches.
"""

import json
import os
import sqlite3

from .errors import ReplayError, StateError
from .hashing import content_digest

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hits (
    ledger_seq INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    sub_batch INTEGER NOT NULL,
    split_reason TEXT,
    rule_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    client_key TEXT NOT NULL,
    destination TEXT NOT NULL,
    bucket INTEGER NOT NULL,
    weight_bp INTEGER NOT NULL,
    backend_id TEXT NOT NULL,
    backend_seq INTEGER NOT NULL,
    param_digest TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    destination TEXT NOT NULL,
    seq_from INTEGER NOT NULL,
    seq_to INTEGER NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS param_versions (
    backend_id TEXT NOT NULL,
    backend_seq INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    quant_version INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    model_dim INTEGER NOT NULL,
    hidden_dim INTEGER NOT NULL,
    param_digest TEXT NOT NULL,
    PRIMARY KEY (backend_id, backend_seq)
);
CREATE TABLE IF NOT EXISTS rebuilds (
    rebuild_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ledger_upper_bound INTEGER NOT NULL,
    status TEXT NOT NULL,
    incoming_during_rebuild INTEGER NOT NULL DEFAULT 0,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    rebuild_id TEXT NOT NULL,
    ledger_seq_from INTEGER NOT NULL,
    ledger_seq_to INTEGER NOT NULL,
    param_versions TEXT NOT NULL,
    body TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_intent (
    batch_id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    destination TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class LedgerStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = __import__("threading").RLock()
        self._initialize()

    def _initialize(self):
        with self._lock:
            self.conn.executescript(SCHEMA)
            stored = self._get_meta("format_version")
            if stored is None:
                self._put_meta("format_version", str(FORMAT_VERSION))
            elif int(stored) != FORMAT_VERSION:
                raise StateError(
                    f"unsupported format version {stored}, want {FORMAT_VERSION}",
                    field="meta.format_version",
                )

    def _get_meta(self, key):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _put_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def close(self):
        with self._lock:
            self.conn.close()

    def register_param_version(self, record):
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO param_versions(backend_id, backend_seq, "
                "model_version, quant_version, seed, model_dim, hidden_dim, "
                "param_digest) VALUES(?,?,?,?,?,?,?,?)",
                (record["backend_id"], record["backend_seq"],
                 record["model_version"], record["quant_version"],
                 record["seed"], record["model_dim"], record["hidden_dim"],
                 record["param_digest"]),
            )
            self.conn.commit()

    def param_versions(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM param_versions ORDER BY backend_id, backend_seq"
            ).fetchall()
            return [dict(r) for r in rows]

    def param_digest(self, backend_id, backend_seq):
        with self._lock:
            row = self.conn.execute(
                "SELECT param_digest FROM param_versions WHERE backend_id=? "
                "AND backend_seq=?", (backend_id, backend_seq)
            ).fetchone()
            return row["param_digest"] if row else None

    def mark_batch_intent(self, batch_id, rule_id, destination):
        from datetime import datetime, timezone

        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO batch_intent(batch_id, rule_id, "
                "destination, created_at) VALUES(?,?,?,?)",
                (batch_id, rule_id, destination,
                 datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()

    def commit_batch(self, batch_id, rule_id, destination, hits, split_reasons):
        """Atomically commit hits, merge inputs and the progress cursor."""
        if not hits:
            return
        from datetime import datetime, timezone

        with self._lock, self.conn:
            next_seq_row = self.conn.execute(
                "SELECT COALESCE(MAX(ledger_seq), 0) + 1 AS next FROM hits"
            ).fetchone()
            next_seq = next_seq_row["next"]
            seq_from = next_seq
            for offset, hit in enumerate(hits):
                payload = {
                    "batch_id": batch_id,
                    "sub_batch": hit["sub_batch"],
                    "split_reason": split_reasons.get(hit["sub_batch"]),
                    "rule_id": hit["rule_id"],
                    "sample_id": hit["sample_id"],
                    "client_key": hit["client_key"],
                    "destination": hit["destination"],
                    "bucket": hit["bucket"],
                    "weight_bp": hit["weight_bp"],
                    "backend_id": hit["backend_id"],
                    "backend_seq": hit["backend_seq"],
                    "param_digest": hit["param_digest"],
                    "output_digest": hit["output_digest"],
                    "input_digest": hit["input_digest"],
                    "elapsed_ms": hit["elapsed_ms"],
                }
                self.conn.execute(
                    "INSERT INTO hits(ledger_seq, batch_id, sub_batch, "
                    "split_reason, rule_id, sample_id, client_key, destination, "
                    "bucket, weight_bp, backend_id, backend_seq, param_digest, "
                    "output_digest, input_digest, elapsed_ms, content_digest) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (next_seq + offset, batch_id, payload["sub_batch"],
                     payload["split_reason"], payload["rule_id"],
                     payload["sample_id"], payload["client_key"],
                     payload["destination"], payload["bucket"],
                     payload["weight_bp"], payload["backend_id"],
                     payload["backend_seq"], payload["param_digest"],
                     payload["output_digest"], payload["input_digest"],
                     payload["elapsed_ms"], content_digest(payload)),
                )
            seq_to = next_seq + len(hits) - 1
            batch_payload = {
                "batch_id": batch_id, "rule_id": rule_id,
                "destination": destination, "seq_from": seq_from,
                "seq_to": seq_to, "status": "committed",
            }
            self.conn.execute(
                "INSERT INTO batches(batch_id, rule_id, destination, seq_from, "
                "seq_to, status, content_digest) VALUES(?,?,?,?,?,?,?)",
                (batch_id, rule_id, destination, seq_from, seq_to,
                 "committed", content_digest(batch_payload)),
            )
            self.conn.execute("DELETE FROM batch_intent WHERE batch_id=?",
                              (batch_id,))
            self._put_meta("progress_cursor", str(seq_to))
            self._put_meta("progress_updated_at",
                           datetime.now(timezone.utc).isoformat())

    def max_ledger_seq(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(ledger_seq), 0) AS m FROM hits"
            ).fetchone()
            return row["m"]

    def count_hits(self):
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) c FROM hits").fetchone()["c"]

    def hits_in_range(self, seq_from, seq_to):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM hits WHERE ledger_seq BETWEEN ? AND ? "
                "ORDER BY ledger_seq", (seq_from, seq_to)
            ).fetchall()
            return [dict(r) for r in rows]

    def all_hits(self):
        return self.hits_in_range(1, self.max_ledger_seq())

    def pending_intents(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM batch_intent ORDER BY rowid"
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_intent_unknown(self, intent):
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO batches(batch_id, rule_id, destination, "
                "seq_from, seq_to, status, content_digest) VALUES(?,?,?,?,?,?,?)",
                (intent["batch_id"], intent["rule_id"], intent["destination"],
                 -1, -1, "unknown_rerun_required",
                 content_digest({"batch_id": intent["batch_id"],
                                 "status": "unknown_rerun_required"})),
            )
            self.conn.execute("DELETE FROM batch_intent WHERE batch_id=?",
                              (intent["batch_id"],))
            self.conn.commit()

    def unknown_batches(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM batches WHERE status='unknown_rerun_required'"
            ).fetchall()
            return [dict(r) for r in rows]

    def verify_hit_digests(self):
        """Raise on the first row whose stored content digest does not match."""
        with self._lock:
            rows = self.conn.execute("SELECT * FROM hits ORDER BY ledger_seq")
            for row in rows:
                data = dict(row)
                stored = data.pop("content_digest")
                data.pop("ledger_seq", None)
                if content_digest(data) != stored:
                    raise ReplayError(
                        f"ledger content digest mismatch at ledger_seq="
                        f"{row['ledger_seq']}",
                        field=f"hits[{row['ledger_seq']}].content_digest",
                    )
        return True

    def start_rebuild(self, rebuild_id, upper_bound):
        from datetime import datetime, timezone

        with self._lock:
            payload = {"rebuild_id": rebuild_id,
                       "ledger_upper_bound": upper_bound,
                       "status": "running",
                       "incoming_during_rebuild": 0}
            self.conn.execute(
                "INSERT INTO rebuilds(rebuild_id, started_at, ledger_upper_bound, "
                "status, incoming_during_rebuild, content_digest) "
                "VALUES(?,?,?,?,?,?)",
                (rebuild_id, datetime.now(timezone.utc).isoformat(),
                 upper_bound, "running", 0, content_digest(payload)),
            )
            self.conn.commit()

    def finish_rebuild(self, rebuild_id, incoming):
        with self._lock:
            upper = self.conn.execute(
                "SELECT ledger_upper_bound FROM rebuilds WHERE rebuild_id=?",
                (rebuild_id,),
            ).fetchone()["ledger_upper_bound"]
            payload = {"rebuild_id": rebuild_id,
                       "ledger_upper_bound": upper, "status": "done",
                       "incoming_during_rebuild": incoming}
            self.conn.execute(
                "UPDATE rebuilds SET status=?, incoming_during_rebuild=?, "
                "content_digest=? WHERE rebuild_id=?",
                ("done", incoming, content_digest(payload), rebuild_id),
            )
            self.conn.commit()

    def rebuilds(self):
        with self._lock:
            rows = self.conn.execute("SELECT * FROM rebuilds ORDER BY rowid")
            return [dict(r) for r in rows]

    def save_snapshot(self, snapshot):
        with self._lock:
            existing = self.conn.execute(
                "SELECT snapshot_id FROM snapshots WHERE snapshot_id=?",
                (snapshot["snapshot_id"],),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "DELETE FROM snapshots WHERE snapshot_id=?",
                    (snapshot["snapshot_id"],),
                )
            self.conn.execute(
                "INSERT INTO snapshots(snapshot_id, rebuild_id, ledger_seq_from, "
                "ledger_seq_to, param_versions, body, digest, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (snapshot["snapshot_id"], snapshot["rebuild_id"],
                 snapshot["ledger_seq_from"], snapshot["ledger_seq_to"],
                 json.dumps(snapshot["param_versions"], sort_keys=True),
                 json.dumps(snapshot["body"], sort_keys=True,
                            separators=(",", ":")),
                 snapshot["digest"], snapshot["created_at"]),
            )
            self.conn.commit()

    def load_latest_snapshot(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM snapshots ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            snapshot = dict(row)
            snapshot["param_versions"] = json.loads(snapshot["param_versions"])
            snapshot["body"] = json.loads(snapshot["body"])
            return snapshot

    def replace_all_snapshots(self, snapshot):
        """Session snapshots are replaced wholesale, never merged incrementally."""
        with self._lock:
            self.conn.execute("DELETE FROM snapshots")
            self.conn.commit()
        self.save_snapshot(snapshot)
