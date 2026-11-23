"""SQLite checkpoint ledger with strict integrity validation."""

import json
import os
import sqlite3

from .errors import ChainError, IntegrityError, LedgerError
from .hashing import canon_bytes
from .model import content_digest, parse_shard_blob, shard_digest

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    parent_id TEXT,
    step INTEGER NOT NULL,
    shard_count INTEGER NOT NULL,
    partition_mode TEXT NOT NULL,
    optimizer_digest TEXT NOT NULL,
    rng_state TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    ledger_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shards (
    checkpoint_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    tensors TEXT NOT NULL,
    shard_version INTEGER NOT NULL,
    shard_digest TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    PRIMARY KEY (checkpoint_id, shard_id),
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
);
CREATE TABLE IF NOT EXISTS changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    change_set TEXT NOT NULL,
    ledger_version INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, db_path, shard_dir, create=False):
        self.db_path = db_path
        self.shard_dir = shard_dir
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(shard_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        if create:
            self._create_schema()
        elif not self._schema_exists():
            raise LedgerError("ledger database does not exist", [db_path])

    def _create_schema(self):
        self.conn.executescript(DDL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )

    def _schema_exists(self):
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchone()
        return row is not None

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ reads
    def get_meta(self, key):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value))
        )

    def list_checkpoint_ids(self):
        rows = self.conn.execute(
            "SELECT checkpoint_id FROM checkpoints ORDER BY ledger_version"
        ).fetchall()
        return [row["checkpoint_id"] for row in rows]

    def get_checkpoint(self, checkpoint_id):
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["rng_state"] = json.loads(entry["rng_state"])
        entry["rng_state_canon"] = canon_bytes(entry["rng_state"])
        entry["shards"] = self.get_shards(checkpoint_id)
        return entry

    def get_shards(self, checkpoint_id):
        rows = self.conn.execute(
            "SELECT * FROM shards WHERE checkpoint_id=? ORDER BY shard_index",
            (checkpoint_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["tensors"] = json.loads(item["tensors"])
            result.append(item)
        return result

    def get_shard(self, checkpoint_id, shard_id):
        rows = self.get_shards(checkpoint_id)
        for row in rows:
            if row["shard_id"] == shard_id:
                return row
        return None

    def latest_version(self):
        row = self.conn.execute(
            "SELECT MAX(ledger_version) AS v FROM checkpoints"
        ).fetchone()
        return row["v"] if row and row["v"] is not None else 0

    def latest_checkpoint_id(self):
        row = self.conn.execute(
            "SELECT checkpoint_id FROM checkpoints ORDER BY ledger_version DESC LIMIT 1"
        ).fetchone()
        return row["checkpoint_id"] if row else None

    def list_changes(self):
        rows = self.conn.execute(
            "SELECT * FROM changes ORDER BY seq"
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["change_set"] = json.loads(item["change_set"])
            out.append(item)
        return out

    def has_change(self, change_fingerprint):
        row = self.conn.execute(
            "SELECT 1 FROM changes WHERE change_set=? LIMIT 1",
            (json.dumps(change_fingerprint, sort_keys=True),),
        ).fetchone()
        return row is not None

    def find_change_checkpoint(self, change_fingerprint):
        row = self.conn.execute(
            "SELECT checkpoint_id FROM changes WHERE change_set=? LIMIT 1",
            (json.dumps(change_fingerprint, sort_keys=True),),
        ).fetchone()
        return row["checkpoint_id"] if row else None

    # ----------------------------------------------------------------- writes
    def begin(self):
        self.conn.execute("BEGIN IMMEDIATE")

    def commit(self):
        self.conn.execute("COMMIT")

    def rollback(self):
        self.conn.execute("ROLLBACK")

    def insert_checkpoint(self, entry, shards, change_set):
        self.conn.execute(
            """INSERT INTO checkpoints(checkpoint_id,parent_id,step,shard_count,
               partition_mode,optimizer_digest,rng_state,cursor,ledger_version,
               content_digest) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                entry["checkpoint_id"],
                entry["parent_id"],
                entry["step"],
                entry["shard_count"],
                entry["partition_mode"],
                entry["optimizer_digest"],
                json.dumps(entry["rng_state"], sort_keys=True),
                entry["cursor"],
                entry["ledger_version"],
                entry["content_digest"],
            ),
        )
        for shard in shards:
            self.conn.execute(
                """INSERT INTO shards(checkpoint_id,shard_id,shard_index,tensors,
                   shard_version,shard_digest,blob_path) VALUES(?,?,?,?,?,?,?)""",
                (
                    entry["checkpoint_id"],
                    shard["shard_id"],
                    shard["shard_index"],
                    json.dumps(shard["tensors"]),
                    shard["shard_version"],
                    shard["shard_digest"],
                    shard["blob_path"],
                ),
            )
        self.conn.execute(
            """INSERT INTO changes(checkpoint_id,parent_id,change_set,ledger_version)
               VALUES(?,?,?,?)""",
            (
                entry["checkpoint_id"],
                entry["parent_id"],
                json.dumps(change_set, sort_keys=True),
                entry["ledger_version"],
            ),
        )


    def insert_restored(self, entry_scalar, shards, change_row):
        """Insert a checkpoint and its original change-trace row verbatim."""
        self.conn.execute(
            """INSERT INTO checkpoints(checkpoint_id,parent_id,step,shard_count,
               partition_mode,optimizer_digest,rng_state,cursor,ledger_version,
               content_digest) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                entry_scalar["checkpoint_id"],
                entry_scalar["parent_id"],
                entry_scalar["step"],
                entry_scalar["shard_count"],
                entry_scalar["partition_mode"],
                entry_scalar["optimizer_digest"],
                json.dumps(entry_scalar["rng_state"], sort_keys=True),
                entry_scalar["cursor"],
                entry_scalar["ledger_version"],
                entry_scalar["content_digest"],
            ),
        )
        for shard in shards:
            self.conn.execute(
                """INSERT INTO shards(checkpoint_id,shard_id,shard_index,tensors,
                   shard_version,shard_digest,blob_path) VALUES(?,?,?,?,?,?,?)""",
                (
                    entry_scalar["checkpoint_id"],
                    shard["shard_id"],
                    shard["shard_index"],
                    json.dumps(shard["tensors"]),
                    shard["shard_version"],
                    shard["shard_digest"],
                    shard["blob_path"],
                ),
            )
        self.conn.execute(
            """INSERT INTO changes(seq,checkpoint_id,parent_id,change_set,ledger_version)
               VALUES(?,?,?,?,?)""",
            (
                change_row["seq"],
                change_row["checkpoint_id"],
                change_row["parent_id"],
                json.dumps(change_row["change_set"], sort_keys=True),
                change_row["ledger_version"],
            ),
        )

    # ------------------------------------------------------------- validation
    def validate(self):
        """Full structural + digest validation. Raises E_LEDGER/E_CHAIN/E_INTEGRITY."""
        self._validate_structure()
        self._validate_chain()
        self._validate_digests()

    def _validate_structure(self):
        violations = []
        ids = self.list_checkpoint_ids()
        if len(ids) != len(set(ids)):
            seen, duplicates = set(), []
            for cid in ids:
                if cid in seen:
                    duplicates.append(cid)
                seen.add(cid)
            violations.append(f"duplicate checkpoint_id: {duplicates}")

        versions = []
        for cid in ids:
            entry = self.get_checkpoint(cid)
            versions.append((cid, entry["ledger_version"]))
            shards = entry["shards"]
            if len(shards) != entry["shard_count"]:
                violations.append(
                    f"{cid}: shard_count={entry['shard_count']} but rows={len(shards)}"
                )
            digest_set = [shard["shard_digest"] for shard in shards]
            if len(digest_set) != len(set(digest_set)) and shards:
                violations.append(f"{cid}: shard digest set not unique")
            if entry["partition_mode"] not in ("layer", "unit"):
                violations.append(f"{cid}: unknown partition_mode")

        ordered = [version for _cid, version in versions]
        for index in range(1, len(ordered)):
            if ordered[index] <= ordered[index - 1]:
                violations.append(
                    f"ledger_version not strictly increasing at position {index}: "
                    f"{ordered[index - 1]} -> {ordered[index]}"
                )
        if violations:
            raise LedgerError("ledger structure validation failed", violations)

    def _validate_chain(self):
        violations = []
        ids = set(self.list_checkpoint_ids())
        for cid in ids:
            entry = self.get_checkpoint(cid)
            parent = entry["parent_id"]
            if parent is not None and parent not in ids:
                violations.append(
                    f"{cid}: parent_id {parent!r} does not exist (dangling reference)"
                )
        # cycle detection via parent links
        for cid in ids:
            seen, cursor = set(), cid
            while cursor is not None:
                if cursor in seen:
                    violations.append(f"parent cycle detected involving {cursor}")
                    break
                seen.add(cursor)
                entry = self.get_checkpoint(cursor)
                if entry is None:
                    break
                cursor = entry["parent_id"]
        if violations:
            raise ChainError("checkpoint parent chain invalid", violations)

    def _validate_digests(self):
        violations = []
        for cid in self.list_checkpoint_ids():
            entry = self.get_checkpoint(cid)
            scalar = {
                key: (canon_bytes(value) if key == "rng_state" else value)
                for key, value in entry.items()
                if key not in ("shards", "content_digest", "rng_state_canon")
            }
            expected = content_digest(scalar)
            if expected != entry["content_digest"]:
                violations.append(
                    f"{cid}: content_digest mismatch (stored={entry['content_digest'][:12]}, "
                    f"recomputed={expected[:12]})"
                )
            for shard in entry["shards"]:
                if not os.path.exists(shard["blob_path"]):
                    violations.append(
                        f"{cid}/{shard['shard_id']}: blob missing {shard['blob_path']}"
                    )
                    continue
                with open(shard["blob_path"], "rb") as handle:
                    tensors, momenta = parse_shard_blob(handle.read())
                state = {"tensors": tensors, "momenta": momenta}
                recomputed = shard_digest(shard["tensors"], state)
                if recomputed != shard["shard_digest"]:
                    violations.append(
                        f"{cid}/{shard['shard_id']}: shard_digest mismatch "
                        f"(stored={shard['shard_digest'][:12]} recomputed={recomputed[:12]})"
                    )
        if violations:
            raise IntegrityError("content digest or blob integrity failed", violations)

    # ---------------------------------------------------------- fault injection
    def corrupt_shard_digest(self, checkpoint_id, shard_id):
        self.conn.execute(
            "UPDATE shards SET shard_digest=? WHERE checkpoint_id=? AND shard_id=?",
            ("0" * 64, checkpoint_id, shard_id),
        )

    def corrupt_parent(self, checkpoint_id):
        self.conn.execute(
            "UPDATE checkpoints SET parent_id=? WHERE checkpoint_id=?",
            ("ghost_parent", checkpoint_id),
        )

    def corrupt_content_digest(self, checkpoint_id):
        self.conn.execute(
            "UPDATE checkpoints SET content_digest=? WHERE checkpoint_id=?",
            ("f" * 64, checkpoint_id),
        )
