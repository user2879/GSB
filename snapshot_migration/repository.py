"""SQLite persistence for fragments, ledgers, references and snapshot metadata.

The store is the single durable authority.  Migration work is staged in a
transaction and only the committed generation pointer makes a generation
visible, which is what crash recovery relies on.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional

from .fragments import Fragment, canonical_bytes

SCHEMA_VERSION = 3


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    parent_id TEXT,
    committed INTEGER NOT NULL DEFAULT 0,
    chain_digest TEXT
);
CREATE TABLE IF NOT EXISTS fragments (
    fragment_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    range_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    lower TEXT NOT NULL,
    upper TEXT NOT NULL,
    key_count INTEGER NOT NULL,
    digest TEXT NOT NULL,
    version INTEGER NOT NULL,
    inherited_from TEXT,
    source_fragment_id TEXT,
    keys_blob BLOB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (fragment_id, version)
);
CREATE TABLE IF NOT EXISTS migration_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    generation_id TEXT NOT NULL,
    range_id TEXT NOT NULL,
    disposition TEXT NOT NULL,
    evidence TEXT NOT NULL,
    affected_fragments TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cleanup_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    generation_id TEXT NOT NULL,
    range_id TEXT NOT NULL,
    fragment_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence TEXT NOT NULL,
    UNIQUE (generation_id, range_id, fragment_id)
);
CREATE TABLE IF NOT EXISTS read_refs (
    view_id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL,
    opened_at_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS fragment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    generation_id TEXT NOT NULL,
    range_id TEXT NOT NULL,
    fragment_id TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class Repository:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self):
        with self.transaction():
            self.conn.executescript(SCHEMA)
        current = self.get_meta("schema_version")
        if current is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------------- meta ----------------
    def get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))

    def all_meta(self):
        return {row["key"]: row["value"]
                for row in self.conn.execute("SELECT key, value FROM meta")}

    # ---------------- generations ----------------
    def upsert_generation(self, generation_id, seq, parent_id, chain_digest,
                          committed=False):
        self.conn.execute(
            "INSERT INTO generations(generation_id, seq, parent_id, committed, "
            "chain_digest) VALUES(?,?,?,?,?) "
            "ON CONFLICT(generation_id) DO UPDATE SET seq=excluded.seq, "
            "parent_id=excluded.parent_id, committed=excluded.committed, "
            "chain_digest=excluded.chain_digest",
            (generation_id, seq, parent_id, 1 if committed else 0, chain_digest))

    def mark_committed(self, generation_id):
        self.conn.execute(
            "UPDATE generations SET committed=1 WHERE generation_id=?",
            (generation_id,))

    def committed_generations(self):
        rows = self.conn.execute(
            "SELECT * FROM generations WHERE committed=1 ORDER BY seq").fetchall()
        return [dict(row) for row in rows]

    def is_committed(self, generation_id):
        row = self.conn.execute(
            "SELECT committed FROM generations WHERE generation_id=?",
            (generation_id,)).fetchone()
        return bool(row and row["committed"])

    def generation_row(self, generation_id):
        row = self.conn.execute(
            "SELECT * FROM generations WHERE generation_id=?",
            (generation_id,)).fetchone()
        return dict(row) if row else None

    # ---------------- fragments ----------------
    def insert_fragment(self, frag: Fragment):
        self.conn.execute(
            "INSERT INTO fragments(fragment_id, generation_id, range_id, ordinal, "
            "lower, upper, key_count, digest, version, inherited_from, "
            "source_fragment_id, keys_blob, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (frag.fragment_id, frag.generation_id, frag.range_id, frag.ordinal,
             frag.lower, frag.upper, frag.key_count, frag.digest, frag.version,
             frag.inherited_from, frag.source_fragment_id,
             canonical_bytes(frag.keys), frag.status))

    def supersede_fragment(self, fragment_id, old_version):
        self.conn.execute(
            "UPDATE fragments SET status='superseded' WHERE fragment_id=? AND version=?",
            (fragment_id, old_version))

    def mark_cleaned(self, fragment_id, version):
        self.conn.execute(
            "UPDATE fragments SET status='cleaned' WHERE fragment_id=? AND version=?",
            (fragment_id, version))

    def active_fragments(self, generation_id):
        rows = self.conn.execute(
            "SELECT * FROM fragments WHERE generation_id=? AND status='active' "
            "ORDER BY range_id, ordinal", (generation_id,)).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def fragment_versions(self, fragment_id):
        rows = self.conn.execute(
            "SELECT * FROM fragments WHERE fragment_id=? ORDER BY version",
            (fragment_id,)).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def all_active_fragments(self):
        rows = self.conn.execute(
            "SELECT * FROM fragments WHERE status='active' "
            "ORDER BY generation_id, range_id, ordinal").fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def _row_to_fragment(self, row):
        keys = tuple(json.loads(row["keys_blob"].decode("utf-8")))
        return Fragment(
            fragment_id=row["fragment_id"], generation_id=row["generation_id"],
            range_id=row["range_id"], ordinal=row["ordinal"], lower=row["lower"],
            upper=row["upper"], key_count=row["key_count"], digest=row["digest"],
            version=row["version"], inherited_from=row["inherited_from"],
            source_fragment_id=row["source_fragment_id"], keys=keys,
            status=row["status"])

    # ---------------- ledgers ----------------
    def add_migration_entry(self, seq, generation_id, range_id, disposition,
                            evidence, affected_fragments):
        self.conn.execute(
            "INSERT INTO migration_ledger(seq, generation_id, range_id, "
            "disposition, evidence, affected_fragments) VALUES (?,?,?,?,?,?)",
            (seq, generation_id, range_id, disposition,
             json.dumps(evidence, sort_keys=True),
             json.dumps(affected_fragments, sort_keys=True)))

    def migration_ledger(self, generation_id=None):
        if generation_id:
            rows = self.conn.execute(
                "SELECT * FROM migration_ledger WHERE generation_id=? ORDER BY id",
                (generation_id,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM migration_ledger ORDER BY id").fetchall()
        return [self._ledger_row(row) for row in rows]

    def add_cleanup_entry(self, seq, generation_id, range_id, fragment_id,
                          reason, evidence):
        self.conn.execute(
            "INSERT OR IGNORE INTO cleanup_ledger(seq, generation_id, range_id, "
            "fragment_id, reason, evidence) VALUES (?,?,?,?,?,?)",
            (seq, generation_id, range_id, fragment_id, reason,
             json.dumps(evidence, sort_keys=True)))

    def cleanup_ledger(self):
        rows = self.conn.execute(
            "SELECT * FROM cleanup_ledger ORDER BY id").fetchall()
        return [self._ledger_row(row) for row in rows]

    def add_event(self, seq, generation_id, range_id, fragment_id, event, detail):
        self.conn.execute(
            "INSERT INTO fragment_events(seq, generation_id, range_id, "
            "fragment_id, event, detail) VALUES (?,?,?,?,?,?)",
            (seq, generation_id, range_id, fragment_id, event,
             json.dumps(detail, sort_keys=True)))

    def fragment_events(self, generation_id=None):
        if generation_id:
            rows = self.conn.execute(
                "SELECT * FROM fragment_events WHERE generation_id=? ORDER BY id",
                (generation_id,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM fragment_events ORDER BY id").fetchall()
        return [self._ledger_row(row) for row in rows]

    def _ledger_row(self, row):
        result = dict(row)
        for key in ("evidence", "affected_fragments", "detail"):
            if key in result and isinstance(result[key], str):
                try:
                    result[key] = json.loads(result[key])
                except json.JSONDecodeError:
                    pass
        return result

    # ---------------- read references ----------------
    def open_ref(self, view_id, generation_id, seq):
        self.conn.execute(
            "INSERT INTO read_refs(view_id, generation_id, opened_at_seq) "
            "VALUES (?,?,?)", (view_id, generation_id, seq))

    def close_ref(self, view_id):
        self.conn.execute("DELETE FROM read_refs WHERE view_id=?", (view_id,))

    def active_refs(self):
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM read_refs ORDER BY view_id")]

    def ref_count(self, generation_id):
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM read_refs WHERE generation_id=?",
            (generation_id,)).fetchone()["n"]

    # ---------------- replacement ----------------
    def replace_all(self):
        """Whole-snapshot replacement: wipe derived tables, keep schema meta."""
        for table in ("fragment_events", "cleanup_ledger", "migration_ledger",
                      "read_refs", "fragments", "generations"):
            self.conn.execute(f"DELETE FROM {table}")

    def close(self):
        self.conn.close()
