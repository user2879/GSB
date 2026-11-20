"""SQLite ledger: the single source of truth.

Guarantees:
  * each finished injection is committed in ONE transaction together with
    its observation summaries, request records, transitions and the
    progress cursor;
  * an injection that becomes active first writes an *intent* row with
    committed=0 (its own transaction); after a crash such rows are the
    only trace and are marked `unknown` on recovery;
  * the ledger carries an integer format version and a content digest
    (sha256 over a canonical dump of all data tables) which is refreshed
    inside every transaction; a mismatch on open means tampering ->
    E_SNAPSHOT.
"""

import hashlib
import json
import sqlite3
import threading

FORMAT_VERSION = 1

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS meta("
    " key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS injections("
    " seq INTEGER PRIMARY KEY, stable_key TEXT NOT NULL,"
    " itype TEXT NOT NULL, target TEXT NOT NULL,"
    " params_json TEXT NOT NULL, planned_start_ms INTEGER NOT NULL,"
    " state TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0,"
    " activated_mono_ms INTEGER, reverted_mono_ms INTEGER,"
    " wait_start_mono_ms INTEGER, wait_end_mono_ms INTEGER,"
    " business_code TEXT, evidence TEXT)",
    "CREATE TABLE IF NOT EXISTS transitions("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, inj_seq INTEGER NOT NULL,"
    " from_state TEXT NOT NULL, to_state TEXT NOT NULL,"
    " reason TEXT NOT NULL, mono_ms INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS requests("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, inj_seq INTEGER NOT NULL,"
    " phase TEXT NOT NULL, req_idx INTEGER NOT NULL,"
    " mono_ms INTEGER NOT NULL, ok INTEGER NOT NULL,"
    " latency_ms INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS observations("
    " inj_seq INTEGER NOT NULL, phase TEXT NOT NULL,"
    " samples INTEGER NOT NULL, success INTEGER NOT NULL,"
    " fail INTEGER NOT NULL, rate_fixed INTEGER NOT NULL,"
    " rate_err_bound INTEGER NOT NULL, p50_fixed INTEGER NOT NULL,"
    " p95_fixed INTEGER NOT NULL, quant_err_bound INTEGER NOT NULL,"
    " scale INTEGER NOT NULL, truncation TEXT NOT NULL,"
    " decidable INTEGER NOT NULL, PRIMARY KEY(inj_seq, phase))",
    "CREATE TABLE IF NOT EXISTS progress("
    " id INTEGER PRIMARY KEY CHECK(id=1), cursor INTEGER NOT NULL)",
]

DIGEST_TABLES = ("injections", "transitions", "requests", "observations", "progress")


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.lock = threading.RLock()
        with self.lock, self.conn:
            for stmt in SCHEMA:
                self.conn.execute(stmt)

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value, cur=None):
        (cur or self.conn).execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))

    def meta_dict(self):
        return {k: v for k, v in self.conn.execute("SELECT key,value FROM meta")}

    def content_digest(self):
        h = hashlib.sha256()
        for table in DIGEST_TABLES:
            cur = self.conn.execute("SELECT * FROM %s" % table)
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                h.update(_canonical([table, cols, list(row)]).encode())
                h.update(b"\n")
        return h.hexdigest()

    def verify_digest(self):
        stored = self.get_meta("content_digest")
        return stored is not None and stored == self.content_digest()

    def transaction(self):
        return _Tx(self)

    def close(self):
        with self.lock:
            self.conn.close()


class _Tx:
    """Exclusive locked transaction; refreshes content digest on commit."""

    def __init__(self, ledger):
        self.ledger = ledger

    def __enter__(self):
        self.ledger.lock.acquire()
        self.cur = self.ledger.conn.cursor()
        self.cur.execute("BEGIN IMMEDIATE")
        return self.cur

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.ledger.conn.commit()
                digest = self.ledger.content_digest()
                with self.ledger.conn:
                    self.ledger.set_meta("content_digest", digest)
            else:
                self.ledger.conn.rollback()
        finally:
            self.cur.close()
            self.ledger.lock.release()
        return False
