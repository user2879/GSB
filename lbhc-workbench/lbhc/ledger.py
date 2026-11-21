"""SQLite ledger: probes, transitions, instance view, progress cursor.

Durability rules:
  * a probe result, its state transitions, the instance view update, the
    progress cursor and the hash-chain events commit in ONE transaction;
  * an ``inflight`` row is committed BEFORE the probe starts and removed in
    the same transaction as its result, so a killed process always leaves a
    recoverable, unambiguous trace;
  * a sha256 hash chain over events lets us detect any out-of-band ledger
    tampering (E_LEDGER).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

from .errors import LedgerError

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS probes (
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    round INTEGER NOT NULL,
    path TEXT NOT NULL,
    pool_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    enqueued_ms REAL NOT NULL,
    started_ms REAL NOT NULL,
    finished_ms REAL NOT NULL,
    wait_ms REAL NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY (version, seq)
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    probe_seq INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason_category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instance_view (
    version INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    pool_id TEXT NOT NULL,
    state TEXT NOT NULL,
    consec_success INTEGER NOT NULL,
    consec_failure INTEGER NOT NULL,
    last_reason TEXT NOT NULL,
    last_probe_seq INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (version, instance_id)
);
CREATE TABLE IF NOT EXISTS inflight (
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    instance_id TEXT NOT NULL,
    PRIMARY KEY (version, seq)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    ref_seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    prev_digest TEXT NOT NULL,
    digest TEXT NOT NULL
);
"""

GENESIS = "0" * 64


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class Ledger:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        row = self._db.execute(
            "SELECT digest FROM events ORDER BY id DESC LIMIT 1").fetchone()
        self._last_digest = row[0] if row else GENESIS

    def close(self):
        with self._lock:
            self._db.close()

    # -- meta ------------------------------------------------------------
    def get_meta(self, key, default=None):
        row = self._db.execute("SELECT value FROM meta WHERE key=?",
                               (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        with self._lock:
            self._db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))
            self._db.commit()

    def current_version(self):
        v = self.get_meta("spec_version")
        return int(v) if v is not None else None

    # -- hash chain --------------------------------------------------------
    def _append_event(self, cur, version, kind, ref_seq, payload):
        body = _canonical(payload)
        digest = hashlib.sha256(
            (self._last_digest + "|" + kind + "|" + body).encode()
        ).hexdigest()
        cur.execute(
            "INSERT INTO events(version,kind,ref_seq,payload,prev_digest,"
            "digest) VALUES(?,?,?,?,?,?)",
            (version, kind, ref_seq, body, self._last_digest, digest))
        self._last_digest = digest
        return digest

    def verify_chain(self):
        """Recompute the hash chain; any tampering raises E_LEDGER."""
        prev = GENESIS
        for row in self._db.execute(
                "SELECT id,kind,payload,prev_digest,digest FROM events "
                "ORDER BY id"):
            eid, kind, payload, prev_digest, digest = row
            if prev_digest != prev:
                raise LedgerError(
                    "hash chain link broken: prev_digest mismatch",
                    path=f"events[{eid}]",
                    detail={"expected_prev": prev, "found": prev_digest})
            expect = hashlib.sha256(
                (prev + "|" + kind + "|" + payload).encode()).hexdigest()
            if expect != digest:
                raise LedgerError(
                    "hash chain digest mismatch: ledger content was "
                    "modified outside the workbench",
                    path=f"events[{eid}]",
                    detail={"expected": expect, "found": digest})
            prev = digest
        return True

    # -- inflight (crash recovery) -----------------------------------------
    def add_inflight(self, version, seq, instance_id):
        with self._lock:
            self._db.execute(
                "INSERT INTO inflight(version,seq,instance_id) "
                "VALUES(?,?,?)", (version, seq, instance_id))
            self._db.commit()

    def pending_inflight(self, version):
        return [tuple(r) for r in self._db.execute(
            "SELECT version,seq,instance_id FROM inflight WHERE version=? "
            "ORDER BY seq", (version,))]

    def void_inflight(self, version):
        """Mark leftover (uncommitted) probes as unknown/void, in one tx."""
        pending = self.pending_inflight(version)
        if not pending:
            return []
        with self._lock:
            cur = self._db.cursor()
            for _, seq, iid in pending:
                self._append_event(cur, version, "probe_void", seq, {
                    "seq": seq, "instance_id": iid,
                    "reason": "uncommitted_probe_voided_on_recovery"})
            cur.execute("DELETE FROM inflight WHERE version=?", (version,))
            self._db.commit()
        return pending

    # -- probe commit --------------------------------------------------------
    def record_probe(self, rec, transitions, view_row):
        """Commit one probe result + transitions + view + cursor + events.

        ``rec``        : dict with probe fields
        ``transitions``: list of (from_state, to_state, reason_category)
        ``view_row``   : dict with the instance's fresh view values
        Everything lands in a single transaction; a crash mid-way leaves
        no partial state behind.
        """
        with self._lock:
            cur = self._db.cursor()
            cur.execute(
                "DELETE FROM inflight WHERE version=? AND seq=?",
                (rec["version"], rec["seq"]))
            cur.execute(
                "INSERT INTO probes(version,seq,round,path,pool_id,"
                "instance_id,enqueued_ms,started_ms,finished_ms,wait_ms,"
                "outcome,reason,detail) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["version"], rec["seq"], rec["round"], rec["path"],
                 rec["pool_id"], rec["instance_id"], rec["enqueued_ms"],
                 rec["started_ms"], rec["finished_ms"], rec["wait_ms"],
                 rec["outcome"], rec["reason"], rec["detail"]))
            for frm, to, cat in transitions:
                cur.execute(
                    "INSERT INTO transitions(version,probe_seq,instance_id,"
                    "from_state,to_state,reason_category) "
                    "VALUES(?,?,?,?,?,?)",
                    (rec["version"], rec["seq"], rec["instance_id"],
                     frm, to, cat))
                self._append_event(cur, rec["version"], "transition",
                                   rec["seq"], {
                                       "instance_id": rec["instance_id"],
                                       "probe_seq": rec["seq"],
                                       "from": frm, "to": to,
                                       "reason_category": cat})
            cur.execute(
                "INSERT INTO instance_view(version,instance_id,pool_id,"
                "state,consec_success,consec_failure,last_reason,"
                "last_probe_seq,stale) VALUES(?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(version,instance_id) DO UPDATE SET "
                "state=excluded.state,consec_success=excluded.consec_success,"
                "consec_failure=excluded.consec_failure,"
                "last_reason=excluded.last_reason,"
                "last_probe_seq=excluded.last_probe_seq,stale=0",
                (rec["version"], rec["instance_id"], rec["pool_id"],
                 view_row["state"], view_row["consec_success"],
                 view_row["consec_failure"], view_row["last_reason"],
                 rec["seq"]))
            cur.execute(
                "INSERT INTO meta(key,value) VALUES('cursor',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(rec["seq"]),))
            self._append_event(cur, rec["version"], "probe", rec["seq"], {
                "seq": rec["seq"], "round": rec["round"],
                "path": rec["path"], "pool_id": rec["pool_id"],
                "instance_id": rec["instance_id"],
                "outcome": rec["outcome"], "reason": rec["reason"]})
            self._db.commit()

    # -- views ----------------------------------------------------------------
    def init_view(self, version, spec):
        """Wholesale replace the current view for a new spec version."""
        with self._lock:
            cur = self._db.cursor()
            cur.execute("UPDATE instance_view SET stale=1 WHERE version<>?",
                        (version,))
            for pool in spec.pools:
                for b in pool.backends:
                    state = "draining" if b.role == "draining" else "unknown"
                    reason = ("declared_draining" if b.role == "draining"
                              else "")
                    cur.execute(
                        "INSERT INTO instance_view(version,instance_id,"
                        "pool_id,state,consec_success,consec_failure,"
                        "last_reason,last_probe_seq,stale) "
                        "VALUES(?,?,?,?,0,0,?,-1,0) "
                        "ON CONFLICT(version,instance_id) DO UPDATE SET "
                        "pool_id=excluded.pool_id,state=excluded.state,"
                        "consec_success=0,consec_failure=0,"
                        "last_reason=excluded.last_reason,"
                        "last_probe_seq=-1,stale=0",
                        (version, b.id, pool.id, state, reason))
            self._append_event(cur, version, "view_replace", -1, {
                "version": version,
                "instances": [b.id for p in spec.pools for b in p.backends],
            })
            self._db.commit()

    def view(self, version, include_stale=False):
        sql = ("SELECT instance_id,pool_id,state,consec_success,"
               "consec_failure,last_reason,last_probe_seq,stale "
               "FROM instance_view WHERE version=?")
        if not include_stale:
            sql += " AND stale=0"
        sql += " ORDER BY instance_id"
        return [dict(zip(
            ("instance_id", "pool_id", "state", "consec_success",
             "consec_failure", "last_reason", "last_probe_seq", "stale"),
            r)) for r in self._db.execute(sql, (version,))]

    def probes(self, version):
        cols = ("version", "seq", "round", "path", "pool_id", "instance_id",
                "enqueued_ms", "started_ms", "finished_ms", "wait_ms",
                "outcome", "reason", "detail")
        return [dict(zip(cols, r)) for r in self._db.execute(
            "SELECT version,seq,round,path,pool_id,instance_id,enqueued_ms,"
            "started_ms,finished_ms,wait_ms,outcome,reason,detail "
            "FROM probes WHERE version=? ORDER BY seq", (version,))]

    def transitions(self, version):
        cols = ("id", "version", "probe_seq", "instance_id", "from_state",
                "to_state", "reason_category")
        return [dict(zip(cols, r)) for r in self._db.execute(
            "SELECT id,version,probe_seq,instance_id,from_state,to_state,"
            "reason_category FROM transitions WHERE version=? "
            "ORDER BY probe_seq, id", (version,))]

    def max_round(self, version):
        row = self._db.execute(
            "SELECT MAX(round) FROM probes WHERE version=?",
            (version,)).fetchone()
        return row[0] if row and row[0] is not None else -1

    def next_seq(self, version):
        row = self._db.execute(
            "SELECT MAX(seq) FROM probes WHERE version=?",
            (version,)).fetchone()
        return (row[0] + 1) if row and row[0] is not None else 0

    def stale_versions(self):
        return [r[0] for r in self._db.execute(
            "SELECT DISTINCT version FROM instance_view WHERE stale=1 "
            "ORDER BY version")]

    def event_count(self):
        return self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
