"""SQLite persistence for leases, sector versions, batch results,
adjudication traces and session snapshots. Every write commits
immediately so a forced kill only loses incomplete records."""
from __future__ import annotations

import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS leases(
  lease_id TEXT PRIMARY KEY, node_id TEXT, sectors TEXT,
  seq_lo INTEGER, seq_hi INTEGER, grant_mono REAL, expire_mono REAL,
  state TEXT, progress INTEGER);
CREATE TABLE IF NOT EXISTS transitions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT,
  from_state TEXT, to_state TEXT, reason TEXT, progress INTEGER, mono REAL);
CREATE TABLE IF NOT EXISTS sector_versions(
  sector_id TEXT PRIMARY KEY, version INTEGER, owner TEXT);
CREATE TABLE IF NOT EXISTS batch_results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT, node_id TEXT,
  op TEXT, agg TEXT, sectors TEXT, seq_lo INTEGER, seq_hi INTEGER,
  delta REAL, count INTEGER, versions TEXT, value REAL, expected REAL,
  match INTEGER, mono REAL);
CREATE TABLE IF NOT EXISTS adjudications(
  id INTEGER PRIMARY KEY AUTOINCREMENT, req_id TEXT, attempt INTEGER,
  node_id TEXT, sectors TEXT, result TEXT, detail TEXT, mono REAL);
CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY AUTOINCREMENT, wl TEXT, mono REAL,
  leases TEXT, sector_versions TEXT, trace TEXT);
"""


class Store:
    def __init__(self, path):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA synchronous=FULL")
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    def close(self):
        with self._lock:
            self.db.close()

    def get_meta(self, key, default=None):
        with self._lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?",
                                  (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        with self._lock:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))
            self.db.commit()

    def insert_lease(self, lease):
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO leases VALUES(?,?,?,?,?,?,?,?,?)",
                (lease.lease_id, lease.node_id, json.dumps(list(lease.sectors)),
                 lease.seq_lo, lease.seq_hi, lease.grant_mono, lease.expire_mono,
                 lease.state, lease.progress))
            self.db.commit()

    def update_lease(self, lease_id, state, progress):
        with self._lock:
            self.db.execute("UPDATE leases SET state=?, progress=? WHERE lease_id=?",
                            (state, progress, lease_id))
            self.db.commit()

    def load_leases(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT lease_id,node_id,sectors,seq_lo,seq_hi,grant_mono,"
                "expire_mono,state,progress FROM leases ORDER BY rowid").fetchall()
        return [{"lease_id": r[0], "node_id": r[1], "sectors": json.loads(r[2]),
                 "seq_lo": r[3], "seq_hi": r[4], "grant_mono": r[5],
                 "expire_mono": r[6], "state": r[7], "progress": r[8]}
                for r in rows]

    def replace_leases(self, lease_dicts):
        with self._lock:
            self.db.execute("DELETE FROM leases")
            for d in lease_dicts:
                self.db.execute(
                    "INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?)",
                    (d["lease_id"], d["node_id"], json.dumps(list(d["sectors"])),
                     d["seq_lo"], d["seq_hi"], d["grant_mono"], d["expire_mono"],
                     d["state"], d.get("progress", 0)))
            self.db.commit()

    def log_transition(self, lease_id, from_state, to_state, reason, progress, mono):
        with self._lock:
            self.db.execute("INSERT INTO transitions VALUES(NULL,?,?,?,?,?,?)",
                            (lease_id, from_state, to_state, reason, progress, mono))
            self.db.commit()

    def list_transitions(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT lease_id,from_state,to_state,reason,progress,mono "
                "FROM transitions ORDER BY id").fetchall()
        return [{"lease_id": r[0], "from": r[1], "to": r[2], "reason": r[3],
                 "progress": r[4], "mono": r[5]} for r in rows]

    def init_sector_versions(self, sectors, owners):
        with self._lock:
            for sid in sectors:
                self.db.execute(
                    "INSERT OR IGNORE INTO sector_versions VALUES(?,?,?)",
                    (sid, 0, owners[sid]))
            self.db.commit()

    def set_sector_version(self, sector_id, version):
        with self._lock:
            self.db.execute("UPDATE sector_versions SET version=? WHERE sector_id=?",
                            (version, sector_id))
            self.db.commit()

    def get_sector_versions(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT sector_id,version FROM sector_versions").fetchall()
        return {r[0]: r[1] for r in rows}

    def replace_sector_versions(self, versions, owners):
        with self._lock:
            self.db.execute("DELETE FROM sector_versions")
            for sid, ver in versions.items():
                self.db.execute("INSERT INTO sector_versions VALUES(?,?,?)",
                                (sid, ver, owners.get(sid, "")))
            self.db.commit()

    def insert_batch_result(self, rec):
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO batch_results VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["lease_id"], rec["node_id"], rec["op"], rec["agg"],
                 json.dumps(rec["sectors"]), rec["seq_lo"], rec["seq_hi"],
                 rec["delta"], rec["count"], json.dumps(rec["versions"]),
                 rec["value"], rec["expected"], int(rec["match"]), rec["mono"]))
            self.db.commit()
            return cur.lastrowid

    def list_batches(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT id,lease_id,node_id,op,agg,sectors,seq_lo,seq_hi,delta,"
                "count,versions,value,expected,match,mono FROM batch_results "
                "ORDER BY id").fetchall()
        return [{"id": r[0], "lease_id": r[1], "node_id": r[2], "op": r[3],
                 "agg": r[4], "sectors": json.loads(r[5]), "seq_lo": r[6],
                 "seq_hi": r[7], "delta": r[8], "count": r[9],
                 "versions": json.loads(r[10]), "value": r[11],
                 "expected": r[12], "match": bool(r[13]), "mono": r[14]}
                for r in rows]

    def insert_adjudication(self, req_id, attempt, node_id, sectors, result,
                            detail, mono):
        with self._lock:
            self.db.execute("INSERT INTO adjudications VALUES(NULL,?,?,?,?,?,?,?)",
                            (req_id, attempt, node_id, json.dumps(list(sectors)),
                             result, json.dumps(detail), mono))
            self.db.commit()

    def list_adjudications(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT req_id,attempt,node_id,sectors,result,detail,mono "
                "FROM adjudications ORDER BY id").fetchall()
        return [{"req_id": r[0], "attempt": r[1], "node_id": r[2],
                 "sectors": json.loads(r[3]), "result": r[4],
                 "detail": json.loads(r[5]), "mono": r[6]} for r in rows]

    def save_snapshot(self, wl, mono, leases, sector_versions, trace):
        with self._lock:
            self.db.execute("INSERT INTO snapshots VALUES(NULL,?,?,?,?,?)",
                            (wl, mono, json.dumps(leases),
                             json.dumps(sector_versions), json.dumps(trace)))
            self.db.commit()

    def load_snapshot(self):
        with self._lock:
            row = self.db.execute(
                "SELECT wl,mono,leases,sector_versions,trace FROM snapshots "
                "ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"wl": row[0], "mono": row[1], "leases": json.loads(row[2]),
                "sector_versions": json.loads(row[3]), "trace": json.loads(row[4])}
