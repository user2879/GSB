"""SQLite 归属账本: 同一账本混存 r1/r2/r3 三种格式."""

import json
import sqlite3

from .canon import canonical_json, digest_of
from . import records as records_mod

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    request_id TEXT PRIMARY KEY,
    fmt TEXT NOT NULL,
    payload TEXT NOT NULL,
    seq INTEGER NOT NULL
)
"""


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def reset(self):
        self.conn.execute("DELETE FROM records")
        self.conn.commit()

    def insert(self, request_id, fmt, payload_dict):
        cur = self.conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM records")
        seq = cur.fetchone()[0]
        self.conn.execute(
            "INSERT INTO records(request_id, fmt, payload, seq) VALUES (?,?,?,?)",
            (request_id, fmt, canonical_json(payload_dict), seq),
        )
        self.conn.commit()

    def read_all(self, fmt=None):
        sql = "SELECT request_id, fmt, payload FROM records"
        params = ()
        if fmt is not None:
            sql += " WHERE fmt = ?"
            params = (fmt,)
        sql += " ORDER BY seq"
        return [
            {"request_id": r[0], "fmt": r[1], "payload": r[2]}
            for r in self.conn.execute(sql, params)
        ]

    def decode_all(self, supported_slots, fmt=None):
        out = []
        for row in self.read_all(fmt=fmt):
            out.append(records_mod.decode(json.loads(row["payload"]), supported_slots))
        return out

    def rewrite(self, request_id, fmt, payload_dict):
        with self.conn:
            self.conn.execute(
                "UPDATE records SET fmt = ?, payload = ? WHERE request_id = ?",
                (fmt, canonical_json(payload_dict), request_id),
            )

    def formats(self):
        rows = self.conn.execute("SELECT DISTINCT fmt FROM records").fetchall()
        return sorted(r[0] for r in rows)

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def content_digest(self):
        rows = self.conn.execute(
            "SELECT request_id, fmt, payload FROM records ORDER BY request_id"
        ).fetchall()
        return digest_of([list(r) for r in rows])

    def fmt_of(self, request_id):
        row = self.conn.execute(
            "SELECT fmt FROM records WHERE request_id = ?", (request_id,)
        ).fetchone()
        return row[0] if row else None
