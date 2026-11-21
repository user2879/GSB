"""SQLite 账本与会话快照持久化。

每个批次在单个事务内完整提交（meta/batches/connections/borrows/
partials/cursor），崩溃时未提交事务自动回滚，恢复只能看到完整批次。
快照额外记录表摘要、槽位参数版本、阈值与账本摘要；版本或摘要不一致时
整体替换，并稳定列出被替换项。
"""

import sqlite3

from . import LEDGER_FORMAT_VERSION, SNAPSHOT_FORMAT_VERSION
from .digest import canonical_json, content_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    committed_ms INTEGER NOT NULL,
    digest TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS connections (
    batch_id INTEGER NOT NULL,
    conn_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS borrows (
    batch_id INTEGER NOT NULL,
    borrower INTEGER NOT NULL,
    lender INTEGER NOT NULL,
    key_cap INTEGER NOT NULL,
    deadline_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS partials (
    batch_id INTEGER NOT NULL,
    conn_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursor (
    batch_id INTEGER PRIMARY KEY,
    last_row_index INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS crash_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    last_committed_batch INTEGER NOT NULL,
    connections_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    table_digest TEXT NOT NULL,
    slot_params_version INTEGER NOT NULL,
    thresholds_json TEXT NOT NULL,
    ledger_digest TEXT NOT NULL,
    replaced_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def is_initialized(self):
        return self.get_meta("format_version") is not None

    def initialize(self, table_digest, slot_params_version, thresholds):
        existing = self.get_meta("format_version")
        replaced = []
        if existing is not None:
            current_table = self.get_meta("table_digest")
            current_version = self.get_meta("slot_params_version")
            current_thresholds = self.get_meta("thresholds")
            mismatch = (
                int(existing) != LEDGER_FORMAT_VERSION
                or current_table != table_digest
                or int(current_version) != slot_params_version
                or current_thresholds != canonical_json(thresholds).decode("utf-8")
            )
            if mismatch:
                replaced = self._replace_all()
            else:
                return replaced
        if existing is not None and int(existing) != LEDGER_FORMAT_VERSION:
            replaced = self._replace_all()
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('table_digest', ?)",
            (table_digest,),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('slot_params_version', ?)",
            (str(slot_params_version),),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('thresholds', ?)",
            (canonical_json(thresholds).decode("utf-8"),),
        )
        self.conn.commit()
        return replaced

    def _replace_all(self):
        tables = ("batches", "connections", "borrows", "partials", "cursor", "snapshot", "crash_events")
        replaced = []
        for table in tables:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count:
                replaced.append({"table": table, "rows_replaced": count})
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
        return sorted(replaced, key=lambda x: x["table"])

    def get_meta(self, key):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def last_batch_id(self):
        row = self.conn.execute("SELECT MAX(batch_id) AS m FROM batches").fetchone()
        return row["m"] if row["m"] is not None else -1

    def append_batch(self, batch):
        bid = batch["batch_id"]
        digest = content_digest(batch["payload"])
        with self.conn:
            self.conn.execute(
                "INSERT INTO batches(batch_id, committed_ms, digest, payload) VALUES(?,?,?,?)",
                (bid, batch["committed_ms"], digest,
                 canonical_json(batch["payload"]).decode("utf-8")),
            )
            for c in batch["connections"]:
                self.conn.execute(
                    "INSERT INTO connections(batch_id, conn_id, payload) VALUES(?,?,?)",
                    (bid, c["conn_id"], canonical_json(c).decode("utf-8")),
                )
            for br in batch["borrows"]:
                self.conn.execute(
                    "INSERT INTO borrows(batch_id, borrower, lender, key_cap, deadline_ms)"
                    " VALUES(?,?,?,?,?)",
                    (bid, br["borrower"], br["lender"], br["key_cap"], br["deadline_ms"]),
                )
            for part in batch["partials"]:
                self.conn.execute(
                    "INSERT INTO partials(batch_id, conn_id, payload) VALUES(?,?,?)",
                    (bid, part["conn_id"], canonical_json(part).decode("utf-8")),
                )
            self.conn.execute(
                "INSERT INTO cursor(batch_id, last_row_index) VALUES(?,?)",
                (bid, batch["last_row_index"]),
            )
        return digest

    def load_batch(self, batch_id):
        row = self.conn.execute(
            "SELECT * FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row is None:
            return None
        import json
        payload = json.loads(row["payload"])
        connections = [
            json.loads(r["payload"])
            for r in self.conn.execute(
                "SELECT payload FROM connections WHERE batch_id=? ORDER BY conn_id",
                (batch_id,),
            )
        ]
        partials = [
            json.loads(r["payload"])
            for r in self.conn.execute(
                "SELECT payload FROM partials WHERE batch_id=? ORDER BY conn_id",
                (batch_id,),
            )
        ]
        borrows = [dict(r) for r in self.conn.execute(
            "SELECT borrower, lender, key_cap, deadline_ms FROM borrows WHERE batch_id=?"
            " ORDER BY borrower", (batch_id,))]
        cursor = self.conn.execute(
            "SELECT last_row_index FROM cursor WHERE batch_id=?", (batch_id,)
        ).fetchone()["last_row_index"]
        return {
            "batch_id": batch_id,
            "committed_ms": row["committed_ms"],
            "digest": row["digest"],
            "payload": payload,
            "connections": connections,
            "partials": partials,
            "borrows": borrows,
            "last_row_index": cursor,
        }

    def all_batch_ids(self):
        return [r[0] for r in self.conn.execute(
            "SELECT batch_id FROM batches ORDER BY batch_id")]

    def write_snapshot(self, table_digest, slot_params_version, thresholds, replaced):
        ledger_digest = self.ledger_digest()
        with self.conn:
            self.conn.execute("DELETE FROM snapshot")
            self.conn.execute(
                "INSERT INTO snapshot(id, format_version, table_digest, slot_params_version,"
                " thresholds_json, ledger_digest, replaced_json) VALUES(1,?,?,?,?,?,?)",
                (SNAPSHOT_FORMAT_VERSION, table_digest, slot_params_version,
                 canonical_json(thresholds).decode("utf-8"), ledger_digest,
                 canonical_json(replaced).decode("utf-8")),
            )

    def read_snapshot(self):
        row = self.conn.execute("SELECT * FROM snapshot WHERE id=1").fetchone()
        if row is None:
            return None
        import json
        return {
            "format_version": row["format_version"],
            "table_digest": row["table_digest"],
            "slot_params_version": row["slot_params_version"],
            "thresholds": json.loads(row["thresholds_json"]),
            "ledger_digest": row["ledger_digest"],
            "replaced": json.loads(row["replaced_json"]),
        }

    def record_crash(self, last_committed_batch, unknown_connections):
        with self.conn:
            self.conn.execute(
                "INSERT INTO crash_events(last_committed_batch, connections_json) VALUES(?,?)",
                (last_committed_batch, canonical_json(unknown_connections).decode("utf-8")),
            )

    def crash_events(self):
        import json
        return [
            {"last_committed_batch": r[0], "connections": json.loads(r[1])}
            for r in self.conn.execute(
                "SELECT last_committed_batch, connections_json FROM crash_events ORDER BY id")
        ]

    def clear_crash_events(self):
        with self.conn:
            self.conn.execute("DELETE FROM crash_events")

    def ledger_digest(self):
        rows = [r[0] for r in self.conn.execute(
            "SELECT digest FROM batches ORDER BY batch_id")]
        return content_digest({"format_version": LEDGER_FORMAT_VERSION, "batch_digests": rows})

    def verify_batch_digests(self):
        import json
        for row in self.conn.execute("SELECT batch_id, digest, payload FROM batches ORDER BY batch_id"):
            actual = content_digest(json.loads(row["payload"]))
            if actual != row["digest"]:
                return row["batch_id"], actual, row["digest"]
        return None
