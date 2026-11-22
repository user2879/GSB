"""SQLite 账本: 迁移记录、指令回执摘要、逐事件状态与进度游标。

- 每个事件批次在同一事务中写入迁移记录、指令回执摘要、逐事件状态
  与进度游标; 账本携带整数格式版本与内容摘要。
- 强制终止后重启只依据已完整提交的记录恢复; 任何处于 "running"
  状态的指令(未随批次完整提交)在恢复时标记为 unknown 并要求重跑。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .errors import DiagError
from .util import canonical, digest_obj

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    entity TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    mono_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS commands(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cmd_seq INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    instance TEXT NOT NULL,
    domain TEXT NOT NULL,
    action TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    enqueue_mono_ms INTEGER NOT NULL,
    start_mono_ms INTEGER NOT NULL,
    end_mono_ms INTEGER NOT NULL,
    receipt_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS event_states(
    event_seq INTEGER PRIMARY KEY,
    instance_states TEXT NOT NULL,
    domain_states TEXT NOT NULL,
    expected TEXT NOT NULL,
    receipts_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursor(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    event_seq INTEGER NOT NULL,
    event_ts_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS domain_marks(
    domain TEXT PRIMARY KEY,
    isolated_event_ts INTEGER NOT NULL
);
"""

DIGEST_TABLES = ("transitions", "commands", "event_states", "cursor", "domain_marks")


class Ledger:
    def __init__(self, path, versions: dict[str, str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        if fresh:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),))
            for key, value in sorted(versions.items()):
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)", (key, value))
            self.conn.execute(
                "INSERT INTO cursor(id, event_seq, event_ts_ms) VALUES(1, 0, 0)")
            self.conn.commit()
        else:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='format_version'").fetchone()
            stored = int(row[0]) if row else -1
            if stored != FORMAT_VERSION:
                self.conn.close()
                raise DiagError("E_SPEC",
                                f"ledger format version {stored} != {FORMAT_VERSION}",
                                {"field": "format_version", "line": 0})
            for key, value in sorted(versions.items()):
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    (key, value))
                self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- 元信息 ----
    def meta(self) -> dict[str, str]:
        return {k: v for k, v in self.conn.execute("SELECT key, value FROM meta")}

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def cursor(self) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT event_seq, event_ts_ms FROM cursor WHERE id=1").fetchone()
        return (row[0], row[1]) if row else (0, 0)

    # ---- 批次事务 ----
    @contextmanager
    def batch(self):
        """同一事务写入一个事件批次的全部记录并推进游标。"""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self
            self.set_meta("digest", self._compute_digest())
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def add_transition(self, tr: dict) -> None:
        self.conn.execute(
            "INSERT INTO transitions(event_seq, kind, entity, from_state, to_state,"
            " reason, mono_ms) VALUES(?,?,?,?,?,?,?)",
            (tr["event_seq"], tr["kind"], tr["entity"], tr["from"], tr["to"],
             tr["reason"], tr["mono_ms"]))

    def add_command(self, res) -> None:
        import json
        self.conn.execute(
            "INSERT INTO commands(cmd_seq, event_seq, instance, domain, action,"
            " attempt, status, enqueue_mono_ms, start_mono_ms, end_mono_ms,"
            " receipt_json, error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (res.cmd_seq, res.event_seq, res.instance, res.domain, res.action,
             res.attempt, res.status, res.enqueue_mono_ms, res.start_mono_ms,
             res.end_mono_ms,
             json.dumps(res.receipt, sort_keys=True) if res.receipt else None,
             res.error))

    def add_event_state(self, event_seq: int, instance_states: dict,
                        domain_states: dict, expected: dict,
                        receipts_digest: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO event_states(event_seq, instance_states,"
            " domain_states, expected, receipts_digest) VALUES(?,?,?,?,?)",
            (event_seq, canonical(instance_states), canonical(domain_states),
             canonical(expected), receipts_digest))

    def set_cursor(self, event_seq: int, event_ts_ms: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cursor(id, event_seq, event_ts_ms)"
            " VALUES(1, ?, ?)", (event_seq, event_ts_ms))

    def set_domain_mark(self, domain: str, isolated_event_ts: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO domain_marks(domain, isolated_event_ts)"
            " VALUES(?, ?)", (domain, isolated_event_ts))

    # ---- 恢复与查询 ----
    def recover(self) -> list[dict]:
        """强制终止后的恢复: 未完整提交的指令标记为 unknown 并要求重跑。"""
        rows = self.conn.execute(
            "SELECT id, instance, cmd_seq FROM commands WHERE status='running'"
        ).fetchall()
        self.conn.execute(
            "UPDATE commands SET status='unknown',"
            " error='interrupted before batch commit' WHERE status='running'")
        self.conn.commit()
        return [{"id": r[0], "instance": r[1], "cmd_seq": r[2]} for r in rows]

    def restore_state(self) -> dict:
        """依据已提交迁移记录重建实例健康与域隔离状态。"""
        health: dict[str, str] = {}
        domain_state: dict[str, str] = {}
        for kind, entity, to_state in self.conn.execute(
                "SELECT kind, entity, to_state FROM transitions ORDER BY id"):
            if kind == "instance":
                health[entity] = to_state
            else:
                domain_state[entity] = to_state
        marks = {d: ts for d, ts in self.conn.execute(
            "SELECT domain, isolated_event_ts FROM domain_marks")}
        return {"health": health, "domain_state": domain_state,
                "isolated_since_ts": marks}

    def transitions(self) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        return self.conn.execute(
            "SELECT * FROM transitions ORDER BY id").fetchall()

    def commands(self, event_seq: int | None = None) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        if event_seq is None:
            return self.conn.execute("SELECT * FROM commands ORDER BY id").fetchall()
        return self.conn.execute(
            "SELECT * FROM commands WHERE event_seq=? ORDER BY id",
            (event_seq,)).fetchall()

    def event_state(self, event_seq: int) -> sqlite3.Row | None:
        self.conn.row_factory = sqlite3.Row
        return self.conn.execute(
            "SELECT * FROM event_states WHERE event_seq=?", (event_seq,)).fetchone()

    def event_states_range(self, up_to_seq: int) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        return self.conn.execute(
            "SELECT * FROM event_states WHERE event_seq<=? ORDER BY event_seq",
            (up_to_seq,)).fetchall()

    def max_cmd_seq(self) -> int:
        row = self.conn.execute("SELECT MAX(cmd_seq) FROM commands").fetchone()
        return row[0] or 0

    def _compute_digest(self) -> str:
        payload = {}
        for table in DIGEST_TABLES:
            cur = self.conn.execute(f"SELECT * FROM {table} ORDER BY 1")
            payload[table] = [[col for col in row] for row in cur.fetchall()]
        return digest_obj(payload)

    def digest(self) -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='digest'").fetchone()
        return row[0] if row else ""
