"""持久化账本（SQLite 事件溯源）。

设计要点：

* 每条事件写入独立事务，只有完整提交的事件在重启后可见（强制终止时
  未提交事务随进程消失，对应任务按“未知”处理并要求重跑）。
* 事件携带单调 seq 与链式内容摘要 prev_digest -> digest，格式版本为
  整数 LEDGER_FORMAT_VERSION。
* 打开账本时识别格式版本并校验内容摘要：不一致则整体替换状态，旧账本
  复制为只读备份，绝不在损坏账本上做增量合并。
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time

from .errors import LedgerError

LEDGER_FORMAT_VERSION = 3


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


class Ledger:
    def __init__(self, path):
        self.path = path
        self.backup_path = None
        self.replaced = False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_or_recover()
        self.seq, self.digest = self._head()

    # ---------- 初始化 / 格式识别 ----------
    def _init_or_recover(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'")
        if cur.fetchone() is None:
            self._create_schema()
            return
        fmt = self._meta("format_version")
        if fmt is None or int(fmt) != LEDGER_FORMAT_VERSION:
            self._replace_because(
                f"格式版本不一致: 账本={fmt} 期望={LEDGER_FORMAT_VERSION}")
            return
        if not self._verify_chain():
            self._replace_because("内容摘要链校验失败")

    def _create_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
            CREATE TABLE events (
                seq INTEGER PRIMARY KEY,
                ts_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                prev_digest TEXT NOT NULL
            );
            CREATE TABLE cursors (name TEXT PRIMARY KEY, v TEXT NOT NULL);
            """)
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),))
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES('created_ms', ?)",
            (str(int(time.monotonic() * 1000)),))

    def _replace_because(self, reason):
        self.conn.close()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{self.path}.bak-{stamp}"
        shutil.copy2(self.path, backup)
        os.chmod(backup, 0o444)
        for suffix in ("-wal", "-shm"):
            side = self.path + suffix
            if os.path.exists(side):
                side_bak = backup + suffix
                shutil.copy2(side, side_bak)
                try:
                    os.chmod(side_bak, 0o444)
                except OSError:
                    pass
        os.remove(self.path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.remove(self.path + suffix)
        self.backup_path = backup
        self.replaced = True
        self.replace_reason = reason
        self.conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _meta(self, key):
        cur = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _head(self):
        cur = self.conn.execute(
            "SELECT seq, digest FROM events ORDER BY seq DESC LIMIT 1")
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0, "0" * 64)

    def _verify_chain(self):
        prev = "0" * 64
        for seq, payload, digest, old_prev in self.conn.execute(
                "SELECT seq, payload, digest, prev_digest FROM events ORDER BY seq"):
            if old_prev != prev:
                return False
            expect = hashlib.sha256(
                (prev + _canonical({"seq": seq, "payload": json.loads(payload)}))
                .encode("utf-8")).hexdigest()
            if expect != digest:
                return False
            prev = digest
        return True

    # ---------- 写入 ----------
    def append(self, kind, payload):
        """完整提交一条事件；返回事件头 (seq, digest)。"""
        self.seq += 1
        body = _canonical({"seq": self.seq, "payload": payload})
        digest = hashlib.sha256((self.digest + body).encode("utf-8")).hexdigest()
        self.conn.execute(
            "INSERT INTO events(seq,ts_ms,kind,payload,digest,prev_digest)"
            " VALUES(?,?,?,?,?,?)",
            (self.seq, int(time.monotonic() * 1000), kind,
             json.dumps(payload, sort_keys=True, ensure_ascii=False),
             digest, self.digest))
        self.digest = digest
        return self.seq, digest

    def set_cursor(self, name, value):
        self.conn.execute(
            "INSERT INTO cursors(name,v) VALUES(?,?) "
            "ON CONFLICT(name) DO UPDATE SET v=excluded.v",
            (name, json.dumps(value, sort_keys=True)))

    def get_cursor(self, name, default=None):
        cur = self.conn.execute("SELECT v FROM cursors WHERE name=?", (name,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else default

    # ---------- 读取 / 恢复 ----------
    def events(self, kind=None):
        if kind is None:
            cur = self.conn.execute(
                "SELECT seq,kind,payload FROM events ORDER BY seq")
        else:
            cur = self.conn.execute(
                "SELECT seq,kind,payload FROM events WHERE kind=? ORDER BY seq",
                (kind,))
        return [(seq, k, json.loads(p)) for seq, k, p in cur]

    def snapshot_state(self):
        """按事件顺序折叠出当前状态，供重启恢复。"""
        state = {
            "manifests": {},       # version -> manifest 快照
            "compilations": {},    # version -> {artifacts, propagation}
            "task_outcomes": {},   # task_id -> 最新结局
            "receipts": {},        # task_id -> 回执摘要
            "latency": [],         # 全部延迟观测（不剔除）
            "cursors": {},
        }
        for _, kind, payload in self.events():
            if kind == "manifest_snapshot":
                state["manifests"][payload["version"]] = payload
            elif kind == "compile":
                state["compilations"][payload["version"]] = payload
            elif kind == "task_outcome":
                state["task_outcomes"][payload["task_id"]] = payload
            elif kind == "receipt":
                state["receipts"][payload["task_id"]] = payload
            elif kind == "latency":
                state["latency"].append(payload)
        for name, value in self.conn.execute("SELECT name,v FROM cursors"):
            state["cursors"][name] = json.loads(value)
        return state

    def close(self):
        self.conn.close()

    def assert_intact(self):
        if not self._verify_chain():
            raise LedgerError("账本内容摘要校验失败", field="events",
                              basis="链式 digest 必须逐条匹配")
