"""会话快照: 本地 SQLite 持久化、版本识别与强制终止后的恢复。

持久化内容: 配置版本、脚本内容摘要、缩放因子口径、已封存窗口集合、
当前窗口部分观测、完整批次记录与逐项事件轨迹, 并附整体内容摘要。

恢复规则:
  * 只能依据账本中已完整写入的批次记录恢复;
  * 未完成批次标记为未知并要求重跑, 此时不得声称能生成退出报告;
  * 快照版本或摘要不一致时整体替换, 并以 E_SNAPSHOT 报告差异。
"""
import json
import sqlite3

from . import fixedpoint as fp
from .config import config_digest
from .errors import E_SNAPSHOT, BizError
from .ledger import (Batch, Event, Ledger, ObservationItem, Window,
                     canonical, digest_of)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER, kind TEXT, rtt_ms INTEGER,
    clock_ms INTEGER, arrival_order INTEGER);
CREATE TABLE IF NOT EXISTS batches (
    idx INTEGER, event_lo INTEGER, event_hi INTEGER, clock_ms INTEGER,
    config_version INTEGER, complete INTEGER, digest TEXT);
CREATE TABLE IF NOT EXISTS windows (
    idx INTEGER, batch_lo INTEGER, batch_hi INTEGER, sealed INTEGER,
    stale INTEGER, version INTEGER, digest TEXT);
CREATE TABLE IF NOT EXISTS items (
    window_idx INTEGER, name TEXT, value INTEGER, error_bound INTEGER,
    batch_lo INTEGER, batch_hi INTEGER, affected INTEGER, halted INTEGER,
    version INTEGER);
CREATE TABLE IF NOT EXISTS trace (event_seq INTEGER, item TEXT, value INTEGER);
CREATE TABLE IF NOT EXISTS anomalies (seq INTEGER, code TEXT, rendered TEXT);
"""

DATA_TABLES = ("events", "batches", "windows", "items", "trace", "anomalies")


class SnapshotStore:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- 元信息 --------------------------------------------------------
    def load_meta(self):
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {k: v for k, v in rows}

    def negotiate_version(self, config, script):
        """版本识别: 配置或脚本变化时版本号单调递增。

        返回 (config_version, changed, diff_details)。
        """
        meta = self.load_meta()
        want = {
            "config_digest": config_digest(config),
            "script_digest": script.digest,
            "scale_factor": str(config.scale_factor),
        }
        if not meta:
            return 1, False, {}
        diff = {k: {"stored": meta.get(k), "current": v}
                for k, v in sorted(want.items()) if meta.get(k) != v}
        if not diff:
            return int(meta["config_version"]), False, {}
        return int(meta["config_version"]) + 1, True, diff

    # ---- 写入 ----------------------------------------------------------
    def save_session(self, ledger, events, finished, script_digest="",
                     stale_archive=()):
        """持久化会话; 只写入已完整批次的覆盖事件, 未完成批次不落盘。"""
        cur = self.conn.cursor()
        complete_events = len(ledger.batches) * ledger.config.batch_size
        events = list(events)[:complete_events]
        with self.conn:
            for table in DATA_TABLES:
                cur.execute("DELETE FROM %s" % table)
            cur.execute("DELETE FROM meta")
            meta = {
                "config_version": str(ledger.config_version),
                "config_digest": config_digest(ledger.config),
                "script_digest": script_digest,
                "scale_factor": str(ledger.config.scale_factor),
                "session_finished": "1" if finished else "0",
                "stale_archive": json.dumps(list(stale_archive)),
            }
            for key, value in meta.items():
                cur.execute("INSERT INTO meta VALUES (?, ?)", (key, value))
            for event in events:
                cur.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                            (event.event_seq, event.kind, event.rtt_ms,
                             event.clock_ms, event.arrival_order))
            for batch in ledger.batches:
                cur.execute("INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (batch.index, batch.event_lo, batch.event_hi,
                             batch.clock_ms, batch.config_version,
                             1 if batch.complete else 0, batch.digest))
            for window in ledger.all_windows():
                cur.execute("INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (window.index, window.batch_lo, window.batch_hi,
                             1 if window.sealed else 0, 1 if window.stale else 0,
                             window.version, window.digest))
                for name in sorted(window.items):
                    item = window.items[name]
                    cur.execute(
                        "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (window.index, item.name, item.value, item.error_bound,
                         item.batch_lo, item.batch_hi, 1 if item.affected else 0,
                         1 if item.halted else 0, item.version))
            for event_seq, item, value in ledger.trace:
                cur.execute("INSERT INTO trace VALUES (?, ?, ?)",
                            (event_seq, item, value))
            for seq, anomaly in enumerate(ledger.anomalies):
                cur.execute("INSERT INTO anomalies VALUES (?, ?, ?)",
                            (seq, anomaly.code, anomaly.render()))
            digest = self._content_digest(cur)
            cur.execute("INSERT INTO meta VALUES ('content_digest', ?)", (digest,))
        return digest

    @staticmethod
    def _content_digest(cur):
        dump = {}
        for table in DATA_TABLES:
            rows = cur.execute(
                "SELECT * FROM %s ORDER BY rowid" % table).fetchall()
            dump[table] = [list(row) for row in rows]
        meta = {k: v for k, v in cur.execute(
            "SELECT key, value FROM meta WHERE key != 'content_digest'")}
        dump["meta"] = sorted(meta.items())
        return digest_of(dump)

    # ---- 校验 ----------------------------------------------------------
    def verify(self):
        """内容摘要校验; 不一致时返回 E_SNAPSHOT 业务异常, 一致返回 None。"""
        meta = self.load_meta()
        stored = meta.get("content_digest")
        if stored is None:
            return None
        actual = self._content_digest(self.conn.cursor())
        if actual != stored:
            return BizError(E_SNAPSHOT, "snapshot content digest mismatch",
                            stored=stored, actual=actual)
        return None

    # ---- 恢复 ----------------------------------------------------------
    def load_events(self):
        rows = self.conn.execute(
            "SELECT event_seq, kind, rtt_ms, clock_ms, arrival_order "
            "FROM events ORDER BY rowid").fetchall()
        return [Event(event_seq=r[0], kind=r[1], rtt_ms=r[2],
                      clock_ms=r[3], arrival_order=r[4]) for r in rows]

    def load_windows(self):
        """按窗口序号稳定顺序加载窗口及其观测项。"""
        windows = []
        for row in self.conn.execute(
                "SELECT idx, batch_lo, batch_hi, sealed, stale, version, digest "
                "FROM windows ORDER BY idx"):
            window = Window(index=row[0], batch_lo=row[1], batch_hi=row[2],
                            sealed=bool(row[3]), stale=bool(row[4]),
                            version=row[5], digest=row[6])
            for item_row in self.conn.execute(
                    "SELECT name, value, error_bound, batch_lo, batch_hi, "
                    "affected, halted, version FROM items "
                    "WHERE window_idx = ? ORDER BY name", (row[0],)):
                window.items[item_row[0]] = ObservationItem(
                    name=item_row[0], value=item_row[1],
                    error_bound=item_row[2], batch_lo=item_row[3],
                    batch_hi=item_row[4], affected=bool(item_row[5]),
                    halted=bool(item_row[6]), version=item_row[7])
            windows.append(window)
        return windows

    def load_trace(self):
        return [(r[0], r[1], r[2]) for r in self.conn.execute(
            "SELECT event_seq, item, value FROM trace ORDER BY rowid")]

    def recover(self, config, script):
        """强制终止后的恢复。

        返回 RecoveryResult; 快照摘要不一致时整体替换并抛 E_SNAPSHOT。
        """
        problem = self.verify()
        if problem is not None:
            with self.conn:
                for table in DATA_TABLES:
                    self.conn.execute("DELETE FROM %s" % table)
                self.conn.execute("DELETE FROM meta")
            raise problem
        meta = self.load_meta()
        if not meta:
            return RecoveryResult(ledger=None, unknown_batches=[],
                                  recovered_events=0)
        events = self.load_events()
        ledger = Ledger(config, int(meta["config_version"]))
        for event in events:
            ledger.ingest_event(event)
        # 未完成批次: 已持久化事件覆盖的批之外, 脚本剩余部分标记为未知
        total_batches = -(-len(script.entries) // config.batch_size)
        done_batches = len(ledger.batches)
        finished = meta.get("session_finished") == "1"
        unknown = [] if finished else list(range(done_batches, total_batches))
        return RecoveryResult(ledger=ledger, unknown_batches=unknown,
                              recovered_events=len(events))


class RecoveryResult:
    def __init__(self, ledger, unknown_batches, recovered_events):
        self.ledger = ledger
        self.unknown_batches = unknown_batches
        self.recovered_events = recovered_events

    @property
    def exit_report_available(self):
        """存在未知批次时不得声称能生成退出报告。"""
        return not self.unknown_batches and self.ledger is not None


def diff_windows(old_windows, new_windows):
    """分窗口整体替换的稳定差异清单(新增/删除/版本变化)。"""
    old = {w.index: w for w in old_windows}
    new = {w.index: w for w in new_windows}
    added = [{"window": i, "items": sorted(new[i].items)}
             for i in sorted(new) if i not in old]
    removed = [{"window": i, "items": sorted(old[i].items)}
               for i in sorted(old) if i not in new]
    changed = [{"window": i, "items": sorted(new[i].items),
                "old_version": old[i].version, "new_version": new[i].version}
               for i in sorted(new) if i in old and old[i].version != new[i].version]
    return {"added": added, "removed": removed, "version_changed": changed}
