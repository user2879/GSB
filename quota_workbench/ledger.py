"""诊断会话账本：SQLite 持久化、版本识别、整体替换恢复与重放校验。"""

import json
import sqlite3

from .errors import WorkbenchError
from .models import canonical, digest

LEDGER_FORMAT_VERSION = 1
DOMAINS = ("arbitration", "migration", "attribution", "events")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conclusions (
    domain TEXT NOT NULL,
    volume_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (domain, volume_id, metric, payload)
);
CREATE TABLE IF NOT EXISTS cursor (
    batch_seq INTEGER PRIMARY KEY,
    volume_id TEXT NOT NULL,
    batch_digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def read_meta(self):
        return {k: v for k, v in self.conn.execute("SELECT key, value FROM meta")}

    def _write_meta(self, conn, updates):
        for k, v in updates.items():
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                         (k, str(v)))

    def check_snapshot(self, sources_digest, rule_version):
        """版本或摘要不一致时以 E_SNAPSHOT 拒绝，调用方须整体替换恢复。"""
        meta = self.read_meta()
        if not meta:
            return "empty"
        mismatches = []
        if int(meta.get("format_version", -1)) != LEDGER_FORMAT_VERSION:
            mismatches.append({"field": "format_version",
                               "ledger": meta.get("format_version"),
                               "expected": LEDGER_FORMAT_VERSION})
        if meta.get("sources_digest") != sources_digest:
            mismatches.append({"field": "sources_digest",
                               "ledger": meta.get("sources_digest"),
                               "expected": sources_digest})
        if meta.get("rule_version") != rule_version:
            mismatches.append({"field": "rule_version",
                               "ledger": meta.get("rule_version"),
                               "expected": rule_version})
        if mismatches:
            raise WorkbenchError("E_SNAPSHOT",
                                 "账本快照版本或内容摘要不一致，须整体替换恢复",
                                 {"mismatches": mismatches})
        return "ok"

    def commit_batch(self, batch_seq, volume_id, rows, meta_updates):
        """同一事务写入该批归一摘要、仲裁/迁移/归因结论与进度游标。"""
        with self.conn:
            for domain, payload in rows:
                self.conn.execute(
                    "INSERT INTO conclusions(domain, volume_id, metric, payload)"
                    " VALUES (?, ?, ?, ?)",
                    (domain, volume_id, payload.get("metric", "-"),
                     canonical(payload)))
            batch_digest = digest([[d, p] for d, p in rows])
            self.conn.execute(
                "INSERT INTO cursor(batch_seq, volume_id, batch_digest)"
                " VALUES (?, ?, ?)", (batch_seq, volume_id, batch_digest))
            meta = dict(meta_updates)
            meta["format_version"] = LEDGER_FORMAT_VERSION
            self._write_meta(self.conn, meta)

    def full_replace(self, all_rows, meta_updates):
        """以整体替换方式恢复结论状态：清空后一次性重建，而非逐条修补。"""
        with self.conn:
            self.conn.execute("DELETE FROM conclusions")
            self.conn.execute("DELETE FROM cursor")
            for domain, volume_id, payload in all_rows:
                self.conn.execute(
                    "INSERT INTO conclusions(domain, volume_id, metric, payload)"
                    " VALUES (?, ?, ?, ?)",
                    (domain, volume_id, payload.get("metric", "-"),
                     canonical(payload)))
            meta = dict(meta_updates)
            meta["format_version"] = LEDGER_FORMAT_VERSION
            self._write_meta(self.conn, meta)

    def all_rows(self):
        cur = self.conn.execute(
            "SELECT domain, volume_id, metric, payload FROM conclusions"
            " ORDER BY domain, volume_id, metric, payload")
        return cur.fetchall()

    def committed_volumes(self):
        cur = self.conn.execute("SELECT volume_id FROM cursor ORDER BY batch_seq")
        return [r[0] for r in cur.fetchall()]

    def content_digest(self):
        return digest(self.all_rows())

    def replay_verify(self, recomputed_rows):
        """把重算结论与账本逐项比较；首个不一致以 E_REPLAY 报明
        卷、指标与字段依据。recomputed_rows: [(domain, volume_id, payload)]。"""
        ledger_rows = sorted((d, v, p) for d, v, _, p in self.all_rows())
        expected = sorted((d, v, canonical(p)) for d, v, p in recomputed_rows)
        if len(ledger_rows) != len(expected):
            raise WorkbenchError(
                "E_REPLAY",
                f"重放行数不一致：账本 {len(ledger_rows)}，重算 {len(expected)}",
                {"volume_id": None, "metric": None, "field": "row_count"})
        for (ld, lv, lp), (ed, ev, ep) in zip(ledger_rows, expected):
            if (ld, lv) != (ed, ev):
                raise WorkbenchError(
                    "E_REPLAY", "重放结论域不一致",
                    {"volume_id": lv, "metric": None, "field": "domain",
                     "ledger": [ld, lv], "expected": [ed, ev]})
            if lp != ep:
                lobj = json.loads(lp)
                eobj = json.loads(ep)
                field = next(
                    (k for k in sorted(set(lobj) | set(eobj))
                     if lobj.get(k) != eobj.get(k)), None)
                raise WorkbenchError(
                    "E_REPLAY",
                    f"重放差异定位于卷 {ev} 指标 {eobj.get('metric')} 字段 {field}",
                    {"volume_id": ev, "metric": eobj.get("metric"),
                     "field": field,
                     "ledger_value": lobj.get(field),
                     "expected_value": eobj.get(field)})
        return True
