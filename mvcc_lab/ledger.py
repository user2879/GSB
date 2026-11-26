"""SQLite 账本：原子提交、版本识别与整体替换恢复（业务要求 7、8）。"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from . import SCHEMA_VERSION
from .digest import canonical, digest
from .errors import LedgerError
from .loader import spec_digests
from .model import Spec

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    committed_at TEXT,
    state TEXT NOT NULL,
    content_digest TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    round_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    read_point_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    summary_digest TEXT NOT NULL,
    PRIMARY KEY (round_id, query_id)
);
CREATE TABLE IF NOT EXISTS reason_chains (
    round_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (round_id, query_id, seq)
);
CREATE TABLE IF NOT EXISTS schedule_traces (
    round_id INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    PRIMARY KEY (round_id, query_id)
);
CREATE TABLE IF NOT EXISTS progress (
    round_id INTEGER PRIMARY KEY,
    cursor_json TEXT NOT NULL,
    unknown_query_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS replacement_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    replaced_at TEXT NOT NULL,
    backup_path TEXT NOT NULL,
    diff_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, spec: Spec):
        self.path = Path(path)
        self.spec = spec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=10
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(_DDL)
        self.round_id = 0

    # ------------------------- 版本识别与整体替换 -------------------------

    def identify_or_replace(self) -> dict:
        stored = self._load_meta()
        wanted = spec_digests(self.spec)
        if not stored:
            self._write_identity(wanted)
            return {"action": "initialized", "diff": [], "backup": None}
        diff = []
        for key in ("transactions_digest", "chain_digest",
                    "readpoints_digest", "queries_digest", "content_digest",
                    "spec_version"):
            if str(stored.get(key)) != str(wanted.get(key)):
                diff.append({
                    "field": key,
                    "stored": stored.get(key),
                    "current": wanted.get(key),
                })
        schema_ok = int(stored.get("schema_version", -1)) == SCHEMA_VERSION
        if not diff and schema_ok:
            return {"action": "matched", "diff": [], "backup": None}

        # 整体替换而非增量合并：保留只读备份与差异列表。
        self.conn.close()
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        backup = self.path.with_name(
            f"{self.path.name}.backup-{timestamp}.readonly.sqlite"
        )
        shutil.copy2(self.path, backup)
        backup.chmod(0o444)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar_backup = backup.with_name(backup.name + suffix)
                shutil.copy2(sidecar, sidecar_backup)
                sidecar_backup.chmod(0o444)
        self.path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=10
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(_DDL)
        self._write_identity(wanted)
        self.conn.execute(
            "INSERT INTO replacement_log(replaced_at, backup_path, diff_json)"
            " VALUES (?, ?, ?)",
            (datetime.now().isoformat(), str(backup), canonical(diff)),
        )
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        return {"action": "replaced", "diff": diff, "backup": str(backup)}

    def _load_meta(self) -> dict:
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {key: value for key, value in rows}

    def _write_identity(self, digests: dict) -> None:
        payload = {"schema_version": str(SCHEMA_VERSION),
                   **{key: str(value) for key, value in digests.items()}}
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            self.conn.execute("DELETE FROM meta;")
            self.conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                list(payload.items()),
            )
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

    # ------------------------------- 轮次 --------------------------------

    def start_round(self, unknown_query_ids) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) + 1 FROM rounds"
        ).fetchone()
        self.round_id = int(row[0])
        self.conn.execute("BEGIN IMMEDIATE;")
        self.conn.execute(
            "INSERT INTO rounds(round_id, started_at, state)"
            " VALUES (?, ?, 'open')",
            (self.round_id, datetime.now().isoformat()),
        )
        self.conn.execute(
            "INSERT INTO progress(round_id, cursor_json, unknown_query_ids)"
            " VALUES (?, ?, ?)",
            (self.round_id, canonical({"committed_queries": []}),
             canonical(sorted(unknown_query_ids))),
        )
        # 轮次意图单独提交：强制终止后据此识别未完成轮次。
        self.conn.execute("COMMIT;")
        return self.round_id

    def commit_round(self, summaries: list[dict], traces: list[dict],
                     cursor: dict) -> str:
        """同一事务写入逐查询结论、理由链、调度轨迹与进度游标。"""
        content = digest({
            "summaries": summaries,
            "traces": traces,
            "cursor": cursor,
        })
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            for summary in summaries:
                self.conn.execute(
                    "INSERT INTO decisions(round_id, query_id, read_point_id,"
                    " row_key, summary_json, summary_digest)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (self.round_id, summary["query_id"],
                     summary["read_point_id"], summary["row_key"],
                     canonical(summary), digest(summary)),
                )
                for seq, reason in enumerate(summary["reason_chain"]):
                    self.conn.execute(
                        "INSERT INTO reason_chains(round_id, query_id, seq,"
                        " reason) VALUES (?, ?, ?, ?)",
                        (self.round_id, summary["query_id"], seq, reason),
                    )
            for trace in traces:
                self.conn.execute(
                    "INSERT INTO schedule_traces(round_id, query_id, trace_json)"
                    " VALUES (?, ?, ?)",
                    (self.round_id, trace["query_id"], canonical(trace)),
                )
            self.conn.execute(
                "UPDATE progress SET cursor_json=?, unknown_query_ids=? "
                "WHERE round_id=?",
                (canonical(cursor), canonical([]), self.round_id),
            )
            self.conn.execute(
                "UPDATE rounds SET state='committed', committed_at=?,"
                " content_digest=? WHERE round_id=?",
                (datetime.now().isoformat(), content, self.round_id),
            )
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        return content

    # ------------------------------- 恢复 --------------------------------

    def recover(self) -> dict:
        """只承认完整提交轮次；未提交轮次查询标记为未知并要求重跑。"""
        rounds = self.conn.execute(
            "SELECT round_id, state FROM rounds ORDER BY round_id"
        ).fetchall()
        committed = [rid for rid, state in rounds if state == "committed"]
        open_rounds = [rid for rid, state in rounds if state != "committed"]
        last_committed = max(committed) if committed else None
        unknown: set[str] = set()
        for rid in open_rounds:
            row = self.conn.execute(
                "SELECT unknown_query_ids FROM progress WHERE round_id=?",
                (rid,),
            ).fetchone()
            if row:
                unknown.update(canonical_loads(row[0]))
        decisions = {}
        if last_committed is not None:
            for qid, payload, digest_value in self.conn.execute(
                "SELECT query_id, summary_json, summary_digest FROM decisions"
                " WHERE round_id=? ORDER BY query_id",
                (last_committed,),
            ):
                if digest(canonical_loads(payload)) != digest_value:
                    raise LedgerError(
                        "账本结论摘要自校验失败", object_id=qid,
                        field="summary_digest",
                    )
                decisions[qid] = canonical_loads(payload)
        return {
            "last_committed_round": last_committed,
            "open_rounds": open_rounds,
            "unknown_query_ids": sorted(unknown),
            "committed_decisions": decisions,
        }

    def close(self) -> None:
        self.conn.close()


def canonical_loads(text: str):
    import json
    return json.loads(text)
