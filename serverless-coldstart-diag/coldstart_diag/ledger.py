"""SQLite 调度账本：每分钟一个事务，携带格式版本与内容摘要。

强制终止语义：只有 COMMIT 成功的分钟才会出现在 ledger.minutes；
重启恢复时按分钟连续性判定未决（计算中未提交）分钟，未决分钟标记为 UNKNOWN 并要求重跑。
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from . import LEDGER_FORMAT_VERSION
from .engine import aggregate
from .spec import validate_manifest

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ledger_format_version INTEGER NOT NULL,
    declaration_version INTEGER NOT NULL,
    scale INTEGER NOT NULL,
    manifest_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    expected_minutes INTEGER NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS minutes (
    run_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    declaration_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (run_id, minute)
);
CREATE TABLE IF NOT EXISTS stale_minutes (
    run_id TEXT NOT NULL,
    declaration_version INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def canonical_digest(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def start_run(self, run_id, manifest, expected_minutes):
        validate_manifest(manifest)
        with self.conn:
            seq = self.conn.execute("SELECT COALESCE(MAX(created_seq),0)+1 FROM runs").fetchone()[0]
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    LEDGER_FORMAT_VERSION,
                    manifest["declaration_version"],
                    manifest["scale"],
                    canonical_digest(manifest),
                    "RUNNING",
                    expected_minutes,
                    seq,
                ),
            )

    def append_minute(self, run_id, result):
        payload = result.to_dict()
        digest = canonical_digest(payload)
        # 单分钟单事务：强制终止时要么整分钟可见，要么完全不可见
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO minutes VALUES (?,?,?,?,?)",
                (run_id, result.minute, result.declaration_version,
                 json.dumps(payload, sort_keys=True, separators=(",", ":")), digest),
            )

    def finalize_run(self, run_id, status="COMPLETE"):
        with self.conn:
            self.conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    def mark_stale(self, run_id, declaration_version):
        """把某声明版本下的分钟整体移入 stale_minutes，当前视图不再可见。"""
        with self.conn:
            rows = self.conn.execute(
                "SELECT run_id, minute, declaration_version, payload_json FROM minutes "
                "WHERE declaration_version=? ORDER BY run_id, minute", (declaration_version,)
            ).fetchall()
            for row in rows:
                self.conn.execute(
                    "INSERT INTO stale_minutes VALUES (?,?,?,?)",
                    (row["run_id"], row["declaration_version"], row["minute"], row["payload_json"]),
                )
            self.conn.execute(
                "DELETE FROM minutes WHERE declaration_version=?", (declaration_version,)
            )
        return len(rows)

    def load_minutes(self, run_id):
        rows = self.conn.execute(
            "SELECT payload_json, digest FROM minutes WHERE run_id=? ORDER BY minute",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if canonical_digest(payload) != row["digest"]:
                raise ValueError(f"分钟内容摘要不一致: run={run_id}")
            out.append(payload)
        return out

    def get_run(self, run_id):
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self):
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_seq").fetchall()
        return [dict(r) for r in rows]

    def recover(self, run_id):
        """依据已完整提交的分钟恢复，未决（缺失/不连续）分钟标记 UNKNOWN 要求重跑。"""
        run = self.get_run(run_id)
        if run is None:
            return None
        committed = [
            row[0]
            for row in self.conn.execute(
                "SELECT minute FROM minutes WHERE run_id=? ORDER BY minute", (run_id,)
            ).fetchall()
        ]
        expected = run["expected_minutes"]
        committed_set = set(committed)
        pending = [m for m in range(expected) if m not in committed_set]
        contiguous = 0
        for m in range(expected):
            if m in committed_set:
                contiguous += 1
            else:
                break
        status = "COMPLETE" if not pending else "INTERRUPTED"
        if run["status"] == "RUNNING" and pending:
            self.finalize_run(run_id, "INTERRUPTED")
        return {
            "run_id": run_id,
            "ledger_format_version": run["ledger_format_version"],
            "declaration_version": run["declaration_version"],
            "scale": run["scale"],
            "expected_minutes": expected,
            "committed_minutes": committed,
            "recovered_minutes": contiguous,
            "pending_minutes": pending,
            "status": status,
            "note": "未决分钟状态为 UNKNOWN，必须重跑；无法生成强制终止时刻的退出报告",
        }

    def recovered_aggregate(self, run_id, scale):
        payloads = self.load_minutes(run_id)
        from .engine import MinuteResult

        results = [MinuteResult(**p) for p in payloads]
        return aggregate(results, scale)
