"""SQLite 权威账本。

设计要点：
* 每个批次在单条事务内同时写入批次记录、全部回执、对照与进度游标；
* 批次携带链式摘要（前批摘要 | 当前规范化载荷），形成只追加证据链；
* 声明演进后旧 spec_version 整体标记 stale 且不参与当前展示，采用整体替换；
* 快照版本/内容摘要与磁盘不一致时整体替换并保留只读备份。
"""

import os
import shutil
import sqlite3
import stat

from . import LEDGER_FORMAT_VERSION
from .digest import canonical_json, chain_digest, content_digest, sha256_hex
from .errors import ReplayError, SnapshotError, SpecError
from .spec import spec_to_jsonable


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spec_snapshots (
    spec_version INTEGER PRIMARY KEY,
    spec_digest TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    spec_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    chain_digest TEXT NOT NULL,
    PRIMARY KEY (spec_version, batch_id)
);
CREATE TABLE IF NOT EXISTS batches (
    spec_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    adopted_format TEXT NOT NULL,
    param_digest TEXT NOT NULL,
    request_seqs TEXT NOT NULL,
    destination_distribution TEXT NOT NULL,
    request_destinations TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    per_request_outputs TEXT NOT NULL,
    conclusion_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    chain_digest TEXT NOT NULL,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    wait_ms INTEGER,
    run_ms INTEGER,
    retry_count INTEGER,
    status TEXT NOT NULL,
    PRIMARY KEY (spec_version, batch_id)
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    merge_seq INTEGER NOT NULL,
    source_format TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    adopted INTEGER NOT NULL,
    reject_reason TEXT,
    raw_envelope BLOB NOT NULL,
    raw_len INTEGER NOT NULL,
    normalized_json TEXT NOT NULL,
    filled_fields_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS latency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    created_ms INTEGER NOT NULL
);
"""


def _now_ms() -> int:
    import time
    return time.monotonic_ns() // 1_000_000


class Ledger:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(SCHEMA)
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('format_version',?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
            elif int(row["value"]) != LEDGER_FORMAT_VERSION:
                raise SnapshotError(
                    "账本格式版本与当前软件不兼容",
                    stored=row["value"], expected=LEDGER_FORMAT_VERSION,
                )

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 声明快照与陈旧防护 ------------------------------------------------
    def spec_digest_of(self, spec) -> str:
        payload = {
            k: v for k, v in spec_to_jsonable(spec).items()
            if k != "source_paths"
        }
        return content_digest(payload, prefix="spec")

    def active_spec_version(self):
        row = self.conn.execute(
            "SELECT MAX(spec_version) AS v FROM spec_snapshots WHERE status='active'"
        ).fetchone()
        return row["v"] if row and row["v"] is not None else None

    def register_spec(self, spec) -> dict:
        """整体替换式登记声明，返回变更差异（新增/移除/变更，顺序稳定）。"""
        digest = self.spec_digest_of(spec)
        active = self.active_spec_version()
        diff = {"added": [], "removed": [], "changed": [],
                "replaced": False, "backup": None}
        if active is not None:
            old = self._snapshot(active)
            if old["spec_digest"] == digest:
                if active != spec.spec_version:
                    raise SpecError(
                        "声明版本必须单调递增",
                        field="spec_version", stored=active,
                        incoming=spec.spec_version,
                    )
                return diff
            if active >= spec.spec_version:
                raise SpecError(
                    "声明版本必须单调递增",
                    field="spec_version", stored=active,
                    incoming=spec.spec_version,
                )
            backup = self._replace_with_backup(spec, digest)
            diff["replaced"] = True
            diff["backup"] = backup
            diff.update(self._diff_workers(old["spec_json"], spec))
        else:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO spec_snapshots VALUES(?,?,?,'active',?)",
                    (spec.spec_version, digest,
                     canonical_json(spec_to_jsonable(spec)).decode("utf-8"),
                     _now_ms()),
                )
            diff["added"] = sorted(w.worker_id for w in spec.workers)
        self._verify_disk_summary(spec, digest)
        return diff

    def _snapshot(self, version):
        import json
        row = self.conn.execute(
            "SELECT * FROM spec_snapshots WHERE spec_version=?", (version,)
        ).fetchone()
        return {"spec_digest": row["spec_digest"],
                "spec_json": json.loads(row["spec_json"])}

    def _replace_with_backup(self, spec, digest):
        backup_path = f"{self.path}.bak-v{self.active_spec_version()}-{_now_ms()}"
        self.conn.commit()
        shutil.copy2(self.path, backup_path)
        os.chmod(backup_path, stat.S_IRUSR | stat.S_IRGRP)
        with self.conn:
            self.conn.execute(
                "UPDATE spec_snapshots SET status='stale'"
            )
            self.conn.execute(
                "UPDATE batches SET status='stale' WHERE status='committed'"
            )
            self.conn.execute(
                "INSERT INTO spec_snapshots VALUES(?,?,?,'active',?)",
                (spec.spec_version, digest,
                    canonical_json(spec_to_jsonable(spec)).decode("utf-8"),
                    _now_ms()),
            )
        return os.path.abspath(backup_path)

    def _diff_workers(self, old_spec_json, new_spec):
        old_workers = {w["worker_id"]: w for w in old_spec_json["workers"]}
        new_workers = {w.worker_id: w.__dict__ for w in new_spec.workers}
        added = sorted(set(new_workers) - set(old_workers))
        removed = sorted(set(old_workers) - set(new_workers))
        changed = []
        for wid in sorted(set(old_workers) & set(new_workers)):
            old_fields, new_fields = old_workers[wid], new_workers[wid]
            fields = sorted(
                key for key in new_fields
                if old_fields.get(key) != new_fields[key]
            )
            if fields:
                changed.append([wid, fields])
        return {"added": added, "removed": removed, "changed": changed}

    def _verify_disk_summary(self, spec, digest):
        row = self.conn.execute(
            "SELECT spec_digest FROM spec_snapshots WHERE spec_version=?",
            (spec.spec_version,),
        ).fetchone()
        if row is None or row["spec_digest"] != digest:
            raise SnapshotError(
                "磁盘声明摘要与内存状态不一致，已整体替换",
                spec_version=spec.spec_version,
            )

    # ---- 批次原子提交与恢复 ------------------------------------------------
    def committed_batches(self, spec_version: int):
        rows = self.conn.execute(
            "SELECT batch_id FROM batches WHERE spec_version=? AND status='committed'"
            " ORDER BY seq", (spec_version,)
        ).fetchall()
        return [row["batch_id"] for row in rows]

    def commit_batch(self, spec_version: int, seq: int, task, conclusion: dict,
                     receipts: list, timing: dict, prev_digest: str) -> str:
        """单事务写入批次归属、参数摘要、回执摘要与进度游标。"""
        payload = canonical_json(
            {"batch_id": conclusion["batch_id"],
             "content_digest": conclusion["content_digest"],
             "request_seqs": conclusion["request_seqs"],
             "output_digest": conclusion["output_digest"]}
        )
        digest = chain_digest(prev_digest, payload)
        with self.conn:
            self.conn.execute(
                "INSERT INTO batches VALUES("
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'committed')",
                (
                    spec_version, task.batch_id, seq, task.rule_id,
                    task.worker_id, conclusion["param_version"],
                    conclusion["source_format"], conclusion["param_digest"],
                    canonical_json(conclusion["request_seqs"]).decode(),
                    canonical_json(
                        conclusion["destination_distribution"]).decode(),
                    canonical_json(
                        conclusion["request_destinations"]).decode(),
                    conclusion["output_digest"],
                    canonical_json(
                        conclusion["per_request_outputs"]).decode(),
                    canonical_json(conclusion).decode(),
                    conclusion["content_digest"], digest,
                    timing.get("wait_start_ms"), timing.get("wait_end_ms"),
                    timing.get("wait_ms"), timing.get("run_ms"),
                    timing.get("retry_count", 0),
                ),
            )
            for item in receipts:
                self.conn.execute(
                    "INSERT INTO receipts(spec_version,batch_id,merge_seq,"
                    "source_format,param_version,adopted,reject_reason,"
                    "raw_envelope,raw_len,normalized_json,filled_fields_json)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        spec_version, task.batch_id, item["merge_seq"],
                        item["normalized"]["source_format"],
                        item["normalized"]["param_version"],
                        1 if item["adopted"] else 0, item.get("reject_reason"),
                        item["raw"], item["normalized"]["raw_len"],
                        canonical_json(item["normalized"]).decode(),
                        canonical_json(
                            item["normalized"]["filled_fields"]).decode(),
                    ),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO progress VALUES(?,?,?,?)",
                (spec_version, task.batch_id, seq, digest),
            )
        return digest

    def last_chain_digest(self, spec_version: int) -> str:
        row = self.conn.execute(
            "SELECT chain_digest FROM batches WHERE spec_version=? "
            "AND status='committed' ORDER BY seq DESC LIMIT 1",
            (spec_version,),
        ).fetchone()
        return row["chain_digest"] if row else sha256_hex(b"genesis")

    def recovery_state(self, spec, plan) -> dict:
        """只承认完整提交批次；计划内未提交批次一律标注 unknown 待重跑。"""
        committed = self.committed_batches(spec.spec_version)
        planned = [task.batch_id for task in plan]
        unknown = [bid for bid in planned if bid not in committed]
        return {
            "spec_version": spec.spec_version,
            "committed": committed,
            "unknown": unknown,
            "resume_from_seq": len(committed),
            "exit_report": None,
        }

    def verify_replay(self, spec_version: int, rederived: dict) -> None:
        """按账本记录逐项比较重推导结论，定位首个不一致批/请求/字段。"""
        rows = self.conn.execute(
            "SELECT * FROM batches WHERE spec_version=? AND status='committed'"
            " ORDER BY seq", (spec_version,)
        ).fetchall()
        stored = {row["batch_id"]: row for row in rows}
        for batch_id in sorted(set(stored) | set(rederived)):
            if batch_id not in stored:
                raise ReplayError(
                    "重放出现账本缺失批次", batch_id=batch_id, field="<batch>"
                )
            if batch_id not in rederived:
                raise ReplayError(
                    "重放缺少账本已有批次", batch_id=batch_id, field="<batch>"
                )
            row, derived = stored[batch_id], rederived[batch_id]
            mismatch = self._first_field_mismatch(row, derived)
            if mismatch is not None:
                raise ReplayError(
                    "重放结论与账本逐项比较不一致",
                    batch_id=batch_id, **mismatch,
                )

    @staticmethod
    def _first_field_mismatch(row, derived):
        import json
        scalar_fields = (
            "worker_id", "param_version", "adopted_format", "param_digest",
            "output_digest", "content_digest",
        )
        stored_conclusion = json.loads(row["conclusion_json"])
        for field_name in scalar_fields:
            stored = row[field_name]
            actual = derived.get(
                "source_format" if field_name == "adopted_format"
                else field_name
            )
            if stored != actual:
                return {"seq": None, "field": field_name,
                        "stored": stored, "actual": actual}
        for seq, destination in derived["request_destinations"]:
            stored_dest = dict(stored_conclusion["request_destinations"]).get(seq)
            if stored_dest != destination:
                return {"seq": seq, "field": "destination",
                        "stored": stored_dest, "actual": destination}
        stored_outputs = dict(json.loads(row["per_request_outputs"]))
        for seq, digest in derived["per_request_outputs"]:
            if stored_outputs.get(seq) != digest:
                return {"seq": seq, "field": "output_digest",
                        "stored": stored_outputs.get(seq), "actual": digest}
        if row["content_digest"] != derived["content_digest"]:
            return {"seq": None, "field": "content_digest",
                    "stored": row["content_digest"],
                    "actual": derived["content_digest"]}
        return None

    # ---- 快速视图（单次读取）与延迟统计 ------------------------------------
    def preview_rows(self, spec_version: int):
        """一次 SELECT 读取全部当前批次结论，不触发任何重算。"""
        rows = self.conn.execute(
            "SELECT batch_id,worker_id,request_seqs,output_digest,"
            "destination_distribution,conclusion_json,content_digest "
            "FROM batches WHERE spec_version=? AND status='committed'"
            " ORDER BY seq", (spec_version,)
        ).fetchall()
        return [dict(row) for row in rows]

    def record_latency(self, kind: str, elapsed_ms: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO latency(kind,elapsed_ms,created_ms) VALUES(?,?,?)",
                (kind, int(elapsed_ms), _now_ms()),
            )

    def latency_stats(self, kind: str):
        rows = self.conn.execute(
            "SELECT elapsed_ms FROM latency WHERE kind=?", (kind,)
        ).fetchall()
        values = [row["elapsed_ms"] for row in rows]
        if not values:
            return {"kind": kind, "count": 0, "avg_q1000": 0, "max_ms": 0}
        # 整数定点：均值放大 1000 倍保留，避免浮点不确定。
        avg_q1000 = sum(values) * 1000 // len(values)
        return {"kind": kind, "count": len(values),
                "avg_q1000": avg_q1000, "max_ms": max(values)}

    def receipts_of_batch(self, spec_version: int, batch_id: str):
        rows = self.conn.execute(
            "SELECT merge_seq,source_format,param_version,adopted,"
            "reject_reason,raw_len,filled_fields_json FROM receipts "
            "WHERE spec_version=? AND batch_id=? ORDER BY merge_seq",
            (spec_version, batch_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def verify_chain(self, spec_version: int):
        rows = self.conn.execute(
            "SELECT chain_digest,content_digest,request_seqs,output_digest,"
            "batch_id FROM batches WHERE spec_version=? AND status='committed'"
            " ORDER BY seq", (spec_version,)
        ).fetchall()
        prev = sha256_hex(b"genesis")
        for row in rows:
            payload = canonical_json(
                {"batch_id": row["batch_id"],
                 "content_digest": row["content_digest"],
                 "request_seqs": json_loads_ordered(row["request_seqs"]),
                 "output_digest": row["output_digest"]}
            )
            expect = chain_digest(prev, payload)
            if expect != row["chain_digest"]:
                raise SnapshotError(
                    "批次链式摘要校验失败", batch_id=row["batch_id"]
                )
            prev = row["chain_digest"]
        return prev


def json_loads_ordered(text):
    import json
    return json.loads(text)
