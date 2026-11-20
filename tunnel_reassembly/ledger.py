"""sqlite3 账本与会话快照。

设计要点：
  - 每个会话一个 sqlite 文件，WAL 关闭、单连接串行写入，
    每批记录在一个事务内提交；强制终止时只可能丢失最后一个未提交事务，
    已完整写入（committed）的记录才可用于恢复。
  - 快照携带 content_digest（规范化 JSON 的 sha256），重开时逐字段比对；
    版本或摘要不一致时整体替换通道、参数与窗口状态，不做增量合并。
  - 恢复时窗口内未落定的组统一标记 UNKNOWN，要求重跑，
    不生成强制终止时刻的退出报告。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .config import canonical_digest
from .errors import BusinessError, E_SNAPSHOT
from .model import (
    EstimateVersion,
    FlipRecord,
    FragmentEvent,
    GroupOutcome,
    ReassemblyResult,
    WindowEvent,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    config_version TEXT NOT NULL,
    trace_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    group_seq INTEGER NOT NULL,
    frag_offset INTEGER NOT NULL,
    frag_length INTEGER NOT NULL,
    total_frags INTEGER NOT NULL,
    send_ms INTEGER NOT NULL,
    arrive_ms INTEGER NOT NULL,
    payload_b64 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS estimates (
    channel_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    version INTEGER NOT NULL,
    skew_ms INTEGER,
    sample_count INTEGER NOT NULL,
    qerror_ms INTEGER NOT NULL,
    PRIMARY KEY (channel_id, sample_index)
);
CREATE TABLE IF NOT EXISTS window_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    window_seq INTEGER NOT NULL,
    action TEXT NOT NULL,
    estimate_version INTEGER NOT NULL,
    base_ms INTEGER NOT NULL,
    close_ms INTEGER,
    anchor_group INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS flips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    group_seq INTEGER NOT NULL,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    from_adjusted_ms INTEGER NOT NULL,
    to_adjusted_ms INTEGER NOT NULL,
    from_window INTEGER,
    to_window INTEGER,
    flip_kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    channel_id TEXT NOT NULL,
    group_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    estimate_version INTEGER NOT NULL,
    window_seq INTEGER,
    adjusted_ms INTEGER,
    digest TEXT,
    expected_digest TEXT,
    reason TEXT,
    PRIMARY KEY (channel_id, group_seq)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_ms INTEGER NOT NULL,
    config_version TEXT NOT NULL,
    trace_version TEXT NOT NULL,
    estimate_versions_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


class Ledger:
    """单会话账本。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def set_meta(self, key: str, value: object) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> object | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def register_channel(self, spec_dict: dict, config_version: str, trace_version: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO channels(channel_id, spec_json, config_version, trace_version) "
            "VALUES(?, ?, ?, ?)",
            (spec_dict["channel_id"], json.dumps(spec_dict, sort_keys=True), config_version, trace_version),
        )

    def save_result(self, result: ReassemblyResult) -> None:
        """在单个事务内完整写入一条通道的全部裁决记录。"""
        cid = result.channel_id
        with self.conn:
            self.conn.execute("DELETE FROM fragments WHERE channel_id=?", (cid,))
            self.conn.execute("DELETE FROM estimates WHERE channel_id=?", (cid,))
            self.conn.execute("DELETE FROM window_events WHERE channel_id=?", (cid,))
            self.conn.execute("DELETE FROM flips WHERE channel_id=?", (cid,))
            self.conn.execute("DELETE FROM outcomes WHERE channel_id=?", (cid,))

            for frag in result.fragments:
                import base64

                self.conn.execute(
                    "INSERT INTO fragments(channel_id, group_seq, frag_offset, frag_length, "
                    "total_frags, send_ms, arrive_ms, payload_b64) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        frag.group_seq,
                        frag.frag_offset,
                        frag.frag_length,
                        frag.total_frags,
                        frag.send_ms,
                        frag.arrive_ms,
                        base64.b64encode(frag.payload).decode("ascii"),
                    ),
                )
            for idx, est in enumerate(result.estimates):
                self.conn.execute(
                    "INSERT INTO estimates(channel_id, sample_index, version, skew_ms, "
                    "sample_count, qerror_ms) VALUES(?,?,?,?,?,?)",
                    (cid, idx, est.version, est.skew_ms, est.sample_count, est.qerror_ms),
                )
            for evt in result.window_events:
                self.conn.execute(
                    "INSERT INTO window_events(channel_id, window_seq, action, estimate_version, "
                    "base_ms, close_ms, anchor_group) VALUES(?,?,?,?,?,?,?)",
                    (cid, evt.window_seq, evt.action, evt.estimate_version, evt.base_ms, evt.close_ms, evt.anchor_group),
                )
            for flip in result.flips:
                self.conn.execute(
                    "INSERT INTO flips(channel_id, group_seq, from_version, to_version, "
                    "from_adjusted_ms, to_adjusted_ms, from_window, to_window, flip_kind) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        flip.group_seq,
                        flip.from_version,
                        flip.to_version,
                        flip.from_adjusted_ms,
                        flip.to_adjusted_ms,
                        flip.from_window,
                        flip.to_window,
                        flip.flip_kind,
                    ),
                )
            for out in result.outcomes:
                self.conn.execute(
                    "INSERT INTO outcomes(channel_id, group_seq, status, estimate_version, "
                    "window_seq, adjusted_ms, digest, expected_digest, reason) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        out.group_seq,
                        out.status,
                        out.estimate_version,
                        out.window_seq,
                        out.adjusted_ms,
                        out.digest,
                        out.expected_digest,
                        out.reason,
                    ),
                )


def _result_payload(result: ReassemblyResult) -> dict:
    return {
        "channel_id": result.channel_id,
        "estimates": [asdict(v) for v in result.estimates],
        "window_events": [asdict(v) for v in result.window_events],
        "flips": [asdict(v) for v in result.flips],
        "outcomes": [asdict(v) for v in result.outcomes],
    }


class Ledger(Ledger):  # noqa: F811 - 同文件扩展同名类
    def write_snapshot(
        self,
        snapshot_id: str,
        created_ms: int,
        config_version: str,
        trace_version: str,
        results: list[ReassemblyResult],
    ) -> str:
        estimate_versions = {
            r.channel_id: max((v.version for v in r.estimates if v.skew_ms is not None), default=0)
            for r in results
        }
        payload = {
            "results": [_result_payload(r) for r in sorted(results, key=lambda r: r.channel_id)],
            "estimate_versions": estimate_versions,
        }
        content_digest = canonical_digest(payload)
        with self.conn:
            self.conn.execute(
                "INSERT INTO snapshots(snapshot_id, created_ms, config_version, trace_version, "
                "estimate_versions_json, content_digest, payload_json) VALUES(?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    created_ms,
                    config_version,
                    trace_version,
                    json.dumps(estimate_versions, sort_keys=True),
                    content_digest,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        return content_digest

    def list_snapshots(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT snapshot_id, created_ms, config_version, trace_version, content_digest "
            "FROM snapshots ORDER BY created_ms, snapshot_id"
        ).fetchall()
        return [
            {
                "snapshot_id": r[0],
                "created_ms": r[1],
                "config_version": r[2],
                "trace_version": r[3],
                "content_digest": r[4],
            }
            for r in rows
        ]

    def load_latest_snapshot(
        self, expected_config_version: str, expected_trace_version: str
    ) -> dict:
        """恢复入口：版本与摘要校验失败即 E_SNAPSHOT，要求整体替换。"""
        row = self.conn.execute(
            "SELECT snapshot_id, config_version, trace_version, content_digest, payload_json "
            "FROM snapshots ORDER BY created_ms DESC, snapshot_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise BusinessError(E_SNAPSHOT, "账本中不存在任何完整快照，无法恢复")
        snapshot_id, config_version, trace_version, content_digest, payload_json = row
        if config_version != expected_config_version or trace_version != expected_trace_version:
            raise BusinessError(
                E_SNAPSHOT,
                (
                    f"快照版本与当前会话不一致（快照 config={config_version} "
                    f"trace={trace_version}；当前 config={expected_config_version} "
                    f"trace={expected_trace_version}），须整体替换而非增量合并"
                ),
                path=f"snapshots.{snapshot_id}",
            )
        payload = json.loads(payload_json)
        if canonical_digest(payload) != content_digest:
            raise BusinessError(
                E_SNAPSHOT,
                f"快照内容摘要不符（记录 {content_digest}），判定快照损坏，整体替换",
                path=f"snapshots.{snapshot_id}.content_digest",
            )
        payload["snapshot_id"] = snapshot_id
        payload["content_digest"] = content_digest
        return payload

    def mark_unknown_inflight(self) -> int:
        """强制终止恢复：窗口内未落定（PENDING/无结局）的组标记 UNKNOWN。"""
        cur = self.conn.execute(
            "UPDATE outcomes SET status='UNKNOWN', "
            "reason='强制终止时窗口内未落定，依据已提交账本恢复并要求重跑' "
            "WHERE status='PENDING'"
        )
        self.conn.commit()
        return cur.rowcount

    def load_fragments(self, channel_id: str) -> list[FragmentEvent]:
        import base64

        rows = self.conn.execute(
            "SELECT group_seq, frag_offset, frag_length, total_frags, send_ms, arrive_ms, payload_b64 "
            "FROM fragments WHERE channel_id=? ORDER BY group_seq, frag_offset",
            (channel_id,),
        ).fetchall()
        return [
            FragmentEvent(
                channel_id=channel_id,
                group_seq=r[0],
                frag_offset=r[1],
                frag_length=r[2],
                total_frags=r[3],
                send_ms=r[4],
                arrive_ms=r[5],
                payload=base64.b64decode(r[6]),
            )
            for r in rows
        ]

    def load_outcomes(self, channel_id: str) -> list[GroupOutcome]:
        rows = self.conn.execute(
            "SELECT group_seq, status, estimate_version, window_seq, adjusted_ms, digest, "
            "expected_digest, reason FROM outcomes WHERE channel_id=? ORDER BY group_seq",
            (channel_id,),
        ).fetchall()
        return [GroupOutcome(channel_id, *r) for r in rows]

    def load_estimates(self, channel_id: str) -> list[EstimateVersion]:
        rows = self.conn.execute(
            "SELECT version, skew_ms, sample_count, qerror_ms FROM estimates "
            "WHERE channel_id=? ORDER BY sample_index",
            (channel_id,),
        ).fetchall()
        return [EstimateVersion(*r) for r in rows]

    def load_window_events(self, channel_id: str) -> list[WindowEvent]:
        rows = self.conn.execute(
            "SELECT window_seq, action, estimate_version, base_ms, close_ms, anchor_group "
            "FROM window_events WHERE channel_id=? ORDER BY id",
            (channel_id,),
        ).fetchall()
        return [WindowEvent(channel_id, *r) for r in rows]

    def load_flips(self, channel_id: str) -> list[FlipRecord]:
        rows = self.conn.execute(
            "SELECT group_seq, from_version, to_version, from_adjusted_ms, to_adjusted_ms, "
            "from_window, to_window, flip_kind FROM flips WHERE channel_id=? ORDER BY id",
            (channel_id,),
        ).fetchall()
        return [FlipRecord(channel_id, *r) for r in rows]

    def replace_all(self) -> None:
        """整体替换：清空既有通道、参数、窗口状态与快照，仅保留库结构。"""
        with self.conn:
            for table in (
                "fragments",
                "estimates",
                "window_events",
                "flips",
                "outcomes",
                "snapshots",
                "channels",
            ):
                self.conn.execute(f"DELETE FROM {table}")
