"""SQLite 账本持久化、快照版本识别整体替换与恢复重放校验。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .errors import ReplayError, SnapshotError

LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS call_outcomes (
    batch_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    upstream TEXT NOT NULL,
    downstream TEXT NOT NULL,
    client_key TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    bucket INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    connect_port INTEGER NOT NULL,
    port_kind TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (batch_id, seq)
);
CREATE TABLE IF NOT EXISTS service_states (
    batch_id INTEGER NOT NULL,
    service_id TEXT NOT NULL,
    samples INTEGER NOT NULL,
    success_rate_fixed INTEGER NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (batch_id, service_id)
);
CREATE TABLE IF NOT EXISTS degrade_triggers (
    batch_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    service_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    source TEXT NOT NULL,
    strategy TEXT NOT NULL,
    capacity_used INTEGER NOT NULL,
    PRIMARY KEY (batch_id, seq, service_id)
);
CREATE TABLE IF NOT EXISTS propagation_hops (
    batch_id INTEGER NOT NULL,
    hop_order INTEGER NOT NULL,
    degraded_service TEXT NOT NULL,
    upstream_service TEXT NOT NULL,
    edge_mode TEXT NOT NULL,
    threshold_fixed INTEGER,
    availability_fixed INTEGER,
    propagated INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (batch_id, hop_order)
);
CREATE TABLE IF NOT EXISTS domain_plans (
    batch_id INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    degraded_services TEXT NOT NULL,
    total_capacity INTEGER NOT NULL,
    total_capacity_fixed INTEGER NOT NULL,
    availability_rate_fixed INTEGER NOT NULL,
    source_plan TEXT NOT NULL,
    indistinguishable INTEGER NOT NULL,
    judgment_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, domain_id)
);
CREATE TABLE IF NOT EXISTS cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_seq INTEGER NOT NULL,
    last_batch INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, topology_version: str,
                 topology_digest: str) -> None:
        self.path = Path(path)
        self.topology_version = topology_version
        self.topology_digest = topology_digest
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._initialize_meta()

    def _initialize_meta(self) -> None:
        existing = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('topology_version', ?)",
                (self.topology_version,))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('topology_digest', ?)",
                (self.topology_digest,))
            self.conn.execute(
                "INSERT INTO cursor(id, last_seq, last_batch) VALUES(1, 0, 0)")
            self.conn.commit()

    @classmethod
    def open_or_replace(cls, path: str | Path, topology_version: str,
                        topology_digest: str, *,
                        raise_on_mismatch: bool = False
                        ) -> tuple["Ledger", dict]:
        """打开快照；版本/摘要不符则整体替换（非增量合并）。

        ``raise_on_mismatch=True`` 时先以 E_SNAPSHOT 报明依据，再由调用方
        重新打开执行整体替换，保证业务码可观测且替换语义不变。
        """
        path = Path(path)
        replaced = {"replaced": False, "reason": None}
        mismatch = cls._detect_mismatch(path, topology_digest)
        if mismatch is not None:
            replaced = mismatch
            if raise_on_mismatch:
                raise SnapshotError(
                    f"快照内容摘要/版本不符，触发整体替换: {replaced['reason']}",
                    field="topology_digest",
                    detail=replaced,
                )
            path.unlink()
        ledger = cls(path, topology_version, topology_digest)
        ledger._write_digest_marker(replaced)
        return ledger, replaced

    @staticmethod
    def _detect_mismatch(path: Path, topology_digest: str) -> dict | None:
        if not path.exists():
            return None
        probe = sqlite3.connect(path)
        try:
            probe.row_factory = sqlite3.Row
            rows = {r["key"]: r["value"] for r in
                    probe.execute("SELECT key, value FROM meta")}
        except sqlite3.DatabaseError:
            rows = {}
        finally:
            probe.close()
        if rows.get("format_version") != str(LEDGER_FORMAT_VERSION):
            return {"replaced": True,
                    "reason": f"format_version={rows.get('format_version')}"}
        if rows.get("topology_digest") != topology_digest:
            return {"replaced": True,
                    "reason": (f"digest_mismatch stored="
                               f"{rows.get('topology_digest')} "
                               f"declared={topology_digest}")}
        return None

    def _write_digest_marker(self, replaced: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('last_replace', ?)",
            (json.dumps(replaced, ensure_ascii=False),))
        self.conn.commit()

    def append_batch(self, batch_id: int, records: list, states: dict,
                     triggers: list, hops: list, domain_plans: dict,
                     last_seq: int) -> str:
        """同一事务写入批次全部表并更新游标，返回内容摘要。"""
        payload = {
            "batch_id": batch_id,
            "records": [r.__dict__ for r in records],
            "states": {sid: s.__dict__ for sid, s in states.items()},
            "triggers": [t.__dict__ for t in triggers],
            "hops": [h.__dict__ for h in hops],
            "plans": {did: _plan_dict(p) for did, p in domain_plans.items()},
            "last_seq": last_seq,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.conn:  # 单事务
            for r in records:
                self.conn.execute(
                    """INSERT INTO call_outcomes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (batch_id, r.seq, r.upstream, r.downstream, r.client_key,
                     r.domain_id, r.outcome, r.bucket, r.attempts, r.wait_ms,
                     r.latency_ms, r.connect_port, r.port_kind, r.error))
            for sid in sorted(states):
                s = states[sid]
                self.conn.execute(
                    "INSERT INTO service_states VALUES(?,?,?,?,?)",
                    (batch_id, sid, s.samples, s.success_rate_fixed, s.state))
            for trigger in triggers:
                self.conn.execute(
                    "INSERT INTO degrade_triggers VALUES(?,?,?,?,?,?,?)",
                    (batch_id, trigger.seq, trigger.service_id, trigger.domain_id,
                     trigger.source, trigger.strategy, trigger.capacity_used))
            for order, hop in enumerate(hops):
                self.conn.execute(
                    "INSERT INTO propagation_hops VALUES(?,?,?,?,?,?,?,?,?)",
                    (batch_id, order, hop.degraded_service, hop.upstream_service,
                     hop.edge_mode, hop.threshold_fixed, hop.availability_fixed,
                     1 if hop.propagated else 0, hop.reason))
            for did in sorted(domain_plans):
                plan = domain_plans[did]
                self.conn.execute(
                    "INSERT INTO domain_plans VALUES(?,?,?,?,?,?,?,?,?)",
                    (batch_id, did, ",".join(plan.degraded_services),
                     plan.total_capacity, plan.total_capacity_fixed,
                     plan.availability_rate_fixed, plan.source_plan,
                     1 if plan.indistinguishable else 0,
                     json.dumps(plan.judgment, ensure_ascii=False, sort_keys=True)))
            self.conn.execute(
                "UPDATE cursor SET last_seq=?, last_batch=? WHERE id=1",
                (last_seq, batch_id))
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                (f"batch_digest.{batch_id}", digest))
        return digest

    def cursor(self) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM cursor WHERE id=1").fetchone()

    def stored_outcomes(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM call_outcomes ORDER BY batch_id, seq").fetchall()

    def close(self) -> None:
        self.conn.close()


def _plan_dict(plan) -> dict:
    data = plan.__dict__.copy()
    data["degraded_services"] = list(plan.degraded_services)
    return data


def verify_replay(stored_rows: list, replay_records: list) -> None:
    """按账本拓扑版本重放后逐字段比对，定位首个不一致。"""
    fields = ("outcome", "upstream", "downstream", "domain_id", "bucket",
              "attempts", "port_kind")
    replay_by_seq = {r.seq: r for r in replay_records}
    for row in stored_rows:
        seq = row["seq"]
        if seq not in replay_by_seq:
            raise ReplayError(
                f"重放缺少账本事件 seq={seq}", field="seq", line=seq,
                detail={"seq": seq})
        replay = replay_by_seq[seq]
        for field_name in fields:
            stored_value = row[field_name]
            replay_value = getattr(replay, field_name)
            if stored_value != replay_value:
                raise ReplayError(
                    f"首个不一致位于 seq={seq} 服务 {row['downstream']} "
                    f"字段 {field_name}: 账本={stored_value} 重放={replay_value}",
                    field=field_name, line=seq,
                    detail={"seq": seq, "service_id": row["downstream"],
                            "stored": stored_value, "replay": replay_value})
    if len(replay_records) != len(stored_rows):
        raise ReplayError(
            f"重放事件数 {len(replay_records)} 与账本 {len(stored_rows)} 不一致",
            field="seq", line=min(len(stored_rows) + 1, len(replay_records) + 1))
