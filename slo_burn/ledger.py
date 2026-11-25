"""SQLite 持久化账本：唯一可复核事实源。

所有写入在单事务内提交并携带整数格式版本与内容摘要（SHA-256）。
快照版本或摘要不一致时整体替换，禁止增量合并。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from . import LEDGER_FORMAT_VERSION
from .errors import LedgerError
from .models import WindowConclusion
from .stream import CoverageRecord
from .params import VersionedSpec

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS specs (
    service_id TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    effective_from INTEGER NOT NULL,
    effective_to INTEGER,
    payload TEXT NOT NULL,
    PRIMARY KEY (service_id, param_version)
);
CREATE TABLE IF NOT EXISTS batches (
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    successes INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    source_line INTEGER NOT NULL,
    source_batch_seq INTEGER NOT NULL,
    source_stream TEXT NOT NULL,
    commit_seq INTEGER NOT NULL,
    PRIMARY KEY (service_id, minute)
);
CREATE TABLE IF NOT EXISTS conclusions (
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    param_version INTEGER NOT NULL,
    path TEXT NOT NULL,
    total INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    budget_total_scaled INTEGER NOT NULL,
    remaining_budget_scaled INTEGER NOT NULL,
    burn_rate_num INTEGER NOT NULL,
    burn_rate_den INTEGER NOT NULL,
    alert INTEGER NOT NULL,
    alert_reasons TEXT NOT NULL,
    chunk_capacity INTEGER,
    stale INTEGER NOT NULL DEFAULT 0,
    commit_seq INTEGER NOT NULL,
    PRIMARY KEY (service_id, minute, param_version)
);
CREATE TABLE IF NOT EXISTS coverage (
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    kept_line INTEGER NOT NULL,
    kept_batch_seq INTEGER NOT NULL,
    kept_stream TEXT NOT NULL,
    covered_line INTEGER NOT NULL,
    covered_batch_seq INTEGER NOT NULL,
    covered_stream TEXT NOT NULL,
    rule TEXT NOT NULL,
    commit_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS adjudications (
    event_seq INTEGER NOT NULL,
    observed_ms INTEGER NOT NULL,
    slot_id INTEGER NOT NULL,
    service_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    decision TEXT NOT NULL,
    basis TEXT NOT NULL,
    prior_owner TEXT,
    commit_seq INTEGER NOT NULL,
    PRIMARY KEY (event_seq, commit_seq)
);
CREATE TABLE IF NOT EXISTS pending_batches (
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    commit_seq INTEGER NOT NULL,
    PRIMARY KEY (service_id, minute)
);
CREATE TABLE IF NOT EXISTS commits (
    commit_seq INTEGER PRIMARY KEY,
    committed_ms INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    detail TEXT NOT NULL
);
"""


def digest_payload(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        # DDL 在 sqlite3 自动事务模式下执行，避免 executescript 隐式提交干扰外层事务
        connection = self.connection
        connection.execute("BEGIN")
        try:
            for statement in [part.strip() for part in _SCHEMA.split(";") if part.strip()]:
                connection.execute(statement)
            row = connection.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
            elif int(row[0]) != LEDGER_FORMAT_VERSION:
                raise LedgerError(
                    "账本格式版本不受支持，须整体替换而非增量合并",
                    expected=LEDGER_FORMAT_VERSION,
                    actual=int(row[0]),
                )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    @contextmanager
    def transaction(self):
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def next_commit_seq(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(commit_seq), 0) + 1 FROM commits"
        ).fetchone()
        return int(row[0])

    def write_commit(
        self,
        kind: str,
        conclusions: list[WindowConclusion],
        points: list | None = None,
        coverage: list[CoverageRecord] | None = None,
        adjudications: list | None = None,
        version_chains: dict[str, list[VersionedSpec]] | None = None,
        detail: dict[str, Any] | None = None,
        stale_minutes: dict[str, set[int]] | None = None,
    ) -> int:
        """原子提交一批结论；旧版本结论整体标记陈旧。"""
        seq = self.next_commit_seq()
        digest_rows = [item.__dict__ for item in conclusions]
        content_digest = digest_payload(digest_rows)
        with self.transaction() as connection:
            if version_chains is not None:
                connection.execute("DELETE FROM specs")
                for chain in version_chains.values():
                    for versioned in chain:
                        connection.execute(
                            "INSERT INTO specs(service_id, param_version, effective_from,"
                            " effective_to, payload) VALUES(?,?,?,?,?)",
                            (
                                versioned.spec.service_id,
                                versioned.spec.param_version,
                                versioned.effective_from,
                                versioned.effective_to,
                                json.dumps(versioned.spec.__dict__, sort_keys=True),
                            ),
                        )
            if points:
                for service_id, point in points:
                    connection.execute(
                        "INSERT OR REPLACE INTO batches(service_id, minute, successes,"
                        " failures, source_line, source_batch_seq, source_stream, commit_seq)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (
                            service_id,
                            point.minute,
                            point.successes,
                            point.failures,
                            point.source_line,
                            point.source_batch_seq,
                            point.source_stream,
                            seq,
                        ),
                    )
            if coverage:
                for record in coverage:
                    connection.execute(
                        "INSERT INTO coverage(service_id, minute, kept_line, kept_batch_seq,"
                        " kept_stream, covered_line, covered_batch_seq, covered_stream,"
                        " rule, commit_seq) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            record.service_id,
                            record.minute,
                            record.kept_line,
                            record.kept_batch_seq,
                            record.kept_stream,
                            record.covered_line,
                            record.covered_batch_seq,
                            record.covered_stream,
                            record.rule,
                            seq,
                        ),
                    )
            if adjudications:
                for decision in adjudications:
                    connection.execute(
                        "INSERT INTO adjudications(event_seq, observed_ms, slot_id,"
                        " service_id, lease_id, version, decision, basis, prior_owner,"
                        " commit_seq) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            decision.event_seq,
                            decision.observed_ms,
                            decision.slot_id,
                            decision.service_id,
                            decision.lease_id,
                            decision.version,
                            decision.decision,
                            decision.basis,
                            decision.prior_owner,
                            seq,
                        ),
                    )
            # 每分钟的当前参数版本：只有非当前版本的历史行才标记陈旧，
            # 保证每个 (service, minute) 恰有一条活跃结论参与展示与重放。
            current_version_at: dict[tuple[str, int], int] = {}
            if version_chains is not None:
                for service_id, chain in version_chains.items():
                    service_minutes = sorted(
                        item.minute for item in conclusions if item.service_id == service_id
                    )
                    for minute in service_minutes:
                        for chain_item in chain:
                            if chain_item.covers(minute):
                                current_version_at[(service_id, minute)] = (
                                    chain_item.spec.param_version
                                )
                                break
            for item in conclusions:
                current_version = current_version_at.get(
                    (item.service_id, item.minute), item.param_version
                )
                if item.param_version != current_version:
                    stale_flag = 1
                else:
                    stale_flag = 0
                connection.execute(
                    "INSERT OR REPLACE INTO conclusions(service_id, minute, param_version,"
                    " path, total, failures, budget_total_scaled, remaining_budget_scaled,"
                    " burn_rate_num, burn_rate_den, alert, alert_reasons, chunk_capacity,"
                    " stale, commit_seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item.service_id,
                        item.minute,
                        item.param_version,
                        item.path,
                        item.total,
                        item.failures,
                        item.budget_total_scaled,
                        item.remaining_budget_scaled,
                        item.burn_rate_num,
                        item.burn_rate_den,
                        int(item.alert),
                        json.dumps(item.alert_reasons),
                        item.chunk_capacity,
                        stale_flag,
                        seq,
                    ),
                )
            connection.execute(
                "INSERT INTO commits(commit_seq, committed_ms, kind, content_digest,"
                " format_version, detail) VALUES(?,?,?,?,?,?)",
                (
                    seq,
                    __import__("slo_burn.evaluator", fromlist=["now_ms"]).now_ms(),
                    kind,
                    content_digest,
                    LEDGER_FORMAT_VERSION,
                    json.dumps(detail or {}, sort_keys=True),
                ),
            )
        return seq

    def mark_stale(self, service_id: str, minutes: Iterable[int]) -> int:
        seq = self.next_commit_seq()
        with self.transaction() as connection:
            for minute in minutes:
                connection.execute(
                    "UPDATE conclusions SET stale=1, commit_seq=? WHERE service_id=? AND minute=?",
                    (seq, service_id, minute),
                )
            connection.execute(
                "INSERT INTO commits(commit_seq, committed_ms, kind, content_digest,"
                " format_version, detail) VALUES(?,?,?,?,?,?)",
                (
                    seq,
                    __import__("slo_burn.evaluator", fromlist=["now_ms"]).now_ms(),
                    "mark-stale",
                    digest_payload([{"service_id": service_id, "minutes": list(minutes)}]),
                    LEDGER_FORMAT_VERSION,
                    json.dumps({"service_id": service_id}),
                ),
            )
        return seq

    def register_pending(self, service_id: str, minute: int, payload: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO pending_batches(service_id, minute, payload, status,"
                " commit_seq) VALUES(?,?,?,?,?)",
                (service_id, minute, json.dumps(payload, sort_keys=True), "unknown", self.next_commit_seq()),
            )

    def clear_pending(self, service_id: str, minute: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM pending_batches WHERE service_id=? AND minute=?",
                (service_id, minute),
            )

    def pending_batches(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT service_id, minute, payload, status FROM pending_batches ORDER BY service_id, minute"
        ).fetchall()
        return [
            {"service_id": row[0], "minute": row[1], "payload": json.loads(row[2]), "status": row[3]}
            for row in rows
        ]

    def active_conclusions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT service_id, minute, param_version, path, total, failures,"
            " budget_total_scaled, remaining_budget_scaled, burn_rate_num, burn_rate_den,"
            " alert, alert_reasons, chunk_capacity FROM conclusions WHERE stale=0"
            " ORDER BY service_id, minute"
        ).fetchall()
        columns = [
            "service_id", "minute", "param_version", "path", "total", "failures",
            "budget_total_scaled", "remaining_budget_scaled", "burn_rate_num",
            "burn_rate_den", "alert", "alert_reasons", "chunk_capacity",
        ]
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            item["alert"] = bool(item["alert"])
            item["alert_reasons"] = tuple(json.loads(item["alert_reasons"]))
            result.append(item)
        return result

    def latest_commit(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT commit_seq, committed_ms, kind, content_digest, format_version, detail"
            " FROM commits ORDER BY commit_seq DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "commit_seq": row[0],
            "committed_ms": row[1],
            "kind": row[2],
            "content_digest": row[3],
            "format_version": row[4],
            "detail": json.loads(row[5]),
        }

    def verify_format(self) -> None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        if row is None or int(row[0]) != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "快照格式版本不一致，须整体替换",
                expected=LEDGER_FORMAT_VERSION,
                actual=None if row is None else int(row[0]),
            )

    def replace_all(self) -> None:
        """快照不一致时的整体替换：清空全部业务表（保留 meta）。"""
        with self.transaction() as connection:
            for table in (
                "commits", "conclusions", "batches", "coverage",
                "adjudications", "pending_batches", "specs",
            ):
                connection.execute(f"DELETE FROM {table}")
