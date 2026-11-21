"""SQLite 演练账本：迁移记录、跳过溯源、版本与进度游标同事务落盘。"""

import json
import os
import sqlite3

from . import LEDGER_FORMAT_VERSION
from .errors import LedgerError
from .graph import topological_order
from .models import digest_of

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    plan_version INTEGER PRIMARY KEY,
    plan_id TEXT NOT NULL,
    plan_payload TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS event_sets (
    plan_version INTEGER PRIMARY KEY,
    event_payload TEXT NOT NULL,
    event_digest TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version INTEGER NOT NULL,
    stage_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    clock_ms INTEGER NOT NULL,
    source_event_id TEXT,
    payload_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skip_traces (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    transition_seq INTEGER NOT NULL,
    failed_root TEXT NOT NULL,
    path_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    plan_version INTEGER NOT NULL,
    stage_id TEXT NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    unknown INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (plan_version, stage_id)
);
CREATE TABLE IF NOT EXISTS event_conclusions (
    plan_version INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (plan_version, event_id)
);
CREATE TABLE IF NOT EXISTS progress (
    plan_version INTEGER PRIMARY KEY,
    cursor_stage TEXT,
    state TEXT NOT NULL,
    updated_seq INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self):
        with self.conn:
            self.conn.executescript(SCHEMA)
            row = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('head_digest', ?)",
                    (digest_of({"ledger": "empty", "format": LEDGER_FORMAT_VERSION}),),
                )
            else:
                if int(row[0]) != LEDGER_FORMAT_VERSION:
                    raise LedgerError(
                        f"账本格式版本不兼容: {row[0]} != {LEDGER_FORMAT_VERSION}",
                        location=self.path,
                    )
                self._verify_chain()

    def close(self):
        self.conn.close()

    def _head_digest(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='head_digest'").fetchone()
        return row[0] if row else None

    def _set_head_digest(self, digest):
        self.conn.execute("UPDATE meta SET value=? WHERE key='head_digest'", (digest,))

    def _next_digest(self, entry):
        return digest_of({"prev": self._head_digest(), "entry": entry})

    def _verify_chain(self):
        """逐条重放迁移记录，校验内容摘要链；任一记录被篡改即 E_LEDGER。"""
        expected = digest_of({"ledger": "empty", "format": LEDGER_FORMAT_VERSION})
        rows = self.conn.execute("SELECT * FROM transitions ORDER BY seq").fetchall()
        for row in rows:
            entry = {
                "seq": row["seq"],
                "plan_version": row["plan_version"],
                "stage_id": row["stage_id"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "reason": row["reason"],
                "clock_ms": row["clock_ms"],
                "source_event_id": row["source_event_id"],
                "payload_digest": row["payload_digest"],
            }
            expected = digest_of({"prev": expected, "entry": entry})
        if expected != self._head_digest():
            raise LedgerError("账本内容摘要链校验失败，账本可能已被篡改或写入不完整", location=self.path)
        for row in self.conn.execute("SELECT payload_json, payload_digest FROM outcomes"):
            if digest_of(json.loads(row["payload_json"])) != row["payload_digest"]:
                raise LedgerError("阶段结局内容摘要不符", location=self.path)

    def save_plan(self, plan, events):
        plan_payload = {
            "plan_id": plan.plan_id,
            "version": plan.version,
            "stages": [stage.to_dict() for stage in plan.stages],
        }
        event_payload = [event.to_dict() for event in events.events]
        with self.conn:
            exists = self.conn.execute(
                "SELECT 1 FROM plans WHERE plan_version=?", (plan.version,)
            ).fetchone()
            if exists:
                raise LedgerError(f"预案版本已存在，禁止覆盖: v{plan.version}", location=str(plan.version))
            seq = self._next_seq()
            self.conn.execute(
                "INSERT INTO plans VALUES(?, ?, ?, ?, ?)",
                (
                    plan.version,
                    plan.plan_id,
                    json.dumps(plan_payload, ensure_ascii=False, sort_keys=True),
                    digest_of(plan_payload),
                    seq,
                ),
            )
            self.conn.execute(
                "INSERT INTO event_sets VALUES(?, ?, ?, ?)",
                (
                    plan.version,
                    json.dumps(event_payload, ensure_ascii=False, sort_keys=True),
                    digest_of(event_payload),
                    seq,
                ),
            )

    def _next_seq(self):
        row = self.conn.execute("SELECT COALESCE(MAX(created_seq), 0) + 1 AS seq FROM plans").fetchone()
        return row["seq"]

    def latest_version(self):
        row = self.conn.execute("SELECT MAX(plan_version) AS version FROM plans").fetchone()
        return row["version"]

    def load_plan_events(self, version):
        plan_row = self.conn.execute("SELECT * FROM plans WHERE plan_version=?", (version,)).fetchone()
        event_row = self.conn.execute("SELECT * FROM event_sets WHERE plan_version=?", (version,)).fetchone()
        if plan_row is None:
            raise LedgerError(f"账本缺少预案版本 v{version}", location=str(version))
        from .models import EventSet, Plan

        plan_payload = json.loads(plan_row["plan_payload"])
        if digest_of(plan_payload) != plan_row["plan_digest"]:
            raise LedgerError("预案内容摘要不符", location=f"plans.v{version}")
        event_payload = json.loads(event_row["event_payload"])
        if digest_of(event_payload) != event_row["event_digest"]:
            raise LedgerError("注入事件内容摘要不符", location=f"event_sets.v{version}")
        return Plan.from_dict(plan_payload, version=version), EventSet.from_list(event_payload)

    def list_versions(self):
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT plan_version FROM plans ORDER BY plan_version"
            ).fetchall()
        ]

    def append_simulation(self, plan, simulation, mark_stale_versions=None):
        """把一次模拟的迁移、溯源、结局与事件结论原子写入。"""
        mark_stale_versions = set(mark_stale_versions or ())
        topo = topological_order(plan.stages)
        with self.conn:
            for old_version in sorted(mark_stale_versions):
                self.conn.execute(
                    "UPDATE outcomes SET stale=1 WHERE plan_version=?", (old_version,)
                )
                self.conn.execute(
                    "UPDATE event_conclusions SET stale=1 WHERE plan_version=?", (old_version,)
                )
            last_cursor = None
            for transition in simulation.transitions:
                stage = transition["stage_id"]
                entry_digest = digest_of(
                    {
                        "plan_version": transition["plan_version"],
                        "stage_id": stage,
                        "from_state": transition["from_state"],
                        "to_state": transition["to_state"],
                        "reason": transition["reason"],
                        "clock_ms": transition["clock_ms"],
                        "source_event_id": transition["source_event_id"],
                    }
                )
                cur = self.conn.execute(
                    "INSERT INTO transitions(plan_version, stage_id, from_state, to_state, reason, "
                    "clock_ms, source_event_id, payload_digest) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        transition["plan_version"],
                        stage,
                        transition["from_state"],
                        transition["to_state"],
                        transition["reason"],
                        transition["clock_ms"],
                        transition["source_event_id"],
                        entry_digest,
                    ),
                )
                transition_seq = cur.lastrowid
                if transition["to_state"] == "skipped":
                    outcome = simulation.outcomes[stage]
                    self.conn.execute(
                        "INSERT INTO skip_traces(transition_seq, failed_root, path_json) VALUES(?, ?, ?)",
                        (
                            transition_seq,
                            outcome["upstream_failure"],
                            json.dumps(outcome["skip_trace"], ensure_ascii=False),
                        ),
                    )
                self._set_head_digest(self._next_digest(
                    {
                        "seq": transition_seq,
                        "plan_version": transition["plan_version"],
                        "stage_id": stage,
                        "from_state": transition["from_state"],
                        "to_state": transition["to_state"],
                        "reason": transition["reason"],
                        "clock_ms": transition["clock_ms"],
                        "source_event_id": transition["source_event_id"],
                        "payload_digest": entry_digest,
                    }
                ))
                last_cursor = stage
                self.conn.execute(
                    "INSERT INTO progress(plan_version, cursor_stage, state, updated_seq) "
                    "VALUES(?, ?, ?, ?) ON CONFLICT(plan_version) DO UPDATE SET "
                    "cursor_stage=excluded.cursor_stage, state=excluded.state, updated_seq=excluded.updated_seq",
                    (plan.version, last_cursor, transition["to_state"], transition_seq),
                )

            for stage_id in topo:
                payload = simulation.outcomes[stage_id]
                self.conn.execute(
                    "INSERT INTO outcomes(plan_version, stage_id, state, payload_json, payload_digest, stale, unknown) "
                    "VALUES(?, ?, ?, ?, ?, 0, 0) ON CONFLICT(plan_version, stage_id) DO UPDATE SET "
                    "state=excluded.state, payload_json=excluded.payload_json, "
                    "payload_digest=excluded.payload_digest, stale=0, unknown=0",
                    (
                        plan.version,
                        stage_id,
                        payload["state"],
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        digest_of(payload),
                    ),
                )
            for conclusion in simulation.event_conclusions:
                self.conn.execute(
                    "INSERT INTO event_conclusions(plan_version, event_id, payload_json, stale) "
                    "VALUES(?, ?, ?, 0) ON CONFLICT(plan_version, event_id) DO UPDATE SET "
                    "payload_json=excluded.payload_json, stale=0",
                    (
                        plan.version,
                        conclusion["event_id"],
                        json.dumps(conclusion, ensure_ascii=False, sort_keys=True),
                    ),
                )

    def mark_unknown_and_progress(self, version, unknown_stages):
        with self.conn:
            for stage_id in unknown_stages:
                row = self.conn.execute(
                    "SELECT 1 FROM outcomes WHERE plan_version=? AND stage_id=?",
                    (version, stage_id),
                ).fetchone()
                if row is None:
                    payload = {
                        "stage_id": stage_id,
                        "state": "unknown",
                        "plan_version": version,
                        "start_clock_ms": None,
                        "end_clock_ms": None,
                        "reason": "running_not_committed",
                        "source_event_id": None,
                        "criterion": {},
                        "criterion_passed": None,
                        "failed_actions": [],
                        "skip_trace": [],
                        "upstream_failure": None,
                    }
                    self.conn.execute(
                        "INSERT INTO outcomes(plan_version, stage_id, state, payload_json, "
                        "payload_digest, stale, unknown) VALUES(?, ?, 'unknown', ?, ?, 0, 1)",
                        (version, stage_id,
                         json.dumps(payload, ensure_ascii=False, sort_keys=True),
                         digest_of(payload)),
                    )
                else:
                    self.conn.execute(
                        "UPDATE outcomes SET unknown=1, state='unknown' WHERE plan_version=? AND stage_id=?",
                        (version, stage_id),
                    )

    def active_outcomes(self, version):
        rows = self.conn.execute(
            "SELECT payload_json FROM outcomes WHERE plan_version=? ORDER BY stage_id",
            (version,),
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def active_event_conclusions(self, version):
        rows = self.conn.execute(
            "SELECT payload_json FROM event_conclusions WHERE plan_version=? AND stale=0 ORDER BY event_id",
            (version,),
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def transitions(self, version):
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM transitions WHERE plan_version=? ORDER BY seq", (version,)
            ).fetchall()
        ]

    def progress(self, version):
        row = self.conn.execute(
            "SELECT * FROM progress WHERE plan_version=?", (version,)
        ).fetchone()
        return dict(row) if row else None

    def recover(self, version):
        """依据已完整提交的迁移恢复；running 未提交阶段标记 unknown，拒绝生成退出报告。"""
        plan, events = self.load_plan_events(version)
        topo = topological_order(plan.stages)
        transitions = self.transitions(version)
        completed_terminal = set()
        running_left = set()
        for transition in transitions:
            if transition["to_state"] in ("succeeded", "failed", "skipped"):
                completed_terminal.add(transition["stage_id"])
            if transition["to_state"] == "running":
                running_left.add(transition["stage_id"])
        unknown_stages = sorted(
            stage_id
            for stage_id in topo
            if stage_id in running_left and stage_id not in completed_terminal
        )
        if unknown_stages:
            self.mark_unknown_and_progress(version, unknown_stages)
        outcomes = {}
        for payload in self.active_outcomes(version):
            outcomes[payload["stage_id"]] = payload
        return {
            "plan": plan,
            "events": events,
            "outcomes": outcomes,
            "unknown_stages": unknown_stages,
            "recovered_transition_count": len(transitions),
            "can_build_exit_report": not unknown_stages,
        }

    def stats(self, version):
        rows = self.conn.execute(
            "SELECT state, COUNT(*) AS n FROM outcomes WHERE plan_version=? AND stale=0 GROUP BY state",
            (version,),
        ).fetchall()
        stats = {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0, "unknown": 0}
        for row in rows:
            key = "unknown" if row["state"] == "unknown" else row["state"]
            stats[key] = row["n"]
            stats["total"] += row["n"]
        return stats

    def ledger_digest(self):
        return self._head_digest()
