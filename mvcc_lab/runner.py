"""判定轮次编排、恢复与一致性重放（业务要求 4、7、8）。"""

import json
import os
import signal
from pathlib import Path

from . import RULE_VERSION
from .digest import canonical
from .errors import ReplayError
from .executor import ReplayExecutor
from .ledger import Ledger
from .scheduler import Scheduler
from .visibility import cause_proportions, decide_all


def _compare_field(query_id, read_point_id, row_key, field, expected, actual):
    if expected != actual:
        raise ReplayError(
            f"重放结论与一次性完整判定在字段 {field!r} 不一致",
            object_id=f"{read_point_id}/{row_key}/{query_id}",
            field=field,
            detail={"expected": expected, "actual": actual},
        )


def compare_decisions(expected_summaries, actual_summaries) -> None:
    """逐项比较；差异定位到首个不一致的读点、行键与字段名。"""
    expected = {s["query_id"]: s for s in expected_summaries}
    actual = {s["query_id"]: s for s in actual_summaries}
    for qid in sorted(set(expected) | set(actual)):
        if qid not in actual:
            raise ReplayError("重放缺少查询结论", object_id=qid,
                              field="query_id")
        if qid not in expected:
            raise ReplayError("重放出现多余查询结论", object_id=qid,
                              field="query_id")
        exp, got = expected[qid], actual[qid]
        for field in ("read_point_id", "row_key", "visible_version",
                      "visible_txn_id", "excluded", "reason_chain",
                      "tie_applied", "error_influenced", "causes",
                      "rule_version"):
            _compare_field(qid, exp["read_point_id"], exp["row_key"],
                           field, exp.get(field), got.get(field))


class Runner:
    def __init__(self, spec, config_path: str, ledger_path: str,
                 injection: str = "none", batch_size: int = 2,
                 kill_after: int | None = None, replay: bool = False):
        self.spec = spec
        self.config_path = config_path
        self.ledger_path = ledger_path
        self.injection = injection
        self.batch_size = batch_size
        self.kill_after = kill_after
        self.replay = replay
        self.ledger = Ledger(ledger_path, spec)

    def run_full_round(self) -> dict:
        identity = self.ledger.identify_or_replace()
        recovery = self.ledger.recover()
        unknown = set(recovery["unknown_query_ids"])
        if recovery["last_committed_round"] is None:
            unknown = {q.query_id for q in self.spec.queries}
        # 强制终止模拟：在轮次意图提交后、结论提交前杀死本进程。
        round_id = self.ledger.start_round(unknown)
        if self.kill_after is not None and len(unknown) > self.kill_after:
            self._force_exit(round_id)

        with ReplayExecutor(self.config_path,
                            injection=self._executor_injection()) as executor:
            scheduler = Scheduler(self.spec, batch_size=self.batch_size)
            summaries = scheduler.run(executor)
            scheduler.verify_within_limits()
            peaks = scheduler.peaks()
            traces = scheduler.occupancy_trace()

        # 仅重放未知查询；已完整提交的结论保持账本原值。
        if recovery["committed_decisions"] and not unknown:
            summaries = list(recovery["committed_decisions"].values())
        cursor = {
            "committed_queries": sorted(s["query_id"] for s in summaries),
            "total_queries": len(self.spec.queries),
        }
        content_digest = self.ledger.commit_round(summaries, traces, cursor)
        decisions = decide_all(self.spec,
                               forced_rule_conflict=(self.injection ==
                                                     "rule_conflict"))
        proportions = cause_proportions(decisions)
        report = {
            "rule_version": RULE_VERSION,
            "round_id": round_id,
            "identity": identity,
            "recovery": {
                "last_committed_round": recovery["last_committed_round"],
                "open_rounds": recovery["open_rounds"],
                "unknown_query_ids": recovery["unknown_query_ids"],
            },
            "peaks": peaks,
            "queries_committed": len(summaries),
            "content_digest": content_digest,
            "cause_proportions": proportions,
            "summaries": summaries,
            "schedule_trace": traces,
        }
        if self.replay:
            self._replay_check(report, decisions)
        return report

    def _executor_injection(self) -> str:
        # 超时/端口/字段缺失注入直接透传给执行器；强制终止不改变协议。
        if self.injection in ("silent", "port_busy", "missing_field",
                              "rule_conflict", "silent_hold"):
            return self.injection
        return "none"

    def _replay_check(self, report: dict, inproc_decisions) -> None:
        """按账本记录重放剩余判定并与一次性完整判定逐项比较。"""
        expected = [d.summary() for d in inproc_decisions]
        actual = report["summaries"]
        compare_decisions(expected, actual)
        report["replay_check"] = {
            "status": "consistent",
            "compared_queries": len(actual),
        }

    def _force_exit(self, round_id: int):
        # 显式不生成退出报告：仅保留已提交的轮次意图记录。
        self.ledger.close()
        os.kill(os.getpid(), signal.SIGKILL)
        raise SystemExit(137)  # pragma: no cover - SIGKILL 不会返回

    def close(self):
        self.ledger.close()


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(
        canonical(report) + "\n", encoding="utf-8"
    )
