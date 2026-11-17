"""Stable, Tk-independent view model for the diagnostic dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .reporting import build_report


TABLE_COLUMNS = (
    ("minute", "分钟", 58),
    ("workload_id", "工作负载", 105),
    ("requests", "请求", 65),
    ("latency", "延迟ms", 70),
    ("errors", "错误", 60),
    ("expected", "期望", 58),
    ("final", "最终", 58),
    ("decision", "决策", 70),
    ("budget_used", "预算占用", 74),
    ("cooldown", "冷却", 58),
    ("path", "路径", 95),
)


@dataclass(frozen=True)
class DashboardModel:
    report: dict[str, Any]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self.report["rows"])

    @property
    def budget_curve(self) -> list[dict[str, Any]]:
        return list(self.report["budget_curve"])

    @property
    def cooling_suppressions(self) -> list[dict[str, Any]]:
        return list(self.report["cooling_suppressions"])

    @property
    def title(self) -> str:
        run = self.report["run"]
        return (
            f"Run {run['run_id']} · {run['declaration_version']} · "
            f"path={run['path']} · ledger={self.report['ledger_digest'][:12]}"
        )

    @property
    def status_line(self) -> str:
        run = self.report["run"]
        available = "可出具退出报告" if self.report["exit_report_available"] else "仅已提交前缀，不可出具退出报告"
        return (
            f"状态 {run['status']} · {run['committed_minutes']}/{run['minute_count']} 分钟 · "
            f"预算 {run['budget']} · 峰值估计 {run['peak_memory_estimate_bytes']} B · {available}"
        )

    def table_row(self, row: dict[str, Any]) -> tuple[str, ...]:
        observation = row["observation"]
        return (
            str(row["minute"]),
            row["workload_id"],
            str(observation["requests"]),
            str(observation["avg_latency_ms"]),
            str(observation["errors"]),
            str(row["expected_replicas"]),
            str(row["final_replicas"]),
            row["decision_type"],
            f"{row['budget_used']}/{self.report['run']['budget']}",
            "E_COOLDOWN" if row["cooldown_suppressed"] else "-",
            row["path"],
        )

    def evidence_text(self, row: dict[str, Any]) -> str:
        lines = [
            f"minute={row['minute']} workload={row['workload_id']}",
            f"decision={row['decision_type']} expected={row['expected_replicas']} "
            f"budget_target={row['budget_target_replicas']} final={row['final_replicas']}",
            f"window={json.dumps(row['window_stats'], ensure_ascii=False, sort_keys=True)}",
            f"rule={row['selected_rule_id']}",
            "rule_evaluations="
            + json.dumps(row["rule_evaluations"], ensure_ascii=False, sort_keys=True),
        ]
        if row["cooldown_message"]:
            lines.append(row["cooldown_message"])
        if row["budget_clipped"]:
            lines.append(
                "budget_clip amount={} affected={}".format(
                    row["budget_clip_amount"], row["budget_affected_workloads"]
                )
            )
        lines.append("依据链:")
        lines.extend(f"- {item}" for item in row["evidence"])
        return "\n".join(lines)


def load_dashboard_model(db_path: str | Path, run_id: str) -> DashboardModel:
    with Ledger(db_path) as ledger:
        return DashboardModel(build_report(ledger, run_id))


def self_check(db_path: str | Path, run_id: str) -> dict[str, Any]:
    model = load_dashboard_model(db_path, run_id)
    rows = model.rows
    first = model.table_row(rows[0]) if rows else ()
    return {
        "ok": True,
        "business_code": "OK",
        "rows": len(rows),
        "suppressions": len(model.cooling_suppressions),
        "budget_points": len(model.budget_curve),
        "stable_order": [item["minute_index"] for item in rows]
        == sorted(item["minute_index"] for item in rows),
        "title": model.title,
        "first_row": first,
        "exit_report_available": model.report["exit_report_available"],
    }
