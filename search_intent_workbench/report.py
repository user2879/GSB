"""JSON diagnostics and a dependency-free HTML visualization."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, atomic_write_text


REPORT_VERSION = "diagnostic-report-v1"


def build_report(
    run: dict[str, Any],
    *,
    command_checks: list[dict[str, Any]],
    recovery_checks: list[dict[str, Any]],
    concurrency_check: dict[str, Any],
    fault_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent_distribution = Counter(
        record["intent"]["primary_intent"]
        for record in run.get("queries", [])
        if record["intent"]["primary_intent"] is not None
    )
    suppression_by_entry: Counter[str] = Counter()
    suppression_by_type: Counter[str] = Counter()
    accepted_by_type: Counter[str] = Counter()
    for record in run.get("queries", []):
        for slot in record["slots"]["slots"]:
            accepted_by_type[slot["entity_type"]] += 1
        for slot in record["slots"]["suppressed"]:
            suppression_by_entry[slot["entry_id"]] += 1
            suppression_by_type[slot["entity_type"]] += 1

    boundary_samples: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for record in run.get("queries", []):
        scores = record["intent"]["intent_scores"]
        primary = scores[0] if scores else None
        second = scores[1] if len(scores) > 1 else None
        margin_units = (
            primary["score_units"] - second["score_units"]
            if primary is not None and second is not None
            else (primary["score_units"] if primary is not None else 0)
        )
        boundary = {
            "query_id": record["query_id"],
            "primary_intent": record["intent"]["primary_intent"],
            "primary_score": primary["score_formatted"] if primary else None,
            "runner_up_intent": second["intent"] if second else None,
            "runner_up_score": second["score_formatted"] if second else None,
            "margin_units": margin_units,
        }
        boundary_samples.append(boundary)
        per_query.append(
            {
                "query_id": record["query_id"],
                "intent_label": record["input"]["intent_label"],
                "session_id": record["input"]["session_id"],
                "turn": record["input"]["turn"],
                "normalized_text": record["normalization"]["normalized_text"],
                "slots": record["slots"]["slots"],
                "suppressed_slots": record["slots"]["suppressed"],
                "intent": record["intent"],
                "rewrite": record["rewrite"],
                "cache_status": {
                    stage: record["stages"][stage]["cache_status"]
                    for stage in ("normalization", "slots", "intent", "rewrite")
                },
            }
        )
    boundary_samples.sort(key=lambda item: (item["margin_units"], item["query_id"]))

    total_slots = sum(accepted_by_type.values())
    total_suppressed = sum(suppression_by_entry.values())
    return {
        "report_version": REPORT_VERSION,
        "global_version": run.get("global_version"),
        "command_index": run.get("command_index"),
        "metadata": run["metadata"],
        "summary": {
            "query_count": len(per_query),
            "failure_count": len(run.get("failures", [])),
            "accepted_slot_count": total_slots,
            "suppressed_slot_count": total_suppressed,
            "suppression_ratio": round(
                total_suppressed / (total_slots + total_suppressed), 6
            )
            if total_slots + total_suppressed
            else 0.0,
        },
        "intent_distribution": dict(sorted(intent_distribution.items())),
        "slot_suppression": {
            "accepted_by_entity_type": dict(sorted(accepted_by_type.items())),
            "suppressed_by_entity_type": dict(sorted(suppression_by_type.items())),
            "suppressed_by_entry_id": dict(sorted(suppression_by_entry.items())),
        },
        "boundary_intent_samples": boundary_samples[:5],
        "per_query": per_query,
        "failures": run.get("failures", []),
        "reuse_report": run.get("reuse_report", {}),
        "command_evolution": command_checks,
        "serial_vs_concurrent": concurrency_check,
        "recovery_checks": recovery_checks,
        "fault_injection_hits": (fault_report or {}).get("injections", []),
        "fault_injection_summary": (fault_report or {}).get("summary", {}),
    }


def write_report(
    report: dict[str, Any], json_path: str | Path, html_path: str | Path | None = None
) -> None:
    atomic_write_json(json_path, report)
    if html_path is not None:
        atomic_write_text(html_path, render_html(report))


def _distribution_rows(report: dict[str, Any]) -> str:
    distribution = report.get("intent_distribution", {})
    maximum = max(distribution.values(), default=1)
    rows: list[str] = []
    for intent, count in distribution.items():
        width = max(2, round(100 * count / maximum))
        rows.append(
            '<div class="bar-row"><span>'
            + html.escape(str(intent))
            + "</span><div class=\"track\"><div class=\"bar\" style=\"width:"
            + str(width)
            + '%\"></div></div><b>'
            + str(count)
            + "</b></div>"
        )
    return "".join(rows) or "<p>No classified queries.</p>"


def _query_rows(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for query in report.get("per_query", []):
        scores = query["intent"]["intent_scores"]
        score_text = " · ".join(
            f'{item["intent"]}={item["score_formatted"]}' for item in scores
        )
        slot_text = ", ".join(
            f'{slot["entity_type"]}:{slot["matched_text"]}' for slot in query["slots"]
        ) or "—"
        suppressed_text = ", ".join(
            f'{slot["entry_id"]}@{slot["token_start"]}'
            for slot in query["suppressed_slots"]
        ) or "—"
        rows.append(
            "<tr><td>"
            + html.escape(query["query_id"])
            + "</td><td>"
            + html.escape(query["normalized_text"])
            + "</td><td>"
            + html.escape(query["intent"]["primary_intent"] or "")
            + "</td><td>"
            + html.escape(score_text)
            + "</td><td>"
            + html.escape(slot_text)
            + "</td><td>"
            + html.escape(suppressed_text)
            + "</td><td>"
            + html.escape(query["rewrite"]["text"])
            + "</td></tr>"
        )
    return "".join(rows)


def _command_rows(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for step in report.get("command_evolution", []):
        recomputed = step.get("recomputed_counts", {})
        rows.append(
            "<tr><td>"
            + str(step.get("step", ""))
            + "</td><td>"
            + html.escape(step.get("command_id", ""))
            + "</td><td>"
            + html.escape(step.get("type", ""))
            + "</td><td>"
            + ("PASS" if step.get("incremental_equals_full") else "FAIL")
            + "</td><td>"
            + html.escape(", ".join(f"{key}:{value}" for key, value in recomputed.items()))
            + "</td></tr>"
        )
    return "".join(rows)


def _fault_rows(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for injection in report.get("fault_injection_hits", []):
        rows.append(
            "<tr><td>"
            + html.escape(injection.get("injection_id", ""))
            + "</td><td>"
            + html.escape(injection.get("kind", ""))
            + "</td><td>"
            + ("HIT" if injection.get("detected") else "MISS")
            + "</td><td>"
            + html.escape(", ".join(injection.get("observed_codes", [])))
            + "</td></tr>"
        )
    return "".join(rows)


def render_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    recovery = report.get("recovery_checks", [])
    concurrency = report.get("serial_vs_concurrent", {})
    recovery_text = ", ".join(
        f'{item.get("name")}={item.get("status")}' for item in recovery
    ) or "none"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Query Intent Diagnosis Workbench</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#65717e; --line:#dce3ea; --accent:#1769aa; --ok:#13795b; --bad:#b42318; }}
body {{ margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif; color:var(--ink); background:#f4f7fa; }}
main {{ max-width:1320px; margin:0 auto; padding:28px 24px 60px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:30px 0 12px; }}
.sub {{ color:var(--muted); margin-bottom:22px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.card {{ background:white; border:1px solid var(--line); border-radius:10px; padding:15px; }}
.card b {{ font-size:23px; display:block; }} .card span {{ color:var(--muted); }}
.panel {{ background:white; border:1px solid var(--line); border-radius:10px; padding:18px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:820px; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px 9px; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }} .bar-row {{ display:grid; grid-template-columns:170px 1fr 45px; gap:10px; align-items:center; margin:7px 0; }}
.track {{ height:12px; background:#e9eef3; border-radius:99px; overflow:hidden; }} .bar {{ height:100%; background:linear-gradient(90deg,#1769aa,#52a8d8); }}
.pass {{ color:var(--ok); font-weight:700; }} .fail {{ color:var(--bad); font-weight:700; }}
code {{ background:#eef2f5; padding:2px 5px; border-radius:4px; }}
</style>
</head>
<body><main>
<h1>本地电商查询意图诊断工作台</h1>
<div class="sub">Global version {html.escape(str(report.get("global_version")))} · config <code>{html.escape(report.get("metadata", {}).get("config_hash", "")[:16])}</code> · serial/concurrent equal: <span class="{'pass' if concurrency.get('equal') else 'fail'}">{'PASS' if concurrency.get('equal') else 'FAIL'}</span></div>
<div class="cards">
<div class="card"><b>{summary.get("query_count", 0)}</b><span>有效查询</span></div>
<div class="card"><b>{summary.get("failure_count", 0)}</b><span>失败清单</span></div>
<div class="card"><b>{summary.get("accepted_slot_count", 0)}</b><span>实体槽位</span></div>
<div class="card"><b>{summary.get("suppressed_slot_count", 0)}</b><span>抑制候选</span></div>
<div class="card"><b>{concurrency.get("serial_peak_workers", 0)} / {concurrency.get("concurrent_peak_workers", 0)}</b><span>串行 / 并发峰值</span></div>
<div class="card"><b>{html.escape(recovery_text)}</b><span>快照恢复判定</span></div>
</div>
<h2>意图分布</h2><div class="panel">{_distribution_rows(report)}</div>
<h2>命令演进一致性</h2><div class="panel"><table><thead><tr><th>Step</th><th>Command</th><th>Type</th><th>Incremental == Full</th><th>Recomputed counts</th></tr></thead><tbody>{_command_rows(report)}</tbody></table></div>
<h2>异常注入命中</h2><div class="panel"><table><thead><tr><th>ID</th><th>Kind</th><th>Result</th><th>Observed codes</th></tr></thead><tbody>{_fault_rows(report)}</tbody></table></div>
<h2>逐查询诊断</h2><div class="panel"><table><thead><tr><th>Query</th><th>Normalized</th><th>Primary intent</th><th>Scores</th><th>Slots</th><th>Suppressed</th><th>Rewrite</th></tr></thead><tbody>{_query_rows(report)}</tbody></table></div>
</main></body></html>
"""
