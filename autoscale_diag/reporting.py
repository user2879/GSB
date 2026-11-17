"""Stable JSON, snapshot and self-contained HTML reports."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

from .ledger import Ledger


REPORT_VERSION = "1.0"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_report(ledger: Ledger, run_id: str) -> dict[str, Any]:
    run = ledger.get_run(run_id)
    if run is None:
        from .errors import ledger_error

        raise ledger_error("run does not exist", "run_id", run_id=run_id)
    committed = ledger.load_committed(run_id)
    suppressions = ledger.cooling_suppressions(run_id)
    recovery_events = ledger.recovery_events(run_id)
    complete = run["status"] == "complete" and len(committed) == run["minute_count"]
    peak_memory = max(
        (int(item["peak_memory_estimate_bytes"]) for item in committed), default=0
    )

    minute_rows: list[dict[str, Any]] = []
    budget_curve: list[dict[str, Any]] = []
    decision_counts = {"expand": 0, "shrink": 0, "hold": 0}
    clipping_events = 0
    for header in committed:
        budget_curve.append(
            {
                "minute_index": header["minute_index"],
                "minute": header["minute"],
                "budget_used": header["budget_used"],
                "budget": run["budget"],
            }
        )
        for item in header["decisions"]:
            decision = item["decision"]
            decision_type = decision["decision_type"]
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            if decision["budget_clipped"]:
                clipping_events += 1
            minute_rows.append(
                {
                    "minute_index": header["minute_index"],
                    "minute": header["minute"],
                    "path": header["path"],
                    "budget_used": header["budget_used"],
                    "workload_id": decision["workload_id"],
                    "concurrency_per_replica": next(
                        workload["concurrency_per_replica"]
                        for workload in run["spec"]["workloads"]
                        if workload["id"] == decision["workload_id"]
                    ),
                    "observation": decision["observation"],
                    "window_stats": decision["window_stats"],
                    "selected_rule_id": decision["selected_rule_id"],
                    "rule_evaluations": decision["rule_evaluations"],
                    "expected_replicas": decision["expected_replicas"],
                    "previous_replicas": decision["previous_replicas"],
                    "attempted_replicas": decision["attempted_replicas"],
                    "budget_target_replicas": decision["budget_target_replicas"],
                    "final_replicas": decision["final_replicas"],
                    "decision_type": decision_type,
                    "budget_clipped": decision["budget_clipped"],
                    "budget_clip_amount": decision["budget_clip_amount"],
                    "budget_affected_workloads": decision["budget_affected_workloads"],
                    "cooldown_active": decision["cooldown_active"],
                    "cooldown_suppressed": decision["cooldown_suppressed"],
                    "cooldown_code": decision["cooldown_code"],
                    "cooldown_trigger_minute": decision["cooldown_trigger_minute"],
                    "cooldown_until_minute": decision["cooldown_until_minute"],
                    "cooldown_message": decision["cooldown_message"],
                    "evidence": decision["evidence"],
                }
            )

    minute_rows.sort(key=lambda item: (item["minute_index"], item["workload_id"]))
    budget_curve.sort(key=lambda item: item["minute_index"])
    digest_payload = {
        "declaration_version": run["declaration_version"],
        "spec_sha256": run["spec_sha256"],
        "budget": run["budget"],
        "path": run["path"],
        "window_size": run["window_size"],
        "minutes": [
            {
                "minute_index": header["minute_index"],
                "minute": header["minute"],
                "path": header["path"],
                "budget_used": header["budget_used"],
                "budget": header["budget"],
                "decisions": [
                    item["decision"] for item in sorted(header["decisions"], key=lambda row: row["workload_id"])
                ],
            }
            for header in committed
        ],
    }
    ledger_digest = hashlib.sha256(_json(digest_payload).encode("utf-8")).hexdigest()

    business_codes: list[str] = []
    if complete:
        business_codes.append("OK")
    else:
        business_codes.append("E_LEDGER")
    if suppressions:
        business_codes.append("E_COOLDOWN")

    auto_path = (
        "full"
        if run["minute_count"] < run["policy"]["aggregation_threshold"]
        else "incremental"
    )
    report = {
        "report_version": REPORT_VERSION,
        "ok": complete,
        "business_codes": business_codes,
        "exit_report_available": complete,
        "run": {
            "run_id": run["run_id"],
            "declaration_version": run["declaration_version"],
            "spec_sha256": run["spec_sha256"],
            "status": run["status"],
            "path": run["path"],
            "window_capacity": run["window_size"],
            "peak_memory_estimate_bytes": peak_memory,
            "budget": run["budget"],
            "minute_count": run["minute_count"],
            "committed_minutes": len(committed),
            "created_at": run["created_at"],
            "completed_at": run["completed_at"],
        },
        "path_selection": {
            "mode": "auto" if run["path"] == auto_path else "forced",
            "rule": "full when minute_count < aggregation_threshold, otherwise incremental",
            "aggregation_threshold": run["policy"]["aggregation_threshold"],
            "minute_count": run["minute_count"],
            "auto_would_select": auto_path,
            "forced": run["path"] != auto_path,
            "selected": run["path"],
        },
        "summary": {
            "decision_counts": dict(sorted(decision_counts.items())),
            "cooling_suppressions": len(suppressions),
            "budget_clipping_events": clipping_events,
            "max_budget_used": max(
                (item["budget_used"] for item in budget_curve), default=0
            ),
        },
        "ledger_digest": ledger_digest,
        "budget_curve": budget_curve,
        "rows": minute_rows,
        "cooling_suppressions": suppressions,
        "recovery_events": recovery_events,
        "policy": run["policy"],
        "diagnostic_note": (
            "complete committed ledger"
            if complete
            else "prefix only; uncommitted minutes are unknown and must be rerun"
        ),
    }
    if not complete:
        report["next_uncommitted_minute_index"] = len(committed)
    return report


def write_text_atomic(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _svg_budget_chart(report: dict[str, Any]) -> str:
    curve = report["budget_curve"]
    if not curve:
        return '<p class="empty">No committed minutes.</p>'
    width, height = 920, 240
    left, right, top, bottom = 48, 18, 20, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    budget = max(1, int(report["run"]["budget"]))
    count = max(1, len(curve))

    def x(index: int) -> float:
        return left + (plot_width * index / max(1, count - 1))

    def y(value: int) -> float:
        return top + plot_height - (plot_height * value / budget)

    points = " ".join(f"{x(i):.1f},{y(item['budget_used']):.1f}" for i, item in enumerate(curve))
    budget_y = y(budget)
    labels = []
    for index, item in enumerate(curve):
        if len(curve) <= 16 or index % max(1, len(curve) // 12) == 0:
            labels.append(
                f'<text x="{x(index):.1f}" y="{height - 12}" text-anchor="middle">{item["minute"]}</text>'
            )
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(item["budget_used"]):.1f}" r="3">'
        f'<title>minute {item["minute"]}: {item["budget_used"]}/{budget}</title></circle>'
        for i, item in enumerate(curve)
    )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Budget occupancy curve">
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
  <line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" class="axis"/>
  <line x1="{left}" y1="{budget_y:.1f}" x2="{width - right}" y2="{budget_y:.1f}" class="budget-line"/>
  <text x="{left + 4}" y="{budget_y - 5:.1f}" class="budget-label">budget {budget}</text>
  <polyline points="{points}" class="budget-path"/>
  {dots}
  {''.join(labels)}
  <text x="{width - right}" y="{height - 12}" text-anchor="end">minute</text>
</svg>
"""


def render_html(report: dict[str, Any]) -> str:
    escaped_codes = ", ".join(html.escape(code) for code in report["business_codes"])
    row_html = []
    for row in report["rows"]:
        observation = row["observation"]
        cooling = row["cooldown_message"] or ""
        clip = f"{row['budget_clip_amount']}" if row["budget_clipped"] else "0"
        row_html.append(
            "<tr data-minute='{mi}' data-workload='{wid}'>"
            "<td>{minute}</td><td>{wid}</td><td>{req}</td><td>{lat}</td><td>{err}</td>"
            "<td>{expected}</td><td>{budget_target}</td><td>{final}</td><td>{decision}</td>"
            "<td>{budget_used}</td><td>{clip}</td><td>{cooldown}</td>"
            "</tr>".format(
                mi=row["minute_index"],
                wid=html.escape(row["workload_id"]),
                minute=row["minute"],
                req=observation["requests"],
                lat=observation["avg_latency_ms"],
                err=observation["errors"],
                expected=row["expected_replicas"],
                budget_target=row["budget_target_replicas"],
                final=row["final_replicas"],
                decision=html.escape(row["decision_type"]),
                budget_used=row["budget_used"],
                clip=clip,
                cooldown=html.escape(cooling),
            )
        )
    suppression_html = []
    for item in report["cooling_suppressions"]:
        suppression_html.append(
            "<tr><td>{minute}</td><td>{workload}</td><td>{code}</td><td>{trigger}</td>"
            "<td>{attempted}</td><td>{final}</td><td>{message}</td></tr>".format(
                minute=item["minute_index"],
                workload=html.escape(item["workload_id"]),
                code=html.escape(item["code"]),
                trigger=item["trigger_minute"],
                attempted=item["attempted_replicas"],
                final=item["final_replicas"],
                message=html.escape(item["message"]),
            )
        )
    recovery_html = []
    for item in report["recovery_events"]:
        recovery_html.append(
            "<li><strong>{event}</strong> minute_index={minute_index} minute={minute} "
            "<code>{detail}</code></li>".format(
                event=html.escape(item["event_type"]),
                minute_index=item["minute_index"],
                minute=item["minute"],
                detail=html.escape(json.dumps(item["detail"], ensure_ascii=False, sort_keys=True)),
            )
        )
    print_ready = "yes" if report["exit_report_available"] else "no"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Elastic Scaling Diagnostic Workbench</title>
  <style>
    :root {{ color-scheme: dark; --bg:#10151d; --panel:#18212c; --line:#354150; --text:#e8eef6; --muted:#9fb0c3; --accent:#57c2a6; --warn:#f0b45c; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Segoe UI, Arial, sans-serif; }}
    header, main {{ max-width:1280px; margin:auto; padding:20px; }}
    h1 {{ margin:0 0 6px; font-size:26px; }} h2 {{ margin-top:28px; font-size:18px; }}
    .muted {{ color:var(--muted); }} .pill {{ display:inline-block; padding:3px 9px; border:1px solid var(--line); border-radius:999px; margin:3px 5px 3px 0; }}
    .good {{ border-color:var(--accent); color:var(--accent); }} .warn {{ border-color:var(--warn); color:var(--warn); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }}
    .value {{ font-size:22px; font-weight:600; }} table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
    th, td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); position:sticky; top:0; background:#1d2835; }}
    .table-wrap {{ max-height:620px; overflow:auto; border:1px solid var(--line); border-radius:10px; }}
    code {{ color:#b7ddff; }} svg {{ width:100%; height:auto; background:var(--panel); border:1px solid var(--line); border-radius:10px; }}
    .axis {{ stroke:#516072; stroke-width:1; }} .budget-line {{ stroke:var(--warn); stroke-dasharray:7 5; }}
    .budget-path {{ fill:none; stroke:var(--accent); stroke-width:2.5; }} circle {{ fill:var(--accent); }}
    text {{ fill:var(--muted); font-size:11px; }} .budget-label {{ fill:var(--warn); }}
  </style>
</head>
<body>
<header>
  <h1>Elastic Scaling Diagnostic Workbench</h1>
  <div class="muted">Run <code>{html.escape(report['run']['run_id'])}</code> · declaration <code>{html.escape(report['run']['declaration_version'])}</code> · ledger <code>{html.escape(report['ledger_digest'])}</code></div>
  <div><span class="pill {'good' if report['exit_report_available'] else 'warn'}">{escaped_codes}</span><span class="pill">path {html.escape(report['run']['path'])}</span><span class="pill">exit report {print_ready}</span></div>
</header>
<main>
  <section class="grid">
    <div class="card"><div class="muted">Committed minutes</div><div class="value">{report['run']['committed_minutes']} / {report['run']['minute_count']}</div></div>
    <div class="card"><div class="muted">Global budget</div><div class="value">{report['run']['budget']}</div></div>
    <div class="card"><div class="muted">Peak memory estimate</div><div class="value">{report['run']['peak_memory_estimate_bytes']} B</div></div>
    <div class="card"><div class="muted">Cooldown suppressions</div><div class="value">{report['summary']['cooling_suppressions']}</div></div>
    <div class="card"><div class="muted">Budget clips</div><div class="value">{report['summary']['budget_clipping_events']}</div></div>
  </section>
  <p class="muted">{html.escape(report['diagnostic_note'])}. Path rule: {html.escape(report['path_selection']['rule'])}; selected {html.escape(report['path_selection']['selected'])}; window capacity {report['run']['window_capacity']}.</p>
  <h2>Budget occupancy</h2>
  {_svg_budget_chart(report)}
  <h2>Minute decisions</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Minute</th><th>Workload</th><th>Req</th><th>Latency</th><th>Errors</th><th>Expected</th><th>Budget target</th><th>Final</th><th>Decision</th><th>Budget used</th><th>Clip</th><th>Cooldown / evidence</th></tr></thead>
    <tbody>{''.join(row_html)}</tbody>
  </table></div>
  <h2>Cooling suppressions</h2>
  <div class="table-wrap"><table><thead><tr><th>Minute</th><th>Workload</th><th>Code</th><th>Trigger</th><th>Attempted</th><th>Final</th><th>Message</th></tr></thead><tbody>{''.join(suppression_html) or '<tr><td colspan="7">None</td></tr>'}</tbody></table></div>
  <h2>Recovery events</h2>
  <ul>{''.join(recovery_html) or '<li>None</li>'}</ul>
</main>
</body>
</html>
"""


def write_report(
    ledger: Ledger,
    run_id: str,
    *,
    json_path: str | Path | None = None,
    html_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_report(ledger, run_id)
    if json_path:
        write_json_atomic(json_path, report)
    if html_path:
        write_text_atomic(html_path, render_html(report))
    if snapshot_path:
        write_json_atomic(
            snapshot_path,
            {
                "snapshot_version": "1.0",
                "sqlite_path": str(ledger.path),
                "report": report,
            },
        )
    return report
