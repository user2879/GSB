"""JSON diagnostic report and a dependency-free HTML visualization."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .util import atomic_write_text, canonical_json


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return html.escape(str(value))


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def render_html(report: dict[str, Any]) -> str:
    stable = report.get("stable_result", {})
    summary = stable.get("summary", {})
    manifest = report.get("snapshot_manifest") or {}
    failures = report.get("failures", [])
    ledger = report.get("invalidation_reuse_ledger", [])
    layer_errors = summary.get("layer_errors", [])
    command_responses = report.get("command_responses", [])
    validation = report.get("validation", {})

    cards = [
        ("完成层", f"{summary.get('completed_layers', 0)}/{summary.get('total_layers', 0)}"),
        ("最大绝对误差", _fmt(summary.get("max_abs_error"))),
        ("均方误差", _fmt(summary.get("mse"))),
        ("失败项", str(len(failures))),
        ("快照修订", _fmt(manifest.get("revision"))),
        ("增量=全量", _fmt(validation.get("incremental_equals_full"))),
        ("快照恢复一致", _fmt(validation.get("snapshot_resume_equals_full"))),
        ("预算内响应", _fmt(validation.get("all_commands_within_budget"))),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )

    layer_max = max(
        [float(row.get("max_abs_error") or 0.0) for row in layer_errors] or [1.0]
    )
    layer_rows = []
    for row in layer_errors:
        width = 100.0 * float(row.get("max_abs_error") or 0.0) / layer_max if layer_max else 0.0
        layer_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('layer')))}</td>"
            f"<td>{_fmt(row.get('max_abs_error'))}</td>"
            f"<td>{_fmt(row.get('mse'))}</td>"
            f"<td>{_fmt(row.get('corrected_max_abs_error'))}</td>"
            f"<td>{_fmt(row.get('corrected_mse'))}</td>"
            f"<td>{row.get('sample_count')}</td>"
            f'<td><div class="bar"><i style="width:{width:.2f}%"></i></div></td>'
            "</tr>"
        )

    command_rows = []
    for row in command_responses:
        codes = ", ".join(row.get("codes", [])) or "ok"
        command_rows.append(
            "<tr>"
            f"<td>{row.get('command_index')}</td>"
            f"<td>{html.escape(str(row.get('op')))}</td>"
            f"<td>{html.escape(str(row.get('status')))}</td>"
            f"<td>{html.escape(codes)}</td>"
            f"<td>{row.get('result_generation')}</td>"
            f"<td>{_fmt(row.get('completed_layers'))}</td>"
            f"<td>{_fmt(row.get('elapsed_ms'), 4)} ms</td>"
            f"<td>{_fmt(row.get('within_budget'))}</td>"
            "</tr>"
        )

    failure_rows = []
    for issue in failures:
        failure_rows.append(
            "<tr>"
            f"<td>{html.escape(str(issue.get('code')))}</td>"
            f"<td>{_fmt(issue.get('command_index'))}</td>"
            f"<td>{_fmt(issue.get('line_number'))}</td>"
            f"<td>{html.escape(str(issue.get('layer') or '—'))}</td>"
            f"<td>{html.escape(str(issue.get('message')))}</td>"
            "</tr>"
        )

    ledger_rows = []
    for entry in ledger[-80:]:
        stages = entry.get("stages", {})
        stage_text = ", ".join(f"{key}:{value}" for key, value in stages.items())
        ledger_rows.append(
            "<tr>"
            f"<td>{entry.get('command_index')}</td>"
            f"<td>{html.escape(str(entry.get('layer')))}</td>"
            f"<td>{html.escape(str(entry.get('action')))}</td>"
            f"<td>{html.escape(str(entry.get('reason')))}</td>"
            f"<td>{html.escape(stage_text)}</td>"
            "</tr>"
        )

    stage_rows = []
    for stage, version in sorted((manifest.get("artifact_versions") or {}).items()):
        stage_rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage))}</td>"
            f"<td><code>{html.escape(str(version))}</code></td>"
            "</tr>"
        )

    plan = stable.get("scheduling", {})
    batch_text = ", ".join("[" + ", ".join(batch) + "]" for batch in plan.get("batches", []))
    payload_json = html.escape(canonical_json(report))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Compression Workbench</title>
<style>
:root {{ color-scheme: light dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#2f81f7; --ok:#3fb950; --warn:#d29922; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:28px 32px 12px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:.2px; }}
h2 {{ margin:28px 0 12px; font-size:18px; }}
p {{ color:var(--muted); margin:0; }}
main {{ padding:20px 32px 48px; max-width:1400px; margin:auto; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:15px 16px; }}
.card span {{ color:var(--muted); display:block; font-size:12px; text-transform:uppercase; letter-spacing:.6px; }}
.card strong {{ display:block; margin-top:6px; font-size:22px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.4px; }}
tr:last-child td {{ border-bottom:0; }}
.bar {{ height:8px; min-width:120px; background:#21262d; border-radius:999px; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:linear-gradient(90deg,#2f81f7,#58a6ff); }}
.toolbar {{ display:flex; gap:10px; margin-bottom:10px; }}
input {{ flex:1; max-width:340px; background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:9px 11px; }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
pre {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; overflow:auto; max-height:420px; }}
details {{ margin-top:12px; }}
summary {{ cursor:pointer; color:var(--accent); }}
.ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }}
</style>
</head>
<body>
<header>
  <h1>模型压缩与部署诊断工作台</h1>
  <p>逐层分位标定 · 定点重量化 · 饱和截断 · 软标签校正 · 指纹级复用 · 原子快照</p>
</header>
<main>
  <div class="grid">{card_html}</div>

  <h2>逐层误差归因</h2>
  <div class="toolbar"><input id="layerFilter" placeholder="过滤层名，例如 layer.2"></div>
  <table id="layerTable">
    <thead><tr><th>层</th><th>最大绝对误差</th><th>MSE</th><th>校正后最大误差</th><th>校正后 MSE</th><th>样本</th><th>误差分布</th></tr></thead>
    <tbody>{''.join(layer_rows) or '<tr><td colspan="7">暂无完成层</td></tr>'}</tbody>
  </table>

  <h2>命令与延迟预算</h2>
  <table>
    <thead><tr><th>#</th><th>命令</th><th>状态</th><th>代码</th><th>结果代次</th><th>完成层</th><th>耗时</th><th>预算内</th></tr></thead>
    <tbody>{''.join(command_rows) or '<tr><td colspan="8">无命令</td></tr>'}</tbody>
  </table>

  <h2>失效 / 复用账本</h2>
  <table>
    <thead><tr><th>命令</th><th>层</th><th>动作</th><th>原因</th><th>阶段</th></tr></thead>
    <tbody>{''.join(ledger_rows) or '<tr><td colspan="5">空</td></tr>'}</tbody>
  </table>

  <h2>失败明细</h2>
  <table>
    <thead><tr><th>代码</th><th>命令</th><th>行</th><th>层</th><th>消息</th></tr></thead>
    <tbody>{''.join(failure_rows) or '<tr><td colspan="5">无失败</td></tr>'}</tbody>
  </table>

  <h2>确定性调度计划</h2>
  <pre>{html.escape(batch_text or '无')}
点预算: {_fmt(plan.get('ledger', [{}])[0].get('remaining_before') if plan.get('ledger') else None)} · 待算: {html.escape(str(plan.get('pending_layers', [])))}</pre>

  <h2>快照阶段版本</h2>
  <table>
    <thead><tr><th>阶段</th><th>产物版本</th></tr></thead>
    <tbody>{''.join(stage_rows) or '<tr><td colspan="2">尚无快照</td></tr>'}</tbody>
  </table>

  <details>
    <summary>查看完整 JSON 报告</summary>
    <pre>{payload_json}</pre>
  </details>
</main>
<script>
const filter = document.getElementById('layerFilter');
filter.addEventListener('input', () => {{
  const needle = filter.value.trim().toLowerCase();
  document.querySelectorAll('#layerTable tbody tr').forEach((row) => {{
    row.style.display = !needle || row.textContent.toLowerCase().includes(needle) ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""


def write_html_report(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_text(path, render_html(report))
