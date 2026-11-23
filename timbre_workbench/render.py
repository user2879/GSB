"""Dependency-free static HTML diagnostics dashboard."""

from __future__ import annotations

import html
import json
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value))


def _bar(value: float, maximum: float = 1.0) -> str:
    width = max(0.0, min(100.0, (float(value) / maximum) * 100.0 if maximum else 0.0))
    return f'<span class="bar"><i style="width:{width:.2f}%"></i></span>'


def render_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    diagnostics = report.get("diagnostics", {})
    cache_events = diagnostics.get("cache_events", {})
    quantization = report.get("quantization", {})
    dimensions = [row["name"] for row in quantization.get("dimensions", [])]
    failures = report.get("failures", [])
    conflicts = report.get("conflicts", [])

    segment_rows = []
    for segment in report.get("segments", []):
        assignment = segment.get("assignment", {})
        feature_values = segment.get("features", {})
        dimension_records = segment.get("description", {}).get("dimension_records", [])
        quant_text = "<br>".join(
            f"{_e(row['dimension'])}: {_e(row['quantized_integer'])} / {_e(row['tier_bits'])}b "
            f"step {_e(row['step'])} residual {_e(row['residual'])}"
            for row in dimension_records
        )
        segment_rows.append(
            "<tr>"
            f"<td>{_e(segment['segment_index'])}</td>"
            f"<td><strong>{_e(segment['file'])}</strong><br><small>{_e(segment['segment_id'])}</small></td>"
            f"<td>{_e(segment['start_sample'])}–{_e(segment['end_sample'])}<br><small>{_e(segment['sample_count'])} samples</small></td>"
            f"<td>{_e(segment['annotation_speaker_id'])}</td>"
            f"<td>{_e(feature_values.get('valid_frame_count'))} frames<br><small>F0 {_e(feature_values.get('f0_mean_hz'))} Hz · rate {_e(feature_values.get('speech_rate_sps'))}/s</small></td>"
            f"<td>{quant_text}</td>"
            f"<td><strong>{_e(assignment.get('speaker_id'))}</strong><br><small>sim {_e(assignment.get('similarity_to_previous'))} · {_e(assignment.get('decision'))}</small></td>"
            f"<td><code>{_e(segment['description_key'][:16])}…</code><br><code>{_e(segment['assignment_key'][:16])}…</code></td>"
            "</tr>"
        )

    speaker_cards = []
    for speaker in report.get("speakers", []):
        vector = speaker.get("vector", [])
        bars = "".join(
            f'<div class="vector-row"><span>{_e(name)}</span>{_bar(value)}<b>{float(value):.6f}</b></div>'
            for name, value in zip(dimensions, vector)
        )
        participants = ", ".join(
            f"{row['file']} / {row['segment_id']} (w={row['weight']})" for row in speaker.get("participants", [])
        )
        speaker_cards.append(
            '<article class="speaker-card">'
            f"<h3>{_e(speaker['speaker_id'])}</h3>"
            f"<p><span class=\"tag\">{_e(speaker['segment_count'])} segments</span>"
            f"<span class=\"tag\">{_e(speaker['weight_denominator'])} frame weight</span>"
            + ('<span class="tag warn">degenerate</span>' if speaker.get('aggregation_degenerate') else "")
            + "</p>"
            f"<div class=\"vector\">{bars}</div>"
            f"<small>{_e(participants)}</small>"
            f"<p><code>{_e(speaker['aggregation_key'][:24])}…</code></p>"
            "</article>"
        )

    quant_rows = "".join(
        "<tr>"
        f"<td>{_e(row['name'])}</td>"
        f"<td>{_e(row['source'])}</td>"
        f"<td>{float(row['minimum']):.9f} … {float(row['maximum']):.9f}</td>"
        f"<td>{_e(row['bits'])} bits / {_e(row['levels'])} levels</td>"
        f"<td>{_e(row['step'])}</td>"
        f"<td>{_e(row['tier_mode'])}</td>"
        "</tr>"
        for row in quantization.get("dimensions", [])
    )
    failure_rows = "".join(
        "<tr>"
        f"<td><code>{_e(row.get('error_code'))}</code></td>"
        f"<td>{_e(row.get('file'))}</td>"
        f"<td>{_e(row.get('stage'))}</td>"
        f"<td>{_e(row.get('message'))}</td>"
        "</tr>"
        for row in failures
    )
    conflict_rows = "".join(
        "<tr>"
        f"<td>{_e(row['file'])}</td>"
        f"<td>{_e(row['previous_segment_id'])}</td>"
        f"<td>{_e(row['segment_id'])}</td>"
        f"<td>{_e(row['similarity'])}</td>"
        f"<td>{_e(row['predicted_same'])}</td>"
        f"<td>{_e(row['annotated_same'])}</td>"
        f"<td>{_e(row['resolution'])}</td>"
        "</tr>"
        for row in conflicts
    )
    snapshot_warnings = diagnostics.get("snapshot_warnings", [])
    warning_html = "".join(
        f'<div class="alert"><code>{_e(row.get("error_code"))}</code> {_e(row.get("message"))}</div>'
        for row in snapshot_warnings
    )
    scripts = """
    <script>
    const search = document.getElementById('segment-search');
    if (search) {
      search.addEventListener('input', () => {
        const needle = search.value.toLowerCase();
        document.querySelectorAll('#segments tbody tr').forEach(row => {
          row.style.display = row.innerText.toLowerCase().includes(needle) ? '' : 'none';
        });
      });
    }
    </script>
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>说话人音色分析诊断工作台</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a31; --panel2:#17213d; --ink:#e8eefc; --muted:#91a1c7; --line:#273556; --accent:#67e8f9; --good:#34d399; --warn:#fbbf24; --bad:#fb7185; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at 20% -10%,#172554 0,#0b1020 38%,#070b16 100%); color:var(--ink); font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif; }}
main {{ max-width:1480px; margin:0 auto; padding:32px; }}
h1,h2,h3 {{ margin:0 0 12px; }}
h1 {{ font-size:28px; letter-spacing:-.02em; }}
h2 {{ font-size:19px; margin-top:28px; }}
p {{ margin:6px 0; }}
.muted, small {{ color:var(--muted); }}
.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:18px; }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:12px; margin:22px 0; }}
.card,.speaker-card,.panel {{ background:linear-gradient(160deg,var(--panel),var(--panel2)); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 12px 32px #0004; }}
.card span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.card strong {{ display:block; font-size:28px; margin-top:4px; }}
.speakers {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }}
.speaker-card h3 {{ color:var(--accent); }}
.tag {{ display:inline-block; background:#24345e; border-radius:999px; padding:2px 9px; margin:2px 5px 2px 0; font-size:12px; }}
.tag.warn {{ background:#4a3211; color:var(--warn); }}
.vector-row {{ display:grid; grid-template-columns:110px 1fr 92px; gap:8px; align-items:center; margin:7px 0; }}
.vector-row b {{ font-variant-numeric:tabular-nums; text-align:right; font-weight:500; }}
.bar {{ display:block; height:8px; background:#0a1020; border-radius:999px; overflow:hidden; border:1px solid #26365c; }}
.bar i {{ display:block; height:100%; background:linear-gradient(90deg,#22d3ee,#818cf8); }}
table {{ width:100%; border-collapse:collapse; background:#0f172c; border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
th,td {{ text-align:left; vertical-align:top; padding:9px 10px; border-bottom:1px solid var(--line); }}
th {{ color:#b9c8ea; background:#16203a; font-size:12px; letter-spacing:.04em; text-transform:uppercase; position:sticky; top:0; }}
tr:hover td {{ background:#131d36; }}
code {{ color:#a5f3fc; font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; word-break:break-all; }}
.alert {{ background:#3d2510; color:#fde68a; border:1px solid #7c4a13; padding:12px 14px; border-radius:12px; margin:12px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.toolbar {{ display:flex; gap:10px; align-items:center; margin:18px 0 8px; }}
input {{ width:min(460px,100%); background:#0e1730; color:var(--ink); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.scroll {{ overflow:auto; max-height:650px; border-radius:14px; }}
@media (max-width:900px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .grid2 {{ grid-template-columns:1fr; }} main {{ padding:18px; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <div class="muted">LOCAL DETERMINISTIC ANALYSIS · {_e(report.get('schema_version'))}</div>
      <h1>说话人音色分析诊断工作台</h1>
      <p class="muted">命令游标 {_e(report.get('command_cursor'))} · 全局版本 {_e(report.get('global_version'))} · 配置 {_e(report.get('effective_config_hash', '')[:18])}…</p>
    </div>
    <div><code>canonical {_e(report.get('canonical_digest', '')[:24])}…</code></div>
  </section>
  {warning_html}
  <section class="cards">
    <div class="card"><span>已完成段</span><strong>{_e(summary.get('completed_segments'))}</strong></div>
    <div class="card"><span>说话人</span><strong>{_e(summary.get('speaker_count'))}</strong></div>
    <div class="card"><span>向量维度</span><strong>{_e(summary.get('vector_dimension_count'))}</strong></div>
    <div class="card"><span>复用产物</span><strong>{_e(cache_events.get('reused_count'))}</strong></div>
    <div class="card"><span>冲突 / 失败</span><strong>{_e(summary.get('conflict_count'))} / {_e(summary.get('failure_count'))}</strong></div>
  </section>

  <h2>说话人级向量</h2>
  <section class="speakers">{''.join(speaker_cards) or '<p class="muted">尚无聚合结果</p>'}</section>

  <h2>逐维自适应量化</h2>
  <div class="scroll"><table><thead><tr><th>维度</th><th>来源</th><th>动态范围</th><th>位宽/档位</th><th>步长</th><th>选择模式</th></tr></thead><tbody>{quant_rows}</tbody></table></div>

  <h2>段级诊断</h2>
  <div class="toolbar"><input id="segment-search" placeholder="过滤文件、段、说话人、特征…"></div>
  <div class="scroll"><table id="segments"><thead><tr><th>#</th><th>文件 / 段</th><th>采样范围</th><th>标注</th><th>有效帧</th><th>量化（整数 / 档位 / 步长 / 残差）</th><th>归属</th><th>缓存键</th></tr></thead><tbody>{''.join(segment_rows)}</tbody></table></div>

  <section class="grid2">
    <div>
      <h2>归属冲突</h2>
      <div class="scroll"><table><thead><tr><th>文件</th><th>前段</th><th>当前段</th><th>相似度</th><th>预测同人</th><th>标注同人</th><th>处理</th></tr></thead><tbody>{conflict_rows}</tbody></table></div>
    </div>
    <div>
      <h2>失败清单</h2>
      <div class="scroll"><table><thead><tr><th>错误</th><th>文件</th><th>阶段</th><th>信息</th></tr></thead><tbody>{failure_rows}</tbody></table></div>
    </div>
  </section>

  <h2>增量与并发诊断</h2>
  <section class="panel">
    <p>失效 {_e(cache_events.get('invalidated_count'))} · 重算 {_e(cache_events.get('recomputed_count'))} · 复用 {_e(cache_events.get('reused_count'))} · 强制重建 {_e(cache_events.get('forced_rebuild'))}</p>
    <p class="muted">受影响段：{_e(', '.join(summary.get('affected_segment_ids', [])) or '—')}</p>
    <p class="muted">线程峰值 {_e(diagnostics.get('threading', {}).get('peak_workers_observed'))} / 上限 {_e(diagnostics.get('threading', {}).get('max_workers_configured'))} · 稳定排序 {_e(diagnostics.get('threading', {}).get('stable_sort_keys'))}</p>
    <pre><code>{_e(json.dumps(diagnostics.get('task_orders', {}), ensure_ascii=False, indent=2))}</code></pre>
  </section>
</main>
{scripts}
</body>
</html>
"""

