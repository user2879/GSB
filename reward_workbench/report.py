"""诊断报告：JSON 明细 + 自包含 HTML 可视化（无外部依赖）。"""
from __future__ import annotations

import html
import json
from typing import Any


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)


def _bar_chart(title: str, rows: list[tuple[str, int]]) -> str:
    if not rows:
        return f"<h3>{html.escape(title)}</h3><p>无数据</p>"
    max_v = max(v for _, v in rows) or 1
    width = 320
    bars = []
    for label, value in rows:
        w = int(width * value / max_v)
        bars.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(str(label))}</span>'
            f'<span class="bar" style="width:{w}px"></span>'
            f'<span class="bar-val">{value}</span></div>'
        )
    return f'<h3>{html.escape(title)}</h3><div class="bars">{"".join(bars)}</div>'


def _diff_sparkline(diffs: list[int]) -> str:
    if not diffs:
        return "<p>无差值</p>"
    width, height, pad = 520, 140, 20
    lo, hi = min(diffs), max(diffs)
    span = (hi - lo) or 1
    n = len(diffs)
    points = []
    for i, d in enumerate(diffs):
        x = pad + (width - 2 * pad) * (i / max(1, n - 1))
        y = height - pad - (height - 2 * pad) * ((d - lo) / span)
        points.append(f"{x:.1f},{y:.1f}")
    zero_y = height - pad - (height - 2 * pad) * ((0 - lo) / span)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" role="img">'
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width-pad}" y2="{zero_y:.1f}" '
        f'class="zero"/>'
        f'<polyline points="{" ".join(points)}" class="line"/>'
        f"</svg>"
    )


def render_html(payload: dict[str, Any]) -> str:
    results = payload.get("canonical_results", {})
    boundaries = results.get("boundaries", [])
    summaries = results.get("summaries", [])
    scores = results.get("scores", [])
    failures = payload.get("failures", [])
    budget = payload.get("budget_ledger", {})

    label_counts = {"stable": 0, "critical": 0, "conflict": 0}
    for bd in boundaries:
        label_counts[bd["label"]] = label_counts.get(bd["label"], 0) + 1
    label_chart = _bar_chart(
        "边界判定分布",
        [("稳定 stable", label_counts["stable"]),
         ("临界 critical", label_counts["critical"]),
         ("冲突 conflict", label_counts["conflict"])],
    )

    batch_counts: dict[str, int] = {}
    for g in summaries:
        batch_counts[g["batch_id"]] = batch_counts.get(g["batch_id"], 0) + g["total"]
    batch_chart = _bar_chart(
        "各批次处理对数", sorted(batch_counts.items()))

    diffs = [s["diff"] for s in scores]
    spark = _diff_sparkline(diffs)

    summary_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            g["batch_id"], g["annotator_id"], g["time_partition"], g["total"],
            g["stable"], g["critical"], g["conflict"],
            g["stable_ratio"], g["critical_ratio"], g["conflict_ratio"],
            g["mean_diff_str"],
        )) + "</tr>"
        for g in summaries
    )
    boundary_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            b["pair_id"], b["label"], b["diff"], b["threshold"],
            b["at_threshold"], b["tie_rank"],
        )) + "</tr>"
        for b in payload.get("boundary_list", boundaries)
    )
    failure_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            f["sort_key"], f["code"], f["stage"], f["message"], f["injection_id"],
        )) + "</tr>"
        for f in failures
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>LLM 偏好对齐奖励建模诊断工作台</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; border-left: 4px solid #2563eb; padding-left: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #d1d5db; padding: 4px 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.bar-row {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
.bar-label {{ width: 130px; }} .bar {{ height: 14px; background: #2563eb; margin: 0 8px; border-radius: 3px; }}
.bar-val {{ font-variant-numeric: tabular-nums; }}
.spark {{ border: 1px solid #e5e7eb; border-radius: 6px; background: #fafafa; }}
.line {{ fill: none; stroke: #16a34a; stroke-width: 2; }} .zero {{ stroke: #9ca3af; stroke-dasharray: 4 4; }}
.kpi {{ display: inline-block; background: #eff6ff; border-radius: 8px; padding: 10px 16px; margin-right: 12px; }}
.kpi b {{ font-size: 18px; }}
</style></head><body>
<h1>LLM 偏好对齐与奖励建模 · 可视化诊断工作台</h1>
<div>
  <span class="kpi">偏好对总数 <b>{len(scores)}</b></span>
  <span class="kpi">已消耗预算 <b>{budget.get('consumed', 0)}</b> / {budget.get('global_budget', 0)}</span>
  <span class="kpi">剩余预算 <b>{budget.get('remaining', 0)}</b></span>
  <span class="kpi">失败条目 <b>{len(failures)}</b></span>
</div>
<h2>分布图</h2>
{label_chart}
{batch_chart}
<h2>奖励差值走势（定点整数）</h2>
{spark}
<h2>批次 × 标注者 × 时间分区 汇总</h2>
<table><thead><tr><th>批次</th><th>标注者</th><th>时间分区</th><th>总数</th>
<th>稳定</th><th>临界</th><th>冲突</th><th>稳定比例</th><th>临界比例</th>
<th>冲突比例</th><th>平均差值</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>边界样本清单（可审计）</h2>
<table><thead><tr><th>对 ID</th><th>判定</th><th>差值</th><th>阈值</th>
<th>恰等于阈值</th><th>字典序决断序号</th></tr></thead><tbody>{boundary_rows}</tbody></table>
<h2>失败清单（业务码）</h2>
<table><thead><tr><th>排序键</th><th>业务码</th><th>阶段</th><th>信息</th>
<th>注入编号</th></tr></thead><tbody>{failure_rows or '<tr><td colspan=5>无</td></tr>'}</tbody></table>
</body></html>"""


def write_html(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_html(payload))
