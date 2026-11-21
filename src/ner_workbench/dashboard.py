from __future__ import annotations

import html
import json
from typing import Any, Iterable


TYPE_COLORS = {
    "人物": ("#fee2e2", "#991b1b"),
    "地点": ("#dcfce7", "#166534"),
    "机构": ("#dbeafe", "#1e40af"),
    "时间": ("#fef3c7", "#92400e"),
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _card(label: str, value: object, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{_esc(label)}</div>'
        f'<div class="metric-value">{_esc(value)}</div>'
        f'<div class="metric-note">{_esc(note)}</div>'
        "</div>"
    )


def _highlight_chars(sample: dict[str, Any]) -> str:
    chars = list(sample.get("chars", []))
    spans: dict[int, dict[str, Any]] = {}
    for entity in sample.get("entities", []):
        start = int(entity["start"])
        end = int(entity["end"])
        for position in range(start, min(end + 1, len(chars))):
            spans[position] = entity
    output: list[str] = []
    index = 0
    while index < len(chars):
        entity = spans.get(index)
        if entity is None:
            output.append(f"<span>{_esc(chars[index])}</span>")
            index += 1
            continue
        end = int(entity["end"])
        text = "".join(chars[index : end + 1])
        entity_type = str(entity.get("type", ""))
        foreground, background = TYPE_COLORS.get(entity_type, ("#ede9fe", "#5b21b6"))
        output.append(
            f'<mark class="entity" style="color:{foreground};background:{background}" '
            f'title="{_esc(entity_type)} [{index},{end}]">{_esc(text)}</mark>'
        )
        index = end + 1
    return "".join(output)


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]], empty: str = "无记录") -> str:
    row_list = list(rows)
    if not row_list:
        return f'<div class="empty">{_esc(empty)}</div>'
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in row_list
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_dashboard(report: dict[str, Any]) -> str:
    det = report.get("deterministic", {})
    summary = det.get("summary", {})
    cfg = report.get("configuration", {}).get("normalized", {})
    model = report.get("model", {})
    snapshot = det.get("snapshot", {})
    samples = det.get("samples", [])
    failures = det.get("failures", [])
    injections = det.get("injections", {})
    cache = det.get("cache", {})
    collisions = det.get("feature_hash_collisions", {})
    budget = det.get("budget", {})
    diagnostics = report.get("diagnostics", {})

    total_budget = max(1, int(budget.get("global_char_budget", 0)))
    consumed_budget = int(budget.get("consumed_char_budget", 0))
    budget_percent = min(100.0, 100.0 * consumed_budget / total_budget)

    stage_totals: dict[str, float] = {}
    for timings in diagnostics.get("stage_timings_ms", {}).values():
        for stage in ("emission", "decode", "merge"):
            stage_totals[stage] = stage_totals.get(stage, 0.0) + float(timings.get(stage, 0.0))
    max_stage = max(stage_totals.values(), default=1.0) or 1.0
    stage_bars = "".join(
        f'<div class="bar-row"><span>{_esc(stage)}</span>'
        f'<div class="bar"><i style="width:{100.0 * value / max_stage:.2f}%"></i></div>'
        f'<b>{value:.3f} ms</b></div>'
        for stage, value in stage_totals.items()
    )

    type_legend = "".join(
        f'<span class="legend"><i style="background:{background};color:{foreground}"></i>{_esc(entity_type)}</span>'
        for entity_type, (foreground, background) in TYPE_COLORS.items()
    )

    sample_cards = []
    for sample in samples:
        repairs = sample.get("repairs", [])
        repair_html = ""
        if repairs:
            repair_html = '<details class="repairs"><summary>修复记录 ' + _esc(len(repairs)) + '</summary><ul>' + "".join(
                f'<li><code>{_esc(item.get("code"))}</code> {_esc(item.get("reason"))}</li>' for item in repairs
            ) + "</ul></details>"
        entity_badges = "".join(
            f'<span class="entity-badge" style="color:{TYPE_COLORS.get(str(entity.get("type")), ("#5b21b6", "#ede9fe"))[0]};'
            f'background:{TYPE_COLORS.get(str(entity.get("type")), ("#5b21b6", "#ede9fe"))[1]}">'
            f'{_esc(entity.get("type"))} · {int(entity.get("start"))}-{int(entity.get("end"))}</span>'
            for entity in sample.get("entities", [])
        )
        sample_cards.append(
            '<article class="sample-card" data-sample="' + _esc(sample.get("sample_id")) + '">'
            '<div class="sample-head">'
            f'<h3>{_esc(sample.get("sample_id"))}</h3>'
            f'<span class="score">path {_esc(sample.get("path_score"))}</span>'
            "</div>"
            f'<div class="text-line">{_highlight_chars(sample)}</div>'
            f'<div class="badges">{entity_badges or "<span class=empty>无实体</span>"}</div>'
            f'<div class="sample-meta">字符 {_esc(sample.get("char_count"))} · 打分 {_esc(sample.get("scored_chars"))} · '
            f'实体 {_esc(len(sample.get("entities", [])))} · 修复 {_esc(len(repairs))}</div>'
            f"{repair_html}</article>"
        )

    failure_rows = [
        (
            item.get("sample_id"),
            item.get("stage"),
            item.get("code"),
            item.get("exception_type"),
            item.get("injection_id") or "-",
            item.get("message"),
        )
        for item in failures
    ]
    configured_rows = [
        (item.get("injection_id"), item.get("sample_id"), item.get("stage"), item.get("kind"), "configured")
        for item in injections.get("configured", [])
    ]
    hit_rows = [
        (
            item.get("injection_id"),
            item.get("sample_id"),
            item.get("stage"),
            item.get("kind"),
            item.get("code"),
            item.get("detail"),
        )
        for item in injections.get("hits", [])
    ]
    eviction_rows = [
        (
            item.get("sequence"),
            item.get("evicted_block"),
            item.get("incoming_block"),
            item.get("reason"),
        )
        for item in cache.get("evictions", [])
    ]
    recompute_rows = [
        (
            item.get("sequence"),
            item.get("block_key"),
            item.get("checksum"),
            item.get("matches_first_compute"),
        )
        for item in cache.get("recompilation_log", [])
    ]
    timing_rows = [
        (
            sample_id,
            round(float(timings.get("emission", 0.0)), 6),
            round(float(timings.get("decode", 0.0)), 6),
            round(float(timings.get("merge", 0.0)), 6),
            round(float(timings.get("total", 0.0)), 6),
        )
        for sample_id, timings in sorted(diagnostics.get("stage_timings_ms", {}).items())
    ]
    collision_rows = [
        (item.get("bucket"), item.get("key_count"), ", ".join(item.get("keys", [])[:6]))
        for item in collisions.get("colliding_buckets", [])
    ]
    budget_rows = [
        (
            item.get("sequence"),
            item.get("sample_id"),
            item.get("requested_chars"),
            item.get("effective_quota"),
            item.get("allocated"),
            f'{item.get("before")} → {item.get("after")}',
            item.get("code") or "admitted",
        )
        for item in budget.get("deduction_sequence", [])
    ]

    embedded = json.dumps(report, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NER 确定性调度与预算诊断工作台</title>
<style>
:root {{ --ink:#172033; --muted:#65728a; --line:#dce3ef; --bg:#f4f7fc; --panel:#ffffff; --accent:#1357d8; --ok:#087f5b; --warn:#b45309; --bad:#b91c1c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at top left,#e7efff 0,#f4f7fc 32rem); color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif; }}
main {{ max-width:1440px; margin:0 auto; padding:28px 30px 60px; }}
header.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:22px; }}
h1 {{ margin:0; font-size:28px; letter-spacing:-.02em; }}
.subtitle {{ color:var(--muted); margin-top:7px; }}
.status {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:8px 14px; background:#e8f5ee; color:var(--ok); font-weight:700; white-space:nowrap; }}
.status.interrupted {{ background:#fff4e6; color:var(--warn); }}
.digest {{ font:12px ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--muted); margin-top:8px; overflow-wrap:anywhere; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; box-shadow:0 7px 24px rgba(37,63,112,.05); }}
.metric-label {{ color:var(--muted); font-size:12px; }}
.metric-value {{ font-size:25px; font-weight:760; letter-spacing:-.03em; }}
.metric-note {{ color:var(--muted); font-size:11px; min-height:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(12,1fr); gap:16px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 10px 30px rgba(37,63,112,.045); min-width:0; }}
.span-4 {{ grid-column:span 4; }} .span-6 {{ grid-column:span 6; }} .span-8 {{ grid-column:span 8; }} .span-12 {{ grid-column:span 12; }}
@media (max-width:1000px) {{ .span-4,.span-6,.span-8 {{ grid-column:span 12; }} }}
h2 {{ margin:0 0 12px; font-size:16px; }}
h3 {{ margin:0; font-size:15px; }}
.kv {{ display:grid; grid-template-columns:150px 1fr; gap:7px 12px; font-size:13px; }}
.kv dt {{ color:var(--muted); }} .kv dd {{ margin:0; overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
.progress {{ height:16px; border-radius:999px; background:#e8edf6; overflow:hidden; }}
.progress > i {{ display:block; height:100%; background:linear-gradient(90deg,#1357d8,#38bdf8); }}
.progress-caption {{ display:flex; justify-content:space-between; margin-top:8px; color:var(--muted); }}
.bar-row {{ display:grid; grid-template-columns:90px 1fr 100px; align-items:center; gap:10px; margin:9px 0; font-size:12px; }}
.bar {{ height:11px; border-radius:999px; background:#edf1f7; overflow:hidden; }} .bar i {{ display:block; height:100%; background:#2f6fe4; }}
.bar-row b {{ text-align:right; font-weight:600; }}
.legend {{ display:inline-flex; align-items:center; gap:6px; margin:0 12px 8px 0; color:var(--muted); }}
.legend i {{ width:12px; height:12px; border-radius:4px; display:inline-block; }}
.sample-tools {{ display:flex; gap:10px; margin-bottom:12px; }}
input[type=search] {{ flex:1; border:1px solid var(--line); border-radius:10px; padding:10px 12px; font:inherit; background:#fbfcff; }}
.samples {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:12px; }}
.sample-card {{ border:1px solid var(--line); border-radius:13px; padding:14px; background:#fcfdff; }}
.sample-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px; }}
.score {{ color:var(--muted); font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.text-line {{ font-size:20px; line-height:1.9; letter-spacing:.02em; padding:10px 12px; background:#fff; border:1px solid #edf1f7; border-radius:10px; }}
.text-line span {{ white-space:pre-wrap; }}
mark.entity {{ border-radius:5px; padding:2px 1px; font-weight:700; }}
.badges {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; }}
.entity-badge {{ border-radius:999px; padding:3px 8px; font-size:11px; font-weight:700; }}
.sample-meta {{ margin-top:9px; color:var(--muted); font-size:11px; }}
.repairs {{ margin-top:8px; color:var(--muted); font-size:12px; }} .repairs ul {{ margin:7px 0 0; padding-left:18px; }}
.table-wrap {{ overflow:auto; max-height:520px; border:1px solid #edf1f7; border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ text-align:left; padding:8px 9px; border-bottom:1px solid #eef2f8; vertical-align:top; }}
th {{ position:sticky; top:0; background:#f8faff; color:var(--muted); font-weight:650; z-index:1; }}
tr:last-child td {{ border-bottom:none; }}
code {{ font:11px ui-monospace,SFMono-Regular,Consolas,monospace; color:#334155; }}
.empty {{ color:var(--muted); padding:10px 2px; }}
.breach {{ color:var(--bad); font-weight:700; }}
.footer-note {{ margin-top:22px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<main>
<header class="hero">
  <div>
    <h1>NER 确定性调度与预算诊断工作台</h1>
    <div class="subtitle">字级哈希特征 · 定点发射打分 · 对数域 BIESO 维特比 · 双预算并行调度 · 原子快照</div>
    <div class="digest">deterministic digest: {_esc(report.get('deterministic_digest'))}</div>
  </div>
  <div class="status {'interrupted' if report.get('status') != 'complete' else ''}">● {_esc(report.get('status'))}<br><small>workers={_esc(cfg.get('max_workers'))}</small></div>
</header>

<section class="metrics">
  {_card("成功 / 有效样本", f"{summary.get('successful_sample_count', 0)} / {summary.get('valid_sample_count', 0)}")}
  {_card("失败终态", summary.get('failed_sample_count', 0), f"加载失败 {summary.get('load_failure_count', 0)}")}
  {_card("实体片段", summary.get('entity_count', 0), f"修复 {summary.get('repair_count', 0)}")}
  {_card("字级预算", f"{consumed_budget}/{summary.get('global_char_budget', 0)}", f"剩余 {summary.get('remaining_char_budget', 0)}")}
  {_card("缓存淘汰", summary.get('cache_eviction_count', 0), f"重算 {summary.get('cache_recomputation_count', 0)}")}
  {_card("注入命中", summary.get('injection_hit_count', 0), f"配置 {summary.get('injection_configured_count', 0)}")}
  {_card("哈希冲突", summary.get('collision_events', 0), f"桶数 {collisions.get('hash_buckets', 0)}")}
</section>

<section class="grid">
  <div class="panel span-6">
    <h2>全局字级预算</h2>
    <div class="progress"><i style="width:{budget_percent:.2f}%"></i></div>
    <div class="progress-caption"><span>已消耗 {consumed_budget}</span><span>预算 {total_budget}</span></div>
    <div class="kv" style="margin-top:16px">
      <dt>单样本配额</dt><dd>{_esc(cfg.get('per_sample_quota'))}</dd>
      <dt>处理单元</dt><dd>{_esc(cfg.get('processing_unit'))}</dd>
      <dt>字级批大小</dt><dd>{_esc(cfg.get('char_batch_size'))}</dd>
      <dt>定点 / 量化</dt><dd>{_esc(cfg.get('fixed_decimals'))} 位 / q{_esc(cfg.get('quantization_level'))}</dd>
    </div>
  </div>
  <div class="panel span-6">
    <h2>阶段耗时汇总</h2>
    {stage_bars or '<div class="empty">无阶段耗时</div>'}
    <div class="kv" style="margin-top:14px">
      <dt>总墙钟</dt><dd>{_esc(diagnostics.get('total_wall_ms'))} ms</dd>
      <dt>交互预算</dt><dd>{_esc(diagnostics.get('latency_budget_ms'))} ms</dd>
      <dt>超预算记录</dt><dd class="{'breach' if diagnostics.get('latency_breaches') else ''}">{_esc(len(diagnostics.get('latency_breaches', [])))}</dd>
    </div>
  </div>

  <div class="panel span-4">
    <h2>模型与哈希</h2>
    <dl class="kv">
      <dt>模型版本</dt><dd>{_esc(model.get('model_version'))}</dd>
      <dt>参数哈希</dt><dd>{_esc(model.get('model_hash'))}</dd>
      <dt>标签数</dt><dd>{_esc(model.get('label_count'))}</dd>
      <dt>维度 / 桶</dt><dd>{_esc(model.get('feature_dim'))} / {_esc(model.get('hash_buckets'))}</dd>
      <dt>特征命中</dt><dd>{_esc(collisions.get('total_feature_hits'))}</dd>
      <dt>占用桶</dt><dd>{_esc(collisions.get('occupied_buckets'))}</dd>
      <dt>冲突桶</dt><dd>{_esc(collisions.get('colliding_bucket_count'))}</dd>
    </dl>
  </div>
  <div class="panel span-4">
    <h2>配置与快照</h2>
    <dl class="kv">
      <dt>配置哈希</dt><dd>{_esc(report.get('configuration', {}).get('config_hash'))}</dd>
      <dt>语料哈希</dt><dd>{_esc(report.get('configuration', {}).get('corpus_hash'))}</dd>
      <dt>快照状态</dt><dd>{_esc(snapshot.get('status'))}</dd>
      <dt>恢复点</dt><dd>{_esc(snapshot.get('recovery_point_sample_id') or '无')}</dd>
      <dt>并发数</dt><dd>{_esc(report.get('run_metadata', {}).get('max_workers'))}</dd>
      <dt>全局版本</dt><dd>{_esc(report.get('snapshot_manifest', {}).get('global_version'))}</dd>
    </dl>
  </div>
  <div class="panel span-4">
    <h2>缓存驻留与重算</h2>
    <dl class="kv">
      <dt>缓存预算</dt><dd>{_esc(cfg.get('cache_blocks'))} blocks</dd>
      <dt>访问 / 命中</dt><dd>{_esc(cache.get('accesses'))} / {_esc(cache.get('hits'))}</dd>
      <dt>未命中</dt><dd>{_esc(cache.get('misses'))}</dd>
      <dt>重算</dt><dd>{_esc(cache.get('recomputations', cache.get('recomputation_count')))}</dd>
      <dt>校验不一致</dt><dd class="{'breach' if cache.get('checksum_mismatches') else ''}">{_esc(cache.get('checksum_mismatches'))}</dd>
      <dt>驻留块</dt><dd>{_esc(len(cache.get('resident_blocks', [])))}</dd>
    </dl>
  </div>

  <div class="panel span-12">
    <h2>实体高亮与逐样本诊断</h2>
    <div>{type_legend}</div>
    <div class="sample-tools"><input id="sample-filter" type="search" placeholder="筛选样本 ID、实体文本或修复码…"></div>
    <div class="samples" id="sample-list">{''.join(sample_cards) or '<div class="empty">无成功样本</div>'}</div>
  </div>

  <div class="panel span-12">
    <h2>失败清单</h2>
    {_table(("样本", "阶段", "业务码", "异常类型", "注入", "说明"), failure_rows)}
  </div>
  <div class="panel span-6">
    <h2>注入配置与命中</h2>
    {_table(("注入编号", "样本", "阶段", "类型", "状态/码", "说明"), hit_rows or configured_rows)}
  </div>
  <div class="panel span-6">
    <h2>预算扣减序列</h2>
    {_table(("序号", "样本", "请求", "配额", "扣减", "前→后", "状态"), budget_rows)}
  </div>
  <div class="panel span-6">
    <h2>LRU 淘汰序列</h2>
    {_table(("序号", "淘汰块", "新块", "依据"), eviction_rows)}
  </div>
  <div class="panel span-6">
    <h2>淘汰后重算清单</h2>
    {_table(("序号", "字块", "SHA-256", "与首次一致"), recompute_rows)}
  </div>
  <div class="panel span-6">
    <h2>哈希冲突桶</h2>
    {_table(("桶", "键数", "示例键"), collision_rows)}
  </div>
  <div class="panel span-6">
    <h2>逐样本阶段耗时</h2>
    {_table(("样本", "发射 ms", "解码 ms", "合并 ms", "总计 ms"), timing_rows)}
  </div>
</section>
<div class="footer-note">本页为离线自包含 HTML；数据来自 report.json。兼容性以 deterministic 区段的 SHA-256 摘要为准，墙钟耗时仅用于诊断。</div>
</main>
<script>
const rawReport = {embedded};
const filter = document.getElementById('sample-filter');
if (filter) {{
  filter.addEventListener('input', () => {{
    const needle = filter.value.trim().toLowerCase();
    document.querySelectorAll('.sample-card').forEach(card => {{
      card.style.display = !needle || card.textContent.toLowerCase().includes(needle) ? '' : 'none';
    }});
  }});
}}
</script>
</body>
</html>
"""
