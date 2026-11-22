from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StemScope · 分轨渲染诊断工作台</title>
<style>
:root{--bg:#0a0e14;--panel:#111824;--panel2:#162132;--line:#26344a;--text:#e7edf6;--muted:#8fa0b8;--cyan:#4bd8e8;--lime:#a8e86b;--amber:#ffbf69;--red:#ff6b7a;--violet:#ad8bff;--shadow:0 18px 48px rgba(0,0,0,.32)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% 0%,#18243a 0,transparent 33%),radial-gradient(circle at 90% 10%,#172b2a 0,transparent 30%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
header{padding:28px 34px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:22px;align-items:flex-end;background:linear-gradient(180deg,rgba(15,23,35,.96),rgba(10,14,20,.74))}
h1{margin:0;font-size:25px;letter-spacing:-.02em}h2{font-size:15px;margin:0 0 12px;letter-spacing:.02em;text-transform:uppercase;color:#c9d6e8}.sub{color:var(--muted);font-size:12px;margin-top:6px}.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:5px 10px;font-size:11px;color:var(--muted);background:#0e1622}.pill.ok{color:var(--lime);border-color:#365127;background:#132016}.pill.bad{color:var(--red);border-color:#5b2931;background:#211318}
main{padding:22px 30px 48px;max-width:1540px;margin:auto}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:var(--shadow)}.metric{font-size:24px;font-weight:700;letter-spacing:-.02em}.metric small{font-size:11px;color:var(--muted);font-weight:500}.metric.cyan{color:var(--cyan)}.metric.lime{color:var(--lime)}.metric.amber{color:var(--amber)}.metric.violet{color:var(--violet)}
section{margin-top:20px}.split{display:grid;grid-template-columns:1.35fr .65fr;gap:16px}.split3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.track{display:grid;grid-template-columns:190px minmax(230px,1fr) 180px;gap:18px;align-items:center;padding:14px 0;border-bottom:1px solid rgba(38,52,74,.68)}.track:last-child{border-bottom:0}.trackname{font-weight:700;font-size:16px;text-transform:capitalize}.tag{display:inline-block;font-size:10px;padding:2px 7px;border:1px solid var(--line);border-radius:999px;color:var(--muted);margin-right:4px;margin-top:5px}.wave{height:74px;width:100%;background:#0a111b;border:1px solid #1b2b40;border-radius:9px}.trackctl{display:grid;gap:8px}.gainrow{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}input[type=range]{width:100%;accent-color:var(--cyan)}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:8px 9px;border-bottom:1px solid rgba(38,52,74,.58);vertical-align:top}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}tr:hover td{background:rgba(75,216,232,.035)}.num{text-align:right;font-variant-numeric:tabular-nums}.status{font-size:10px;border-radius:999px;padding:2px 7px;border:1px solid var(--line)}.status.reuse{color:var(--lime);border-color:#365127}.status.recompute{color:var(--amber);border-color:#5f4724}.status.bad{color:var(--red);border-color:#5b2931}.scroll{max-height:420px;overflow:auto;border-radius:10px}
.timeline{display:grid;gap:8px}.cmd{display:grid;grid-template-columns:90px 120px 1fr 90px 100px;gap:10px;align-items:center;padding:9px 11px;background:#0c1420;border:1px solid #1b2b40;border-radius:9px;font-size:11px}.bar{height:7px;border-radius:999px;background:#1a293c;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime))}.oktext{color:var(--lime)}.badtext{color:var(--red)}.ambertext{color:var(--amber)}
.checks{display:flex;flex-wrap:wrap;gap:7px}.loss{font-size:11px;padding:9px 11px;border-left:2px solid var(--amber);background:#151b20;border-radius:0 8px 8px 0;margin-bottom:6px}.muted{color:var(--muted)}.code{font-family:ui-monospace,monospace;font-size:11px;color:#bcd0e8}.footer{color:var(--muted);font-size:11px;padding:18px 2px}.notice{font-size:11px;color:var(--muted);margin:8px 0 14px}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:10px}.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:4px}.dot.cyan{background:var(--cyan)}.dot.lime{background:var(--lime)}.dot.amber{background:var(--amber)}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,1fr)}.split,.split3{grid-template-columns:1fr}.track{grid-template-columns:1fr}.cmd{grid-template-columns:1fr 1fr}.card{padding:13px}}
</style>
</head>
<body>
<header>
  <div><h1>StemScope · 分轨渲染诊断工作台</h1><div class="sub">确定性重混 / 分轨级增量失效 / 预算内响应 / 原子阶段快照</div></div>
  <div><span id="overall" class="pill">loading</span> <span class="pill mono" id="reportId"></span></div>
</header>
<main>
  <div class="grid" id="metrics"></div>
  <div class="checks" id="checks" style="margin-top:14px"></div>

  <section class="split">
    <div class="card">
      <h2>分轨视图与本地视觉预览</h2>
      <div class="notice">滑杆仅用于视觉预演；权威帧值来自服务端固定点重混与逐帧明细。滑块不会伪造增量结果。</div>
      <div id="tracks"></div>
    </div>
    <div class="card">
      <h2>最终混音</h2>
      <div id="mixwave"></div>
      <div class="legend"><span><i class="dot cyan"></i>最终 PCM</span><span><i class="dot amber"></i>峰值参考</span></div>
      <div id="mixmeta" style="margin-top:12px"></div>
    </div>
  </section>

  <section class="card">
    <h2>增量交互时间线</h2>
    <div class="timeline" id="timeline"></div>
  </section>

  <section class="split">
    <div class="card"><h2>逐片段峰值与缓存动作</h2><div class="scroll"><table id="segments"></table></div></div>
    <div class="card"><h2>预算语义</h2><div id="budget"></div></div>
  </section>

  <section class="split">
    <div class="card"><h2>三格式校验与互转</h2><div id="formats"></div></div>
    <div class="card"><h2>旁路元数据 / 丢失字段</h2><div id="losses"></div></div>
  </section>

  <section class="split">
    <div class="card"><h2>快照与恢复判定</h2><div id="snapshot"></div></div>
    <div class="card"><h2>失败清单（稳定排序）</h2><div id="failures"></div></div>
  </section>

  <section class="card"><h2>注入故障链路</h2><div id="injections"></div></section>
  <div class="footer" id="footer"></div>
</main>
<script>
const report = __REPORT_JSON__;
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v) => Number(v ?? 0).toLocaleString('en-US');
function pill(ok, text){return `<span class="pill ${ok?'ok':'bad'}">${esc(text)}</span>`}
function metric(label, value, cls=''){return `<div class="card"><div class="metric ${cls}">${value}</div><div class="muted" style="font-size:11px;margin-top:4px">${esc(label)}</div></div>`}

const checks = report.acceptance?.checks || [];
$('overall').className = `pill ${report.acceptance?.passed?'ok':'bad'}`;
$('overall').textContent = report.acceptance?.passed ? '全部验收通过' : '存在失败项';
$('reportId').textContent = report.report_id || '';
$('checks').innerHTML = checks.map(c => pill(c.passed, `${c.passed?'PASS':'FAIL'} · ${c.id}`)).join('');
const fr = report.final_render || {};
$('metrics').innerHTML = [
  metric('最终峰值', num(fr.aggregate_peak), 'cyan'),
  metric('输出片段', num(fr.completed_segments), ''),
  metric('最终复用任务', num(fr.reused?.length || 0), 'lime'),
  metric('预算超限响应', num(report.interaction?.budget_hits || 0), 'amber'),
  metric('报告指纹', `<span style="font-size:13px">${esc((report.report_fingerprint||'').slice(0,12))}</span>`, 'violet')
].join('');

function waveSvg(points, cls='cyan'){
  if(!points || !points.length) return '<div class="wave"></div>';
  const w=900,h=100,mid=h/2;
  const step=w/Math.max(1,points.length);
  const paths=[];
  points.forEach((p,i)=>{const x=i*step; const y0=mid-(p[1]/32768)*45; const y1=mid-(p[0]/32768)*45; paths.push(`M${x.toFixed(2)} ${y0.toFixed(2)}L${(x+Math.max(.7,step*.7)).toFixed(2)} ${y1.toFixed(2)}`)});
  return `<svg class="wave" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${paths.join('')}" stroke="var(--${cls})" stroke-width="1.2" opacity=".85"/><line x1="0" y1="50" x2="${w}" y2="50" stroke="#22344a" stroke-width=".8"/></svg>`;
}
const tracks = report.visualization?.tracks || [];
$('tracks').innerHTML = tracks.map((t,idx)=>`<div class="track"><div><div class="trackname">${esc(t.name)}</div><div><span class="tag">${esc(t.source_format_version)}</span><span class="tag">${num(t.frames)} frames</span></div><div class="muted mono" style="font-size:10px;margin-top:6px">max |x| ${num(t.max_abs_amplitude)}</div></div><div id="wave_${idx}">${waveSvg(t.waveform)}</div><div class="trackctl"><div class="gainrow"><span>gain ${esc(t.gain)}</span><span id="eff_${idx}">Q ${num(t.quantized_gain_int)}</span></div><input id="gain_${idx}" type="range" min="0" max="200" value="${Math.round(t.gain*100)}"><div class="gainrow"><span>${t.muted?'MUTED':'ACTIVE'}</span><span>crop ${esc(t.start_ms)}–${esc(t.end_ms)} ms</span></div></div></div>`).join('');
tracks.forEach((t,idx)=>{const slider=$(`gain_${idx}`); slider.addEventListener('input',()=>{const g=Number(slider.value)/100; $(`eff_${idx}`).textContent=`visual ${g.toFixed(2)}×`; $(`wave_${idx}`).innerHTML=waveSvg(t.waveform.map(p=>[Math.round(p[0]*g),Math.round(p[1]*g)]));});});
$('mixwave').innerHTML = waveSvg(report.visualization?.mix_waveform, 'cyan');
$('mixmeta').innerHTML = `<div class="mono muted" style="font-size:11px">${esc((fr.pcm_sha256||'').slice(0,32))}…</div><div class="muted" style="font-size:11px">${num(fr.frames)} frames · ${num(fr.pcm_bytes)} bytes</div>`;

const timeline = report.interaction?.responses || [];
$('timeline').innerHTML = timeline.map(r=>`<div class="cmd"><span class="code">${esc(r.command_id)}</span><span class="${r.status==='completed'?'oktext':'badtext'}">${esc(r.status)}</span><div><div class="bar"><span style="width:${Math.min(100,(r.type==='budget'?100:r.elapsed_ms/r.budget_ms*100)).toFixed(1)}%"></span></div><div class="muted" style="font-size:10px;margin-top:4px">${esc(r.type||'')} · requested v${esc(r.requested_param_version)} · summary v${esc(r.summary?.param_version)}</div></div><span class="mono ${r.status==='completed'?'oktext':'badtext'}">${Number(r.elapsed_ms).toFixed(0)} ms</span><span class="mono muted">budget ${Number(r.budget_ms).toFixed(0)} ms</span></div>`).join('');

const segs = fr.segments || [];
$('segments').innerHTML = `<thead><tr><th>#</th><th>clip / chunk</th><th class="num">frames</th><th class="num">peak</th><th>mix cache</th><th class="num">detail sha</th></tr></thead><tbody>${segs.map(s=>`<tr><td class="num">${s.segment_index}</td><td>${esc(s.clip_id)} / ${s.chunk_index}</td><td class="num">${s.frames}</td><td class="num">${num(s.peak)}</td><td>${s.cache_action==='reused'?'<span class="status reuse">reused</span>':s.cache_action==='recomputed'?'<span class="status recompute">recomputed</span>':'<span class="status">cached</span>'}</td><td class="num mono" style="font-size:10px">${esc((s.detail_sha256||'').slice(0,10))}</td></tr>`).join('')}</tbody>`;

const b = report.interaction || {};
$('budget').innerHTML = `<div class="grid" style="grid-template-columns:1fr 1fr">${metric('预算内完成', num(b.completed_within_budget), 'lime')}${metric('超限返回旧摘要', num(b.budget_hits), 'amber')}</div><div class="muted" style="font-size:11px;margin-top:12px">${esc(b.semantics_note || '')}</div>`;

const fv = report.format_validation || {};
$('formats').innerHTML = `<div class="checks" style="margin-bottom:10px">${(fv.versions||[]).map(v=>`<span class="pill ok">${esc(v)}</span>`).join('')}</div><div class="muted" style="font-size:11px">${num(fv.clip_count||0)} clips · ${esc(fv.pcm_equality ? '三格式 PCM 逐帧一致' : 'PCM 不一致')}</div><div class="scroll" style="margin-top:10px"><table><thead><tr><th>clip</th><th>v1 fp</th><th>v2 fp</th><th>v3 fp</th></tr></thead><tbody>${(fv.clips||[]).map(c=>`<tr><td>${esc(c.clip_id)}</td>${['v1','v2','v3'].map(v=>`<td class="mono" style="font-size:10px">${esc((c.versions?.[v]?.package_fingerprint||'').slice(0,12))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;

const losses = report.conversions?.loss_fields || [];
$('losses').innerHTML = losses.length ? losses.map(l=>`<div class="loss"><b>${esc(l.field)}</b> · ${esc(l.source_track)} · ${esc(l.source_format_version)} → ${esc(l.target_format_version)}<div class="mono muted" style="font-size:10px;margin-top:3px">${esc(l.value)}</div></div>`).join('') : '<div class="muted">本次转换没有不可表达字段。</div>';

const snap = report.snapshot || {};
$('snapshot').innerHTML = `<table><tbody><tr><td>初始审计</td><td>${esc(snap.initial_status)}</td></tr><tr><td>续跑审计</td><td>${pill(snap.resume?.valid, snap.resume?.status||'')}</td></tr><tr><td>续跑缓存命中</td><td class="num">${num(snap.resume?.mix_hits||0)} mix / ${num(snap.resume?.parse_hits||0)} parse</td></tr><tr><td>检查点提交</td><td class="num">${num(snap.interactive_checkpoint_commits||0)}</td></tr><tr><td>版本陈旧拒绝</td><td>${pill(snap.stale_version?.rejected, 'E_SNAPSHOT_STALE')}</td></tr><tr><td>配置哈希陈旧拒绝</td><td>${pill(snap.stale_config?.rejected, 'E_SNAPSHOT_STALE')}</td></tr><tr><td>临时目录残留</td><td class="num">${num(snap.temp_dirs_remaining||0)}</td></tr></tbody></table><div class="muted mono" style="font-size:10px;margin-top:10px">${esc(snap.global_version||'')}</div>`;

const failures = report.diagnostics?.failures || [];
$('failures').innerHTML = failures.length ? `<div class="scroll"><table><thead><tr><th>code</th><th>stage</th><th>track</th><th>message</th></tr></thead><tbody>${failures.map(f=>`<tr><td class="badtext code">${esc(f.code)}</td><td>${esc(f.stage)}</td><td>${esc(f.track)}</td><td>${esc(f.message)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="muted">当前成功运行没有失败项。</div>';

const inj = report.injections || [];
$('injections').innerHTML = `<table><thead><tr><th>case</th><th>format</th><th>operation</th><th>expected</th><th>observed</th><th>result</th></tr></thead><tbody>${inj.map(x=>`<tr><td>${esc(x.case_id)}</td><td>${esc(x.format_version)}</td><td>${esc(x.operation)}</td><td class="code">${esc(x.expected_code)}</td><td class="code">${esc((x.observed_codes||[]).join(', '))}</td><td>${pill(x.passed,'PASS/FAIL')}</td></tr>`).join('')}</tbody></table>`;
$('footer').textContent = `StemScope report ${report.report_id} · deterministic payload fingerprint ${report.report_fingerprint} · timings are measured separately and excluded from the repeatability fingerprint`;
</script>
</body>
</html>
'''


def generate_dashboard(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__REPORT_JSON__", embedded)
    path.write_text(html, encoding="utf-8")
    return path
