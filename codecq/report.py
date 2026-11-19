"""Interactive diagnostic view.

Reads only the committed, version-stamped ledger snapshot and renders a
self-contained HTML report (no external assets, no network): per-frame
PSNR/SSIM curves, shot bands, decoded frame sequence and anomaly markers.
"""
from __future__ import annotations

from pathlib import Path

from . import ledger as ledger_mod
from . import util

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>codecq 诊断视图</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#14171c;color:#dde3ea}
h1{font-size:18px} h2{font-size:14px;margin:12px 0 4px}
select,option{background:#1d222b;color:#dde3ea}
.meta{color:#8b94a1;font-size:12px;margin-bottom:8px}
canvas{background:#1d222b;border:1px solid #2c3340;display:block;margin-bottom:8px}
table{border-collapse:collapse;font-size:12px;margin-bottom:12px}
td,th{border:1px solid #2c3340;padding:2px 8px;text-align:left}
.finding{color:#ff7b72}.ok{color:#7ee787}
#readout{font-size:12px;color:#8b94a1;height:16px}
</style>
</head>
<body>
<h1>codecq 编解码质量诊断视图</h1>
<div class="meta" id="ledgerMeta"></div>
<label>编码记录: <select id="encSel"></select></label>
<div id="readout"></div>
<canvas id="chart" width="960" height="280"></canvas>
<canvas id="chart2" width="960" height="200"></canvas>
<h2>检出异常</h2><div id="findings"></div>
<h2>分段汇总</h2><div id="segments"></div>
<h2>音频对齐</h2><div id="audio"></div>
<h2>解码帧序（前 40 帧）</h2><div id="seq"></div>
<script>
const LEDGER = __DATA__;
const encSel = document.getElementById('encSel');
document.getElementById('ledgerMeta').textContent =
  'ledger ' + LEDGER.version + ' | source ' + LEDGER.source.video.width + 'x' +
  LEDGER.source.video.height + '@' + LEDGER.source.video.fps + ' frames=' +
  LEDGER.source.video.frames + ' | overall ' + LEDGER.hashes.overall.slice(0,16);
const okEncodes = LEDGER.encodes.filter(e => e.status === 'ok');
okEncodes.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.id;encSel.appendChild(o);});
const shots = LEDGER.source.shots;
function anomalyFrames(enc){
  const s = new Set();
  for(const f of enc.findings){
    if(f.location.frames) f.location.frames.forEach(x=>s.add(x));
    if(f.location.frame !== undefined) s.add(f.location.frame);
  }
  return s;
}
function draw(canvasId, key, ymin, ymax){
  const enc = okEncodes[encSel.selectedIndex];
  const cv = document.getElementById(canvasId);
  const ctx = cv.getContext('2d');
  const W=cv.width,H=cv.height,pad=30;
  ctx.clearRect(0,0,W,H);
  const n = LEDGER.source.video.frames;
  const xs = i => pad + (W-2*pad)*i/(n-1);
  const ys = v => H-pad - (H-2*pad)*(v-ymin)/(ymax-ymin);
  shots.forEach((s,idx)=>{ if(idx%2===0){ctx.fillStyle='#222a36';ctx.fillRect(xs(s.start),pad,xs(s.end)-xs(s.start),H-2*pad);} });
  ctx.strokeStyle='#2c3340';ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  ctx.fillStyle='#8b94a1';ctx.font='10px sans-serif';
  ctx.fillText(ymax.toFixed(1),2,pad+8);ctx.fillText(ymin.toFixed(1),2,H-pad);
  shots.forEach(s=>{ctx.fillText('shot'+s.index,xs(s.start)+3,pad+10);});
  const m = enc.metrics.frames;
  ctx.strokeStyle='#6cb6ff';ctx.beginPath();
  m.forEach((f,i)=>{const x=xs(f.src),y=ys(Math.max(ymin,Math.min(ymax,f[key])));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();
  const an = anomalyFrames(enc);
  ctx.fillStyle='#ff7b72';
  an.forEach(f=>{ctx.fillRect(xs(f)-1,pad,2,H-2*pad);});
  cv.onmousemove = ev=>{
    const r=cv.getBoundingClientRect();
    const i=Math.round((ev.clientX-r.left-pad)/(W-2*pad)*(n-1));
    const f=m.find(x=>x.src===i);
    document.getElementById('readout').textContent = f?
      `frame ${f.src}: psnr_y=${f.psnr_y} psnr_w=${f.psnr_weighted} ssim_y=${f.ssim_y}`:'';
  };
}
function render(){
  const enc = okEncodes[encSel.selectedIndex];
  draw('chart','psnr_y',20,60);
  draw('chart2','ssim_y',0.8,1.0);
  const fd = document.getElementById('findings');
  fd.innerHTML = enc.findings.length?
    '<table><tr><th>code</th><th>location</th><th>detail</th></tr>'+
    enc.findings.map(f=>`<tr><td class="finding">${f.code}</td><td>${JSON.stringify(f.location)}</td><td>${f.detail}</td></tr>`).join('')+'</table>'
    :'<span class="ok">无检出</span>';
  document.getElementById('segments').innerHTML =
    '<table><tr><th>shot</th><th>range</th><th>present</th><th>missing</th><th>psnr_y mean</th><th>ssim_y mean</th></tr>'+
    enc.metrics.segments.map(s=>`<tr><td>${s.shot}</td><td>[${s.start},${s.end})</td><td>${s.frames_present}</td><td>${s.frames_missing.join(',')||'-'}</td><td>${s.psnr_y_mean??'-'}</td><td>${s.ssim_y_mean??'-'}</td></tr>`).join('')+'</table>';
  document.getElementById('audio').innerHTML =
    '<table><tr><th>shot</th><th>offset(samples)</th><th>gain</th><th>similarity</th><th>joint av offset(s)</th></tr>'+
    enc.audio_eval.segments.map(s=>`<tr><td>${s.shot}</td><td>${s.offset_samples}</td><td>${s.gain_ratio}</td><td>${s.similarity}</td><td>${s.joint_av_offset_seconds}</td></tr>`).join('')+'</table>';
  document.getElementById('seq').innerHTML =
    '<table><tr><th>dec</th><th>src</th><th>pts_ms</th><th>pix_fmt</th></tr>'+
    enc.frames.slice(0,40).map(f=>`<tr><td>${f.dec}</td><td>${f.src}</td><td>${f.pts_ms}</td><td>${f.pix_fmt}</td></tr>`).join('')+'</table>';
}
encSel.onchange = render;
render();
</script>
</body>
</html>
"""


def render_report(ledger_dir, out_path) -> dict:
    """Build report.html from the committed ledger snapshot only."""
    ledger = ledger_mod.load_committed(ledger_dir)
    html = TEMPLATE.replace("__DATA__", util.canonical_json(ledger))
    out_path = Path(out_path)
    util.atomic_write(out_path, html.encode("utf-8"))
    return {"version": ledger["version"], "path": str(out_path)}
