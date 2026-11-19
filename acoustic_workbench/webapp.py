"""Local read-only HTTP diagnostic dashboard."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import DiagnosticService


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Room Acoustic Incremental Diagnostics</title>
<style>
:root{--bg:#0f1218;--panel:#171c25;--line:#2a3342;--text:#e6edf6;--muted:#8d9bad;--amber:#ffb454;--green:#4fd18b;--blue:#6ea8fe;--red:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:650}.muted{color:var(--muted)}.pill{border:1px solid var(--line);padding:2px 8px;border-radius:20px}
main{display:grid;grid-template-columns:minmax(560px,1fr) 440px;gap:14px;padding:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
.panel h2{font-size:13px;margin:0 0 10px;color:#b9c6d8;font-weight:600;letter-spacing:.03em;text-transform:uppercase}
#projection{width:100%;height:auto;border-radius:6px;background:#05070b;cursor:crosshair;display:block}
.legend{display:flex;gap:14px;margin-top:9px;color:var(--muted);flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:1px}
svg{width:100%;height:auto;display:block;background:#10141c;border-radius:7px}.node{cursor:pointer}.node rect{stroke:#46536a;stroke-width:1}.node text{fill:#cfd9e6;font-size:9px;pointer-events:none}.edge{stroke:#303b4c;stroke-width:.8;fill:none}.edge.parameter{stroke-dasharray:3 4}.edge.hot{stroke:var(--amber);stroke-width:2}.node.selected rect{stroke:#fff;stroke-width:2}
details{border-top:1px solid var(--line);padding-top:8px;margin-top:10px}summary{cursor:pointer;color:var(--blue)}pre{white-space:pre-wrap;word-break:break-word;background:#0d1118;padding:9px;border-radius:6px;max-height:360px;overflow:auto}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.kv{display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #202836;padding:3px 0}.kv span:first-child{color:var(--muted)}.chain{max-height:210px;overflow:auto;border-left:2px solid var(--line);padding-left:9px;margin-top:5px}.chain-item{margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}button{background:#202838;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}
@media(max-width:1100px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>Room Acoustic Incremental Diagnostics</h1><span id="revision" class="pill">loading</span><span id="summary" class="muted"></span><span class="muted">read-only snapshot</span></header>
<main>
<section class="panel">
  <h2>Equirectangular projection / fixed tile boundaries</h2>
  <canvas id="projection" width="960" height="480"></canvas>
  <div class="legend"><span><i class="dot" style="background:var(--green)"></i>reused tile</span><span><i class="dot" style="background:var(--amber)"></i>recomputed tile</span><span>click tile or graph node</span></div>
  <h2 style="margin-top:16px">Dependency graph</h2>
  <svg id="graph" viewBox="0 0 1400 580" aria-label="dependency graph"></svg>
</section>
<aside class="panel">
  <h2>Node inspection</h2>
  <div id="detail"><span class="muted">Select a projection tile or graph node.</span></div>
</aside>
</main>
<script>
let S=null, selected=null, byId={}, tileSpecs=[];
const ns='http://www.w3.org/2000/svg';
function el(n,attrs={}){const e=document.createElementNS(ns,n);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
function text(t,attrs={}){const e=el('text',attrs);e.textContent=t;return e}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function nodePos(node){
  let r=0,c=0;
  if(node.payload&&node.payload.tile){r=node.payload.tile.row;c=node.payload.tile.col}
  if(node.type==='rir')return [70,70+parseInt(node.id.split(':c')[1])*82];
  if(node.type==='sph')return [280,70+parseInt(node.id.split(':c')[1])*82];
  if(node.type==='energy')return [500+c*50,60+r*125];
  if(node.type==='tile')return [970+c*50,60+r*125];
  return [20,20]
}
function paramPos(node,index){return [20,40+index*18]}
function colorFor(node){if(node.status==='reused')return '#173a2b';if(node.status==='recomputed')return '#4a3415';return '#202838'}
function strokeFor(node){if(node.status==='reused')return '#4fd18b';if(node.status==='recomputed')return '#ffb454';return '#46536a'}
async function load(){
  const response=await fetch('/api/snapshot',{cache:'no-store'}); if(!response.ok)throw new Error(await response.text());
  S=await response.json(); byId=Object.fromEntries(S.nodes.map(n=>[n.id,n])); tileSpecs=S.tiles;
  document.getElementById('revision').textContent='revision '+S.revision+' · '+S.scene_hash.slice(0,12);
  const b=S.build||{}; document.getElementById('summary').textContent=`closure ${b.closure_size||0} · reused ${b.reused_count||0} · recomputed ${b.recomputed_count||0}`;
  await drawProjection(); drawGraph(); if(S.nodes.length)selectNode('tile:r0c0');
}
function drawProjection(){
  return new Promise(resolve=>{const canvas=document.getElementById('projection'),ctx=canvas.getContext('2d');const img=new Image();img.onload=()=>{ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);overlayTiles(ctx,canvas);resolve()};img.onerror=()=>{ctx.fillStyle='#211';ctx.fillRect(0,0,canvas.width,canvas.height);resolve()};img.src='data:image/png;base64,'+S.projection.preview_png_base64});
}
function overlayTiles(ctx,canvas){
  const sx=canvas.width/S.width, sy=canvas.height/S.height;
  for(const t of S.tiles){const [x0,x1,y0,y1]=t.pixel_bounds;const isSel=selected==='tile:r'+t.row+'c'+t.col;ctx.strokeStyle=isSel?'#ffffff':(t.status==='reused'?'#4fd18b':'#ffb454');ctx.lineWidth=isSel?3:1.5;ctx.strokeRect(x0*sx+.5,y0*sy+.5,(x1-x0)*sx-1,(y1-y0)*sy-1);if(isSel){ctx.fillStyle='rgba(255,255,255,.18)';ctx.fillRect(x0*sx,y0*sy,(x1-x0)*sx,(y1-y0)*sy)}}
}
function drawGraph(){
  const svg=document.getElementById('graph');svg.innerHTML='';byId={};S.nodes.forEach(n=>byId[n.id]=n);
  const positions={};S.nodes.forEach(n=>positions[n.id]=nodePos(n));
  for(const key of S.parameter_keys){positions['param:'+key]=[12,30+S.parameter_keys.indexOf(key)*17]}
  const selectedNode=selected?byId[selected]:null;
  const hot=new Set();if(selectedNode){for(const d of selectedNode.dependencies)hot.add(selected+'|'+d);for(const e of S.edges){if(e.target===selected||e.source===selected)hot.add(e.source+'|'+e.target)}}
  const edgeLayer=el('g');svg.appendChild(edgeLayer);
  for(const e of S.edges){const a=positions[e.source],b=positions[e.target];if(!a||!b)continue;const isHot=hot.has(e.source+'|'+e.target);const path=el('path',{d:`M${a[0]+45},${a[1]+13} C${(a[0]+b[0])/2},${a[1]+13} ${(a[0]+b[0])/2},${b[1]+10} ${b[0]},${b[1]+10}`,class:'edge'+(e.kind==='parameter'?' parameter':'')+(isHot?' hot':'')});edgeLayer.appendChild(path)}
  const nodeLayer=el('g');svg.appendChild(nodeLayer);
  for(const n of S.nodes){const [x,y]=positions[n.id];const g=el('g',{class:'node'+(n.id===selected?' selected':''),transform:`translate(${x},${y})`});g.addEventListener('click',()=>selectNode(n.id));const w=n.type==='rir'||n.type==='sph'?70:42,h=n.type==='rir'||n.type==='sph'?26:20;g.appendChild(el('rect',{width:w,height:h,rx:4,fill:colorFor(n),stroke:strokeFor(n)}));const label=n.type==='rir'?n.id.replace('rir:','RIR '):n.type==='sph'?n.id.replace('sh:','SH '):n.id.replace('energy:','E ').replace('tile:','T ');g.appendChild(text(label,{x:w/2,y:h/2+3,'text-anchor':'middle'}));nodeLayer.appendChild(g)}
  const colLabels=[[70,'RIR'],[280,'SH COEFFS'],[515,'ENERGY BLOCKS'],[985,'PROJECTION TILES']];for(const [x,l] of colLabels){svg.appendChild(text(l,{x,y:22,fill:'#7d8ca0','font-size':11}))}
}
function selectNode(id){selected=id;drawGraph();const canvas=document.getElementById('projection');overlayTiles(canvas.getContext('2d'),canvas);renderDetail(byId[id]);}
function chainHtml(id,depth=0,seen=new Set()){
  if(!id||seen.has(id))return '';seen.add(id);const n=byId[id];if(!n)return '';const reason=n.reason||{};let h=`<div class="chain-item"><b>${esc(id)}</b> <span class="muted">rev ${n.computed_revision} · ${esc(n.status)}</span><br><span class="muted">${esc(reason.kind||'')}</span>`;if(reason.changed_version_keys&&reason.changed_version_keys.length)h+=`<br><span class="warn">version: ${esc(reason.changed_version_keys.join(', '))}</span>`;if(reason.changed_value_keys&&reason.changed_value_keys.length)h+=`<br><span class="warn">value: ${esc(reason.changed_value_keys.join(', '))}</span>`;if(depth<4){for(const dep of (reason.upstream||n.dependencies)){h+=chainHtml(dep,depth+1,seen)}}return h+'</div>'}
function renderDetail(n){
  if(!n){document.getElementById('detail').innerHTML='<span class="muted">Node not found.</span>';return}
  const reason=n.reason||{};const stamps=Object.entries(n.param_stamps||{}).map(([k,v])=>`<div class="kv"><span>${esc(k)}</span><span>v${v}</span></div>`).join('');const deps=(n.dependencies||[]).map(d=>`<div class="kv"><span>${esc(d)}</span><span>${esc((byId[d]||{}).content_hash||'').slice(0,12)}</span></div>`).join('');
  document.getElementById('detail').innerHTML=`<div class="kv"><span>id</span><b>${esc(n.id)}</b></div><div class="kv"><span>type</span><span>${esc(n.type)}</span></div><div class="kv"><span>status</span><span class="${n.status==='reused'?'good':'warn'}">${esc(n.status)}</span></div><div class="kv"><span>content hash</span><span>${esc(n.content_hash).slice(0,20)}…</span></div><div class="kv"><span>computed revision</span><span>${n.computed_revision}</span></div><div class="kv"><span>reuse source</span><span>${n.last_reuse_source_revision??'—'}</span></div><h2 style="margin-top:13px">Invalidation reason</h2><div class="chain">${chainHtml(n.id)}</div><h2 style="margin-top:13px">Version stamps</h2>${stamps||'<span class="muted">No local parameter stamps.</span>'}<h2 style="margin-top:13px">Upstream hashes</h2>${deps||'<span class="muted">No upstream artifact.</span>'}<details><summary>Raw node payload</summary><pre>${esc(JSON.stringify(n.payload,null,2))}</pre></details>`;
}
document.getElementById('projection').addEventListener('click',ev=>{if(!S)return;const c=ev.currentTarget,r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*c.width/r.width,y=(ev.clientY-r.top)*c.height/r.height,sx=S.width/c.width,sy=S.height/c.height,px=x*sx,py=y*sy;for(const t of S.tiles){const[x0,x1,y0,y1]=t.pixel_bounds;if(px>=x0&&px<x1&&py>=y0&&py<y1){selectNode('tile:r'+t.row+'c'+t.col);break}}});
load().catch(err=>{document.getElementById('detail').textContent='Snapshot load failed: '+err});
</script>
</body>
</html>
"""


class DiagnosticHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: DiagnosticService):
        self.diagnostic_service = service
        super().__init__(address, DiagnosticRequestHandler)


class DiagnosticRequestHandler(BaseHTTPRequestHandler):
    server: DiagnosticHTTPServer

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode("utf-8"))
        elif self.path == "/api/snapshot":
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", self.server.diagnostic_service.snapshot_json())
        elif self.path == "/api/projection.png":
            self._send(HTTPStatus.OK, "image/png", self.server.diagnostic_service.projection_png())
        elif self.path == "/api/health":
            payload = json.dumps({"status": "ok", "read_only": True}, separators=(",", ":")).encode("utf-8")
            self._send(HTTPStatus.OK, "application/json", payload)
        else:
            self._send(HTTPStatus.NOT_FOUND, "application/json", b'{"code":"E_NOT_FOUND"}')

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        self._read_only()

    def do_PUT(self) -> None:
        self._read_only()

    def do_PATCH(self) -> None:
        self._read_only()

    def do_DELETE(self) -> None:
        self._read_only()

    def _read_only(self) -> None:
        payload = b'{"code":"E_SNAPSHOT_CONFLICT","message":"diagnostic view is read-only"}'
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, "application/json", payload, extra_headers={"Allow": "GET, HEAD"})

    def _send(
        self,
        status: HTTPStatus,
        content_type: str,
        payload: bytes,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    service: DiagnosticService,
    host: str = "127.0.0.1",
    port: int = 0,
) -> DiagnosticHTTPServer:
    return DiagnosticHTTPServer((host, int(port)), service)


def run_dashboard(
    workspace: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    ready: threading.Event | None = None,
) -> None:
    service = DiagnosticService.from_workspace(workspace)
    server = create_server(service, host, port)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    print(f"diagnostic dashboard: {url}")
    print(f"snapshot revision: {service.snapshot.revision} hash: {service.snapshot.source_hash[:16]}")
    if ready is not None:
        ready.set()
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
