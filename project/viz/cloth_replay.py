"""Export a complete cloth archive to a self-contained, read-only HTML replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

import numpy as np

from ..cloth_archive import SCHEMA, load_run, sha256

MAX_FILES = 514
MAX_RECORDS = 512
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_VERTEX_INSTANCES = 200_000
MAX_TRIANGLE_INSTANCES = 400_000
MAX_HTML_BYTES = 16 * 1024 * 1024


class ReplayLimitError(ValueError):
    pass


def _limit(name, actual, maximum):
    if actual > maximum:
        raise ReplayLimitError(f"{name}: actual={actual}, limit={maximum}; no frames were omitted")


def source_fingerprint(root):
    """Includes names and sizes, so an extra file cannot escape the comparison."""
    root = Path(root).resolve(strict=True)
    files = sorted(root.iterdir())
    _limit("MAX_FILES", len(files), MAX_FILES)
    total = expanded = 0
    result = {}
    for path in files:
        st = path.lstat()
        if (not stat.S_ISREG(st.st_mode) or path.is_symlink()
                or getattr(st, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"archive contains non-plain member: {path.name}")
        total += st.st_size
        _limit("MAX_ARCHIVE_BYTES", total, MAX_ARCHIVE_BYTES)
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as z:
                expanded += sum(info.file_size for info in z.infolist())
            _limit("MAX_EXPANDED_BYTES", expanded, MAX_EXPANDED_BYTES)
        result[path.name] = dict(size=st.st_size, sha256=sha256(path))
    return result


def payload_from_run(run, fingerprint):
    _limit("MAX_RECORDS", len(run.records), MAX_RECORDS)
    vertices = sum(len(r.arrays["vertex_ids"]) for r in run.records)
    triangles = sum(len(r.arrays["triangle_ids"]) for r in run.records)
    _limit("MAX_VERTEX_INSTANCES", vertices, MAX_VERTEX_INSTANCES)
    _limit("MAX_TRIANGLE_INSTANCES", triangles, MAX_TRIANGLE_INSTANCES)
    cumulative = np.zeros(3)
    frames = []
    for index, record in enumerate(run.records):
        a, m = record.arrays, record.metadata
        load = m["load"]
        if m["stage"] == "post_solve":
            # One conversion of recorded deltas, no new load ledger or solver.
            cumulative += np.asarray(load["audit"]["external_total"])
        displacement = np.linalg.norm(a["positions"] - a["reference_positions"], axis=1)
        energy = float(a["diag_bend_energy"].sum()) if m["diagnostics"] is not None else None
        frame = {k: a[k].tolist() for k in ("positions", "reference_positions", "triangles", "vertex_ids",
                   "root_vertex_ids", "pinned", "edge_vertex_ids", "cut_root_edges")}
        frame.update(index=index, stage=m["stage"], version=m["version"], time=m["time"], h=m["h"],
                     event=m["event"], load=load, fracture_decision=m.get("fracture_decision"),
                     impulse=a["load_impulse"].tolist() if load is not None else None,
                     max_displacement_m=float(displacement.max(initial=0.)), bend_energy_J=energy,
                     cumulative_input_impulse_Ns=cumulative.tolist(),
                     cumulative_input_impulse_norm_Ns=float(np.linalg.norm(cumulative)))
        frames.append(frame)
    return dict(schema="xpbd-cloth-replay-v1", archive_schema=SCHEMA, run=run.metadata,
                source_files=fingerprint, hash_meaning="computed checksums, not signatures",
                frames=frames, display_only=True)


def encode_payload(payload):
    # No untrusted text can close the non-executable JSON script element.
    text = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XPBD · 真实状态回放</title>
<style>
:root{color-scheme:light;--ink:#17263a;--muted:#627188;--line:#dbe3ed;--blue:#215ce0;--cut:#ec8544}
*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:var(--ink);font:14px/1.5 system-ui,"Segoe UI",sans-serif}
main{max-width:1440px;margin:auto;padding:28px 32px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
.eyebrow{font-size:11px;letter-spacing:2px;color:var(--blue);font-weight:750}h1{font-size:27px;margin:4px 0;font-weight:700}p{margin:5px 0}.muted{color:var(--muted)}
.badge{padding:7px 11px;border:1px solid #b9d5d2;background:#e9f6f2;color:#247466;border-radius:7px;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:18px}.panel{background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.canvashead{display:flex;justify-content:space-between;align-items:center;padding:13px 18px;border-bottom:1px solid var(--line)}
#scene{display:block;width:100%;height:480px;background:#101e31}.legend{display:flex;gap:18px;padding:10px 18px;font-size:12px;flex-wrap:wrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.controls{padding:12px 18px 16px;border-top:1px solid var(--line)}
.buttons{display:flex;align-items:center;gap:7px;flex-wrap:wrap}button,select{border:1px solid var(--line);border-radius:6px;background:white;color:var(--ink);padding:6px 10px;font:inherit;cursor:pointer}
button:hover{background:#edf3ff}button.primary{background:var(--blue);color:white;border-color:var(--blue)}button:disabled{opacity:.35;cursor:default}button[aria-pressed="true"]{border-color:var(--blue);color:var(--blue)}
input[type=range]{accent-color:var(--blue)}#frameSlider{width:100%;margin:12px 0 6px}.small{font-size:12px}.row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:9px 0}
aside{padding:18px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}.value{font-size:21px;font-variant-numeric:tabular-nums;margin-bottom:14px}#stage{font-size:16px;color:var(--blue);font-weight:700}
.rule{border-top:1px solid var(--line);margin:14px 0}.notice{border-left:3px solid #e2a34e;background:#fff8eb;padding:10px 12px;border-radius:3px;font-size:12px;margin-top:13px}
#event{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}#audit{font-size:11px;white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;line-height:1.7}
.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:18px}.chart{padding:14px 16px}.chart h2{font-size:13px;margin:0 0 7px}.chart svg{width:100%;height:150px;display:block}.metric{font-size:19px;font-variant-numeric:tabular-nums}
details{margin-top:18px;padding:13px 18px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;color:var(--muted);max-height:320px;overflow:auto}
footer{padding:16px 2px 0;font-size:12px;color:var(--muted)}#error{color:#a62323;white-space:pre-wrap}
@media(max-width:950px){main{padding:16px}.layout{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:10px}.charts{grid-template-columns:1fr}#scene{height:420px}}
</style></head><body><main>
<header><div><div class="eyebrow">XPBD / STATE REVIEW</div><h1>载荷 · 形变 · 拓扑</h1><p class="muted">读取真实归档，逐条查看完成态与裂后事件。</p></div><span class="badge" id="backend">只读回放</span></header>
<div id="error" role="alert"></div>
<div class="layout"><section class="panel">
<div class="canvashead"><span id="recordLabel">记录</span><div class="buttons"><button id="viewOblique" aria-pressed="true">斜视</button><button id="viewFront" aria-pressed="false">正视</button><button id="viewSide" aria-pressed="false">侧视</button></div></div>
<canvas id="scene" aria-label="布料三角网格；橙色为参考裂边，绿色为固定点"></canvas>
<div class="legend"><span><i class="dot" style="background:#5ba5e7"></i>布面</span><span><i class="dot" style="background:#65d7bb"></i>固定点</span><span><i class="dot" style="background:#ec8544"></i>裂边 / 新生点</span><span>箭头：输入冲量，共同比例</span></div>
<div class="controls"><div class="buttons"><button id="first" aria-label="首帧">|‹</button><button id="prev" aria-label="上一帧">‹</button><button class="primary" id="play">播放</button><button id="next" aria-label="下一帧">›</button><button id="last" aria-label="末帧">›|</button><label>速度 <select id="speed"><option value="2">2 记录/秒</option><option value="6">6 记录/秒</option><option value="12">12 记录/秒</option></select></label><label><input id="showIds" type="checkbox"> 顶点 ID</label></div>
<input id="frameSlider" type="range" min="0" value="0" aria-label="记录索引"><div class="row small"><span id="timeLabel"></span><span>播放间隔独立于物理步长；不插值</span></div>
<div class="row"><label for="magnify">视觉位移放大</label><input id="magnify" type="range" min="1" max="100" step="1" value="1"><strong id="magLabel">×1</strong></div><p class="small muted">放大仅作用于位移，曲线保留真实数值。全序列固定取景，不随帧自动缩放。</p></div>
</section><aside class="panel">
<div><div class="label">记录阶段</div><div id="stage"></div><div class="label" style="margin-top:14px">物理步 / 拓扑版</div><div class="value" id="version"></div><div class="label">顶点 / 三角形</div><div class="value" id="counts"></div></div>
<div><div class="label">拓扑事件</div><div id="event"></div><div class="rule"></div><div class="label">输入冲量 [N·s]</div><div id="audit"></div><div id="supportStatus" class="notice"></div></div>
</aside></div>
<div class="charts"><section class="panel chart"><h2>最大位移 [m]</h2><div class="metric" id="dispValue"></div><svg id="dispChart" viewBox="0 0 480 160" aria-label="最大位移随物理时间变化"></svg></section><section class="panel chart"><h2>bend 诊断能量 [J]</h2><div class="metric" id="energyValue"></div><svg id="energyChart" viewBox="0 0 480 160" aria-label="弯曲能量；缺失和拓扑变化处断线"></svg></section><section class="panel chart"><h2>累计输入冲量范数 [N·s]</h2><div class="metric" id="impulseValue"></div><svg id="impulseChart" viewBox="0 0 480 160" aria-label="累计输入冲量范数"></svg></section></div>
<footer>曲线横轴为归档 time [s]；同一时刻的拓扑事件保留独立记录。能量按当时铰链集合求和，缺失或换拓扑处断线，不表示能量守恒。规定切割 gap=0 不保证事件瞬间已张开。参数为工程示例，非材料标定。</footer>
<details class="panel"><summary>配置、来源与校验和</summary><pre id="provenance"></pre></details>
</main><script id="replayData" type="application/json">@@DATA@@</script>
<script>
"use strict";
try {
const data=JSON.parse(document.getElementById("replayData").textContent), frames=data.frames;
const el=id=>document.getElementById(id), fmt=v=>v===null?"未记录":Number(v).toExponential(5);
const vector=v=>v.map(x=>Number(x).toPrecision(5)).join(", ");
let index=0, timer=null, yaw=.55, pitch=-.25, mag=1;
const canvas=el("scene"), ctx=canvas.getContext("2d"), slider=el("frameSlider");
slider.max=String(frames.length-1);
el("backend").textContent=data.run.backend.toUpperCase()+" / "+data.run.precision+" · 只读";
el("supportStatus").textContent=data.run.support_reaction==="unmeasured"
 ?"完整支撑反力：未测量。固定点外载抵消项仅为记账贡献。"
 :"支撑反力已测量。逐窗冲量、力矩与收支见源归档 support_balance。";
el("provenance").textContent=JSON.stringify({archive_schema:data.archive_schema,run:data.run,source_files:data.source_files,hash_meaning:data.hash_meaning},null,2);
const ref=frames[0].reference_positions, min=[Infinity,Infinity,Infinity], max=[-Infinity,-Infinity,-Infinity];
for(const p of ref)for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}
const center=min.map((v,k)=>(v+max[k])/2);
let radius=1;
let maxImpulse=0;
for(const f of frames)if(f.impulse)for(const j of f.impulse)maxImpulse=Math.max(maxImpulse,Math.hypot(...j));
function position(f,i){return f.positions[i].map((x,k)=>f.reference_positions[i][k]+mag*(x-f.reference_positions[i][k]));}
function bounds(){radius=1e-9;for(const f of frames)for(let i=0;i<f.positions.length;i++){const p=position(f,i);radius=Math.max(radius,Math.hypot(...p.map((v,k)=>v-center[k])));}}
function project(p,w,h){let [x,y,z]=p.map((v,k)=>v-center[k]);const xx=x*Math.cos(yaw)+z*Math.sin(yaw), zz=-x*Math.sin(yaw)+z*Math.cos(yaw), yy=y*Math.cos(pitch)-zz*Math.sin(pitch);const scale=.40*Math.min(w,h)/radius;return [w/2+scale*xx,h/2-scale*yy,y*Math.sin(pitch)+zz*Math.cos(pitch)];}
function line(a,b,color,width=1){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function draw(){
 const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const f=frames[index], points=f.positions.map((_,i)=>project(position(f,i),w,h));
 ctx.strokeStyle="#21334c";ctx.lineWidth=.6;for(let x=0;x<w;x+=40)line([x,0],[x,h],"#1a2b43",.5);for(let y=0;y<h;y+=40)line([0,y],[w,y],"#1a2b43",.5);
 const tris=f.triangles.slice().sort((a,b)=>a.reduce((s,i)=>s+points[i][2],0)-b.reduce((s,i)=>s+points[i][2],0));
 for(const tri of tris){ctx.beginPath();tri.forEach((i,k)=>k?ctx.lineTo(points[i][0],points[i][1]):ctx.moveTo(points[i][0],points[i][1]));ctx.closePath();ctx.fillStyle="rgba(74,144,212,.24)";ctx.fill();ctx.strokeStyle="#5593c5";ctx.lineWidth=1;ctx.stroke();}
 const byId=new Map(f.vertex_ids.map((id,i)=>[id,i])),cut=new Set(f.cut_root_edges.map(e=>e.join(":")));
 for(const edge of f.edge_vertex_ids){const a=byId.get(edge[0]),b=byId.get(edge[1]),r=[f.root_vertex_ids[a],f.root_vertex_ids[b]].sort((x,y)=>x-y);if(cut.has(r.join(":")))line(points[a],points[b],"#ff9c61",3);}
 if(f.impulse&&maxImpulse>0)for(let i=0;i<points.length;i++){const p=position(f,i),tip=p.map((x,k)=>x+f.impulse[i][k]/maxImpulse*radius*.23),q=project(tip,w,h),a=points[i];line(a,q,"#b0bff9",1.2);const angle=Math.atan2(q[1]-a[1],q[0]-a[0]);line(q,[q[0]-5*Math.cos(angle-.5),q[1]-5*Math.sin(angle-.5)],"#b0bff9",1.2);line(q,[q[0]-5*Math.cos(angle+.5),q[1]-5*Math.sin(angle+.5)],"#b0bff9",1.2);}
 const newIds=new Set(f.event?f.event.new_vertex_ids:[]);
 for(let i=0;i<points.length;i++){const p=points[i];ctx.beginPath();ctx.arc(p[0],p[1],newIds.has(f.vertex_ids[i])?5:3,0,Math.PI*2);ctx.fillStyle=f.pinned[i]?"#65d7bb":"#a1c8ea";ctx.fill();if(newIds.has(f.vertex_ids[i])){ctx.strokeStyle="#ff9c61";ctx.lineWidth=2;ctx.stroke();}if(el("showIds").checked){ctx.font="11px system-ui";ctx.fillStyle="#d7e5f5";ctx.fillText(String(f.vertex_ids[i])+" / r"+f.root_vertex_ids[i],p[0]+7,p[1]-7+(f.vertex_ids[i]!==f.root_vertex_ids[i]?19:0));}}
 ctx.fillStyle="#a4b6cd";ctx.font="12px system-ui";ctx.fillText("世界坐标 [m] · 视觉位移 ×"+mag,16,25);
 ctx.fillText("gap="+(f.event?f.event.gap:"—")+" m · "+f.stage,16,h-18);
}
function svgNode(name,attrs,text){const n=document.createElementNS("http://www.w3.org/2000/svg",name);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function chart(id,key,color){
 const svg=el(id);svg.replaceChildren();const w=480,h=160,left=54,right=14,top=14,bottom=30;
 const vals=frames.map(f=>f[key]).filter(v=>v!==null),hi=Math.max(1e-12,...vals)*1.08,tmax=Math.max(1e-12,frames[frames.length-1].time);
 const xy=(f,v)=>[left+(w-left-right)*f.time/tmax,h-bottom-(h-top-bottom)*v/hi];
 for(let k=0;k<3;k++){const v=hi*k/2,y=xy(frames[0],v)[1];svg.append(svgNode("line",{x1:left,y1:y,x2:w-right,y2:y,stroke:"#e4eaf2"}));svg.append(svgNode("text",{x:left-5,y:y+4,"text-anchor":"end","font-size":10,fill:"#738096"},v.toExponential(1)));}
 svg.append(svgNode("text",{x:left,y:h-8,"font-size":10,fill:"#738096"},"0"));svg.append(svgNode("text",{x:w-right,y:h-8,"text-anchor":"end","font-size":10,fill:"#738096"},tmax.toPrecision(4)+" s"));
 let path="",previous=null;
 for(const f of frames){const v=f[key];if(v===null){previous=null;continue;}const [x,y]=xy(f,v);const continuous=previous&&previous.version[0]===f.version[0]&&previous.version[2]===f.version[2];path+=(continuous?"L":"M")+x+","+y;svg.append(svgNode("circle",{cx:x,cy:y,r:2.5,fill:color}));previous=f;}
 svg.append(svgNode("path",{d:path,fill:"none",stroke:color,"stroke-width":1.8}));
 const f=frames[index],x=xy(f,0)[0];svg.append(svgNode("line",{x1:x,y1:top,x2:x,y2:h-bottom,stroke:"#929eb1","stroke-dasharray":"3 3"}));
 if(f[key]!==null){const q=xy(f,f[key]);svg.append(svgNode("circle",{cx:q[0],cy:q[1],r:4.5,fill:color,stroke:"white","stroke-width":1.5}));}
}
function update(){
 const f=frames[index];slider.value=String(index);document.body.dataset.record=String(index);document.body.dataset.stage=f.stage;
 el("recordLabel").textContent="记录 "+(index+1)+" / "+frames.length;
 el("stage").textContent={initial:"初始完成态",post_solve:"求解后 · 诊断有效",post_tear:"裂后事件 · 诊断缺失"}[f.stage];
 el("version").textContent=f.version[1]+" / "+f.version[2]+"  · epoch "+f.version[0];el("counts").textContent=f.vertex_ids.length+" / "+f.triangles.length;
 el("timeLabel").textContent="time "+f.time.toPrecision(7)+" s · solver 步长 "+f.h.toPrecision(4)+" s";
 el("event").textContent=f.event?(f.event.kind==="prescribed_cut"?"规定切割（非应变触发验收）":"应变阈值事件")+"\ncut IDs: "+f.event.cut_vertex_ids.join(" ↔ ")+"\ngap: "+f.event.gap+" m\n父 → 子: "+f.event.parent_vertex_ids.map((v,i)=>v+" → "+f.event.new_vertex_ids[i]).join(", "):"本记录没有新增切割事件。";
 const a=f.load?f.load.audit:null;el("audit").textContent=(a?"total: ["+vector(a.external_total)+"]\nfree: ["+vector(a.external_free)+"]\npinned: ["+vector(a.external_pinned)+"]\n抵消记账: ["+vector(a.pinned_external_cancellation)+"]":"本记录无载荷消费。")+"\n累计: ["+vector(f.cumulative_input_impulse_Ns)+"]";
 el("dispValue").textContent=fmt(f.max_displacement_m);el("energyValue").textContent=fmt(f.bend_energy_J);el("impulseValue").textContent=fmt(f.cumulative_input_impulse_norm_Ns);
 el("prev").disabled=el("first").disabled=index===0;el("next").disabled=el("last").disabled=index===frames.length-1;el("magLabel").textContent="×"+mag+"（仅视觉）";
 draw();chart("dispChart","max_displacement_m","#2862cf");chart("energyChart","bend_energy_J","#b36e31");chart("impulseChart","cumulative_input_impulse_norm_Ns","#288672");
}
function pause(){if(timer!==null)clearInterval(timer);timer=null;el("play").textContent="播放";}
function seek(i){pause();index=Math.max(0,Math.min(frames.length-1,i));update();}
el("first").onclick=()=>seek(0);el("last").onclick=()=>seek(frames.length-1);el("prev").onclick=()=>seek(index-1);el("next").onclick=()=>seek(index+1);slider.oninput=()=>seek(Number(slider.value));
el("play").onclick=()=>{if(timer!==null){pause();return;}if(index===frames.length-1)index=0;update();el("play").textContent="暂停";timer=setInterval(()=>{index++;update();if(index===frames.length-1)pause();},1000/Number(el("speed").value));};
el("speed").onchange=pause;el("magnify").oninput=()=>{mag=Number(el("magnify").value);bounds();update();};el("showIds").onchange=draw;
for(const [id,y,p]of [["viewOblique",.55,-.25],["viewFront",0,0],["viewSide",Math.PI/2,0]])el(id).onclick=()=>{yaw=y;pitch=p;for(const name of ["viewOblique","viewFront","viewSide"])el(name).setAttribute("aria-pressed",String(name===id));draw();};
window.addEventListener("resize",draw);bounds();update();
} catch(error){document.getElementById("error").textContent="回放无法读取："+error.message;}
</script></body></html>'''


def export_replay(archive, output):
    root = Path(archive).resolve(strict=True)
    target = Path(output).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("output must be outside the source archive")
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    before = source_fingerprint(root)
    run = load_run(root)
    payload = payload_from_run(run, before)
    try:
        encoded = encode_payload(payload)
    except (ValueError, TypeError) as error:
        raise ValueError(f"replay payload contains invalid JSON values: {error}") from error
    html = HTML.replace("@@DATA@@", encoded).encode("utf-8")
    _limit("MAX_HTML_BYTES", len(html), MAX_HTML_BYTES)
    if source_fingerprint(root) != before:
        raise ValueError("source archive changed during read")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".cloth-replay-", suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            out.write(html)
        # Same-directory hard link atomically publishes a fully written file,
        # fails if target now exists, and never replaces an existing file.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink()
    return dict(status="complete", records=len(run.records), schema=SCHEMA, bytes=len(html),
                output=str(target), source_files=before, html_sha256=sha256(target))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        summary = export_replay(args.archive, args.out)
    except ReplayLimitError as error:
        print(f"Replay limit: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Replay rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
