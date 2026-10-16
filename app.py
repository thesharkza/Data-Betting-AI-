import streamlit as st
import streamlit.components.v1 as components
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import json
import random
import pandas as pd
import numpy as np
import gc
import time
import threading
import uuid
from math import isfinite
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as esc
from sklearn.ensemble import RandomForestClassifier
 
# ----------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP & AI", page_icon="🪐", layout="wide")
 
# ----------------------------------------
# THEME : Deep Space / Mission Control
# ----------------------------------------
VOID = "#040806"
HULL = "#0a120d"
EDGE = "#1b3a2a"
INK = "#e6f2ea"
DIM = "#7f9a8b"
DEEP = "#15803d"
TEAL = "#2dd4bf"
EMERALD = "#4ade80"
LIME = "#a3e635"
AMBER = "#ffb454"
CORAL = "#ff5d7a"
 
 
def _stars(n, seed, colors):
    rnd = random.Random(seed)
    return ",".join(f"{rnd.randint(0, 100)}vw {rnd.randint(0, 100)}vh {rnd.choice(colors)}" for _ in range(n))
 
 
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
 
@property --a { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
 
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Prompt', 'Segoe UI', sans-serif !important;
}
 
/* ---------- Nebula + starfield ---------- */
.stApp {
    background-color: #040806;
    background-image:
        radial-gradient(1100px 650px at 88% -8%, rgba(34,197,94,0.30), transparent 62%),
        radial-gradient(900px 600px at -8% 28%, rgba(45,212,191,0.15), transparent 62%),
        radial-gradient(800px 520px at 55% 112%, rgba(16,185,129,0.12), transparent 62%);
    background-size: 130% 130%;
    background-attachment: fixed;
    animation: nebula 50s ease-in-out infinite alternate;
    color: #e6f2ea;
}
@keyframes nebula { from { background-position: 0% 0%; } to { background-position: 100% 100%; } }
.stApp::before {
    content: ""; position: fixed; top: 0; left: 0; width: 1px; height: 1px;
    box-shadow: __S1__; opacity: .9; pointer-events: none; z-index: 0;
    animation: twinkle 4s ease-in-out infinite alternate;
}
.stApp::after {
    content: ""; position: fixed; top: 0; left: 0; width: 2px; height: 2px; border-radius: 50%;
    box-shadow: __S2__; pointer-events: none; z-index: 0;
    animation: twinkle 6s ease-in-out -2s infinite alternate;
}
@keyframes twinkle { from { opacity: .25; } to { opacity: 1; } }
 
header[data-testid="stHeader"] { background: transparent; }
.block-container { position: relative; z-index: 1; padding-top: 1.6rem; max-width: 1280px; }
h1, h2, h3, h4 { color: #e6f2ea !important; letter-spacing: -0.01em; }
p, label, span { color: inherit; }
 
/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #14281d; padding-bottom: 2px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(10,18,13,0.5); color: #7f9a8b; padding: 10px 20px;
    border-radius: 12px 12px 0 0; font-weight: 500; transition: color .2s, background .2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #e6f2ea; background: rgba(34,197,94,0.14); }
.stTabs [aria-selected="true"] {
    color: #e6f2ea !important;
    background: linear-gradient(180deg, rgba(34,197,94,0.28), rgba(10,18,13,0.6));
}
.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #22c55e, #2dd4bf); height: 3px; box-shadow: 0 0 14px #22c55e;
}
 
/* ---------- Buttons ---------- */
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e, #2dd4bf); color: #03140a; border: none; font-weight: 600;
    border-radius: 12px; padding: 0.55rem 1.4rem; box-shadow: 0 8px 28px -10px rgba(34,197,94,0.8);
}
.stButton > button:not([kind="primary"]), [data-testid="stFormSubmitButton"] > button:not([kind="primary"]) {
    background: rgba(10,18,13,0.7); color: #d6eadf; border: 1px solid #1b3a2a; border-radius: 12px;
    transition: border-color .2s, box-shadow .2s, transform .2s;
}
.stButton > button:not([kind="primary"]):hover, [data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover {
    border-color: #2dd4bf; color: #86efac; box-shadow: 0 0 22px -6px rgba(45,212,191,0.7); transform: translateY(-1px);
}
 
/* ---------- Inputs / upload / alerts / forms ---------- */
[data-testid="stFileUploader"] section {
    background: rgba(10,18,13,0.55); border: 1px dashed #2f5a43; border-radius: 18px;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stFileUploader"] section:hover { border-color: #2dd4bf; box-shadow: 0 0 30px -10px rgba(45,212,191,0.6); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
    background-color: rgba(10,18,13,0.75) !important; border-color: #1b3a2a !important; border-radius: 10px !important;
}
[data-testid="stAlert"] { background: rgba(10,18,13,0.7); border: 1px solid #1b3a2a; border-radius: 14px; }
[data-testid="stForm"] {
    background: rgba(10,18,13,0.55); border: 1px solid #1b3a2a; border-radius: 16px;
    backdrop-filter: blur(6px);
}
 
/* ---------- Headings ---------- */
.panel-title { font-size: 1.1rem; font-weight: 600; color: #e6f2ea; margin: 0 0 2px 0; display: flex; align-items: center; gap: 10px; }
.panel-title::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #2dd4bf; box-shadow: 0 0 12px #2dd4bf; flex: none; }
.panel-sub { font-size: 0.86rem; color: #7f9a8b; margin: 0 0 8px 18px; }
 
/* ---------- Job cards ---------- */
.job-card {
    position: relative; overflow: hidden; background: rgba(10,18,13,0.65); border: 1px solid #1b3a2a;
    border-radius: 16px; padding: 14px 20px 16px; margin-bottom: 10px;
}
.job-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #ffb454; box-shadow: 0 0 14px #ffb454; }
.job-card.ok::before { background: #4ade80; box-shadow: 0 0 14px #4ade80; }
.job-card.err::before { background: #ff5d7a; box-shadow: 0 0 14px #ff5d7a; }
.job-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.job-name { color: #7f9a8b; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job-pill { font-size: 0.76rem; padding: 3px 11px; border-radius: 999px; white-space: nowrap; font-weight: 500; }
.job-pill.run { color: #ffd08a; background: rgba(255,180,84,0.14); }
.job-pill.ok { color: #4ade80; background: rgba(74,222,128,0.14); }
.job-pill.err { color: #ff8da1; background: rgba(255,93,122,0.14); }
.job-title { color: #e6f2ea; font-size: 1.05rem; font-weight: 600; margin: 8px 0 2px; }
.job-title .vs { color: #7f9a8b; font-weight: 400; padding: 0 6px; }
.job-msg { color: #a7c4b5; font-size: 0.9rem; }
.job-note { color: #7f9a8b; font-size: 0.8rem; margin-top: 6px; }
.job-bar { height: 3px; border-radius: 3px; margin-top: 12px; background: #12201a; overflow: hidden; }
.job-bar i { display: block; height: 100%; width: 40%; border-radius: 3px; background: linear-gradient(90deg, transparent, #2dd4bf, #22c55e, transparent); animation: sweep 1.4s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(260%); } }
 
/* ---------- Pending summary chips ---------- */
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 16px; }
.chip { background: rgba(10,18,13,0.7); border: 1px solid #1b3a2a; border-radius: 999px; padding: 6px 16px; font-size: 0.86rem; color: #a7c4b5; }
.chip b { color: #e6f2ea; font-weight: 600; margin-left: 8px; font-family: 'Space Grotesk', 'Prompt', sans-serif; }
 
/* ---------- Match cards ---------- */
.mc {
    position: relative; border-radius: 22px; padding: 20px 24px 22px; margin: 6px 0 10px;
    background: linear-gradient(160deg, rgba(14,32,22,0.82), rgba(6,12,9,0.82));
    box-shadow: 0 24px 60px -34px rgba(34,197,94,0.7); overflow: hidden;
}
.mc::before {
    content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px; pointer-events: none;
    background: conic-gradient(from var(--a), #1b3a2a 0 62%, #22c55e 80%, #2dd4bf 92%, #1b3a2a 100%);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    animation: spin 7s linear infinite;
}
@keyframes spin { to { --a: 360deg; } }
.mc-teams { display: flex; align-items: center; gap: 16px; margin-bottom: 18px; }
.mc-t { flex: 1; font-size: 1.08rem; font-weight: 600; color: #e6f2ea; line-height: 1.3; }
.mc-t.r { text-align: right; }
.mc-vs {
    position: relative; flex: none; width: 44px; height: 44px; display: grid; place-items: center;
    font-family: 'Space Grotesk', 'Prompt', sans-serif; font-size: .78rem; font-weight: 700; color: #b7e0c8;
}
.mc-vs::before {
    content: ""; position: absolute; inset: 0; border-radius: 50%; border: 1px dashed #2f7a52;
    animation: rot 12s linear infinite;
}
.mc-vs::after {
    content: ""; position: absolute; top: -3px; left: 50%; width: 6px; height: 6px; margin-left: -3px;
    border-radius: 50%; background: #2dd4bf; box-shadow: 0 0 10px #2dd4bf; transform-origin: 3px 25px;
    animation: rot 6s linear infinite;
}
@keyframes rot { to { transform: rotate(360deg); } }
.mc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.mc-mod {
    display: flex; align-items: center; gap: 16px; padding: 14px 16px; border-radius: 16px;
    background: rgba(3,8,5,0.62); border: 1px solid #14281d; transition: border-color .25s, box-shadow .25s;
}
.mc-mod:hover { border-color: var(--c); box-shadow: 0 0 26px -10px var(--c); }
.mc-ring {
    --p: 50; position: relative; flex: none; width: 88px; height: 88px; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), #12201a 0);
    animation: fillring 1.4s cubic-bezier(.2,.8,.2,1) both;
    filter: drop-shadow(0 0 10px color-mix(in srgb, var(--c) 45%, transparent));
}
.mc-ring::before { content: ""; position: absolute; inset: 9px; border-radius: 50%; background: #050a07; }
.mc-ring b { position: relative; font-family: 'Space Grotesk', 'Prompt', sans-serif; font-size: 1.35rem; color: #e6f2ea; font-weight: 700; }
.mc-ring small { font-size: .7rem; color: #7f9a8b; margin-left: 1px; }
@keyframes fillring { from { --p: 0; } }
.mc-info { min-width: 0; }
.mc-k { display: flex; gap: 10px; align-items: baseline; font-size: .76rem; color: #7f9a8b; letter-spacing: .02em; }
.mc-k span { color: #d6eadf; font-weight: 500; font-family: 'Space Grotesk', 'Prompt', sans-serif; font-size: .9rem; }
.mc-d { font-size: .8rem; color: #7f9a8b; margin: 3px 0 8px; }
.mc-v {
    display: inline-block; padding: 5px 12px; border-radius: 10px; font-size: .84rem; font-weight: 500; color: var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent);
}
 
@media (max-width: 900px) { .mc-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""".replace("__S1__", _stars(150, 7, ["#ffffff", "#d8ffe8", "#a7f3d0"])) \
   .replace("__S2__", _stars(45, 21, ["#ffffff", "#99f6e4", "#ffe9a8"]))
 
st.markdown(PAGE_CSS, unsafe_allow_html=True)
 
# ----------------------------------------
# กรอบ HTML สำหรับ Dashboard component (iframe)
# ----------------------------------------
_FRAME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--void:#040806;--hull:#0a120d;--edge:#1b3a2a;--ink:#e6f2ea;--dim:#7f9a8b;--deep:#15803d;--teal:#2dd4bf;--emer:#4ade80;--amber:#ffb454;--coral:#ff5d7a}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:transparent;color:var(--ink);font-family:'Prompt','Segoe UI',sans-serif;overflow-x:hidden}
body::-webkit-scrollbar{width:0}
.num{font-family:'Space Grotesk','Prompt',sans-serif;font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""
 
_FRAME_JS = """
const REDUCE = matchMedia('(prefers-reduced-motion:reduce)').matches;
function countUp(el){
  const v=parseFloat(el.dataset.v), d=parseInt(el.dataset.d||0), t0=performance.now(), D=1300;
  if(REDUCE){el.textContent=v.toFixed(d);return}
  (function f(t){const p=Math.min(1,(t-t0)/D), e=1-Math.pow(1-p,3); el.textContent=(v*e).toFixed(d); if(p<1) requestAnimationFrame(f)})(t0);
}
document.querySelectorAll('[data-v]').forEach(countUp);
document.querySelectorAll('[data-w]').forEach((e,i)=>setTimeout(()=>{e.style.width=e.dataset.w+'%'},150+i*90));
document.querySelectorAll('.tilt').forEach(card=>{
  card.addEventListener('mousemove',e=>{
    const r=card.getBoundingClientRect(), x=(e.clientX-r.left)/r.width-.5, y=(e.clientY-r.top)/r.height-.5;
    card.style.transform='perspective(700px) rotateX('+(-y*8)+'deg) rotateY('+(x*10)+'deg) translateY(-2px)';
    card.style.setProperty('--mx',(x+.5)*100+'%'); card.style.setProperty('--my',(y+.5)*100+'%');
  });
  card.addEventListener('mouseleave',()=>{card.style.transform=''});
});
"""
 
 
def _frame(css, body, js=""):
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>{_FRAME_CSS}{css}</style></head><body>{body}<script>{_FRAME_JS}{js}</script></body></html>")
 
 
# ----------------------------------------
# HERO : ดาวเคราะห์ + starfield warp + นาฬิกา
# ----------------------------------------
HERO_CSS = """
.hero{position:relative;height:176px;border-radius:24px;overflow:hidden;border:1px solid var(--edge);
  background:linear-gradient(120deg,rgba(14,32,22,.88),rgba(5,10,7,.72) 62%);box-shadow:0 30px 70px -40px rgba(34,197,94,.9)}
canvas{position:absolute;inset:0;width:100%;height:100%}
.in{position:relative;display:flex;align-items:center;gap:28px;height:100%;padding:0 32px}
.planet{position:relative;flex:none;width:116px;height:116px}
.sphere{position:absolute;inset:16px;border-radius:50%;z-index:1;
  background:radial-gradient(circle at 32% 28%,#bbf7d0,#22c55e 38%,#0f5132 72%,#03140a);
  box-shadow:0 0 54px rgba(34,197,94,.6),inset -14px -10px 26px rgba(0,0,0,.55)}
.rb,.rf{position:absolute;left:-16px;right:-16px;top:44px;height:30px;border-radius:50%;border:2px solid rgba(74,222,128,.6);transform:rotate(-18deg)}
.rb{z-index:0;border-bottom-color:transparent;box-shadow:0 0 14px rgba(45,212,191,.25)}
.rf{z-index:2;border-top-color:transparent;box-shadow:0 0 14px rgba(45,212,191,.35)}
.moon{position:absolute;inset:0;z-index:3;animation:orb 9s linear infinite}
.moon i{position:absolute;top:-2px;left:50%;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:var(--amber);box-shadow:0 0 14px var(--amber)}
@keyframes orb{to{transform:rotate(360deg)}}
.txt{flex:1;min-width:0}
h1{font-size:1.65rem;font-weight:600;letter-spacing:-.01em;line-height:1.25}
.sub{margin-top:8px;color:var(--dim);font-size:.95rem;min-height:1.4em}
.sub::after{content:"";display:inline-block;width:7px;height:1em;margin-left:3px;vertical-align:-2px;background:var(--teal);animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.tele{flex:none;text-align:right;display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.clock{font-size:1.6rem;font-weight:600;letter-spacing:.04em}
.clock small{display:block;font-size:.72rem;color:var(--dim);font-family:'Prompt',sans-serif;letter-spacing:0;font-weight:400}
.live{display:inline-flex;align-items:center;gap:8px;padding:5px 13px;border-radius:999px;font-size:.8rem;color:#bbf7d0;background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.3)}
.live i{width:8px;height:8px;border-radius:50%;background:var(--emer);box-shadow:0 0 0 0 rgba(74,222,128,.7);animation:ping 1.8s infinite}
@keyframes ping{70%{box-shadow:0 0 0 9px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
@media(max-width:700px){.tele{display:none}.planet{width:84px;height:84px}.sphere{inset:12px}.rb,.rf{top:30px;height:22px}h1{font-size:1.2rem}.in{padding:0 18px;gap:16px}.sub{font-size:.8rem}}
"""
 
HERO_BODY = """
<div class="hero">
  <canvas id="sky"></canvas>
  <div class="in">
    <div class="planet"><div class="rb"></div><div class="sphere"></div><div class="rf"></div><div class="moon"><i></i></div></div>
    <div class="txt"><h1>ระบบวิเคราะห์ราคาบอล VIP &amp; AI</h1><div class="sub" id="typed"></div></div>
    <div class="tele"><div class="clock num" id="clock">--:--:--<small>เวลาประเทศไทย</small></div><span class="live"><i></i>ระบบออนไลน์</span></div>
  </div>
</div>
"""
 
HERO_JS = """
const c=document.getElementById('sky'), x=c.getContext('2d'); let W,H,mx=.72,my=.5,d=1;
function fit(){const r=c.getBoundingClientRect(); d=window.devicePixelRatio||1; W=c.width=r.width*d; H=c.height=r.height*d}
fit(); addEventListener('resize',fit);
const S=Array.from({length:130},()=>({a:Math.random()*6.283,r:Math.random(),z:Math.random()*.9+.1,k:Math.random()<.25}));
document.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect(); mx=(e.clientX-r.left)/r.width; my=(e.clientY-r.top)/r.height});
function pos(s,r,cx,cy){return [cx+Math.cos(s.a)*r*W*.85, cy+Math.sin(s.a)*r*H*1.5]}
function draw(){
  x.clearRect(0,0,W,H); const cx=W*mx, cy=H*my;
  for(const s of S){
    if(!REDUCE){s.r+=.0035*s.z; if(s.r>1.15){s.r=.02;s.a=Math.random()*6.283}}
    const p=pos(s,s.r,cx,cy), q=pos(s,Math.max(0,s.r-.05*s.z),cx,cy);
    x.strokeStyle=(s.k?'rgba(134,239,172,':'rgba(220,255,235,')+Math.min(1,s.r*1.5)*s.z+')';
    x.lineWidth=Math.max(.6,s.z*1.6)*d; x.beginPath(); x.moveTo(q[0],q[1]); x.lineTo(p[0],p[1]); x.stroke();
  }
  if(!REDUCE) requestAnimationFrame(draw);
}
draw();
function tick(){document.getElementById('clock').firstChild.nodeValue=new Date().toLocaleTimeString('th-TH',{timeZone:'Asia/Bangkok',hour12:false})}
tick(); setInterval(tick,1000);
const T='สกัดราคาจากภาพ · วิเคราะห์ · ติดตามผล · พยากรณ์ด้วย Machine Learning แบบเรียลไทม์', el=document.getElementById('typed');
if(REDUCE){el.textContent=T}else{let i=0; (function t(){el.textContent=T.slice(0,++i); if(i<T.length) setTimeout(t,32)})()}
"""
 
# ----------------------------------------
# STAT CARDS
# ----------------------------------------
STATS_CSS = """
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:4px 2px}
.card{--c:var(--teal);--mx:50%;--my:50%;position:relative;overflow:hidden;border-radius:20px;padding:20px 22px 18px;min-height:138px;
  background:radial-gradient(260px circle at var(--mx) var(--my),color-mix(in srgb,var(--c) 18%,transparent),transparent 70%),linear-gradient(160deg,rgba(14,32,22,.85),rgba(6,12,9,.85));
  border:1px solid color-mix(in srgb,var(--c) 38%,#1b3a2a);transition:transform .18s ease-out,box-shadow .25s;
  box-shadow:0 24px 50px -34px var(--c)}
.card:hover{box-shadow:0 24px 60px -26px var(--c)}
.lab{font-size:.86rem;color:var(--dim)}
.val{display:flex;align-items:baseline;gap:6px;margin-top:6px}
.val b{font-size:2.9rem;font-weight:700;line-height:1.1;color:var(--ink);text-shadow:0 0 24px color-mix(in srgb,var(--c) 60%,transparent)}
.val em{font-style:normal;color:var(--dim);font-size:1rem}
.bar{height:6px;border-radius:6px;background:#12201a;margin-top:14px;overflow:hidden}
.bar i{display:block;height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,var(--deep),var(--c));transition:width 1.4s cubic-bezier(.2,.8,.2,1)}
.sat{position:absolute;right:12px;top:12px;width:58px;height:58px;color:var(--c);opacity:.9}
@media(max-width:640px){.val b{font-size:1.8rem}.card{padding:14px 14px 12px;min-height:110px}.lab{font-size:.74rem}.sat{width:36px;height:36px}.row{gap:8px}}
"""
 
 
def _stat_card(label, value, unit, color, pct=None, decimals=1):
    bar = f"<div class='bar'><i data-w='{min(float(pct), 100):g}'></i></div>" if pct is not None else ""
    return (f"<div class='card tilt' style='--c:{color}'><span class='lab'>{esc(label)}</span>"
            f"<div class='val'><b class='num' data-v='{value}' data-d='{decimals}'>0</b><em>{esc(unit)}</em></div>{bar}"
            "<svg class='sat' viewBox='0 0 60 60'><circle cx='30' cy='30' r='24' fill='none' stroke='currentColor' "
            "stroke-opacity='.3' stroke-dasharray='2 5'/><circle r='3.2' fill='currentColor'>"
            "<animateMotion dur='6s' repeatCount='indefinite' path='M30,6 A24,24 0 1,1 29.99,6'/></circle></svg></div>")
 
 
def render_stats_strip(hdp_acc, ou_acc, total_train):
    body = ("<div class='row'>"
            + _stat_card("เรียนรู้จากข้อมูลผลลัพธ์แล้ว", total_train, "คู่", LIME, decimals=0)
            + _stat_card("ความแม่นยำ AI (แฮนดิแคป)", hdp_acc, "%", EMERALD, pct=hdp_acc)
            + _stat_card("ความแม่นยำ AI (สูง-ต่ำ)", ou_acc, "%", TEAL, pct=ou_acc)
            + "</div>")
    components.html(_frame(STATS_CSS, body), height=172)
 
 
# ----------------------------------------
# MODEL PANEL : วงแหวนความแม่นยำ + ปัจจัยสำคัญ
# ----------------------------------------
PANEL_CSS = """
.panel{--c:var(--teal);border-radius:22px;padding:20px 24px 18px;background:linear-gradient(160deg,rgba(14,32,22,.8),rgba(6,12,9,.8));
  border:1px solid #1b3a2a;box-shadow:0 28px 60px -40px var(--c);min-height:330px}
.ph{font-size:1.02rem;font-weight:600}
.ps{font-size:.82rem;color:var(--dim);margin-top:2px}
.body{display:flex;flex-wrap:wrap;align-items:center;gap:22px;margin-top:14px}
.gauge{position:relative;flex:none;width:188px;height:188px;margin:0 auto}
.gauge svg{width:100%;height:100%;overflow:visible}
.center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.center b{font-size:2.3rem;font-weight:700;text-shadow:0 0 22px color-mix(in srgb,var(--c) 60%,transparent)}
.center span{font-size:.78rem;color:var(--dim)}
#arc{transition:stroke-dasharray 1.4s cubic-bezier(.2,.8,.2,1)}
.legend{display:flex;gap:16px;justify-content:center;margin-top:10px;font-size:.78rem;color:var(--dim)}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:0}
.bars{flex:1;min-width:230px;display:flex;flex-direction:column;gap:12px}
.bt{font-size:.78rem;color:var(--dim);margin-bottom:-2px}
.r{display:grid;grid-template-columns:1fr auto;gap:4px 10px;padding:6px 8px;border-radius:10px;transition:background .2s}
.r:hover{background:rgba(34,197,94,.12)}
.n{font-size:.86rem;color:#d6eadf}
.v{font-size:.84rem;color:var(--c);font-weight:600}
.t{grid-column:1/3;height:6px;border-radius:6px;background:#12201a;overflow:hidden}
.t i{display:block;height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,var(--deep),var(--c));transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
"""
 
FEATURE_LABELS = {
    'Expected_Total_Goals': 'ประตูรวมที่คาดหวัง', 'OU_Goal_Diff': 'ส่วนต่างโกลที่คาด', 'Total_Goals_Conceded_Last5': 'เสียประตู 5 นัดรวม',
    'H_GoalsFor_Last5': 'เหย้ายิงได้ 5 นัด', 'H_GoalsAgainst_Last5': 'เหย้าเสีย 5 นัด', 'A_GoalsFor_Last5': 'เยือนยิงได้ 5 นัด', 'A_GoalsAgainst_Last5': 'เยือนเสีย 5 นัด',
    'H_WinRate_Last5': 'WinRate เหย้า 5 นัด', 'A_WinRate_Last5': 'WinRate เยือน 5 นัด', '1X2 เหย้า': 'ราคา 1X2 เหย้า', '1X2 เยือน': 'ราคา 1X2 เยือน',
    'แฮนดิแคป (เหย้า)': 'แต้มต่อ HDP', 'ค่าน้ำ HDP (เหย้า)': 'ค่าน้ำ HDP', 'โกลสูง/ต่ำ': 'ไลน์สูง-ต่ำ', 'ค่าน้ำ (สูง)': 'ค่าน้ำสูง', 'ค่าน้ำ (ต่ำ)': 'ค่าน้ำต่ำ'
}
 
 
def top_features(model, feature_names, n=5):
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(n)
    df_imp['Feature'] = df_imp['Feature'].replace(FEATURE_LABELS)
    return list(zip(df_imp['Feature'], df_imp['Importance']))[::-1]
 
 
def render_model_panel(title, sub, color, correct, total, feats, height=372):
    acc = round((correct / total) * 100, 1) if total > 0 else 0
    circ = 490.088
    arc_len = circ * (correct / total) if total > 0 else 0
    top = max((float(v) for _, v in feats), default=1) or 1
    rows = "".join(
        f"<div class='r'><span class='n'>{esc(str(name))}</span><span class='v num'>{float(v) * 100:.1f}%</span>"
        f"<div class='t'><i data-w='{float(v) / top * 100:.1f}'></i></div></div>" for name, v in feats)
    body = (f"<div class='panel' style='--c:{color}'><div class='ph'>{esc(title)}</div><div class='ps'>{esc(sub)}</div>"
            "<div class='body'><div>"
            "<div class='gauge'><svg viewBox='0 0 200 200'>"
            "<circle cx='100' cy='100' r='92' fill='none' stroke='#2a4a39' stroke-opacity='.55' stroke-dasharray='2 6'/>"
            "<circle cx='100' cy='100' r='78' fill='none' stroke='#ff5d7a' stroke-opacity='.55' stroke-width='14'/>"
            f"<circle id='arc' data-l='{arc_len:.2f}' cx='100' cy='100' r='78' fill='none' stroke='{color}' stroke-width='14' "
            "stroke-dasharray='0 999' style='transform:rotate(-90deg);transform-origin:100px 100px;filter:drop-shadow(0 0 8px " + color + ")'/>"
            f"<g><circle cx='100' cy='8' r='4' fill='{color}'/><animateTransform attributeName='transform' type='rotate' from='0 100 100' to='360 100 100' dur='9s' repeatCount='indefinite'/></g>"
            "</svg>"
            f"<div class='center'><b class='num'><span data-v='{acc}' data-d='1'>0</span>%</b><span>ทายถูก</span></div></div>"
            f"<div class='legend'><span><i style='background:{color}'></i>ถูก {correct} คู่</span>"
            f"<span><i style='background:#ff5d7a'></i>ผิด/เจ๊า {total - correct} คู่</span></div></div>"
            f"<div class='bars'><div class='bt'>ปัจจัยที่ AI ใช้ตัดสินใจมากที่สุด 5 อันดับ</div>{rows}</div></div></div>")
    js = "const arc=document.getElementById('arc'); setTimeout(()=>{arc.style.strokeDasharray=arc.dataset.l+' 999'},120);"
    components.html(_frame(PANEL_CSS, body, js), height=height, scrolling=True)
 
 
def render_hero():
    components.html(_frame(HERO_CSS, HERO_BODY, HERO_JS), height=182)
 
 
# ----------------------------------------
# การตั้งค่า API และ Google Sheets
# ----------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("❌ ไม่พบ GEMINI_API_KEY กรุณาตั้งค่าใน Streamlit Secrets")
 
def _has_gspread_secret():
    try: return "gspread" in st.secrets
    except Exception: return False
 
@st.cache_resource
def init_gspread():
    if _has_gspread_secret():
        client = gspread.service_account_from_dict(dict(st.secrets["gspread"]))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        client = gspread.service_account(filename=os.path.join(base_dir, 'credentials.json'))
    return client.open("ข้อมูลราคาบอลสกัดจากภาพ")
 
try:
    workbook = init_gspread()
except Exception as e:
    st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
    workbook = None
 
def get_ws(wb):
    try: return wb.worksheet("DATA")
    except gspread.exceptions.WorksheetNotFound: return wb.sheet1
 
model = genai.GenerativeModel('gemini-3.7-flash')
 
WATER_IDX = [6, 8, 10, 11]
FMT_LABELS = {"MY": "มาเลเซีย", "HK": "ฮ่องกง", "ID": "อินโดนีเซีย", "EU": "ยุโรป (Decimal)"}
FMT_CHOICES = { "อัตโนมัติ (ให้ AI ตรวจจับ)": None, "มาเลเซีย (Malay)": "MY", "ฮ่องกง (HK)": "HK", "อินโดนีเซีย (Indo)": "ID", "ยุโรป (Decimal)": "EU" }
 
def to_malay(x, fmt):
    x = float(x)
    if fmt == "MY": return round(x, 2)
    if fmt == "EU": hk = x - 1
    elif fmt == "ID": hk = x if x > 0 else (1 / abs(x) if x != 0 else 0)
    else: hk = x
    return round(hk, 2) if hk <= 1 else round(-1 / hk, 2)
 
def guess_format(nums):
    if nums and all(v > 1 for v in nums): return "EU"
    if any(v < -1 for v in nums): return "ID"
    return "MY"
 
def parse_cell(v):
    cleaned = re.sub(r'^[oOuU\s]+', '', str(v).strip().replace('−', '-'))
    try: return float(cleaned) if '.' in cleaned else int(cleaned)
    except Exception: return cleaned
 
# ฟังก์ชันแปลงข้อมูล AI แบบใหม่ ล็อกโครงสร้างให้อ่านผ่าน JSON
def clean_raw_data(raw_text):
    text = raw_text.strip()
    match = re.search(r'\[(.*)\]', text, re.DOTALL)
    if not match:
        raise ValueError("ไม่พบข้อมูลรูปแบบ JSON จาก AI")
        
    try:
        data = json.loads("[" + match.group(1) + "]")
    except Exception as e:
        # ลองคลีนคอมม่าเผื่อ AI พิมพ์เกิน
        cleaned_str = match.group(1).strip()
        if cleaned_str.endswith(','): cleaned_str = cleaned_str[:-1]
        try:
            data = json.loads("[" + cleaned_str + "]")
        except:
            raise ValueError(f"รูปแบบข้อมูลผิดพลาด: {e}")
        
    if len(data) < 12:
        raise ValueError(f"ข้อมูลมาไม่ครบ (ได้มา {len(data)} ช่อง จาก 12 ช่องที่ต้องการ)")
        
    home = str(data[0]).strip()
    away = str(data[1]).strip()
    
    numbers = []
    for i in range(2, 12):
        val_str = str(data[i]).replace('−', '-').replace(',', '').strip()
        try: numbers.append(float(val_str))
        except: numbers.append(0.0)
        
    ai_fmt = str(data[12]).strip().upper() if len(data) > 12 else "UNK"
    if ai_fmt not in ["MY", "HK", "ID", "EU"]: ai_fmt = "UNK"
    
    # บังคับล็อกแถวให้มี 12 คอลัมน์เป๊ะๆ ป้องกันการไหลไปทับช่อง M, N, O เด็ดขาด
    row_data = [home, away] + numbers
    return row_data, ai_fmt
 
def convert_water_to_malay(row, user_fmt, ai_fmt):
    new_row = list(row)
    idx = [i for i in WATER_IDX if i < len(new_row) and isinstance(new_row[i], (int, float))]
    if not idx: return new_row, "ℹ️ ไม่พบค่าน้ำที่เป็นตัวเลข", ""
    fmt, source = (user_fmt, "คุณเลือกเอง") if user_fmt else (ai_fmt, "AI ตรวจพบ")
    if fmt not in FMT_LABELS:
        fmt, source = guess_format([new_row[i] for i in idx]), "ระบบประเมินเอง"
    for i in idx: new_row[i] = to_malay(new_row[i], fmt)
    return new_row, f"อิงรูปแบบ: {FMT_LABELS.get(fmt, fmt)} ({source})", None
 
FORM5_HEADERS = ["H_WinRate_Last5", "H_GoalsFor_Last5", "H_GoalsAgainst_Last5", "A_WinRate_Last5", "A_GoalsFor_Last5", "A_GoalsAgainst_Last5"]
 
def validate_form5(values):
    if len(values) != 6 or any(v is None or v == "" or isinstance(v, bool) for v in values):
        raise ValueError('กรุณากรอกข้อมูลให้ครบ 6 ช่อง')
    nums = [float(v) for v in values]
    return [nums[0] / 100, int(nums[1]), int(nums[2]), nums[3] / 100, int(nums[4]), int(nums[5])]
 
def save_row_to_sheet(ws, row, form5):
    normalized = validate_form5(form5)
    col_a, col_b = ws.col_values(1), ws.col_values(2)
    last_row = max((i + 1 for i, v in enumerate(col_a) if str(v).strip()), default=1)
    new_home, new_away = str(row[0]).strip(), str(row[1]).strip()
    target_row = next((i + 1 for i in range(1, len(col_a)) if str(col_a[i]).strip() == new_home and (str(col_b[i]).strip() if i < len(col_b) else "") == new_away), None)
    overwritten = target_row is not None
    if not overwritten: target_row = max(2, last_row + 1)
 
    requests = []
    if ws.col_count < 29: requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "COLUMNS", "length": 29 - ws.col_count}})
    if target_row > ws.row_count: requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "ROWS", "length": target_row - ws.row_count}})
 
    def typed(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool): return {"userEnteredValue": {"numberValue": value}}
        return {"userEnteredValue": {"stringValue": str(value)}}
 
    def update(start_row, start_col, cells, fields):
        return {"updateCells": {"start": {"sheetId": ws.id, "rowIndex": start_row, "columnIndex": start_col}, "rows": [{"values": cells}], "fields": fields}}
 
    requests.append(update(0, 23, [typed(h) for h in FORM5_HEADERS], "userEnteredValue"))
    
    # อัปเดตข้อมูลบอล 12 ช่อง ห้ามเกิน ป้องกันการทับคอลัมน์ M
    requests.append(update(target_row - 1, 0, [typed(v) for v in row[:12]], "userEnteredValue"))
    
    form_cells = [typed(v) for v in normalized]
    for i, cell in enumerate(form_cells):
        cell["userEnteredFormat"] = {"numberFormat": {"type": "PERCENT" if i in (0, 3) else "NUMBER", "pattern": "0.0%" if i in (0, 3) else "0"}}
    requests.append(update(target_row - 1, 23, form_cells, "userEnteredValue,userEnteredFormat.numberFormat"))
    ws.spreadsheet.batch_update({"requests": requests})
    return target_row, overwritten
 
def update_match_score_in_sheet(ws, home_team, away_team, score_text):
    col_a = ws.col_values(1)
    col_b = ws.col_values(2)
    target_row = None
    for i in range(1, len(col_a)):
        if str(col_a[i]).strip() == str(home_team).strip() and str(col_b[i]).strip() == str(away_team).strip():
            target_row = i + 1
            break
    if target_row:
        ws.update(f'M{target_row}', [[score_text]])
        return True
    return False
 
class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.sheet_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.data_version = 0
 
    def create(self, name, img_bytes, user_fmt, owner_id):
        job_id = uuid.uuid4().hex[:8]
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "name": name, "status": "queued", "msg": "รอคิว...", "created": time.time(), "finished": None, "img": img_bytes, "user_fmt": user_fmt, "result": None, "note": None, "warn": None, "owner_id": owner_id, "row_data": None, "form5": None}
        return job_id
    def get(self, job_id):
        with self.lock: return dict(self.jobs.get(job_id)) if job_id in self.jobs else None
    def update(self, job_id, **kw):
        with self.lock:
            if job_id in self.jobs: self.jobs[job_id].update(kw)
    def bump_version(self):
        with self.lock: self.data_version += 1
    def snapshot(self):
        with self.lock: return sorted((dict(j) for j in self.jobs.values()), key=lambda j: j["created"], reverse=True)
    def clear_finished(self, owner_id):
        with self.lock:
            for k in [k for k, j in self.jobs.items() if j["status"] in ("done", "error") and j.get("owner_id") == owner_id]: self.jobs.pop(k)
 
@st.cache_resource
def get_job_manager(): return JobManager()
 
JOBS = get_job_manager()
if "form5_owner_id" not in st.session_state: st.session_state["form5_owner_id"] = uuid.uuid4().hex
 
def extract_row(img_bytes, user_fmt, model):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    
    # คำสั่งใหม่: บังคับให้ AI ตอบกลับเป็นโค้ด JSON Array ห้ามอธิบาย
    prompt = (
        "สกัดข้อมูลจากตารางราคาบอลในภาพนี้ เลือกมา 1 คู่ที่เห็นชัดที่สุด\n"
        "**คำสั่งสำคัญ: ต้องตอบกลับเป็น JSON Array โค้ดดิบๆ เท่านั้น ห้ามพิมพ์ข้อความอธิบายใดๆ ทั้งสิ้น**\n"
        "รูปแบบโครงสร้างที่ต้องการ:\n"
        "[\"ชื่อทีมเหย้า\", \"ชื่อทีมเยือน\", 1X2เหย้า, 1X2เสมอ, 1X2เยือน, HDPเหย้า, น้ำHDPเหย้า, HDPเยือน, น้ำHDPเยือน, โกลสูงต่ำ, น้ำสูง, น้ำต่ำ, \"รูปแบบน้ำเช่น MY\"]\n"
        "ตัวอย่างผลลัพธ์:\n"
        "[\"Mardin 1969 Spor\", \"Karagumruk\", 2.10, 3.50, 3.20, -0.5, 0.95, 0.5, -0.85, 2.5, 0.80, -0.90, \"MY\"]"
    )
    
    for attempt in range(3):
        try:
            resp = model.generate_content([{"mime_type": "image/jpeg", "data": buf.getvalue()}, prompt])
            row, ai_fmt = clean_raw_data(resp.text.strip())
            return convert_water_to_malay(row, user_fmt, ai_fmt)
        except Exception: time.sleep(2)
    raise RuntimeError("อ่านภาพไม่สำเร็จ (AI ตอบผิดรูปแบบโครงสร้าง)")
 
def process_job(job_id, workbook, model):
    try:
        job = JOBS.get(job_id)
        JOBS.update(job_id, status="reading", msg="🤖 AI กำลังอ่านราคา...")
        row, note, warn = extract_row(job["img"], job["user_fmt"], model)
        JOBS.update(job_id, status="ready", msg='อ่านราคาแล้ว', row_data=list(row), note=note, warn=warn, img=None)
    except Exception as e:
        JOBS.update(job_id, status="error", msg=f"{e}", finished=time.time())
    finally: gc.collect()
 
def save_confirmed_job(job_id, workbook):
    job = JOBS.get(job_id)
    if not job or job["status"] != "saving": return
    try:
        with JOBS.sheet_lock:
            ws = get_ws(workbook)
            target_row, overwritten = save_row_to_sheet(ws, job["row_data"], job["form5"])
        JOBS.bump_version()
        JOBS.update(job_id, status="done", msg="บันทึกสำเร็จ", finished=time.time(), result={"home": str(job["row_data"][0]), "away": str(job["row_data"][1]), "row": target_row, "overwritten": overwritten})
    except Exception as exc:
        JOBS.update(job_id, status="error", msg=f"Save error: {exc}", finished=time.time())
 
def submit_form5(job_id, values, owner_id):
    job = JOBS.get(job_id)
    if job and job.get("owner_id") == owner_id:
        JOBS.update(job_id, status="saving", msg="Saving...", form5=list(values))
        JOBS.pool.submit(save_confirmed_job, job_id, workbook)
 
# ----------------------------------------
# UI : รายการงาน + ฟอร์ม 5 นัด
# ----------------------------------------
STATUS_UI = {"queued": ("รอคิว", "run"), "reading": ("กำลังอ่านภาพ", "run"), "ready": ("อ่านราคาแล้ว", "ok"),
             "saving": ("กำลังบันทึก", "run"), "done": ("บันทึกแล้ว", "ok"), "error": ("ไม่สำเร็จ", "err")}
 
def panel_header(title, sub): st.markdown(f"<p class='panel-title'>{title}</p><p class='panel-sub'>{sub}</p>", unsafe_allow_html=True)
 
def render_jobs():
    jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if not jobs: return
    for j in jobs:
        cls = "ok" if j["status"] == "done" else "err" if j["status"] == "error" else "run"
        label, pill_cls = STATUS_UI.get(j["status"], (j["status"], "run"))
        head = (f"<div class='job-head'><span class='job-name'>{esc(j['name'])}</span>"
                f"<span class='job-pill {pill_cls}'>{label}</span></div>")
        pair = j.get("row_data")
        res = j.get("result")
        if j["status"] == "done" and res:
            where = "บันทึกทับแถวเดิม" if res.get("overwritten") else "คู่ใหม่"
            title = (f"<div class='job-title'>{esc(res['home'])}<span class='vs'>vs</span>{esc(res['away'])}"
                     f"<span class='job-msg'> · แถว {res['row']} ({where})</span></div>")
        elif pair:
            title = f"<div class='job-title'>{esc(str(pair[0]))}<span class='vs'>vs</span>{esc(str(pair[1]))}</div>"
        else:
            title = ""
        body = title + f"<div class='job-msg'>{esc(j['msg'])}</div>"
        if j.get("note"): body += f"<div class='job-note'>{esc(str(j['note']))}</div>"
        if j.get("warn"): body += f"<div class='job-note' style='color:#ffd08a'>{esc(str(j['warn']))}</div>"
        if j["status"] in ("queued", "reading", "saving"): body += "<div class='job-bar'><i></i></div>"
        st.markdown(f"<div class='job-card {cls}'>{head}{body}</div>", unsafe_allow_html=True)
        if j["status"] == "ready":
            if st.button('กรอกฟอร์ม 5 นัด', key=f"open_form5_{j['id']}"):
                st.session_state["form5_editing"] = j["id"]
                st.rerun()
 
def render_form5_editor():
    job_id = st.session_state.get("form5_editing")
    job = JOBS.get(job_id) if job_id else None
    if not job or job["status"] != "ready": return
    panel_header("ฟอร์ม 5 นัดล่าสุด", f"{esc(str(job['row_data'][0]))} vs {esc(str(job['row_data'][1]))}")
    with st.form(f"form5_{job_id}"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"**เหย้า** · {job['row_data'][0]}")
            h_win = st.number_input('Win H', step=20.)
            h_for = st.number_input('GF H', step=1)
            h_ag = st.number_input('GA H', step=1)
        with right:
            st.markdown(f"**เยือน** · {job['row_data'][1]}")
            a_win = st.number_input('Win A', step=20.)
            a_for = st.number_input('GF A', step=1)
            a_ag = st.number_input('GA A', step=1)
        if st.form_submit_button('บันทึก', type="primary"):
            submit_form5(job_id, [h_win, h_for, h_ag, a_win, a_for, a_ag], st.session_state["form5_owner_id"])
            st.session_state.pop("form5_editing", None)
            st.rerun()
 
@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_data(_workbook, version=0):
    ws = get_ws(_workbook)
    return pd.DataFrame(ws.get_all_records())
 
# ----------------------------------------
# UI ฟันธง + ช่องกรอกผลสกอร์
# ----------------------------------------
def _pred_tone(text, kind):
    text = str(text)
    if "ข้าม" in text: return "skip"
    if kind == "hdp": return ("cs" if "VIP" in text else "cf") if "รอง" in text else ("gs" if "VIP" in text else "gf")
    return "cs" if "มั่นใจ" in text and "ต่ำ" in text else "cf" if "ต่ำ" in text else "gs" if "มั่นใจ" in text else "gf"
 
_TONE_COLORS = {
    "hdp": {"gs": EMERALD, "gf": "#a7f3c0", "cs": "#fde047", "cf": "#fef08a", "skip": "#4f6b5c"},
    "ou": {"gs": TEAL, "gf": "#99f6e4", "cs": AMBER, "cf": "#ffd08a", "skip": "#4f6b5c"},
}
 
def _fmt_num(v, signed=False):
    try: return f"{float(v):+g}" if signed else f"{float(v):g}"
    except (TypeError, ValueError): return "-"
 
def _match_card_html(home, away, hdp_line, ou_line, hdp_prob, ou_prob, hdp_verdict, ou_verdict):
    c1 = _TONE_COLORS["hdp"][_pred_tone(hdp_verdict, "hdp")]
    c2 = _TONE_COLORS["ou"][_pred_tone(ou_verdict, "ou")]
 
    def mod(kind, line_txt, prob, verdict, color, desc):
        return (f"<div class='mc-mod' style='--c:{color}'>"
                f"<div class='mc-ring' style='--p:{prob:g}'><b>{prob:g}<small>%</small></b></div>"
                f"<div class='mc-info'><div class='mc-k'>{kind}<span>{esc(line_txt)}</span></div>"
                f"<div class='mc-d'>{desc}</div><span class='mc-v'>{esc(verdict)}</span></div></div>")
 
    return ("<div class='mc'>"
            f"<div class='mc-teams'><span class='mc-t'>{esc(home)}</span><span class='mc-vs'>VS</span><span class='mc-t r'>{esc(away)}</span></div>"
            "<div class='mc-grid'>"
            + mod("HANDICAP", f"เหย้า {_fmt_num(hdp_line, True)}", hdp_prob, hdp_verdict, c1, "โอกาสเหย้าชนะแฮนดิแคป")
            + mod("OVER / UNDER", f"ไลน์ {_fmt_num(ou_line)}", ou_prob, ou_verdict, c2, "โอกาสออกสูง")
            + "</div></div>")
 
def render_prediction_cards(df_pred, workbook):
    for idx, r in df_pred.iterrows():
        home = str(r.get('ทีมเหย้า', '-'))
        away = str(r.get('ทีมเยือน', '-'))
        hdp_prob = float(r.get('AI_โอกาสเหย้าชนะHDP(%)', 50))
        ou_prob = float(r.get('AI_โอกาสออกสูง(%)', 50))
        hdp_verdict = str(r.get('AI_ฟันธง_HDP', '-'))
        ou_verdict = str(r.get('AI_ฟันธง_สูงต่ำ', '-'))
        hdp_line = r.get('แฮนดิแคป (เหย้า)', 0)
        ou_line = r.get('โกลสูง/ต่ำ', 0)
 
        with st.container():
            st.markdown(_match_card_html(home, away, hdp_line, ou_line, hdp_prob, ou_prob, hdp_verdict, ou_verdict),
                        unsafe_allow_html=True)
 
            with st.form(key=f"score_form_{idx}_{home}_{away}"):
                sc1, sc2, sc3 = st.columns([2, 1, 1])
                with sc1: score_input = st.text_input("กรอกผลสกอร์ (เช่น 2-1)", placeholder="ทีมเหย้า-ทีมเยือน", key=f"score_input_{idx}")
                with sc3:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("💾 บันทึกผลลงชีต")
                
                if submitted:
                    if score_input.strip():
                        success = update_match_score_in_sheet(get_ws(workbook), home, away, score_input.strip())
                        if success:
                            st.success(f"บันทึกผลสกอร์ {score_input} ของ {home} vs {away} เรียบร้อยแล้ว!")
                            time.sleep(1)
                            st.rerun()
                        else: st.error("ไม่พบคู่นี้ใน Google Sheets กรุณาตรวจสอบชื่อทีม")
                    else: st.warning("กรุณากรอกผลสกอร์ก่อนกดบันทึก")
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
 
# ----------------------------------------
# หัวข้อหลัก
# ----------------------------------------
render_hero()
 
tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "🧠 AI Dashboard (สถิติ & ทำนาย)"])
 
with tab1:
    panel_header("สกัดราคาบอลจากภาพ", "ลากรูปตารางราคามาวางได้หลายรูป AI จะอ่านราคาแล้วรอให้กรอกฟอร์ม 5 นัดก่อนบันทึก")
    fmt_choice = st.selectbox("รูปแบบค่าน้ำ", list(FMT_CHOICES.keys()))
    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{up_n}")
    if uploaded_files and workbook:
        for f in uploaded_files: JOBS.pool.submit(process_job, JOBS.create(f.name, f.getvalue(), FMT_CHOICES[fmt_choice], st.session_state["form5_owner_id"]), workbook, model)
        st.session_state["up_n"] = up_n + 1
        st.rerun()
    if hasattr(st, "fragment"): st.fragment(run_every=2)(render_jobs)()
    else: render_jobs()
    render_form5_editor()
 
# ----------------------------------------
# TAB 2 : AI DASHBOARD
# ----------------------------------------
with tab2:
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        if st.button("🔄 รีเฟรชข้อมูล"): 
            st.cache_data.clear()
            st.rerun()
 
        df = load_dashboard_data(workbook, JOBS.data_version)
        
        numeric_cols_raw = ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)',
                            'H_WinRate_Last5', 'H_GoalsFor_Last5', 'H_GoalsAgainst_Last5', 'A_WinRate_Last5', 'A_GoalsFor_Last5', 'A_GoalsAgainst_Last5']
        
        numeric_cols = [c for c in numeric_cols_raw if c in df.columns]
        
        for col in numeric_cols:
            if df[col].dtype == object: df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
 
        if all(c in df.columns for c in ['H_GoalsFor_Last5', 'A_GoalsAgainst_Last5', 'A_GoalsFor_Last5', 'H_GoalsAgainst_Last5', 'โกลสูง/ต่ำ']):
            df['Exp_Goals_Home'] = (df['H_GoalsFor_Last5'] + df['A_GoalsAgainst_Last5']) / 10.0
            df['Exp_Goals_Away'] = (df['A_GoalsFor_Last5'] + df['H_GoalsAgainst_Last5']) / 10.0
            df['Expected_Total_Goals'] = df['Exp_Goals_Home'] + df['Exp_Goals_Away']
            df['OU_Goal_Diff'] = df['Expected_Total_Goals'] - df['โกลสูง/ต่ำ']
            df['Total_Goals_Conceded_Last5'] = df['H_GoalsAgainst_Last5'] + df['A_GoalsAgainst_Last5']
 
        if 'ผล HDP' in df.columns and 'ผลสูง/ต่ำ' in df.columns:
            def map_hdp_result(res):
                res = str(res).strip()
                if res == "" or res.lower() in ("nan", "none"): return np.nan
                if 'เจ้าบ้านได้เต็ม' in res or 'เจ้าบ้านได้ครึ่ง' in res: return 1
                if 'เจ้าบ้านเสียเต็ม' in res or 'เจ้าบ้านเสียครึ่ง' in res or 'เจ๊า' in res: return 0
                return np.nan 
 
            def map_ou_result(res):
                res = str(res).strip()
                if res == "" or res.lower() in ("nan", "none"): return np.nan
                if 'สูงได้เต็ม' in res or 'สูงได้ครึ่ง' in res: return 1
                if 'ต่ำได้เต็ม' in res or 'ต่ำได้ครึ่ง' in res or 'เจ๊า' in res: return 0
                return np.nan 
 
            df['Target_HDP'] = df['ผล HDP'].apply(map_hdp_result)
            df['Target_OU'] = df['ผลสูง/ต่ำ'].apply(map_ou_result)
            
            if 'ผลสกอร์' in df.columns:
                df_predict = df[df['ผลสกอร์'].astype(str).str.strip().isin(["", "nan", "None"]) & df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()
            else:
                df_predict = df[df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()
 
            core_features = ['1X2 เหย้า', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'Target_HDP', 'Target_OU']
            missing_cols = [c for c in core_features if c not in df.columns]
            
            if missing_cols:
                st.info(f"ℹ️ ระบบกำลังรอข้อมูลผลบอล หรือไม่พบคอลัมน์: {', '.join(missing_cols)} ในชีตของคุณ")
            else:
                df_train = df.dropna(subset=core_features).copy()
                
                if len(df_train) > 5:
                    features = numeric_cols.copy()
                    new_ou_features = ['Expected_Total_Goals', 'OU_Goal_Diff', 'Total_Goals_Conceded_Last5']
                    if all(c in df_train.columns for c in new_ou_features): features.extend(new_ou_features)
                    
                    train_medians = df_train[features].median().fillna(0)
                    X_train = df_train[features].fillna(train_medians)
                    
                    model_hdp = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                    model_hdp.fit(X_train, df_train['Target_HDP'])
                    model_ou = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                    model_ou.fit(X_train, df_train['Target_OU'])
 
                    hdp_correct = (model_hdp.predict(X_train) == df_train['Target_HDP']).sum()
                    ou_correct = (model_ou.predict(X_train) == df_train['Target_OU']).sum()
                    total_train = len(df_train)
                    
                    hdp_acc = round((hdp_correct / total_train) * 100, 1) if total_train > 0 else 0
                    ou_acc = round((ou_correct / total_train) * 100, 1) if total_train > 0 else 0
 
                    # --- แถบสรุปตัวเลข (HTML component) ---
                    render_stats_strip(hdp_acc, ou_acc, total_train)
 
                    # --- วงแหวนความแม่นยำ + ปัจจัยสำคัญ (HTML component) ---
                    col_hdp, col_ou = st.columns(2, gap="medium")
                    with col_hdp:
                        render_model_panel("แฮนดิแคป", "ความแม่นยำจากผลการเรียนรู้ประวัติศาสตร์ทั้งหมด", EMERALD,
                                           int(hdp_correct), total_train, top_features(model_hdp, X_train.columns))
                    with col_ou:
                        render_model_panel("สูง-ต่ำ", "ความแม่นยำจากผลการเรียนรู้ประวัติศาสตร์ทั้งหมด", TEAL,
                                           int(ou_correct), total_train, top_features(model_ou, X_train.columns))
 
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    panel_header("แมตช์ที่รอทำนาย (AI Predictions)", "ระบบวิเคราะห์ความน่าจะเป็นและช่องกรอกผลสกอร์อัปเดต Google Sheets")
                    
                    if len(df_predict) > 0:
                        X_pred = df_predict[features].fillna(train_medians)
                        
                        def get_win_prob(model, X):
                            probs = model.predict_proba(X)
                            if hasattr(model, 'classes_') and 1 in model.classes_: return probs[:, list(model.classes_).index(1)]
                            elif probs.shape[1] > 1: return probs[:, 1]
                            return np.zeros(len(X))
                        
                        df_predict['AI_โอกาสเหย้าชนะHDP(%)'] = (get_win_prob(model_hdp, X_pred) * 100).round(1)
                        df_predict['AI_โอกาสออกสูง(%)'] = (get_win_prob(model_ou, X_pred) * 100).round(1)
                        
                        def hdp_decision(prob):
                            if prob >= 60: return "⭐⭐⭐ ไม้หนัก VIP (แทงเหย้า)"
                            elif prob >= 55: return "⭐⭐ ลงทุนเหย้า"
                            elif prob <= 40: return "❌ รองทีมเยือน VIP"
                            elif prob <= 45: return "🔻 รองทีมเยือน"
                            else: return "⚠️ ข้าม (โอกาสไม่ชัดเจน)"
                            
                        def ou_decision(prob):
                            if prob >= 60: return "🔥 สูง (Over) มั่นใจมาก"
                            elif prob >= 55: return "📈 น่าลงทุนสูง"
                            elif prob <= 40: return "🛡️ ต่ำ (Under) มั่นใจมาก"
                            elif prob <= 45: return "📉 น่าลงทุนต่ำ"
                            else: return "⚠️ ข้าม (O/U)"
 
                        df_predict['AI_ฟันธง_HDP'] = df_predict['AI_โอกาสเหย้าชนะHDP(%)'].apply(hdp_decision)
                        df_predict['AI_ฟันธง_สูงต่ำ'] = df_predict['AI_โอกาสออกสูง(%)'].apply(ou_decision)
                        
                        n_hdp = sum(1 for v in df_predict['AI_ฟันธง_HDP'] if _pred_tone(v, "hdp") != "skip")
                        n_ou = sum(1 for v in df_predict['AI_ฟันธง_สูงต่ำ'] if _pred_tone(v, "ou") != "skip")
                        st.markdown(f"<div class='chips'><span class='chip'>แมตช์ที่รอทำนาย<b>{len(df_predict)}</b></span>"
                                    f"<span class='chip'>สัญญาณแฮนดิแคป<b>{n_hdp}</b></span>"
                                    f"<span class='chip'>สัญญาณสูง-ต่ำ<b>{n_ou}</b></span></div>", unsafe_allow_html=True)
                        render_prediction_cards(df_predict, workbook)
                    else:
                        st.success("🎉 ไม่มีคู่ค้างที่ยังไม่ลงผลสกอร์ ทุกคู่ใน Google Sheets บันทึกผลครบถ้วนแล้ว!")
                else:
                    st.info("ℹ️ ต้องการข้อมูลที่มีผลการแข่งขันในอดีตอย่างน้อย 5 แถวขึ้นไป เพื่อให้ AI เริ่มประมวลผลสถิติ")
        else:
            st.warning("⚠️ ไม่พบคอลัมน์ 'ผล HDP' หรือ 'ผลสูง/ต่ำ' ใน Google Sheets สำหรับเทรน AI")
