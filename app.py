import streamlit as st
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import json
import pandas as pd
import numpy as np
import gc
import time
import threading
import uuid
import traceback
from math import isfinite
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as esc
 
# 🚀 นำเข้า Machine Learning Ensemble & Models (V5.0 Ultimate Hybrid)
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
 
# ----------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP & AI", page_icon="⚽", layout="wide")
 
# ----------------------------------------
# THEME : Black & Emerald (Cyberpunk Monitor)
# ----------------------------------------
BG = "#050907"
SURFACE = "#0c130f"
LINE = "#1b2a22"
TEXT = "#e6f2ea"
MUTED = "#7f9a8b"
GREEN = "#22c55e"
GREEN_SOFT = "#86efac"
AMBER = "#d9a441"
RED = "#e5484d"
CYAN = "#0ea5e9"
CYAN_SOFT = "#7dd3fc"
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea { font-family: 'Prompt', 'Segoe UI', sans-serif !important; }
.stApp { background: #050907; color: #e6f2ea; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3, h4 { color: #e6f2ea !important; letter-spacing: -0.01em; }
p, label, span { color: inherit; }
 
/* 🌟 แก้ปัญหาข้อความเททับกันใน Streamlit Container */
.element-container { overflow: visible !important; }
.stMarkdown div { line-height: 1.5; }
 
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1b2a22; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7f9a8b; padding: 10px 18px; border-radius: 10px 10px 0 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #22c55e !important; background: #0c130f; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22c55e; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background: #22c55e; color: #03140a; border: none; font-weight: 600; border-radius: 10px; padding: 0.55rem 1.4rem; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover { background: #4ade80; color: #03140a; }
.stButton > button:not([kind="primary"]), [data-testid="stFormSubmitButton"] > button:not([kind="primary"]) { background: #0c130f; color: #cfe3d6; border: 1px solid #1b2a22; border-radius: 10px; transition: border-color .2s, box-shadow .2s; }
.stButton > button:not([kind="primary"]):hover, [data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover { border-color: #22c55e; color: #4ade80; box-shadow: 0 0 20px -8px rgba(34,197,94,0.7); }
[data-testid="stFileUploader"] section { background: #0c130f; border: 1px dashed #2a4136; border-radius: 14px; transition: border-color .2s, box-shadow .2s; }
[data-testid="stFileUploader"] section:hover { border-color: #22c55e; box-shadow: 0 0 28px -12px rgba(34,197,94,0.6); }
[data-testid="stAlert"] { background: #0c130f; border: 1px solid #1b2a22; border-radius: 12px; }
.page-title { font-size: 1.9rem; font-weight: 600; margin: 0; color: #e6f2ea; }
.page-sub { color: #7f9a8b; margin: 4px 0 22px 0; font-size: 0.95rem; }
 
/* ================= INPUTS (ช่องกรอกข้อมูล) กลับมาแล้ว ================= */
label[data-testid="stWidgetLabel"] p { color: #a7c4b5; font-size: 0.85rem; font-weight: 400; }
[data-baseweb="input"], [data-baseweb="base-input"] { background-color: #080e0b !important; border-radius: 10px !important; }
[data-baseweb="input"] { border: 1px solid #1b2a22 !important; transition: border-color .2s, box-shadow .2s; }
[data-baseweb="input"]:focus-within { border-color: #22c55e !important; box-shadow: 0 0 0 1px rgba(34,197,94,0.55), 0 0 22px -6px rgba(34,197,94,0.65); }
[data-testid="stNumberInput"] input { font-family: 'JetBrains Mono', monospace !important; font-weight: 500; color: #e6f2ea; }
[data-testid="stNumberInput"] button { background: #0c130f !important; color: #7f9a8b !important; border: none !important; transition: background .2s, color .2s; }
[data-testid="stNumberInput"] button:hover { background: rgba(34,197,94,0.14) !important; color: #4ade80 !important; }
[data-baseweb="select"] > div { background-color: #080e0b !important; border: 1px solid #1b2a22 !important; border-radius: 10px !important; }
[data-baseweb="select"] > div:focus-within { border-color: #22c55e !important; box-shadow: 0 0 22px -8px rgba(34,197,94,0.6); }
 
/* ฟอร์ม = แผงควบคุม */
[data-testid="stForm"] { position: relative; background: linear-gradient(180deg, #0b120e, #080e0b); border: 1px solid #1b2a22; border-radius: 16px; padding: 18px 20px 10px; }
[data-testid="stForm"]::before { content: ""; position: absolute; left: 20px; right: 20px; top: -1px; height: 2px; background: linear-gradient(90deg, transparent, #22c55e, transparent); opacity: .7; }
 
/* ช่องกรอกผลสกอร์ = สกอร์บอร์ด */
[class*="st-key-score_input_"] input { font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 700 !important; text-align: center; letter-spacing: .32em; color: #4ade80 !important; text-shadow: 0 0 14px rgba(74,222,128,0.5); }
[class*="st-key-score_input_"] input::placeholder { color: #2a4136 !important; text-shadow: none; }
[class*="st-key-score_form_"] [data-testid="stForm"], [data-testid="stForm"][class*="st-key-score_form_"] { border-radius: 0 0 18px 18px; border-top: 1px dashed #1b2a22; background: #080e0b; padding: 16px 20px 8px; }
[class*="st-key-score_form_"] [data-testid="stForm"]::before, [data-testid="stForm"][class*="st-key-score_form_"]::before { display: none; }
[class*="st-key-score_form_"] [data-testid="stFormSubmitButton"] { margin-top: 1.75rem; }
 
/* หัวฟอร์ม 5 นัด */
.fx-side { --c: #22c55e; display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 12px; margin-bottom: 10px; border: 1px solid color-mix(in srgb, var(--c) 45%, #1b2a22); background: color-mix(in srgb, var(--c) 9%, transparent); font-weight: 500; color: #e6f2ea; }
.fx-side.away { --c: #d9a441; }
.fx-side small { font-family: 'JetBrains Mono', monospace; color: var(--c); letter-spacing: .1em; font-size: .72rem; }
 
/* ================= JOB CARDS (การ์ด Tab 1 กลับมาแล้ว) ================= */
.job-card { position: relative; overflow: hidden; background: #0c130f; border: 1px solid #1b2a22; border-left: 4px solid #1b2a22; border-radius: 14px; padding: 14px 20px 16px; margin-bottom: 10px; }
.job-card.ok { border-left-color: #22c55e; }
.job-card.err { border-left-color: #e5484d; }
.job-card.run { border-left-color: #d9a441; }
.job-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.job-name { color: #7f9a8b; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job-pill { font-size: 0.76rem; padding: 3px 11px; border-radius: 999px; white-space: nowrap; font-weight: 500; }
.job-pill.run { color: #e8bb62; background: rgba(217,164,65,0.14); }
.job-pill.ok { color: #4ade80; background: rgba(34,197,94,0.15); }
.job-pill.err { color: #f06b70; background: rgba(229,72,77,0.14); }
.job-title { color: #e6f2ea; font-size: 1.05rem; font-weight: 600; margin: 8px 0 2px; }
.job-title .vs { color: #7f9a8b; font-weight: 400; padding: 0 6px; }
.job-msg { color: #a7c4b5; font-size: 0.9rem; }
.job-note { color: #7f9a8b; font-size: 0.8rem; margin-top: 6px; }
.job-bar { height: 3px; border-radius: 3px; margin-top: 12px; background: #17241d; overflow: hidden; }
.job-bar .j-bar { display: block; height: 100%; width: 40%; border-radius: 3px; background: linear-gradient(90deg, transparent, #22c55e, #86efac, transparent); animation: sweep 1.4s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(260%); } }
 
/* ================= KPI (monitor tiles) ================= */
.kpi-wrap { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 22px; width: 100%; }
.kpi { flex: 1 1 calc(33.333% - 16px); min-width: 250px; --k: #7f9a8b; position: relative; overflow: hidden; border: 1px solid #1b2a22; border-radius: 16px; padding: 20px 24px 22px; background-color: #0c130f; background-image: linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px); background-size: 100% 24px, 24px 100%; box-sizing: border-box; }
.kpi::before, .kpi::after { content: ""; position: absolute; width: 11px; height: 11px; border: 2px solid var(--k); opacity: .7; }
.kpi::before { top: 8px; left: 8px; border-right: none; border-bottom: none; }
.kpi::after { bottom: 8px; right: 8px; border-left: none; border-top: none; }
.kpi.hero { --k: #22c55e; background-color: #0d2417; border-color: #1f6b3d; box-shadow: 0 0 40px -12px rgba(34,197,94,0.35); }
.kpi.hero-cyan { --k: #0ea5e9; background-color: #08283d; border-color: #0284c7; box-shadow: 0 0 40px -12px rgba(14,165,233,0.35); }
.kpi-label { display: flex; align-items: center; gap: 9px; color: #7f9a8b; font-size: 0.9rem; margin-bottom: 6px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 700; line-height: 1.15; color: #e6f2ea; }
.kpi.hero .kpi-value { font-size: 3.2rem; color: #4ade80; text-shadow: 0 0 22px rgba(74,222,128,0.45); }
.kpi.hero-cyan .kpi-value { font-size: 3.2rem; color: #38bdf8; text-shadow: 0 0 22px rgba(56,189,248,0.45); }
.kpi-unit { font-family: 'Prompt', sans-serif; font-size: 1rem; color: #7f9a8b; font-weight: 400; margin-left: 4px; text-shadow: none; }
.kpi-bar { height: 6px; background: #17241d; border-radius: 6px; margin-top: 14px; overflow: hidden; }
.kpi-bar > div { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; }
.kpi.hero-cyan .kpi-bar > div { background: linear-gradient(90deg, #0284c7, #38bdf8); }
 
/* ================= MONITOR & TICKER ================= */
.led { --lc: var(--k, #22c55e); display: inline-block; flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--lc); box-shadow: 0 0 0 0 color-mix(in srgb, var(--lc) 70%, transparent); animation: ping 1.8s infinite; }
@keyframes ping { 70% { box-shadow: 0 0 0 9px color-mix(in srgb, var(--lc) 0%, transparent); } 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--lc) 0%, transparent); } }
.mon-status { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 11px 18px; margin: 6px 0 10px; background: #0a110d; border: 1px solid #1b2a22; border-radius: 14px; font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: #7f9a8b; }
.mon-status b { color: #4ade80; letter-spacing: .1em; font-weight: 700; }
.mon-status em { font-style: normal; color: #e6f2ea; margin-left: 6px; }
 
.ticker-container { width: 100%; overflow: hidden; border-top: 1px solid #14211a; border-bottom: 1px solid #14211a; background: #080e0b; margin-bottom: 16px; position: relative; }
.ticker-track { display: flex; white-space: nowrap; width: max-content; padding: 9px 0; animation: marquee var(--dur, 40s) linear infinite; }
@keyframes marquee { to { transform: translateX(-50%); } }
.tk { display: inline-flex; align-items: center; font-size: .84rem; color: #a7c4b5; margin-right: 38px; }
.tk b { color: #e6f2ea; font-weight: 500; margin-right: 6px; }
.tk .d { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin: 0 6px 0 12px; background: var(--c); box-shadow: 0 0 8px var(--c); }
 
/* ================= MATCH CARDS ================= */
.mon { position: relative; overflow: hidden; border-radius: 18px; padding: 14px 22px 22px; margin-bottom: 16px; background: linear-gradient(rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 100% 26px, linear-gradient(90deg, rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 26px 100%, linear-gradient(180deg, #0e1712, #0b110e); border: 1px solid #1b2a22; }
.mon::after { content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%; pointer-events: none; background: linear-gradient(180deg, transparent, rgba(74,222,128,0.07), transparent); animation: scan 5.5s linear infinite; }
@keyframes scan { to { transform: translateY(350%); } }
.cn { position: absolute; width: 12px; height: 12px; border: 2px solid #22c55e; opacity: .65; pointer-events: none; }
.cn.tl { top: 8px; left: 8px; border-right: none; border-bottom: none; }
.cn.tr { top: 8px; right: 8px; border-left: none; border-bottom: none; }
.cn.bl { bottom: 8px; left: 8px; border-right: none; border-top: none; }
.cn.br { bottom: 8px; right: 8px; border-left: none; border-top: none; }
.mon-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #7f9a8b; padding: 0 6px; }
.mon-id { color: #cfe3d6; letter-spacing: .1em; }
.mon-st { padding: 2px 10px; border-radius: 999px; border: 1px solid #1b2a22; font-family: 'Prompt', sans-serif; }
.mon-st.on { color: #4ade80; border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.1); }
 
.sig { margin-left: auto; display: flex; align-items: flex-end; gap: 3px; height: 14px; }
.sig .s-bar { width: 4px; border-radius: 1px; background: #1f2e26; }
.sig .s-bar:nth-child(1) { height: 5px; } .sig .s-bar:nth-child(2) { height: 8px; } .sig .s-bar:nth-child(3) { height: 11px; } .sig .s-bar:nth-child(4) { height: 14px; }
.sig .s-bar.on { background: #4ade80; box-shadow: 0 0 6px rgba(74,222,128,0.7); }
 
.mon-teams { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 16px 0 14px; padding: 0 6px; width: 100%; }
.mon-t { flex: 1; font-weight: 600; font-size: 1.1rem; color: #e6f2ea; line-height: 1.3; }
.mon-t.r { text-align: right; }
.ecg { flex: 0 0 60px; height: 30px; }
.ecg .pulse { stroke-dasharray: 22 78; animation: ecg 2.6s linear infinite; filter: drop-shadow(0 0 4px #4ade80); }
@keyframes ecg { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }
 
.mon-ch { display: flex; flex-direction: row; gap: 14px; width: 100%; box-sizing: border-box; }
.ch { flex: 1; min-width: 0; --c: #22c55e; padding: 14px 16px; border-radius: 14px; background: rgba(5,9,7,0.72); border: 1px solid #14211a; box-sizing: border-box; }
.ch-h { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #7f9a8b; letter-spacing: .04em; }
.ch-k { color: var(--c); font-weight: 700; flex: none; }
.ch-l { color: #cfe3d6; font-size: .82rem; text-align: right; }
.ch-body { display: flex; align-items: center; gap: 14px; margin-top: 6px; }
.dial { flex: 0 0 80px; width: 80px; height: 48px; overflow: visible; }
.ch-info { flex: 1; min-width: 0; }
.ch-val { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; line-height: 1; color: var(--c); text-shadow: 0 0 18px color-mix(in srgb, var(--c) 55%, transparent); }
.ch-val small { font-size: .9rem; color: #7f9a8b; margin-left: 2px; }
 
.ch-d { font-size: .78rem; color: #7f9a8b; margin: 6px 0 9px; line-height: 1.3; }
.ch-v { display: block; padding: 6px 12px; border-radius: 8px; font-size: .82rem; font-weight: 500; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); line-height: 1.4; word-wrap: break-word; margin-top: 4px; }
 
.vu { display: grid; grid-template-columns: repeat(var(--n, 20), 1fr); gap: 3px; margin-top: 12px; }
.vu .v-bar { height: 9px; border-radius: 2px; background: #17241d; }
.vu .v-bar.on { background: var(--c); box-shadow: 0 0 8px color-mix(in srgb, var(--c) 60%, transparent); animation: vuon .35s calc(var(--i) * 40ms) both; }
@keyframes vuon { from { background: #17241d; box-shadow: none; } }
.vu-l { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: #7f9a8b; margin-top: 5px; padding: 0 2px; }
 
.mon.solo { border-radius: 18px; margin-bottom: 10px; }
 
/* 📱 Responsive Mobile Layout Fix */
@media (max-width: 768px) {
    .kpi { flex: 1 1 100%; min-width: 100%; }
    .mon-ch { flex-direction: column; }
    .mon-t { font-size: 1rem; }
    .ecg { display: none; }
    .mon-teams { justify-content: flex-start; }
    .mon-t.r { text-align: left; }
    .mon-t.r::before { content: "vs "; color: #5d7a69; font-weight: 400; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)
 
# =============================================================================
# CYBER MONITOR CSS (override layer v2)
# วางต่อท้าย st.markdown("""<style> ... </style>""") ก้อนเดิมทันที
# (ต้องอยู่ "หลัง" CSS เดิม เพื่อให้ทับค่าเดิมได้)  ไม่ต้องแก้ CSS เดิมเลย
# =============================================================================
CYBER_CSS = """
<style>
/* ================= CYBER MONITOR V6 : override layer ================= */
:root { --cy:#00e5ff; --mg:#ff2bd6; --gn:#39ff88; --yl:#fcee0a; --rd:#ff3b5c; }
 
/* พื้นหลัง: เรืองแสงมุมจอ + สแกนไลน์บางๆ */
.stApp {
  background:
    radial-gradient(900px 520px at 100% 0%, rgba(255,43,214,.10), transparent 60%),
    radial-gradient(800px 520px at 0% 100%, rgba(0,229,255,.09), transparent 60%),
    #04070a;
}
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: repeating-linear-gradient(0deg, rgba(0,229,255,.028) 0 1px, transparent 1px 3px);
}
 
/* หัวเรื่อง */
.page-title {
  background: linear-gradient(90deg, #00e5ff, #39ff88 55%, #ff2bd6);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 12px rgba(0,229,255,.35));
  font-weight: 700; line-height: 1.25;
}
.page-sub { border-left: 2px solid #ff2bd6; padding-left: 10px; }
 
/* แท็บ: ไฮไลต์ไซแอน */
.stTabs [aria-selected="true"] { color:#00e5ff !important; text-shadow:0 0 12px rgba(0,229,255,.6); }
.stTabs [data-baseweb="tab-highlight"] { background-color:#00e5ff !important; box-shadow:0 0 12px #00e5ff; }
 
/* การ์ดเดิม: มุมตัดเฉียงแบบ HUD */
.kpi, .mon, .job-card {
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
  border-radius: 0 !important;
}
.kpi.hero { border-color:#39ff88; box-shadow:none; }
.kpi.hero-cyan { border-color:#00e5ff; box-shadow:none; }
.kpi.hero .kpi-value { color:#39ff88; text-shadow:0 0 22px rgba(57,255,136,.55); }
.kpi.hero-cyan .kpi-value { color:#00e5ff; text-shadow:0 0 22px rgba(0,229,255,.55); }
.kpi-bar, .kpi-bar > div { border-radius: 0; }
 
/* ================= HUD PANEL (st.container(key="hud_...")) ================= */
[class*="st-key-hud_"] {
  position: relative; margin: 14px 0; padding: 16px 18px 10px;
  border: 1px solid rgba(0,229,255,.25);
  background: linear-gradient(180deg, rgba(8,20,26,.92), rgba(5,9,12,.94));
  clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
}
[class*="st-key-hud_"]::before {
  content:""; position:absolute; left:0; right:16px; top:0; height:2px;
  background: linear-gradient(90deg, #00e5ff, rgba(255,43,214,.6) 55%, transparent);
}
.hud-title { display:flex; align-items:center; gap:10px; margin:0 0 2px;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; letter-spacing:.06em; color:#00e5ff; }
.hud-title .led { --lc:#00e5ff; }
.hud-title .sub { margin-left:auto; text-align:right; color:#7f9a8b; letter-spacing:0;
  font-family:'Prompt',sans-serif; font-size:.74rem; }
.hud-note { color:#7f9a8b; font-size:.76rem; margin:2px 0 8px; line-height:1.5; }
.hud-note b { color:#a7c4b5; font-weight:500; }
 
/* ชิปตัวเลข */
.chip-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0 6px; }
.chip { --c:#00e5ff; padding:9px 12px; border-left:3px solid var(--c);
  border-top:1px solid color-mix(in srgb, var(--c) 30%, #12202a);
  border-right:1px solid color-mix(in srgb, var(--c) 30%, #12202a);
  border-bottom:1px solid color-mix(in srgb, var(--c) 30%, #12202a);
  background: color-mix(in srgb, var(--c) 7%, #060c10); min-width:0; }
.chip-k { font-size:.7rem; color:#7f9a8b; line-height:1.3; }
.chip-v { font-family:'JetBrains Mono',monospace; font-size:1.25rem; font-weight:700; color:var(--c);
  text-shadow:0 0 14px color-mix(in srgb, var(--c) 55%, transparent); }
 
/* แถบผลย้อนหลัง (streak strip) */
.fs-wrap { margin:6px 0 14px; }
.fs-h { display:flex; justify-content:space-between; gap:8px; margin-bottom:6px;
  font-family:'JetBrains Mono',monospace; font-size:.74rem; color:#7f9a8b; }
.fs-h b { color:#e6f2ea; font-weight:500; }
.fs-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(14px, 1fr)); gap:4px; }
.fs-cell { height:20px; background:var(--c); opacity:.92;
  box-shadow:0 0 8px color-mix(in srgb, var(--c) 55%, transparent); }
.fs-cell.last { outline:1px solid #e6f2ea; outline-offset:1px; }
.fs-legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:.72rem; color:#7f9a8b; }
.fs-legend span::before { content:""; display:inline-block; width:9px; height:9px; margin-right:6px;
  background:var(--c); box-shadow:0 0 6px var(--c); }
 
/* สถานะโมเดล */
.mdl-wrap { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin:8px 0; }
.mdl { --c:#39ff88; padding:14px 16px; border-left:3px solid var(--c);
  border-top:1px solid color-mix(in srgb, var(--c) 35%, #12202a);
  border-right:1px solid color-mix(in srgb, var(--c) 35%, #12202a);
  border-bottom:1px solid color-mix(in srgb, var(--c) 35%, #12202a);
  background: color-mix(in srgb, var(--c) 6%, #060c10); min-width:0; }
.mdl-h { display:flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
  font-size:.78rem; letter-spacing:.06em; color:var(--c); }
.mdl-h .led { --lc:var(--c); }
.mdl-s { margin-left:auto; padding:1px 10px; border:1px solid var(--c); font-size:.72rem; }
.mdl-r { display:flex; justify-content:space-between; gap:8px; margin-top:7px; font-size:.8rem; color:#a7c4b5; }
.mdl-r b { font-family:'JetBrains Mono',monospace; color:#e6f2ea; font-weight:500; }
.mdl-n { margin-top:9px; font-size:.76rem; color:#7f9a8b; line-height:1.5; word-break:break-word; }
 
/* Plotly: ให้กลืนกับธีม และไม่ล้นจอ */
[data-testid="stPlotlyChart"] { width:100%; overflow:hidden; }
.js-plotly-plot .plotly .modebar { display:none !important; }
 
/* ================= PANEL HEADERS (เดิมไม่มี CSS) ================= */
.panel-title { margin: 6px 0 0; font-size: 1.15rem; font-weight: 600; color: #e6f2ea; letter-spacing: .01em; text-shadow: 0 0 18px rgba(0,229,255,.25); }
.panel-sub { margin: 2px 0 14px; padding-left: 10px; border-left: 2px solid #ff2bd6; color: #7f9a8b; font-size: .85rem; line-height: 1.5; }
.empty-state { margin: 8px 0 12px; padding: 18px 16px; text-align: center; color: #7f9a8b; font-size: .85rem; line-height: 1.6;
  border: 1px dashed rgba(0,229,255,.28); background: rgba(0,229,255,.03); }
 
/* ================= UNIFIED MATCH CARD : สัญญาณ + กรอกสกอร์ ในการ์ดเดียว ================= */
[class*="st-key-matchcard_"] {
  position: relative; gap: 0 !important; margin: 0 0 20px; padding: 0;
  border: 1px solid rgba(0,229,255,.30); background: #070d10;
  clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px));
}
[class*="st-key-matchcard_"]::before {
  content: ""; position: absolute; left: 0; right: 18px; top: 0; height: 2px; z-index: 2;
  background: linear-gradient(90deg, #00e5ff, rgba(255,43,214,.65) 55%, transparent);
}
[class*="st-key-matchcard_"] .element-container { margin: 0; }
[class*="st-key-matchcard_"] .mon { border: 0 !important; margin: 0 !important; clip-path: none !important; }
[class*="st-key-matchcard_"] .mc-sec {
  display: flex; align-items: center; gap: 10px; padding: 12px 22px 2px;
  border-top: 1px dashed rgba(0,229,255,.30); background: rgba(4,8,11,.80);
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .06em; color: #00e5ff; }
[class*="st-key-matchcard_"] .mc-sec .led { --lc: #00e5ff; }
[class*="st-key-matchcard_"] .mc-sec .sub { margin-left: auto; text-align: right; letter-spacing: 0; color: #7f9a8b; font-family: 'Prompt', sans-serif; font-size: .74rem; }
[class*="st-key-matchcard_"] [data-testid="stForm"] {
  border: 0 !important; border-radius: 0 !important; background: rgba(4,8,11,.80) !important; padding: 8px 22px 10px !important; }
[class*="st-key-matchcard_"] [data-testid="stForm"]::before { display: none; }
[class*="st-key-matchcard_"] [data-testid="stFormSubmitButton"] { margin-top: 1.75rem; }
[class*="st-key-matchcard_"] [data-testid="stAlert"] { margin: 0 22px 12px; }
 
/* ช่องกรอกสกอร์ = สกอร์บอร์ด (แก้ชื่อคลาสให้ตรงกับ key จริง t2_score_input_) */
[class*="st-key-t2_score_input_"] input {
  font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 700 !important;
  text-align: center; letter-spacing: .32em; color: #39ff88 !important; text-shadow: 0 0 14px rgba(57,255,136,.5); }
[class*="st-key-t2_score_input_"] input::placeholder { color: #2a4136 !important; text-shadow: none; }
 
/* ================= KPI sub / TABLE / MISC ================= */
.kpi-sub { margin-top: 8px; font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #7f9a8b; }
.chip-row { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.cy-scroll { overflow-x: auto; margin: 6px 0 10px; }
.cy-table { width: 100%; min-width: 440px; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: .78rem; }
.cy-table th { padding: 7px 10px; text-align: right; white-space: nowrap; color: #7f9a8b; font-weight: 500; border-bottom: 1px solid rgba(0,229,255,.28); }
.cy-table td { padding: 7px 10px; text-align: right; white-space: nowrap; color: #e6f2ea; border-bottom: 1px solid #12202a; }
.cy-table th:first-child, .cy-table td:first-child { text-align: left; }
.cy-table td:nth-child(2) { text-align: left; }
.cy-table tr.sel td { color: #39ff88; background: rgba(57,255,136,.06); }
.mdl-wrap.three { grid-template-columns: repeat(2, 1fr); }
 
/* ================= MOBILE ================= */
@media (max-width: 768px) {
  .block-container { padding: 1.2rem .8rem 3rem; }
  .page-title { font-size: 1.35rem; }
  .page-sub { font-size: .8rem; margin-bottom: 14px; }
  .stTabs [data-baseweb="tab-list"] { overflow-x:auto; flex-wrap:nowrap; scrollbar-width:none; }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display:none; }
  .stTabs [data-baseweb="tab"] { white-space:nowrap; padding:8px 12px; font-size:.85rem; }
 
  .kpi { padding:16px 18px 18px; }
  .kpi-value { font-size:1.9rem; }
  .kpi.hero .kpi-value, .kpi.hero-cyan .kpi-value { font-size:2.5rem; }
  .mon-status { font-size:.72rem; padding:9px 12px; gap:6px 14px; }
 
  [class*="st-key-hud_"] { padding:14px 12px 6px; margin:12px 0; }
  .hud-title { flex-wrap:wrap; font-size:.74rem; }
  .hud-title .sub { margin-left:0; text-align:left; flex-basis:100%; }
  .chip-row { grid-template-columns:repeat(2,1fr); gap:8px; }
  .chip-v { font-size:1.1rem; }
  .mdl-wrap { grid-template-columns:1fr; }
  .fs-grid { grid-template-columns:repeat(auto-fill, minmax(12px, 1fr)); gap:3px; }
  .fs-cell { height:18px; }
  .ch-val { font-size:1.55rem; }
  .vu-l { font-size:.64rem; }
  [class*="st-key-matchcard_"] { margin-bottom: 16px; }
  [class*="st-key-matchcard_"] .mon { padding: 12px 14px 16px; }
  [class*="st-key-matchcard_"] .mc-sec { padding: 10px 14px 0; flex-wrap: wrap; }
  [class*="st-key-matchcard_"] .mc-sec .sub { margin-left: 0; text-align: left; flex-basis: 100%; }
  [class*="st-key-matchcard_"] [data-testid="stForm"] { padding: 6px 14px 8px !important; }
  [class*="st-key-matchcard_"] [data-testid="stFormSubmitButton"] { margin-top: 0; }
  [class*="st-key-matchcard_"] [data-testid="stAlert"] { margin: 0 14px 10px; }
  [class*="st-key-t2_score_input_"] input { font-size: 1.35rem !important; letter-spacing: .24em; }
  .cy-table { font-size: .72rem; }
}
@media (max-width: 380px) {
  .chip-row { grid-template-columns:1fr; }
}
@media (prefers-reduced-motion: reduce) { .stApp::before { display:none; } }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)
 
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
 
# ----------------------------------------
# การจัดการ Data Extraction & Google Sheets
# ----------------------------------------
model = genai.GenerativeModel('gemini-3.7-flash')
 
# ประกาศตัวแปร FIELD_NAMES และ CRITICAL_IDX เพื่อป้องกันแอปพังตอนอ่านค่า
WATER_IDX = [6, 8, 10, 11]
CRITICAL_IDX = [5, 6, 7, 8, 9, 10, 11]
FIELD_NAMES = ["ทีมเหย้า", "ทีมเยือน", "1X2เหย้า", "1X2เสมอ", "1X2เยือน", "HDPเหย้า", "น้ำHDPเหย้า", "HDPเยือน", "น้ำHDPเยือน", "โกลสูงต่ำ", "น้ำสูง", "น้ำต่ำ"]
 
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
 
def clean_raw_data(raw_text):
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text).strip()
    
    match_array = re.search(r'\[(.*)\]', text, re.DOTALL)
    match_obj = re.search(r'\{.*\}', text, re.DOTALL)
    
    if match_array: json_str = "[" + match_array.group(1).strip() + "]"
    elif match_obj: json_str = match_obj.group(0).strip()
    else: json_str = text
        
    json_str = re.sub(r',\s*\]$', ']', json_str)
    json_str = re.sub(r',\s*\}$', '}', json_str)
    
    try: data = json.loads(json_str)
    except Exception: raise ValueError(f"รูปแบบข้อมูลผิดพลาด -> สิ่งที่ AI ตอบมาคือ: {text[:80]}...")
        
    if isinstance(data, dict): data = list(data.values())
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list): data = data[0]
    if not isinstance(data, list): raise ValueError(f"ข้อมูลไม่ใช่ Array -> สิ่งที่ AI ตอบมาคือ: {str(data)[:80]}...")
    if len(data) < 12: raise ValueError(f"ข้อมูลมาไม่ครบ (ได้ {len(data)} ช่อง จาก 12) -> ข้อมูลที่ได้: {str(data)[:80]}...")
        
    home = str(data[0]).strip()
    away = str(data[1]).strip()
 
    numbers, bad = [], []
    for i in range(2, 12):
        val_str = str(data[i]).replace('−', '-').replace(',', '').strip()
        try:
            v = float(val_str)
            if not isfinite(v): raise ValueError
            numbers.append(v)
        except (TypeError, ValueError):
            numbers.append(None)      # ไม่ใช่ 0.0 อีกต่อไป
            bad.append(i)
 
    critical_bad = [FIELD_NAMES[i] for i in bad if i in CRITICAL_IDX]
    if critical_bad:   # raise -> extract_row จะ retry ให้อัตโนมัติ และถ้ายังพลาดงานจะขึ้น error
        raise ValueError(f"อ่านค่าไม่ได้: {', '.join(critical_bad)} -> AI ตอบ: {str(data)[:80]}")
    warn = ("ช่องที่อ่านไม่ได้ (เว้นว่างในชีต): " + ", ".join(FIELD_NAMES[i] for i in bad)) if bad else None
 
    ai_fmt = str(data[12]).strip().upper() if len(data) > 12 else "UNK"
    if ai_fmt not in ["MY", "HK", "ID", "EU"]: ai_fmt = "UNK"
    return [home, away] + numbers, ai_fmt, warn
 
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
    with JOBS.sheet_lock:
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
        if value is None or (isinstance(value, float) and not isfinite(value)): return {}   # เซลล์ว่าง
        if isinstance(value, (int, float)) and not isinstance(value, bool): return {"userEnteredValue": {"numberValue": value}}
        return {"userEnteredValue": {"stringValue": str(value)}}
 
    def update(start_row, start_col, cells, fields):
        return {"updateCells": {"start": {"sheetId": ws.id, "rowIndex": start_row, "columnIndex": start_col}, "rows": [{"values": cells}], "fields": fields}}
 
    requests.append(update(0, 23, [typed(h) for h in FORM5_HEADERS], "userEnteredValue"))
    requests.append(update(target_row - 1, 0, [typed(v) for v in row[:12]], "userEnteredValue"))
    
    form_cells = [typed(v) for v in normalized]
    for i, cell in enumerate(form_cells):
        cell["userEnteredFormat"] = {"numberFormat": {"type": "PERCENT" if i in (0, 3) else "NUMBER", "pattern": "0.0%" if i in (0, 3) else "0"}}
    requests.append(update(target_row - 1, 23, form_cells, "userEnteredValue,userEnteredFormat.numberFormat"))
    ws.spreadsheet.batch_update({"requests": requests})
    return target_row, overwritten
 
def update_match_score_in_sheet(workbook, home_team, away_team, score_text, *, sheet_row=None, expected_snapshot=None):
    """Tab 2 only: validate row identity; write one RAW text cell; verify by reading.
 
    Compatible with the former four-argument call. With no row reference, refuse
    ambiguous pairs rather than taking their first matching occurrence. Locking
    only protects threads using the same JOBS instance, not other app replicas or
    people sorting/editing the sheet. No automatic retries of dispatched writes.
    """
    score = _t2_parse_score(score_text)
    if score is None:
        raise ScoreConflictError('กรอกสกอร์เป็นข้อความ เหย้า-เยือน เช่น 2-1 หรือ 0-0 เท่านั้น')
    normalized_score = f'{score[0]}-{score[1]}'
    home = _t2_canonical(home_team, name=True)
    away = _t2_canonical(away_team, name=True)
    if not home or not away:
        raise ScoreConflictError('ชื่อทีมไม่ครบ จึงไม่สามารถระบุแถวที่จะบันทึกได้')
    with JOBS.sheet_lock:
        ws = workbook.worksheet('DATA')
        _t2_check_headers(_t2_values(ws, 'A1:M1'))
        if sheet_row is None:
            data = _t2_data_from_grid(_t2_values(ws, 'A1:AC'))
            matches = data.loc[data['ทีมเหย้า'].map(lambda v: _t2_canonical(v, name=True)).eq(home)
                               & data['ทีมเยือน'].map(lambda v: _t2_canonical(v, name=True)).eq(away)]
            if matches.empty:
                return False
            if len(matches) != 1:
                raise ScoreConflictError('พบชื่อคู่ทีมซ้ำหลายแถว กรุณารีเฟรชแล้วเลือกการ์ดที่มีเลขแถวชัดเจน')
            target = int(matches.iloc[0]['_SheetRow'])
            if expected_snapshot is None:
                expected_snapshot = matches.iloc[0]['_Snapshot']
        else:
            if isinstance(sheet_row, (bool, np.bool_)):
                raise ScoreConflictError('เลขแถวไม่ถูกต้อง')
            text_row = str(sheet_row).strip()
            if not re.fullmatch(r'[0-9]+', text_row) or int(text_row) < 2:
                raise ScoreConflictError('เลขแถวไม่ถูกต้อง')
            target = int(text_row)
            if not expected_snapshot:
                raise ScoreConflictError('ไม่มีข้อมูลอ้างอิงของแถว กรุณารีเฟรชก่อนบันทึก')
        current = _t2_read_target(ws, target)
        if not _t2_target_matches(current, home, away, expected_snapshot):
            raise ScoreConflictError('ข้อมูลในแถวนี้เปลี่ยนไปหรือมีการย้ายแถว จึงยังไม่เขียนสกอร์ กรุณารีเฟรชและตรวจคู่แข่งขันอีกครั้ง')
        write_key = _t2_write_key(workbook, ws, target)
        journal = _t2_score_journal()
        previous = journal.get(write_key)
        if previous is not None:
            if _t2_confirm_pending(ws, previous):
                del journal[write_key]
                if _t2_parse_score(previous['score']) == score:
                    return True
                raise ScoreConflictError('ยืนยันว่าคำสั่งก่อนหน้าบันทึกสกอร์อื่นแล้ว จึงไม่เขียนทับ')
            raise ScoreWriteUncertainError('มีคำสั่งลงสกอร์แถวนี้ที่ยังยืนยันไม่ได้ จึงไม่ส่งซ้ำ กดตรวจยืนยันสกอร์ที่บันทึกค้างและตรวจ DATA ก่อน')
        old_score = current[12]
        if not _t2_blank(old_score):
            if _t2_parse_score(old_score) == score:
                return True
            raise ScoreConflictError('แถวนี้มีสกอร์หรือค่าที่ไม่ใช่ช่องว่างอยู่แล้ว จึงไม่เขียนทับ กรุณาตรวจคอลัมน์ M ใน DATA')
        formula_rows = ws.get(range_name=f'M{target}', value_render_option='FORMULA')
        formula_value = formula_rows[0][0] if formula_rows and formula_rows[0] else ''
        if not _t2_blank(formula_value):
            raise ScoreConflictError('ช่องสกอร์มีสูตรหรือมีข้อมูลเปลี่ยนระหว่างตรวจ จึงไม่เขียนทับ กรุณาตรวจคอลัมน์ M')
        entry = {'row': target, 'snapshot': _t2_fingerprint(current), 'score': normalized_score}
        journal[write_key] = entry
        try:
            ws.update(range_name=f'M{target}', values=[[normalized_score]], value_input_option='RAW')
        except Exception as write_error:
            # A timeout does not prove failure. A read may confirm it succeeded.
            try:
                confirmed = _t2_confirm_pending(ws, entry)
            except Exception:
                confirmed = False
            if confirmed:
                journal.pop(write_key, None)
                return True
            response_code = getattr(getattr(write_error, 'response', None), 'status_code', None)
            if response_code in (400, 401, 403, 404, 422, 429) or isinstance(write_error, TypeError):
                # Explicit rejection / local argument mismatch: no accepted write.
                journal.pop(write_key, None)
                raise
            raise ScoreWriteUncertainError('ส่งคำสั่งลงสกอร์แล้ว แต่ยังยืนยันผลไม่ได้ จึงไม่เขียนซ้ำ กรุณาตรวจ DATA และกดตรวจยืนยันสกอร์ที่บันทึกค้าง') from write_error
        try:
            confirmed = _t2_confirm_pending(ws, entry)
        except Exception as read_error:
            raise ScoreWriteUncertainError('บริการตอบรับคำสั่งแล้ว แต่ยังอ่านกลับยืนยันไม่ได้ กรุณาตรวจ DATA ก่อนบันทึกซ้ำ') from read_error
        if not confirmed:
            raise ScoreWriteUncertainError('ข้อมูลที่อ่านกลับไม่ตรงกับแถวและสกอร์ที่ส่ง จึงยังไม่แจ้งว่าสำเร็จ กรุณาตรวจ DATA ก่อนดำเนินการต่อ')
        journal.pop(write_key, None)
        return True
 
# ----------------------------------------
# Job Manager (คิวงานเบื้องหลัง)
# ----------------------------------------
class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.sheet_lock = threading.RLock()
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
 
@st.cache_resource
def get_job_manager(): return JobManager()
 
JOBS = get_job_manager()
if "form5_owner_id" not in st.session_state: st.session_state["form5_owner_id"] = uuid.uuid4().hex
 
def extract_row(img_bytes, user_fmt, model):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    
    prompt = (
        "กรุณาดึงข้อมูลจากตารางราคาบอลในภาพนี้ เลือกมา 1 คู่ที่เห็นชัดที่สุด\n"
        "**สำคัญมาก: ตอบกลับเฉพาะข้อมูลแบบ JSON Array เท่านั้น ห้ามพิมพ์คำอธิบายหรือตัวหนังสืออื่นๆ เด็ดขาด**\n"
        "[\"ทีมเหย้า\", \"ทีมเยือน\", 1X2เหย้า, 1X2เสมอ, 1X2เยือน, HDPเหย้า, น้ำHDPเหย้า, HDPเยือน, น้ำHDPเยือน, โกลสูงต่ำ, น้ำสูง, น้ำต่ำ, \"รูปแบบน้ำเช่น MY\"]\n"
        "ตัวอย่างที่ต้องการ:\n"
        "[\"Arsenal\", \"Chelsea\", 2.10, 3.50, 3.20, -0.5, 0.95, 0.5, -0.85, 2.5, 0.80, -0.90, \"MY\"]"
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
    ]
    
    last_error = ""
    for attempt in range(3):
        try:
            resp = model.generate_content([{"mime_type": "image/jpeg", "data": buf.getvalue()}, prompt], safety_settings=safety_settings)
            if getattr(resp, 'prompt_feedback', None) and resp.prompt_feedback.block_reason:
                raise ValueError("ภาพถูกเซ็นเซอร์โดยระบบ Safety ของ Google")
                
            raw_text = resp.text.strip()
            if not raw_text: raise ValueError("AI ตอบกลับเป็นค่าว่าง (Empty Response)")
                
            row, ai_fmt, parse_warn = clean_raw_data(raw_text)
            row, note, fmt_warn = convert_water_to_malay(row, user_fmt, ai_fmt)
            return row, note, (" | ".join(w for w in (parse_warn, fmt_warn) if w) or None)
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
            
    raise RuntimeError(f"{last_error}")
 
def extract_form_stats(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    
    prompt = (
        "กรุณาดึงข้อมูลสถิติจากภาพ (ฟอร์มการเล่น) โดยตอบกลับเป็น JSON Array ที่มีตัวเลข 6 ตัวแบบนี้เท่านั้น ห้ามพิมพ์คำอธิบายอื่นเด็ดขาด:\n"
        "[WinRateเหย้า, ประตูได้เหย้า, ประตูเสียเหย้า, WinRateเยือน, ประตูได้เยือน, ประตูเสียเยือน]\n"
        "หมายเหตุ: ทีมบนคือเหย้า ทีมล่างคือเยือน WinRate ให้ใส่เฉพาะตัวเลข (เช่น 40% ให้ใส่ 40)"
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
    ]
    
    last_error = ""
    for attempt in range(3):
        try:
            resp = model.generate_content([{"mime_type": "image/jpeg", "data": buf.getvalue()}, prompt], safety_settings=safety_settings)
            raw_text = resp.text.strip()
            
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r'\s*```$', '', raw_text).strip()
            
            match_array = re.search(r'\[(.*)\]', raw_text, re.DOTALL)
            if match_array:
                data = json.loads("[" + match_array.group(1).strip() + "]")
                if len(data) >= 6:
                    return [float(data[0]), int(data[1]), int(data[2]), float(data[3]), int(data[4]), int(data[5])]
            raise ValueError(f"รูปแบบไม่ถูกต้อง: {raw_text}")
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
            
    raise RuntimeError(f"ไม่สามารถอ่านภาพสถิติได้: {last_error}")
 
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
# UI: Upload & Tasks
# ----------------------------------------
STATUS_UI = {"queued": ("รอคิว", "run"), "reading": ("กำลังอ่านภาพ", "run"), "ready": ("อ่านราคาแล้ว", "ok"),
             "saving": ("กำลังบันทึก", "run"), "done": ("บันทึกแล้ว", "ok"), "error": ("ไม่สำเร็จ", "err")}
 
def panel_header(title, sub): st.markdown(f"<p class='panel-title'>{title}</p><p class='panel-sub'>{sub}</p>", unsafe_allow_html=True)
 
def _render_job_signal(res):
    ai = get_ai()
    if ai is None or ai["status"] != "ok": return
    dfp = ai["df_predict"]
    if len(dfp) == 0 or 'AI_ฟันธง_HDP' not in dfp.columns: return
    m = dfp[(dfp['ทีมเหย้า'].astype(str).str.strip() == str(res['home']).strip()) & (dfp['ทีมเยือน'].astype(str).str.strip() == str(res['away']).strip())]
    if m.empty: return
    st.markdown(_card_from_row(int(res['row']), m.iloc[-1], solo=True), unsafe_allow_html=True)
 
def render_jobs(polling=False):
    jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if not jobs: return
    if polling and not any(j["status"] in ("queued", "reading", "saving") for j in jobs): st.rerun()
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
        else: title = ""
        body = title + f"<div class='job-msg'>{esc(j['msg'])}</div>"
        if j.get("note"): body += f"<div class='job-note'>{esc(str(j['note']))}</div>"
        if j.get("warn"): body += f"<div class='job-note' style='color:#e8bb62'>{esc(str(j['warn']))}</div>"
        if j["status"] in ("queued", "reading", "saving"): body += "<div class='job-bar'><div class='j-bar'></div></div>"
        st.markdown(f"<div class='job-card {cls}'>{head}{body}</div>", unsafe_allow_html=True)
        if j["status"] == "done" and res: _render_job_signal(res)
        if j["status"] == "ready":
            if st.button('กรอกฟอร์ม 5 นัด', key=f"open_form5_{j['id']}"):
                st.session_state["form5_editing"] = j["id"]
                st.rerun()
 
def clear_owner_jobs(owner_id):
    with JOBS.lock:
        for k in [k for k, j in JOBS.jobs.items() if j.get("owner_id") == owner_id and j["status"] != "saving"]:
            JOBS.jobs.pop(k)
 
def clear_upload_page():
    clear_owner_jobs(st.session_state["form5_owner_id"])
    st.session_state.pop("form5_editing", None)
    st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1
 
def _f5_number(label, key, step, as_int=False):
    # ไม่ส่ง value ซ้ำเมื่อมีค่าใน session_state แล้ว (กันคำเตือนของ Streamlit)
    kwargs = {} if key in st.session_state else {"value": 0 if as_int else 0.0}
    return st.number_input(label, step=step, key=key, **kwargs)
 
def render_form5_editor():
    job_id = st.session_state.get("form5_editing")
    job = JOBS.get(job_id) if job_id else None
    if not job or job["status"] != "ready": return
    
    panel_header("ฟอร์ม 5 นัดล่าสุด", "อัปโหลดภาพสถิติเพื่อให้ AI กรอกอัตโนมัติ หรือกรอกข้อมูลเอง")
    
    stat_file = st.file_uploader("📸 อัปโหลดภาพฟอร์มสถิติ", type=['png', 'jpg', 'jpeg'], key=f"stat_up_{job_id}")
    
    if stat_file and st.session_state.get(f"stat_processed_{job_id}") != stat_file.name:
        with st.spinner("🤖 AI กำลังอ่านข้อมูลสถิติ..."):
            try:
                form_data = extract_form_stats(stat_file.getvalue(), model)
                st.session_state[f"{job_id}_h_win"] = form_data[0]
                st.session_state[f"{job_id}_h_for"] = form_data[1]
                st.session_state[f"{job_id}_h_ag"] = form_data[2]
                st.session_state[f"{job_id}_a_win"] = form_data[3]
                st.session_state[f"{job_id}_a_for"] = form_data[4]
                st.session_state[f"{job_id}_a_ag"] = form_data[5]
                st.session_state[f"stat_processed_{job_id}"] = stat_file.name
                st.toast("✅ ดึงข้อมูลสถิติสำเร็จ!")
                st.rerun()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
 
    with st.form(f"form5_{job_id}"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"<div class='fx-side'><small>HOME</small>{esc(str(job['row_data'][0]))}</div>", unsafe_allow_html=True)
            h_win = _f5_number('Win Rate เหย้า (%)', f"{job_id}_h_win", 20.)
            h_for = _f5_number('ประตูได้ (รวม 5 นัด)', f"{job_id}_h_for", 1, as_int=True)
            h_ag = _f5_number('ประตูเสีย (รวม 5 นัด)', f"{job_id}_h_ag", 1, as_int=True)
        with right:
            st.markdown(f"<div class='fx-side away'><small>AWAY</small>{esc(str(job['row_data'][1]))}</div>", unsafe_allow_html=True)
            a_win = _f5_number('Win Rate เยือน (%)', f"{job_id}_a_win", 20.)
            a_for = _f5_number('ประตูได้ (รวม 5 นัด)', f"{job_id}_a_for", 1, as_int=True)
            a_ag = _f5_number('ประตูเสีย (รวม 5 นัด)', f"{job_id}_a_ag", 1, as_int=True)
            
        if st.form_submit_button('💾 บันทึกทั้งหมด', type="primary"):
            submit_form5(job_id, [h_win, h_for, h_ag, a_win, a_for, a_ag], st.session_state["form5_owner_id"])
            st.session_state.pop("form5_editing", None)
            st.rerun()
 
# ----------------------------------------
# AI : V5.3-AI-TRAPS (UI / OCR / sheet writes are not modified)
# -----------------------------------------------------------------------------
# This block is the complete prediction engine. It intentionally does not write
# to Sheets, change widgets, invent missing odds, or train on legacy formulas.
import copy as _ml_copy
import hashlib as _ml_hashlib
 
ML_PATCH_VERSION = "V5.3-AI-TRAPS"
ML_CONFIG = {
    "seed": 42,
    "min_rows": 100,
    "min_groups": 20,
    "min_partition": 10,
    "min_validation": 50,
    "min_calibration": 30,
    "min_form_training_rows": 60,
    "min_line_examples": 15,
    
    # 🎯 [จุดสำคัญที่ปรับจูน] 
    "min_probability": 58.0,        # [ปรับจาก 55.0] ให้เข้าทำเฉพาะคู่ที่โมเดลมั่นใจเกิน 58% เพื่อสู้กับค่าน้ำ
    "min_ev": 0.04,                 # [ปรับจาก 0.02] เข้าทำเมื่อได้ Value Bet (ผลตอบแทนคาดหวัง) คุ้มค่าความเสี่ยงมากขึ้น
    "probability_stress": 0.05,     # [ปรับจาก 0.03] เผื่อ Margin of Error ให้กว้างขึ้น ป้องกันราคาไหลหลอก
    "baseline_smoothing": 1.0,
    "baseline_mix": 0.05,           # [ปรับจาก 0.10] ลดการพึ่งพาค่าเฉลี่ยสถิติพื้นฐาน ให้ AI ตัดสินใจจาก Data เชิงลึกมากขึ้น
    
    "min_validation_gain": 0.005,
    "max_brier_degradation": 0.005,
    "max_top_label_ece": 0.15,
    "temperatures": (0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    "max_iter": 120,
    "threads": 2,
    
    # 🎯 [Money Management]
    "fractional_kelly": 0.25,        # ใช้ Kelly 1/4 เพื่อป้องกันพอร์ตแตกจากความแปรปรวน
    "max_theoretical_fraction": 0.03, # จำกัดไม่ให้ไม้ไหนเกิน 3% ของพอร์ตรวม แม้ AI จะมั่นใจมากก็ตาม
}
# Thresholds above are engineering defaults, NOT statistically proven guarantees.
# No chronology is invented: this legacy schema supports grouped research only.
ML_STATES = ("FL", "HL", "PUSH", "HW", "FW")
ML_STATE_LABELS = ("เสียเต็ม", "เสียครึ่ง", "คืนทุน", "ได้ครึ่ง", "ได้เต็ม")
ML_ODDS_COLUMNS = ["1X2 เหย้า", "1X2 เสมอ", "1X2 เยือน", "แฮนดิแคป (เหย้า)", "ค่าน้ำ HDP (เหย้า)", "แฮนดิแคป (เยือน)", "ค่าน้ำ HDP (เยือน)", "โกลสูง/ต่ำ", "ค่าน้ำ (สูง)", "ค่าน้ำ (ต่ำ)"]
_ML_HOME = "ทีมเหย้า"
_ML_AWAY = "ทีมเยือน"
_ML_SCORE = "ผลสกอร์"
_ML_PROB_COLS = {
    "HDP": "AI_โอกาสเหย้าชนะHDP(%)",
    "OU": "AI_โอกาสออกสูง(%)",
}
_ML_VERDICT_COLS = {
    "HDP": "AI_ฟันธง_HDP",
    "OU": "AI_ฟันธง_สูงต่ำ",
}
 
 
def _ml_number(value):
    if isinstance(value, (bool, np.bool_)) or value is None:
        return np.nan
    try:
        result = float(str(value).strip().replace("−", "-"))
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan
 
 
def _ml_blank(value):
    if value is None:
        return True
    return str(value).strip().lower() in ("", "nan", "none", "null", "<na>", "nat")
 
 
def _ml_win_rate(value):
    text = str(value).strip()
    percent = text.endswith("%")
    number = _ml_number(text[:-1] if percent else value)
    if not np.isfinite(number) or number < 0:
        return np.nan
    if percent or number > 1:
        number /= 100.0
    return number if 0 <= number <= 1 else np.nan
 
 
def _ml_goal_count(value):
    number = _ml_number(value)
    return number if np.isfinite(number) and 0 <= number <= 500 and float(number).is_integer() else np.nan
 
 
def _ml_malay_decimal(value):
    """DATA's four water columns are Malay, converted by the frozen uploader.
 
    Out-of-range inputs are rejected, not guessed to be HK/Decimal/Indonesian.
    """
    value = _ml_number(value)
    if not np.isfinite(value) or value == 0 or abs(value) > 1:
        return np.nan
    return 1.0 + (value if value > 0 else 1.0 / abs(value))
 
 
def _ml_score(value):
    # Reject datetime, Excel serial numbers and booleans before string conversion.
    if not isinstance(value, str):
        return None
    text = value.strip().replace("−", "-").replace("–", "-")
    match = re.fullmatch(r"([0-9]{1,2})\s*[-:]\s*([0-9]{1,2})", text)
    return tuple(map(int, match.groups())) if match else None
 
 
def _ml_valid_line(value, market):
    if market not in ("HDP", "OU"):
        return False
    value = _ml_number(value)
    return bool(np.isfinite(value) and abs(value) <= 100
                and abs(value * 4 - round(value * 4)) < 1e-7
                and (market == "HDP" or value > 0))
 
 
def _ml_settlement(score, line, market):
    if score is None or not _ml_valid_line(line, market):
        return np.nan
    if (not isinstance(score, (tuple, list)) or len(score) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or v < 0 for v in score)):
        return np.nan
    line = float(line)
    parts = (np.floor(line * 2) / 2, np.ceil(line * 2) / 2)
    differences = ([score[0] - score[1] + p for p in parts] if market == "HDP"
                   else [sum(score) - p for p in parts])
    return int(round(float(np.mean(np.sign(differences))) * 2)) + 2
 
 
def _ml_allowed(line, market):
    if not _ml_valid_line(line, market):
        raise ValueError("ไลน์ไม่ถูกต้อง")
    family = int(round(float(line) * 4)) % 4
    allowed = np.array([True, False, False, False, True])
    if family == 0:
        allowed[2] = True
    elif family in (1, 3):
        half_win = (family == 1) if market == "HDP" else (family == 3)
        allowed[3 if half_win else 1] = True
    if market == "OU" and np.isclose(float(line), 0.25):
        allowed[0] = False
    return allowed
 
 
def _ml_probability(raw, lines, market, temperature=1.0):
    raw = np.asarray(raw, dtype=float)
    lines = np.asarray(lines, dtype=float)
    if len(raw) == 0:
        return np.empty((0, 5), dtype=float)
    if (raw.shape != (len(lines), 5) or not np.isfinite(raw).all()
            or (raw < 0).any() or (raw.sum(axis=1) <= 0).any()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("ความน่าจะเป็นไม่ถูกต้อง")
    masks = np.vstack([_ml_allowed(line, market) for line in lines])
    logits = np.log(np.clip(raw, 1e-12, 1.0)) / float(temperature)
    logits[~masks] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)
 
 
def _ml_raw_probabilities(estimator, X):
    raw = np.full((len(X), 5), 1e-12, dtype=float)
    if len(X):
        classes = np.asarray(estimator.classes_, dtype=int)
        if len(set(classes)) != len(classes) or (classes < 0).any() or (classes > 4).any():
            raise ValueError("Invalid model class mapping")
        raw[:, classes] = estimator.predict_proba(X)
    return raw
 
 
def _ml_log_loss(y, probabilities):
    y = np.asarray(y, dtype=int)
    return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean()) if len(y) else np.nan
 
 
def _ml_home_share(probabilities):
    """Legacy display contract: P(home/over positive | not a push)."""
    p = np.asarray(probabilities, dtype=float)
    decisive = p[:, [0, 1, 3, 4]].sum(axis=1)
    result = np.full(len(p), np.nan)
    np.divide(p[:, 3:].sum(axis=1), decisive, out=result, where=decisive > 1e-12)
    return result
 
 
def _ml_metrics(y, probabilities):
    """Keep the legacy >=55% directional KPI; add explicit denominators.
 
    Policy results are computed separately by _ml_policy_metrics. Never relabel
    this filtered directional KPI as the win rate of recommended selections.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if not len(y):
        return {"n": 0, "correct": 0, "accuracy": np.nan, "log_loss": np.nan,
                "brier": np.nan, "ece": np.nan, "directional_n": 0,
                "directional_correct": 0, "class_accuracy": np.nan}
    share = _ml_home_share(p)
    sign = np.where(share > 0.5, 1, -1)
    outcome = np.sign(y - 2)
    use = np.isfinite(share) & ((share >= 0.55) | (share <= 0.45)) & (outcome != 0)
    n = int(use.sum())
    correct = int((sign[use] == outcome[use]).sum())
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    top_correct = predicted == y
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & ((confidence < high) if high < .999 else (confidence <= 1))
        if selected.any():
            ece += float(selected.mean()) * abs(float(top_correct[selected].mean()) - float(confidence[selected].mean()))
    return {"n": len(y), "correct": int(top_correct.sum()),
            "accuracy": correct / n * 100 if n else np.nan,
            "class_accuracy": float(top_correct.mean() * 100),
            "directional_n": n, "directional_correct": correct,
            "directional_coverage": n / len(y), "push_n": int((y == 2).sum()),
            "log_loss": _ml_log_loss(y, p),
            "brier": float(np.square(p - np.eye(5)[y]).sum(axis=1).mean()),
            "ece": float(ece)}
 
 
def _ml_returns(decimal):
    value = _ml_number(decimal)
    if not np.isfinite(value) or value <= 1:
        return np.full(5, np.nan)
    return np.array([-1., -.5, 0., (value - 1) / 2, value - 1])
 
 
def _ml_expected_value(probabilities, decimal, reverse=False):
    p = np.asarray(probabilities, dtype=float)
    r = _ml_returns(decimal)
    if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1) or not np.isfinite(r).all()):
        return np.nan
    return float((p[::-1] if reverse else p) @ r)
 
 
def _ml_kelly_fraction(probabilities, decimal, side, config=None):
    """Bounded fractional expected-log optimum, not the old mean/variance proxy.
 
    Kept for diagnostics only. Research verdicts never recommend a bankroll size.
    """
    cfg = ML_CONFIG if config is None else config
    p = np.asarray(probabilities, dtype=float)
    returns = _ml_returns(decimal)
    if side == 1:
        returns = returns[::-1]
    if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1) or not np.isfinite(returns).all()
            or np.dot(p, returns) <= 0):
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-9
    for _ in range(60):
        middle = (lo + hi) / 2
        derivative = float(np.dot(p, returns / (1 + middle * returns)))
        if derivative > 0:
            lo = middle
        else:
            hi = middle
    return float(min((lo + hi) / 2 * cfg["fractional_kelly"], cfg["max_theoretical_fraction"]))
 
 
def _ml_stress_ev(p, decimal, reverse=False, amount=None):
    """Worst expected return in a total-variation ball, moving mass high to low.
 
    A sensitivity test using a user-set radius, NOT a confidence interval.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (5,) or not np.isfinite(p).all() or not np.isfinite(_ml_number(decimal)) or decimal <= 1:
        return np.nan
    if (p < 0).any() or not np.isclose(p.sum(), 1):
        return np.nan
    q = (p[::-1] if reverse else p).copy()
    feasible = np.flatnonzero(q > 0)
    worst = int(feasible[0])
    remaining = float(ML_CONFIG["probability_stress"] if amount is None else amount)
    if not 0 <= remaining <= 1:
        raise ValueError("Invalid stress radius")
    for high in range(4, worst, -1):
        move = min(remaining, q[high], 1.0 - q[worst])
        q[high] -= move
        q[worst] += move
        remaining -= move
        if remaining <= 1e-12:
            break
    return float(q @ _ml_returns(decimal))
 
 
def _ml_choices(probabilities, first_prices, second_prices, config=None):
    cfg = ML_CONFIG if config is None else config
    p = np.asarray(probabilities, dtype=float)
    first_prices, second_prices = np.asarray(first_prices, float), np.asarray(second_prices, float)
    if p.shape != (len(first_prices), 5) or len(second_prices) != len(first_prices):
        raise ValueError("Probability and price row counts do not match")
    first = np.array([_ml_expected_value(x, v) for x, v in zip(p, first_prices)])
    second = np.array([_ml_expected_value(x, v, True) for x, v in zip(p, second_prices)])
    side = np.where(np.nan_to_num(first, nan=-np.inf) >= np.nan_to_num(second, nan=-np.inf), 0, 1)
    price = np.where(side == 0, first_prices, second_prices)
    ev = np.where(side == 0, first, second)
    stressed = np.array([_ml_stress_ev(x, v, bool(s), cfg["probability_stress"]) for x, v, s in zip(p, price, side)])
    candidate = np.isfinite(ev) & (ev >= cfg["min_ev"]) & (stressed > 0)
    kelly = np.array([_ml_kelly_fraction(x, v, int(s), cfg) for x, v, s in zip(p, price, side)])
    return {"side": side, "ev": ev, "stress_ev": stressed, "price": price,
            "candidate": candidate, "kelly": kelly}
 
 
def _ml_prepare(df_in):
    """Derive labels from checked scores only. Raw source columns remain intact."""
    if not isinstance(df_in, pd.DataFrame):
        raise ValueError("DATA must be a DataFrame")
    df = df_in.copy(deep=True).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("หัวคอลัมน์ DATA ซ้ำกัน")
    for col in (_ML_HOME, _ML_AWAY, _ML_SCORE):
        if col not in df:
            df[col] = ""
    for col in ML_ODDS_COLUMNS + FORM5_HEADERS:
        if col not in df:
            df[col] = np.nan
            
    # Retain invalid named rows for data-quality reporting; drop truly empty rows.
    keep = ~(df[_ML_HOME].map(_ml_blank) & df[_ML_AWAY].map(_ml_blank))
    df = df.loc[keep].reset_index(drop=True)
    num = df[ML_ODDS_COLUMNS].apply(lambda column: column.map(_ml_number))
    X = pd.DataFrame(index=df.index)
    
    odds = num[ML_ODDS_COLUMNS[:3]].where(lambda values: values > 1)
    inv = 1.0 / odds
    total = inv.sum(axis=1, min_count=3)
    for label, col in zip(("Home", "Draw", "Away"), ML_ODDS_COLUMNS[:3]):
        X[f"Market_{label}_Prob"] = inv[col] / total
    X["Bookie_Margin"] = total - 1.0
    X["Has_1X2"] = odds.notna().all(axis=1).astype(float)
    X["1X2_Variance"] = X[["Market_Home_Prob", "Market_Draw_Prob", "Market_Away_Prob"]].var(axis=1)
    
    for feature, offset in (("HDP_Home_Decimal", 4), ("HDP_Away_Decimal", 6),
                            ("Over_Decimal", 8), ("Under_Decimal", 9)):
        X[feature] = df[ML_ODDS_COLUMNS[offset]].map(_ml_malay_decimal)
        
    X["Water_HDP_Diff"] = X["HDP_Home_Decimal"] - X["HDP_Away_Decimal"]
    X["Water_OU_Diff"] = X["Over_Decimal"] - X["Under_Decimal"]
    
    for market, offset in (("HDP", 3), ("OU", 7)):
        line = num[ML_ODDS_COLUMNS[offset]]
        X[market + "_Line"] = line.where(line.map(lambda v: _ml_valid_line(v, market)))
        X[market + "_Family"] = np.round(X[market + "_Line"] * 4) % 4
        
    for col in FORM5_HEADERS:
        X[col] = df[col].map(_ml_win_rate if "WinRate" in col else _ml_goal_count)
        
    X["Form_Present_Count"] = X[FORM5_HEADERS].notna().sum(axis=1).astype(float)
    X["Form_Complete"] = X[FORM5_HEADERS].notna().all(axis=1).astype(float)
    for col in FORM5_HEADERS:
        X[col + "_Missing"] = X[col].isna().astype(float)
        
    h_for, h_against = X[FORM5_HEADERS[1]] / 5.0, X[FORM5_HEADERS[2]] / 5.0
    a_for, a_against = X[FORM5_HEADERS[4]] / 5.0, X[FORM5_HEADERS[5]] / 5.0
    X["Form_Home_Goal_Avg"] = h_for
    X["Form_Away_Goal_Avg"] = a_for
    X["Form_WinRate_Difference"] = X[FORM5_HEADERS[0]] - X[FORM5_HEADERS[3]]
    X["Expected_Total_Goals"] = (h_for + a_against + a_for + h_against) / 2.0
    X["Form_Total_vs_Line"] = X["Expected_Total_Goals"] - X["OU_Line"]
 
    # ==========================================
    # 🚨 [V5.5 REALITY CHECK & ANTI-BAIT TRAPS] 🚨
    # ==========================================
    # 1. HDP vs Form Mismatch & Fake Steam Indicator
    X["HDP_vs_Form_Mismatch"] = X["Form_WinRate_Difference"] + (X["HDP_Line"] / 2.5)
    X["Is_Fake_Steam"] = (X["HDP_vs_Form_Mismatch"].abs() > 1.0).astype(float)

    # 2. 1X2 Anchor Divergence (Noisy Crash Detection)
    X["HDP_Home_Implied_Prob"] = 1.0 / (X["HDP_Home_Decimal"] + 0.001)
    X["1X2_vs_HDP_Anomaly"] = (X["Market_Home_Prob"] - X["HDP_Home_Implied_Prob"]).abs()
    X["Noisy_Crash_Flag"] = ((X["1X2_vs_HDP_Anomaly"] > 0.15) & (X["Bookie_Margin"] < 0.08)).astype(float)

    # 3. Over-Bait Risk Indicator
    avg_conceded_total = (X[FORM5_HEADERS[2]] + X[FORM5_HEADERS[5]]) / 10.0
    X["Over_Bait_Risk"] = ((X["OU_Line"] >= 3.0) & (avg_conceded_total < 1.5)).astype(float)

    # 4. Cup High HDP Penalty Check (ถ้ามีคอลัมน์ Tournament_Type ใน Google Sheet)
    if "Tournament_Type" in df.columns:
        is_cup = df["Tournament_Type"].map(lambda v: 1 if str(v).strip().lower() in ["cup", "acl", "ถ้วย"] else 0)
        X["Cup_High_HDP_Risk"] = ((X["HDP_Line"] <= -1.25) & (is_cup == 1)).astype(float)
    else:
        X["Cup_High_HDP_Risk"] = 0.0

    # 5. Feature ดั้งเดิม
    X["Home_Goal_Pressure"] = X["Form_Home_Goal_Avg"] - (X[FORM5_HEADERS[2]] / 5.0)
    X["Away_Goal_Pressure"] = X["Form_Away_Goal_Avg"] - (X[FORM5_HEADERS[5]] / 5.0)
    X["HDP_to_OU_Ratio"] = X["HDP_Line"].abs() / (X["OU_Line"] + 0.001)
    X["Form_vs_Juice_Bait"] = X["Form_WinRate_Difference"] * X["Water_HDP_Diff"]
    X["Defensive_Fragility"] = (X[FORM5_HEADERS[2]] / 5.0) * (X[FORM5_HEADERS[5]] / 5.0)

    means = X["Expected_Total_Goals"].to_numpy(float)
    lines = X["OU_Line"].to_numpy(float)
    usable = np.isfinite(means) & (means >= 0) & np.isfinite(lines)
    for feature in ("Poisson_Over_Positive", "Poisson_Under_Positive", "Poisson_Push"):
        X[feature] = np.nan
    if usable.any():
        low = np.floor(lines[usable] * 2) / 2
        high = np.ceil(lines[usable] * 2) / 2
        X.loc[usable, "Poisson_Over_Positive"] = stats.poisson.sf(np.floor(low), means[usable])
        X.loc[usable, "Poisson_Under_Positive"] = stats.poisson.cdf(np.ceil(high) - 1, means[usable])
        whole = np.isclose(low, high) & np.isclose(low, np.round(low))
        X.loc[usable, "Poisson_Push"] = np.where(whole, stats.poisson.pmf(np.round(low), means[usable]), 0.0)
        
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    
    names_ok = ~df[_ML_HOME].map(_ml_blank) & ~df[_ML_AWAY].map(_ml_blank)
    names_ok &= df[_ML_HOME].astype(str).str.strip().str.casefold().ne(df[_ML_AWAY].astype(str).str.strip().str.casefold())
    df["_ML_Group"] = [json.dumps(sorted([re.sub(r"\s+", " ", str(h).strip().casefold()),
                                         re.sub(r"\s+", " ", str(a).strip().casefold())]), ensure_ascii=False)
                        for h, a in zip(df[_ML_HOME], df[_ML_AWAY])]
    score = df[_ML_SCORE].map(_ml_score)
    score_blank = df[_ML_SCORE].map(_ml_blank)
    df["_ML_InvalidScore"] = ~score_blank & score.isna()
    df["_ML_Pending"] = names_ok & score_blank
    
    away_line = num[ML_ODDS_COLUMNS[5]]
    matching_hdp = np.isclose(X["HDP_Line"], -away_line, equal_nan=False)
    
    for market, first, second in (("HDP", "HDP_Home_Decimal", "HDP_Away_Decimal"),
                                  ("OU", "Over_Decimal", "Under_Decimal")):
        ok = names_ok & X[[market + "_Line", first, second]].notna().all(axis=1)
        if market == "HDP":
            ok &= matching_hdp
        df["_ML_Valid_" + market] = ok
        target = [_ml_settlement(s, v, market) if valid else np.nan
                  for s, v, valid in zip(score, X[market + "_Line"], ok)]
        df["Target_" + market] = target
        
    return df, X
 
 
def _ml_split(groups, config=None):
    """Stable four-way GROUP split, explicitly NOT a chronological backtest."""
    cfg = ML_CONFIG if config is None else config
    groups = np.asarray(groups, dtype=str)
    if len(np.unique(groups)) < cfg["min_groups"]:
        raise ValueError("กลุ่มข้อมูลอิสระยังน้อยเกินไป")
    def take(subset, fraction, seed):
        split = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        a, b = next(split.split(subset, groups=groups[subset]))
        return subset[a], subset[b]
    rest, test = take(np.arange(len(groups)), .15, cfg["seed"])
    rest, validation = take(rest, .15 / .85, cfg["seed"] + 1)
    train, calibration = take(rest, .15 / .70, cfg["seed"] + 2)
    parts = dict(train=train, calibration=calibration, validation=validation, test=test)
    if min(map(len, parts.values())) < cfg["min_partition"]:
        raise ValueError("จำนวนข้อมูลในชุดฝึกหรือทดสอบไม่พอ")
    return parts
 
 
def _ml_training_payload(df, X, market):
    valid = df["_ML_Valid_" + market] & df["Target_" + market].notna()
    buckets = {}
    for position in np.flatnonzero(valid.to_numpy()):
        row = X.iloc[position]
        values = [float(v) if np.isfinite(v) else None for v in row]
        group = df.iloc[position]["_ML_Group"]
        key = json.dumps([group, values], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        buckets.setdefault(key, set()).add(int(df.iloc[position]["Target_" + market]))
    # Same input and pair with conflicting labels cannot be resolved without IDs.
    conflicts = sum(len(labels) != 1 for labels in buckets.values())
    rows = []
    for key in sorted(buckets):
        labels = buckets[key]
        if len(labels) == 1:
            group, values = json.loads(key)
            rows.append([group, next(iter(labels)), values])
    payload = json.dumps({"features": list(X.columns), "rows": rows}, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    audit = {"eligible_rows": int(valid.sum()), "unique_training_rows": len(rows),
             "conflicting_input_groups": conflicts,
             "excluded_or_collapsed": int(valid.sum()) - len(rows)}
    return payload, audit
 
 
def _ml_baseline_fit(y, lines, smoothing):
    counts = np.full((4, 5), float(smoothing))
    for label, line in zip(y, lines):
        counts[int(round(float(line) * 4)) % 4, int(label)] += 1
    return counts
 
 
def _ml_baseline_raw(counts, lines):
    families = np.round(np.asarray(lines) * 4).astype(int) % 4
    values = counts[families]
    return values / values.sum(axis=1, keepdims=True)
 
 
@st.cache_resource(show_spinner=False)
def _ml_execution_lock():
    # threadpoolctl changes process-level limits: serialize its scoped use here.
    return threading.RLock()
 
 
def _ml_candidate_raw(bundle, name, X, lines):
    base = _ml_baseline_raw(bundle["baseline"], lines)
    if name == "baseline":
        return base
    mix = bundle["config"]["baseline_mix"]
    if name == "blend":
        names = list(bundle["models"])
        if not names:
            return base
        raw = np.mean([_ml_raw_probabilities(bundle["models"][n][0], X[bundle["models"][n][1]]) for n in names], axis=0)
    else:
        estimator, columns = bundle["models"][name]
        raw = _ml_raw_probabilities(estimator, X[columns])
    return (1 - mix) * raw + mix * base
 
 
def _ml_predict_bundle(bundle, X, lines):
    with _ml_execution_lock(), threadpool_limits(limits=bundle["config"]["threads"]):
        raw = _ml_candidate_raw(bundle, bundle["selected"], X, lines)
        return _ml_probability(raw, lines, bundle["market"], bundle["temperature"])
 
 
def _ml_selection_mask(p, first, second, lines, report, config=None):
    cfg = ML_CONFIG if config is None else config
    choices = _ml_choices(p, first, second, cfg)
    share = _ml_home_share(p)
    selected_prob = np.where(choices["side"] == 0, share, 1 - share) * 100
    support = np.array([report["family_counts"].get(int(round(float(line) * 4)) % 4, 0) for line in lines])
    bounds = report.get("line_range", [-np.inf, np.inf])
    in_range = (np.asarray(lines) >= bounds[0]) & (np.asarray(lines) <= bounds[1])
    mask = (choices["candidate"] & (selected_prob >= cfg["min_probability"])
            & (support >= cfg["min_line_examples"]) & in_range & bool(report["gate_pass"]))
    return mask, choices, selected_prob
 
 
def _ml_policy_metrics(y, p, first, second, lines, report, config=None):
    mask, choices, _ = _ml_selection_mask(p, first, second, lines, report, config)
    selected = np.flatnonzero(mask)
    returns = np.array([_ml_returns(choices["price"][i])[int(y[i]) if choices["side"][i] == 0 else 4 - int(y[i])] for i in selected])
    decisive = returns != 0
    n = int(decisive.sum())
    return {"eligible_test_rows": len(y), "research_signals": len(selected),
            "decisive_n": n, "push_n": int((~decisive).sum()),
            "positive_n": int((returns > 0).sum()),
            "positive_rate_excluding_push": float((returns > 0).sum() / n * 100) if n else None,
            "unit_return_sum": float(returns.sum()),
            "unit_return_mean": float(returns.mean()) if len(returns) else None,
            "stake_recommendations": 0,
            "note": "Grouped held-out simulation; no timestamps; not live prediction history."}
 
 
@st.cache_resource(max_entries=4, show_spinner=False)
def _ml_train_cached(payload, market, config_json, engine_version):
    """Only canonical labeled training records affect this resource cache.
 
    Pending rows, legacy formulas, row order and sheet row numbers do not.
    No pickle/model file is loaded. A process restart retrains lazily.
    """
    cfg = json.loads(config_json)
    source = json.loads(payload)
    records = source["rows"]
    if len(records) < cfg["min_rows"]:
        raise ValueError(f"{market}: ต้องมีข้อมูลที่ใช้ได้อย่างน้อย {cfg['min_rows']} รายการ")
    groups = np.array([r[0] for r in records])
    y = np.array([r[1] for r in records], dtype=int)
    X = pd.DataFrame([r[2] for r in records], columns=source["features"], dtype=float)
    parts = _ml_split(groups, cfg)
    train, cal, val, test = (parts[key] for key in ("train", "calibration", "validation", "test"))
    lines = X[market + "_Line"].to_numpy(float)
    baseline = _ml_baseline_fit(y[train], lines[train], cfg["baseline_smoothing"])
    bundle = {"baseline": baseline, "models": {}, "market": market, "config": cfg}
    candidates = ["baseline"]
    errors = []
    active = [col for col in X if X.iloc[train][col].nunique(dropna=True) > 1]
    form_features = (set(FORM5_HEADERS)
                     | {c for c in X if c.startswith(("Form_", "Poisson_", "Expected_")) or c.endswith("_Missing")}
                     | {"HDP_vs_Form_Mismatch", "Home_Goal_Pressure", "Away_Goal_Pressure", "Defensive_Fragility"})
    price_features = [c for c in active if c not in form_features]
    model_feature_sets = [("price", price_features)]
    if int(X.iloc[train]["Form_Complete"].sum()) >= cfg["min_form_training_rows"] and set(active) - set(price_features):
        model_feature_sets.append(("price_form", active))
    with _ml_execution_lock(), threadpool_limits(limits=cfg["threads"]):
        if len(np.unique(y[train])) > 1:
            for number, (name, columns) in enumerate(model_feature_sets):
                if not columns:
                    continue
                estimator = HistGradientBoostingClassifier(
                    max_iter=cfg["max_iter"], max_depth=3, max_leaf_nodes=7,
                    learning_rate=.035, l2_regularization=8.0,
                    min_samples_leaf=max(15, min(50, len(train) // 30)),
                    early_stopping=False, random_state=cfg["seed"] + number)
                try:
                    estimator.fit(X.iloc[train][columns], y[train])
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    errors.append({"candidate": name, "type": type(exc).__name__})
                    continue
                bundle["models"][name] = (estimator, columns)
                candidates.append(name)
        if len(bundle["models"]) > 1:
            candidates.append("blend")
        evaluation = {}
        for name in candidates:
            raw_cal = _ml_candidate_raw(bundle, name, X.iloc[cal], lines[cal])
            temperature = min(cfg["temperatures"], key=lambda t: _ml_log_loss(y[cal], _ml_probability(raw_cal, lines[cal], market, t)))
            p_val = _ml_probability(_ml_candidate_raw(bundle, name, X.iloc[val], lines[val]), lines[val], market, temperature)
            evaluation[name] = {"temperature": float(temperature), "validation": _ml_metrics(y[val], p_val)}
        # Select ONLY on validation. Test labels never select, calibrate or gate.
        base_metrics = evaluation["baseline"]["validation"]
        selected = "baseline"
        for name in candidates[1:]:
            metrics = evaluation[name]["validation"]
            if (metrics["log_loss"] <= base_metrics["log_loss"] - cfg["min_validation_gain"]
                    and metrics["brier"] <= base_metrics["brier"] + cfg["max_brier_degradation"]
                    and metrics["log_loss"] < evaluation[selected]["validation"]["log_loss"]):
                selected = name
        val_metrics = evaluation[selected]["validation"]
        reasons = []
        if len(val) < cfg["min_validation"]:
            reasons.append("ชุดตรวจสอบน้อยเกินไป")
        if len(cal) < cfg["min_calibration"]:
            reasons.append("ชุดปรับความน่าจะเป็นน้อยเกินไป")
        if selected == "baseline":
            reasons.append("ยังไม่ดีกว่าวิธีพื้นฐานตามเกณฑ์")
        if val_metrics["ece"] > cfg["max_top_label_ece"]:
            reasons.append("ความน่าจะเป็นยังคลาดเคลื่อนมาก")
        bundle.update(selected=selected, temperature=evaluation[selected]["temperature"])
        p_test = _ml_probability(_ml_candidate_raw(bundle, selected, X.iloc[test], lines[test]), lines[test], market, bundle["temperature"])
        p_base = _ml_probability(_ml_baseline_raw(baseline, lines[test]), lines[test], market, evaluation["baseline"]["temperature"])
    families = np.round(lines[train] * 4).astype(int) % 4
    report = {
        "market": market, "status": "ok", "engine_version": engine_version,
        "model_id": _ml_hashlib.sha256((payload + market + config_json + engine_version).encode()).hexdigest()[:16],
        "selected_model": selected, "temperature": bundle["temperature"],
        "evaluation_mode": "GROUPED_RESEARCH_ONLY", "deployment_ready": False,
        "gate_pass": not reasons, "gate_reasons": reasons,
        "family_counts": {f: int((families == f).sum()) for f in range(4)},
        "line_range": [float(lines[train].min()), float(lines[train].max())],
        "split_counts": {key: len(value) for key, value in parts.items()},
        "candidate_validation": evaluation, "validation": val_metrics,
        "test": _ml_metrics(y[test], p_test), "baseline_test": _ml_metrics(y[test], p_base),
        "model_errors": errors, "train_form_complete": int(X.iloc[train]["Form_Complete"].sum()),
        "note": "ผลทดลองแบบแบ่งกลุ่ม ยังไม่ได้ทดสอบตามเวลาจริง",
    }
    first, second = (("HDP_Home_Decimal", "HDP_Away_Decimal") if market == "HDP" else ("Over_Decimal", "Under_Decimal"))
    report["policy_test"] = _ml_policy_metrics(y[test], p_test, X.iloc[test][first], X.iloc[test][second], lines[test], report, cfg)
    # Store only hashes for split auditing, not names or private input records.
    report["split_group_hashes"] = {
        key: sorted({_ml_hashlib.sha256(str(groups[i]).encode()).hexdigest() for i in indices})
        for key, indices in parts.items()}
    bundle["report"] = report
    return bundle
 
 
def _ml_fit_market(df, X, market):
    payload, audit = _ml_training_payload(df, X, market)
    config_json = json.dumps(ML_CONFIG, sort_keys=True, allow_nan=False)
    bundle = _ml_train_cached(payload, market, config_json, ML_PATCH_VERSION)
    report = _ml_copy.deepcopy(bundle["report"])
    report["data_audit"] = audit
    pending = np.flatnonzero(df["_ML_Pending"].to_numpy())
    p = np.full((len(pending), 5), np.nan)
    use = df.iloc[pending]["_ML_Valid_" + market].to_numpy(bool)
    if use.any():
        positions = pending[use]
        p[use] = _ml_predict_bundle(bundle, X.iloc[positions], X.iloc[positions][market + "_Line"].to_numpy(float))
    return report, p
 
 
def _ml_verdict(market, side, prob_val, ev, stress_ev, line, valid, report, kelly,
                row_data=None, features_data=None, min_prob=58.0, min_ev=None):
    """Neutral research results with V5.5 Hard Rule Safeguards."""
    prefix = "⚠️ ข้าม"
    if not valid:
        return prefix + " (ราคาหรือไลน์ไม่ครบ/ผิดรูปแบบ)"
    if report.get("status") != "ok":
        return prefix + " (โมเดลตลาดนี้ยังไม่พร้อม)"
    if not report.get("gate_pass", False):
        reason = "; ".join(report.get("gate_reasons", [])) or "ไม่ผ่านเกณฑ์คุณภาพ"
        return prefix + " (" + reason + ") | ค่าประเมินเพื่อทดลอง"
        
    # 🚨 [V5.5 HARD RULES - กรองกับดักราคาหลอกก่อนอนุมัติสัญญาณ]
    if features_data is not None:
        try:
            # 1. ดักจับ Noisy Crash
            if float(features_data.get("Noisy_Crash_Flag", 0)) > 0:
                return prefix + " (ตรวจพบ Noisy Crash / ราคาไหลหลอก ขาด 1X2 Anchor รองรับ)"
            # 2. ดักจับ Over-Baiting (ถ้าจะฟันธงสูง แต่ติดกับดัก Over-Bait)
            if market == "OU" and side == 0 and float(features_data.get("Over_Bait_Risk", 0)) > 0:
                return prefix + " (เสี่ยง Over-Bait / สถิติประตูเสียน้อยกว่าเรต OU ที่เปิดล่อ)"
            # 3. ดักจับ บอลถ้วยต่อแพง
            if market == "HDP" and float(features_data.get("Cup_High_HDP_Risk", 0)) > 0:
                return prefix + " (บอลถ้วยทวีปต่อแพงเกิน -1.25 เสี่ยงชนะไม่ขาด/รองเจ้าบ้าน)"
        except Exception:
            pass

    family = int(round(float(line) * 4)) % 4
    if report["family_counts"].get(family, 0) < ML_CONFIG["min_line_examples"]:
        return prefix + " (ตัวอย่างไลน์ชนิดนี้น้อยเกินไป)"
    bounds = report.get("line_range", [-np.inf, np.inf])
    if not bounds[0] <= float(line) <= bounds[1]:
        return prefix + " (ไลน์นอกช่วงข้อมูลฝึก)"
        
    min_ev = ML_CONFIG["min_ev"] if min_ev is None else min_ev
    if not np.isfinite(prob_val) or prob_val < min_prob:
        return prefix + " (ความน่าจะเป็นไม่ถึงเกณฑ์)"
    if not np.isfinite(ev) or not np.isfinite(stress_ev) or ev < min_ev or stress_ev <= 0:
        return prefix + " (ผลตอบแทนคาดหมายไม่ผ่านเกณฑ์เผื่อความคลาดเคลื่อน)"
        
    desc = ((_ML_HOME if side == 0 else _ML_AWAY) if market == "HDP"
            else ("สูง" if side == 0 else "ต่ำ"))
            
    recommended_bet = max(1.0, kelly * 100) 
    
    return (f"⭐ ฟันธง: {desc} | โอกาสชนะ: {prob_val:.1f}% (EV: +{ev:.2f}) | "
            f"💰 แนะนำลงทุน: {recommended_bet:.1f}% ของพอร์ต")
 
 
@st.cache_data(ttl=300, show_spinner=False)
def compute_ai(df_in):
    try:
        df, X = _ml_prepare(df_in)
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "error", "level": "error", "msg": str(exc), "engine_version": ML_PATCH_VERSION}
    pending_positions = np.flatnonzero(df["_ML_Pending"].to_numpy())
    pending = df.iloc[pending_positions].copy()
    reports = {}
    market_failures = []
    for market, first, second in (("HDP", "HDP_Home_Decimal", "HDP_Away_Decimal"),
                                  ("OU", "Over_Decimal", "Under_Decimal")):
        try:
            report, p = _ml_fit_market(df, X, market)
        except Exception as exc:
            # A market-local failure must not suppress the other market.
            # No traceback locals, keys or provider responses are included.
            report = {"market": market, "status": "unavailable", "gate_pass": False,
                      "error_type": type(exc).__name__, "family_counts": {},
                      "engine_version": ML_PATCH_VERSION,
                      "msg": str(exc) if isinstance(exc, ValueError) else "คำนวณตลาดนี้ไม่สำเร็จ"}
            p = np.full((len(pending), 5), np.nan)
            market_failures.append(market)
        reports[market] = report
        home_prob = _ml_home_share(p) * 100
        pending[_ML_PROB_COLS[market]] = np.round(home_prob, 1)
        choices = _ml_choices(p, X.iloc[pending_positions][first], X.iloc[pending_positions][second])
        selected_prob = np.where(choices["side"] == 0, home_prob, 100 - home_prob)
        pending[_ML_VERDICT_COLS[market]] = [
            _ml_verdict(market, int(choices["side"][i]), selected_prob[i], choices["ev"][i],
                        choices["stress_ev"][i], X.iloc[pos][market + "_Line"],
                        bool(df.iloc[pos]["_ML_Valid_" + market]), report, choices["kelly"][i],
                        row_data=df.iloc[pos], features_data=X.iloc[pos],
                        min_prob=ML_CONFIG["min_probability"])
            for i, pos in enumerate(pending_positions)]
        for k, state in enumerate(ML_STATES):
            pending[f"AI_{market}_{state}"] = p[:, k]
        pending[f"AI_{market}_Positive_Unconditional"] = p[:, 3:].sum(axis=1)
        pending[f"AI_{market}_SelectedSide"] = choices["side"]
        pending[f"AI_{market}_EV"] = choices["ev"]
        pending[f"AI_{market}_StressEV"] = choices["stress_ev"]
        pending[f"AI_{market}_TheoreticalFraction"] = choices["kelly"]
    history = df.loc[df["Target_HDP"].notna() | df["Target_OU"].notna()].copy()
    # Frozen widgets cannot render missing gauges honestly. Keep such rows in the
    # detailed output, but omit them from the legacy paired-gauge card output.
    # Tab2's independent raw loader still lists them and accepts their scores.
    complete = np.isfinite(pending[list(_ML_PROB_COLS.values())].to_numpy(float)).all(axis=1)
    display = pending.loc[complete].copy()
    def accuracy(market):
        return round(float(reports[market].get("test", {}).get("accuracy", np.nan)), 1)
    status = "ok" if len(market_failures) < 2 else "few"
    return {"status": status, "level": "warning" if market_failures else "info",
            "msg": " / ".join(reports[m].get("msg", "") for m in market_failures),
            "engine_version": ML_PATCH_VERSION,
            "df_predict": display, "df_predict_details": pending,
            "df_history": history, "total_train": len(history),
            "hdp_acc": accuracy("HDP"), "ou_acc": accuracy("OU"),
            "reports": reports,
            "data_quality": {"named_rows": len(df), "pending": len(pending),
                             "invalid_scores": int(df["_ML_InvalidScore"].sum()),
                             "cards_without_both_probabilities": int((~complete).sum())},
            "evaluation_mode": "GROUPED_RESEARCH_ONLY"}
 
 
# The original DATA loader is retained for compatibility with callers.
def load_dashboard_data(_workbook, version=0):
    with JOBS.sheet_lock:
        return pd.DataFrame(get_ws(_workbook).get_all_records())
 
 
def get_ai():
    if not workbook:
        return None
    try:
        # Reuse the existing 5.1.4 raw DATA reader: UNFORMATTED_VALUE preserves
        # numeric Excel-date scores so _ml_score rejects them. No API writes here.
        data = _t2_load_data(workbook, _t2_workbook_key(workbook), JOBS.data_version)
        return compute_ai(data)
    except Exception as exc:
        return {"status": "error", "level": "error", "engine_version": ML_PATCH_VERSION,
                "msg": "โหลดข้อมูลสำหรับ AI ไม่สำเร็จ: " + type(exc).__name__}
 
 
# ----------------------------------------
# UI Components & Gauges
# ----------------------------------------
def _pred_tone(text, kind):
    text = str(text)
    if "อันตราย" in text: return "err"
    if "ข้าม" in text: return "skip"
    if "VIP" in text: return "gs"
    if "⭐⭐" in text: return "gf"
    if "⭐" in text: return "cs"
    return "cf"
 
_TONE_COLORS = {
    "hdp": {"gs": GREEN, "gf": "#4ade80", "cs": "#e8bb62", "cf": AMBER, "err": RED, "skip": "#4f6b5c"},
    "ou": {"gs": "#38bdf8", "gf": CYAN_SOFT, "cs": "#e8bb62", "cf": AMBER, "err": RED, "skip": "#4f6b5c"},
}
_TONE_STRENGTH = {"gs": 4, "gf": 3, "cs": 2, "cf": 1, "err": 4, "skip": 0}
 
def _fmt_num(v, signed=False):
    try: return f"{float(v):+g}" if signed else f"{float(v):g}"
    except (TypeError, ValueError): return "-"
 
_ECG_PATH = "M0 15 H58 L64 15 L70 5 L78 25 L84 15 H128 L134 10 L140 15 H200"
 
def _ecg_svg():
    return ("<svg class='ecg' viewBox='0 0 200 30' preserveAspectRatio='none'>"
            f"<path d='{_ECG_PATH}' fill='none' stroke='#1f3a2c' stroke-width='1.5' vector-effect='non-scaling-stroke'/>"
            f"<path class='pulse' pathLength='100' d='{_ECG_PATH}' fill='none' stroke='#4ade80' stroke-width='2' vector-effect='non-scaling-stroke'/></svg>")
 
def _dial_svg(p, color, cls="dial"):
    p = max(0.0, min(float(p), 100.0))
    arc = 157.08 * p / 100
    rot = -90 + p * 1.8
    return (f"<svg class='{cls}' viewBox='0 0 120 72'>"
            "<path d='M10 62 A50 50 0 0 1 110 62' fill='none' stroke='#17241d' stroke-width='9' stroke-linecap='round'/>"
            "<line x1='60' y1='6' x2='60' y2='12' stroke='#2a4136' stroke-width='2'/>"
            f"<path class='dial-arc' d='M10 62 A50 50 0 0 1 110 62' fill='none' stroke='{color}' stroke-width='9' stroke-linecap='round' stroke-dasharray='{arc:.2f} 200'/>"
            f"<g class='needle' style='--rot:{rot:.1f}deg'><line x1='60' y1='62' x2='60' y2='27' stroke='#e6f2ea' stroke-width='2' stroke-linecap='round'/></g>"
            f"<circle cx='60' cy='62' r='5' fill='#0c130f' stroke='{color}' stroke-width='2'/></svg>")
 
def _vu(p, n=20, cls=""):
    lit = int(round(max(0.0, min(float(p), 100.0)) / 100 * n))
    # เปลี่ยนจาก <i> เป็น <div class='v-bar'> ป้องกัน Streamlit ลบทิ้ง
    cells = "".join(f"<div class='v-bar {'on' if i < lit else ''}' style='--i:{i}'></div>" for i in range(n))
    return f"<div class='vu {cls}' style='--n:{n}'>{cells}</div>"
 
def _channel_html(ch, name, line_txt, prob, verdict, color, desc, lo_lbl, hi_lbl):
    p = max(0.0, min(float(prob), 100.0))
    return (f"<div class='ch' style='--c:{color}'>"
            f"<div class='ch-h'><span class='ch-k'>{ch}</span><span>{name}</span><span class='ch-l'>{esc(line_txt)}</span></div>"
            "<div class='ch-body'>"
            f"{_dial_svg(p, color)}"
            f"<div class='ch-info'><div class='ch-val'>{p:g}<small>%</small></div><div class='ch-d'>{desc}</div><div class='ch-v'>{esc(verdict)}</div></div>"
            "</div>"
            f"{_vu(p, 20)}<div class='vu-l'><span>{lo_lbl}</span><span>50%</span><span>{hi_lbl}</span></div></div>")
 
def _match_card_html(n, home, away, hdp_line, ou_line, hdp_prob, ou_prob, hdp_verdict, ou_verdict, solo=False):
    t1, t2 = _pred_tone(hdp_verdict, "hdp"), _pred_tone(ou_verdict, "ou")
    strength = max(_TONE_STRENGTH[t1], _TONE_STRENGTH[t2])
    # เปลี่ยนจาก <i> เป็น <div class='s-bar'> 
    bars = "".join(f"<div class='s-bar {'on' if k < strength else ''}'></div>" for k in range(4))
    status = "<span class='mon-st on'>มีสัญญาณ</span>" if strength > 0 else "<span class='mon-st'>ข้าม</span>"
    tag = "บันทึกแล้ว" if solo else "รอผลสกอร์"
    return (f"<div class='mon{' solo' if solo else ''}'><span class='cn tl'></span><span class='cn tr'></span><span class='cn bl'></span><span class='cn br'></span>"
            f"<div class='mon-bar'><span class='led'></span><span class='mon-id'>MATCH #{n:02d}</span>{status}<span>{tag}</span><span class='sig'>{bars}</span></div>"
            f"<div class='mon-teams'><span class='mon-t'>{esc(home)}</span>{_ecg_svg()}<span class='mon-t r'>{esc(away)}</span></div>"
            "<div class='mon-ch'>"
            + _channel_html("CH1", "HANDICAP", f"เหย้า {_fmt_num(hdp_line, True)}", hdp_prob, hdp_verdict,
                            _TONE_COLORS["hdp"][t1], "โอกาสคว้ากำไรแฮนดิแคป", "เยือน", "เหย้า")
            + _channel_html("CH2", "OVER / UNDER", f"ไลน์ {_fmt_num(ou_line)}", ou_prob, ou_verdict,
                            _TONE_COLORS["ou"][t2], "โอกาสคว้ากำไรสูงต่ำ", "ต่ำ", "สูง")
            + "</div></div>")
 
def _card_from_row(n, r, solo=False):
    return _match_card_html(n, str(r.get('ทีมเหย้า', '-')), str(r.get('ทีมเยือน', '-')),
                            r.get('แฮนดิแคป (เหย้า)', 0), r.get('โกลสูง/ต่ำ', 0),
                            float(r.get('AI_โอกาสเหย้าชนะHDP(%)', 50)), float(r.get('AI_โอกาสออกสูง(%)', 50)),
                            str(r.get('AI_ฟันธง_HDP', '-')), str(r.get('AI_ฟันธง_สูงต่ำ', '-')), solo)
 
def render_monitor_header(df_pred):
    n_hdp = sum(1 for v in df_pred['AI_ฟันธง_HDP'] if _pred_tone(v, "hdp") != "skip")
    n_ou = sum(1 for v in df_pred['AI_ฟันธง_สูงต่ำ'] if _pred_tone(v, "ou") != "skip")
    st.markdown(f"<div class='mon-status'><span class='led'></span><b>MONITORING</b>"
                f"<span>แมตช์ที่รอทำนาย<em>{len(df_pred)}</em></span>"
                f"<span>สัญญาณแฮนดิแคป<em>{n_hdp}</em></span>"
                f"<span>สัญญาณสูง-ต่ำ<em>{n_ou}</em></span></div>", unsafe_allow_html=True)
    items = []
    for _, r in df_pred.iterrows():
        v1, v2 = str(r.get('AI_ฟันธง_HDP', '-')), str(r.get('AI_ฟันธง_สูงต่ำ', '-'))
        c1 = _TONE_COLORS["hdp"][_pred_tone(v1, "hdp")]
        c2 = _TONE_COLORS["ou"][_pred_tone(v2, "ou")]
        items.append(f"<span class='tk'><b>{esc(str(r.get('ทีมเหย้า', '-')))} vs {esc(str(r.get('ทีมเยือน', '-')))}</b>"
                     f"<span class='d' style='--c:{c1}'></span>HDP {float(r.get('AI_โอกาสเหย้าชนะHDP(%)', 50)):g}%"
                     f"<span class='d' style='--c:{c2}'></span>O/U {float(r.get('AI_โอกาสออกสูง(%)', 50)):g}%</span>")
    if items:
        track = "".join(items) * 2
        st.markdown(f"<div class='ticker-container'><div class='ticker-track' style='--dur:{max(24, len(items) * 7)}s'>{track}</div></div>", unsafe_allow_html=True)
 
def _pair(h, a): return (str(h).strip(), str(a).strip())
def _visible_matches(df_pred):
    hidden = st.session_state.setdefault("hidden_matches", set())
    pairs = [_pair(h, a) for h, a in zip(df_pred.get('ทีมเหย้า', []), df_pred.get('ทีมเยือน', []))]
    hidden.intersection_update(pairs)
    if len(pairs) != len(df_pred): return df_pred
    return df_pred[[p not in hidden for p in pairs]]
 
def _stretch():
    """ความกว้างเต็มคอลัมน์ ให้ใช้ได้ทั้ง Streamlit เก่า (use_container_width) และใหม่ (width='stretch')."""
    try:
        major, minor = (int(x) for x in st.__version__.split('.')[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
 
 
def render_prediction_cards(df_pred, workbook, offset=0):
    """Tab 2: ผลทำนายและช่องบันทึกสกอร์อยู่ในการ์ดใบเดียวกัน (ไม่แตะ Tab 1)."""
    for n, (_, row) in enumerate(df_pred.iterrows(), start=offset + 1):
        sheet_row = int(row['_SheetRow'])
        snapshot = str(row['_Snapshot'])
        home, away = str(row.get('ทีมเหย้า', '')), str(row.get('ทีมเยือน', ''))
        token = f'{sheet_row}_{snapshot[:16]}'
        with st.container(key=f'matchcard_{token}'):
            _t2_render_card(n, row)
            st.markdown(f"<div class='mc-sec'><span class='led'></span><span>บันทึกผลสกอร์</span>"
                        f"<span class='sub'>DATA แถว {sheet_row} · ตรวจชื่อทีมก่อนลงผล</span></div>",
                        unsafe_allow_html=True)
            with st.form(key=f't2_score_form_{token}'):
                sc1, sc2 = st.columns([3, 2])
                with sc1:
                    score_input = st.text_input('กรอกผลสกอร์ เหย้า-เยือน (เช่น 2-1)', placeholder='0-0', key=f't2_score_input_{token}')
                    final_confirmed = st.checkbox('ยืนยันว่าแข่งขันจบแล้วและตรวจสกอร์ถูกต้อง', key=f't2_final_{token}')
                with sc2:
                    submitted = st.form_submit_button('บันทึกผลลงชีต', **_stretch())
            if submitted:
                if not final_confirmed:
                    st.warning('กรุณายืนยันว่าแข่งขันจบแล้วก่อนบันทึกผล')
                    continue
                score = _t2_parse_score(score_input)
                if score is None:
                    st.warning('กรอกผลสกอร์เป็น เหย้า-เยือน เช่น 2-1 หรือ 0-0')
                    continue
                saved = False
                try:
                    saved = update_match_score_in_sheet(workbook, home, away, score_input,
                        sheet_row=sheet_row, expected_snapshot=snapshot)
                except Exception as exc:
                    _t2_error(exc, f'บันทึกผลสกอร์ที่แถว {sheet_row}')
                if saved:
                    _t2_finish_saved(sheet_row, f'{score[0]}-{score[1]}')
                    # Rerun is deliberately outside the save try/except.
                    st.rerun()
                elif saved is False:
                    st.caption('หากยังไม่ยืนยันว่าบันทึกสำเร็จ ให้ตรวจข้อความด้านบนก่อนส่งคำสั่งใหม่')
 
# ----------------------------------------
# Dashboard : KPI & Charts
# ----------------------------------------
def _kpi_html(hdp_acc, ou_acc, total_train, hdp_n=None, ou_n=None):
    def finite(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return None
 
    def tile(cls, label, value, n):
        number = finite(value)
        shown = "–" if number is None else f"{number:g}"
        unit = "" if number is None else "<span class='kpi-unit'>%</span>"
        width = 0 if number is None else min(max(number, 0), 100)
        sub = f"<div class='kpi-sub'>ตัวอย่างทดสอบ n = {int(n)}</div>" if finite(n) is not None else (
            "<div class='kpi-sub'>ยังไม่มีผลทดสอบ</div>" if number is None else "")
        return (f"<div class='kpi {cls}'><div class='kpi-label'><span class='led'></span>{label}</div>"
                f"<div class='kpi-value'>{shown}{unit}</div>"
                f"<div class='kpi-bar'><div style='width:{width}%'></div></div>{sub}</div>")
 
    return ("<div class='kpi-wrap'>"
            + tile("hero", "แฮนดิแคป · ทายถูกฝั่งตามเกณฑ์เดิม", hdp_acc, hdp_n)
            + tile("hero-cyan", "สูง/ต่ำ · ทายถูกฝั่งตามเกณฑ์เดิม", ou_acc, ou_n)
            + f"<div class='kpi'><div class='kpi-label'><span class='led'></span>รายการประวัติที่โมเดลได้รับ</div>"
              f"<div class='kpi-value'>{total_train}<span class='kpi-unit'>คู่</span></div></div></div>")
 
# =============================================================================
# CYBER MONITOR DASHBOARD (v2)
# แทนที่ฟังก์ชัน render_dashboard(ai) เดิมทั้งฟังก์ชัน ด้วยโค้ดทั้งก้อนนี้
# (ตำแหน่งเดิม: ต่อจาก _kpi_html และก่อนบล็อก "PATCH 5.1.4")
# ใช้ของเดิม: _kpi_html, esc, np, pd, go, datetime/timezone/timedelta
# =============================================================================
from plotly.subplots import make_subplots
 
NEON_CYAN, NEON_MAG, NEON_GREEN = "#00e5ff", "#ff2bd6", "#39ff88"
NEON_YEL, NEON_RED, NEON_GREY = "#fcee0a", "#ff3b5c", "#6b7f74"
_CY_MONO = "JetBrains Mono, Prompt, monospace"
_CY_GRID = "rgba(0,229,255,0.09)"
_CY_BG = "#050907"
CY_PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
 
# 4 = ได้เต็ม ... 0 = เสียเต็ม (ตรงกับ Target_HDP / Target_OU เดิม)
_OUT_LABELS = {4: "ได้เต็ม", 3: "ได้ครึ่ง", 2: "คืนทุน", 1: "เสียครึ่ง", 0: "เสียเต็ม"}
_OUT_COLORS = {4: NEON_GREEN, 3: NEON_CYAN, 2: NEON_GREY, 1: NEON_YEL, 0: NEON_RED}
_OUT_ORDER = [4, 3, 2, 1, 0]
 
 
def _cy_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
 
 
def _cy_layout(fig, height=320, **extra):
    """ธีม Plotly กลาง: โปร่งใส, กริดบาง, ฟอนต์โมโน, ปิดการลาก/ซูม (กันจอมือถือเลื่อนติด)"""
    base = dict(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_CY_MONO, color="#a7c4b5", size=11),
        margin=dict(t=8, b=8, l=6, r=8), dragmode=False, hovermode="x unified",
        hoverlabel=dict(bgcolor=_CY_BG, bordercolor=NEON_CYAN,
                        font=dict(family=_CY_MONO, color="#e6f2ea", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    base.update(extra)
    fig.update_layout(**base)
    fig.update_xaxes(showgrid=True, gridcolor=_CY_GRID, gridwidth=1, zeroline=False,
                     linecolor="#1b2a22", tickfont=dict(size=10), fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor=_CY_GRID, gridwidth=1, zeroline=True,
                     zerolinecolor="rgba(255,43,214,0.35)", zerolinewidth=1,
                     linecolor="#1b2a22", tickfont=dict(size=10), fixedrange=True)
    return fig
 
 
def _cy_units(series):
    """Target 0..4 -> -1, -0.5, 0, +0.5, +1 ; ค่าว่างนับเป็น 0 (เหมือนกราฟเดิม)"""
    if series is None:
        return np.zeros(0)
    t = pd.to_numeric(series, errors="coerce")
    t = t.where(t.between(0, 4))
    return ((t - 2) / 2).fillna(0.0).to_numpy(dtype=float)
 
 
def _cy_counts(series):
    if series is None:
        return {k: 0 for k in _OUT_ORDER}
    t = pd.to_numeric(series, errors="coerce").dropna()
    t = t[t.isin(list(_OUT_ORDER))].astype(int)
    return {k: int((t == k).sum()) for k in _OUT_ORDER}
 
 
def _cy_drawdown(cum):
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((cum - peak).min())
 
 
def _cy_glow(fig, x, y, color, name, row, fill=False):
    for width, alpha in ((13, 0.05), (6, 0.13)):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=_cy_rgba(color, alpha), width=width)), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2.2),
        fill="tozeroy" if fill else None, fillcolor=_cy_rgba(color, 0.06) if fill else None,
        hovertemplate="%{y:+.1f}<extra>" + name + "</extra>"), row=row, col=1)
 
 
def _cy_endpoint(fig, x, y, color, row):
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=9, color=color, line=dict(color=_CY_BG, width=2))), row=row, col=1)
    fig.add_annotation(x=x, y=y, text=f"{y:+.1f}", showarrow=False, xanchor="left", xshift=9,
                       font=dict(family=_CY_MONO, size=11, color=color), row=row, col=1)
 
 
def build_trend_figure(df_hist):
    """บน: ดัชนีสะสม HDP/OU (เส้นนีออน) · ล่าง: โมเมนตัมเฉลี่ย 10 แถวล่าสุด"""
    hdp = _cy_units(df_hist["Target_HDP"] if "Target_HDP" in df_hist else None)
    ou = _cy_units(df_hist["Target_OU"] if "Target_OU" in df_hist else None)
    n = max(len(hdp), len(ou))
    if n == 0:
        return None, {}
    hdp = hdp if len(hdp) == n else np.zeros(n)
    ou = ou if len(ou) == n else np.zeros(n)
    x = np.arange(1, n + 1)
    c_hdp, c_ou = np.cumsum(hdp), np.cumsum(ou)
    m_hdp = pd.Series(hdp).rolling(10, min_periods=1).mean().to_numpy()
    m_ou = pd.Series(ou).rolling(10, min_periods=1).mean().to_numpy()
 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07)
    _cy_glow(fig, x, c_hdp, NEON_CYAN, "แฮนดิแคป", 1, fill=True)
    _cy_glow(fig, x, c_ou, NEON_MAG, "สูง/ต่ำ", 1, fill=True)
    _cy_endpoint(fig, n, float(c_hdp[-1]), NEON_CYAN, 1)
    _cy_endpoint(fig, n, float(c_ou[-1]), NEON_MAG, 1)
    _cy_glow(fig, x, m_hdp, NEON_CYAN, "โมเมนตัม HDP", 2)
    _cy_glow(fig, x, m_ou, NEON_MAG, "โมเมนตัม O/U", 2)
    _cy_layout(fig, height=430, margin=dict(t=30, b=8, l=6, r=46))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_yaxes(range=[-1.05, 1.05], row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    stats = {"n": n, "hdp_last": float(c_hdp[-1]), "ou_last": float(c_ou[-1]),
             "hdp_dd": _cy_drawdown(c_hdp), "ou_dd": _cy_drawdown(c_ou)}
    return fig, stats
 
 
def build_donut_figure(counts):
    total = sum(counts.values())
    if total == 0:
        return None
    fig = go.Figure(go.Pie(
        labels=[_OUT_LABELS[k] for k in _OUT_ORDER], values=[counts[k] for k in _OUT_ORDER],
        hole=0.66, sort=False, direction="clockwise", rotation=90, textinfo="none",
        marker=dict(colors=[_OUT_COLORS[k] for k in _OUT_ORDER], line=dict(color=_CY_BG, width=3)),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    _cy_layout(fig, height=300, margin=dict(t=6, b=6, l=6, r=6), hovermode="closest",
               legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.add_annotation(text=f"<b>{total}</b>", x=0.5, y=0.54, showarrow=False,
                       font=dict(family=_CY_MONO, size=30, color=NEON_CYAN))
    fig.add_annotation(text="ผลรวม 2 ตลาด", x=0.5, y=0.40, showarrow=False,
                       font=dict(family="Prompt, sans-serif", size=11, color="#7f9a8b"))
    return fig
 
 
def build_stack_figure(hdp_counts, ou_counts):
    fig = go.Figure()
    for k in _OUT_ORDER:
        vals = [hdp_counts[k], ou_counts[k]]
        fig.add_trace(go.Bar(
            y=["HDP", "O/U"], x=vals, orientation="h", name=_OUT_LABELS[k],
            marker=dict(color=_OUT_COLORS[k], line=dict(color=_CY_BG, width=2)),
            text=[str(v) if v else "" for v in vals], textposition="inside", insidetextanchor="middle",
            textfont=dict(family=_CY_MONO, size=11, color="#03140a"),
            hovertemplate="%{y} · " + _OUT_LABELS[k] + ": %{x}<extra></extra>"))
    _cy_layout(fig, height=300, margin=dict(t=6, b=6, l=6, r=8), hovermode="closest",
               barmode="stack", barnorm="percent", bargap=0.35,
               legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed", tickfont=dict(size=12, color="#e6f2ea"))
    return fig
 
 
# ----------------------------- HTML blocks -----------------------------------
def _cy_title(text, sub=""):
    return (f"<div class='hud-title'><span class='led'></span><span>{esc(text)}</span>"
            f"<span class='sub'>{esc(sub)}</span></div>")
 
 
def _cy_strip(title, series, n=40):
    t = pd.to_numeric(series, errors="coerce").dropna() if series is not None else pd.Series(dtype=float)
    vals = [int(v) for v in t[t.isin(list(_OUT_ORDER))].tolist()][-n:]
    if not vals:
        return ""
    cells = "".join(
        f"<div class='fs-cell{' last' if i == len(vals) - 1 else ''}' style='--c:{_OUT_COLORS[v]}' "
        f"title='{_OUT_LABELS[v]}'></div>" for i, v in enumerate(vals))
    return (f"<div class='fs-wrap'><div class='fs-h'><span>{esc(title)}</span>"
            f"<span>ล่าสุด <b>{_OUT_LABELS[vals[-1]]}</b> · {len(vals)} แถว</span></div>"
            f"<div class='fs-grid'>{cells}</div></div>")
 
 
def _cy_legend():
    return ("<div class='fs-legend'>" + "".join(
        f"<span style='--c:{_OUT_COLORS[k]}'>{_OUT_LABELS[k]}</span>" for k in _OUT_ORDER) + "</div>")
 
 
def _cy_fmt(value, digits=3):
    try:
        f = float(value)
        return f"{f:.{digits}f}" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"
 
 
def _cy_model_html(reports):
    tiles = []
    for key, name in (("HDP", "HANDICAP"), ("OU", "OVER / UNDER")):
        r = (reports or {}).get(key) or {}
        if r.get("status") != "ok":
            color, tag, note = NEON_RED, "OFFLINE", str(r.get("msg") or "โมเดลตลาดนี้ยังไม่พร้อม")
        elif r.get("gate_pass"):
            color, tag, note = NEON_GREEN, "PASS", "ผ่านเกณฑ์คุณภาพ — ยังเป็นผลทดลองแบบแบ่งกลุ่ม ไม่ใช่การทดสอบตามเวลา"
        else:
            color, tag = NEON_YEL, "HOLD"
            note = "; ".join(r.get("gate_reasons") or []) or "ยังไม่ผ่านเกณฑ์คุณภาพ"
        val = r.get("validation") or {}
        rows = sum((r.get("split_counts") or {}).values())
        tiles.append(
            f"<div class='mdl' style='--c:{color}'><div class='mdl-h'><span class='led'></span>"
            f"<span>{name}</span><span class='mdl-s'>{tag}</span></div>"
            f"<div class='mdl-r'><span>โมเดลที่เลือก</span><b>{esc(str(r.get('selected_model', '-')))}</b></div>"
            f"<div class='mdl-r'><span>Log loss (ชุดตรวจสอบ)</span><b>{_cy_fmt(val.get('log_loss'))}</b></div>"
            f"<div class='mdl-r'><span>ECE (ความเพี้ยนของความน่าจะเป็น)</span><b>{_cy_fmt(val.get('ece'))}</b></div>"
            f"<div class='mdl-r'><span>แถวที่ใช้ฝึก/ทดสอบ</span><b>{rows}</b></div>"
            f"<div class='mdl-n'>{esc(note)}</div></div>")
    return f"<div class='mdl-wrap'>{''.join(tiles)}</div>"
 
 
def _cy_plot(fig, key=None):
    st.plotly_chart(fig, theme=None, config=CY_PLOT_CONFIG, key=f"cy_{key}" if key else None, **_stretch())
 
 
# ----------------------------- generic helpers (v2) --------------------------
def _cy_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
 
 
def _cy_chip_row(items):
    """items = [(label, value, color), ...]"""
    cells = "".join(
        f"<div class='chip' style='--c:{c}'><div class='chip-k'>{esc(str(k))}</div>"
        f"<div class='chip-v'>{esc(str(v))}</div></div>" for k, v, c in items)
    return f"<div class='chip-row'>{cells}</div>"
 
 
def _cy_table(headers, rows, selected=()):
    head = "".join(f"<th>{esc(str(h))}</th>" for h in headers)
    lines = []
    for i, row in enumerate(rows):
        cls = " class='sel'" if i in selected else ""
        cells = "".join(f"<td>{esc(str(c))}</td>" for c in row)
        lines.append(f"<tr{cls}>{cells}</tr>")
    return (f"<div class='cy-scroll'><table class='cy-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(lines)}</tbody></table></div>")
 
 
def _cy_empty(text):
    return f"<div class='empty-state'>{esc(text)}</div>"
 
 
_CY_MARKETS = (("HDP", "HDP", NEON_CYAN), ("OU", "O/U", NEON_MAG))
 
 
# ----------------------------- extra figures ---------------------------------
def build_compare_figure(reports):
    """โมเดลที่เลือก vs วิธีพื้นฐาน บนชุดทดสอบ (ยิ่งต่ำยิ่งดี)"""
    metrics = (("log_loss", "Log loss"), ("brier", "Brier"), ("ece", "ECE"))
    fig, added = go.Figure(), 0
    for key, label, color in _CY_MARKETS:
        r = (reports or {}).get(key) or {}
        test, base = r.get("test"), r.get("baseline_test")
        if r.get("status") != "ok" or not test or not base:
            continue
        for title, source, shade in ((f"{label} โมเดล", test, color), (f"{label} พื้นฐาน", base, _cy_rgba(color, 0.35))):
            fig.add_trace(go.Bar(
                x=[m[1] for m in metrics], y=[_cy_float(source.get(m[0])) for m in metrics], name=title,
                marker=dict(color=shade, line=dict(color=_CY_BG, width=1)),
                hovertemplate="%{x}: %{y:.4f}<extra>" + title + "</extra>"))
            added += 1
    if not added:
        return None
    _cy_layout(fig, height=320, margin=dict(t=6, b=8, l=6, r=8), barmode="group", bargap=0.28,
               legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_yaxes(rangemode="tozero")
    return fig
 
 
def build_coverage_figure(reports):
    """จำนวนตัวอย่างฝึกแยกตามเศษของไลน์ (.00/.25/.50/.75)"""
    labels = [".00", ".25", ".50", ".75"]
    fig, added = go.Figure(), 0
    for key, label, color in _CY_MARKETS:
        counts = ((reports or {}).get(key) or {}).get("family_counts") or {}
        if not counts:
            continue
        values = [int(counts.get(i, counts.get(str(i), 0))) for i in range(4)]
        fig.add_trace(go.Bar(x=labels, y=values, name=label,
                             marker=dict(color=color, line=dict(color=_CY_BG, width=1)),
                             hovertemplate="ไลน์ %{x}: %{y} แถว<extra>" + label + "</extra>"))
        added += 1
    if not added:
        return None
    minimum = ML_CONFIG["min_line_examples"]
    _cy_layout(fig, height=280, margin=dict(t=18, b=8, l=6, r=8), barmode="group", bargap=0.3,
               legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.add_hline(y=minimum, line=dict(color=NEON_YEL, width=1, dash="dot"), annotation_text=f"ขั้นต่ำ {minimum}",
                  annotation_position="top left", annotation_font=dict(size=10, color=NEON_YEL))
    return fig
 
 
# ----------------------------- HTML blocks (v2) ------------------------------
def _cy_chips(stats):
    return _cy_chip_row([("ดัชนีสะสมล่าสุด · แฮนดิแคป", f"{stats['hdp_last']:+.1f}", NEON_CYAN),
                         ("ดัชนีสะสมล่าสุด · สูง/ต่ำ", f"{stats['ou_last']:+.1f}", NEON_MAG),
                         ("ย่อลึกสุด · แฮนดิแคป", f"{stats['hdp_dd']:+.1f}", NEON_YEL),
                         ("ย่อลึกสุด · สูง/ต่ำ", f"{stats['ou_dd']:+.1f}", NEON_YEL)])
 
 
def _cy_candidates_html(reports):
    rows, selected = [], []
    for key, label, _ in _CY_MARKETS:
        r = (reports or {}).get(key) or {}
        for name, info in (r.get("candidate_validation") or {}).items():
            val = (info or {}).get("validation") or {}
            chosen = name == r.get("selected_model")
            if chosen:
                selected.append(len(rows))
            rows.append([label, ("★ " if chosen else "") + str(name), _cy_fmt(info.get("temperature"), 2),
                         _cy_fmt(val.get("log_loss")), _cy_fmt(val.get("brier")), _cy_fmt(val.get("ece"))])
    if not rows:
        return _cy_empty("ยังไม่มีข้อมูลตัวเลือกโมเดล")
    return _cy_table(["ตลาด", "ตัวเลือกโมเดล", "Temp", "Log loss", "Brier", "ECE"], rows, selected)
 
 
def _cy_policy_html(reports):
    tiles = []
    for key, label, _ in _CY_MARKETS:
        r = (reports or {}).get(key) or {}
        policy = r.get("policy_test") if r.get("status") == "ok" else None
        if not policy:
            tiles.append(f"<div class='mdl' style='--c:{NEON_GREY}'><div class='mdl-h'><span class='led'></span>"
                         f"<span>{label}</span><span class='mdl-s'>N/A</span></div>"
                         f"<div class='mdl-n'>ยังไม่มีผลจำลองของตลาดนี้</div></div>")
            continue
        signals = int(policy.get("research_signals") or 0)
        total = int(policy.get("eligible_test_rows") or 0)
        rate = _cy_float(policy.get("positive_rate_excluding_push"))
        unit_sum = _cy_float(policy.get("unit_return_sum")) or 0.0
        unit_mean = _cy_float(policy.get("unit_return_mean"))
        if not signals:
            color, tag = NEON_GREY, "NO SIGNAL"
            note = ("ไม่ผ่านเกณฑ์คุณภาพของโมเดล จึงไม่มีสัญญาณในชุดทดสอบ" if not r.get("gate_pass")
                    else "ไม่มีรายการที่ผ่านเกณฑ์ในชุดทดสอบ")
        elif unit_sum > 0:
            color, tag, note = NEON_GREEN, "POSITIVE", "จำลองบนชุดทดสอบแบบแบ่งกลุ่ม ไม่ใช่ผลจริงตามเวลา และยังไม่หักค่าธรรมเนียม"
        else:
            color, tag, note = NEON_RED, "NEGATIVE", "จำลองบนชุดทดสอบแบบแบ่งกลุ่ม ไม่ใช่ผลจริงตามเวลา และยังไม่หักค่าธรรมเนียม"
        tiles.append(
            f"<div class='mdl' style='--c:{color}'><div class='mdl-h'><span class='led'></span>"
            f"<span>{label}</span><span class='mdl-s'>{tag}</span></div>"
            f"<div class='mdl-r'><span>สัญญาณ / แถวทดสอบ</span><b>{signals} / {total}</b></div>"
            f"<div class='mdl-r'><span>แพ้ชนะชัดเจน · คืนทุน</span><b>{int(policy.get('decisive_n') or 0)} · {int(policy.get('push_n') or 0)}</b></div>"
            f"<div class='mdl-r'><span>อัตราได้กำไร (ไม่นับคืนทุน)</span><b>{'-' if rate is None else f'{rate:.1f}%'}</b></div>"
            f"<div class='mdl-r'><span>ผลรวม · เฉลี่ยต่อไม้ (หน่วย)</span><b>{unit_sum:+.2f} · {'-' if unit_mean is None else f'{unit_mean:+.3f}'}</b></div>"
            f"<div class='mdl-n'>{esc(note)}</div></div>")
    return f"<div class='mdl-wrap'>{''.join(tiles)}</div>"
 
 
def _cy_audit_html(reports):
    rows = []
    for key, label, _ in _CY_MARKETS:
        r = (reports or {}).get(key) or {}
        audit, split = r.get("data_audit") or {}, r.get("split_counts") or {}
        rows.append([label, audit.get("eligible_rows", "-"), audit.get("unique_training_rows", "-"),
                     audit.get("conflicting_input_groups", "-"), audit.get("excluded_or_collapsed", "-"),
                     split.get("train", "-"), split.get("calibration", "-"), split.get("validation", "-"),
                     split.get("test", "-"), r.get("train_form_complete", "-")])
    return _cy_table(["ตลาด", "ใช้ได้", "ไม่ซ้ำ", "ขัดแย้ง", "ตัด/รวม", "Train", "Cal", "Val", "Test", "ฟอร์มครบ"], rows)
 
 
# ----------------------------- dashboard sections ----------------------------
def _cy_history_section(ai):
    df_hist = ai.get("df_history", pd.DataFrame())
    has_hist = (isinstance(df_hist, pd.DataFrame) and len(df_hist) > 0
                and ("Target_HDP" in df_hist.columns or "Target_OU" in df_hist.columns))
    if not has_hist:
        st.markdown(_cy_empty("ยังไม่มีข้อมูลที่บันทึกผลสกอร์เพียงพอสำหรับสร้างกราฟ — ลงสกอร์ในแท็บสัญญาณเพื่อเริ่มสะสมข้อมูล"),
                    unsafe_allow_html=True)
        return
    hdp_series = df_hist["Target_HDP"] if "Target_HDP" in df_hist else None
    ou_series = df_hist["Target_OU"] if "Target_OU" in df_hist else None
 
    fig, stats = build_trend_figure(df_hist)
    if fig is not None:
        with st.container(key="hud_trend"):
            st.markdown(_cy_title("MARKET TREND", f"{stats['n']} แถวตามลำดับในชีต"), unsafe_allow_html=True)
            st.markdown(_cy_chips(stats), unsafe_allow_html=True)
            _cy_plot(fig, "trend")
            st.markdown("<div class='hud-note'><b>ส่วนบน</b> ดัชนีสะสม: เพิ่ม = เจ้าบ้าน/สูงได้ · ลด = เสีย "
                        "&nbsp;|&nbsp; <b>ส่วนล่าง</b> โมเมนตัมเฉลี่ย 10 แถวล่าสุด "
                        "&nbsp;|&nbsp; คะแนน −1…+1 ไม่รวมค่าน้ำ ไม่ใช่กำไรจริง และไม่ใช่ลำดับเวลาแข่งขัน</div>",
                        unsafe_allow_html=True)
 
    hdp_c, ou_c = _cy_counts(hdp_series), _cy_counts(ou_series)
    total_c = {k: hdp_c[k] + ou_c[k] for k in _OUT_ORDER}
    if sum(total_c.values()) > 0:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            with st.container(key="hud_donut"):
                st.markdown(_cy_title("RESULT MIX", "ฝั่งเจ้าบ้านและสูง"), unsafe_allow_html=True)
                _cy_plot(build_donut_figure(total_c), "donut")
        with col_b:
            with st.container(key="hud_stack"):
                st.markdown(_cy_title("MARKET SPLIT", "สัดส่วนแยกตามตลาด"), unsafe_allow_html=True)
                _cy_plot(build_stack_figure(hdp_c, ou_c), "stack")
        st.caption("สัดส่วนผลย้อนหลังฝั่งเจ้าบ้านและสูง — ไม่ใช่ผลทำนายของ AI")
 
    strip = _cy_strip("HANDICAP · 40 ผลล่าสุด", hdp_series) + _cy_strip("OVER / UNDER · 40 ผลล่าสุด", ou_series)
    if strip:
        with st.container(key="hud_strip"):
            st.markdown(_cy_title("FORM STRIP", "ซ้าย = เก่า · ขวา = ล่าสุด"), unsafe_allow_html=True)
            st.markdown(strip + _cy_legend(), unsafe_allow_html=True)
 
 
def _cy_model_section(reports):
    with st.container(key="hud_model"):
        st.markdown(_cy_title("MODEL STATUS", "ผลจากชุดตรวจสอบ ไม่ใช่ผลอนาคต"), unsafe_allow_html=True)
        st.markdown(_cy_model_html(reports), unsafe_allow_html=True)
 
    with st.container(key="hud_compare"):
        st.markdown(_cy_title("MODEL vs BASELINE", "ชุดทดสอบ · ยิ่งต่ำยิ่งดี"), unsafe_allow_html=True)
        fig = build_compare_figure(reports)
        if fig is None:
            st.markdown(_cy_empty("ยังไม่มีผลเปรียบเทียบ — โมเดลของตลาดยังไม่พร้อมหรือข้อมูลยังน้อยเกินไป"), unsafe_allow_html=True)
        else:
            _cy_plot(fig, "compare")
        st.markdown("<div class='hud-note'><b>ตัวเลือกโมเดลในชุดตรวจสอบ</b> · ★ = ตัวที่ระบบเลือก "
                    "(เลือกจากชุดตรวจสอบเท่านั้น ชุดทดสอบไม่ถูกใช้ตัดสิน)</div>", unsafe_allow_html=True)
        st.markdown(_cy_candidates_html(reports), unsafe_allow_html=True)
 
    with st.container(key="hud_policy"):
        st.markdown(_cy_title("SIGNAL SIMULATION", "จำลองสัญญาณบนชุดทดสอบ"), unsafe_allow_html=True)
        st.markdown(_cy_policy_html(reports), unsafe_allow_html=True)
 
 
def _cy_data_section(ai, reports):
    quality = ai.get("data_quality") or {}
    invalid = int(quality.get("invalid_scores") or 0)
    with st.container(key="hud_dataquality"):
        st.markdown(_cy_title("DATA QUALITY", "จากชีต DATA ที่โหลดล่าสุด"), unsafe_allow_html=True)
        st.markdown(_cy_chip_row([
            ("แถวที่มีชื่อทีม", quality.get("named_rows", "-"), NEON_CYAN),
            ("รอผลสกอร์", quality.get("pending", "-"), NEON_MAG),
            ("สกอร์ผิดรูปแบบ", invalid, NEON_RED if invalid else NEON_GREY),
            ("การ์ดที่ผลทำนายไม่ครบ 2 ตลาด", quality.get("cards_without_both_probabilities", "-"), NEON_YEL)]),
            unsafe_allow_html=True)
 
    with st.container(key="hud_audit"):
        st.markdown(_cy_title("TRAINING AUDIT", "ข้อมูลที่ใช้ฝึกและการแบ่งชุด"), unsafe_allow_html=True)
        st.markdown(_cy_audit_html(reports), unsafe_allow_html=True)
        errors = [f"{label}: {e.get('candidate')} ({e.get('type')})"
                  for key, label, _ in _CY_MARKETS for e in ((reports.get(key) or {}).get("model_errors") or [])]
        note = ("<b>ขัดแย้ง</b> = อินพุตเดียวกันแต่ผลต่างกัน (ถูกตัดออก) · <b>ตัด/รวม</b> = แถวที่ถูกยุบหรือตัด · "
                "<b>ฟอร์มครบ</b> = จำนวนแถวฝึกที่มีฟอร์ม 5 นัดครบ")
        if errors:
            note += "<br>โมเดลที่ฝึกไม่สำเร็จ: " + esc(", ".join(errors))
        st.markdown(f"<div class='hud-note'>{note}</div>", unsafe_allow_html=True)
 
    with st.container(key="hud_coverage"):
        st.markdown(_cy_title("LINE COVERAGE", "ตัวอย่างฝึกแยกตามเศษของไลน์"), unsafe_allow_html=True)
        fig = build_coverage_figure(reports)
        if fig is None:
            st.markdown(_cy_empty("ยังไม่มีข้อมูลความครอบคลุมของไลน์"), unsafe_allow_html=True)
        else:
            _cy_plot(fig, "coverage")
            ranges = [f"{label}: {r['line_range'][0]:+g} ถึง {r['line_range'][1]:+g}"
                      for key, label, _ in _CY_MARKETS
                      for r in [reports.get(key) or {}] if r.get("line_range")]
            st.markdown("<div class='hud-note'>ไลน์ที่มีตัวอย่างต่ำกว่าเส้นประจะถูกข้ามสัญญาณ · ช่วงไลน์ที่ใช้ฝึก → "
                        + esc(" · ".join(ranges)) + "</div>", unsafe_allow_html=True)
 
 
# ----------------------------- main renderer ---------------------------------
def render_dashboard(ai):
    if not isinstance(ai, dict) or ai.get("status") not in ("ok", "few"):
        st.warning("⚠️ ยังไม่มีข้อมูลสรุปสำหรับ Dashboard (หรือโหลดไม่สำเร็จ)")
        return
    reports = ai.get("reports") or {}
    if ai.get("status") == "few":
        st.warning("โมเดลทั้งสองตลาดยังไม่พร้อม: " + str(ai.get("msg") or "ข้อมูลยังไม่พอ") +
                   " — ส่วนข้อมูลและสถานะยังแสดงได้ตามปกติ")
 
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")
    st.markdown(f"<div class='mon-status'><span class='led'></span><b>สรุปข้อมูล</b>"
                f"<span>รายการประวัติ<em>{ai.get('total_train', 0)}</em>คู่</span>"
                f"<span>เอนจิน<em>{esc(str(ai.get('engine_version', '-')))}</em></span>"
                f"<span>อัปเดตล่าสุด<em>{stamp}</em></span></div>", unsafe_allow_html=True)
 
    def directional_n(key):
        return ((reports.get(key) or {}).get("test") or {}).get("directional_n")
    st.markdown(_kpi_html(ai.get("hdp_acc"), ai.get("ou_acc"), ai.get("total_train", 0),
                          directional_n("HDP"), directional_n("OU")), unsafe_allow_html=True)
    st.info("รายงานจากโมเดล ไม่ใช่ผลรับรองความแม่นยำในอนาคต · อัตราทายถูกฝั่งคิดเฉพาะตัวอย่างทดสอบที่ผ่านเกณฑ์ 55% "
            "และไม่คืนทุน · การแบ่งข้อมูลเป็นการสุ่มตามกลุ่มคู่ทีม ไม่ใช่แบ่งตามเวลา")
 
    tab_history, tab_model, tab_data = st.tabs(["📈 ผลย้อนหลัง", "🧪 คุณภาพโมเดล", "🗂️ ข้อมูลและการฝึก"])
    with tab_history:
        _cy_history_section(ai)
    with tab_model:
        _cy_model_section(reports)
    with tab_data:
        _cy_data_section(ai, reports)
 
 
# -----------------------------------------------------------------------------
# PATCH 5.1.4: Tab 2/3 only. Original Tab 1 and all of its dependencies are frozen.
# No monkey-patching of the upload workflow; no new sheet or metadata columns.
# -----------------------------------------------------------------------------
class ScoreConflictError(ValueError):
    """The displayed row no longer matches, or a different score already exists."""
 
 
class ScoreWriteUncertainError(RuntimeError):
    """A dispatched write cannot yet be confirmed. Do not resend automatically."""
 
 
@st.cache_resource
def _t2_score_journal():
    # Shared across sessions in this process; not a distributed/durable lock.
    # Access is always protected by the original JOBS.sheet_lock.
    return {}
 
 
def _t2_blank(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()
 
 
def _t2_parse_score(value):
    """Accept only a text score, not Excel dates, booleans, numbers or formulas."""
    if not isinstance(value, str):
        return None
    text = value.strip().replace('\u2212', '-').replace('\u2013', '-')
    match = re.fullmatch(r'([0-9]{1,2})\s*[-:]\s*([0-9]{1,2})', text)
    return tuple(map(int, match.groups())) if match else None
 
 
def _t2_canonical(value, *, name=False):
    if _t2_blank(value):
        return ''
    if name:
        return ' '.join(str(value).split())
    if isinstance(value, (bool, np.bool_)):
        return 'bool:' + str(bool(value))
    try:
        n = float(value)
        if np.isfinite(n):
            return format(n, '.15g')
    except (TypeError, ValueError, OverflowError):
        pass
    return str(value).strip()
 
 
def _t2_fingerprint(values):
    """Fingerprint A:L and X:AC; ignore score and legacy formula columns M:W."""
    import hashlib
    padded = list(values[:29]) + [''] * max(0, 29 - len(values))
    inputs = padded[:12] + padded[23:29]
    normalized = [_t2_canonical(v, name=i < 2) for i, v in enumerate(inputs)]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode('utf-8')).hexdigest()
 
 
def _t2_record_fingerprint(record):
    values = [record.get(c, '') for c in ['ทีมเหย้า', 'ทีมเยือน'] + ML_ODDS_COLUMNS]
    values += [''] * 11
    values += [record.get(c, '') for c in FORM5_HEADERS]
    return _t2_fingerprint(values)
 
 
def _t2_values(ws, cell_range):
    return ws.get(range_name=cell_range, value_render_option='UNFORMATTED_VALUE',
                  date_time_render_option='SERIAL_NUMBER')
 
 
def _t2_check_headers(values):
    required = ['ทีมเหย้า', 'ทีมเยือน'] + ML_ODDS_COLUMNS + ['ผลสกอร์']
    actual = [str(v).strip() for v in (values[0] if values else [])[:13]]
    if actual != required:
        raise ScoreConflictError('หัวคอลัมน์ A:M ของ DATA ไม่ตรงกับต้นฉบับ จึงหยุดก่อนเขียนข้อมูล กรุณาตรวจตำแหน่งคอลัมน์')
 
 
def _t2_data_from_grid(grid):
    _t2_check_headers(grid)
    columns = ['ทีมเหย้า', 'ทีมเยือน'] + ML_ODDS_COLUMNS + ['ผลสกอร์'] + FORM5_HEADERS
    columns += ['_SheetRow', '_Snapshot', '_ScoreStatus', '_Issue']
    records = []
    for sheet_row, values in enumerate(grid[1:], start=2):
        vals = list(values[:29]) + [''] * max(0, 29 - len(values))
        if _t2_blank(vals[0]) and _t2_blank(vals[1]):
            continue
        record = dict(zip(['ทีมเหย้า', 'ทีมเยือน'] + ML_ODDS_COLUMNS + ['ผลสกอร์'], vals[:13]))
        record.update(dict(zip(FORM5_HEADERS, vals[23:29])))
        names_ok = not _t2_blank(vals[0]) and not _t2_blank(vals[1])
        score = _t2_parse_score(vals[12])
        status = ('pending' if _t2_blank(vals[12]) else 'finished' if score is not None else 'invalid')
        issue = '' if status != 'invalid' else 'ผลสกอร์ไม่ใช่ข้อความรูปแบบ 2-1 (อาจเป็นวันที่หรือตัวเลข)'
        if not names_ok:
            status, issue = 'invalid', 'ชื่อทีมเหย้าหรือทีมเยือนไม่ครบ'
        record.update(_SheetRow=sheet_row, _Snapshot=_t2_fingerprint(vals),
                      _ScoreStatus=status, _Issue=issue)
        records.append(record)
    return pd.DataFrame(records, columns=columns)
 
 
@st.cache_data(ttl=20, show_spinner=False)
def _t2_load_data(_workbook, workbook_key, data_version=0):
    # Use DATA explicitly: a missing DATA tab must not redirect score writes.
    with JOBS.sheet_lock:
        ws = _workbook.worksheet('DATA')
        grid = _t2_values(ws, 'A1:AC')
    return _t2_data_from_grid(grid)
 
 
def _t2_workbook_key(wb):
    return str(getattr(wb, 'id', id(wb)))
 
 
def _t2_refresh():
    _t2_load_data.clear()
    # Preserve the original refresh semantics without replacing its functions.
    st.cache_data.clear()
 
 
def _t2_read_target(ws, row_number):
    rows = _t2_values(ws, f'A{row_number}:AC{row_number}')
    vals = list(rows[0]) if rows else []
    return vals[:29] + [''] * max(0, 29 - len(vals))
 
 
def _t2_target_matches(values, home, away, expected_snapshot):
    names = (_t2_canonical(values[0], name=True), _t2_canonical(values[1], name=True))
    return (names == (_t2_canonical(home, name=True), _t2_canonical(away, name=True))
            and (expected_snapshot is None or _t2_fingerprint(values) == expected_snapshot))
 
 
def _t2_write_key(wb, ws, row_number):
    return (_t2_workbook_key(wb), str(ws.id), int(row_number))
 
 
def _t2_confirm_pending(ws, entry):
    vals = _t2_read_target(ws, entry['row'])
    return (_t2_fingerprint(vals) == entry['snapshot']
            and _t2_parse_score(vals[12]) == _t2_parse_score(entry['score']))
 
 
def _t2_reconcile(wb):
    """Read only: never retries an uncertain write."""
    confirmed, unresolved = 0, 0
    with JOBS.sheet_lock:
        ws = wb.worksheet('DATA')
        _t2_check_headers(_t2_values(ws, 'A1:M1'))
        journal = _t2_score_journal()
        prefix = (_t2_workbook_key(wb), str(ws.id))
        for key, entry in list(journal.items()):
            if key[:2] != prefix:
                continue
            if _t2_confirm_pending(ws, entry):
                del journal[key]
                confirmed += 1
            else:
                unresolved += 1
    _t2_load_data.clear()
    return confirmed, unresolved
 
 
def _t2_error(exc, context):
    # No exception message/locals/full API response in support details.
    # User-authored validation exceptions contain only fixed, safe messages.
    import importlib.metadata
    code = uuid.uuid4().hex[:10]
    if isinstance(exc, (ScoreConflictError, ScoreWriteUncertainError)):
        message = str(exc)
    else:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        message = {403: 'ไม่มีสิทธิ์อ่านหรือแก้ไขชีต กรุณาตรวจสิทธิ์บัญชีเชื่อมต่อ',
                   404: 'ไม่พบชีตหรือทรัพยากรที่ระบุ',
                   429: 'เรียกบริการถี่เกินกำหนด กรุณาตรวจสถานะก่อนลองอีกครั้ง'}.get(
                      status, 'ดำเนินการไม่สำเร็จ กรุณาส่งรายละเอียดด้านล่างเพื่อตรวจสอบ')
    st.error(f'{message} (รหัสอ้างอิง {code})')
    versions = {}
    for package in ('streamlit', 'gspread'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'ไม่พบข้อมูลรุ่น'
    frames = [{'file': os.path.basename(frame.filename), 'line': frame.lineno,
               'function': frame.name} for frame in traceback.extract_tb(exc.__traceback__)]
    details = {'รหัสอ้างอิง': code, 'ขั้นตอน': context,
               'ชนิดข้อผิดพลาด': type(exc).__name__, 'ตำแหน่ง': frames[-6:],
               'รุ่นไลบรารี': versions, 'รุ่นปรับปรุงแท็บ2': '5.1.4-TAB1-LOCKED'}
    with st.expander('รายละเอียดสำหรับส่งตรวจ (ไม่รวมกุญแจ API หรือข้อมูลบัญชี)'):
        st.code(json.dumps(details, ensure_ascii=False, indent=2), language='json')
 
 
def _t2_pending_view(raw, ai):
    pending = raw.loc[raw['_ScoreStatus'].eq('pending')].copy()
    prediction_columns = ['AI_โอกาสเหย้าชนะHDP(%)', 'AI_โอกาสออกสูง(%)',
                          'AI_ฟันธง_HDP', 'AI_ฟันธง_สูงต่ำ']
    for col in prediction_columns:
        pending[col] = np.nan if '(%)' in col else ''
    if not isinstance(ai, dict) or ai.get('status') != 'ok':
        return pending
    predictions = ai.get('df_predict', pd.DataFrame())
    matches = {}
    for _, record in predictions.iterrows():
        key = _t2_record_fingerprint(record)
        matches.setdefault(key, []).append(record)
    for index, record in pending.iterrows():
        candidates = matches.get(record['_Snapshot'], [])
        # Never guess which duplicate row a prediction belongs to.
        if len(candidates) == 1:
            for col in prediction_columns:
                pending.at[index, col] = candidates[0].get(col, pending.at[index, col])
    return pending
 
 
def _t2_render_card(number, record):
    probabilities = [record.get('AI_โอกาสเหย้าชนะHDP(%)'), record.get('AI_โอกาสออกสูง(%)')]
    usable = all(isinstance(p, (int, float, np.number)) and np.isfinite(p) for p in probabilities)
    if usable:
        # Only Tab 2's returned HTML is relabelled. Shared renderers stay unchanged.
        html = _card_from_row(number, record)
        html = html.replace('โอกาสคว้ากำไรแฮนดิแคป', 'โอกาสเหย้าได้ เมื่อไม่นับคืนทุน')
        html = html.replace('โอกาสคว้ากำไรสูงต่ำ', 'โอกาสสูงได้ เมื่อไม่นับคืนทุน')
        st.markdown(html, unsafe_allow_html=True)
    else:
        home, away = esc(str(record.get('ทีมเหย้า', ''))), esc(str(record.get('ทีมเยือน', '')))
        st.markdown(
            "<div class='mon'><span class='cn tl'></span><span class='cn tr'></span><span class='cn bl'></span><span class='cn br'></span>"
            f"<div class='mon-bar'><span class='led' style='--lc:#6b7f74'></span><span class='mon-id'>MATCH #{number:02d}</span>"
            "<span class='mon-st'>ไม่มีผลทำนาย</span><span>รอผลสกอร์</span></div>"
            f"<div class='mon-teams'><span class='mon-t'>{home}</span>{_ecg_svg()}<span class='mon-t r'>{away}</span></div>"
            "<div class='ch-d'>ยังไม่มีผลทำนายที่จับคู่กับแถวนี้ได้แน่นอน (ราคา/ไลน์อาจไม่ครบ หรือโมเดลยังไม่พร้อม) "
            "แต่ยังบันทึกผลสกอร์ได้ตามปกติ</div></div>", unsafe_allow_html=True)
 
 
def _t2_finish_saved(row_number, score_text):
    # Data is already confirmed. Failure to refresh must not be reported as a failed write.
    st.session_state['_t2_saved_notice'] = f'ยืนยันบันทึกผลสกอร์ {score_text} ที่ DATA!M{row_number} แล้ว'
    try:
        JOBS.bump_version()
        _t2_load_data.clear()
    except Exception:
        st.session_state['_t2_saved_notice'] += ' กรุณารีเฟรชหน้าเว็บเพื่อโหลดข้อมูลล่าสุด'
 
 
def _t2_flags(pending):
    """คืน (มีผลทำนายครบ 2 ตลาด, มีสัญญาณ ⭐ อย่างน้อย 1 ตลาด) เป็นอาร์เรย์ bool."""
    probabilities = pending[list(_ML_PROB_COLS.values())].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    has_prediction = np.isfinite(probabilities).all(axis=1)
    verdicts = pending[list(_ML_VERDICT_COLS.values())].astype(str)
    starred = verdicts.apply(lambda column: column.str.contains('⭐', regex=False)).any(axis=1).to_numpy(bool)
    return has_prediction, has_prediction & starred
 
 
def _t2_render_tab(wb, ai):
    panel_header('สัญญาณการทำนายและบันทึกผลสกอร์',
                 f'เอนจิน {ML_PATCH_VERSION} · ผลทำนายและช่องกรอกสกอร์อยู่ในการ์ดใบเดียวกัน')
    if wb is None:
        st.error('ไม่สามารถเชื่อมต่อ Google Sheets ได้')
        return
    notice = st.session_state.pop('_t2_saved_notice', None)
    if notice:
        st.success(notice)
    left, right = st.columns(2)
    with left:
        st.button('รีเฟรชข้อมูล', key='refresh_signals', on_click=_t2_refresh)
    with right:
        if st.button('ตรวจยืนยันสกอร์ที่บันทึกค้าง (อ่านอย่างเดียว)', key='t2_reconcile'):
            try:
                confirmed, unresolved = _t2_reconcile(wb)
                st.info(f'ตรวจยืนยันได้ {confirmed} รายการ / ยังยืนยันไม่ได้ {unresolved} รายการ')
            except Exception as exc:
                _t2_error(exc, 'ตรวจผลคำสั่งบันทึกค้าง')
    try:
        raw = _t2_load_data(wb, _t2_workbook_key(wb), JOBS.data_version)
    except Exception as exc:
        _t2_error(exc, 'อ่านข้อมูลดิบของแท็บ 2')
        return
    if not isinstance(ai, dict) or ai.get('status') != 'ok':
        st.warning('ระบบทำนายเดิมยังไม่พร้อม ส่วนลงสกอร์ด้านล่างทำงานแยกกัน ไม่ต้องรอให้โมเดลฝึกสำเร็จ')
    st.caption('ตรวจชื่อทีม เลขแถว และผลการแข่งขันก่อนบันทึก ห้ามจัดเรียง/แทรกแถวหรือเปิดหลายโปรแกรมเขียน DATA พร้อมกัน')
    st.caption('เปอร์เซ็นต์ของโมเดลเป็นโอกาสเฉพาะกรณีไม่คืนทุน ไม่ใช่ผลรับรองความแม่นยำหรือกำไร')
    invalid = raw.loc[raw['_ScoreStatus'].eq('invalid')]
    if len(invalid):
        with st.expander(f'ข้อมูลที่ต้องตรวจในชีต {len(invalid)} รายการ (ไม่บันทึกทับอัตโนมัติ)'):
            display = invalid[['_SheetRow', 'ทีมเหย้า', 'ทีมเยือน', 'ผลสกอร์', '_Issue']].copy()
            display['ผลสกอร์'] = display['ผลสกอร์'].map(str)
            st.dataframe(display.rename(columns={'_SheetRow': 'แถว DATA', '_Issue': 'ประเด็นที่พบ'}), **_stretch())
    pending = _t2_pending_view(raw, ai)
    if pending.empty:
        st.success('ไม่มีรายการที่ชื่อทีมครบและช่องผลสกอร์ว่างใน DATA')
        return
 
    has_prediction, has_signal = _t2_flags(pending)
    st.markdown(_cy_chip_row([
        ('รอผลสกอร์ทั้งหมด', f'{len(pending):,}', NEON_CYAN),
        ('มีผลทำนายครบ', int(has_prediction.sum()), NEON_GREEN),
        ('มีสัญญาณ ⭐', int(has_signal.sum()), NEON_YEL),
        ('ลงสกอร์ได้ (ไม่มีผลทำนาย)', int((~has_prediction).sum()), NEON_GREY)]), unsafe_allow_html=True)
    if has_prediction.any():
        render_monitor_header(pending[has_prediction])
 
    options = {'ทั้งหมด': np.ones(len(pending), dtype=bool), 'มีผลทำนาย': has_prediction, 'เฉพาะมีสัญญาณ ⭐': has_signal}
    mode = st.radio('แสดงรายการ', list(options), horizontal=True, key='_t2_filter')
    view = pending[options[mode]]
    if view.empty:
        st.markdown(_cy_empty('ไม่มีรายการตามตัวกรองนี้ — ลองเลือก "ทั้งหมด"'), unsafe_allow_html=True)
        return
 
    page_size = 20
    pages = (len(view) + page_size - 1) // page_size
    page_key = '_t2_page'
    old_page = st.session_state.get(page_key, 1)
    if not isinstance(old_page, int) or not 1 <= old_page <= pages:
        st.session_state[page_key] = 1
    page = st.selectbox('หน้ารายการรอผล', list(range(1, pages + 1)), key=page_key)
    start = (page - 1) * page_size
    st.caption(f'แสดง {len(view):,} รายการ — รายการที่ {start + 1} ถึง {min(start + page_size, len(view))}')
    render_prediction_cards(view.iloc[start:start + page_size], wb, offset=start)
 
 
def _t3_quality(wb):
    if wb is None:
        return
    try:
        raw = _t2_load_data(wb, _t2_workbook_key(wb), JOBS.data_version)
    except Exception as exc:
        _t2_error(exc, 'ตรวจคุณภาพข้อมูลสำหรับแท็บ 3')
        return
    counts = {key: int(raw['_ScoreStatus'].eq(key).sum()) for key in ('finished', 'pending', 'invalid')}
    with st.container(key='hud_rawstatus'):
        st.markdown(_cy_title('DATA STATUS', 'อ่านตรงจากคอลัมน์ M ของชีต DATA'), unsafe_allow_html=True)
        st.markdown(_cy_chip_row([
            ('สกอร์รูปแบบถูกต้อง', counts['finished'], NEON_GREEN),
            ('รอลงผลสกอร์', counts['pending'], NEON_CYAN),
            ('ต้องตรวจสอบ', counts['invalid'], NEON_RED if counts['invalid'] else NEON_GREY),
            ('แถวทั้งหมด', len(raw), NEON_MAG)]), unsafe_allow_html=True)
        st.markdown("<div class='hud-note'>ตรวจจากชื่อทีมและคอลัมน์ M โดยไม่ใช้สูตรเก่า "
                    "ตัวเลขนี้ไม่ใช่จำนวนตัวอย่างที่โมเดลฝึกจริง (ดูแท็บ \"ข้อมูลและการฝึก\")</div>", unsafe_allow_html=True)
 
# ----------------------------------------
# หัวข้อหลัก (Tabs)
# ----------------------------------------
st.markdown(f"<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI ({ML_PATCH_VERSION})</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>วิเคราะห์ HDP และสูง/ต่ำจากภาพราคา · บันทึกสกอร์ · ตรวจคุณภาพโมเดลและข้อมูลในที่เดียว</p>", unsafe_allow_html=True)
 
AI = get_ai()
 
tab1, tab2, tab3 = st.tabs(["📸 อัปโหลดราคาบอล", "📡 สัญญาณการทำนาย", "📊 Dashboard"])
 
with tab1:
    st.info("💡 **Tips V5.0:** ถ่ายภาพตารางราคาให้เห็น 1 คู่ชัดเจน เพื่อให้ AI ดึงข้อมูลและคำนวณขนาดไม้ลงทุนให้อย่างแม่นยำ")
    fmt_choice = st.selectbox("รูปแบบค่าน้ำ", list(FMT_CHOICES.keys()))
    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{up_n}")
    if uploaded_files and workbook:
        for f in uploaded_files: JOBS.pool.submit(process_job, JOBS.create(f.name, f.getvalue(), FMT_CHOICES[fmt_choice], st.session_state["form5_owner_id"]), workbook, model)
        st.session_state["up_n"] = up_n + 1
        st.rerun()
 
    my_jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if my_jobs or st.session_state.get("form5_editing"):
        st.button("🧹 เคลียข้อมูลหน้านี้", key="clear_upload", on_click=clear_upload_page, help="ล้างรายการงาน การ์ดสัญญาณ ฟอร์ม 5 นัด และไฟล์ที่เลือกไว้ในหน้านี้")
 
    polling = any(j["status"] in ("queued", "reading", "saving") for j in my_jobs)
    if hasattr(st, "fragment"): st.fragment(run_every=2 if polling else None)(render_jobs)(polling)
    else: render_jobs(polling)
    render_form5_editor()
 
with tab2:
    _t2_render_tab(workbook, AI)
 
with tab3:
    panel_header("สรุปผลและคุณภาพข้อมูล", f"เอนจิน {ML_PATCH_VERSION} · ข้อมูลดิบ ผลย้อนหลัง และคุณภาพโมเดลแสดงแยกกันชัดเจน")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("รีเฟรชข้อมูล", key="refresh_dashboard", on_click=_t2_refresh)
        _t3_quality(workbook)
        if not isinstance(AI, dict):
            st.warning("ยังโหลดรายงานโมเดลไม่ได้ แต่สามารถตรวจข้อมูลและลงสกอร์ในแท็บ 2 ได้")
        elif AI.get("status") == "error":
            st.warning("โหลดรายงานโมเดลไม่สำเร็จ: " + str(AI.get("msg", "")) + " — ยังตรวจข้อมูลและลงสกอร์ในแท็บ 2 ได้")
        else:
            render_dashboard(AI)
