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

# 🚀 นำเข้า Machine Learning Model (ใช้ v3.0 The Ultimate Model)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

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
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1b2a22; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7f9a8b; padding: 10px 18px; border-radius: 10px 10px 0 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #22c55e !important; background: #0c130f; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22c55e; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background: #22c55e; color: #03140a; border: none; font-weight: 600; border-radius: 10px; padding: 0.55rem 1.4rem; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover { background: #4ade80; color: #03140a; }
.stButton > button:not([kind="primary"]), [data-testid="stFormSubmitButton"] > button:not([kind="primary"]) {
    background: #0c130f; color: #cfe3d6; border: 1px solid #1b2a22; border-radius: 10px; transition: border-color .2s, box-shadow .2s;
}
.stButton > button:not([kind="primary"]):hover, [data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover {
    border-color: #22c55e; color: #4ade80; box-shadow: 0 0 20px -8px rgba(34,197,94,0.7);
}
[data-testid="stFileUploader"] section { background: #0c130f; border: 1px dashed #2a4136; border-radius: 14px; transition: border-color .2s, box-shadow .2s; }
[data-testid="stFileUploader"] section:hover { border-color: #22c55e; box-shadow: 0 0 28px -12px rgba(34,197,94,0.6); }
[data-testid="stAlert"] { background: #0c130f; border: 1px solid #1b2a22; border-radius: 12px; }
.page-title { font-size: 1.9rem; font-weight: 600; margin: 0; color: #e6f2ea; }
.page-sub { color: #7f9a8b; margin: 4px 0 22px 0; font-size: 0.95rem; }

/* ================= INPUTS (ช่องกรอกข้อมูล) ================= */
label[data-testid="stWidgetLabel"] p { color: #a7c4b5; font-size: 0.85rem; font-weight: 400; }
[data-baseweb="input"], [data-baseweb="base-input"] { background-color: #080e0b !important; border-radius: 10px !important; }
[data-baseweb="input"] { border: 1px solid #1b2a22 !important; transition: border-color .2s, box-shadow .2s; }
[data-baseweb="input"]:focus-within {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 1px rgba(34,197,94,0.55), 0 0 22px -6px rgba(34,197,94,0.65);
}
[data-testid="stNumberInput"] input { font-family: 'JetBrains Mono', monospace !important; font-weight: 500; color: #e6f2ea; }
[data-testid="stNumberInput"] button { background: #0c130f !important; color: #7f9a8b !important; border: none !important; transition: background .2s, color .2s; }
[data-testid="stNumberInput"] button:hover { background: rgba(34,197,94,0.14) !important; color: #4ade80 !important; }
[data-baseweb="select"] > div { background-color: #080e0b !important; border: 1px solid #1b2a22 !important; border-radius: 10px !important; }
[data-baseweb="select"] > div:focus-within { border-color: #22c55e !important; box-shadow: 0 0 22px -8px rgba(34,197,94,0.6); }

/* ฟอร์ม = แผงควบคุม */
[data-testid="stForm"] {
    position: relative; background: linear-gradient(180deg, #0b120e, #080e0b); border: 1px solid #1b2a22;
    border-radius: 16px; padding: 18px 20px 10px;
}
[data-testid="stForm"]::before {
    content: ""; position: absolute; left: 20px; right: 20px; top: -1px; height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent); opacity: .7;
}

/* ช่องกรอกผลสกอร์ = สกอร์บอร์ด */
[class*="st-key-score_input_"] input {
    font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 700 !important;
    text-align: center; letter-spacing: .32em; color: #4ade80 !important; text-shadow: 0 0 14px rgba(74,222,128,0.5);
}
[class*="st-key-score_input_"] input::placeholder { color: #2a4136 !important; text-shadow: none; }
[class*="st-key-score_form_"] [data-testid="stForm"], [data-testid="stForm"][class*="st-key-score_form_"] {
    border-radius: 0 0 18px 18px; border-top: 1px dashed #1b2a22; background: #080e0b; padding: 16px 20px 8px;
}
[class*="st-key-score_form_"] [data-testid="stForm"]::before, [data-testid="stForm"][class*="st-key-score_form_"]::before { display: none; }
[class*="st-key-score_form_"] [data-testid="stFormSubmitButton"] { margin-top: 1.75rem; }

/* หัวฟอร์ม 5 นัด */
.fx-side {
    --c: #22c55e; display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 12px; margin-bottom: 10px;
    border: 1px solid color-mix(in srgb, var(--c) 45%, #1b2a22); background: color-mix(in srgb, var(--c) 9%, transparent);
    font-weight: 500; color: #e6f2ea;
}
.fx-side.away { --c: #d9a441; }
.fx-side small { font-family: 'JetBrains Mono', monospace; color: var(--c); letter-spacing: .1em; font-size: .72rem; }

/* ================= KPI (monitor tiles) ================= */
.kpi-wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 12px 0 22px; }
.kpi {
    --k: #7f9a8b; position: relative; overflow: hidden; border: 1px solid #1b2a22; border-radius: 16px; padding: 20px 24px 22px;
    background-color: #0c130f;
    background-image: linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px);
    background-size: 100% 24px, 24px 100%;
}
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
.kpi-bar > div { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; animation: grow 1.4s cubic-bezier(.2,.8,.2,1); }
.kpi.hero-cyan .kpi-bar > div { background: linear-gradient(90deg, #0284c7, #38bdf8); }
@keyframes grow { from { width: 0; } }
.panel-title { font-size: 1.05rem; font-weight: 500; color: #e6f2ea; margin: 0 0 2px 0; }
.panel-sub { font-size: 0.85rem; color: #7f9a8b; margin: 0 0 6px 0; }

/* ================= JOB CARDS ================= */
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
.job-bar i { display: block; height: 100%; width: 40%; border-radius: 3px; background: linear-gradient(90deg, transparent, #22c55e, #86efac, transparent); animation: sweep 1.4s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(260%); } }

/* ================= MONITOR ================= */
.led { --lc: var(--k, #22c55e); display: inline-block; flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--lc); box-shadow: 0 0 0 0 color-mix(in srgb, var(--lc) 70%, transparent); animation: ping 1.8s infinite; }
@keyframes ping { 70% { box-shadow: 0 0 0 9px color-mix(in srgb, var(--lc) 0%, transparent); } 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--lc) 0%, transparent); } }
.mon-status {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 11px 18px; margin: 6px 0 10px;
    background: #0a110d; border: 1px solid #1b2a22; border-radius: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: #7f9a8b;
}
.mon-status b { color: #4ade80; letter-spacing: .1em; font-weight: 700; }
.mon-status em { font-style: normal; color: #e6f2ea; margin-left: 6px; }
.ticker {
    overflow: hidden; border-top: 1px solid #14211a; border-bottom: 1px solid #14211a; background: #080e0b; margin-bottom: 16px;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
}
.ticker-track { display: flex; width: max-content; padding: 9px 0; animation: marquee var(--dur, 40s) linear infinite; }
.ticker:hover .ticker-track { animation-play-state: paused; }
@keyframes marquee { to { transform: translateX(-50%); } }
.tk { white-space: nowrap; font-size: .84rem; color: #a7c4b5; margin-right: 38px; }
.tk b { color: #e6f2ea; font-weight: 500; margin-right: 6px; }
.tk .d { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin: 0 6px 0 12px; background: var(--c); box-shadow: 0 0 8px var(--c); }

.mon {
    position: relative; overflow: hidden; border-radius: 18px 18px 0 0; padding: 14px 22px 22px; margin-bottom: -14px;
    background:
        linear-gradient(rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 100% 26px,
        linear-gradient(90deg, rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 26px 100%,
        linear-gradient(180deg, #0e1712, #0b110e);
    border: 1px solid #1b2a22; border-bottom: none; transition: border-color .25s, box-shadow .25s;
}
.mon:hover { border-color: #1f6b3d; box-shadow: 0 0 40px -18px rgba(34,197,94,0.6); }
.mon::after {
    content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%; pointer-events: none;
    background: linear-gradient(180deg, transparent, rgba(74,222,128,0.07), transparent); animation: scan 5.5s linear infinite;
}
@keyframes scan { to { transform: translateY(350%); } }
.cn { position: absolute; width: 12px; height: 12px; border: 2px solid #22c55e; opacity: .65; pointer-events: none; }
.cn.tl { top: 8px; left: 8px; border-right: none; border-bottom: none; }
.cn.tr { top: 8px; right: 8px; border-left: none; border-bottom: none; }
.cn.bl { bottom: 8px; left: 8px; border-right: none; border-top: none; }
.cn.br { bottom: 8px; right: 8px; border-left: none; border-top: none; }
.mon-bar { display: flex; align-items: center; gap: 12px; font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #7f9a8b; padding: 0 6px; }
.mon-id { color: #cfe3d6; letter-spacing: .1em; }
.mon-st { padding: 2px 10px; border-radius: 999px; border: 1px solid #1b2a22; font-family: 'Prompt', sans-serif; }
.mon-st.on { color: #4ade80; border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.1); }
.sig { margin-left: auto; display: flex; align-items: flex-end; gap: 3px; height: 14px; }
.sig i { width: 4px; border-radius: 1px; background: #1f2e26; }
.sig i:nth-child(1) { height: 5px; } .sig i:nth-child(2) { height: 8px; } .sig i:nth-child(3) { height: 11px; } .sig i:nth-child(4) { height: 14px; }
.sig i.on { background: #4ade80; box-shadow: 0 0 6px rgba(74,222,128,0.7); }
.mon-teams { display: flex; align-items: center; gap: 16px; margin: 16px 0 14px; padding: 0 6px; }
.mon-t { flex: 0 1 auto; max-width: 34%; font-weight: 600; font-size: 1.1rem; color: #e6f2ea; line-height: 1.3; }
.mon-t.r { text-align: right; }
.ecg { flex: 1; min-width: 60px; height: 30px; }
.ecg .pulse { stroke-dasharray: 22 78; animation: ecg 2.6s linear infinite; filter: drop-shadow(0 0 4px #4ade80); }
@keyframes ecg { from { stroke-dashoffset: 100; } to { stroke-dashoffset: 0; } }
.mon-ch { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.ch { --c: #22c55e; padding: 14px 16px 14px; border-radius: 14px; background: rgba(5,9,7,0.72); border: 1px solid #14211a; transition: border-color .25s, box-shadow .25s; }
.ch:hover { border-color: var(--c); box-shadow: 0 0 26px -12px var(--c); }
.ch-h { display: flex; align-items: baseline; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #7f9a8b; letter-spacing: .04em; }
.ch-k { color: var(--c); font-weight: 700; }
.ch-l { margin-left: auto; color: #cfe3d6; font-size: .82rem; }
.ch-body { display: flex; align-items: center; gap: 14px; margin-top: 6px; }
.dial { flex: none; width: 132px; overflow: visible; }
.dial-arc { animation: arcdraw 1.3s cubic-bezier(.2,.8,.2,1) both; filter: drop-shadow(0 0 5px var(--c)); }
@keyframes arcdraw { from { stroke-dasharray: 0 200; } }
.needle { transform-origin: 60px 62px; transform: rotate(var(--rot)); animation: needle 1.3s cubic-bezier(.2,.8,.2,1) both; }
@keyframes needle { from { transform: rotate(-90deg); } }
.ch-val { font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 700; line-height: 1; color: var(--c); text-shadow: 0 0 18px color-mix(in srgb, var(--c) 55%, transparent); }
.ch-val small { font-size: .9rem; color: #7f9a8b; margin-left: 2px; }
.ch-d { font-size: .78rem; color: #7f9a8b; margin: 6px 0 9px; }
.ch-v { display: inline-block; padding: 4px 11px; border-radius: 8px; font-size: .82rem; font-weight: 500; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); }
.vu { display: grid; grid-template-columns: repeat(var(--n, 20), 1fr); gap: 3px; margin-top: 12px; }
.vu.sm { margin-top: 2px; }
.vu.sm i { height: 6px; }
.vu i { height: 9px; border-radius: 2px; background: #17241d; }
.vu i.on { background: var(--c); box-shadow: 0 0 8px color-mix(in srgb, var(--c) 60%, transparent); animation: vuon .35s calc(var(--i) * 40ms) both; }
@keyframes vuon { from { background: #17241d; box-shadow: none; } }
.vu-l { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: .66rem; color: #5d7a69; margin-top: 5px; }

.mon.solo { border-radius: 18px; border-bottom: 1px solid #1b2a22; margin-bottom: 10px; }

/* ===== Dashboard scope panels ===== */
.scope-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr)); gap: 16px; margin-bottom: 8px; }
.scope {
    --c: #22c55e; --k: var(--c); position: relative; overflow: hidden; border-radius: 18px; padding: 16px 22px 24px;
    border: 1px solid #1b2a22; background-color: #0b110e;
    background-image: linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px);
    background-size: 100% 26px, 26px 100%; transition: border-color .25s, box-shadow .25s;
}
.scope:hover { border-color: color-mix(in srgb, var(--c) 55%, #1b2a22); box-shadow: 0 0 40px -20px var(--c); }
.scope::after {
    content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%; pointer-events: none;
    background: linear-gradient(180deg, transparent, color-mix(in srgb, var(--c) 9%, transparent), transparent); animation: scan 6.5s linear infinite;
}
.scope-h { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #7f9a8b; }
.scope-h b { color: var(--c); }
.scope-h .ecg { flex: 1; height: 22px; }
.scope-body { display: flex; flex-wrap: wrap; align-items: center; gap: 22px; margin-top: 14px; }
.scope-dial { flex: none; width: 210px; margin: 0 auto; text-align: center; }
.dial.big { width: 100%; max-width: 210px; }
.sd-val { font-family: 'JetBrains Mono', monospace; font-size: 2.7rem; font-weight: 700; line-height: 1.1; margin-top: -4px; color: var(--c); text-shadow: 0 0 22px color-mix(in srgb, var(--c) 55%, transparent); }
.sd-val small { font-size: 1rem; color: #7f9a8b; margin-left: 2px; }
.sd-lab { font-size: .8rem; color: #7f9a8b; }
.scope-side { flex: 1; min-width: 240px; }
.cnt { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: .82rem; color: #7f9a8b; }
.cnt b { font-family: 'JetBrains Mono', monospace; color: #e6f2ea; margin: 0 3px; }
.cnt .bad b { color: #f06b70; }
.fh { font-size: .76rem; color: #7f9a8b; margin: 18px 0 4px; }
.fr { display: grid; grid-template-columns: 1fr auto; gap: 3px 10px; padding: 6px 0; }
.fn { font-size: .86rem; color: #cfe3d6; }
.fv { font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: var(--c); font-weight: 700; }
.fr .vu { grid-column: 1 / 3; }

@media (max-width: 900px) {
    .kpi-wrap { grid-template-columns: 1fr 1fr; }
    .kpi-wrap .kpi:last-child { grid-column: 1 / -1; }
    .mon-ch { grid-template-columns: 1fr; }
    .mon-t { max-width: 40%; font-size: .98rem; }
}
@media (max-width: 640px) {
    .block-container { padding: 1rem .75rem 3rem; }
    .page-title { font-size: 1.45rem; }
    .page-sub { font-size: .85rem; margin-bottom: 14px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: .9rem; }
    .kpi { padding: 16px 16px 20px; }
    .kpi-value { font-size: 1.9rem; }
    .kpi.hero .kpi-value, .kpi.hero-cyan .kpi-value { font-size: 2.3rem; }
    .kpi-label { font-size: .78rem; }
    .mon { padding: 12px 14px 18px; }
    .mon-bar { flex-wrap: wrap; gap: 8px 10px; }
    .mon-teams { flex-wrap: wrap; gap: 4px 10px; margin: 12px 0 10px; }
    .mon-t, .mon-t.r { max-width: 100%; flex: 1 1 100%; text-align: left; }
    .mon-t.r::before { content: "vs "; color: #5d7a69; font-weight: 400; }
    .ecg { display: none; }
    .dial { width: 100px; }
    .ch-val { font-size: 1.8rem; }
    .scope { padding: 14px 14px 20px; }
    .scope-dial { width: 180px; }
    .sd-val { font-size: 2.3rem; }
    .scope-side { min-width: 100%; }
    .vu { gap: 2px; }
    [class*="st-key-score_form_"] [data-testid="stFormSubmitButton"] { margin-top: 0; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)

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

# 🚀 โหลดโมเดล Flash ตัวใหม่ล่าสุด ซึ่งเก่งเรื่องการสกัดข้อมูลไวและแม่นยำ
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

# 🚀 [อัปเกรด] ฟังก์ชันแปลงข้อมูล AI แบบใหม่ ใช้ Regex ดึงข้อมูลโดยไม่ต้องบังคับ JSON MIME type
def clean_raw_data(raw_text):
    text = raw_text.strip()
    
    # 1. ลบ Markdown ออก
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text).strip()
    
    # 2. ค้นหาเฉพาะก้อนข้อมูลที่อยู่ในวงเล็บ [...] หรือ {...}
    match_array = re.search(r'\[(.*)\]', text, re.DOTALL)
    match_obj = re.search(r'\{.*\}', text, re.DOTALL)
    
    if match_array:
        json_str = "[" + match_array.group(1).strip() + "]"
    elif match_obj:
        json_str = match_obj.group(0).strip()
    else:
        json_str = text
        
    # เคลียร์ comma เกินตัวสุดท้ายทิ้ง
    json_str = re.sub(r',\s*\]$', ']', json_str)
    json_str = re.sub(r',\s*\}$', '}', json_str)
    
    try:
        data = json.loads(json_str)
    except Exception:
        raise ValueError(f"รูปแบบข้อมูลผิดพลาด -> สิ่งที่ AI ตอบมาคือ: {text[:80]}...")
        
    # หาก AI ส่งมาเป็น Object/Dict ให้ดึงเฉพาะค่า Values ออกมาเรียง
    if isinstance(data, dict):
        data = list(data.values())
        
    # หาก AI ส่งมาเป็น List ซ้อน List (เช่น [["ทีม", ...]]) ให้ดึงชั้นแรกออกมา
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]
        
    if not isinstance(data, list):
        raise ValueError(f"ข้อมูลไม่ใช่ Array -> สิ่งที่ AI ตอบมาคือ: {str(data)[:80]}...")
        
    if len(data) < 12:
        raise ValueError(f"ข้อมูลมาไม่ครบ (ได้ {len(data)} ช่อง จาก 12) -> ข้อมูลที่ได้: {str(data)[:80]}...")
        
    home = str(data[0]).strip()
    away = str(data[1]).strip()
    
    numbers = []
    for i in range(2, 12):
        val_str = str(data[i]).replace('−', '-').replace(',', '').strip()
        try: numbers.append(float(val_str))
        except: numbers.append(0.0)
        
    ai_fmt = str(data[12]).strip().upper() if len(data) > 12 else "UNK"
    if ai_fmt not in ["MY", "HK", "ID", "EU"]: ai_fmt = "UNK"
    
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

# 🚀 [อัปเกรด] นำ Prompt และ Safety Settings จากเวอร์ชัน Tab แรกที่ทำงานได้สมบูรณ์แบบมาใช้
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
            # 🚀 ปล่อยอิสระ ไม่บังคับ MIME Type เพื่อป้องกัน AI ตอบ [] เวลามันลังเล
            resp = model.generate_content(
                [{"mime_type": "image/jpeg", "data": buf.getvalue()}, prompt],
                safety_settings=safety_settings
            )
            
            if getattr(resp, 'prompt_feedback', None) and resp.prompt_feedback.block_reason:
                raise ValueError("ภาพถูกเซ็นเซอร์โดยระบบ Safety ของ Google")
                
            raw_text = resp.text.strip()
            if not raw_text:
                raise ValueError("AI ตอบกลับเป็นค่าว่าง (Empty Response)")
                
            row, ai_fmt = clean_raw_data(raw_text)
            return convert_water_to_malay(row, user_fmt, ai_fmt)
            
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
            
    raise RuntimeError(f"{last_error}")

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

def _render_job_signal(res):
    """แสดงการ์ดสัญญาณของคู่ที่เพิ่งบันทึกสำเร็จ"""
    ai = get_ai()
    if ai is None: return
    if ai["status"] != "ok":
        _show_ai_message(ai)
        return
    dfp = ai["df_predict"]
    if len(dfp) == 0 or 'AI_ฟันธง_HDP' not in dfp.columns:
        st.caption("ยังไม่มีสัญญาณสำหรับคู่นี้ (คู่นี้อาจมีผลการแข่งขันแล้ว)")
        return
    m = dfp[(dfp['ทีมเหย้า'].astype(str).str.strip() == str(res['home']).strip()) & (dfp['ทีมเยือน'].astype(str).str.strip() == str(res['away']).strip())]
    if m.empty:
        st.caption("ยังไม่พบสัญญาณของคู่นี้ในรายการที่รอทำนาย — ลองกดรีเฟรชในแท็บสัญญาณการทำนาย")
        return
    st.markdown(_card_from_row(int(res['row']), m.iloc[-1], solo=True), unsafe_allow_html=True)

def render_jobs(polling=False):
    jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if not jobs: return
    if polling and not any(j["status"] in ("queued", "reading", "saving") for j in jobs):
        st.rerun()  # งานเสร็จหมดแล้ว: รีเฟรชทั้งหน้าเพื่อหยุดการดึงสถานะ
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
        if j.get("warn"): body += f"<div class='job-note' style='color:#e8bb62'>{esc(str(j['warn']))}</div>"
        if j["status"] in ("queued", "reading", "saving"): body += "<div class='job-bar'><i></i></div>"
        st.markdown(f"<div class='job-card {cls}'>{head}{body}</div>", unsafe_allow_html=True)
        if j["status"] == "done" and res:
            _render_job_signal(res)
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

def render_form5_editor():
    job_id = st.session_state.get("form5_editing")
    job = JOBS.get(job_id) if job_id else None
    if not job or job["status"] != "ready": return
    panel_header("ฟอร์ม 5 นัดล่าสุด", "Win Rate กรอกเป็นเปอร์เซ็นต์ (เช่น 60) ส่วนประตูกรอกเป็นยอดรวม 5 นัด")
    with st.form(f"form5_{job_id}"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"<div class='fx-side'><small>HOME</small>{esc(str(job['row_data'][0]))}</div>", unsafe_allow_html=True)
            h_win = st.number_input('Win Rate เหย้า (%)', step=20., key=f"{job_id}_h_win")
            h_for = st.number_input('ประตูได้ (รวม 5 นัด)', step=1, key=f"{job_id}_h_for")
            h_ag = st.number_input('ประตูเสีย (รวม 5 นัด)', step=1, key=f"{job_id}_h_ag")
        with right:
            st.markdown(f"<div class='fx-side away'><small>AWAY</small>{esc(str(job['row_data'][1]))}</div>", unsafe_allow_html=True)
            a_win = st.number_input('Win Rate เยือน (%)', step=20., key=f"{job_id}_a_win")
            a_for = st.number_input('ประตูได้ (รวม 5 นัด)', step=1, key=f"{job_id}_a_for")
            a_ag = st.number_input('ประตูเสีย (รวม 5 นัด)', step=1, key=f"{job_id}_a_ag")
        if st.form_submit_button('บันทึก', type="primary"):
            submit_form5(job_id, [h_win, h_for, h_ag, a_win, a_for, a_ag], st.session_state["form5_owner_id"])
            st.session_state.pop("form5_editing", None)
            st.rerun()

@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_data(_workbook, version=0):
    ws = get_ws(_workbook)
    return pd.DataFrame(ws.get_all_records())

# ----------------------------------------
# AI : เทรนโมเดล + ทำนาย (🚀 ใช้งาน HistGradientBoosting ML v3.0 The Ultimate Model 🚀)
# ----------------------------------------
FEATURE_LABELS = {
    'Implied_Home_Prob': 'โอกาสชนะ 1X2 เหย้า',
    'Implied_Draw_Prob': 'โอกาสออกเสมอ 1X2',
    'Implied_Away_Prob': 'โอกาสชนะ 1X2 เยือน',
    'Bookie_Margin': 'ค่าน้ำโต๊ะ (Margin)',
    'HDP_vs_1X2_Diff': 'ส่วนต่างราคาหลอก HDP',
    '1X2_Variance': 'ความห่างชั้น (Variance)',
    'Expected_Total_Goals': 'ประตูรวมที่คาดหวัง',
    'OU_Goal_Diff': 'ส่วนต่างโกลที่คาด',
    'Total_Goals_Conceded_Last5': 'เสียประตู 5 นัดรวม',
    'H_GoalsFor_Last5': 'เหย้ายิงได้ 5 นัด',
    'H_GoalsAgainst_Last5': 'เหย้าเสีย 5 นัด',
    'A_GoalsFor_Last5': 'เยือนยิงได้ 5 นัด',
    'A_GoalsAgainst_Last5': 'เยือนเสีย 5 นัด',
    'H_WinRate_Last5': 'WinRate เหย้า 5 นัด',
    'A_WinRate_Last5': 'WinRate เยือน 5 นัด',
    '1X2 เหย้า': 'ราคา 1X2 เหย้า',
    '1X2 เสมอ': 'ราคา 1X2 เสมอ',
    '1X2 เยือน': 'ราคา 1X2 เยือน',
    'แฮนดิแคป (เหย้า)': 'แต้มต่อ HDP',
    'ค่าน้ำ HDP (เหย้า)': 'ค่าน้ำ HDP',
    'โกลสูง/ต่ำ': 'ไลน์สูง-ต่ำ',
    'ค่าน้ำ (สูง)': 'ค่าน้ำสูง',
    'ค่าน้ำ (ต่ำ)': 'ค่าน้ำต่ำ'
}

def top_features(model, feature_names, n=5):
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(n)
    df_imp['Feature'] = df_imp['Feature'].replace(FEATURE_LABELS)
    return list(zip(df_imp['Feature'], df_imp['Importance']))[::-1]

@st.cache_data(ttl=60, show_spinner=False)
def compute_ai(df_in):
    df = df_in.copy()

    odds_raw = ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)']
    stats_raw = ['H_WinRate_Last5', 'H_GoalsFor_Last5', 'H_GoalsAgainst_Last5', 'A_WinRate_Last5', 'A_GoalsFor_Last5', 'A_GoalsAgainst_Last5']
    numeric_cols_raw = odds_raw + stats_raw
    
    for col in numeric_cols_raw:
        if col in df.columns:
            if df[col].dtype == object: df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if all(c in df.columns for c in ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)']):
        df['Implied_Home_Prob'] = 1 / df['1X2 เหย้า']
        df['Implied_Draw_Prob'] = 1 / df['1X2 เสมอ'] 
        df['Implied_Away_Prob'] = 1 / df['1X2 เยือน']
        
        df['Bookie_Margin'] = df['Implied_Home_Prob'] + df['Implied_Draw_Prob'] + df['Implied_Away_Prob'] - 1
        
        m = (df['1X2 เหย้า'] + df['1X2 เสมอ'] + df['1X2 เยือน']) / 3.0
        df['1X2_Variance'] = ((df['1X2 เหย้า'] - m)**2 + (df['1X2 เสมอ'] - m)**2 + (df['1X2 เยือน'] - m)**2) / 2.0
        df['1X2_Variance'] = df['1X2_Variance'].fillna(0)
        
        df['HDP_vs_1X2_Diff'] = df['Implied_Home_Prob'] - (0.5 - df['แฮนดิแคป (เหย้า)'] * 0.1)

    if all(c in df.columns for c in ['H_GoalsFor_Last5', 'A_GoalsAgainst_Last5', 'A_GoalsFor_Last5', 'H_GoalsAgainst_Last5', 'โกลสูง/ต่ำ']):
        df['Exp_Goals_Home'] = (df['H_GoalsFor_Last5'] + df['A_GoalsAgainst_Last5']) / 10.0
        df['Exp_Goals_Away'] = (df['A_GoalsFor_Last5'] + df['H_GoalsAgainst_Last5']) / 10.0
        df['Expected_Total_Goals'] = df['Exp_Goals_Home'] + df['Exp_Goals_Away']
        df['OU_Goal_Diff'] = df['Expected_Total_Goals'] - df['โกลสูง/ต่ำ']
        df['Total_Goals_Conceded_Last5'] = df['H_GoalsAgainst_Last5'] + df['A_GoalsAgainst_Last5']

    if not ('ผล HDP' in df.columns and 'ผลสูง/ต่ำ' in df.columns):
        return {"status": "nocols", "level": "warning", "msg": "⚠️ ไม่พบคอลัมน์ 'ผล HDP' หรือ 'ผลสูง/ต่ำ' ใน Google Sheets"}

    def map_hdp_result(res):
        res = str(res).strip()
        if res == "" or res.lower() in ("nan", "none"): return np.nan
        if 'เจ้าบ้านได้เต็ม' in res or 'เจ้าบ้านได้ครึ่ง' in res: return 1
        if 'เจ้าบ้านเสียเต็ม' in res or 'เจ้าบ้านเสียครึ่ง' in res: return 0
        return np.nan

    def map_ou_result(res):
        res = str(res).strip()
        if res == "" or res.lower() in ("nan", "none"): return np.nan
        if 'สูงได้เต็ม' in res or 'สูงได้ครึ่ง' in res: return 1
        if 'ต่ำได้เต็ม' in res or 'ต่ำได้ครึ่ง' in res: return 0
        return np.nan

    df['Target_HDP'] = df['ผล HDP'].apply(map_hdp_result)
    df['Target_OU'] = df['ผลสูง/ต่ำ'].apply(map_ou_result)

    if 'ผลสกอร์' in df.columns:
        df_predict = df[df['ผลสกอร์'].astype(str).str.strip().isin(["", "nan", "None"]) & df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()
    else:
        df_predict = df[df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()

    core_features = ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'Target_HDP', 'Target_OU']
    missing_cols = [c for c in core_features if c not in df.columns]
    if missing_cols:
        return {"status": "missing", "level": "info", "msg": f"ℹ️ รอข้อมูล หรือไม่พบคอลัมน์: {', '.join(missing_cols)}"}

    df_train_hdp = df.dropna(subset=['Target_HDP']).copy()
    df_train_ou = df.dropna(subset=['Target_OU']).copy()
    
    if len(df_train_hdp) <= 5 or len(df_train_ou) <= 5:
        return {"status": "few", "level": "info", "msg": "ℹ️ ต้องการข้อมูลที่มีผลการแข่งขันในอดีตอย่างน้อย 5 แถวขึ้นไป เพื่อให้ AI เริ่มเรียนรู้"}

    if len(df_train_hdp['Target_HDP'].unique()) < 2 or len(df_train_ou['Target_OU'].unique()) < 2:
        return {"status": "few", "level": "info", "msg": "ℹ️ ข้อมูลผลลัพธ์ที่บันทึกไว้ในอดีตมีเพียงหน้าเดียว (เช่น มีแต่ข้อมูลแทงชนะ) AI ต้องการข้อมูลทั้งคู่ที่ชนะและแพ้เพื่อหาความต่าง"}

    odds_features = [c for c in odds_raw if c in df.columns] + ['Implied_Home_Prob', 'Implied_Draw_Prob', 'Implied_Away_Prob', 'Bookie_Margin', 'HDP_vs_1X2_Diff', '1X2_Variance']
    odds_features = [c for c in odds_features if c in df.columns]
    
    stats_features = [c for c in stats_raw if c in df.columns] + ['Expected_Total_Goals', 'OU_Goal_Diff', 'Total_Goals_Conceded_Last5']
    stats_features = [c for c in stats_features if c in df.columns]
    
    all_features = odds_features + stats_features

    # 🚀 [FIX v3.0] กรองคอลัมน์ที่ข้อมูลเหมือนกันหมดทุกแถวทิ้ง (Zero-Variance)
    valid_features_hdp = [f for f in all_features if df_train_hdp[f].nunique(dropna=True) > 1]
    valid_features_ou = [f for f in all_features if df_train_ou[f].nunique(dropna=True) > 1]

    if not valid_features_hdp or not valid_features_ou:
        return {"status": "few", "level": "info", "msg": "ℹ️ ข้อมูลราคายังไม่มีความหลากหลายพอ (ข้อมูลมีค่าซ้ำกันหมดทุกคอลัมน์) ให้เพิ่มข้อมูลคู่อื่นๆ อีกสักนิด"}

    X_train_hdp = df_train_hdp[valid_features_hdp]
    y_train_hdp = df_train_hdp['Target_HDP']
    model_hdp = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, random_state=42, early_stopping=False)
    model_hdp.fit(X_train_hdp, y_train_hdp)
    
    X_train_ou = df_train_ou[valid_features_ou]
    y_train_ou = df_train_ou['Target_OU']
    model_ou = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, random_state=42, early_stopping=False)
    model_ou.fit(X_train_ou, y_train_ou)

    # จัดการ RandomForest สำหรับแสดง Feature Importance Dashboard
    rf_hdp_feats = [f for f in odds_features if f in valid_features_hdp]
    rf_ou_feats = [f for f in odds_features if f in valid_features_ou]
    
    if not rf_hdp_feats: rf_hdp_feats = valid_features_hdp[:1]
    if not rf_ou_feats: rf_ou_feats = valid_features_ou[:1]

    rf_hdp = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(df_train_hdp[rf_hdp_feats].fillna(0), y_train_hdp)
    rf_ou = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(df_train_ou[rf_ou_feats].fillna(0), y_train_ou)

    hdp_correct = int((model_hdp.predict(X_train_hdp) == y_train_hdp).sum())
    ou_correct = int((model_ou.predict(X_train_ou) == y_train_ou).sum())
    
    total_hdp_train = len(df_train_hdp)
    total_ou_train = len(df_train_ou)

    hdp_acc = round((hdp_correct / total_hdp_train) * 100, 1) if total_hdp_train > 0 else 0
    ou_acc = round((ou_correct / total_ou_train) * 100, 1) if total_ou_train > 0 else 0

    if len(df_predict) > 0:
        probs_hdp = model_hdp.predict_proba(df_predict[valid_features_hdp])[:, 1]
        probs_ou = model_ou.predict_proba(df_predict[valid_features_ou])[:, 1]

        df_predict['AI_โอกาสเหย้าชนะHDP(%)'] = (np.array(probs_hdp) * 100).round(1)
        df_predict['AI_โอกาสออกสูง(%)'] = (np.array(probs_ou) * 100).round(1)

        # --- อัปเกรด: ระบบพลิกคำใบ้ ต่อ/รอง ตามสถานะแต้มต่อจริงแบบอัตโนมัติ ---
        def get_hdp_decision(row):
            prob = row['AI_โอกาสเหย้าชนะHDP(%)']
            try: hdp = float(row['แฮนดิแคป (เหย้า)'])
            except: hdp = 0.0
            
            # ตรวจสอบว่าใครคือทีมต่อ ใครคือทีมรอง จากเครื่องหมาย +/-
            if hdp < 0:    # เจ้าบ้านราคาติดลบ แปลว่า "เจ้าบ้านต่อ"
                home_txt, away_txt = "ต่อเจ้าบ้าน", "รองทีมเยือน"
            elif hdp > 0:  # เจ้าบ้านราคาเป็นบวก แปลว่า "เจ้าบ้านรอง"
                home_txt, away_txt = "รองเจ้าบ้าน", "ต่อทีมเยือน"
            else:          # ราคาเสมอ
                home_txt, away_txt = "แทงเจ้าบ้าน", "แทงทีมเยือน"
                
            # ให้คำแนะนำโดยอิงจากโอกาสชนะของ "เจ้าบ้าน"
            if prob >= 60: return f"⭐⭐⭐ VIP ไม้หนัก ({home_txt})"
            elif prob >= 53: return f"⭐⭐ น่าลงทุน ({home_txt})"
            elif prob <= 42: return f"⭐⭐⭐ VIP ไม้หนัก ({away_txt})"
            elif prob <= 47: return f"⭐⭐ น่าลงทุน ({away_txt})"
            else: return "⚠️ ข้าม (ราคาก้ำกึ่ง 50/50)"

        def ou_decision(prob):
            if prob >= 60: return "⭐⭐⭐ VIP สูง (Over) โคตรมั่นใจ" 
            elif prob >= 55: return "⭐⭐ สูง (Over) มั่นใจมาก" 
            elif prob >= 52: return "⭐ น่าลงทุน (สูง Over)" 
            else: return "⚠️ ข้าม (O/U ทรงไม่ชัดเจน)"

        # เรียกใช้ฟังก์ชันใหม่ โดยส่งข้อมูลไปประมวลผลทั้งแถว (เพื่อเช็คราคา HDP)
        df_predict['AI_ฟันธง_HDP'] = df_predict.apply(get_hdp_decision, axis=1)
        df_predict['AI_ฟันธง_สูงต่ำ'] = df_predict['AI_โอกาสออกสูง(%)'].apply(ou_decision)

    return {"status": "ok", "df_predict": df_predict, "hdp_acc": hdp_acc, "ou_acc": ou_acc,
            "hdp_correct": hdp_correct, "ou_correct": ou_correct, "total_train": max(total_hdp_train, total_ou_train),
            "hdp_feats": top_features(rf_hdp, rf_hdp_feats), "ou_feats": top_features(rf_ou, rf_ou_feats)}

def get_ai():
    if not workbook: return None
    try:
        return compute_ai(load_dashboard_data(workbook, JOBS.data_version))
    except Exception as e:
        err_detail = traceback.format_exc()
        return {"status": "error", "level": "error", "msg": f"โหลดข้อมูลไม่สำเร็จ: {e}", "trace": err_detail}

def _show_ai_message(ai):
    level = ai.get("level", "info")
    msg = ai.get("msg", "")
    if level == "error":
        st.error(msg)
        if "trace" in ai:
            with st.expander("🛠️ ข้อมูลสำหรับนักพัฒนา (Traceback Error)"):
                st.code(ai["trace"], language="python")
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

# ----------------------------------------
# ชิ้นส่วนภาพแบบ Monitor (เกจ, VU meter, ECG)
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
    cells = "".join(f"<i class='{'on' if i < lit else ''}' style='--i:{i}'></i>" for i in range(n))
    return f"<div class='vu {cls}' style='--n:{n}'>{cells}</div>"

def _channel_html(ch, name, line_txt, prob, verdict, color, desc, lo_lbl, hi_lbl):
    p = max(0.0, min(float(prob), 100.0))
    return (f"<div class='ch' style='--c:{color}'>"
            f"<div class='ch-h'><span class='ch-k'>{ch}</span><span>{name}</span><span class='ch-l'>{esc(line_txt)}</span></div>"
            "<div class='ch-body'>"
            f"{_dial_svg(p, color)}"
            f"<div><div class='ch-val'>{p:g}<small>%</small></div><div class='ch-d'>{desc}</div><span class='ch-v'>{esc(verdict)}</span></div>"
            "</div>"
            f"{_vu(p, 20)}<div class='vu-l'><span>{lo_lbl}</span><span>50%</span><span>{hi_lbl}</span></div></div>")

def _match_card_html(n, home, away, hdp_line, ou_line, hdp_prob, ou_prob, hdp_verdict, ou_verdict, solo=False):
    t1, t2 = _pred_tone(hdp_verdict, "hdp"), _pred_tone(ou_verdict, "ou")
    strength = max(_TONE_STRENGTH[t1], _TONE_STRENGTH[t2])
    bars = "".join(f"<i class='{'on' if k < strength else ''}'></i>" for k in range(4))
    status = "<span class='mon-st on'>มีสัญญาณ</span>" if strength > 0 else "<span class='mon-st'>ข้าม</span>"
    tag = "บันทึกแล้ว" if solo else "รอผลสกอร์"
    return (f"<div class='mon{' solo' if solo else ''}'><span class='cn tl'></span><span class='cn tr'></span><span class='cn bl'></span><span class='cn br'></span>"
            f"<div class='mon-bar'><span class='led'></span><span class='mon-id'>MATCH #{n:02d}</span>{status}<span>{tag}</span><span class='sig'>{bars}</span></div>"
            f"<div class='mon-teams'><span class='mon-t'>{esc(home)}</span>{_ecg_svg()}<span class='mon-t r'>{esc(away)}</span></div>"
            "<div class='mon-ch'>"
            + _channel_html("CH1", "HANDICAP", f"เหย้า {_fmt_num(hdp_line, True)}", hdp_prob, hdp_verdict,
                            _TONE_COLORS["hdp"][t1], "โอกาสเหย้าชนะแฮนดิแคป", "เยือน", "เหย้า")
            + _channel_html("CH2", "OVER / UNDER", f"ไลน์ {_fmt_num(ou_line)}", ou_prob, ou_verdict,
                            _TONE_COLORS["ou"][t2], "โอกาสออกสูง", "ต่ำ", "สูง")
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
    track = "".join(items) * 2
    st.markdown(f"<div class='ticker'><div class='ticker-track' style='--dur:{max(24, len(items) * 7)}s'>{track}</div></div>",
                unsafe_allow_html=True)

def _pair(h, a): return (str(h).strip(), str(a).strip())

def _visible_matches(df_pred):
    hidden = st.session_state.setdefault("hidden_matches", set())
    pairs = [_pair(h, a) for h, a in zip(df_pred.get('ทีมเหย้า', []), df_pred.get('ทีมเยือน', []))]
    hidden.intersection_update(pairs)
    if len(pairs) != len(df_pred): return df_pred
    return df_pred[[p not in hidden for p in pairs]]

def render_prediction_cards(df_pred, workbook):
    for n, (idx, r) in enumerate(df_pred.iterrows(), start=1):
        home = str(r.get('ทีมเหย้า', '-'))
        away = str(r.get('ทีมเยือน', '-'))

        with st.container():
            st.markdown(_card_from_row(n, r), unsafe_allow_html=True)

            with st.form(key=f"score_form_{idx}_{home}_{away}"):
                sc1, sc2 = st.columns([3, 2])
                with sc1: score_input = st.text_input("กรอกผลสกอร์ เหย้า-เยือน (เช่น 2-1)", placeholder="0-0", key=f"score_input_{idx}")
                with sc2: submitted = st.form_submit_button("💾 บันทึกผลลงชีต", use_container_width=True)

                if submitted:
                    if score_input.strip():
                        success = update_match_score_in_sheet(get_ws(workbook), home, away, score_input.strip())
                        if success:
                            st.session_state.setdefault("hidden_matches", set()).add(_pair(home, away))
                            JOBS.bump_version()
                            st.toast(f"บันทึกผลสกอร์ {score_input.strip()} ของ {home} vs {away} เรียบร้อยแล้ว", icon="✅")
                            st.rerun()
                        else: st.error("ไม่พบคู่นี้ใน Google Sheets กรุณาตรวจสอบชื่อทีม")
                    else: st.warning("กรุณากรอกผลสกอร์ก่อนกดบันทึก")
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ----------------------------------------
# Dashboard : KPI + เกจความแม่นยำ + ปัจจัยสำคัญ
# ----------------------------------------
def _kpi_html(hdp_acc, ou_acc, total_train):
    return ("<div class='kpi-wrap'>"
            f"<div class='kpi hero'><div class='kpi-label'><span class='led'></span>ความแม่นยำ AI (แฮนดิแคป)</div>"
            f"<div class='kpi-value'>{hdp_acc}<span class='kpi-unit'>%</span></div><div class='kpi-bar'><div style='width:{min(hdp_acc, 100)}%'></div></div></div>"
            f"<div class='kpi hero-cyan'><div class='kpi-label'><span class='led'></span>ความแม่นยำ AI (สูง-ต่ำ)</div>"
            f"<div class='kpi-value'>{ou_acc}<span class='kpi-unit'>%</span></div><div class='kpi-bar'><div style='width:{min(ou_acc, 100)}%'></div></div></div>"
            f"<div class='kpi'><div class='kpi-label'><span class='led'></span>เรียนรู้จากข้อมูลผลลัพธ์แล้ว</div>"
            f"<div class='kpi-value'>{total_train}<span class='kpi-unit'>คู่</span></div></div></div>")

def _scope_html(ch, name, color, acc, correct, total, feats):
    top = max((float(v) for _, v in feats), default=1) or 1
    rows = "".join(f"<div class='fr'><span class='fn'>{esc(str(nm))}</span><span class='fv'>{float(v) * 100:.1f}%</span>{_vu(float(v) / top * 100, 20, 'sm')}</div>"
                   for nm, v in feats)
    return (f"<div class='scope' style='--c:{color}'>"
            f"<div class='scope-h'><span class='led'></span><b>{ch}</b><span>{name}</span>{_ecg_svg()}</div>"
            "<div class='scope-body'>"
            f"<div class='scope-dial'>{_dial_svg(acc, color, 'dial big')}<div class='sd-val'>{acc:g}<small>%</small></div><div class='sd-lab'>ความแม่นยำของ AI</div></div>"
            "<div class='scope-side'>"
            f"<div class='cnt'><span>ทายถูก<b>{correct}</b>คู่</span><span class='bad'>ทายผิด/เจ๊า<b>{total - correct}</b>คู่</span></div>"
            f"{_vu(acc, 25)}"
            "<div class='fh'>ปัจจัยที่ AI ใช้ตัดสินใจมากที่สุด 5 อันดับ</div>"
            f"{rows}</div></div></div>")

def render_dashboard(ai):
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")
    st.markdown(f"<div class='mon-status'><span class='led'></span><b>DASHBOARD</b>"
                f"<span>เรียนรู้จาก<em>{ai['total_train']}</em>คู่</span><span>อัปเดตล่าสุด<em>{stamp}</em></span></div>", unsafe_allow_html=True)
    st.markdown(_kpi_html(ai["hdp_acc"], ai["ou_acc"], ai["total_train"]), unsafe_allow_html=True)
    st.markdown("<div class='scope-grid'>"
                + _scope_html("CH1", "แฮนดิแคป", GREEN, ai["hdp_acc"], ai["hdp_correct"], ai["total_train"], ai["hdp_feats"])
                + _scope_html("CH2", "สูง-ต่ำ", "#38bdf8", ai["ou_acc"], ai["ou_correct"], ai["total_train"], ai["ou_feats"])
                + "</div>", unsafe_allow_html=True)

# ----------------------------------------
# หัวข้อหลัก
# ----------------------------------------
st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI (v3.0 The Ultimate Model)</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>สกัดราคาจากภาพ วิเคราะห์ ติดตามผล และพยากรณ์ด้วย HistGradientBoosting ML แบบเรียลไทม์</p>", unsafe_allow_html=True)

AI = get_ai()

tab1, tab2, tab3 = st.tabs(["📸 อัปโหลดราคาบอล", "📡 สัญญาณการทำนาย", "📊 Dashboard"])

# ----------------------------------------
# TAB 1 : อัปโหลดราคาบอล
# ----------------------------------------
with tab1:
    fmt_choice = st.selectbox("รูปแบบค่าน้ำ", list(FMT_CHOICES.keys()))
    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{up_n}")
    if uploaded_files and workbook:
        for f in uploaded_files: JOBS.pool.submit(process_job, JOBS.create(f.name, f.getvalue(), FMT_CHOICES[fmt_choice], st.session_state["form5_owner_id"]), workbook, model)
        st.session_state["up_n"] = up_n + 1
        st.rerun()

    my_jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if my_jobs or st.session_state.get("form5_editing"):
        st.button("🧹 เคลียข้อมูลหน้านี้", key="clear_upload", on_click=clear_upload_page,
                  help="ล้างรายการงาน การ์ดสัญญาณ ฟอร์ม 5 นัด และไฟล์ที่เลือกไว้ในหน้านี้")

    polling = any(j["status"] in ("queued", "reading", "saving") for j in my_jobs)
    if hasattr(st, "fragment"): st.fragment(run_every=2 if polling else None)(render_jobs)(polling)
    else: render_jobs(polling)
    render_form5_editor()

# ----------------------------------------
# TAB 2 : สัญญาณการทำนาย
# ----------------------------------------
with tab2:
    panel_header("📡 สัญญาณการทำนาย (AI Predictions)", "แมตช์ที่รอผลสกอร์ — บันทึกผลแล้วการ์ดจะหายไปเอง")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("🔄 รีเฟรชข้อมูล", key="refresh_signals", on_click=st.cache_data.clear)
        if AI["status"] != "ok":
            _show_ai_message(AI)
        else:
            df_view = _visible_matches(AI["df_predict"]) if len(AI["df_predict"]) > 0 else AI["df_predict"]
            if len(df_view) > 0:
                render_monitor_header(df_view)
                render_prediction_cards(df_view, workbook)
            else:
                st.success("🎉 ไม่มีคู่ค้างที่ยังไม่ลงผลสกอร์ ทุกคู่ใน Google Sheets บันทึกผลครบถ้วนแล้ว!")

# ----------------------------------------
# TAB 3 : Dashboard
# ----------------------------------------
with tab3:
    panel_header("📊 Dashboard", "ความแม่นยำของโมเดลและปัจจัยที่ AI ใช้ตัดสินใจ")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("🔄 รีเฟรชข้อมูล", key="refresh_dashboard", on_click=st.cache_data.clear)
        if AI["status"] != "ok":
            _show_ai_message(AI)
        else:
            render_dashboard(AI)
