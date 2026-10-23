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

/* ================= INPUTS (ช่องกรอกข้อมูล) ================= */
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

/* ================= KPI (monitor tiles) ================= */
.kpi-wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 12px 0 22px; }
.kpi { --k: #7f9a8b; position: relative; overflow: hidden; border: 1px solid #1b2a22; border-radius: 16px; padding: 20px 24px 22px; background-color: #0c130f; background-image: linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px); background-size: 100% 24px, 24px 100%; }
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
.mon-status { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 11px 18px; margin: 6px 0 10px; background: #0a110d; border: 1px solid #1b2a22; border-radius: 14px; font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: #7f9a8b; }
.mon-status b { color: #4ade80; letter-spacing: .1em; font-weight: 700; }
.mon-status em { font-style: normal; color: #e6f2ea; margin-left: 6px; }
.ticker { overflow: hidden; border-top: 1px solid #14211a; border-bottom: 1px solid #14211a; background: #080e0b; margin-bottom: 16px; -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); }
.ticker-track { display: flex; width: max-content; padding: 9px 0; animation: marquee var(--dur, 40s) linear infinite; }
.ticker:hover .ticker-track { animation-play-state: paused; }
@keyframes marquee { to { transform: translateX(-50%); } }
.tk { white-space: nowrap; font-size: .84rem; color: #a7c4b5; margin-right: 38px; }
.tk b { color: #e6f2ea; font-weight: 500; margin-right: 6px; }
.tk .d { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin: 0 6px 0 12px; background: var(--c); box-shadow: 0 0 8px var(--c); }

.mon { position: relative; overflow: hidden; border-radius: 18px 18px 0 0; padding: 14px 22px 22px; margin-bottom: -14px; background: linear-gradient(rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 100% 26px, linear-gradient(90deg, rgba(34,197,94,0.045) 1px, transparent 1px) 0 0 / 26px 100%, linear-gradient(180deg, #0e1712, #0b110e); border: 1px solid #1b2a22; border-bottom: none; transition: border-color .25s, box-shadow .25s; }
.mon:hover { border-color: #1f6b3d; box-shadow: 0 0 40px -18px rgba(34,197,94,0.6); }
.mon::after { content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%; pointer-events: none; background: linear-gradient(180deg, transparent, rgba(74,222,128,0.07), transparent); animation: scan 5.5s linear infinite; }
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

/* 📱 Responsive Mobile Layout Fix */
@media (max-width: 640px) {
    .block-container { padding: 1rem .75rem 3rem; }
    .page-title { font-size: 1.45rem; }
    .kpi { padding: 16px 16px 20px; }
    .kpi-value { font-size: 1.9rem; }
    .kpi.hero .kpi-value, .kpi.hero-cyan .kpi-value { font-size: 2.3rem; }
    .mon { padding: 12px 14px 18px; }
    .mon-t, .mon-t.r { max-width: 100%; flex: 1 1 100%; text-align: left; }
    .mon-t.r::before { content: "vs "; color: #5d7a69; font-weight: 400; }
    .ecg { display: none; }
    .dial { width: 100px; }
    .ch-val { font-size: 1.8rem; }
    
    /* บังคับเรียงซ้อนบน-ล่างบนหน้าจอมือถือ ไม่บีบอัดสเกล */
    .mon-ch { grid-template-columns: 1fr !important; gap: 16px !important; }
    .ch-h { font-size: 0.8rem; }
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

def update_match_score_in_sheet(workbook, home_team, away_team, score_text):
    home, away = str(home_team).strip(), str(away_team).strip()
    with JOBS.sheet_lock:
        ws = get_ws(workbook)
        col_a, col_b, col_m = ws.col_values(1), ws.col_values(2), ws.col_values(13)
        cell = lambda col, i: str(col[i]).strip() if i < len(col) else ""
        target_row = next((i + 1 for i in range(1, len(col_a))
                           if cell(col_a, i) == home and cell(col_b, i) == away and not cell(col_m, i)), None)
        if target_row is None: return False
        ws.update(f'M{target_row}', [[score_text]])
        return True
    return False

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
        if j["status"] in ("queued", "reading", "saving"): body += "<div class='job-bar'><i></i></div>"
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
            h_win = st.number_input('Win Rate เหย้า (%)', step=20., key=f"{job_id}_h_win", value=float(st.session_state.get(f"{job_id}_h_win", 0.0)))
            h_for = st.number_input('ประตูได้ (รวม 5 นัด)', step=1, key=f"{job_id}_h_for", value=int(st.session_state.get(f"{job_id}_h_for", 0)))
            h_ag = st.number_input('ประตูเสีย (รวม 5 นัด)', step=1, key=f"{job_id}_h_ag", value=int(st.session_state.get(f"{job_id}_h_ag", 0)))
        with right:
            st.markdown(f"<div class='fx-side away'><small>AWAY</small>{esc(str(job['row_data'][1]))}</div>", unsafe_allow_html=True)
            a_win = st.number_input('Win Rate เยือน (%)', step=20., key=f"{job_id}_a_win", value=float(st.session_state.get(f"{job_id}_a_win", 0.0)))
            a_for = st.number_input('ประตูได้ (รวม 5 นัด)', step=1, key=f"{job_id}_a_for", value=int(st.session_state.get(f"{job_id}_a_for", 0)))
            a_ag = st.number_input('ประตูเสีย (รวม 5 นัด)', step=1, key=f"{job_id}_a_ag", value=int(st.session_state.get(f"{job_id}_a_ag", 0)))
            
        if st.form_submit_button('💾 บันทึกทั้งหมด', type="primary"):
            submit_form5(job_id, [h_win, h_for, h_ag, a_win, a_for, a_ag], st.session_state["form5_owner_id"])
            st.session_state.pop("form5_editing", None)
            st.rerun()

# ----------------------------------------
# AI : Machine Learning V5.0 (Ultimate Hybrid)
# ----------------------------------------
ML_PATCH_VERSION = "V5.0-Ultimate-Hybrid"
ML_CONFIG = {
    "seed": 42, "min_rows": 100, 
    "min_ev": 0.02,                   # เกณฑ์ EV ขั้นต่ำ 2%
    "probability_stress": 0.03,       # เผื่อความคลาดเคลื่อนทางสถิติ 3%
    "min_line_examples": 15,
}
ML_STATES = ("FL", "HL", "PUSH", "HW", "FW")
ML_STATE_LABELS = ("เสียเต็ม", "เสียครึ่ง", "คืนทุน", "ได้ครึ่ง", "ได้เต็ม")

ML_ODDS_COLUMNS = ["1X2 เหย้า", "1X2 เสมอ", "1X2 เยือน", "แฮนดิแคป (เหย้า)", "ค่าน้ำ HDP (เหย้า)", "แฮนดิแคป (เยือน)", "ค่าน้ำ HDP (เยือน)", "โกลสูง/ต่ำ", "ค่าน้ำ (สูง)", "ค่าน้ำ (ต่ำ)"]

def _ml_number(value):
    if isinstance(value, (bool, np.bool_)) or value is None: return np.nan
    try:
        number = float(str(value).strip().replace("\u2212", "-"))
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError): return np.nan

def _ml_win_rate(value):
    text = str(value).strip()
    percent = text.endswith("%")
    number = _ml_number(text[:-1] if percent else value)
    if not np.isfinite(number) or number < 0: return np.nan
    if percent or number > 1: number /= 100.0
    return number if 0 <= number <= 1 else np.nan

def _ml_malay_decimal(value):
    v = _ml_number(value)
    if not np.isfinite(v) or v == 0: return np.nan
    if v > 1.0: return v  # Decimal ยุโรป/ฮ่องกง
    if v < -1.0: return 1.0 + (1.0 / abs(v)) # อินโดนีเซีย
    return 1.0 + (v if v > 0 else 1.0 / abs(v)) # มาเลย์

def _ml_blank(value):
    return str(value).strip().lower() in ("", "nan", "none", "null", "<na>")

def _ml_score(value):
    text = str(value).strip().replace("\u2212", "-").replace("\u2013", "-")
    match = re.fullmatch(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", text)
    return tuple(map(int, match.groups())) if match else None

def _ml_valid_line(value, market):
    value = _ml_number(value)
    return bool(np.isfinite(value) and abs(value) <= 100 and abs(value * 4 - round(value * 4)) < 1e-7 and (market == "HDP" or value > 0))

def _ml_settlement(score, line, market):
    if score is None or not _ml_valid_line(line, market): return np.nan
    line = float(line)
    parts = (np.floor(line * 2) / 2, np.ceil(line * 2) / 2)
    if market == "HDP": differences = [score[0] - score[1] + part for part in parts]
    else: differences = [sum(score) - part for part in parts]
    return int(round(float(np.mean(np.sign(differences))) * 2)) + 2

def _ml_allowed(line, market):
    if not _ml_valid_line(line, market): return np.ones(5, dtype=bool)
    family = int(round(float(line) * 4)) % 4
    allowed = np.array([True, False, False, False, True])
    if family == 0: allowed[2] = True
    elif family in (1, 3):
        half_win = (family == 1) if market == "HDP" else (family == 3)
        allowed[3 if half_win else 1] = True
    if market == "OU" and np.isclose(float(line), 0.25): allowed[0] = False
    return allowed

def _ml_probability(raw, lines, market, temperature=1.0):
    raw = np.asarray(raw, dtype=float)
    if len(raw) == 0: return np.empty((0, 5), dtype=float)
    masks = np.vstack([_ml_allowed(line, market) for line in lines])
    logits = np.log(np.clip(raw, 1e-12, 1.0)) / float(temperature)
    logits[~masks] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def _ml_raw_probabilities(model, X):
    probabilities = np.full((len(X), 5), 1e-12, dtype=float)
    if len(X):
        probabilities[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    return probabilities

def _ml_log_loss(y, probabilities):
    return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())

def _ml_metrics(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    actual = np.eye(5)[y]
    return {
        "n": len(y), "correct": int((predicted == y).sum()), "accuracy": float((predicted == y).mean()),
        "log_loss": _ml_log_loss(y, probabilities), "brier": float(np.square(probabilities - actual).sum(axis=1).mean()),
    }

# ==========================================
# 🛠️ ฟังก์ชันส่วนที่หายไป (ใส่กลับให้ครบแล้วครับ)
# ==========================================
def _ml_returns(decimal):
    decimal = float(decimal)
    return np.array([-1.0, -0.5, 0.0, (decimal - 1) / 2, decimal - 1])

def _ml_expected_value(probabilities, decimal, reverse=False):
    if not np.isfinite(decimal) or decimal <= 1: return np.nan
    p = np.asarray(probabilities, dtype=float)
    return float((p[::-1] if reverse else p) @ _ml_returns(decimal))

def _ml_home_share(probabilities):
    """P(เหย้า/สูงชนะ | ไม่คืนทุน) = (HW+FW)/(FL+HL+HW+FW) ฝั่งตรงข้ามคือ 1 - ค่านี้"""
    p = np.asarray(probabilities, dtype=float)
    decisive = p[:, [0, 1, 3, 4]].sum(axis=1)
    return np.where(decisive > 1e-9, p[:, 3:].sum(axis=1) / np.maximum(decisive, 1e-9), 0.5)
# ==========================================

def _ml_label_state(value, market):
    text = str(value).strip()
    if market == "HDP":
        labels = {"เจ้าบ้านเสียเต็ม": 0, "เจ้าบ้านเสียครึ่ง": 1, "คืนทุน": 2, "เจ้าบ้านได้ครึ่ง": 3, "เจ้าบ้านได้เต็ม": 4}
    else:
        labels = {"ต่ำได้เต็ม": 0, "ต่ำได้ครึ่ง": 1, "คืนทุน": 2, "สูงได้ครึ่ง": 3, "สูงได้เต็ม": 4}
    return labels.get(text, np.nan)

def _ml_prepare(df_in):
    df = df_in.copy(deep=True).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    for column in ML_ODDS_COLUMNS: df[column] = df[column].map(_ml_number) if column in df else np.nan
    for column in FORM5_HEADERS:
        if column not in df: df[column] = np.nan
        elif "WinRate" in column: df[column] = df[column].map(_ml_win_rate)
        else: df[column] = df[column].map(_ml_number)
    for column in ("ผลสกอร์", "ผล HDP", "ผลสูง/ต่ำ"):
        if column not in df: df[column] = ""
    
    groups = [json.dumps([re.sub(r"\s+", " ", str(h).strip().casefold()), re.sub(r"\s+", " ", str(a).strip().casefold())], ensure_ascii=False) for h, a in zip(df["ทีมเหย้า"], df["ทีมเยือน"])]
    df["_ML_Group"] = groups
    
    X = pd.DataFrame(index=df.index)
    odds = df[["1X2 เหย้า", "1X2 เสมอ", "1X2 เยือน"]].where(lambda values: values > 1)
    inverse = 1.0 / odds
    inverse_total = inverse.sum(axis=1, min_count=3)
    
    # 🌟 1. Mathematical Features
    X["True_Home_Prob"] = inverse["1X2 เหย้า"] / inverse_total
    X["True_Draw_Prob"] = inverse["1X2 เสมอ"] / inverse_total
    X["True_Away_Prob"] = inverse["1X2 เยือน"] / inverse_total
    X["Bookie_Margin"] = inverse_total - 1.0
    X["1X2_Variance"] = X[["True_Home_Prob", "True_Draw_Prob", "True_Away_Prob"]].var(axis=1)

    for name, source in (("HDP_Home_Decimal", "ค่าน้ำ HDP (เหย้า)"), ("HDP_Away_Decimal", "ค่าน้ำ HDP (เยือน)"), ("Over_Decimal", "ค่าน้ำ (สูง)"), ("Under_Decimal", "ค่าน้ำ (ต่ำ)")):
        X[name] = df[source].map(_ml_malay_decimal)

    # 🌟 2. Data Imputation: กู้คืนค่าน้ำที่ OCR อาจอ่านพลาด โดยใช้ Bookie Margin มาตรฐาน (1.05)
    for c1, c2 in [("HDP_Home_Decimal", "HDP_Away_Decimal"), ("Over_Decimal", "Under_Decimal")]:
        missing_c2 = X[c2].isna() & X[c1].notna()
        if missing_c2.any(): X.loc[missing_c2, c2] = 1.0 / (1.05 - (1.0 / X.loc[missing_c2, c1]))
        missing_c1 = X[c1].isna() & X[c2].notna()
        if missing_c1.any(): X.loc[missing_c1, c1] = 1.0 / (1.05 - (1.0 / X.loc[missing_c1, c2]))

    X["Water_HDP_Diff"] = X["HDP_Home_Decimal"] - X["HDP_Away_Decimal"]
    X["Water_OU_Diff"] = X["Over_Decimal"] - X["Under_Decimal"]

    for market, column in (("HDP", "แฮนดิแคป (เหย้า)"), ("OU", "โกลสูง/ต่ำ")):
        X[market + "_Line"] = df[column].where(df[column].map(lambda v: _ml_valid_line(v, market)))
        X[market + "_Family"] = np.round(X[market + "_Line"] * 4) % 4
        
    X["HDP_vs_TrueProb_Diff"] = X["True_Home_Prob"] - (0.5 - X["HDP_Line"] * 0.1)

    for column in FORM5_HEADERS: X[column] = df[column]

    # 🌟 3. Poisson Distribution & Dynamic Expected Goals
    avg_h_gf, avg_h_ga = X["H_GoalsFor_Last5"] / 5.0, X["H_GoalsAgainst_Last5"] / 5.0
    avg_a_gf, avg_a_ga = X["A_GoalsFor_Last5"] / 5.0, X["A_GoalsAgainst_Last5"] / 5.0
    
    X["Expected_Total_Goals"] = (avg_h_gf + avg_a_ga) / 2.0 + (avg_a_gf + avg_h_ga) / 2.0
    X["Form_Home_Goal_Avg"] = avg_h_gf
    X["Form_Away_Goal_Avg"] = avg_a_gf
    X["Form_Total_vs_Line"] = X["Expected_Total_Goals"] - X["OU_Line"]
    
    # คำนวณเปอร์เซ็นต์สูง/ต่ำ ด้วยสูตรปัวซง
    safe_lines = np.floor(X["OU_Line"].fillna(0))
    safe_goals = X["Expected_Total_Goals"].fillna(2.5)
    X["Poisson_Under_Prob"] = stats.poisson.cdf(safe_lines, safe_goals)
    X["Poisson_Over_Prob"] = 1.0 - X["Poisson_Under_Prob"]
    
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    names_ok = ~df["ทีมเหย้า"].map(_ml_blank) & ~df["ทีมเยือน"].map(_ml_blank)
    base_valid = names_ok & odds.notna().all(axis=1)
    
    df["_ML_Valid_HDP"] = base_valid & X[["HDP_Line", "HDP_Home_Decimal", "HDP_Away_Decimal"]].notna().all(axis=1)
    df["_ML_Valid_OU"] = base_valid & X[["OU_Line", "Over_Decimal", "Under_Decimal"]].notna().all(axis=1)
    
    scores = df["ผลสกอร์"].map(_ml_score)
    for market, label_column, line_column in (("HDP", "ผล HDP", "HDP_Line"), ("OU", "ผลสูง/ต่ำ", "OU_Line")):
        targets = []
        for i in range(len(df)):
            label = _ml_label_state(df.at[i, label_column], market)
            result = _ml_settlement(scores.iloc[i], X.at[i, line_column], market)
            if np.isfinite(result): pass
            elif _ml_blank(df.at[i, "ผลสกอร์"]) and np.isfinite(label): result = label
            if np.isfinite(result) and not _ml_allowed(X.at[i, line_column], market)[int(result)]: result = np.nan
            targets.append(result)
        df["Target_" + market] = targets
    df["_ML_Pending"] = df[["ผลสกอร์", "ผล HDP", "ผลสูง/ต่ำ"]].apply(lambda c: c.map(_ml_blank)).all(axis=1)
    return df, X

def _ml_split(groups):
    groups = np.asarray(groups)
    indices = np.arange(len(groups))
    def take(subset, test_size, seed):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(subset, groups=groups[subset]))
        return subset[train], subset[test]
    rest, test = take(indices, 0.15, ML_CONFIG["seed"])
    rest, validation = take(rest, 0.15 / 0.85, ML_CONFIG["seed"] + 1)
    train, calibration = take(rest, 0.15 / 0.70, ML_CONFIG["seed"] + 2)
    return {"train": train, "calibration": calibration, "validation": validation, "test": test}

# Dynamic Half-Kelly Criterion
def _ml_kelly_fraction(probabilities, decimal, side):
    returns = _ml_returns(decimal)
    if side == 1: returns = returns[::-1]
    ev = float(np.dot(probabilities, returns))
    if ev <= 0: return 0.0
    var = float(np.dot(probabilities, (returns - ev)**2))
    if var <= 0: return 0.0
    half_kelly = (ev / var) * 0.5
    return min(half_kelly, 0.05) # Cap at 5% of Bankroll

def _ml_choices(probabilities, first_prices, second_prices):
    first = np.array([_ml_expected_value(p, price) for p, price in zip(probabilities, first_prices)])
    second = np.array([_ml_expected_value(p, price, True) for p, price in zip(probabilities, second_prices)])
    side = np.where(np.nan_to_num(first, nan=-np.inf) >= np.nan_to_num(second, nan=-np.inf), 0, 1)
    ev = np.where(side == 0, first, second)
    price = np.where(side == 0, np.asarray(first_prices), np.asarray(second_prices))
    stressed = ev - ML_CONFIG["probability_stress"] * price
    candidate = np.isfinite(ev) & (ev >= ML_CONFIG["min_ev"]) & (stressed > 0)
    kellys = np.array([_ml_kelly_fraction(p, pr, s) for p, pr, s in zip(probabilities, price, side)])
    return {"side": side, "ev": ev, "stress_ev": stressed, "price": price, "candidate": candidate, "kelly": kellys}

def _ml_fit_market(df, X, market):
    valid = df["_ML_Valid_" + market] & df["Target_" + market].notna()
    indices = np.flatnonzero(valid.to_numpy())
    data = df.iloc[indices]
    if len(data) < ML_CONFIG["min_rows"]: raise ValueError(f"ML {market}: ต้องมีข้อมูลอย่างน้อย {ML_CONFIG['min_rows']} แถว")
    y = data["Target_" + market].to_numpy(dtype=int)
    split = _ml_split(data["_ML_Group"].to_numpy())
    train, calibration, validation, test = [split[key] for key in ("train", "calibration", "validation", "test")]
    X_all = X.iloc[indices]
    features = [c for c in X.columns if X_all.iloc[train][c].nunique(dropna=True) > 1]
    X_all = X_all[features]
    lines = X.iloc[indices][market + "_Line"].to_numpy()
    
    with threadpool_limits(limits=2):
        # Ensemble Voting (3 HistGradientBoosting)
        m1 = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.03, l2_regularization=5.0, random_state=42)
        m2 = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.01, l2_regularization=2.0, random_state=43)
        m3 = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=10.0, random_state=44)
        
        estimator = VotingClassifier(estimators=[('m1', m1), ('m2', m2), ('m3', m3)], voting='soft')
        estimator.fit(X_all.iloc[train], y[train])
        
        raw_cal = _ml_raw_probabilities(estimator, X_all.iloc[calibration])
        temperatures = (0.8, 1.0, 1.25, 1.5, 2.0)
        temperature = min(temperatures, key=lambda t: _ml_log_loss(y[calibration], _ml_probability(raw_cal, lines[calibration], market, t)))
        
        def predict(part): return _ml_probability(_ml_raw_probabilities(estimator, X_all.iloc[part]), lines[part], market, temperature)
        
        p_val = predict(validation)
        val_metrics = _ml_metrics(y[validation], p_val)
        gate = bool(len(validation) >= 10)
        
        p_test = predict(test)
        p_train = predict(train)
        
        pending_pos = np.flatnonzero(df["_ML_Pending"].to_numpy())
        pending_lines = X.iloc[pending_pos][market + "_Line"].to_numpy()
        pending_probs = _ml_probability(_ml_raw_probabilities(estimator, X.iloc[pending_pos][features]), pending_lines, market, temperature)
        
    family_counts = {family: int((np.round(lines[train] * 4).astype(int) % 4 == family).sum()) for family in range(4)}
    
    report = {
        "market": market, "test": _ml_metrics(y[test], p_test), "gate_pass": gate,
        "family_counts": family_counts
    }
    return report, pending_probs

def _ml_verdict(market, side, prob_val, ev, stress_ev, line, valid, report, kelly,
                row_data=None, features_data=None, min_prob=55.0, min_ev=None):
    min_ev = ML_CONFIG["min_ev"] if min_ev is None else min_ev
    if not valid: return "⚠️ ข้าม (ราคาผิด/ไม่ครบ)"
    if row_data is None: return "⚠️ ข้าม (ไม่มีข้อมูลแถวสำหรับตรวจ Guardrail)"

    side = int(side)
    home_1x2 = _ml_number(row_data.get("1X2 เหย้า", np.nan))
    water_col = {("HDP", 0): "ค่าน้ำ HDP (เหย้า)", ("HDP", 1): "ค่าน้ำ HDP (เยือน)",
                 ("OU", 0): "ค่าน้ำ (สูง)", ("OU", 1): "ค่าน้ำ (ต่ำ)"}[(market, side)]
    water_decimal = _ml_malay_decimal(row_data.get(water_col, np.nan))
    if not np.isfinite(water_decimal):
        return "⚠️ ข้าม (ค่าน้ำฝั่งที่เลือกอ่านไม่ได้/อยู่นอกช่วงแปลงค่า)"

    # 🛡️ DYNAMIC GUARDRAILS (อัปเดตใหม่ตาม Data Insights 2165 แถว + กฎเดิม)
    if market == "HDP":
        # 1. กฎใหม่: บล็อกเรตราคาล่อแมงเม่า: ต่อเจ้าบ้าน 0.5 และ 0.75 
        if side == 0 and line in [-0.5, -0.75]:
            if prob_val < 62.0:  
                return "⚠️ ข้าม (ราคาต่อ 0.5/0.75 เจ้าบ้านมีความเสี่ยงสูง)"
                
        true_home_prob = features_data.get("True_Home_Prob", np.nan) if features_data is not None else np.nan
        if np.isfinite(true_home_prob):
            if true_home_prob < 0.45 and line <= -0.5 and side == 0:
                return "⚠️ ข้าม (ราคาต่อล่อแมงเม่า สวนทางสถิติ 1X2)"
            if true_home_prob > 0.60 and line >= 0.0 and side == 0:
                return "⚠️ ข้าม (ราคาผิดปกติ เจ้าบ้านฟอร์มโหดแต่โต๊ะไม่กล้าให้ต่อ)"
                
    elif market == "OU":
        # 2. กฎใหม่: บล็อกการแทงต่ำ (Under) เมื่อค่าน้ำกดจ่ายน้อยกว่า 2.0 (น้ำดำ/ล่อให้แทง)
        if side == 1 and water_decimal < 2.0:
            return "⚠️ ข้าม (โต๊ะกดค่าน้ำต่ำเพื่อล่อ สถิติมักโดนยิงท้ายเกม)"
            
        if line >= 3.25:
            if side == 0: return "⚠️ ข้าม (เรตสูงผิดปกติ สถิติมักออกต่ำ)"
            if water_decimal < 1.85: return "⚠️ ข้าม (ออกต่ำจริง แต่ค่าน้ำเอาเปรียบเกินไป)"
            
        poisson_over = features_data.get("Poisson_Over_Prob", np.nan) if features_data is not None else np.nan
        if np.isfinite(poisson_over):
            if side == 0 and poisson_over < 0.35: return "⚠️ ข้าม (สถิติ Poisson ชี้ว่าโอกาสสูงน้อยเกินไป)"
            if side == 1 and poisson_over > 0.65: return "⚠️ ข้าม (สถิติ Poisson ชี้ว่าโอกาสต่ำน้อยเกินไป)"

    family = int(round(float(line) * 4)) % 4
    f_count = report["family_counts"].get(family, 0)
    if f_count < ML_CONFIG["min_line_examples"]:
        return "⚠️ ข้าม (ตัวอย่างชนิดไลน์นี้น้อยไป)"
    if not np.isfinite(prob_val) or prob_val < min_prob:
        return f"⚠️ ข้าม (โอกาสชนะน้อยเกินไป: {prob_val:.1f}%)"
    if not np.isfinite(ev) or ev < min_ev or stress_ev <= 0:
        return "⚠️ ข้าม (ความได้เปรียบ EV ไม่ผ่านเกณฑ์)"

    is_high_water = water_decimal >= 1.90
    is_strong_fav = bool(market == "HDP" and np.isfinite(home_1x2) and home_1x2 <= 1.47 and line <= -1.25 and side == 0)
    if is_high_water or is_strong_fav: kelly = min(kelly * 1.5, 0.05)
    k_pct = kelly * 100

    if prob_val >= 65.0 or (prob_val >= 60.0 and is_high_water): level = "⭐⭐⭐ VIP ไม้หนัก"
    elif prob_val >= 60.0 or is_strong_fav: level = "⭐⭐ น่าลงทุน"
    else: level = "⭐ ทดลองติดตาม"

    desc = "สูง Over" if side == 0 else "ต่ำ Under"
    if market == "HDP":
        if line < 0: desc = "ต่อเจ้าบ้าน" if side == 0 else "รองทีมเยือน"
        elif line > 0: desc = "รองเจ้าบ้าน" if side == 0 else "ต่อทีมเยือน"
        else: desc = "เจ้าบ้าน" if side == 0 else "ทีมเยือน"

    warnings = []
    if f_count < 50: warnings.append("ข้อมูลน้อย")
    if (59.5 <= prob_val <= 60.5) or (64.5 <= prob_val <= 65.5): warnings.append("คาบเส้น")
    warn_text = f" ⚠️({','.join(warnings)})" if warnings else ""
    return f"{level} ({desc}) | โอกาส {prob_val:.1f}% | แนะนำ {k_pct:.1f}%{warn_text}" 

@st.cache_data(ttl=300, show_spinner=False)
def compute_ai(df_in):
    try:
        df, X = _ml_prepare(df_in)
        reports, probs = {}, {}
        for m in ("HDP", "OU"):
            reports[m], probs[m] = _ml_fit_market(df, X, m)
    except ValueError as e: return {"status": "few", "level": "warning", "msg": str(e)}
    
    pending_pos = np.flatnonzero(df["_ML_Pending"].to_numpy())
    pending = df.iloc[pending_pos].copy()
    history = df[~df["_ML_Pending"]].copy()
    
    for m, p_col, v_col, c1, c2 in (("HDP", "AI_โอกาสเหย้าชนะHDP(%)", "AI_ฟันธง_HDP", "HDP_Home_Decimal", "HDP_Away_Decimal"),
                                    ("OU", "AI_โอกาสออกสูง(%)", "AI_ฟันธง_สูงต่ำ", "Over_Decimal", "Under_Decimal")):
        p = probs[m]
        prob_home_over = _ml_home_share(p) * 100
        prob_away_under = 100.0 - prob_home_over
        
        # Calculate for pending
        pending[p_col] = np.round(prob_home_over, 1)
        choices = _ml_choices(p, X.iloc[pending_pos][c1], X.iloc[pending_pos][c2])
        side_probs = np.where(choices["side"] == 0, prob_home_over, prob_away_under)
        
        pending[v_col] = [
            _ml_verdict(m, int(choices["side"][i]), side_probs[i], choices["ev"][i], choices["stress_ev"][i],
                        X.iloc[pos][m + "_Line"], bool(df.iloc[pos]["_ML_Valid_" + m]), reports[m], choices["kelly"][i], 
                        row_data=df.iloc[pos], features_data=X.iloc[pos])
            for i, pos in enumerate(pending_pos)
        ]
        
    return {
        "status": "ok", "df_predict": pending, "df_history": history, "total_train": len(history),
        "hdp_acc": round(reports["HDP"]["test"]["accuracy"] * 100, 1),
        "ou_acc": round(reports["OU"]["test"]["accuracy"] * 100, 1),
    }

def load_dashboard_data(_workbook, version=0):
    with JOBS.sheet_lock:
        return pd.DataFrame(get_ws(_workbook).get_all_records())

def get_ai():
    if not workbook: return None
    try: 
        return compute_ai(load_dashboard_data(workbook, JOBS.data_version))
    except Exception as e: 
        err_trace = traceback.format_exc()
        return {
            "status": "error", 
            "level": "error", 
            "msg": f"โหลดข้อมูลไม่สำเร็จ: {type(e).__name__} - {str(e)}", 
            "trace": err_trace
        }

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
        st.markdown(f"<div class='ticker'><div class='ticker-track' style='--dur:{max(24, len(items) * 7)}s'>{track}</div></div>", unsafe_allow_html=True)

def _pair(h, a): return (str(h).strip(), str(a).strip())
def _visible_matches(df_pred):
    hidden = st.session_state.setdefault("hidden_matches", set())
    pairs = [_pair(h, a) for h, a in zip(df_pred.get('ทีมเหย้า', []), df_pred.get('ทีมเยือน', []))]
    hidden.intersection_update(pairs)
    if len(pairs) != len(df_pred): return df_pred
    return df_pred[[p not in hidden for p in pairs]]

def render_prediction_cards(df_pred, workbook):
    for n, (idx, r) in enumerate(df_pred.iterrows(), start=1):
        home, away = str(r.get('ทีมเหย้า', '-')), str(r.get('ทีมเยือน', '-'))
        with st.container():
            st.markdown(_card_from_row(n, r), unsafe_allow_html=True)
            with st.form(key=f"score_form_{idx}_{home}_{away}"):
                sc1, sc2 = st.columns([3, 2])
                with sc1: score_input = st.text_input("กรอกผลสกอร์ เหย้า-เยือน (เช่น 2-1)", placeholder="0-0", key=f"score_input_{idx}")
                with sc2: submitted = st.form_submit_button("💾 บันทึกผลลงชีต", use_container_width=True)
                if submitted:
                    if score_input.strip():
                        success = update_match_score_in_sheet(workbook, home, away, score_input.strip())
                        if success:
                            st.session_state.setdefault("hidden_matches", set()).add(_pair(home, away))
                            JOBS.bump_version()
                            st.toast(f"บันทึกผลสกอร์ {score_input.strip()} ของ {home} vs {away} เรียบร้อยแล้ว", icon="✅")
                            st.rerun()
                        else: st.error("ไม่พบคู่นี้ที่ยังไม่มีสกอร์ใน Google Sheets กรุณาตรวจสอบชื่อทีม")
                    else: st.warning("กรุณากรอกผลสกอร์ก่อนกดบันทึก")
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ----------------------------------------
# Dashboard : KPI & Charts
# ----------------------------------------
def _kpi_html(hdp_acc, ou_acc, total_train):
    return ("<div class='kpi-wrap'>"
            f"<div class='kpi hero'><div class='kpi-label'><span class='led'></span>HDP · ความแม่นยำชุดทดสอบ</div>"
            f"<div class='kpi-value'>{hdp_acc}<span class='kpi-unit'>%</span></div><div class='kpi-bar'><div style='width:{min(hdp_acc, 100)}%'></div></div></div>"
            f"<div class='kpi hero-cyan'><div class='kpi-label'><span class='led'></span>O/U · ความแม่นยำชุดทดสอบ</div>"
            f"<div class='kpi-value'>{ou_acc}<span class='kpi-unit'>%</span></div><div class='kpi-bar'><div style='width:{min(ou_acc, 100)}%'></div></div></div>"
            f"<div class='kpi'><div class='kpi-label'><span class='led'></span>เรียนรู้จากข้อมูลผลลัพธ์แล้ว</div>"
            f"<div class='kpi-value'>{total_train}<span class='kpi-unit'>คู่</span></div></div></div>")

def render_dashboard(ai):
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")
    st.markdown(f"<div class='mon-status'><span class='led'></span><b>DASHBOARD</b>"
                f"<span>เรียนรู้จาก<em>{ai['total_train']}</em>คู่</span><span>อัปเดตล่าสุด<em>{stamp}</em></span></div>", unsafe_allow_html=True)
    st.markdown(_kpi_html(ai["hdp_acc"], ai["ou_acc"], ai["total_train"]), unsafe_allow_html=True)
    
    st.success("✅ **ระบบ V5.0 Ultimate Hybrid ทำงานสมบูรณ์แบบ:**")
    st.markdown("""
    - **Ensemble Voting:** ใช้ HistGradientBoosting 3 โมเดลช่วยกันโหวตกรองความผิดพลาด (Soft Voting)
    - **Risk Management & Ranking:** กรองบิลที่โอกาสชนะ < 55% ทิ้ง 100% พร้อมจัดระดับ ⭐⭐⭐ VIP (สำหรับบิลที่โอกาส > 60% ขึ้นไปและได้ค่าน้ำ 0.90+)
    - **Confidence Indicators:** แจ้งเตือนสัญญาณเสี่ยงสวิงด้วยแท็ก `⚠️(ข้อมูลน้อย)` และ `⚠️(คาบเส้น)` อัตโนมัติ
    - **System Guardrails:** ตรวจจับและบล็อก 'ราคาล่อแมงเม่า' ทั้งราคาต่อ ปป./ครึ่งลูกที่เสี่ยงหลอก และราคาโกลสูง/ต่ำที่เกิน 3.25 อัตโนมัติ
    """)

    st.markdown("---")
    st.markdown("<h3 style='color: #4ade80;'>📊 ข้อมูลเชิงลึกสำหรับการลงทุน (Investment Insights)</h3>", unsafe_allow_html=True)

    df_hist = ai.get("df_history", pd.DataFrame())
    
    # เช็คว่ามีข้อมูล Target_HDP (ที่ AI ประมวลผลสถานะ ได้/เสีย จากสกอร์ให้แล้ว)
    if len(df_hist) > 0 and 'Target_HDP' in df_hist.columns:
        # 1. โดนัทชาร์ตสรุปผลประกอบการจริง (W/L/D Breakdown)
        # แมปค่า 0-4 ที่ AI คำนวณกลับมาเป็นป้ายกำกับ
        label_map = {0: "เสียเต็ม", 1: "เสียครึ่ง", 2: "คืนทุน", 3: "ได้ครึ่ง", 4: "ได้เต็ม"}
        results = {"ได้เต็ม": 0, "ได้ครึ่ง": 0, "คืนทุน": 0, "เสียครึ่ง": 0, "เสียเต็ม": 0}
        
        for t in df_hist['Target_HDP'].dropna():
            if t in label_map: results[label_map[t]] += 1
        for t in df_hist['Target_OU'].dropna():
            if t in label_map: results[label_map[t]] += 1

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**สัดส่วนผลประกอบการจริง (HDP & O/U)**")
            labels = list(results.keys())
            values = list(results.values())
            colors = ['#22c55e', '#86efac', '#7f9a8b', '#d9a441', '#e5484d']
            
            fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6f2ea', margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # 2. Cumulative Profit Curve
            st.markdown("**กราฟจำลองกำไรสะสม (Cumulative Profit Curve)**")
            
            # คำนวณกำไรตามสเกล 0-4 (0=เสีย1ยูนิต, 4=ได้1ยูนิต)
            unit_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            profits = []
            
            for _, row in df_hist.iterrows():
                p = 0
                if pd.notna(row.get('Target_HDP')): p += unit_map.get(row['Target_HDP'], 0)
                if pd.notna(row.get('Target_OU')): p += unit_map.get(row['Target_OU'], 0)
                profits.append(p)
                
            cumulative_profit = np.cumsum(profits)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=cumulative_profit, mode='lines', fill='tozeroy', 
                                      line=dict(color='#22c55e', width=3), 
                                      fillcolor='rgba(34, 197, 94, 0.2)'))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6f2ea', yaxis_title="Units", xaxis_title="จำนวนบิล (Match)", 
                               margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.info("ยังไม่มีข้อมูลสถิติที่บันทึกผลสกอร์เพียงพอสำหรับสร้างกราฟ (กราฟจะปรากฏเมื่อมีการกรอกผลสกอร์)")

# ----------------------------------------
# หัวข้อหลัก (Tabs)
# ----------------------------------------
st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI (v5.0 The Ultimate Hybrid)</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>อัปเกรด Ensemble AI โหวต 3 ระดับ พร้อมระบบคำนวณเงินเดิมพันอัตโนมัติ, Ranking และ Confidence Indicator</p>", unsafe_allow_html=True)

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
    panel_header("📡 สัญญาณการทำนาย (AI Predictions)", "แมตช์ที่รอผลสกอร์ — บันทึกผลแล้วการ์ดจะหายไปเอง")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("🔄 รีเฟรชข้อมูล", key="refresh_signals", on_click=st.cache_data.clear)
        if AI["status"] != "ok":
            st.warning(AI["msg"])
            if "trace" in AI:
                with st.expander("🛠️ กางดู Error Log ที่นี่ (ส่งข้อมูลนี้ให้ผู้พัฒนา)"): 
                    st.code(AI["trace"], language="python")
        else:
            df_view = _visible_matches(AI["df_predict"]) if len(AI["df_predict"]) > 0 else AI["df_predict"]
            if len(df_view) > 0:
                render_monitor_header(df_view)
                render_prediction_cards(df_view, workbook)
            else:
                st.success("🎉 ไม่มีคู่ค้างที่ยังไม่ลงผลสกอร์ ทุกคู่ใน Google Sheets บันทึกผลครบถ้วนแล้ว!")

with tab3:
    panel_header("📊 Dashboard", "ความแม่นยำของโมเดล V5.0 และสถานะการเรียนรู้")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("🔄 รีเฟรชข้อมูล", key="refresh_dashboard", on_click=st.cache_data.clear)
        if AI["status"] != "ok":
            st.warning(AI["msg"])
            if "trace" in AI:
                with st.expander("🛠️ กางดู Error Log ที่นี่ (ส่งข้อมูลนี้ให้ผู้พัฒนา)"): 
                    st.code(AI["trace"], language="python")
        else:
            render_dashboard(AI)
