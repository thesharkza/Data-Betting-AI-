import streamlit as st
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP & AI", page_icon="⚽", layout="wide")

# ----------------------------------------
# THEME : Black & Emerald
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Prompt', 'Segoe UI', sans-serif !important;
}
.stApp { background: #050907; color: #e6f2ea; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

h1, h2, h3, h4 { color: #e6f2ea !important; letter-spacing: -0.01em; }
p, label, span { color: inherit; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1b2a22; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #7f9a8b; padding: 10px 18px;
    border-radius: 10px 10px 0 0; font-weight: 500;
}
.stTabs [aria-selected="true"] { color: #22c55e !important; background: #0c130f; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #22c55e; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: #22c55e; color: #03140a; border: none; font-weight: 600;
    border-radius: 10px; padding: 0.55rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #4ade80; color: #03140a; }

/* Uploader */
[data-testid="stFileUploader"] section {
    background: #0c130f; border: 1px dashed #2a4136; border-radius: 14px;
}

/* Alerts */
[data-testid="stAlert"] { background: #0c130f; border: 1px solid #1b2a22; border-radius: 12px; }

/* Page heading */
.page-title { font-size: 1.9rem; font-weight: 600; margin: 0; color: #e6f2ea; }
.page-sub { color: #7f9a8b; margin: 4px 0 22px 0; font-size: 0.95rem; }

/* KPI strip */
.kpi-wrap { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 16px; margin-bottom: 22px; }
.kpi {
    background: #0c130f; border: 1px solid #1b2a22; border-radius: 16px; padding: 22px 26px;
}
.kpi.hero {
    background: linear-gradient(160deg, #0f2a1b 0%, #0c130f 70%);
    border-color: #1f6b3d; box-shadow: 0 0 40px -12px rgba(34,197,94,0.35);
}
.kpi-label { color: #7f9a8b; font-size: 0.9rem; margin-bottom: 6px; }
.kpi-value { font-size: 2.6rem; font-weight: 600; line-height: 1.15; color: #e6f2ea; }
.kpi.hero .kpi-value { font-size: 3.6rem; color: #4ade80; }
.kpi-unit { font-size: 1rem; color: #7f9a8b; font-weight: 400; margin-left: 4px; }
.kpi-bar { height: 6px; background: #17241d; border-radius: 6px; margin-top: 14px; overflow: hidden; }
.kpi-bar > div { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; }

/* Panels */
.panel-title { font-size: 1.05rem; font-weight: 500; color: #e6f2ea; margin: 0 0 2px 0; }
.panel-sub { font-size: 0.85rem; color: #7f9a8b; margin: 0 0 6px 0; }

/* Rule cards */
.rule-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }
.rule-card { background: #0c130f; border: 1px solid #1b2a22; border-radius: 16px; padding: 20px 22px; }
.rule-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.rule-name { color: #e6f2ea; font-weight: 500; font-size: 1rem; }
.rule-tag { font-size: 0.78rem; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.rule-rate-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; margin: 10px 0 10px 0; }
.rule-rate { font-size: 2.4rem; font-weight: 600; line-height: 1.1; }
.trend-box { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; padding-bottom: 4px; }
.trend { font-size: 0.9rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.trend.up { color: #4ade80; background: rgba(34,197,94,0.15); }
.trend.down { color: #f06b70; background: rgba(229,72,77,0.14); }
.trend.flat { color: #7f9a8b; background: rgba(127,154,139,0.12); font-weight: 500; }
.trend-prev { font-size: 0.78rem; color: #7f9a8b; white-space: nowrap; }
.trend-sum { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 14px 0; }
.rule-bar { height: 6px; background: #17241d; border-radius: 6px; overflow: hidden; }
.rule-bar > div { height: 100%; border-radius: 6px; }
.rule-meta { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 14px; }
.rule-meta span { color: #7f9a8b; font-size: 0.85rem; }
.rule-meta b { color: #cfe3d6; font-weight: 500; margin-left: 4px; }
.t-g { color: #4ade80; } .t-a { color: #e8bb62; } .t-r { color: #f06b70; }
.b-g { background: #22c55e; } .b-a { background: #d9a441; } .b-r { background: #e5484d; }
.tag-g { background: rgba(34,197,94,0.15); color: #4ade80; }
.tag-a { background: rgba(217,164,65,0.14); color: #e8bb62; }
.tag-r { background: rgba(229,72,77,0.14); color: #f06b70; }

/* Background jobs */
.job-card { background: #0c130f; border: 1px solid #1b2a22; border-left: 4px solid #1b2a22; border-radius: 14px; padding: 16px 20px; margin-bottom: 10px; }
.job-card.ok { border-left-color: #22c55e; }
.job-card.err { border-left-color: #e5484d; }
.job-card.run { border-left-color: #d9a441; }
.job-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.job-name { color: #7f9a8b; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job-pill { font-size: 0.78rem; padding: 3px 10px; border-radius: 999px; white-space: nowrap; font-weight: 500; }
.job-pill.ok { color: #4ade80; background: rgba(34,197,94,0.15); }
.job-pill.err { color: #f06b70; background: rgba(229,72,77,0.14); }
.job-pill.run { color: #e8bb62; background: rgba(217,164,65,0.14); }
.job-title { color: #e6f2ea; font-size: 1.1rem; font-weight: 600; margin: 8px 0 4px 0; }
.job-msg { color: #a7c4b5; font-size: 0.9rem; }
.job-note { color: #7f9a8b; font-size: 0.8rem; margin-top: 8px; }

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    background: #0c130f; color: #cfe3d6; border: 1px solid #1b2a22; border-radius: 10px;
}
.stButton > button:not([kind="primary"]):hover { border-color: #22c55e; color: #4ade80; }

/* Result card (tab 1) */
.result-card {
    background: #0c130f; border: 1px solid #1b2a22; border-left: 4px solid #22c55e;
    border-radius: 14px; padding: 24px 28px;
}
.result-card h3 { margin: 0 0 12px 0; color: #e6f2ea; }
.result-card .vs { color: #7f9a8b; font-weight: 400; padding: 0 6px; }
.result-row { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid #111c16; }
.result-key { color: #7f9a8b; min-width: 170px; }
.result-val { color: #e6f2ea; font-weight: 500; }
.result-val.hl { color: #4ade80; }

/* AI Predictions */
.pred-summary { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 18px 0; }
.pred-chip { background: #0c130f; border: 1px solid #1b2a22; border-radius: 999px; padding: 6px 14px; font-size: 0.85rem; color: #a7c4b5; }
.pred-chip b { color: #e6f2ea; font-weight: 600; margin-left: 6px; }
.pred-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 16px; }
.pred-card {
    background: linear-gradient(180deg, #0e1712 0%, #0c130f 100%);
    border: 1px solid #1b2a22; border-radius: 18px; padding: 20px 22px;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.pred-card:hover { border-color: #1f6b3d; box-shadow: 0 0 30px -14px rgba(34,197,94,0.45); }
.pred-teams { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.pred-team { flex: 1; font-weight: 600; color: #e6f2ea; font-size: 1.02rem; line-height: 1.3; }
.pred-team.away { text-align: right; }
.pred-vs { font-size: 0.72rem; color: #7f9a8b; border: 1px solid #1b2a22; border-radius: 999px; padding: 2px 10px; }
.pred-sections { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.pred-box { background: #09100c; border: 1px solid #14211a; border-radius: 14px; padding: 14px 16px; }
.pred-box-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.pred-kind { color: #7f9a8b; font-size: 0.78rem; letter-spacing: 0.04em; }
.pred-line { color: #cfe3d6; font-size: 0.85rem; font-weight: 500; }
.pred-pct { font-size: 2rem; font-weight: 600; line-height: 1.1; margin: 8px 0 10px 0; }
.pred-pct small { font-size: 0.9rem; font-weight: 400; color: #7f9a8b; margin-left: 2px; }
.pred-track { position: relative; height: 6px; background: #17241d; border-radius: 6px; }
.pred-fill { height: 100%; border-radius: 6px; }
.pred-mid { position: absolute; left: 50%; top: -3px; width: 1px; height: 12px; background: #3a5547; }
.pred-verdict { margin-top: 14px; font-size: 0.82rem; padding: 5px 11px; border-radius: 10px; display: inline-block; font-weight: 500; }
.tone-gs { color: #4ade80; } .tone-gf { color: #86efac; } .tone-cs { color: #67e8f9; } .tone-cf { color: #a5f3fc; } .tone-skip { color: #7f9a8b; }
.fill-gs { background: linear-gradient(90deg, #16a34a, #4ade80); }
.fill-gf { background: linear-gradient(90deg, #15803d, #86efac); }
.fill-cs { background: linear-gradient(90deg, #0e7490, #67e8f9); }
.fill-cf { background: linear-gradient(90deg, #155e75, #a5f3fc); }
.fill-skip { background: #4b5f54; }
.bg-gs { background: rgba(34,197,94,0.16); color: #4ade80; }
.bg-gf { background: rgba(34,197,94,0.09); color: #86efac; }
.bg-cs { background: rgba(34,211,238,0.15); color: #67e8f9; }
.bg-cf { background: rgba(34,211,238,0.09); color: #a5f3fc; }
.bg-skip { background: rgba(127,154,139,0.12); color: #7f9a8b; }

@media (max-width: 900px) {
    .kpi-wrap { grid-template-columns: 1fr; }
    .pred-grid { grid-template-columns: 1fr; }
    .pred-sections { grid-template-columns: 1fr; }
}
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
    try:
        return "gspread" in st.secrets
    except Exception:
        return False

@st.cache_resource
def init_gspread():
    if _has_gspread_secret():
        client = gspread.service_account_from_dict(dict(st.secrets["gspread"]))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        creds_path = os.path.join(base_dir, 'credentials.json')
        if not os.path.exists(creds_path):
            raise FileNotFoundError('ไม่พบไฟล์ credentials.json ในโฟลเดอร์โครงการ และยังไม่ได้ตั้งค่า st.secrets["gspread"]')
        client = gspread.service_account(filename=creds_path)
    return client.open("ข้อมูลราคาบอลสกัดจากภาพ")

try:
    workbook = init_gspread()
except Exception as e:
    st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
    workbook = None

model = genai.GenerativeModel('gemini-3.7-flash')

# ----------------------------------------
# แปลงค่าน้ำเป็นระบบมาเลเซีย (Malay Odds) เสมอ
# ----------------------------------------
WATER_IDX = [6, 8, 10, 11]  # ค่าน้ำ HDP เหย้า / HDP เยือน / น้ำสูง / น้ำต่ำ
FMT_LABELS = {"MY": "มาเลเซีย", "HK": "ฮ่องกง", "ID": "อินโดนีเซีย", "EU": "ยุโรป (Decimal)"}
FMT_CHOICES = {
    "อัตโนมัติ (ให้ AI ตรวจจับ)": None,
    "มาเลเซีย (Malay)": "MY",
    "ฮ่องกง (HK)": "HK",
    "อินโดนีเซีย (Indo)": "ID",
    "ยุโรป (Decimal)": "EU",
}

def to_malay(x, fmt):
    x = float(x)
    if fmt == "MY":
        return round(x, 2)
    if fmt == "EU":
        hk = x - 1
    elif fmt == "ID":
        hk = x if x > 0 else (1 / abs(x) if x != 0 else 0)
    else:  # HK
        hk = x
    return round(hk, 2) if hk <= 1 else round(-1 / hk, 2)

def guess_format(nums):
    if nums and all(v > 1 for v in nums):
        return "EU"
    if any(v < -1 for v in nums):
        return "ID"
    return "MY"

def parse_cell(v):
    cleaned = re.sub(r'^[oOuU\s]+', '', str(v).strip().replace('−', '-'))
    try:
        return float(cleaned) if '.' in cleaned else int(cleaned)
    except Exception:
        return cleaned

def clean_raw
