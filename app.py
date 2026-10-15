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
CYAN = "#0ea5e9"
CYAN_SOFT = "#7dd3fc"

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

/* Page heading */
.page-title { font-size: 1.9rem; font-weight: 600; margin: 0; color: #e6f2ea; }
.page-sub { color: #7f9a8b; margin: 4px 0 22px 0; font-size: 0.95rem; }

/* KPI strip */
.kpi-wrap { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 22px; }
.kpi {
    background: #0c130f; border: 1px solid #1b2a22; border-radius: 16px; padding: 22px 26px;
}
.kpi.hero {
    background: linear-gradient(160deg, #0f2a1b 0%, #0c130f 70%);
    border-color: #1f6b3d; box-shadow: 0 0 40px -12px rgba(34,197,94,0.35);
}
.kpi.hero-cyan {
    background: linear-gradient(160deg, #082f49 0%, #0c130f 70%);
    border-color: #0284c7; box-shadow: 0 0 40px -12px rgba(14,165,233,0.35);
}
.kpi-label { color: #7f9a8b; font-size: 0.9rem; margin-bottom: 6px; }
.kpi-value { font-size: 2.6rem; font-weight: 600; line-height: 1.15; color: #e6f2ea; }
.kpi.hero .kpi-value { font-size: 3.6rem; color: #4ade80; }
.kpi.hero-cyan .kpi-value { font-size: 3.6rem; color: #38bdf8; }
.kpi-unit { font-size: 1rem; color: #7f9a8b; font-weight: 400; margin-left: 4px; }
.kpi-bar { height: 6px; background: #17241d; border-radius: 6px; margin-top: 14px; overflow: hidden; }
.kpi-bar > div { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; }
.kpi.hero-cyan .kpi-bar > div { background: linear-gradient(90deg, #0284c7, #38bdf8); }

/* Panels */
.panel-title { font-size: 1.05rem; font-weight: 500; color: #e6f2ea; margin: 0 0 2px 0; }
.panel-sub { font-size: 0.85rem; color: #7f9a8b; margin: 0 0 6px 0; }

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

@media (max-width: 900px) {
    .kpi-wrap { grid-template-columns: 1fr; }
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
            raise FileNotFoundError('ไม่พบไฟล์ credentials.json')
        client = gspread.service_account(filename=creds_path)
    return client.open("ข้อมูลราคาบอลสกัดจากภาพ")

try:
    workbook = init_gspread()
except Exception as e:
    st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
    workbook = None

model = genai.GenerativeModel('gemini-3.7-flash')

WATER_IDX = [6, 8, 10, 11]
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

def clean_raw_data(raw_text):
    text = raw_text.replace('`', '').replace('\n', ' ').strip()
    raw_data = [item.strip().replace('−', '-') for item in text.split(',')]
    if not 12 <= len(raw_data) <= 13:
        raise ValueError(f"AI คืนค่ามา {len(raw_data)} ช่อง (ต้องเป็น 12-13 ช่อง)")
    m = re.search(r'\b(MY|HK|ID|EU)\b', raw_data[12].upper()) if len(raw_data) > 12 else None
    ai_fmt = m.group(1) if m else "UNK"
    row_data = raw_data[:2] + [parse_cell(x) for x in raw_data[2:12]]
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

FORM5_HEADERS = [
    "H_WinRate_Last5", "H_GoalsFor_Last5", "H_GoalsAgainst_Last5",
    "A_WinRate_Last5", "A_GoalsFor_Last5", "A_GoalsAgainst_Last5",
]

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
    requests.append(update(target_row - 1, 0, [typed(v) for v in row], "userEnteredValue"))
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
    for i in range
