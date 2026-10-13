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

@media (max-width: 900px) { .kpi-wrap { grid-template-columns: 1fr; } }
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

def clean_raw_data(raw_text):
    text = raw_text.replace('`', '').replace('\n', ' ').strip()
    raw_data = [item.strip().replace('−', '-') for item in text.split(',')]
    if not 12 <= len(raw_data) <= 13:
        raise ValueError(f"AI คืนค่ามา {len(raw_data)} ช่อง (ต้องเป็น 12-13 ช่อง) อาจมีเครื่องหมายจุลภาคปนในชื่อทีม กรุณาอ่านภาพใหม่อีกครั้ง")
    m = re.search(r'\b(MY|HK|ID|EU)\b', raw_data[12].upper()) if len(raw_data) > 12 else None
    ai_fmt = m.group(1) if m else "UNK"
    row_data = raw_data[:2] + [parse_cell(x) for x in raw_data[2:12]]
    return row_data, ai_fmt

def convert_water_to_malay(row, user_fmt, ai_fmt):
    new_row = list(row)
    idx = [i for i in WATER_IDX if i < len(new_row) and isinstance(new_row[i], (int, float))]
    if not idx:
        return new_row, "ℹ️ ไม่พบค่าน้ำที่เป็นตัวเลข จึงไม่ได้แปลงราคา", "ตรวจสอบค่าน้ำในชีตอีกครั้ง"

    fmt, source = (user_fmt, "คุณเลือกเอง") if user_fmt else (ai_fmt, "AI ตรวจพบ")
    if fmt not in FMT_LABELS:
        fmt, source = guess_format([new_row[i] for i in idx]), "ระบบประเมินเอง เพราะ AI ไม่แน่ใจ"

    changes = []
    for i in idx:
        old = new_row[i]
        new_row[i] = to_malay(old, fmt)
        changes.append(f"{old:g} → {new_row[i]:g}")

    if fmt == "MY":
        note = f"✅ ค่าน้ำเป็นแบบมาเลเซียอยู่แล้ว ({source})"
    else:
        note = f"🔄 แปลงค่าน้ำจาก{FMT_LABELS[fmt]} → มาเลเซีย ({source}): " + "  |  ".join(changes)

    warn = None
    if fmt == "EU" and any(row[i] <= 1 for i in idx):
        warn = "⚠️ ระบุเป็นรูปแบบยุโรป (Decimal) แต่มีค่าน้ำ ≤ 1 ซึ่งไม่เข้าเงื่อนไข ควรเลือกรูปแบบราคาให้ตรงกับภาพ"
    elif any(abs(new_row[i]) > 1 for i in idx):
        warn = "⚠️ ค่าน้ำหลังแปลงเกิน ±1 ซึ่งผิดปกติสำหรับระบบมาเลเซีย กรุณาเลือกรูปแบบราคาให้ตรงกับภาพแล้วอัปโหลดใหม่"
    elif source.startswith("ระบบประเมิน"):
        warn = "⚠️ AI ไม่แน่ใจรูปแบบราคา ระบบจึงเดาให้ ควรเช็กค่าน้ำที่บันทึกอีกครั้ง"
    return new_row, note, warn

FORM5_HEADERS = [
    "H_WinRate_Last5", "H_GoalsFor_Last5", "H_GoalsAgainst_Last5",
    "A_WinRate_Last5", "A_GoalsFor_Last5", "A_GoalsAgainst_Last5",
]

def validate_form5(values):
    if len(values) != 6 or any(v is None or v == "" or isinstance(v, bool) for v in values):
        raise ValueError('กรุณากรอกข้อมูลให้ครบ 6 ช่อง; ช่องว่างไม่ใช่ 0')
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('ข้อมูลต้องเป็นตัวเลขที่ถูกต้อง') from exc
    if not all(isfinite(v) for v in nums):
        raise ValueError('ข้อมูลต้องเป็นตัวเลขที่ถูกต้อง')
    for i in (0, 3):
        if not 0 <= nums[i] <= 100 or abs(nums[i] / 20 - round(nums[i] / 20)) > 1e-8:
            raise ValueError('Win rate ของ 5 นัดใช้ 0, 20, 40, 60, 80 หรือ 100%')
    for i in (1, 2, 4, 5):
        if nums[i] < 0 or not nums[i].is_integer():
            raise ValueError('ประตูได้/เสียต้องเป็นจำนวนเต็มตั้งแต่ 0 ขึ้นไป')
    return [nums[0] / 100, int(nums[1]), int(nums[2]),
            nums[3] / 100, int(nums[4]), int(nums[5])]

def save_row_to_sheet(ws, row, form5):
    normalized = validate_form5(form5)
    if len(row) != 12:
        raise ValueError("Price data must have exactly 12 cells (A:L).")
    col_a = ws.col_values(1)
    col_b = ws.col_values(2)
    last_row = max((i + 1 for i, v in enumerate(col_a) if str(v).strip()), default=1)
    new_home, new_away = str(row[0]).strip(), str(row[1]).strip()
    if not new_home or not new_away:
        raise ValueError("Both team names are required.")
    target_row = None
    for i in range(1, len(col_a)):
        away = str(col_b[i]).strip() if i < len(col_b) else ""
        if str(col_a[i]).strip() == new_home and away == new_away:
            target_row = i + 1
            break
    overwritten = target_row is not None
    if not overwritten:
        target_row = max(2, last_row + 1)

    all_headers = ws.row_values(1)
    headers = (all_headers + [""] * 29)[23:29]
    for i, (old, expected) in enumerate(zip(headers, FORM5_HEADERS)):
        if str(old).strip() not in ("", expected):
            raise ValueError(f"X:AC header conflict (column {24+i}): {old!r}; expected {expected!r}. Nothing saved.")
    unnamed = [i for i, h in enumerate(headers) if not str(h).strip()]
    if unnamed and ws.col_count >= 24:
        last_col = min(ws.col_count, 29)
        last_letter = ('X', 'Y', 'Z', 'AA', 'AB', 'AC')[last_col - 24]
        existing = ws.get(f"X2:{last_letter}", value_render_option="FORMULA")
        if any(i < len(r) and str(r[i]).strip() for r in existing for i in unnamed):
            raise ValueError("X:AC contains data without matching headers. Review these columns first; nothing saved.")

    requests = []
    if ws.col_count < 29:
        requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "COLUMNS", "length": 29 - ws.col_count}})
    if target_row > ws.row_count:
        requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "ROWS", "length": target_row - ws.row_count}})

    def typed(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not isfinite(value):
                raise ValueError("Numeric price data must be finite.")
            return {"userEnteredValue": {"numberValue": value}}
        return {"userEnteredValue": {"stringValue": str(value)}}

    def update(start_row, start_col, cells, fields):
        return {"updateCells": {
            "start": {"sheetId": ws.id, "rowIndex": start_row, "columnIndex": start_col},
            "rows": [{"values": cells}], "fields": fields,
        }}

    if unnamed:
        requests.append(update(0, 23, [typed(h) for h in FORM5_HEADERS], "userEnteredValue"))
    requests.append(update(target_row - 1, 0, [typed(v) for v in row], "userEnteredValue"))
    form_cells = [typed(v) for v in normalized]
    for i, cell in enumerate(form_cells):
        cell["userEnteredFormat"] = {"numberFormat": {
            "type": "PERCENT" if i in (0, 3) else "NUMBER",
            "pattern": "0.0%" if i in (0, 3) else "0",
        }}
    requests.append(update(target_row - 1, 23, form_cells,
                           "userEnteredValue,userEnteredFormat.numberFormat"))
    ws.spreadsheet.batch_update({"requests": requests})
    return target_row, overwritten

def wait_for_result(ws, target_row, tries=10):
    time.sleep(2)
    for _ in range(tries):
        r = ws.row_values(target_row)
        if len(r) > 15 and str(r[15]).strip():
            return r
        time.sleep(1)
    return ws.row_values(target_row)

WIN_RATE_MODE = "current"

def calc_win_stats(df_comp):
    c = df_comp['ผลเปรียบเทียบ'].value_counts()
    wf, wh = int(c.get('ชนะเต็ม', 0)), int(c.get('ชนะครึ่ง', 0))
    lf, lh = int(c.get('แพ้เต็ม', 0)), int(c.get('แพ้ครึ่ง', 0))
    draws = int(c.get('เจ๊า', 0))
    total = len(df_comp)
    total_wins = wf + wh * 0.5
    denom = (total_wins + lf + lh * 0.5) if WIN_RATE_MODE == "symmetric" else (total - draws)
    win_rate = round(total_wins / denom * 100, 2) if denom > 0 else 0
    return total, total_wins, win_rate

def get_money_management(rec_text, conf_text):
    if "ข้าม" in str(rec_text):
        return "0 Unit (ข้าม ห้ามลงทุน) 🛑", rec_text

    match_pct = re.search(r'(\d+(\.\d+)?)%', str(conf_text))
    match_n = re.search(r'n=(\d+)', str(conf_text))

    if match_pct:
        pct_val = float(match_pct.group(1))
        n_val = int(match_n.group(1)) if match_n else 0

        if n_val < 5 and "SNIPER" in str(rec_text):
            return "⭐ 0.5 Unit (ช่วงทดสอบโมเดลใหม่ 🧪)", rec_text

        if pct_val < 55 and "SNIPER" in str(rec_text):
            modified_rec = f"⏸️ พักชั่วคราว: {rec_text} (สถิติตกเหลือ {pct_val}%)"
            return "0 Unit (รอสถิติฟื้นตัว 🛑)", modified_rec

        if pct_val >= 70: return "⭐⭐⭐⭐ 3 Units (Max Bet 🎯)", rec_text
        elif pct_val >= 60: return "⭐⭐⭐ 2 Units (High 🚀)", rec_text
        elif pct_val >= 55: return "⭐⭐ 1 Unit (Normal 👍)", rec_text
        else: return "⭐ 0.5 Unit (Low / Test 🧪)", rec_text
    else:
        return "⭐ 0.5 Unit (รอเก็บสถิติ ⏳)", rec_text

# ----------------------------------------
# ระบบงานเบื้องหลัง: อ่านภาพ + บันทึกชีตบนเซิร์ฟเวอร์
# ----------------------------------------
EXTRACT_PROMPT = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ,รหัสรูปแบบราคาน้ำ (ตอบรหัสเดียว: MY=มาเลเซีย มีทั้งบวกและลบ, HK=ฮ่องกง, ID=อินโดนีเซีย, EU=Decimal เช่น 1.90, UNK=ไม่แน่ใจ) ค่าน้ำทุกค่าให้คัดลอกตามที่เห็นในภาพทุกประการรวมเครื่องหมายลบ ห้ามแปลงค่าเอง โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"

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
            self.jobs[job_id] = {"id": job_id, "name": name, "status": "queued", "msg": "รอคิว...",
                                 "created": time.time(), "finished": None, "img": img_bytes,
                                 "user_fmt": user_fmt, "result": None, "note": None, "warn": None,
                                 "owner_id": owner_id, "row_data": None, "form5": None}
            if len(self.jobs) > 40:
                done = sorted((j for j in self.jobs.values() if j["status"] in ("done", "error")),
                              key=lambda j: j["created"])
                for j in done[:len(self.jobs) - 40]:
                    self.jobs.pop(j["id"], None)
        return job_id

    def get(self, job_id):
        with self.lock:
            j = self.jobs.get(job_id)
            return dict(j) if j else None

    def update(self, job_id, **kw):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kw)

    def bump_version(self):
        with self.lock:
            self.data_version += 1

    def snapshot(self):
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()), key=lambda j: j["created"], reverse=True)

    def clear_finished(self, owner_id):
        with self.lock:
            for k in [k for k, j in self.jobs.items() if j["status"] in ("done", "error") and j.get("owner_id") == owner_id]:
                self.jobs.pop(k)

@st.cache_resource
def get_job_manager():
    return JobManager()

JOBS = get_job_manager()
if "form5_owner_id" not in st.session_state:
    st.session_state["form5_owner_id"] = uuid.uuid4().hex

def validate_row(row):
    if len(row) < 12:
        return "ข้อมูลไม่ครบ 12 ช่อง"
    if not str(row[0]).strip() or not str(row[1]).strip():
        return "อ่านชื่อทีมไม่ได้"
    water = [row[i] for i in WATER_IDX if isinstance(row[i], (int, float))]
    if not water:
        return "อ่านค่าน้ำไม่ได้เลย"
    bad = [v for v in water if abs(v) > 1]
    if bad:
        return f"ค่าน้ำ {bad[0]:g} เกิน ±1 หลังแปลงเป็นมาเลเซีย (น่าจะรูปแบบราคาผิด)"
    return None

def extract_row(img_bytes, user_fmt, model):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    data = buf.getvalue()

    last_err = None
    for attempt in range(3):
        try:
            resp = model.generate_content([{"mime_type": "image/jpeg", "data": data}, EXTRACT_PROMPT])
            row, ai_fmt = clean_raw_data(resp.text.strip())
            return convert_water_to_malay(row, user_fmt, ai_fmt)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"อ่านภาพไม่สำเร็จหลังลอง 3 ครั้ง: {last_err}")

def process_job(job_id, workbook, model):
    try:
        job = JOBS.get(job_id)
        if not workbook:
            raise RuntimeError("ไม่สามารถเชื่อมต่อ Google Sheets ได้")

        JOBS.update(job_id, status="reading", msg="🤖 AI กำลังอ่านราคาจากภาพ...")
        row, note, warn = extract_row(job["img"], job["user_fmt"], model)

        problem = validate_row(row)
        if problem:
            JOBS.update(job_id, status="error", msg=f"ไม่บันทึก: {problem}", note=note, finished=time.time())
            return
        if any(not isinstance(row[i], (int, float)) for i in WATER_IDX):
            warn = ((warn + " ") if warn else "") + "⚠️ มีค่าน้ำที่อ่านไม่ได้บางช่อง ควรเช็กแถวนี้ในชีต"

        JOBS.update(job_id, status="ready", msg='อ่านราคาแล้ว กรอก Win rate และประตูได้/เสียก่อนบันทึก',
                    row_data=list(row), note=note, warn=warn, img=None)
    except Exception as e:
        JOBS.update(job_id, status="error", msg=f"{e}", finished=time.time())
    finally:
        gc.collect()

def save_confirmed_job(job_id, workbook):
    job = JOBS.get(job_id)
    if not job or job["status"] != "saving":
        return
    row = job["row_data"]
    try:
        if not workbook:
            raise RuntimeError("Google Sheets is not connected.")
        ws = workbook.sheet1
        with JOBS.sheet_lock:
            target_row, overwritten = save_row_to_sheet(ws, row, job["form5"])
    except Exception as exc:
        JOBS.update(job_id, status="error", msg=f"Save not confirmed: {exc}. Check the sheet before retrying.", finished=time.time())
        return

    JOBS.bump_version()
    warn = job.get("warn")
    try:
        updated = wait_for_result(ws, target_row)
    except Exception as exc:
        updated = []
        warn = ((warn + " ") if warn else "") + f"Saved row {target_row}; result read failed: {exc}"
    rec = updated[15] if len(updated) > 15 and str(updated[15]).strip() else "Waiting for sheet calculation"
    confidence = updated[17] if len(updated) > 17 else "-"
    radar = updated[18] if len(updated) > 18 else "-"
    mm_text, rec = get_money_management(rec, confidence)
    JOBS.update(job_id, status="done", msg="", img=None, finished=time.time(), warn=warn,
                result={"home": str(row[0]), "away": str(row[1]), "rec": rec, "mm": mm_text,
                        "confidence": confidence, "radar": radar,
                        "row": target_row, "overwritten": overwritten})

def submit_form5(job_id, values, owner_id):
    validate_form5(values)
    with JOBS.lock:
        job = JOBS.jobs.get(job_id)
        if not job or job.get("owner_id") != owner_id:
            raise ValueError("This form does not belong to this browser session.")
        if job["status"] not in ("ready", "error") or not job.get("row_data"):
            raise ValueError("This item is already being saved, saved, or not ready.")
        job.update(status="saving", msg="Saving prices and Form5 to Google Sheets...",
                   form5=list(values), finished=None)
    try:
        JOBS.pool.submit(save_confirmed_job, job_id, workbook)
    except RuntimeError:
        JOBS.update(job_id, status="ready", msg="Could not start saving. Please submit again.")
        raise

def submit_job(name, img_bytes, user_fmt):
    job_id = JOBS.create(name, img_bytes, user_fmt, st.session_state["form5_owner_id"])
    JOBS.pool.submit(process_job, job_id, workbook, model)

def retry_job(job_id):
    j = JOBS.get(job_id)
    if j and j.get("owner_id") == st.session_state["form5_owner_id"] and j["status"] == "error" and j["img"] and not j.get("row_data"):
        JOBS.update(job_id, status="queued", msg="รอคิว (ลองใหม่)", result=None, finished=None)
        JOBS.pool.submit(process_job, job_id, workbook, model)

STATUS_UI = {"ready": ('รอกรอกฟอร์ม 5 นัด', "run"), "queued": ("รอคิว", "run"), "reading": ("กำลังอ่านภาพ", "run"),
             "saving": ("กำลังบันทึก", "run"), "done": ("บันทึกแล้ว", "ok"), "error": ("ไม่สำเร็จ", "err")}

def render_jobs():
    jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if not jobs:
        st.caption('ยังไม่มีงาน — อัปโหลดรูป ระบบจะรอกรอกฟอร์มก่อนบันทึก')
        return

    active = sum(1 for j in jobs if j["status"] in ("queued", "reading", "saving"))
    c1, c2 = st.columns([4, 1])
    with c1:
        panel_header("งานล่าสุด", f"กำลังทำ {active} งาน · ปิดหน้านี้หรือสลับแอปได้ งานจะทำต่อบนเซิร์ฟเวอร์")
    with c2:
        st.button("🧹 ล้างรายการที่จบแล้ว", on_click=JOBS.clear_finished, args=(st.session_state["form5_owner_id"],), key="clear_jobs")

    for j in jobs:
        label, cls = STATUS_UI[j["status"]]
        head = (f"<div class='job-head'><span class='job-name'>{esc(j['name'])}</span>"
                f"<span class='job-pill {cls}'>{label}</span></div>")
        if j["status"] == "done":
            r = j["result"]
            where = "บันทึกทับแถวเดิม" if r["overwritten"] else "คู่ใหม่"
            body = (f"<div class='job-title'>{esc(r['home'])}<span class='vs'> vs </span>{esc(r['away'])}"
                    f"<span class='job-msg'> · แถว {r['row']} ({where})</span></div>"
                    f"<div class='result-row'><span class='result-key'>🎯 แนะนำลงทุน</span><span class='result-val hl'>{esc(str(r['rec']))}</span></div>"
                    f"<div class='result-row'><span class='result-key'>💰 Money Mgt</span><span class='result-val'>{esc(r['mm'])}</span></div>"
                    f"<div class='result-row'><span class='result-key'>📊 สถิติความเชื่อมั่น</span><span class='result-val'>{esc(str(r['confidence']))}</span></div>"
                    f"<div class='result-row'><span class='result-key'>🚨 เช็กราคา</span><span class='result-val'>{esc(str(r['radar']))}</span></div>")
        else:
            pair = j.get("row_data")
            title = (f"<div class='job-title'>{esc(str(pair[0]))} vs {esc(str(pair[1]))}</div>" if pair else "")
            body = title + f"<div class='job-msg' style='margin-top:8px'>{esc(j['msg'])}</div>"
        extra = ""
        if j.get("note"):
            extra += f"<div class='job-note'>{esc(j['note'])}</div>"
        if j.get("warn"):
            extra += f"<div class='job-note' style='color:#e8bb62'>{esc(j['warn'])}</div>"
        st.markdown(f"<div class='job-card {cls}'>{head}{body}{extra}</div>", unsafe_allow_html=True)
        if j.get("row_data") and j["status"] in ("ready", "error"):
            if st.button('กรอกฟอร์ม 5 นัด', key=f"open_form5_{j['id']}"):
                st.session_state["form5_editing"] = j["id"]
                st.rerun()
        if j["status"] == "error" and j.get("img") and not j.get("row_data"):
            st.button("🔁 ลองใหม่", key=f"retry_{j['id']}", on_click=retry_job, args=(j["id"],))

def render_form5_editor():
    job_id = st.session_state.get("form5_editing")
    job = JOBS.get(job_id) if job_id else None
    if not job or job.get("owner_id") != st.session_state["form5_owner_id"]:
        return
    if job["status"] not in ("ready", "error") or not job.get("row_data"):
        return
    row = job["row_data"]
    st.subheader('กรอกฟอร์ม 5 นัด')
    st.caption('Win rate: ชนะ 3 จาก 5 นัด กรอก 60; ประตูกรอกเป็นยอดรวม ไม่ใช่ค่าเฉลี่ย')
    with st.form(f"form5_{job_id}", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            st.write('เจ้าบ้าน' + " : " + str(row[0]))
            h_win = st.number_input('Win rate 5 นัดล่าสุด (%)', min_value=0.0, max_value=100.0,
                                    value=None, step=20.0, key=f"{job_id}_h_win")
            h_for = st.number_input('ประตูได้รวม 5 นัด', min_value=0, value=None, step=1, key=f"{job_id}_h_for")
            h_against = st.number_input('ประตูเสียรวม 5 นัด', min_value=0, value=None, step=1, key=f"{job_id}_h_against")
        with right:
            st.write('ทีมเยือน' + " : " + str(row[1]))
            a_win = st.number_input('Win rate 5 นัดล่าสุด (%)', min_value=0.0, max_value=100.0,
                                    value=None, step=20.0, key=f"{job_id}_a_win")
            a_for = st.number_input('ประตูได้รวม 5 นัด', min_value=0, value=None, step=1, key=f"{job_id}_a_for")
            a_against = st.number_input('ประตูเสียรวม 5 นัด', min_value=0, value=None, step=1, key=f"{job_id}_a_against")
        preview = st.form_submit_button('ตรวจข้อมูลก่อนบันทึก')
        save = st.form_submit_button('ยืนยันบันทึกลง Google Sheets', type="primary")
    if preview or save:
        values = [h_win, h_for, h_against, a_win, a_for, a_against]
        try:
            normalized = validate_form5(values)
            if save:
                submit_form5(job_id, values, st.session_state["form5_owner_id"])
                st.session_state.pop("form5_editing", None)
                st.rerun()
            else:
                st.success('ข้อมูลผ่านการตรวจ; ยังไม่บันทึกข้อมูลคู่แข่งขัน')
                st.write(f"{row[0]}: {normalized[0]:.0%}, GF={normalized[1]}, GA={normalized[2]} | "
                         f"{row[1]}: {normalized[3]:.0%}, GF={normalized[4]}, GA={normalized[5]}")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

if hasattr(st, "fragment"):
    render_jobs_live = st.fragment(run_every=2)(render_jobs)
else:
    def render_jobs_live():
        st.button("🔄 อัปเดตสถานะ", key="refresh_jobs")
        render_jobs()

def base_layout(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Prompt, sans-serif", color=MUTED, size=13),
        margin=dict(l=10, r=10, t=10, b=10), height=height,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=TEXT)),
    )
    return fig

HIST_SHEET = "ประวัติสถิติ"

def update_rate_history(workbook, rates):
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M")
    try:
        try:
            ws = workbook.worksheet(HIST_SHEET)
        except gspread.exceptions.WorksheetNotFound:
            ws = workbook.add_worksheet(title=HIST_SHEET, rows=1000, cols=3)
            ws.append_row(["เวลาบันทึก", "กฎ", "อัตราชนะ (%)"])

        history = {}
        for r in ws.get_all_values()[1:]:
            if len(r) >= 3:
                try:
                    history.setdefault(r[1].strip(), []).append((r[0], float(r[2].replace(',', '.'))))
                except ValueError:
                    pass

        new_rows = []
        for rule, v in rates.items():
            h = history.setdefault(rule, [])
            if not h or abs(h[-1][1] - v) >= 0.01:
                h.append((stamp, v))
                new_rows.append([stamp, rule, v])
        if new_rows:
            ws.append_rows(new_rows, value_input_option="USER_ENTERED")
        return history, None
    except Exception as e:
        return {r: [(stamp, v)] for r, v in rates.items()}, str(e)

@st.cache_data(ttl=60, show_spinner=False)
def cached_rate_history(_workbook, rates):
    return update_rate_history(_workbook, rates)

@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_data(_workbook, version=0):
    df = pd.DataFrame(_workbook.sheet1.get_all_records())
    df_stats = pd.DataFrame(_workbook.worksheet("สรุปสถิติ").get_all_records())
    return df, df_stats

def trend_delta(h):
    return h[-1][1] - h[-2][1] if len(h) >= 2 else None

def trend_html(h):
    if not h:
        return "<div class='trend-box'><span class='trend flat'>● เริ่มติดตาม</span></div>"
    d = trend_delta(h)
    if d is None:
        return "<div class='trend-box'><span class='trend flat'>● เริ่มติดตาม</span></div>"
    stamp, prev = h[-1][0], h[-2][1]
    cls, icon = ("up", "▲") if d > 0 else ("down", "▼")
    d_txt = f"{d:+.2f}".rstrip('0').rstrip('.')
    when = f"{stamp[8:10]}/{stamp[5:7]}"
    return (f"<div class='trend-box'><span class='trend {cls}'>{icon} {d_txt}</span>"
            f"<span class='trend-prev'>จาก {prev:g}% · {when}</span></div>")

def panel_header(title, sub):
    st.markdown(f"<p class='panel-title'>{title}</p><p class='panel-sub'>{sub}</p>", unsafe_allow_html=True)

# ----------------------------------------
# ฟังก์ชันส่วนเสริม: AI Machine Learning Predictions (ทายทั้งแฮนดิแคป และ สกอร์สูง-ต่ำ)
# ----------------------------------------
def render_ai_prediction_section(df):
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    panel_header("🤖 AI Machine Learning Predictions (แฮนดิแคป & สกอร์สูง-ต่ำ)", "ระบบวิเคราะห์ความน่าจะเป็นและฟันธงผลการแข่งขันเฉพาะแมตช์ที่ยังไม่ลงผลสกอร์")
    
    # 1. แปลงคอลัมน์ตัวเลขให้พร้อมใช้งาน
    numeric_cols = ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'ผล HDP' in df.columns and 'ผลสูง/ต่ำ' in df.columns:
        
        # Mapping ผล HDP (เจ้าบ้านชนะ HDP หรือไม่)
        def map_hdp_result(res):
            res = str(res).strip()
            if res == "" or res.lower() == "nan": return np.nan
            if 'เจ้าบ้านได้เต็ม' in res or 'เจ้าบ้านได้ครึ่ง' in res: return 1
            if 'เจ้าบ้านเสียเต็ม' in res or 'เจ้าบ้านเสียครึ่ง' in res: return 0
            return np.nan 

        # Mapping ผล สูง-ต่ำ (ออกสูงหรือไม่)
        def map_ou_result(res):
            res = str(res).strip()
            if res == "" or res.lower() == "nan": return np.nan
            if 'สูงได้เต็ม' in res or 'สูงได้ครึ่ง' in res: return 1
            if 'ต่ำได้เต็ม' in res or 'ต่ำได้ครึ่ง' in res: return 0
            return np.nan 

        df['Target_HDP'] = df['ผล HDP'].apply(map_hdp_result)
        df['Target_OU'] = df['ผลสูง/ต่ำ'].apply(map_ou_result)
        
        # ข้อมูลสำหรับ "ให้ AI ทำนาย" (เฉพาะแถวที่ยังไม่ลงผลสกอร์)
        if 'ผลสกอร์' in df.columns:
            df_predict = df[df['ผลสกอร์'].astype(str).str.strip().isin(["", "nan", "None"]) & df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()
        else:
            df_predict = df[df['ผล HDP'].astype(str).str.strip().isin(["", "nan", "None"])].copy()

        df_train = df.dropna(subset=['1X2 เหย้า', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'Target_HDP', 'Target_OU']).copy()

        if len(df_train) > 5 and len(df_predict) > 0:
            features = ['1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)']
            X_train = df_train[features].fillna(0)
            
            # --- 1. โมเดลทำนาย แฮนดิแคป ---
            model_hdp = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model_hdp.fit(X_train, df_train['Target_HDP'])
            
            # --- 2. โมเดลทำนาย สูง-ต่ำ ---
            model_ou = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model_ou.fit(X_train, df_train['Target_OU'])
            
            # ทำนายผลเฉพาะคู่ที่ยังไม่แข่ง
            X_pred = df_predict[features].fillna(0)
            
            # คำนวณ % โอกาส
            df_predict['AI_โอกาสเหย้าชนะHDP(%)'] = (model_hdp.predict_proba(X_pred)[:, 1] * 100).round(1)
            df_predict['AI_โอกาสออกสูง(%)'] = (model_ou.predict_proba(X_pred)[:, 1] * 100).round(1)
            
            # ฟันธง HDP
            def hdp_decision(prob):
                if prob >= 58: return "⭐⭐⭐ ไม้หนัก (แทงเหย้า)"
                elif prob >= 53: return "⭐⭐ ลงทุนเหย้า"
                elif prob <= 42: return "❌ รองทีมเยือน VIP"
                elif prob <= 47: return "🔻 รองทีมเยือน"
                else: return "⚠️ ข้าม (HDP)"
                
            # ฟันธง สูง-ต่ำ
            def ou_decision(prob):
                if prob >= 58: return "🔥 สูง (Over) มั่นใจสูง"
                elif prob >= 53: return "📈 น่าแทงสูง"
                elif prob <= 42: return "🛡️ ต่ำ (Under) มั่นใจสูง"
                elif prob <= 47: return "📉 น่าแทงต่ำ"
                else: return "⚠️ ข้าม (O/U)"

            df_predict['AI_ฟันธง_HDP'] = df_predict['AI_โอกาสเหย้าชนะHDP(%)'].apply(hdp_decision)
            df_predict['AI_ฟันธง_สูงต่ำ'] = df_predict['AI_โอกาสออกสูง(%)'].apply(ou_decision)
            
            # เลือกคอลัมน์ที่จะแสดงบนตาราง
            display_cols = [c for c in ['ทีมเหย้า', 'ทีมเยือน', 'แฮนดิแคป (เหย้า)', 'AI_โอกาสเหย้าชนะHDP(%)', 'AI_ฟันธง_HDP', 'โกลสูง/ต่ำ', 'AI_โอกาสออกสูง(%)', 'AI_ฟันธง_สูงต่ำ'] if c in df_predict.columns]
            
            st.markdown(f"พบแมตช์ที่ยังไม่ลงผลสกอร์ทั้งหมด **{len(df_predict)} คู่**:")
            st.dataframe(df_predict[display_cols], use_container_width=True)
            
        elif len(df_train) <= 5:
            st.info("ℹ️ ต้องการข้อมูลที่มีผลการแข่งขันในอดีต (ทั้ง HDP และ สูง-ต่ำ) อย่างน้อย 5 แถวขึ้นไป เพื่อให้ AI เริ่มประมวลผล")
        else:
            st.success("🎉 ไม่มีคู่ค้างที่ยังไม่ลงผลสกอร์ ทุกคู่ใน Google Sheets บันทึกผลครบถ้วนแล้ว!")
    else:
        st.warning("⚠️ ไม่พบคอลัมน์ 'ผล HDP' หรือ 'ผลสูง/ต่ำ' ใน Google Sheets สำหรับเทรน AI")

# ----------------------------------------
# หัวข้อหลัก
# ----------------------------------------
st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>สกัดราคาจากภาพ วิเคราะห์ ติดตามผล และพยากรณ์ด้วย Machine Learning ในที่เดียว</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "📊 Dashboard & AI"])

# ----------------------------------------
# TAB 1
# ----------------------------------------
with tab1:
    st.subheader("สกัดราคาบอล & วิเคราะห์ VIP")
    st.write('ลากรูปมาวางได้ (เลือกได้หลายรูป) อ่านราคาแล้วกรอก Win rate และประตูได้/เสีย 5 นัด จึงกดยืนยันบันทึก')

    fmt_choice = st.selectbox(
        "รูปแบบค่าน้ำในภาพ", list(FMT_CHOICES.keys()),
        help="ใช้กับรูปที่อัปโหลดต่อจากนี้ ระบบจะแปลงค่าน้ำเป็นแบบมาเลเซียก่อนบันทึกเสมอ เลือก 'อัตโนมัติ' ถ้าไม่แน่ใจ")

    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=True, key=f"uploader_{up_n}")
    if uploaded_files:
        if not workbook:
            st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
        else:
            for f in uploaded_files:
                submit_job(f.name, f.getvalue(), FMT_CHOICES[fmt_choice])
            st.session_state["up_n"] = up_n + 1
            st.rerun()

    render_jobs_live()
    render_form5_editor()

# ----------------------------------------
# TAB 2 : Dashboard & AI
# ----------------------------------------
with tab2:
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        try:
            if st.button("🔄 รีเฟรชข้อมูล"):
                st.cache_data.clear()
                st.rerun()

            df, df_stats = load_dashboard_data(workbook, JOBS.data_version)
            df_stats = df_stats.copy()
            if not df_stats.empty:
                df_stats.columns = [str(c).strip() for c in df_stats.columns]

            if not df.empty and 'ผลเปรียบเทียบ' in df.columns:
                df_comp = df[df['ผลเปรียบเทียบ'].astype(str).str.contains('ชนะ|แพ้|เจ๊า', na=False)].copy()
                total, total_wins, win_rate = calc_win_stats(df_comp)

                # --- 1. KPI ---
                st.markdown(f"""<div class="kpi-wrap">
<div class="kpi hero">
<div class="kpi-label">Win Rate รวมทั้งหมด</div>
<div class="kpi-value">{win_rate}<span class="kpi-unit">%</span></div>
<div class="kpi-bar"><div style="width:{min(win_rate, 100)}%"></div></div>
</div>
<div class="kpi">
<div class="kpi-label">แมตช์ที่จบแล้ว</div>
<div class="kpi-value">{total}<span class="kpi-unit">คู่</span></div>
</div>
<div class="kpi">
<div class="kpi-label">คะแนนชนะสะสม</div>
<div class="kpi-value">{total_wins:g}<span class="kpi-unit">แต้ม</span></div>
</div>
</div>""", unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns([1, 1.25], gap="large")

                # --- 2. Donut ---
                with col_chart1:
                    panel_header("สัดส่วนผลลัพธ์", "ผลของทุกคู่ที่จบแล้ว")
                    color_map = {
                        'ชนะเต็ม': GREEN, 'ชนะครึ่ง': GREEN_SOFT,
                        'เจ๊า': '#4b5f54', 'แพ้ครึ่ง': '#b8575b', 'แพ้เต็ม': RED
                    }
                    counts = df_comp['ผลเปรียบเทียบ'].value_counts()
                    order = [k for k in color_map if k in counts.index]
                    pie_fig = go.Figure(go.Pie(
                        labels=order, values=[counts[k] for k in order],
                        hole=0.72, sort=False, direction='clockwise',
                        marker=dict(colors=[color_map[k] for k in order], line=dict(color=BG, width=3)),
                        textinfo='none', hovertemplate='%{label}: %{value} คู่ (%{percent})<extra></extra>'
                    ))
                    base_layout(pie_fig, 380)
                    pie_fig.update_layout(
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5,
                                    font=dict(color=TEXT, size=13)),
                        annotations=[
                            dict(text=f"<b>{win_rate}%</b>", x=0.5, y=0.55, showarrow=False,
                                 font=dict(size=34, color=TEXT)),
                            dict(text="Win Rate", x=0.5, y=0.43, showarrow=False,
                                 font=dict(size=13, color=MUTED)),
                        ]
                    )
                    st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})

                # --- 3. Bar แยกตามกฎ ---
                with col_chart2:
                    panel_header("อัตราชนะแยกตามกฎ", "เขียว ≥ 60%  ·  เหลือง 55-60%  ·  แดง < 55%")

                    if not df_stats.empty and 'อัตราชนะ' in df_stats.columns:
                        plot_df = df_stats.copy()
                        rule_col = plot_df.columns[0]
                        plot_df['WinRate_Val'] = pd.to_numeric(
                            plot_df['อัตราชนะ'].astype(str).str.replace('%', '').str.strip(),
                            errors='coerce').fillna(0)
                        plot_df = plot_df[plot_df['WinRate_Val'] > 0].sort_values('WinRate_Val', ascending=True)

                        bar_colors = [GREEN if v >= 60 else AMBER if v >= 55 else RED for v in plot_df['WinRate_Val']]

                        fig_bar = go.Figure(go.Bar(
                            x=plot_df['WinRate_Val'], y=plot_df[rule_col], orientation='h',
                            marker=dict(color=bar_colors, cornerradius=6),
                            text=[f"{v:g}%" for v in plot_df['WinRate_Val']], textposition='outside',
                            textfont=dict(color=TEXT, size=13), cliponaxis=False,
                            hovertemplate='%{y}: %{x}%<extra></extra>'
                        ))
                        base_layout(fig_bar, max(380, 46 * len(plot_df) + 40))
                        fig_bar.update_layout(
                            xaxis=dict(range=[0, 108], visible=False),
                            yaxis=dict(showgrid=False, title="", tickfont=dict(size=13, color=TEXT)),
                            bargap=0.45, showlegend=False,
                            margin=dict(l=10, r=40, t=10, b=10),
                        )
                        fig_bar.add_vline(x=55, line=dict(color=LINE, width=1.5, dash='dot'))
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("ยังไม่มีข้อมูลสถิติเพียงพอสำหรับสร้างกราฟ")

            # --- 4. การ์ดสถิติรายกฎ ---
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            panel_header("สถิติรายกฎ", "เรียงจากอัตราชนะสูงสุด · ลูกศรเทียบกับค่าก่อนหน้าที่ระบบบันทึกไว้")

            if not df_stats.empty and 'อัตราชนะ' in df_stats.columns:
                tbl = df_stats.copy()
                tbl['_v'] = pd.to_numeric(
                    tbl['อัตราชนะ'].astype(str).str.replace('%', '').str.strip(),
                    errors='coerce').fillna(0)
                tbl = tbl.sort_values('_v', ascending=False)
                rule_col = df_stats.columns[0]
                extra_cols = [c for c in df_stats.columns if c not in (rule_col, 'อัตราชนะ')]

                rates = {str(r[rule_col]).strip(): round(float(r['_v']), 2)
                         for _, r in tbl.iterrows() if r['_v'] > 0}
                history, hist_err = cached_rate_history(workbook, rates)
                if hist_err:
                    st.warning(f"บันทึกประวัติสถิติไม่สำเร็จ จึงยังแสดงลูกศรไม่ได้: {hist_err}")

                deltas = [trend_delta(history.get(n, [])) for n in rates]
                n_up = sum(1 for d in deltas if d is not None and d > 0)
                n_down = sum(1 for d in deltas if d is not None and d < 0)
                if n_up or n_down:
                    chips = []
                    if n_up: chips.append(f"<span class='trend up'>▲ เพิ่มขึ้น {n_up} กฎ</span>")
                    if n_down: chips.append(f"<span class='trend down'>▼ ลดลง {n_down} กฎ</span>")
                    st.markdown(f"<div class='trend-sum'>{''.join(chips)}</div>", unsafe_allow_html=True)

                cards = ["<div class='rule-grid'>"]
                for _, row in tbl.iterrows():
                    v = row['_v']
                    name = str(row[rule_col]).strip()
                    k, tag = ("g", "แนะนำ") if v >= 60 else ("a", "ระวัง") if v >= 55 else ("r", "หลีกเลี่ยง")
                    meta = "".join(f"<span>{esc(str(c))}<b>{esc(str(row[c]))}</b></span>" for c in extra_cols
                                   if str(row[c]).strip() != "")
                    cards.append(
                        f"<div class='rule-card'>"
                        f"<div class='rule-top'><span class='rule-name'>{esc(name)}</span>"
                        f"<span class='rule-tag tag-{k}'>{tag}</span></div>"
                        f"<div class='rule-rate-row'><div class='rule-rate t-{k}'>{esc(str(row['อัตราชนะ']))}</div>"
                        f"{trend_html(history.get(name, []))}</div>"
                        f"<div class='rule-bar'><div class='b-{k}' style='width:{min(v, 100)}%'></div></div>"
                        f"<div class='rule-meta'>{meta}</div>"
                        f"</div>")
                cards.append("</div>")
                st.markdown("".join(cards), unsafe_allow_html=True)
            else:
                st.warning("ไม่พบข้อมูลสถิติในตาราง")

            # --- 5. เพิ่มการเรียกใช้งาน AI Machine Learning Predictions ตรงนี้ ---
            render_ai_prediction_section(df)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
