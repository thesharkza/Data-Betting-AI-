"""Form5 input upgrade. See README_TH.md for schema, installation and limitations."""
import streamlit as st
import gspread
from google import genai
from google.genai import types
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import logging
from form_data import (
    BASE_HEADERS, FORM_HEADERS, FORM_SCOPES, ENTRY_MODES, TeamForm, MatchMeta,
    ValidationError, SchemaConflict, DuplicateFixture, validate_quote,
    parse_local_datetime, prepare_headers, save_match,
)
logger = logging.getLogger(__name__)
from PIL import Image
import io
import os
import re
import pandas as pd
import plotly.graph_objects as go
import gc
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as esc

# ----------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP", page_icon="⚽", layout="wide")

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
def setting(name, default=None):
    """Settings read in the main thread; secrets are never written to a file."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value if value is not None else os.environ.get(name, default)

GEMINI_API_KEY = setting("GEMINI_API_KEY")
# Original model identifier retained as a default, not claimed to be valid for every account.
GEMINI_MODEL = setting("GEMINI_MODEL", "gemini-3.7-flash")
DATA_WORKSHEET = setting("DATA_WORKSHEET", "DATA")
DATA_TIMEZONE = setting("DATA_TIMEZONE", "Asia/Bangkok")
FORMULA_MAX_ROW_CONFIG = int(setting("FORMULA_MAX_ROW", "2000"))
OWNER = st.session_state.setdefault("draft_owner", uuid.uuid4().hex)

@st.cache_resource
def init_gspread():
    try:
        secret = dict(st.secrets["gspread"]) if "gspread" in st.secrets else None
    except Exception:
        secret = None
    if secret:
        client = gspread.service_account_from_dict(secret)
    else:
        credentials = Path(setting("GOOGLE_APPLICATION_CREDENTIALS", str(Path(__file__).with_name("credentials.json"))))
        if not credentials.exists():
            raise FileNotFoundError('ตั้งค่า st.secrets["gspread"] หรือ credentials.json ก่อนบันทึกจริง')
        client = gspread.service_account(filename=str(credentials))
    sheet_id = setting("GOOGLE_SHEET_ID")
    return client.open_by_key(sheet_id) if sheet_id else client.open(setting("GOOGLE_SHEET_TITLE", "ข้อมูลราคาบอลสกัดจากภาพ"))

try:
    workbook = init_gspread()
    sheet_connection_error = None
except Exception as exc:
    workbook = None
    sheet_connection_error = str(exc)

model = GEMINI_MODEL

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
    """แปลงค่าน้ำจากรูปแบบ fmt เป็น Malay odds (ผ่านค่ากลางแบบ HK)"""
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
    """ใช้เมื่อ AI ไม่แน่ใจ: ทุกค่า > 1 = Decimal, มีค่าลบเกิน -1 = Indo, นอกนั้น = Malay"""
    if nums and all(v > 1 for v in nums):
        return "EU"
    if any(v < -1 for v in nums):
        return "ID"
    return "MY"

def parse_cell(v):
    """แปลงข้อความเป็นตัวเลข (ตัด o/u นำหน้า) ถ้าแปลงไม่ได้คืนเป็นข้อความเดิม"""
    cleaned = re.sub(r'^[oOuU\s]+', '', str(v).strip().replace('−', '-'))
    try:
        return float(cleaned) if '.' in cleaned else int(cleaned)
    except Exception:
        return cleaned

def clean_raw_data(raw_text):
    text = raw_text.replace('`', '').replace('\n', ' ').strip()
    raw_data = [item.strip().replace('−', '-') for item in text.split(',')]
    if not 12 <= len(raw_data) <= 13:
        raise ValueError(f"AI คืนค่ามา {len(raw_data)} ช่อง (ต้องเป็น 12-13 ช่อง) "
                         "อาจมีเครื่องหมายจุลภาคปนในชื่อทีม กรุณาอ่านภาพใหม่อีกครั้ง")
    m = re.search(r'\b(MY|HK|ID|EU)\b', raw_data[12].upper()) if len(raw_data) > 12 else None
    ai_fmt = m.group(1) if m else "UNK"
    row_data = raw_data[:2] + [parse_cell(x) for x in raw_data[2:12]]
    return row_data, ai_fmt

def convert_water_to_malay(row, user_fmt, ai_fmt):
    """แปลงค่าน้ำทั้ง 4 ช่องเป็น Malay -> คืน (แถวใหม่, ข้อความแจ้งผล, ข้อความเตือน)"""
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

# Replaced by form_data.save_match(): no contiguous A:BE writes.


def wait_for_result(ws, target_row, tries=10):
    """รอให้ชีตคำนวณคำแนะนำ (คอลัมน์ P) เสร็จ แทนการ sleep ตายตัว"""
    time.sleep(2)
    for _ in range(tries):
        r = ws.row_values(target_row)
        if len(r) > 15 and str(r[15]).strip():
            return r
        time.sleep(1)
    return ws.row_values(target_row)

# "current" = สูตรเดิม: ชนะครึ่งนับ 0.5 แต้ม แต่ยังนับเป็น 1 เกมในตัวหาร (ไม่นับเจ๊า)
# "symmetric" = ครึ่งชนะ/ครึ่งแพ้ นับ 0.5 เท่ากันทั้งสองฝั่ง: ชนะ / (ชนะ + แพ้)
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

# Money management is read from existing column T; never infer stake from a percentage in R.


# ----------------------------------------
# คิวอ่านภาพ: ยังไม่เขียน Google Sheets จนกว่าผู้ใช้ยืนยันในฟอร์ม
# ----------------------------------------
EXTRACT_PROMPT = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ,รหัสรูปแบบราคาน้ำ (ตอบรหัสเดียว: MY=มาเลเซีย มีทั้งบวกและลบ, HK=ฮ่องกง, ID=อินโดนีเซีย, EU=Decimal เช่น 1.90, UNK=ไม่แน่ใจ) ค่าน้ำทุกค่าให้คัดลอกตามที่เห็นในภาพทุกประการรวมเครื่องหมายลบ ห้ามแปลงค่าเอง โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"

class JobManager:
    """Drafts live in server memory, isolated by browser session. Not a durable queue."""
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.sheet_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.data_version = 0

    def create(self, name, img_bytes, user_fmt, owner):
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id, "owner": owner, "name": name,
                "status": "queued" if img_bytes else "needs_input",
                "msg": "รออ่านภาพ" if img_bytes else "กรอกราคาและฟอร์มก่อนบันทึก",
                "created": time.time(), "finished": None, "img": img_bytes,
                "user_fmt": user_fmt, "result": None, "note": None, "warn": None,
                "row_data": None if img_bytes else ["", ""] + [None] * 10,
            }
        return job_id

    def get(self, job_id, owner):
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job and job["owner"] == owner else None

    def update(self, job_id, owner, **kwargs):
        with self.lock:
            if job_id in self.jobs and self.jobs[job_id]["owner"] == owner:
                self.jobs[job_id].update(kwargs)

    def bump_version(self):
        with self.lock:
            self.data_version += 1

    def snapshot(self, owner):
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values() if j["owner"] == owner),
                          key=lambda j: j["created"], reverse=True)

    def clear_finished(self, owner):
        with self.lock:
            for key in [key for key, job in self.jobs.items()
                        if job["owner"] == owner and job["status"] in ("done", "error")]:
                self.jobs.pop(key)


@st.cache_resource
def get_job_manager():
    return JobManager()

JOBS = get_job_manager()

def validate_row(row):
    try:
        validate_quote(row)
        return None
    except ValidationError as exc:
        return str(exc)


def extract_row(img_bytes, user_fmt, model_name):
    """Only image price extraction. No form statistics are guessed or generated."""
    if not GEMINI_API_KEY or not model_name:
        raise RuntimeError("กรุณาตั้งค่า GEMINI_API_KEY และ GEMINI_MODEL หรือใช้การกรอกเอง")
    with Image.open(io.BytesIO(img_bytes)) as original:
        img = original.convert('RGB')
        img.thumbnail((1600, 1600))
        with io.BytesIO() as buf:
            img.save(buf, format='JPEG', quality=85)
            data = buf.getvalue()
    last_err = None
    for attempt in range(3):
        try:
            with genai.Client(api_key=GEMINI_API_KEY) as client:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_bytes(data=data, mime_type="image/jpeg"), EXTRACT_PROMPT],
                )
            if not response.text:
                raise ValueError("AI ไม่คืนข้อความราคา")
            row, ai_fmt = clean_raw_data(response.text.strip())
            return convert_water_to_malay(row, user_fmt, ai_fmt)
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"อ่านภาพไม่สำเร็จ: {last_err}")


def process_job(job_id, owner, model_name):
    """Worker reads the image only. No Sheets calls and no st.* calls here."""
    try:
        job = JOBS.get(job_id, owner)
        if not job:
            return
        JOBS.update(job_id, owner, status="reading", msg="AI กำลังอ่านราคาเท่านั้น ยังไม่บันทึก")
        row, note, warn = extract_row(job["img"], job["user_fmt"], model_name)
        problem = validate_row(row)
        if problem:
            warn = ((warn + " | ") if warn else "") + "โปรดแก้ราคาก่อนบันทึก: " + problem
        JOBS.update(job_id, owner, status="needs_input", row_data=row, note=note, warn=warn,
                    msg="อ่านแล้ว กรุณาตรวจราคาและกรอกฟอร์ม 5 นัด จากนั้นกดยืนยันบันทึก")
    except Exception as exc:
        logger.exception("Image extraction failed for draft %s", job_id)
        JOBS.update(job_id, owner, status="error", msg=str(exc), finished=time.time())
    finally:
        gc.collect()


def submit_job(name, img_bytes, user_fmt):
    job_id = JOBS.create(name, img_bytes, user_fmt, OWNER)
    JOBS.pool.submit(process_job, job_id, OWNER, model)
    return job_id


def retry_job(job_id):
    job = JOBS.get(job_id, OWNER)
    if job and job["status"] == "error" and job.get("img"):
        JOBS.update(job_id, OWNER, status="queued", msg="รออ่านภาพใหม่", result=None, finished=None)
        JOBS.pool.submit(process_job, job_id, OWNER, model)


STATUS_UI = {"needs_input": ("รอตรวจและกรอกฟอร์ม", "run"), "queued": ("รอคิว", "run"), "reading": ("กำลังอ่านภาพ", "run"),
             "saving": ("กำลังบันทึก", "run"), "done": ("บันทึกแล้ว", "ok"), "error": ("ไม่สำเร็จ", "err")}

def render_jobs():
    jobs = JOBS.snapshot(OWNER)
    if not jobs:
        st.caption("ยังไม่มีรายการ อัปโหลดภาพหรือกดกรอกราคาเองเพื่อเริ่มต้น")
        return
    active = sum(j["status"] in ("queued", "reading", "saving") for j in jobs)
    c1, c2 = st.columns([4, 1])
    with c1:
        panel_header("รายการของคุณ", f"กำลังประมวลผล {active} รายการ · รายการรอกรอกยังไม่บันทึกลงชีต")
    with c2:
        st.button("ล้างรายการที่จบแล้ว", on_click=JOBS.clear_finished, args=(OWNER,), key="clear_jobs")
    for job in jobs[:20]:
        label, css = STATUS_UI.get(job["status"], (job["status"], "run"))
        head = (f"<div class='job-head'><span class='job-name'>{esc(job['name'])}</span>"
                f"<span class='job-pill {css}'>{label}</span></div>")
        if job["status"] == "done":
            result = job["result"]
            action = "พบรายการเดิมที่บันทึกสำเร็จแล้ว" if result.get("already_saved") else (
                "แก้ไขนัดเดิมที่ยืนยัน" if result["overwritten"] else "รายการใหม่")
            body = (f"<div class='job-title'>{esc(result['home'])} vs {esc(result['away'])}</div>"
                    f"<div class='job-msg'>แถว {result['row']} · {action}</div>"
                    f"<div class='result-row'><span class='result-key'>P — สูตรเดิม</span>"
                    f"<span class='result-val'>{esc(str(result['rec']))}</span></div>"
                    f"<div class='result-row'><span class='result-key'>T — ขนาดจำลอง</span>"
                    f"<span class='result-val'>{esc(str(result['mm']))}</span></div>"
                    f"<div class='result-row'><span class='result-key'>R — สถิติกลุ่มกฎเดิม</span>"
                    f"<span class='result-val'>{esc(str(result['confidence']))}</span></div>")
        else:
            row = job.get("row_data") or []
            names = f"{esc(str(row[0]))} vs {esc(str(row[1]))}" if len(row) >= 2 else ""
            body = f"<div class='job-title'>{names}</div><div class='job-msg'>{esc(job['msg'])}</div>"
        notes = "".join(f"<div class='job-note'>{esc(str(job[k]))}</div>"
                        for k in ("note", "warn") if job.get(k))
        st.markdown(f"<div class='job-card {css}'>{head}{body}{notes}</div>", unsafe_allow_html=True)
        if job["status"] == "needs_input":
            if st.button("เปิดฟอร์มตรวจสอบและกรอกข้อมูล", key=f"review_{job['id']}"):
                st.session_state["selected_draft"] = job["id"]
                st.rerun()
        if job["status"] == "error" and job.get("img"):
            st.button("อ่านภาพใหม่", key=f"retry_{job['id']}", on_click=retry_job, args=(job["id"],))


if hasattr(st, "fragment"):
    render_jobs_live = st.fragment(run_every=2)(render_jobs)  # รีเฟรชสถานะเองทุก 2 วินาที
else:
    def render_jobs_live():
        st.button("🔄 อัปเดตสถานะ", key="refresh_jobs")
        render_jobs()

# ----------------------------------------
# Helper สำหรับกราฟ
# ----------------------------------------
def base_layout(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Prompt, sans-serif", color=MUTED, size=13),
        margin=dict(l=10, r=10, t=10, b=10), height=height,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=TEXT)),
    )
    return fig

# ----------------------------------------
# ประวัติอัตราชนะรายกฎ (ใช้ทำลูกศรขึ้น/ลง)
# ----------------------------------------
HIST_SHEET = "ประวัติสถิติ"

def update_rate_history(workbook, rates):
    """บันทึกอัตราชนะของแต่ละกฎลงชีต 'ประวัติสถิติ' เมื่อค่าเปลี่ยน
    คืน ({กฎ: [(เวลา, ค่า), ...]}, ข้อความ error หรือ None)"""
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
    # Dashboard intentionally retains original A:W definitions, not team-form win rates.
    ws = _workbook.worksheet(DATA_WORKSHEET)
    records = ws.get("A:W")
    if records:
        headers = list(records[0]) + list(BASE_HEADERS[len(records[0]):])
        rows = [(list(r) + [""] * 23)[:23] for r in records[1:] if any(str(v).strip() for v in r)]
        df = pd.DataFrame(rows, columns=headers[:23])
    else:
        df = pd.DataFrame(columns=BASE_HEADERS)
    try:
        df_stats = pd.DataFrame(_workbook.worksheet("สรุปสถิติ").get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        df_stats = pd.DataFrame()
    return df, df_stats


def trend_delta(h):
    return h[-1][1] - h[-2][1] if len(h) >= 2 else None

def trend_html(h):
    """ป้ายลูกศรบนการ์ด: ▲ เพิ่มขึ้น / ▼ ลดลง เทียบกับค่าก่อนหน้า"""
    if not h:
        return ""
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
# หัวข้อหลัก
# ----------------------------------------

def render_team_form(prefix, label):
    st.markdown(f"#### {label}")
    st.caption("ใช้ผล 90 นาทีรวมทดเจ็บ ไม่รวมต่อเวลาหรือดวลจุดโทษ; ถ้ามีไม่ครบ 5 นัดให้ระบุจำนวนจริง")
    matches = st.number_input("จำนวนเกมที่มีข้อมูล", min_value=1, max_value=5, value=5, step=1, key=prefix+"n")
    c1, c2, c3 = st.columns(3)
    with c1:
        wins = st.number_input("ชนะ", min_value=0, max_value=5, value=None, step=1, key=prefix+"w")
    with c2:
        draws = st.number_input("เสมอ", min_value=0, max_value=5, value=None, step=1, key=prefix+"d")
    with c3:
        losses = st.number_input("แพ้", min_value=0, max_value=5, value=None, step=1, key=prefix+"l")
    goals_for = st.number_input("ประตูได้รวมในชุดย้อนหลัง", min_value=0, value=None, step=1, key=prefix+"gf")
    goals_against = st.number_input("ประตูเสียรวมในชุดย้อนหลัง", min_value=0, value=None, step=1, key=prefix+"ga")
    st.caption("Win rate = ชนะ ÷ จำนวนเกม ระบบคำนวณให้หลังตรวจข้อมูล ไม่ต้องกรอกเปอร์เซ็นต์ซ้ำ")
    return dict(matches=matches, wins=wins, draws=draws, losses=losses,
                goals_for=goals_for, goals_against=goals_against, label=label)


def render_review(job_id):
    job = JOBS.get(job_id, OWNER)
    if not job or job["status"] != "needs_input":
        return
    st.markdown("### ตรวจราคาและกรอกข้อมูลก่อนบันทึก")
    st.info("ข้อมูลฟอร์มชุดนี้ใช้เตรียมพัฒนาโมเดล ยังไม่เปลี่ยนสูตร P2 หรือเพิ่มความแม่นยำให้อัตโนมัติ")
    if job.get("warn"):
        st.warning(job["warn"])
    if job.get("img"):
        with st.expander("ดูภาพราคาต้นฉบับ"):
            st.image(job["img"], width=700)
    prefix = "draft_" + job_id + "_"
    row = job.get("row_data") or ["", ""] + [None] * 10
    with st.form(prefix+"form", clear_on_submit=False, enter_to_submit=False):
        cols = st.columns(2)
        with cols[0]:
            home_name = st.text_input(BASE_HEADERS[0], value=str(row[0] or ""), key=prefix+"home")
        with cols[1]:
            away_name = st.text_input(BASE_HEADERS[1], value=str(row[1] or ""), key=prefix+"away")
        with st.expander("ตรวจสอบ/แก้ไขราคาที่อ่านได้ — ค่าน้ำทั้ง 4 ช่องเป็นมาเลย์", expanded=not bool(job.get("img"))):
            quote_numbers = []
            price_cols = st.columns(3)
            for index in range(2, 12):
                with price_cols[(index-2) % 3]:
                    value = row[index]
                    if not isinstance(value, (int, float)):
                        value = None
                    value = float(value) if value is not None else None
                    quote_numbers.append(st.number_input(BASE_HEADERS[index], value=value,
                                         step=0.01, format="%.2f", key=prefix+f"price{index}"))
        st.markdown("#### ขอบเขตและเวลาของข้อมูล")
        st.caption("วันเวลาใช้ ค.ศ. รูปแบบ YYYY-MM-DD HH:MM (24 ชั่วโมง) และเขตเวลาที่เลือกด้านล่าง")
        zones = ["Asia/Bangkok", "Asia/Singapore", "UTC"]
        if DATA_TIMEZONE not in zones:
            zones.append(DATA_TIMEZONE)
        zone = st.selectbox("เขตเวลา", zones, index=zones.index(DATA_TIMEZONE), key=prefix+"zone")
        c1, c2 = st.columns(2)
        with c1:
            competition = st.text_input("ลีก / รายการแข่งขัน", key=prefix+"competition")
            kickoff = st.text_input("เวลาเริ่มแข่งขัน", placeholder="YYYY-MM-DD HH:MM", key=prefix+"kickoff")
            odds_at = st.text_input("เวลาที่เก็บราคาในภาพ", placeholder="YYYY-MM-DD HH:MM", key=prefix+"odds_at")
        with c2:
            form_at = st.text_input("ข้อมูลฟอร์มเป็นข้อมูล ณ เวลาใด", placeholder="YYYY-MM-DD HH:MM", key=prefix+"form_at")
            scope_labels = {
                "ALL_COMPETITIONS": "รวมเหย้า/เยือน — ทุกรายการ",
                "LEAGUE_ONLY": "รวมเหย้า/เยือน — เฉพาะลีก",
                "VENUE_ALL_COMPETITIONS": "เจ้าบ้านใช้เกมเหย้า / ทีมเยือนใช้เกมเยือน — ทุกรายการ",
                "VENUE_LEAGUE_ONLY": "เจ้าบ้านใช้เกมเหย้า / ทีมเยือนใช้เกมเยือน — เฉพาะลีก",
            }
            scope = st.selectbox("ขอบเขต 5 นัดของทั้งสองทีม", list(FORM_SCOPES),
                                 format_func=lambda x: scope_labels[x], key=prefix+"scope")
            entry_mode = st.selectbox("ประเภทการบันทึก", list(ENTRY_MODES),
                                     format_func=lambda x: "ก่อนแข่ง (PREMATCH)" if x == "PREMATCH" else "ย้อนหลังจากหลักฐานก่อนแข่ง (HISTORICAL)",
                                     key=prefix+"entry_mode")
        form_source = st.text_input("แหล่งข้อมูลฟอร์ม / URL / ชื่อหลักฐาน (ระบุทั้งสองทีมถ้าคนละแหล่ง)", key=prefix+"source")
        st.caption("Form_AsOf ต้องไม่เกินเวลาเก็บราคา และราคา/ฟอร์มต้องอยู่ก่อนเวลาเริ่มแข่ง ไม่ใช้ผลของคู่นี้")
        hcol, acol = st.columns(2)
        with hcol:
            home_inputs = render_team_form(prefix+"h_", "เจ้าบ้าน")
        with acol:
            away_inputs = render_team_form(prefix+"a_", "ทีมเยือน")
        confirmed = st.checkbox("ยืนยันว่าตรวจราคาแล้ว และฟอร์มเป็นข้อมูลก่อนแข่งจริง ไม่รวมผลคู่นี้", key=prefix+"confirmed")
        allow_update = st.checkbox("อนุญาตแก้ไขเฉพาะนัดเดิมที่วันเวลา/ลีกตรงกัน และยังไม่มีผลสกอร์ (ไม่ใช่แค่ชื่อทีมตรง)", key=prefix+"allow_update")
        b1, b2 = st.columns(2)
        with b1:
            preview_pressed = st.form_submit_button("ตรวจข้อมูล / ดู Win rate ที่คำนวณ", use_container_width=True)
        with b2:
            save_pressed = st.form_submit_button("ยืนยันบันทึกลง Google Sheets", type="primary",
                                                 disabled=workbook is None, use_container_width=True)
    if not (preview_pressed or save_pressed):
        return
    try:
        clean_row = validate_quote([home_name, away_name] + quote_numbers)
        home_form = TeamForm.from_inputs(**home_inputs)
        away_form = TeamForm.from_inputs(**away_inputs)
        meta = MatchMeta(
            kickoff=parse_local_datetime(kickoff, zone, "Kickoff"),
            odds_at=parse_local_datetime(odds_at, zone, "Odds capture"),
            form_asof=parse_local_datetime(form_at, zone, "Form as-of"),
            competition=competition, scope=scope, source=form_source,
            entry_mode=entry_mode, image_name=job["name"],
        )
        meta.validate(datetime.now(timezone.utc))
        st.success("ข้อมูลผ่านการตรวจสอบรูปแบบและความสอดคล้อง — ยังไม่ได้ตรวจยืนยันแหล่งข้อมูลภายนอก")
        st.dataframe(pd.DataFrame([home_form.summary(), away_form.summary()],
                     index=["เจ้าบ้าน", "ทีมเยือน"]), use_container_width=True)
        if home_form.matches < 5 or away_form.matches < 5:
            st.warning("อย่างน้อยหนึ่งทีมมีข้อมูลไม่ครบ 5 นัด ระบบเก็บจำนวนจริง ไม่เติมข้อมูลที่ขาดเป็นศูนย์")
        if not save_pressed:
            st.caption("นี่คือตัวอย่างก่อนบันทึก ยังไม่มีการเขียนข้อมูลการแข่งขันลง Google Sheets")
            return
        if not confirmed:
            st.error("กรุณายืนยันว่าตรวจราคาและใช้ข้อมูลฟอร์มก่อนแข่งก่อนบันทึก")
            return
        if workbook is None:
            st.error("ยังไม่เชื่อมต่อ Google Sheets")
            return
        JOBS.update(job_id, OWNER, status="saving", msg="กำลังบันทึกข้อมูลที่ยืนยันแล้ว")
        with st.spinner("บันทึก A:L และ X:BE โดยไม่เขียนทับ M:W..."):
            with JOBS.sheet_lock:
                ws = workbook.worksheet(DATA_WORKSHEET)
                saved = save_match(ws, clean_row, home_form, away_form, meta, job_id,
                                   allow_update=allow_update, max_formula_row=FORMULA_MAX_ROW_CONFIG)
        # Persist receipt in memory before polling formulas; a polling error is not a save error.
        result = {"home": clean_row[0], "away": clean_row[1], "row": saved.row,
                  "overwritten": saved.overwritten, "already_saved": saved.already_saved,
                  "rec": "บันทึกแล้ว / รอสูตรชีตเดิม", "mm": "ดูคอลัมน์ T",
                  "confidence": "ดูคอลัมน์ R", "radar": "ดูคอลัมน์ S"}
        JOBS.update(job_id, OWNER, status="done", result=result, img=None, finished=time.time(),
                    msg="บันทึกสำเร็จ", warn=("พบชื่อทีมเดียวกันในแถวเก่าที่ไม่มีวันแข่ง จึงไม่เขียนทับแถวเก่า" if saved.legacy_same_names else None))
        JOBS.bump_version()
        try:
            updated = wait_for_result(ws, saved.row, tries=4)
            get = lambda index, default: updated[index] if len(updated) > index and str(updated[index]).strip() else default
            result.update(rec=get(15, "รอสูตร P2"), confidence=get(17, "-"),
                          radar=get(18, "-"), mm=get(19, "ยังไม่มีค่าจาก T"))
            JOBS.update(job_id, OWNER, result=result)
        except Exception:
            logger.exception("Row saved, but reading formula output failed")
            JOBS.update(job_id, OWNER, warn="ข้อมูลบันทึกสำเร็จ แต่ยังอ่านผลสูตรไม่ได้ โปรดตรวจในชีต ไม่ต้องบันทึกซ้ำ")
        st.rerun()
    except DuplicateFixture as exc:
        JOBS.update(job_id, OWNER, status="needs_input")
        st.warning(f"พบข้อมูลนัดเดียวกันที่แถว {exc.row} ยังไม่เขียนทับ หากต้องการแก้ไขให้ตรวจและติ๊กยืนยันแก้ไขนัดเดิม")
    except ValidationError as exc:
        JOBS.update(job_id, OWNER, status="needs_input")
        st.error(f"ยังไม่บันทึก: {exc}")
    except Exception as exc:
        JOBS.update(job_id, OWNER, status="needs_input")
        logger.exception("Save failed or response was not received")
        st.error(f"ยังยืนยันผลบันทึกไม่ได้: {exc} — ตรวจชีตก่อน แล้วกดบันทึกซ้ำในรายการเดิมเพื่อให้ระบบตรวจ Record_ID")


st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>อ่านราคา → ตรวจสอบและกรอกฟอร์ม 5 นัด → ยืนยันบันทึก | สูตร P2 เดิมยังไม่ใช้ฟอร์มใหม่</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "📊 Dashboard"])

# ----------------------------------------
# TAB 1
# ----------------------------------------
with tab1:
    st.subheader("อ่านราคา และกรอกฟอร์มก่อนบันทึก")
    st.write("อัปโหลดได้หลายรูป AI อ่านเฉพาะราคา จากนั้นเปิดแต่ละรายการเพื่อกรอกข้อมูลและกดยืนยัน")
    st.caption("รายการรอกรอกอยู่ในหน่วยความจำของ session ไม่ใช่ข้อมูลที่บันทึกแล้ว การรีสตาร์ตเซิร์ฟเวอร์หรือเปิด session ใหม่อาจทำให้รายการรอกรอกหาย")
    if workbook is None:
        st.warning(f"ยังไม่เชื่อมต่อ Google Sheets: {sheet_connection_error} — ทดลองกรอกและตรวจข้อมูลได้ แต่ปุ่มบันทึกถูกปิด")
    if not GEMINI_API_KEY:
        st.info("ยังไม่มี GEMINI_API_KEY ใช้ปุ่มกรอกราคาเองได้ หรือกำหนด Secrets เพื่ออ่านภาพ")
    with st.expander("เตรียมคอลัมน์ใหม่ X:BE (สำหรับชีต DATA เดิม A:W เท่านั้น)"):
        st.write("เพิ่มต่อท้าย W ไม่แทรกกลางตาราง และไม่ใช้กับชีต DATA_V2 ที่มีคอลัมน์ช่วยอยู่แล้ว")
        st.caption("คำสั่งนี้ตรวจหัวตารางเดิมและจะหยุดถ้า X:BE มีข้อมูลหรือสูตรอื่น ไม่ลบข้อมูลเดิม")
        backup_confirm = st.checkbox("สำรอง Google Sheets แล้ว และตรวจว่าชีตปลายทางถูกต้อง", key="schema_backup")
        d1, d2 = st.columns(2)
        with d1:
            inspect_schema = st.button("ตรวจโครงสร้างเท่านั้น", disabled=workbook is None)
        with d2:
            setup_schema = st.button("เพิ่ม / ตรวจหัวคอลัมน์ X:BE", disabled=workbook is None or not backup_confirm)
        if inspect_schema or setup_schema:
            try:
                with JOBS.sheet_lock:
                    plan = prepare_headers(workbook.worksheet(DATA_WORKSHEET), dry_run=not setup_schema)
                st.success("ตรวจผ่าน (ยังไม่เขียน)" if not setup_schema else "เตรียมหัวคอลัมน์สำเร็จ โดยไม่เปลี่ยน A:W")
                st.caption(f"ปลายทาง {DATA_WORKSHEET}!X1:BE1 · จำนวนคำสั่ง {len(plan['requests'])}")
            except Exception as exc:
                st.error(f"หยุดโดยไม่ทับข้อมูลเดิม: {exc}")
    fmt_choice = st.selectbox("รูปแบบค่าน้ำในภาพ", list(FMT_CHOICES.keys()),
                             help="ราคาที่ตรวจในฟอร์มจะเป็นมาเลย์หลังแปลง ควรตรวจภาพต้นฉบับทุกครั้ง")
    if st.button("กรอกราคาเอง / ไม่ใช้ภาพ", key="manual_draft"):
        draft_id = JOBS.create("กรอกเอง", None, "MY", OWNER)
        st.session_state["selected_draft"] = draft_id
        st.rerun()
    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกภาพราคาบอล", type=['png', 'jpg', 'jpeg'],
                                      accept_multiple_files=True, key=f"uploader_{up_n}", disabled=not bool(GEMINI_API_KEY))
    if uploaded_files:
        for uploaded in uploaded_files:
            submit_job(uploaded.name, uploaded.getvalue(), FMT_CHOICES[fmt_choice])
        st.session_state["up_n"] = up_n + 1
        st.rerun()
    render_jobs_live()
    ready = [j for j in JOBS.snapshot(OWNER) if j["status"] == "needs_input"]
    if ready:
        ready_ids = [j["id"] for j in ready]
        preferred = st.session_state.get("selected_draft")
        if preferred not in ready_ids:
            preferred = ready_ids[0]
        labels = {j["id"]: f"{j['name']} | {(j.get('row_data') or [''])[0]} | {j['id'][:6]}" for j in ready}
        selected = st.selectbox("เลือกคู่ที่จะกรอก", ready_ids, index=ready_ids.index(preferred),
                                format_func=lambda value: labels[value], key="review_selection_" + preferred)
        render_review(selected)
    st.caption("Win rate ทีมย้อนหลังใน X/AA เป็นคนละค่ากับ Win rate ผลกลุ่มกฎบน Dashboard เดิม")

# Original Dashboard preserved; it continues to use the original A:W and rule summaries.
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

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
