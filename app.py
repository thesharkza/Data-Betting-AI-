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
.stButton > button[kind="primary"] { background: #22c55e; color: #03140a; border: none; font-weight: 600; border-radius: 10px; padding: 0.55rem 1.4rem; }
.stButton > button[kind="primary"]:hover { background: #4ade80; color: #03140a; }
[data-testid="stFileUploader"] section { background: #0c130f; border: 1px dashed #2a4136; border-radius: 14px; }
.page-title { font-size: 1.9rem; font-weight: 600; margin: 0; color: #e6f2ea; }
.page-sub { color: #7f9a8b; margin: 4px 0 22px 0; font-size: 0.95rem; }
.kpi-wrap { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 22px; }
.kpi { background: #0c130f; border: 1px solid #1b2a22; border-radius: 16px; padding: 22px 26px; }
.kpi.hero { background: linear-gradient(160deg, #0f2a1b 0%, #0c130f 70%); border-color: #1f6b3d; box-shadow: 0 0 40px -12px rgba(34,197,94,0.35); }
.kpi.hero-cyan { background: linear-gradient(160deg, #082f49 0%, #0c130f 70%); border-color: #0284c7; box-shadow: 0 0 40px -12px rgba(14,165,233,0.35); }
.kpi-label { color: #7f9a8b; font-size: 0.9rem; margin-bottom: 6px; }
.kpi-value { font-size: 2.6rem; font-weight: 600; line-height: 1.15; color: #e6f2ea; }
.kpi.hero .kpi-value { font-size: 3.6rem; color: #4ade80; }
.kpi.hero-cyan .kpi-value { font-size: 3.6rem; color: #38bdf8; }
.kpi-unit { font-size: 1rem; color: #7f9a8b; font-weight: 400; margin-left: 4px; }
.kpi-bar { height: 6px; background: #17241d; border-radius: 6px; margin-top: 14px; overflow: hidden; }
.kpi-bar > div { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; }
.kpi.hero-cyan .kpi-bar > div { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.panel-title { font-size: 1.05rem; font-weight: 500; color: #e6f2ea; margin: 0 0 2px 0; }
.panel-sub { font-size: 0.85rem; color: #7f9a8b; margin: 0 0 6px 0; }
.job-card { background: #0c130f; border: 1px solid #1b2a22; border-left: 4px solid #1b2a22; border-radius: 14px; padding: 16px 20px; margin-bottom: 10px; }
.job-card.ok { border-left-color: #22c55e; }
.job-card.err { border-left-color: #e5484d; }
.job-card.run { border-left-color: #d9a441; }
.job-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.job-name { color: #7f9a8b; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job-msg { color: #a7c4b5; font-size: 0.9rem; }
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

def render_jobs():
    jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if not jobs: return
    for j in jobs:
        cls = "ok" if j["status"] == "done" else "err" if j["status"] == "error" else "run"
        st.markdown(f"<div class='job-card {cls}'><div class='job-head'><span class='job-name'>{esc(j['name'])}</span></div><div class='job-msg'>{esc(j['msg'])}</div></div>", unsafe_allow_html=True)
        if j["status"] == "ready":
            if st.button('กรอกฟอร์ม 5 นัด', key=f"open_form5_{j['id']}"):
                st.session_state["form5_editing"] = j["id"]
                st.rerun()

def render_form5_editor():
    job_id = st.session_state.get("form5_editing")
    job = JOBS.get(job_id) if job_id else None
    if not job or job["status"] != "ready": return
    with st.form(f"form5_{job_id}"):
        st.write(f"ฟอร์ม 5 นัด: {job['row_data'][0]} vs {job['row_data'][1]}")
        h_win, h_for, h_ag = st.number_input('Win H', step=20.), st.number_input('GF H', step=1), st.number_input('GA H', step=1)
        a_win, a_for, a_ag = st.number_input('Win A', step=20.), st.number_input('GF A', step=1), st.number_input('GA A', step=1)
        if st.form_submit_button('บันทึก'):
            submit_form5(job_id, [h_win, h_for, h_ag, a_win, a_for, a_ag], st.session_state["form5_owner_id"])
            st.session_state.pop("form5_editing", None)
            st.rerun()

@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_data(_workbook, version=0):
    ws = get_ws(_workbook)
    return pd.DataFrame(ws.get_all_records())

def base_layout(fig, height=380):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Prompt, sans-serif", color=MUTED, size=13), margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig
def panel_header(title, sub): st.markdown(f"<p class='panel-title'>{title}</p><p class='panel-sub'>{sub}</p>", unsafe_allow_html=True)

# ----------------------------------------
# UI ฟันธง + ช่องกรอกผลสกอร์
# ----------------------------------------
def _pred_tone(text, kind):
    if "ข้าม" in text: return "skip"
    if kind == "hdp": return ("cs" if "VIP" in text else "cf") if "รอง" in text else ("gs" if "VIP" in text else "gf")
    return "cs" if "มั่นใจ" in text and "ต่ำ" in text else "cf" if "ต่ำ" in text else "gs" if "มั่นใจ" in text else "gf"

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
            st.markdown(f"""
            <div style='background: linear-gradient(180deg, #0e1712 0%, #0c130f 100%); border: 1px solid #1b2a22; border-radius: 18px; padding: 18px 22px; margin-bottom: 12px;'>
                <div style='font-size: 1.05rem; font-weight: 600; color: #e6f2ea; margin-bottom: 10px; text-align: center;'>
                    ⚽ {esc(home)} <span style='color: #7f9a8b; font-weight: 400; padding: 0 8px;'>VS</span> {esc(away)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown(f"""
                <div style='background: #09100c; border: 1px solid #14211a; border-radius: 14px; padding: 14px 16px;'>
                    <div style='display: flex; justify-content: space-between; font-size: 0.78rem; color: #7f9a8b;'>
                        <span>HANDICAP</span><span>เหย้า {hdp_line:+g}</span>
                    </div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #4ade80; margin: 6px 0;'>{hdp_prob:g}%</div>
                    <div style='font-size: 0.82rem; background: rgba(34,197,94,0.15); color: #4ade80; padding: 4px 10px; border-radius: 8px; display: inline-block; font-weight: 500;'>{esc(hdp_verdict)}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div style='background: #09100c; border: 1px solid #14211a; border-radius: 14px; padding: 14px 16px;'>
                    <div style='display: flex; justify-content: space-between; font-size: 0.78rem; color: #7f9a8b;'>
                        <span>OVER / UNDER</span><span>ไลน์ {ou_line:g}</span>
                    </div>
                    <div style='font-size: 1.8rem; font-weight: 600; color: #38bdf8; margin: 6px 0;'>{ou_prob:g}%</div>
                    <div style='font-size: 0.82rem; background: rgba(14,165,233,0.15); color: #38bdf8; padding: 4px 10px; border-radius: 8px; display: inline-block; font-weight: 500;'>{esc(ou_verdict)}</div>
                </div>
                """, unsafe_allow_html=True)

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
st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>สกัดราคาจากภาพ วิเคราะห์ ติดตามผล และพยากรณ์ด้วย Machine Learning แบบเรียลไทม์</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "🧠 AI Dashboard (สถิติ & ทำนาย)"])

with tab1:
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

                    st.markdown(f"""<div class="kpi-wrap">
                    <div class="kpi hero">
                        <div class="kpi-label">ความแม่นยำ AI (แฮนดิแคป)</div>
                        <div class="kpi-value">{hdp_acc}<span class="kpi-unit">%</span></div>
                        <div class="kpi-bar"><div style="width:{min(hdp_acc, 100)}%"></div></div>
                    </div>
                    <div class="kpi hero-cyan">
                        <div class="kpi-label">ความแม่นยำ AI (สูง-ต่ำ)</div>
                        <div class="kpi-value">{ou_acc}<span class="kpi-unit">%</span></div>
                        <div class="kpi-bar"><div style="width:{min(ou_acc, 100)}%"></div></div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-label">เรียนรู้จากข้อมูลผลลัพธ์แล้ว</div>
                        <div class="kpi-value">{total_train}<span class="kpi-unit">คู่</span></div>
                    </div>
                    </div>""", unsafe_allow_html=True)

                    def plot_ai_donut(correct, total, color_hex):
                        incorrect = total - correct
                        fig = go.Figure(go.Pie(
                            labels=['ทายถูก (Win)', 'ทายผิด/เจ๊า (Loss/Draw)'], values=[correct, incorrect],
                            hole=0.72, sort=False, direction='clockwise',
                            marker=dict(colors=[color_hex, RED], line=dict(color=BG, width=3)),
                            textinfo='none', hovertemplate='%{label}: %{value} คู่ (%{percent})<extra></extra>'
                        ))
                        base_layout(fig, 320)
                        acc = round((correct/total)*100, 1) if total > 0 else 0
                        fig.update_layout(
                            showlegend=True, legend=dict(orientation='h', yanchor='top', y=-0.05, xanchor='center', x=0.5, font=dict(color=TEXT, size=13)),
                            annotations=[dict(text=f"<b>{acc}%</b>", x=0.5, y=0.55, showarrow=False, font=dict(size=34, color=TEXT)), dict(text="Accuracy", x=0.5, y=0.43, showarrow=False, font=dict(size=13, color=MUTED))],
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        return fig

                    def plot_ai_feature_importance(model, feature_names, color_hex):
                        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(5)
                        df_imp['Feature'] = df_imp['Feature'].replace({
                            'Expected_Total_Goals': 'ประตูรวมที่คาดหวัง', 'OU_Goal_Diff': 'ส่วนต่างโกลที่คาด', 'Total_Goals_Conceded_Last5': 'เสียประตู 5 นัดรวม',
                            'H_GoalsFor_Last5': 'เหย้ายิงได้ 5 นัด', 'H_GoalsAgainst_Last5': 'เหย้าเสีย 5 นัด', 'A_GoalsFor_Last5': 'เยือนยิงได้ 5 นัด', 'A_GoalsAgainst_Last5': 'เยือนเสีย 5 นัด',
                            'H_WinRate_Last5': 'WinRate เหย้า 5 นัด', 'A_WinRate_Last5': 'WinRate เยือน 5 นัด', '1X2 เหย้า': 'ราคา 1X2 เหย้า', '1X2 เยือน': 'ราคา 1X2 เยือน',
                            'แฮนดิแคป (เหย้า)': 'แต้มต่อ HDP', 'ค่าน้ำ HDP (เหย้า)': 'ค่าน้ำ HDP', 'โกลสูง/ต่ำ': 'ไลน์สูง-ต่ำ', 'ค่าน้ำ (สูง)': 'ค่าน้ำสูง', 'ค่าน้ำ (ต่ำ)': 'ค่าน้ำต่ำ'
                        })
                        fig = go.Figure(go.Bar(
                            x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
                            marker=dict(color=color_hex, cornerradius=6),
                            text=[f"{v*100:.1f}%" for v in df_imp['Importance']], textposition='outside',
                            textfont=dict(color=TEXT, size=13), cliponaxis=False, hovertemplate='%{y}: น้ำหนัก %{x:.1%}<extra></extra>'
                        ))
                        base_layout(fig, 320)
                        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(showgrid=False, title="", tickfont=dict(size=13, color=TEXT)), bargap=0.45, showlegend=False, margin=dict(l=10, r=40, t=10, b=10))
                        return fig

                    col_hdp1, col_hdp2 = st.columns([1, 1.25], gap="large")
                    with col_hdp1:
                        panel_header("สัดส่วนความแม่นยำ (แฮนดิแคป)", "เทียบจากผลการเรียนรู้ประวัติศาสตร์ทั้งหมด")
                        st.plotly_chart(plot_ai_donut(hdp_correct, total_train, GREEN), use_container_width=True, config={'displayModeBar': False})
                    with col_hdp2:
                        panel_header("ปัจจัยที่มีผลต่อ HDP สูงสุด 5 อันดับ", "น้ำหนักความสำคัญที่ AI ใช้ตัดสินใจ")
                        st.plotly_chart(plot_ai_feature_importance(model_hdp, X_train.columns, GREEN), use_container_width=True, config={'displayModeBar': False})

                    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    
                    col_ou1, col_ou2 = st.columns([1, 1.25], gap="large")
                    with col_ou1:
                        panel_header("สัดส่วนความแม่นยำ (สูง-ต่ำ)", "เทียบจากผลการเรียนรู้ประวัติศาสตร์ทั้งหมด")
                        st.plotly_chart(plot_ai_donut(ou_correct, total_train, CYAN), use_container_width=True, config={'displayModeBar': False})
                    with col_ou2:
                        panel_header("ปัจจัยที่มีผลต่อสูง-ต่ำ สูงสุด 5 อันดับ", "น้ำหนักความสำคัญที่ AI ใช้ตัดสินใจ")
                        st.plotly_chart(plot_ai_feature_importance(model_ou, X_train.columns, CYAN), use_container_width=True, config={'displayModeBar': False})

                    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
                    panel_header("🤖 แมตช์ที่รอทำนาย (AI Predictions)", "ระบบวิเคราะห์ความน่าจะเป็นและช่องกรอกผลสกอร์อัปเดต Google Sheets")
                    
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
                        
                        st.markdown(f"พบแมตช์ที่รอการทำนาย **{len(df_predict)} คู่**:")
                        render_prediction_cards(df_predict, workbook)
                    else:
                        st.success("🎉 ไม่มีคู่ค้างที่ยังไม่ลงผลสกอร์ ทุกคู่ใน Google Sheets บันทึกผลครบถ้วนแล้ว!")
                else:
                    st.info("ℹ️ ต้องการข้อมูลที่มีผลการแข่งขันในอดีตอย่างน้อย 5 แถวขึ้นไป เพื่อให้ AI เริ่มประมวลผลสถิติ")
        else:
            st.warning("⚠️ ไม่พบคอลัมน์ 'ผล HDP' หรือ 'ผลสูง/ต่ำ' ใน Google Sheets สำหรับเทรน AI")
