import streamlit as st
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import pandas as pd
import plotly.graph_objects as go
import gc
import time
from datetime import datetime, timezone, timedelta

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

@st.cache_resource
def init_gspread():
    try:
        if "gspread" in st.secrets:
            creds_dict = dict(st.secrets["gspread"])
            client = gspread.service_account_from_dict(creds_dict)
        else:
            client = gspread.service_account(filename='credentials.json')
        return client.open("ข้อมูลราคาบอลสกัดจากภาพ")
    except Exception as e:
        st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
        return None

workbook = init_gspread()
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

def clean_raw_data(raw_text):
    raw_data = [item.strip().replace('−', '-') for item in raw_text.split(',')]
    m = re.search(r'\b(MY|HK|ID|EU)\b', raw_data[12].upper()) if len(raw_data) > 12 else None
    ai_fmt = m.group(1) if m else "UNK"
    row_data = []
    for i, item in enumerate(raw_data[:12]):
        if i < 2:
            row_data.append(item)
        else:
            cleaned = re.sub(r'^[oOuU\s]+', '', item)
            try:
                row_data.append(float(cleaned) if '.' in cleaned else int(cleaned))
            except:
                row_data.append(cleaned)
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
    if any(abs(new_row[i]) > 1 for i in idx):
        warn = "⚠️ ค่าน้ำหลังแปลงเกิน ±1 ซึ่งผิดปกติสำหรับระบบมาเลเซีย กรุณาเลือกรูปแบบราคาให้ตรงกับภาพแล้วอัปโหลดใหม่"
    elif source.startswith("ระบบประเมิน"):
        warn = "⚠️ AI ไม่แน่ใจรูปแบบราคา ระบบจึงเดาให้ ควรเช็กค่าน้ำที่บันทึกอีกครั้ง"
    return new_row, note, warn

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
st.markdown("<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>สกัดราคาจากภาพ วิเคราะห์ และติดตามผลงานในที่เดียว</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "📊 Dashboard"])

# ----------------------------------------
# TAB 1
# ----------------------------------------
with tab1:
    st.subheader("สกัดราคาบอล & วิเคราะห์ VIP")
    st.write("อัปโหลดภาพตารางราคาเพื่อสกัดข้อมูลส่งเข้า Google Sheets อัตโนมัติ")

    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="ภาพที่อัปโหลด", width=600)

        fmt_choice = st.selectbox(
            "รูปแบบค่าน้ำในภาพ", list(FMT_CHOICES.keys()),
            help="ระบบจะแปลงค่าน้ำเป็นแบบมาเลเซียก่อนบันทึกเสมอ เลือก 'อัตโนมัติ' ถ้าไม่แน่ใจ หรือเลือกเองเมื่อรู้ว่าภาพเป็นแบบไหน")

        if st.button("🚀 อัปโหลดและบันทึกข้อมูล", type="primary"):
            if not workbook:
                st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
            else:
                try:
                    ws_data = workbook.sheet1

                    with st.spinner("📸 กำลังประมวลผลรูปภาพ..."):
                        img = Image.open(uploaded_file)
                        if img.mode != 'RGB': img = img.convert('RGB')
                        img.thumbnail((800, 800))

                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=80)
                        img_bytes = img_byte_arr.getvalue()

                    with st.spinner("🤖 กำลังให้ AI สกัดราคาบอล..."):
                        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ,รหัสรูปแบบราคาน้ำ (ตอบรหัสเดียว: MY=มาเลเซีย มีทั้งบวกและลบ, HK=ฮ่องกง, ID=อินโดนีเซีย, EU=Decimal เช่น 1.90, UNK=ไม่แน่ใจ) ค่าน้ำทุกค่าให้คัดลอกตามที่เห็นในภาพทุกประการรวมเครื่องหมายลบ ห้ามแปลงค่าเอง โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
                        response = model.generate_content([{"mime_type": "image/jpeg", "data": img_bytes}, prompt])

                    gc.collect()

                    if not response or not response.text:
                        st.error("❌ AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                    else:
                        with st.spinner("📊 กำลังบันทึกและรอ Google Sheets ประมวลผลลัพธ์..."):
                            row_data_to_sheet, ai_fmt = clean_raw_data(response.text.strip())
                            row_data_to_sheet, conv_note, conv_warn = convert_water_to_malay(
                                row_data_to_sheet, FMT_CHOICES[fmt_choice], ai_fmt)

                            col_a_values = ws_data.col_values(1)
                            col_b_values = ws_data.col_values(2)
                            last_row = sum(1 for val in col_a_values if str(val).strip() != "")

                            new_home = str(row_data_to_sheet[0]).strip()
                            new_away = str(row_data_to_sheet[1]).strip()
                            target_row = None

                            for i in range(len(col_a_values)):
                                sheet_home = str(col_a_values[i]).strip()
                                sheet_away = str(col_b_values[i]).strip() if i < len(col_b_values) else ""
                                if sheet_home == new_home and sheet_away == new_away:
                                    target_row = i + 1
                                    break

                            if target_row is not None:
                                ws_data.update(range_name=f"A{target_row}", values=[row_data_to_sheet])
                                st.warning(f"🔄 พบข้อมูลคู่ {new_home} vs {new_away} ในระบบ ทำการ **บันทึกทับ** ที่แถว {target_row} เรียบร้อยแล้ว")
                            else:
                                target_row = last_row + 1
                                ws_data.update(range_name=f"A{target_row}", values=[row_data_to_sheet])
                                st.success("✅ บันทึกข้อมูลคู่ใหม่สำเร็จ!")

                            time.sleep(3)
                            updated_row = ws_data.row_values(target_row)

                            rec = updated_row[15] if len(updated_row) > 15 else "กำลังคำนวณ..."
                            confidence = updated_row[17] if len(updated_row) > 17 else "-"
                            radar = updated_row[18] if len(updated_row) > 18 else "-"

                            mm_text, rec = get_money_management(rec, confidence)

                            st.info(conv_note)
                            if conv_warn:
                                st.warning(conv_warn)

                            st.markdown(f"""<div class="result-card">
<h3>{row_data_to_sheet[0]}<span class="vs">vs</span>{row_data_to_sheet[1]}</h3>
<div class="result-row"><span class="result-key">🎯 แนะนำลงทุน</span><span class="result-val hl">{rec}</span></div>
<div class="result-row"><span class="result-key">💰 Money Mgt</span><span class="result-val">{mm_text}</span></div>
<div class="result-row"><span class="result-key">📊 สถิติความเชื่อมั่น</span><span class="result-val">{confidence}</span></div>
<div class="result-row"><span class="result-key">🚨 เช็กราคา</span><span class="result-val">{radar}</span></div>
</div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                finally:
                    gc.collect()

# ----------------------------------------
# TAB 2 : Dashboard
# ----------------------------------------
with tab2:
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        try:
            ws_data = workbook.sheet1
            df = pd.DataFrame(ws_data.get_all_records())

            ws_stats = workbook.worksheet("สรุปสถิติ")
            df_stats = pd.DataFrame(ws_stats.get_all_records())
            if not df_stats.empty:
                df_stats.columns = [str(c).strip() for c in df_stats.columns]

            if not df.empty and 'ผลเปรียบเทียบ' in df.columns:
                df_comp = df[df['ผลเปรียบเทียบ'].astype(str).str.contains('ชนะ|แพ้|เจ๊า', na=False)].copy()
                total = len(df_comp)
                wins_full = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะเต็ม'])
                wins_half = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะครึ่ง'])
                total_wins = wins_full + (wins_half * 0.5)
                draws = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'เจ๊า'])
                win_rate = round((total_wins / (total - draws)) * 100, 2) if (total - draws) > 0 else 0

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
                        annotations=[dict(text=f"<b style='font-size:34px;color:{TEXT}'>{win_rate}%</b><br>Win Rate",
                                          x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED, size=13))]
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
                history, hist_err = update_rate_history(workbook, rates)
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
                    meta = "".join(f"<span>{c}<b>{row[c]}</b></span>" for c in extra_cols
                                   if str(row[c]).strip() != "")
                    cards.append(
                        f"<div class='rule-card'>"
                        f"<div class='rule-top'><span class='rule-name'>{name}</span>"
                        f"<span class='rule-tag tag-{k}'>{tag}</span></div>"
                        f"<div class='rule-rate-row'><div class='rule-rate t-{k}'>{row['อัตราชนะ']}</div>"
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
