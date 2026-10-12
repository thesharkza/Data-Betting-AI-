'Six-field last-five input: see README.md. Price extraction and Dashboard retained.'
import streamlit as st
import gspread
from google import genai
from google.genai import types
from pathlib import Path
import logging
from form_data import (
    BASE_HEADERS, FORM_HEADERS, TeamForm, ValidationError,
    validate_quote, prepare_headers, save_match,
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
# Page configuration is set once by the short app.py launcher.

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

css_path = Path(__file__).with_name("style.css")
st.markdown("<style>" + css_path.read_text(encoding="utf-8-sig") + "</style>", unsafe_allow_html=True)

# ----------------------------------------
# การตั้งค่า API และ Google Sheets
# ----------------------------------------
def setting(name, default=None):
    'Settings read in the main thread; secrets are never written to a file.'
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value if value is not None else os.environ.get(name, default)

GEMINI_API_KEY = setting("GEMINI_API_KEY")
# Original model identifier retained as a default, not claimed to be valid for every account.
GEMINI_MODEL = setting("GEMINI_MODEL", "gemini-3.7-flash")
DATA_WORKSHEET = setting("DATA_WORKSHEET", "DATA")
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
