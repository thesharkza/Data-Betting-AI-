# P2_FORM5_SINGLE_V2 - one application file; visible UI before external services.
# Run: python -m streamlit run app.py
from __future__ import annotations

import streamlit as st
from pathlib import Path
import logging

BUILD_ID = "P2_FORM5_SINGLE_V2"
logger = logging.getLogger(__name__)
st.set_page_config(page_title="P2 Form5", layout="wide")
st.title("P2 Form5")
st.caption(f"Build: {BUILD_ID} | app.py")
startup_notice = st.empty()
startup_notice.info("กำลังเปิดหน้าจอ — ยังไม่เชื่อมต่อ Google Sheets หรือ Gemini")

# Detect a valid-but-truncated Python upload, without enforcing a hash or file length.
try:
    _source_lines = Path(__file__).read_text(encoding="utf-8-sig").splitlines()
except (OSError, UnicodeError) as exc:
    startup_notice.error(f"Cannot read app.py: {type(exc).__name__}")
    st.stop()
if "# END OF P2_FORM5_SINGLE_V2 - complete app.py" not in [line.strip() for line in _source_lines]:
    startup_notice.error(
        f"app.py: ไม่พบส่วนท้ายของไฟล์ (อ่านได้ {len(_source_lines)} บรรทัด) อาจอัปโหลดไม่ครบ; "
        "ใช้ไฟล์ app.py จาก ZIP ทั้งไฟล์ ไม่คัดลอกเฉพาะพรีวิว"
    )
    st.stop()
logger.info("%s: app.py has %d lines", BUILD_ID, len(_source_lines))

# Imports do not connect to the APIs. Show a visible message if a package is missing.
try:
    import gspread
    from google import genai
    from google.genai import types
    from PIL import Image
    import pandas as pd
    import plotly.graph_objects as go
except ImportError as exc:
    startup_notice.error(f"โหลดไลบรารีไม่สำเร็จ: {exc}")
    st.code("python -m pip install -r requirements.txt", language="bash")
    st.stop()

import io
import os
import gc
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as esc

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any, Sequence
import json
import re
import unicodedata

SCHEMA_VERSION = "FORM5_SIMPLE_V2"
FORM_START = 24
FORM_END = 29
FORMULA_MAX_ROW = 2000
LAST_N = 5
BASE_HEADERS: tuple[str, ...] = ('ทีมเหย้า', 'ทีมเยือน', '1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'แฮนดิแคป (เยือน)', 'ค่าน้ำ HDP (เยือน)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)', 'ผลสกอร์', 'ผล HDP', 'ผลสูง/ต่ำ', 'แนะนำลงทุน', 'ผลเปรียบเทียบ', 'ความเชื่อมั่น (สถิติย้อนหลัง)', 'วัดความเบี่ยงเบนผิดปกติ', 'Money Management', 'เลือก', 'กำไร/ขาดทุนจำลองต่อ 1 Unit', 'เพิ่มรหัสกลุ่มกฎสำหรับรวมสถิติ')
FORM_HEADERS = (
    "H_WinRate_Last5", "H_GoalsFor_Last5", "H_GoalsAgainst_Last5",
    "A_WinRate_Last5", "A_GoalsFor_Last5", "A_GoalsAgainst_Last5",
)
RECEIPT_PREFIX = "P2_FORM5_SIMPLE_RECEIPT:"

class ValidationError(ValueError):
    'Input is incomplete or inconsistent; nothing should be written.'

class SchemaConflict(ValidationError):
    'A:W or X:AC conflicts with the supported DATA layout.'

def column_letter(number: int) -> str:
    if number < 1:
        raise ValueError("Column numbers start at 1")
    result = ""
    while number:
        number, rem = divmod(number - 1, 26)
        result = chr(65 + rem) + result
    return result

def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if value is None or value == "" or isinstance(value, bool):
        raise ValidationError(f"{name}: please enter an integer; missing is not zero.")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name}: must be an integer.") from exc
    if not isfinite(num) or not num.is_integer() or num < minimum:
