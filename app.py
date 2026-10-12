"""Six-field last-five input: see README.md. Price extraction and Dashboard retained."""
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
