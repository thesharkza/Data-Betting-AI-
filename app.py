# P2_FORM5_SINGLE_V1 - standalone app.py; no local Python or CSS imports.
# Run: python -m streamlit run app.py
# Requires the existing installed packages and Streamlit Secrets.
from __future__ import annotations

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
        raise ValidationError(f"{name}: must be an integer >= {minimum}.")
    if maximum is not None and num > maximum:
        raise ValidationError(f"{name}: maximum is {maximum}.")
    return int(num)

def cell_text(value: Any) -> str:
    return "" if value is None else str(value).strip()

def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())

def validate_quote(row: Sequence[Any]) -> list[Any]:
    if len(row) != 12:
        raise ValidationError("A:L requires exactly 12 values.")
    if not str(row[0] or "").strip() or not str(row[1] or "").strip():
        raise ValidationError("Both team names are required.")
    if normalized(str(row[0])) == normalized(str(row[1])):
        raise ValidationError("Home and away team cannot have the same name.")
    clean: list[Any] = [str(row[0]).strip(), str(row[1]).strip()]
    for index, value in enumerate(row[2:], start=2):
        if value is None or value == "" or isinstance(value, bool):
            raise ValidationError(f"Column {column_letter(index + 1)}: a number is required.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Column {column_letter(index + 1)}: invalid number.") from exc
        if not isfinite(number):
            raise ValidationError("Prices must be finite numbers.")
        clean.append(number)
    if min(clean[2:5]) <= 1:
        raise ValidationError("All 1X2 decimal odds must be > 1.")
    for i in (6, 8, 10, 11):
        if clean[i] == 0 or abs(clean[i]) > 1:
            raise ValidationError(f"Column {column_letter(i+1)}: Malay odds must be nonzero and within [-1, 1].")
    if clean[9] <= 0:
        raise ValidationError("Over/under line must be > 0.")
    for i in (5, 7, 9):
        if abs(clean[i] * 4 - round(clean[i] * 4)) > 1e-6:
            raise ValidationError(f"Column {column_letter(i+1)}: line must use increments of 0.25.")
    if abs(clean[5] + clean[7]) > 1e-6:
        raise ValidationError("Home and away handicap lines must be opposites.")
    return clean

@dataclass(frozen=True)
class TeamForm:
    'Win percentage is entered in UI units (60 means 60%, not 0.60).'
    win_rate_pct: float
    goals_for: int
