"""Validated last-five input and non-contiguous Google Sheets writes.

No network calls at import time. Google Sheets objects are injected by the app.
A:W is the original schema. Only A:L and X:BE are written for a match.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from math import isfinite
from typing import Any, Sequence
from zoneinfo import ZoneInfo
import re
import unicodedata

SCHEMA_VERSION = "FORM5_V1"
FORM_START = 24  # X, one-based
FORM_END = 57  # BE, one-based
FORMULA_MAX_ROW = 2000
# Populated from the uploaded CSV by the build script (not from a live sheet).
BASE_HEADERS: tuple[str, ...] = ('ทีมเหย้า', 'ทีมเยือน', '1X2 เหย้า', '1X2 เสมอ', '1X2 เยือน', 'แฮนดิแคป (เหย้า)', 'ค่าน้ำ HDP (เหย้า)', 'แฮนดิแคป (เยือน)', 'ค่าน้ำ HDP (เยือน)', 'โกลสูง/ต่ำ', 'ค่าน้ำ (สูง)', 'ค่าน้ำ (ต่ำ)', 'ผลสกอร์', 'ผล HDP', 'ผลสูง/ต่ำ', 'แนะนำลงทุน', 'ผลเปรียบเทียบ', 'ความเชื่อมั่น (สถิติย้อนหลัง)', 'วัดความเบี่ยงเบนผิดปกติ', 'Money Management', 'เลือก', 'กำไร/ขาดทุนจำลองต่อ 1 Unit', 'เพิ่มรหัสกลุ่มกฎสำหรับรวมสถิติ')
FORM_HEADERS = (
    "H_WinRate_Last5", "H_GoalsFor_Last5", "H_GoalsAgainst_Last5",
    "A_WinRate_Last5", "A_GoalsFor_Last5", "A_GoalsAgainst_Last5",
    "H_Matches_Last5", "H_Wins_Last5", "H_Draws_Last5", "H_Losses_Last5",
    "A_Matches_Last5", "A_Wins_Last5", "A_Draws_Last5", "A_Losses_Last5",
    "H_GoalsFor_Avg_Last5", "H_GoalsAgainst_Avg_Last5",
    "A_GoalsFor_Avg_Last5", "A_GoalsAgainst_Avg_Last5",
    "H_GoalDiff_Last5", "A_GoalDiff_Last5",
    "H_PointsPerGame_Last5", "A_PointsPerGame_Last5",
    "Kickoff_At", "Competition", "Odds_Captured_At", "Form_AsOf",
    "Form_Scope", "Form_Source", "Recorded_At", "Fixture_ID",
    "Record_ID", "Schema_Version", "Entry_Mode", "Image_Name",
)
FORM_SCOPES = ("ALL_COMPETITIONS", "LEAGUE_ONLY", "VENUE_ALL_COMPETITIONS", "VENUE_LEAGUE_ONLY")
ENTRY_MODES = ("PREMATCH", "HISTORICAL")


class ValidationError(ValueError):
    """Input cannot safely be persisted."""


class SchemaConflict(ValidationError):
    """The live worksheet does not have the expected schema."""


class DuplicateFixture(ValidationError):
    def __init__(self, row: int):
        self.row = row
        super().__init__(f"Fixture already exists at row {row}. Confirm updating the same unsettled match, or cancel.")


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


@dataclass(frozen=True)
class TeamForm:
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int

    @classmethod
    def from_inputs(cls, *, matches: Any, wins: Any, draws: Any, losses: Any,
                    goals_for: Any, goals_against: Any, label: str) -> "TeamForm":
        form = cls(_integer(matches, f"{label} matches", 1, 5),
                   _integer(wins, f"{label} wins", 0, 5),
                   _integer(draws, f"{label} draws", 0, 5),
                   _integer(losses, f"{label} losses", 0, 5),
                   _integer(goals_for, f"{label} goals for"),
                   _integer(goals_against, f"{label} goals against"))
        form.validate(label)
        return form

    def validate(self, label: str = "Team") -> None:
        for name in ("matches", "wins", "draws", "losses", "goals_for", "goals_against"):
            _integer(getattr(self, name), f"{label} {name}", 1 if name == "matches" else 0,
                     5 if name in ("matches", "wins", "draws", "losses") else None)
        if self.wins + self.draws + self.losses != self.matches:
            raise ValidationError(f"{label}: wins + draws + losses must equal matches ({self.matches}).")
        if self.goals_for < self.wins or self.goals_against < self.losses:
            raise ValidationError(f"{label}: goals are inconsistent with wins/losses (90-minute results).")
        difference = self.goals_for - self.goals_against
        if self.wins == 0 and self.losses == 0 and difference != 0:
            raise ValidationError(f"{label}: all draws require goals for = goals against.")
        if self.losses == 0 and difference < self.wins:
            raise ValidationError(f"{label}: goal difference is inconsistent with wins and no losses.")
        if self.wins == 0 and difference > -self.losses:
            raise ValidationError(f"{label}: goal difference is inconsistent with losses and no wins.")

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches

    def summary(self) -> dict[str, float | int]:
        return {"Matches": self.matches, "W": self.wins, "D": self.draws, "L": self.losses,
                "Win rate (%)": self.win_rate * 100, "GF total": self.goals_for,
                "GA total": self.goals_against, "GF / match": self.goals_for / self.matches,
                "GA / match": self.goals_against / self.matches,
                "Goal difference": self.goals_for - self.goals_against,
                "Points / match": (3 * self.wins + self.draws) / self.matches}


def parse_local_datetime(value: str, zone: str, label: str) -> datetime:
    """UI uses explicit 24-hour YYYY-MM-DD HH:MM. No guessed date or time."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(zone))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(f"{label}: enter YYYY-MM-DD HH:MM in {zone}.") from exc


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name}: timezone is required.")


@dataclass(frozen=True)
class MatchMeta:
    kickoff: datetime
    odds_at: datetime
    form_asof: datetime
    competition: str
    scope: str
    source: str
    entry_mode: str
    image_name: str = ""

    def validate(self, now: datetime) -> None:
        for label, value in (("kickoff", self.kickoff), ("odds time", self.odds_at),
                             ("form as-of", self.form_asof), ("current time", now)):
            _aware(value, label)
        if not self.competition.strip() or not self.source.strip():
            raise ValidationError("Competition and form source are required.")
        if self.scope not in FORM_SCOPES or self.entry_mode not in ENTRY_MODES:
            raise ValidationError("Invalid form scope or entry mode.")
        if not self.form_asof <= self.odds_at < self.kickoff:
            raise ValidationError("Require Form_AsOf <= Odds_Captured_At < Kickoff_At. Do not include this match in last-five data.")
        if self.odds_at > now + timedelta(minutes=5):
            raise ValidationError("The odds capture time cannot be in the future.")
        if self.entry_mode == "PREMATCH" and now >= self.kickoff:
            raise ValidationError("Kickoff has passed. Use HISTORICAL only with genuine pre-match source evidence.")


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


def cell_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())


def fixture_id(row: Sequence[Any], meta: MatchMeta) -> str:
    parts = (normalized(str(row[0])), normalized(str(row[1])), normalized(meta.competition),
             meta.kickoff.astimezone(timezone.utc).isoformat(timespec="minutes"))
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Formula:
    value: str


def form_cells(target: int, row: Sequence[Any], home: TeamForm, away: TeamForm,
               meta: MatchMeta, record_id: str, now: datetime) -> list[Any]:
    """Derived statistics remain live Google Sheets formulas, not hardcoded values."""
    r = target
    values = [
        Formula(f'=IFERROR(AE{r}/AD{r},"")'), home.goals_for, home.goals_against,
        Formula(f'=IFERROR(AI{r}/AH{r},"")'), away.goals_for, away.goals_against,
        home.matches, home.wins, home.draws, home.losses,
        away.matches, away.wins, away.draws, away.losses,
        Formula(f'=IFERROR(Y{r}/AD{r},"")'), Formula(f'=IFERROR(Z{r}/AD{r},"")'),
        Formula(f'=IFERROR(AB{r}/AH{r},"")'), Formula(f'=IFERROR(AC{r}/AH{r},"")'),
        Formula(f'=Y{r}-Z{r}'), Formula(f'=AB{r}-AC{r}'),
        Formula(f'=IFERROR((3*AE{r}+AF{r})/AD{r},"")'),
        Formula(f'=IFERROR((3*AI{r}+AJ{r})/AH{r},"")'),
        meta.kickoff.isoformat(timespec="minutes"), meta.competition.strip(),
        meta.odds_at.isoformat(timespec="minutes"), meta.form_asof.isoformat(timespec="minutes"),
        meta.scope, meta.source.strip(), now.isoformat(timespec="seconds"),
        fixture_id(row, meta), record_id, SCHEMA_VERSION, meta.entry_mode, meta.image_name,
    ]
    assert len(values) == len(FORM_HEADERS) == FORM_END - FORM_START + 1
    return values


def _typed(value: Any) -> dict[str, Any]:
    if isinstance(value, Formula):
        return {"userEnteredValue": {"formulaValue": value.value}}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)):
        return {"userEnteredValue": {"numberValue": value}}
    # RAW text, even when it begins with '='. Never evaluate user-supplied formulas.
    return {"userEnteredValue": {"stringValue": str(value)}}


def update_cells_request(sheet_id: int, row_number: int, column_number: int,
                         values: Sequence[Any]) -> dict[str, Any]:
    return {"updateCells": {
        "start": {"sheetId": sheet_id, "rowIndex": row_number - 1,
                  "columnIndex": column_number - 1},
        "rows": [{"values": [_typed(v) for v in values]}],
        "fields": "userEnteredValue",
    }}


def _format_request(sheet_id: int, row: int, column: int, kind: str, pattern: str) -> dict[str, Any]:
    return {"repeatCell": {
        "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row,
                  "startColumnIndex": column-1, "endColumnIndex": column},
        "cell": {"userEnteredFormat": {"numberFormat": {"type": kind, "pattern": pattern}}},
        "fields": "userEnteredFormat.numberFormat",
    }}


def header_plan(ws: Any) -> list[dict[str, Any]]:
    """Read-only validation; never insert columns between A:W.

    If new headers are incomplete, require all cells below X:BE to be empty.
    A conflicting pre-existing helper block (e.g. DATA_V2) fails closed.
    """
    headers = list(ws.row_values(1, value_render_option="FORMULA"))
    if len(headers) < len(BASE_HEADERS):
        raise SchemaConflict("Expected the original A:W DATA sheet; required headers are missing.")
    for i, expected in enumerate(BASE_HEADERS):
        if normalized(str(headers[i])) != normalized(expected):
            raise SchemaConflict(f"Column {column_letter(i+1)} must be '{expected}'. Do not use DATA_V2 or a shifted schema.")
    existing = [cell_text(value) for value in headers[FORM_START-1:FORM_END]]
    existing += [""] * (len(FORM_HEADERS) - len(existing))
    for i, (actual, expected) in enumerate(zip(existing, FORM_HEADERS)):
        if str(actual).strip() and str(actual).strip() != expected:
            raise SchemaConflict(f"{column_letter(FORM_START+i)}1 already contains '{actual}'. Nothing was written.")
    needs_headers = existing != list(FORM_HEADERS)
    if needs_headers and ws.col_count >= FORM_START and ws.row_count >= 2:
        end = column_letter(min(ws.col_count, FORM_END))
        cells = ws.get(f"X2:{end}{ws.row_count}", value_render_option="FORMULA")
        if any(cell_text(cell) for record in cells for cell in record):
            raise SchemaConflict("X:BE contains data/formulas below incomplete headers. Nothing was overwritten.")
    requests: list[dict[str, Any]] = []
    if ws.col_count < FORM_END:
        requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "COLUMNS",
                                             "length": FORM_END - ws.col_count}})
    if needs_headers:
        requests.append(update_cells_request(ws.id, 1, FORM_START, FORM_HEADERS))
    return requests


def prepare_headers(ws: Any, *, dry_run: bool = True) -> dict[str, Any]:
    requests = header_plan(ws)
    if requests and not dry_run:
        ws.spreadsheet.batch_update({"requests": requests})
    return {"dry_run": dry_run, "requests": requests, "range": "X1:BE1"}


@dataclass(frozen=True)
class SaveResult:
    row: int
    overwritten: bool = False
    already_saved: bool = False
    legacy_same_names: bool = False


def save_match(ws: Any, row: Sequence[Any], home: TeamForm, away: TeamForm,
               meta: MatchMeta, record_id: str, *, allow_update: bool = False,
               now: datetime | None = None, max_formula_row: int = FORMULA_MAX_ROW) -> SaveResult:
    """Caller MUST hold a shared single-process lock around this function.

    One atomic Sheets batch writes A:L and X:BE, never M:W. Idempotency protects
    manual retries after a lost response. This is not a distributed transaction:
    do not run multiple writer processes or sort the sheet during a save.
    """
    now = now or datetime.now(timezone.utc)
    row = validate_quote(row)
    home.validate("Home")
    away.validate("Away")
    meta.validate(now)
    if not re.fullmatch(r"[0-9a-f]{32}", record_id):
        raise ValidationError("Record_ID must be a 32-character lowercase UUID hex value.")
    requests = header_plan(ws)
    input_rows = list(ws.get(f"A2:M{ws.row_count}")) if ws.row_count >= 2 else []
    names = [list(rr[:2]) for rr in input_rows]
    metadata = list(ws.get(f"BA2:BD{ws.row_count}")) if ws.col_count >= 56 and ws.row_count >= 2 else []
    keyed = fixture_id(row, meta)
    same_fixture: list[int] = []
    legacy_same_names = False
    for idx, previous in enumerate(metadata, start=2):
        padded = list(previous) + [""] * 4
        if padded[1] == record_id:
            return SaveResult(idx, already_saved=True)
        if padded[0] == keyed:
            same_fixture.append(idx)
    if len(same_fixture) > 1:
        raise ValidationError("Multiple rows have this Fixture_ID. Resolve duplicates manually before saving.")
    for idx, name_pair in enumerate(names, start=2):
        pair = list(name_pair) + [""] * 2
        same = normalized(cell_text(pair[0])) == normalized(row[0]) and normalized(cell_text(pair[1])) == normalized(row[1])
        previous_meta = metadata[idx-2] if idx-2 < len(metadata) else []
        if same and (not previous_meta or not previous_meta[0]):
            legacy_same_names = True
    if same_fixture:
        target = same_fixture[0]
        if not allow_update:
            raise DuplicateFixture(target)
        actual_score = ws.get(f"M{target}:M{target}", value_render_option="FORMULA")
        if any(cell_text(c) for rr in actual_score for c in rr):
            raise ValidationError(f"Row {target} already has a score in M. Refusing to change historical prices/form.")
        if now >= meta.kickoff:
            raise ValidationError("Refusing to update an existing fixture after kickoff.")
    else:
        used = [1] + [i for i, rr in enumerate(input_rows, 2) if any(cell_text(c) for c in rr)]
        used += [i for i, rr in enumerate(metadata, 2) if any(cell_text(c) for c in rr)]
        target = max(used) + 1
    if target > max_formula_row:
        raise ValidationError(f"Next row is {target}, beyond formula coverage {max_formula_row}. Extend N:W formulas and configuration first.")
    if target > ws.row_count:
        requests.append({"appendDimension": {"sheetId": ws.id, "dimension": "ROWS",
                                             "length": target - ws.row_count}})
    values = form_cells(target, row, home, away, meta, record_id, now)
    requests.extend((update_cells_request(ws.id, target, 1, row),
                     update_cells_request(ws.id, target, FORM_START, values)))
    for col in (24, 27):  # X and AA contain ratios, formatted as percentage.
        requests.append(_format_request(ws.id, target, col, "PERCENT", "0.0%"))
    for col in (38, 39, 40, 41, 44, 45):  # AL:AO and AR:AS
        requests.append(_format_request(ws.id, target, col, "NUMBER", "0.00"))
    ws.spreadsheet.batch_update({"requests": requests})
    return SaveResult(target, overwritten=bool(same_fixture), legacy_same_names=legacy_same_names)
