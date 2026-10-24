"""Run: python -m streamlit run app.py

Complete application entrypoint. Keep the football_ai/ package beside this file.
Default deployment binds to localhost and does not touch the original DATA tab.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import hashlib
import json
import logging
import secrets as random_secrets
import time
import traceback
from dataclasses import asdict
from datetime import datetime, time as clock_time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from football_ai import __version__
from football_ai.config import Settings
from football_ai.domain import (FORM_COLUMNS, STATES, ValidationError, canonical_json, digest, market_errors,
                                number, timestamp, to_decimal, utc_now)
from football_ai.ingestion import available_models, extract_image, image_key
from football_ai.jobs import JobQueue
from football_ai.model import assess_prediction, train_market, training_digest, training_records
from football_ai.registry import Registry
from football_ai.repository import AmbiguousWrite, SheetsStore, SQLiteStore, StorageError
from football_ai.service import (evaluate_frozen, freeze_predictions, materialize, pending_latest, save_quote, save_result)

st.set_page_config(page_title="Football AI v6 | Data & Model Lab", page_icon="\u26bd", layout="wide")
st.markdown("""<style>
.block-container {max-width:1320px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stMetric"] {border:1px solid #254437;border-radius:12px;padding:14px;}
h1,h2,h3 {letter-spacing:-.02em;}
.stCaption {line-height:1.6;}
</style>""", unsafe_allow_html=True)


def read_secrets():
    try:
        return st.secrets.to_dict()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return {}


SECRET_VALUES = read_secrets()
try:
    SETTINGS = Settings.load(SECRET_VALUES)
except (TypeError, ValueError) as exc:
    st.error("Configuration error: " + str(exc)); st.stop()
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def configure_logging(directory: str):
    logger = logging.getLogger("football_ai")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(Path(directory) / "application.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = configure_logging(str(SETTINGS.data_dir))


def error_message(exc: Exception, context: str):
    ref = random_secrets.token_hex(6)
    frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
              for f in traceback.extract_tb(exc.__traceback__)[-8:]]
    cause = exc.__cause__ or exc
    status = getattr(getattr(cause, "response", None), "status_code", getattr(cause, "code", None))
    LOGGER.error("operation_failed op=%s ref=%s type=%s http=%s frames=%s", context, ref,
                 type(exc).__name__, status if isinstance(status, int) else "unknown", frames)
    if isinstance(exc, (ValidationError, StorageError)):
        st.error(str(exc))
    else:
        st.error(f"{context} failed ({type(exc).__name__}). Reference: {ref}. No credentials or stack trace are shown.")


def authenticate():
    if SETTINGS.local_only:
        # Important: a reverse proxy can expose even a loopback service. See README.
        address = st.get_option("server.address")
        if address not in {"localhost", "127.0.0.1", "::1"}:
            st.error("Local-only mode requires server.address=127.0.0.1. For shared hosting configure OIDC and local_only=false.")
            st.stop()
        return "local-admin", "admin"
    if "auth" not in SECRET_VALUES:
        st.error("Shared hosting is locked: configure Streamlit OIDC and administrator/viewer allowlists.")
        st.stop()
    if not getattr(st.user, "is_logged_in", False):
        st.title("Sign in / \u0e40\u0e02\u0e49\u0e32\u0e2a\u0e39\u0e48\u0e23\u0e30\u0e1a\u0e1a")
        if st.button("Sign in", type="primary"):
            st.login()
        st.stop()
    claims = st.user.to_dict()
    if claims.get("exp") and float(claims["exp"]) <= time.time():
        st.logout(); st.stop()
    if claims.get("email_verified") is False:
        st.error("Your identity provider has not verified this email."); st.stop()
    email = str(claims.get("email") or claims.get("preferred_username") or "").casefold()
    if email in SETTINGS.administrators:
        role = "admin"
    elif email in SETTINGS.editors:
        role = "editor"
    elif email in SETTINGS.viewers:
        role = "viewer"
    else:
        st.error("Signed in, but your identity is not on this application's allowlist.")
        if st.button("Sign out"): st.logout()
        st.stop()
    return email, role


ACTOR, ROLE = authenticate()
if "session_id_v6" not in st.session_state:
    st.session_state.session_id_v6 = random_secrets.token_hex(20)
OWNER = ACTOR + ":" + st.session_state.session_id_v6
CAN_EDIT = ROLE in {"admin", "editor"}
CAN_WRITE = CAN_EDIT and SETTINGS.enable_writes


def require_role(role="editor", writes=True):
    if (ROLE != "admin" if role == "admin" else not CAN_EDIT):
        raise ValidationError("You do not have permission for this operation.")
    if writes and not SETTINGS.enable_writes:
        raise ValidationError("Data writes are disabled by configuration.")


@st.cache_resource
def connect_store(namespace: str, backend: str, enabled: bool, spreadsheet_id: str, legacy_sheet: str, directory: str, _credentials: dict):
    if backend == "local":
        return SQLiteStore(Path(directory) / namespace, enabled)
    return SheetsStore(spreadsheet_id, _credentials, legacy_sheet, enabled, Path(directory) / namespace / "uncertain_write.json")


@st.cache_resource
def registry_for(namespace: str, directory: str):
    return Registry(Path(directory) / namespace / "models")


@st.cache_resource
def queues_for(namespace: str):
    return JobQueue(workers=2, max_pending=6, max_per_owner=3), JobQueue(workers=1, max_pending=2, max_per_owner=2)


@st.cache_data(ttl=45, max_entries=4, show_spinner=False)
def load_snapshot(namespace: str, revision: int, _store):
    return _store.snapshot()


REGISTRY = registry_for(SETTINGS.namespace, str(SETTINGS.data_dir))
OCR_QUEUE, TRAIN_QUEUE = queues_for(SETTINGS.namespace)
try:
    STORE = connect_store(SETTINGS.namespace, SETTINGS.backend, SETTINGS.enable_writes, SETTINGS.spreadsheet_id,
                          SETTINGS.legacy_sheet, str(SETTINGS.data_dir), dict(SECRET_VALUES.get("gspread", {})))
except Exception as exc:
    error_message(exc, "Storage connection")
    st.info("Offline use: backend='local'. Google Sheets: configure spreadsheet_id and [gspread] in secrets.toml.")
    st.stop()


@st.cache_data(ttl=120, max_entries=16, show_spinner=False)
def infer_batch(model_id: str, quotes_json: str, namespace: str, _registry):
    rows = json.loads(quotes_json)
    return _registry.load(model_id).predict(rows)


def refresh():
    load_snapshot.clear()
    infer_batch.clear()


def new_draft(data: dict | None = None):
    st.session_state.draft_v6 = dict(data or {})
    st.session_state.draft_token_v6 = random_secrets.token_hex(5)


def apply_extraction(data):
    if "teams_match" in data:
        draft = dict(st.session_state.get("draft_v6", {}))
        for key in ("h_win_percent", "h_goals_for", "h_goals_against", "a_win_percent", "a_goals_for", "a_goals_against", "h_matches", "a_matches"):
            draft[key] = data.get(key)
        if not draft.get("home"): draft["home"] = data.get("home")
        if not draft.get("away"): draft["away"] = data.get("away")
        draft["form_image_sha256"] = data.get("image_sha256")
        draft["form_extractor_model"] = data.get("extractor_model")
        new_draft(draft)
    else:
        new_draft(data)


@st.fragment(run_every=2)
def extraction_status():
    jobs = OCR_QUEUE.snapshot(OWNER)
    if not jobs:
        return
    st.subheader("Image reading jobs / \u0e2a\u0e16\u0e32\u0e19\u0e30\u0e2d\u0e48\u0e32\u0e19\u0e20\u0e32\u0e1e")
    for job in jobs:
        with st.container(border=True):
            st.write(f"{job['name']} | {job['status']}")
            if job["status"] == "error":
                st.warning(job["error"])
            elif job["status"] == "done":
                data = job["result"]
                with st.expander("Extracted values and uncertain fields"):
                    st.json(data)
                mismatch = "teams_match" in data and not data["teams_match"]
                if mismatch:
                    st.warning("Image team names do not exactly match the intended fixture. No automatic mapping is allowed.")
                approved = not mismatch or st.checkbox("I checked both team identities and their home/away mapping", key="map_" + job["id"])
                if st.button("Use in review form", key="use_" + job["id"], disabled=not CAN_EDIT or not approved):
                    apply_extraction(data)
                    st.rerun()
    if st.button("Clear finished image jobs"):
        OCR_QUEUE.clear_finished(OWNER); st.rerun()


@st.fragment(run_every=2)
def training_status():
    for job in TRAIN_QUEUE.snapshot(OWNER):
        with st.container(border=True):
            st.write(f"{job['name']} | {job['status']}")
            if job["status"] == "error": st.warning(job["error"])
            if job["status"] == "done":
                result = job["result"]
                st.success("Artifact saved: " + result["model_id"])
                st.caption("Ready for paper evaluation" if result["gate_pass"] else "Research only: statistical readiness gate did not pass.")
                if st.button("Activate this model", key="activatejob_" + job["id"], disabled=ROLE != "admin"):
                    try:
                        require_role("admin", writes=False)
                        REGISTRY.activate(result["model_id"])
                        infer_batch.clear(); st.rerun()
                    except Exception as exc: error_message(exc, "Model activation")


def text_number(label, key, value=None, help=None):
    return st.text_input(label, value="" if value in (None, "") else str(value), key=key, help=help)


def entry_page(records):
    st.header("Capture & review / \u0e23\u0e31\u0e1a\u0e41\u0e25\u0e30\u0e15\u0e23\u0e27\u0e08\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25")
    st.caption("Original DATA is read-only. New quotes are saved to AI_DATA_V6; no old formulas are changed.")
    api_key = os.environ.get("GEMINI_API_KEY") or str(SECRET_VALUES.get("GEMINI_API_KEY", ""))
    model = st.session_state.get("gemini_model_selected", SETTINGS.gemini_model)
    files = st.file_uploader("Price screenshots (PNG/JPEG, <=12 MB each; at most 3 per batch)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files:
        preview_index = st.selectbox("Preview uploaded image", list(range(len(files))), format_func=lambda i: files[i].name)
        st.image(files[preview_index], caption="Review the original image before saving", width=600)
    if st.button("Read selected price images", disabled=not CAN_EDIT or not api_key or not model):
        try:
            require_role(writes=False)
            if len(files or []) > 3: raise ValidationError("At most 3 images per batch.")
            for f in files or []:
                data = f.getvalue()
                if len(data) > 12 * 1024 * 1024: raise ValidationError("Image exceeds 12 MB.")
                OCR_QUEUE.submit(OWNER, image_key(data, model, "odds"), f.name, extract_image, data, api_key, model)
            st.rerun()
        except Exception as exc: error_message(exc, "Queue image reading")
    if not api_key or not model:
        st.info("Image AI is optional. Configure the key/model in Settings or enter the values manually below.")
    extraction_status()
    if "draft_token_v6" not in st.session_state: new_draft()
    d = dict(st.session_state.get("draft_v6", {})); token = st.session_state.draft_token_v6
    if st.button("Clear review form"):
        new_draft(); st.rerun()
    if d.get("period") in {"OTHER", "UNKNOWN"}:
        st.warning("The market period was not confirmed as full-time 90 minutes. Verify manually before saving.")
    if d.get("uncertain_fields"): st.warning("AI marked uncertain fields: " + ", ".join(d["uncertain_fields"]))
    zone = ZoneInfo("Asia/Bangkok"); local_now = datetime.now(zone)
    # Optional form image extraction is tied to the reviewed team names, not filenames.
    with st.expander("Read a form-statistics image (optional)"):
        stat_file = st.file_uploader("Form image", type=["png", "jpg", "jpeg"], key="form_image_" + token)
        expected_home = st.text_input("Expected home team in form image", value=d.get("home") or "", key="form_home_" + token)
        expected_away = st.text_input("Expected away team in form image", value=d.get("away") or "", key="form_away_" + token)
        if stat_file:
            st.image(stat_file, caption="Verify team identities and totals", width=600)
        if st.button("Read form image", disabled=not CAN_EDIT or not api_key or not model or not stat_file or not expected_home.strip() or not expected_away.strip()):
            try:
                require_role(writes=False)
                data = stat_file.getvalue()
                context = expected_home.strip() + "|" + expected_away.strip()
                OCR_QUEUE.submit(OWNER, image_key(data, model, "form", context), "Form: " + stat_file.name,
                                 extract_image, data, api_key, model, "form", expected_home.strip(), expected_away.strip())
                st.rerun()
            except Exception as exc: error_message(exc, "Queue form image")
    with st.form("review_" + token, clear_on_submit=False):
        a, b = st.columns(2)
        with a:
            home = st.text_input("Home team", value=d.get("home") or "")
            competition = st.text_input("Competition / season", value=d.get("competition") or "")
            kickoff_date = st.date_input("Kickoff date (Bangkok time)", value=None)
            kickoff_time = st.time_input("Kickoff time", value=clock_time(18, 0))
        with b:
            away = st.text_input("Away team", value=d.get("away") or "")
            source = st.text_input("Price source / bookmaker / screenshot reference", value="")
            capture_date = st.date_input("Actual price capture date", value=local_now.date())
            capture_time = st.time_input("Actual price capture time", value=local_now.time().replace(microsecond=0), help="Use the screenshot's actual observation time, not an invented historical timestamp.")
        format_options = ["UNKNOWN", "MY", "HK", "ID", "EU"]
        fmt = st.selectbox("Original HDP / O-U price format (confirm, do not guess)", format_options,
                           index=format_options.index(d.get("odds_format", "UNKNOWN")))
        st.caption("1X2 prices must always be decimal. Blank means unknown; zero is never used as a missing price.")
        a, b, c = st.columns(3)
        with a: p_home = text_number("1X2 home", token + "1h", d.get("home_1x2"))
        with b: p_draw = text_number("1X2 draw", token + "1d", d.get("draw_1x2"))
        with c: p_away = text_number("1X2 away", token + "1a", d.get("away_1x2"))
        a, b = st.columns(2)
        with a:
            st.markdown("#### Handicap")
            h_line = text_number("Home handicap line", token + "hl", d.get("hdp_line"))
            a_line = text_number("Away handicap line", token + "al", d.get("hdp_away_line"))
            h_price = text_number("Home handicap price", token + "hp", d.get("hdp_home_raw"))
            a_price = text_number("Away handicap price", token + "ap", d.get("hdp_away_raw"))
        with b:
            st.markdown("#### Total goals")
            o_line = text_number("Over/under line", token + "ol", d.get("ou_line"))
            o_price = text_number("Over price", token + "op", d.get("over_raw"))
            u_price = text_number("Under price", token + "up", d.get("under_raw"))
        st.markdown("#### Recent completed matches (optional; leave unavailable fields blank)")
        forms = {}
        a, b = st.columns(2)
        for prefix, area, label in (("h", a, "Home"), ("a", b, "Away")):
            with area:
                forms[prefix + "_win_percent"] = text_number(label + " win rate (%)", token + prefix + "wr", d.get(prefix + "_win_percent"))
                forms[prefix + "_goals_for"] = text_number(label + " TOTAL goals for", token + prefix + "gf", d.get(prefix + "_goals_for"))
                forms[prefix + "_goals_against"] = text_number(label + " TOTAL goals against", token + prefix + "ga", d.get(prefix + "_goals_against"))
                forms[prefix + "_matches"] = st.number_input(label + " recent match count", min_value=1, max_value=5, value=int(d.get(prefix + "_matches") or 5), step=1)
        form_date = st.date_input("Form statistics as-of date", value=local_now.date())
        form_time = st.time_input("Form statistics as-of time", value=local_now.time().replace(microsecond=0))
        form_verified = st.checkbox("I verified the form totals and excluded this fixture's outcome")
        reviewed = st.checkbox("I checked teams, prices, +/- signs, source, timestamps, and full-time 90-minute market")
        submitted = st.form_submit_button("Save reviewed quote / \u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25", type="primary", disabled=not CAN_WRITE)
    if submitted:
        try:
            require_role()
            if not reviewed: raise ValidationError("Review and confirm the observed inputs first.")
            if kickoff_date is None: raise ValidationError("Select the actual kickoff date.")
            if fmt == "UNKNOWN": raise ValidationError("Explicitly choose the observed price format.")
            hl = number(h_line, optional=True); al = number(a_line, optional=True)
            if hl is not None and (al is None or abs(hl + al) > 1e-8):
                raise ValidationError("Home and away handicap lines must be exact opposites.")
            def convert(text):
                return None if not text.strip() else to_decimal(text, fmt)
            record = {"home": home.strip(), "away": away.strip(), "competition": competition.strip(), "source": source.strip(),
                      "kickoff_utc": datetime.combine(kickoff_date, kickoff_time, zone).isoformat(),
                      "captured_at_utc": datetime.combine(capture_date, capture_time, zone).isoformat(),
                      "home_1x2": number(p_home), "draw_1x2": number(p_draw), "away_1x2": number(p_away),
                      "hdp_line": hl, "hdp_home_decimal": convert(h_price), "hdp_away_decimal": convert(a_price),
                      "ou_line": number(o_line, optional=True), "over_decimal": convert(o_price), "under_decimal": convert(u_price),
                      "original_format": fmt, "prematch_verified": reviewed, "form_verified": form_verified,
                      "form_asof_utc": datetime.combine(form_date, form_time, zone).isoformat(),
                      "image_sha256": d.get("image_sha256", ""), "extractor_model": d.get("extractor_model", "manual"),
                      "form_image_sha256": d.get("form_image_sha256", ""), "form_extractor_model": d.get("form_extractor_model", "manual")}
            for prefix in ("h", "a"):
                percent = number(forms[prefix + "_win_percent"], optional=True)
                record[prefix + "_win_rate"] = percent / 100 if percent is not None else None
                for suffix in ("goals_for", "goals_against", "matches"):
                    record[prefix + "_" + suffix] = number(forms[prefix + "_" + suffix], optional=True)
            result = save_quote(STORE, record, ACTOR)
            st.success("Saved quote " + result["id"] + ". Original DATA was not changed.")
            refresh(); new_draft()
        except Exception as exc: error_message(exc, "Save quote")


def prediction_page(records):
    st.header("Predictions / \u0e1c\u0e25\u0e17\u0e33\u0e19\u0e32\u0e22")
    active = REGISTRY.active(); artifacts = {}
    for market, model_id in active.items():
        try: artifacts[market] = REGISTRY.load(model_id)
        except Exception as exc: error_message(exc, market + " model load")
    pending = pending_latest(records)
    if not pending:
        st.info("No valid pending fixtures. Invalid records appear in Data quality, not as completed matches."); return
    if not artifacts:
        st.info("Train a model explicitly in Model lab, then activate it. Loading this page never trains a model.")
    probabilities = {}
    for market, artifact in artifacts.items():
        usable = [r for r in pending if not market_errors(r, market)]
        try:
            # Exactly one batch inference per active market; no network reads inside cards.
            probs = infer_batch(artifact.model_id, canonical_json(usable), SETTINGS.namespace, REGISTRY)
            probabilities[market] = dict(zip((r["id"] for r in usable), probs))
        except Exception as exc: error_message(exc, market + " prediction")
    filter_text = st.text_input("Filter teams")
    view = [r for r in pending if filter_text.casefold() in (r["home"] + " " + r["away"]).casefold()]
    per_page = 10; pages = max(1, (len(view) + per_page - 1) // per_page)
    page = st.number_input("Page", min_value=1, max_value=pages, step=1)
    st.caption(f"{len(view)} pending fixtures; {per_page} per page. Times are displayed as stored UTC offsets.")
    for row in view[(page - 1) * per_page:page * per_page]:
        with st.container(border=True):
            st.subheader(row["home"] + "  vs  " + row["away"])
            st.caption(f"Fixture: {row['fixture_id']} | Kickoff: {row.get('kickoff_utc') or 'unknown (legacy)'}")
            cols = st.columns(2)
            for area, market in zip(cols, ("HDP", "OU")):
                with area:
                    artifact = artifacts.get(market); p = probabilities.get(market, {}).get(row["id"])
                    st.markdown("#### " + market)
                    if artifact is None or p is None:
                        st.info("No active model or valid input for this market."); continue
                    choice = assess_prediction(artifact, row, p, SETTINGS.policy)
                    side_label = ("Home", "Away") if market == "HDP" else ("Over", "Under")
                    st.write(f"Reference side: **{side_label[choice['side']]}** | **{choice['status']}**")
                    st.metric("P(positive return), including all outcomes", f"{choice['positive_probability']:.1%}")
                    st.caption(f"P(push) {choice['push_probability']:.1%} | P(negative) {choice['negative_probability']:.1%}")
                    st.write(f"Conditional P(positive | no push): {choice['positive_given_no_push']:.1%}")
                    st.write(f"EV / unit: {choice['ev']:+.3f} | Stress EV: {choice['stress_ev']:+.3f}")
                    if choice["reasons"]: st.caption(" | ".join(choice["reasons"]))
                    if choice["paper_fraction"]:
                        st.caption(f"Unreserved paper-only Kelly cap: {choice['paper_fraction']:.2%}. Portfolio limits are applied when saved.")
                    with st.expander("Five-state probabilities and model identity"):
                        st.dataframe(pd.DataFrame({"state": list(choice["state_probabilities"]), "probability": list(choice["state_probabilities"].values())}), hide_index=True)
                        st.code(artifact.model_id)
            if st.button("Freeze pre-match prediction for later evaluation", key="freeze_" + row["id"], disabled=not CAN_WRITE or not artifacts):
                try:
                    require_role()
                    saved = freeze_predictions(STORE, row["id"], artifacts, SETTINGS.policy, ACTOR)
                    st.success(f"Saved/reused {len(saved)} immutable prediction records. No actual bet was placed.")
                    refresh()
                except Exception as exc: error_message(exc, "Freeze prediction")
            with st.form("score_" + row["id"]):
                score = st.text_input("Verified full-time score (home-away)")
                reason = st.text_input("Result source / verification note")
                if st.form_submit_button("Save result", disabled=not CAN_WRITE):
                    try:
                        require_role()
                        save_result(STORE, row["fixture_id"], score, "FINAL", ACTOR, reason, expected_previous_id="")
                        refresh(); st.rerun()
                    except Exception as exc: error_message(exc, "Save result")


def run_training(rows, market, config, policy):
    data_hash = training_digest(rows, market, config)
    for manifest in REGISTRY.list():
        if manifest["training_digest"] == data_hash and manifest["report"]["policy_config"] == asdict(policy):
            artifact = REGISTRY.load(manifest["model_id"])
            return {"model_id": artifact.model_id, "gate_pass": artifact.report["gate_pass"]}
    artifact = train_market(rows, market, config, policy)
    artifact = REGISTRY.save(artifact)
    return {"model_id": artifact.model_id, "gate_pass": artifact.report["gate_pass"]}


def model_page(records):
    st.header("Model lab / \u0e1d\u0e36\u0e01\u0e41\u0e25\u0e30\u0e1b\u0e23\u0e30\u0e40\u0e21\u0e34\u0e19\u0e42\u0e21\u0e40\u0e14\u0e25")
    st.info("Legacy history without verified times is RESEARCH ONLY. No dates are fabricated. Only explicit training creates model artifacts.")
    st.caption("Candidates: smoothed line-family baseline, one HGB, and optional three-model soft voting. Selection uses validation log loss; the final test is not tuned.")
    a, b = st.columns(2)
    for area, market in zip((a, b), ("HDP", "OU")):
        with area:
            usable, mode = training_records(records, market, SETTINGS.training)
            st.metric(market + " usable unique fixtures", len(usable))
            st.caption(mode)
            if st.button("Train / evaluate " + market, disabled=ROLE != "admin" or len(usable) < SETTINGS.training.min_rows):
                try:
                    require_role("admin", writes=False)
                    key = digest([market, training_digest(records, market, SETTINGS.training), asdict(SETTINGS.policy)])
                    TRAIN_QUEUE.submit(OWNER, key, "Training " + market, run_training, records, market, SETTINGS.training, SETTINGS.policy)
                    st.rerun()
                except Exception as exc: error_message(exc, market + " training")
    training_status()
    manifests = REGISTRY.list()
    if not manifests: return
    selected = st.selectbox("Inspect saved model / rollback target", [m["model_id"] for m in manifests])
    manifest = next(m for m in manifests if m["model_id"] == selected)
    r = manifest["report"]
    st.write(f"**{r['selected_model']}** | {r['mode']} | trained {manifest['trained_at_utc']}")
    if r["gate_pass"]: st.success("Statistical readiness gate passed for PAPER evaluation only.")
    else: st.warning("Research only: " + ", ".join(r["gate_reasons"]))
    st.dataframe(pd.DataFrame([{"partition": k, "n": v} for k, v in r["partition_counts"].items()]), hide_index=True)
    st.dataframe(pd.DataFrame([
        {"model": "selected", "n_test": r["test"]["n"], "log_loss": r["test"]["log_loss"], "brier": r["test"]["brier"], "5-state accuracy": r["test"]["five_state_accuracy"]},
        {"model": "family baseline", "n_test": r["test_baseline"]["n"], "log_loss": r["test_baseline"]["log_loss"], "brier": r["test_baseline"]["brier"], "5-state accuracy": r["test_baseline"]["five_state_accuracy"]},
    ]), hide_index=True)
    st.caption(f"Directional KPI: {r['test']['directional_correct']}/{r['test']['directional_n']} (conditional threshold >=55%, excludes pushes). Not the full policy hit rate.")
    st.write("Numerical-policy test (before live safety gates):", r["test_policy"])
    reliability = pd.DataFrame(r["test"]["reliability"])
    if not reliability.empty:
        st.markdown("#### Reliability: conditional positive return, excluding pushes")
        st.dataframe(reliability, hide_index=True)
        st.line_chart(reliability.set_index("forecast")[["observed"]])
    if st.button("Activate selected model (or rollback)", disabled=ROLE != "admin"):
        try:
            require_role("admin", writes=False)
            REGISTRY.activate(selected); infer_batch.clear(); st.success("Activated. Research models remain gated.")
        except Exception as exc: error_message(exc, "Activate model")
    st.download_button("Download model evaluation manifest", canonical_json(manifest), file_name=selected + ".json", mime="application/json")


def dashboard_page(snapshot, records, issues):
    st.header("Data quality & evaluation / \u0e04\u0e38\u0e13\u0e20\u0e32\u0e1e\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25")
    a, b, c, d = st.columns(4)
    a.metric("Observations (not unique fixtures)", len(records))
    b.metric("Settled observations", sum(r["status"] == "SETTLED" for r in records))
    c.metric("Pending fixtures", len(pending_latest(records)))
    d.metric("Invalid observations", sum(r["status"] == "INVALID" for r in records))
    if issues:
        with st.expander(f"Quality issues ({len(issues)} findings; multiple findings may affect one row)", expanded=True):
            st.dataframe(pd.DataFrame(issues), hide_index=True, height=240)
    form_n = sum(all(r.get(k) not in (None, "") for k in FORM_COLUMNS) for r in records)
    st.caption(f"Complete form: {form_n}/{len(records)} observations. Missing form is not silently filled with zero.")
    st.markdown("#### Frozen pre-match forecasts with observed results")
    evaluated = evaluate_frozen(snapshot)
    st.caption(f"Evaluable frozen forecasts: {len(evaluated)}/{len(snapshot['predictions'])}. Pending/void outcomes, missing sources, changed quotes, and invalid forecast records are excluded; they are not counted as losses or wins.")
    if not evaluated:
        st.info("No eligible frozen forecasts with outcomes yet. This is not replaced by old spreadsheet-rule results or by re-predicting training rows.")
    else:
        frame = pd.DataFrame(evaluated)
        model_filter = st.selectbox("Model version", sorted(frame.model_id.unique()))
        frame = frame[frame.model_id == model_filter]
        policies = sorted(frame.policy_version.unique())
        policy_filter = st.selectbox("Policy version", policies)
        frame = frame[frame.policy_version == policy_filter]
        paper_only = st.checkbox("Show paper candidates only", value=True)
        if paper_only: frame = frame[frame.status == "PAPER_CANDIDATE"]
        if frame.empty:
            st.info("No records meet this filter. Research forecasts are stored separately.")
        else:
            a, b, c = st.columns(3)
            a.metric("Forecasts / unique fixtures", f"{len(frame)} / {frame.fixture_id.nunique()}")
            c.metric("Average log loss", f"{frame.log_loss.mean():.4f}")
            b.metric("Flat-unit SIMULATED yield", f"{frame.flat_unit_return.mean():+.2%}")
            st.caption("Yield assumes one unit at the saved observed price per displayed forecast. It is not actual profit or a compound bankroll result.")
            st.dataframe(frame, hide_index=True)
            # Aggregate simultaneous/overlapping forecasts by settlement-availability date.
            chart = frame.assign(day=frame.result_available_at_utc.str[:10]).groupby("day").flat_unit_return.sum().cumsum()
            st.line_chart(chart.rename("cumulative simulated units"))
    with st.expander("Inspect input observations (old formulas are excluded)"):
        cols = ["id", "fixture_id", "home", "away", "source_row", "status", "score", "kickoff_utc", "hdp_line", "ou_line"]
        st.dataframe(pd.DataFrame(records).reindex(columns=cols), hide_index=True, height=300)
    st.download_button("Download data-quality findings (JSON)", canonical_json(issues), "data_quality.json", "application/json")


def settings_page(snapshot, records):
    st.header("Settings & operations / \u0e15\u0e31\u0e49\u0e07\u0e04\u0e48\u0e32")
    st.write(f"Backend: **{SETTINGS.backend}** | role: **{ROLE}** | writes: **{SETTINGS.enable_writes}**")
    st.warning("Use only ONE app process/replica as writer for Google Sheets. Process locks do not coordinate other deployments or manual edits.")
    if SETTINGS.backend == "sheets":
        st.caption("Initialize creates only AI_DATA_V6, AI_RESULTS_V6 and AI_PREDICTIONS_V6. Existing DATA and formulas remain untouched.")
        acknowledged = st.checkbox("I backed up the workbook and approve creating the three v6 tabs")
        if st.button("Initialize v6 tabs", disabled=ROLE != "admin" or not SETTINGS.enable_writes or not acknowledged):
            try:
                require_role("admin"); STORE.initialize(); refresh(); st.success("v6 tabs initialized.")
            except Exception as exc: error_message(exc, "Initialize tabs")
    else:
        st.caption("Local backend stores NEW records in runtime/<namespace>/observations.sqlite3. The uploaded DATA workbook is kept read-only.")
        workbook_file = st.file_uploader("Import original DATA workbook (.xlsx)", type=["xlsx"])
        if st.button("Validate and import workbook", disabled=ROLE != "admin" or not SETTINGS.enable_writes or workbook_file is None):
            try:
                require_role("admin")
                loaded, found = STORE.import_legacy(workbook_file.getvalue())
                refresh(); st.success(f"Imported {len(loaded)} observations; {len(found)} quality findings. No original formulas were edited.")
                st.rerun()
            except Exception as exc: error_message(exc, "Import workbook")
    if SETTINGS.backend == "sheets" and STORE.uncertain:
        st.error("An earlier write has an unknown outcome. Further writes are blocked until reconciled.")
        operation_id = STORE.uncertain["record"]["id"]
        st.code(operation_id)
        if st.button("Reconcile unknown write", disabled=ROLE != "admin"):
            try:
                require_role("admin", writes=False)
                if STORE.reconcile_uncertain():
                    refresh(); st.rerun()
                else:
                    st.warning("The record is not visible yet. This alone does not prove that the earlier write failed.")
            except Exception as exc: error_message(exc, "Reconcile uncertain write")
        recovery_reason = st.text_input("Administrator's reconciliation evidence / reason")
        confirm_absent = st.checkbox("I independently inspected the workbook and confirmed the earlier operation did not commit")
        if st.button("Release blocked operation after manual review", disabled=ROLE != "admin" or not confirm_absent):
            try:
                require_role("admin")
                STORE.release_uncertain_after_review(operation_id, ACTOR, recovery_reason)
                refresh(); st.rerun()
            except Exception as exc: error_message(exc, "Release uncertain write")
    api_key = os.environ.get("GEMINI_API_KEY") or str(SECRET_VALUES.get("GEMINI_API_KEY", ""))
    st.markdown("#### Gemini (optional)")
    model = st.text_input("Model ID accessible to your API project", value=st.session_state.get("gemini_model_selected", SETTINGS.gemini_model))
    if st.button("Use this model for this session", disabled=not CAN_EDIT):
        st.session_state.gemini_model_selected = model.strip(); st.success("Session model selected. Extraction validates the request at runtime.")
    if st.button("List model IDs from the API", disabled=not CAN_EDIT or not api_key):
        try:
            require_role(writes=False)
            st.write(available_models(api_key))
        except Exception as exc: error_message(exc, "List Gemini models")
    st.caption("Secrets are read from environment variables or .streamlit/secrets.toml. Do not upload credentials or commit them to source control.")
    with st.expander("Result correction / void (administrator only)"):
        fixtures = {}
        for r in records: fixtures.setdefault(r["fixture_id"], r)
        selected = st.selectbox("Fixture to correct", [""] + list(fixtures), format_func=lambda key: "Select..." if not key else f"{fixtures[key]['home']} vs {fixtures[key]['away']} | {fixtures[key]['status']} | {key[-8:]}")
        if selected:
            events = [r for r in snapshot["results"] if r["fixture_id"] == selected]
            previous_id = events[-1]["id"] if events else ""
            with st.form("correction_" + selected):
                score = st.text_input("Correct verified full-time score", value=fixtures[selected].get("score", ""))
                status = st.selectbox("Outcome status", ["FINAL", "VOID"])
                reason = st.text_input("Required verification source and correction explanation")
                confirm = st.checkbox("I verified this correction; preserve the previous entry in the audit trail")
                if st.form_submit_button("Append correction event", disabled=ROLE != "admin" or not SETTINGS.enable_writes):
                    try:
                        require_role("admin")
                        if not confirm: raise ValidationError("Explicit confirmation is required.")
                        save_result(STORE, selected, score, status, ACTOR, reason, allow_correction=True, expected_previous_id=previous_id)
                        refresh(); st.rerun()
                    except Exception as exc: error_message(exc, "Correct result")
    with st.expander("Model and policy configuration (read-only here)"):
        st.json({"training": asdict(SETTINGS.training), "policy": asdict(SETTINGS.policy)})
    st.download_button("Backup app-managed data as JSON", canonical_json({k: snapshot[k] for k in ("quotes", "results", "predictions")}),
                       "football_ai_v6_backup.json", "application/json")
    st.caption("Back up runtime/ and the complete Google workbook separately. JSON above excludes credentials and model binaries.")


st.title("Football AI v6")
st.caption("Validated data. Explicit training. Honest probabilities. Frozen pre-match evaluation. No guaranteed accuracy or profit.")
with st.sidebar:
    st.markdown("### Python AI workspace")
    st.caption(f"v{__version__} | {ACTOR} | {ROLE}")
    page = st.radio("Workspace", ["Capture & review", "Predictions", "Data quality & evaluation", "Model lab", "Settings"])
    if st.button("Refresh data (does not retrain)"):
        refresh(); st.rerun()
    active = REGISTRY.active()
    for market in ("HDP", "OU"):
        st.caption(f"{market}: {active.get(market, 'no active model')}")
    if not SETTINGS.enable_writes: st.warning("Read-only data mode. Set enable_writes=true to save/import.")
    if SETTINGS.local_only: st.info("Local admin mode: do not expose this service through a public proxy.")
    elif st.button("Sign out"): st.logout()
try:
    SNAPSHOT = load_snapshot(SETTINGS.namespace, STORE.revision, STORE)
    RECORDS, ISSUES = materialize(SNAPSHOT)
except Exception as exc:
    error_message(exc, "Load data")
    st.info("The application fails closed on storage/schema errors. Check the DATA headers and access permissions.")
    st.stop()

# Only the selected page executes. No eager hidden tabs and no automatic model training.
if page == "Capture & review": entry_page(RECORDS)
elif page == "Predictions": prediction_page(RECORDS)
elif page == "Data quality & evaluation": dashboard_page(SNAPSHOT, RECORDS, ISSUES)
elif page == "Model lab": model_page(RECORDS)
else: settings_page(SNAPSHOT, RECORDS)
