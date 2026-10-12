# P2_FORM5_SPLIT_V2 - short entrypoint; CSS is in style.css.
from pathlib import Path
import ast
import hashlib
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUILD_ID = "P2_FORM5_SPLIT_V2"
EXPECTED = {'p2_app.py': '2152b8c245fc76062e5dbb8c2d6bc2042fc6ccd5b7b4e64fe4e9352293a95f3a', 'form_data.py': '26e82e7f1bff0971d16a59999ae7fed4b9fc7873c119507e6fe626bc00aa9d60', 'style.css': '80ba38de41118b2bc4fb49d97b1f286bf434eabb48ed054419cf7f0fe0fe45bb'}
st.set_page_config(page_title='ระบบวิเคราะห์ราคาบอล VIP', page_icon='⚽', layout='wide')
st.caption("Build: " + BUILD_ID)
print("Starting " + BUILD_ID + " from " + str(ROOT / "app.py"), flush=True)
for name, expected in EXPECTED.items():
    path = ROOT / name
    try:
        text = path.read_text(encoding="utf-8-sig")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            raise ValueError("File is incomplete or differs from this release. Upload the full file from the ZIP.")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
        st.error(BUILD_ID + " | " + name + ": " + str(exc))
        st.info("Upload ALL files from P2_Form5_SplitFix.zip into the same folder as app.py, then commit to the deployed branch.")
        st.stop()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    runpy.run_path(str(ROOT / "p2_app.py"), run_name="__main__")
except ModuleNotFoundError as exc:
    st.error("Missing dependency: " + str(exc.name) + ". Install this release's requirements.txt.")
    st.stop()
