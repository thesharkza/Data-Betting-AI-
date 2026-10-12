# P2_FORM5_STARTUP_V3 - replaces only the Split V2 launcher.
# No exact-text checksum gate. Checks below are NOT a security/integrity audit.
from pathlib import Path
import ast
import runpy
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
BUILD_ID = "P2_FORM5_STARTUP_V3"
REQUIRED = {
    "p2_app.py": {"JobManager", "render_team_form", "render_review", "load_dashboard_data"},
    "form_data.py": {"TeamForm", "ValidationError", "validate_quote", "prepare_headers", "save_match"},
}


def validate_python(path: Path, required: set[str]) -> int:
    # Read only: do not import the module, execute it, or contact any API here.
    source = path.read_text(encoding="utf-8-sig")
    if not source.strip():
        raise ValueError("File is empty. Replace this file with its complete copy.")
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")  # Also catches e.g. a return outside a function.
    definitions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(required - definitions)
    if missing:
        raise ValueError(
            "Required definitions missing: " + ", ".join(missing)
            + f". This copy has {len(source.splitlines())} lines; it may be partial or a different version."
        )
    if path.name == "p2_app.py":
        # Detect common partial pastes that still have valid Python syntax.
        # This is a compatibility check, not a proof that all behavior is correct.
        tabs = {
            item.context_expr.id: node
            for node in tree.body if isinstance(node, ast.With)
            for item in node.items if isinstance(item.context_expr, ast.Name)
        }
        for tab, call_name in (("tab1", "render_review"), ("tab2", "cached_rate_history")):
            section = tabs.get(tab)
            has_call = section is not None and any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == call_name for node in ast.walk(section)
            )
            if not has_call:
                raise ValueError(
                    f"Required UI section missing: {tab}/{call_name}. "
                    f"This copy has {len(source.splitlines())} lines; replace p2_app.py with its complete copy."
                )
    return len(source.splitlines())


def main() -> None:
    st.set_page_config(page_title='ระบบวิเคราะห์ราคาบอล VIP', page_icon='⚽', layout='wide')
    st.caption("Build: " + BUILD_ID)
    for name, required in REQUIRED.items():
        try:
            validate_python(ROOT / name, required)
        except SyntaxError as exc:
            st.error(f"{BUILD_ID} | {name}: SyntaxError at line {exc.lineno}, column {exc.offset}: {exc.msg}")
            st.info("Repair the named file or replace it with the complete copy. No application code was started.")
            st.stop()
            return
        except (OSError, UnicodeError, ValueError) as exc:
            st.error(f"{BUILD_ID} | {name}: {exc}")
            st.info("Keep app.py, p2_app.py, form_data.py and style.css together. This check does not compare file hashes.")
            st.stop()
            return
    try:
        css = (ROOT / "style.css").read_text(encoding="utf-8-sig")
        if not css.strip():
            raise ValueError("style.css is empty; restore the stylesheet.")
    except (OSError, UnicodeError, ValueError) as exc:
        st.error(f"{BUILD_ID} | style.css: {exc}")
        st.stop()
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        runpy.run_path(str(ROOT / "p2_app.py"), run_name="__main__")
    except ModuleNotFoundError as exc:
        st.error(f"{BUILD_ID} | Missing module: {exc.name}. Check requirements.txt and the deployment logs.")
        st.stop()


if __name__ == "__main__":
    main()
