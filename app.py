        with st.expander(f'ข้อมูลที่ต้องตรวจในชีต {len(invalid)} รายการ (ไม่บันทึกทับอัตโนมัติ)'):
            display = invalid[['_SheetRow', 'ทีมเหย้า', 'ทีมเยือน', 'ผลสกอร์', '_Issue']].copy()
            display['ผลสกอร์'] = display['ผลสกอร์'].map(str)
            st.dataframe(display.rename(columns={'_SheetRow': 'แถว DATA', '_Issue': 'ประเด็นที่พบ'}), **_stretch())
    pending = _t2_pending_view(raw, ai)
    if pending.empty:
        st.success('ไม่มีรายการที่ชื่อทีมครบและช่องผลสกอร์ว่างใน DATA')
        return
 
    has_prediction, has_signal = _t2_flags(pending)
    st.markdown(_cy_chip_row([
        ('รอผลสกอร์ทั้งหมด', f'{len(pending):,}', NEON_CYAN),
        ('มีผลทำนายครบ', int(has_prediction.sum()), NEON_GREEN),
        ('มีสัญญาณ ⭐', int(has_signal.sum()), NEON_YEL),
        ('ลงสกอร์ได้ (ไม่มีผลทำนาย)', int((~has_prediction).sum()), NEON_GREY)]), unsafe_allow_html=True)
    if has_prediction.any():
        render_monitor_header(pending[has_prediction])
 
    options = {'ทั้งหมด': np.ones(len(pending), dtype=bool), 'มีผลทำนาย': has_prediction, 'เฉพาะมีสัญญาณ ⭐': has_signal}
    mode = st.radio('แสดงรายการ', list(options), horizontal=True, key='_t2_filter')
    view = pending[options[mode]]
    if view.empty:
        st.markdown(_cy_empty('ไม่มีรายการตามตัวกรองนี้ — ลองเลือก "ทั้งหมด"'), unsafe_allow_html=True)
        return
 
    page_size = 20
    pages = (len(view) + page_size - 1) // page_size
    page_key = '_t2_page'
    old_page = st.session_state.get(page_key, 1)
    if not isinstance(old_page, int) or not 1 <= old_page <= pages:
        st.session_state[page_key] = 1
    page = st.selectbox('หน้ารายการรอผล', list(range(1, pages + 1)), key=page_key)
    start = (page - 1) * page_size
    st.caption(f'แสดง {len(view):,} รายการ — รายการที่ {start + 1} ถึง {min(start + page_size, len(view))}')
    render_prediction_cards(view.iloc[start:start + page_size], wb, offset=start)
 
 
def _t3_quality(wb):
    if wb is None:
        return
    try:
        raw = _t2_load_data(wb, _t2_workbook_key(wb), JOBS.data_version)
    except Exception as exc:
        _t2_error(exc, 'ตรวจคุณภาพข้อมูลสำหรับแท็บ 3')
        return
    counts = {key: int(raw['_ScoreStatus'].eq(key).sum()) for key in ('finished', 'pending', 'invalid')}
    with st.container(key='hud_rawstatus'):
        st.markdown(_cy_title('DATA STATUS', 'อ่านตรงจากคอลัมน์ M ของชีต DATA'), unsafe_allow_html=True)
        st.markdown(_cy_chip_row([
            ('สกอร์รูปแบบถูกต้อง', counts['finished'], NEON_GREEN),
            ('รอลงผลสกอร์', counts['pending'], NEON_CYAN),
            ('ต้องตรวจสอบ', counts['invalid'], NEON_RED if counts['invalid'] else NEON_GREY),
            ('แถวทั้งหมด', len(raw), NEON_MAG)]), unsafe_allow_html=True)
        st.markdown("<div class='hud-note'>ตรวจจากชื่อทีมและคอลัมน์ M โดยไม่ใช้สูตรเก่า "
                    "ตัวเลขนี้ไม่ใช่จำนวนตัวอย่างที่โมเดลฝึกจริง (ดูแท็บ \"ข้อมูลและการฝึก\")</div>", unsafe_allow_html=True)
 
# ----------------------------------------
# หัวข้อหลัก (Tabs)
# ----------------------------------------
st.markdown(f"<h1 class='page-title'>⚽ ระบบวิเคราะห์ราคาบอล VIP & AI ({ML_PATCH_VERSION})</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-sub'>วิเคราะห์ HDP และสูง/ต่ำจากภาพราคา · บันทึกสกอร์ · ตรวจคุณภาพโมเดลและข้อมูลในที่เดียว</p>", unsafe_allow_html=True)
 
if not SHEET_WRITES_ENABLED:
    st.warning("READ-ONLY REVIEW BUILD: Google Sheets writes are disabled. Core tests do not certify production readiness.")

AI = get_ai()
 
tab1, tab2, tab3 = st.tabs(["📸 อัปโหลดราคาบอล", "📡 สัญญาณการทำนาย", "📊 Dashboard"])
 
with tab1:
    st.info("Research-only build: verify extracted prices and form statistics. No stake sizing or accuracy guarantee.")
    fmt_choice = st.selectbox("รูปแบบค่าน้ำ", list(FMT_CHOICES.keys()))
    up_n = st.session_state.get("up_n", 0)
    uploaded_files = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{up_n}")
    if uploaded_files and workbook:
        for f in uploaded_files: JOBS.pool.submit(process_job, JOBS.create(f.name, f.getvalue(), FMT_CHOICES[fmt_choice], st.session_state["form5_owner_id"]), workbook, model)
        st.session_state["up_n"] = up_n + 1
        st.rerun()
 
    my_jobs = [j for j in JOBS.snapshot() if j.get("owner_id") == st.session_state["form5_owner_id"]]
    if my_jobs or st.session_state.get("form5_editing"):
        st.button("🧹 เคลียข้อมูลหน้านี้", key="clear_upload", on_click=clear_upload_page, help="ล้างรายการงาน การ์ดสัญญาณ ฟอร์ม 5 นัด และไฟล์ที่เลือกไว้ในหน้านี้")
 
    polling = any(j["status"] in ("queued", "reading", "saving") for j in my_jobs)
    if hasattr(st, "fragment"): st.fragment(run_every=2 if polling else None)(render_jobs)(polling)
    else: render_jobs(polling)
    render_form5_editor()
 
with tab2:
    _t2_render_tab(workbook, AI)
 
with tab3:
    panel_header("สรุปผลและคุณภาพข้อมูล", f"เอนจิน {ML_PATCH_VERSION} · ข้อมูลดิบ ผลย้อนหลัง และคุณภาพโมเดลแสดงแยกกันชัดเจน")
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        st.button("รีเฟรชข้อมูล", key="refresh_dashboard", on_click=_t2_refresh)
        _t3_quality(workbook)
        if not isinstance(AI, dict):
            st.warning("ยังโหลดรายงานโมเดลไม่ได้ แต่สามารถตรวจข้อมูลและลงสกอร์ในแท็บ 2 ได้")
        elif AI.get("status") == "error":
            st.warning("โหลดรายงานโมเดลไม่สำเร็จ: " + str(AI.get("msg", "")) + " — ยังตรวจข้อมูลและลงสกอร์ในแท็บ 2 ได้")
        else:
            render_dashboard(AI)
