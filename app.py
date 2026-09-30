import streamlit as st
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # เพิ่มไลบรารีนี้สำหรับกราฟขั้นสูง
import gc
import time

# ----------------------------------------
# ตั้งค่าหน้าเว็บ Streamlit (กำหนดให้รองรับ CSS พิเศษ)
# ----------------------------------------
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP", page_icon="⚽", layout="wide")

# CSS สำหรับการ์ดด้านบน (3D Modern & Glassmorphism)
st.markdown("""
<style>
    .metric-card {
        border-radius: 20px;
        padding: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15), inset 0 2px 2px rgba(255,255,255,0.2);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.25), inset 0 2px 2px rgba(255,255,255,0.3);
    }
    .bg-blue { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    .bg-purple { background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%); }
    .bg-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    
    .card-title { font-size: 1.1rem; font-weight: 500; opacity: 0.9; margin-bottom: 10px; letter-spacing: 1px; }
    .card-value { font-size: 3.5rem; font-weight: 900; margin: 0; text-shadow: 2px 4px 6px rgba(0,0,0,0.2); line-height: 1.2; }
    .card-unit { font-size: 1.2rem; font-weight: 400; opacity: 0.8; }
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
            creds_path = 'credentials.json'
            client = gspread.service_account(filename=creds_path)
        
        workbook = client.open("ข้อมูลราคาบอลสกัดจากภาพ")
        return workbook
    except Exception as e:
        st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
        return None

workbook = init_gspread()
model = genai.GenerativeModel('gemini-3.7-flash')

def clean_raw_data(raw_text):
    raw_data = [item.strip() for item in raw_text.split(',')]
    row_data = []
    for i, item in enumerate(raw_data):
        if i < 2:
            row_data.append(item) 
        else:
            cleaned = re.sub(r'^[oOuU\s]+', '', item)
            try:
                row_data.append(float(cleaned) if '.' in cleaned else int(cleaned))
            except:
                row_data.append(cleaned)
    return row_data[:12]

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
# หัวข้อหลักของแอป
# ----------------------------------------
st.title("⚽ ระบบวิเคราะห์ราคาบอล VIP")

tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "📊 Dashboard แบบ 3 มิติ"])

with tab1:
    st.subheader("สกัดราคาบอล & วิเคราะห์ VIP")
    st.write("อัปโหลดภาพตารางราคาเพื่อสกัดข้อมูลส่งเข้า Google Sheets อัตโนมัติ")
    
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="ภาพที่อัปโหลด", width=600)
        
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
                        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
                        response = model.generate_content([{"mime_type": "image/jpeg", "data": img_bytes}, prompt])
                    
                    gc.collect()

                    if not response or not response.text:
                        st.error("❌ AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                    else:
                        with st.spinner("📊 กำลังบันทึกและรอ Google Sheets ประมวลผลลัพธ์..."):
                            row_data_to_sheet = clean_raw_data(response.text.strip())
                            
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
                            
                            st.markdown(f"""
                            <div style="padding: 25px; background: linear-gradient(to right, #f8fafc, #f1f5f9); border-left: 6px solid #1abc9c; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                                <h3 style="color: #0f172a; margin-top: 0;">⚽ {row_data_to_sheet[0]} <span style="color:#94a3b8;">vs</span> {row_data_to_sheet[1]}</h3>
                                <div style="margin-top: 15px; font-size: 1.1rem; color: #334155;">
                                    <p style="margin: 5px 0;"><strong>🎯 แนะนำลงทุน:</strong> <span style="color: #e11d48; font-weight: bold;">{rec}</span></p>
                                    <p style="margin: 5px 0;"><strong>💰 Money Mgt:</strong> <span style="color: #0284c7; font-weight: bold;">{mm_text}</span></p>
                                    <p style="margin: 5px 0;"><strong>📊 สถิติความเชื่อมั่น:</strong> <span style="color: #047857; font-weight: bold;">{confidence}</span></p>
                                    <p style="margin: 5px 0;"><strong>🚨 เช็กราคา:</strong> {radar}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                finally:
                    gc.collect()

# ----------------------------------------
# TAB 2 : Dashboard อัปเกรดใหม่ (Clean Chart + SaaS Table)
# ----------------------------------------
with tab2:
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        try:
            # --- ดึงข้อมูลจากชีต DATA เพื่อทำกราฟโดนัท ---
            ws_data = workbook.sheet1
            data = ws_data.get_all_records()
            df = pd.DataFrame(data)
            
            # --- ดึงข้อมูลจากชีต สรุปสถิติ เพื่อทำกราฟแท่งแนวนอนและตาราง ---
            ws_stats = workbook.worksheet("สรุปสถิติ")
            data_stats = ws_stats.get_all_records()
            df_stats = pd.DataFrame(data_stats)
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

                # --- 1. การ์ด 3D สรุปผล ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card bg-blue">
                        <div class="card-title">🏟️ แมตช์ทั้งหมด (จบแล้ว)</div>
                        <p class="card-value">{total} <span class="card-unit">คู่</span></p>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card bg-purple">
                        <div class="card-title">🔥 คะแนนชนะสะสม (Win Score)</div>
                        <p class="card-value">{total_wins} <span class="card-unit">แต้ม</span></p>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card bg-green">
                        <div class="card-title">🏆 Win Rate รวมทั้งหมด</div>
                        <p class="card-value">{win_rate} <span class="card-unit">%</span></p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_chart1, col_chart2 = st.columns(2)

                # --- 2. กราฟวงกลมแบบเจาะรู (Donut 3D) ---
                with col_chart1:
                    st.markdown("<h4 style='text-align: center; color: #334155;'>สัดส่วนผลลัพธ์รวมทั้งหมด</h4>", unsafe_allow_html=True)
                    pull_array = [0.1 if label == 'ชนะเต็ม' else 0 for label in df_comp['ผลเปรียบเทียบ'].unique()]
                    
                    pie_fig = px.pie(df_comp, names='ผลเปรียบเทียบ', color='ผลเปรียบเทียบ', 
                                     color_discrete_map={
                                         'ชนะเต็ม': '#10b981', 'ชนะครึ่ง': '#34d399', 
                                         'แพ้เต็ม': '#f43f5e', 'แพ้ครึ่ง': '#fb7185', 'เจ๊า': '#94a3b8'
                                     }, hole=0.5)
                    pie_fig.update_traces(pull=pull_array, textinfo='percent+label', textfont_size=14,
                                          marker=dict(line=dict(color='#ffffff', width=2)))
                    pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), autosize=True, showlegend=False)
                    st.plotly_chart(pie_fig, use_container_width=True)

                # --- 3. กราฟ Win-Rate แยกตามกฎ (Clean & Readability Focus) ---
                with col_chart2:
                    st.markdown("<h4 style='text-align: center; color: #334155;'>🎯 อัตราชนะแยกตามกฎ (Win Rate %)</h4>", unsafe_allow_html=True)
                    
                    if not df_stats.empty and 'อัตราชนะ' in df_stats.columns:
                        plot_df = df_stats.copy()
                        rule_col = plot_df.columns[0]
                        # แปลงเปอร์เซ็นต์เป็นตัวเลข Float
                        plot_df['WinRate_Val'] = plot_df['อัตราชนะ'].astype(str).str.replace('%', '').apply(lambda x: float(x) if x.replace('.','').isdigit() else 0)
                        
                        # เอาเฉพาะกฎที่มีค่าเปอร์เซ็นต์มากกว่า 0 และจัดเรียงจากน้อยไปมาก
                        plot_df = plot_df[plot_df['WinRate_Val'] > 0].sort_values('WinRate_Val', ascending=True)
                        
                        # กำหนดสีแท่งกราฟ (เขียว = 60+, ส้ม = 55+, แดง = ต่ำกว่า 55)
                        colors = ['#10b981' if val >= 60 else '#f59e0b' if val >= 55 else '#ef4444' for val in plot_df['WinRate_Val']]
                        
                        fig_bar = go.Figure(go.Bar(
                            x=plot_df['WinRate_Val'],
                            y=plot_df[rule_col],
                            orientation='h',
                            marker=dict(color=colors),
                            text=plot_df['อัตราชนะ'],
                            textposition='inside',
                            insidetextfont=dict(color='white', size=13, family='Arial Black')
                        ))
                        
                        fig_bar.update_layout(
                            xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
                            yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color='#475569')),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=20, t=10, b=10),
                            height=400, showlegend=False
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("ยังไม่มีข้อมูลสถิติเพียงพอสำหรับสร้างกราฟ")
                        
            # --- 4. ตาราง Leaderboard (SaaS Glassmorphism Style) ---
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #1e293b; text-align: center; margin-bottom: 20px;'>📋 เจาะลึกสถิติรายกฎ (Leaderboard)</h3>", unsafe_allow_html=True)
            
            if not df_stats.empty:
                # สร้างโค้ด HTML CSS สำหรับตารางสไตล์ Modern Web
                html_table = """
                <style>
                    .modern-table-container { width: 100%; overflow-x: auto; padding-bottom: 20px; }
                    .modern-table {
                        width: 100%; border-collapse: separate; border-spacing: 0 12px;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }
                    .modern-table th {
                        background-color: transparent; color: #64748b; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 15px;
                        text-align: center; border-bottom: 2px solid #e2e8f0; white-space: nowrap;
                    }
                    .modern-table th:first-child { text-align: left; }
                    .modern-table tbody tr {
                        background-color: #ffffff;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        border-radius: 12px; transition: all 0.3s ease;
                    }
                    .modern-table tbody tr:hover {
                        transform: translateY(-3px) scale(1.005);
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                    }
                    .modern-table td {
                        padding: 18px 15px; color: #334155; text-align: center;
                        vertical-align: middle; border: none; font-size: 1.05rem;
                    }
                    .modern-table td:first-child { 
                        border-top-left-radius: 12px; border-bottom-left-radius: 12px; 
                        text-align: left; font-weight: 600; color: #0f172a;
                    }
                    .modern-table td:last-child { border-top-right-radius: 12px; border-bottom-right-radius: 12px; }
                    
                    /* สร้างป้ายกำกับ Capsule สำหรับเปอร์เซ็นต์ */
                    .badge-green { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 0.9em; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);}
                    .badge-orange { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 0.9em; box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3);}
                    .badge-red { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 0.9em; box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);}
                </style>
                <div class="modern-table-container">
                <table class="modern-table">
                    <thead><tr>
                """
                # สร้างหัวตาราง
                for col in df_stats.columns:
                    html_table += f"<th>{col}</th>"
                html_table += "</tr></thead><tbody>"
                
                # นำข้อมูลจาก df_stats มาใส่ในตารางทีละแถว
                for _, row in df_stats.iterrows():
                    html_table += "<tr>"
                    for col in df_stats.columns:
                        val = row[col]
                        if col == 'อัตราชนะ':
                            # จัดการเงื่อนไขสีและการใส่ไอคอน
                            val_str = str(val).replace('%', '').strip()
                            try:
                                v_float = float(val_str)
                            except:
                                v_float = 0
                                
                            if v_float >= 60: 
                                badge = "badge-green"
                                icon = "🎯 "
                            elif v_float >= 55: 
                                badge = "badge-orange"
                                icon = "⚠️️ "
                            else: 
                                badge = "badge-red"
                                icon = "🛑 "
                                
                            html_table += f"<td><span class='{badge}'>{icon}{val}</span></td>"
                        else:
                            html_table += f"<td>{val}</td>"
                    html_table += "</tr>"
                
                html_table += "</tbody></table></div>"
                
                # แสดงผลตาราง HTML ออกทางหน้าจอ
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.warning("ไม่พบข้อมูลสถิติในตาราง")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
