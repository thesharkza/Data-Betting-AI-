import streamlit as st
import gspread
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import pandas as pd
import plotly.express as px
import gc
import time  # เพิ่มไลบรารี time สำหรับหน่วงเวลารอ Sheet คำนวณ

# ----------------------------------------
# ตั้งค่าหน้าเว็บ Streamlit (กำหนดให้รองรับ CSS พิเศษ)
# ----------------------------------------
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP", page_icon="⚽", layout="wide")

# CSS สำหรับตกแต่ง Dashboard ให้เป็นแบบ 3D Modern & Glassmorphism
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
    """
    ฟังก์ชันทำความสะอาดข้อมูลดิบ (เหลือหน้าที่แค่จัดฟอร์แมต 12 คอลัมน์แรก)
    """
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

    # คืนค่ากลับไปเฉพาะ 12 คอลัมน์แรก (คอลัมน์ A ถึง L)
    return row_data[:12]

def get_money_management(rec_text, conf_text):
    """
    ฟังก์ชันคำนวณ Money Management พร้อมระบบ Auto-Pause และ Grace Period
    """
    if "ข้าม" in str(rec_text):
        return "0 Unit (ข้าม ห้ามลงทุน) 🛑", rec_text
    
    match_pct = re.search(r'(\d+(\.\d+)?)%', str(conf_text))
    match_n = re.search(r'n=(\d+)', str(conf_text)) # เพิ่มตัวจับค่า n
    
    if match_pct:
        pct_val = float(match_pct.group(1))
        n_val = int(match_n.group(1)) if match_n else 0 # ดึงค่า n
        
        # ถ้ายอด n ยังไม่ถึง 5 แมตช์ ให้ถือว่าอยู่ในช่วง "ทดสอบโมเดล"
        if n_val < 5 and "SNIPER" in str(rec_text):
            return "⭐ 0.5 Unit (ช่วงทดสอบโมเดลใหม่ 🧪)", rec_text
            
        # ระบบ Auto-Pause: จะทำงานก็ต่อเมื่อ n >= 5 และสถิติต่ำกว่า 55% เท่านั้น
        if pct_val < 55 and "SNIPER" in str(rec_text):
            modified_rec = f"⏸️ พักชั่วคราว: {rec_text} (สถิติตกเหลือ {pct_val}%)"
            return "0 Unit (รอสถิติฟื้นตัว 🛑)", modified_rec
            
        # จัดเกรด Unit ตามปกติ หากผ่านเกณฑ์ด่านตรวจแล้ว
        if pct_val >= 70:
            return "⭐⭐⭐⭐ 3 Units (Max Bet 🎯)", rec_text
        elif pct_val >= 60:
            return "⭐⭐⭐ 2 Units (High 🚀)", rec_text
        elif pct_val >= 55:
            return "⭐⭐ 1 Unit (Normal 👍)", rec_text
        else:
            return "⭐ 0.5 Unit (Low / Test 🧪)", rec_text
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
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((800, 800))
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=80)
                        img_bytes = img_byte_arr.getvalue()
                    
                    with st.spinner("🤖 กำลังให้ AI สกัดราคาบอล..."):
                        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
                        
                        response = model.generate_content([
                            {"mime_type": "image/jpeg", "data": img_bytes}, 
                            prompt
                        ])
                    
                    gc.collect()

                    if not response or not response.text:
                        st.error("❌ AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                    else:
                        with st.spinner("📊 กำลังบันทึกและรอ Google Sheets ประมวลผลลัพธ์..."):
                            row_data_to_sheet = clean_raw_data(response.text.strip())
                            
                            col_a_values = ws_data.col_values(1)
                            col_b_values = ws_data.col_values(2) 
                            
                            last_row = 0
                            for i, val in enumerate(col_a_values):
                                if str(val).strip() != "":
                                    last_row = i + 1
                            
                            # 🔍 ระบบค้นหาและบันทึกทับ (Overwrite) หากทีมซ้ำ
                            new_home = str(row_data_to_sheet[0]).strip()
                            new_away = str(row_data_to_sheet[1]).strip()
                            target_row = None
                            
                            # วนลูปเช็คทีมเหย้าและเยือนว่าซ้ำไหม
                            for i in range(len(col_a_values)):
                                sheet_home = str(col_a_values[i]).strip()
                                sheet_away = str(col_b_values[i]).strip() if i < len(col_b_values) else ""
                                
                                if sheet_home == new_home and sheet_away == new_away:
                                    target_row = i + 1
                                    break

                            if target_row is not None:
                                # 1A. กรณีพบข้อมูลซ้ำ -> บันทึกทับแถวเดิม
                                ws_data.update(range_name=f"A{target_row}", values=[row_data_to_sheet])
                                st.warning(f"🔄 พบข้อมูลคู่ {new_home} vs {new_away} ในระบบ ทำการ **บันทึกทับ** ที่แถว {target_row} เรียบร้อยแล้ว")
                            else:
                                # 1B. กรณีไม่พบข้อมูลซ้ำ -> บันทึกบรรทัดใหม่
                                target_row = last_row + 1
                                ws_data.update(range_name=f"A{target_row}", values=[row_data_to_sheet])
                                st.success("✅ บันทึกข้อมูลคู่ใหม่สำเร็จ!")
                                
                            # 2. หน่วงเวลา 3 วินาที รอให้สูตร MAP ใน Sheets คำนวณเสร็จ
                            time.sleep(3)
                            
                            # 3. ดึงข้อมูลทั้งบรรทัดกลับมาเพื่อเอาผลลัพธ์ (ดึงจาก target_row)
                            updated_row = ws_data.row_values(target_row)
                            
                            # คอลัมน์ P (Index 15), คอลัมน์ R (Index 17), คอลัมน์ S (Index 18)
                            rec = updated_row[15] if len(updated_row) > 15 else "กำลังคำนวณ..."
                            confidence = updated_row[17] if len(updated_row) > 17 else "-"
                            radar = updated_row[18] if len(updated_row) > 18 else "-"

                            # 4. ประมวลผล Money Management และรับค่า rec ที่ถูกดัดแปลงจากระบบ Gatekeeper
                            mm_text, rec = get_money_management(rec, confidence)
                            
                            # แสดงผลลัพธ์ที่ดึงมาจาก Google Sheets
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
# TAB 2 : Dashboard แบบ 3 มิติ และตารางสี
# ----------------------------------------
with tab2:
    if not workbook:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        try:
            # 1. ดึงข้อมูล
            ws_data = workbook.sheet1
            data = ws_data.get_all_records()
            df = pd.DataFrame(data)
            
            if not df.empty and 'ผลเปรียบเทียบ' in df.columns:
                df_comp = df[df['ผลเปรียบเทียบ'].astype(str).str.contains('ชนะ|แพ้|เจ๊า', na=False)].copy()
                total = len(df_comp)
                wins_full = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะเต็ม'])
                wins_half = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะครึ่ง'])
                total_wins = wins_full + (wins_half * 0.5)
                draws = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'เจ๊า'])
                win_rate = round((total_wins / (total - draws)) * 100, 2) if (total - draws) > 0 else 0

                # 2. สร้างการ์ดแสดงผล 3D (HTML/CSS)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card bg-blue">
                        <div class="card-title">🏟️ แมตช์ทั้งหมด (จบแล้ว)</div>
                        <p class="card-value">{total} <span class="card-unit">คู่</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card bg-purple">
                        <div class="card-title">🔥 คะแนนชนะสะสม (Win Score)</div>
                        <p class="card-value">{total_wins} <span class="card-unit">แต้ม</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card bg-green">
                        <div class="card-title">🏆 Win Rate รวมทั้งหมด</div>
                        <p class="card-value">{win_rate} <span class="card-unit">%</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # 3. จัดการข้อมูลก่อนวาดกราฟ
                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].astype(str).str.strip()
                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].replace({
                    '': 'ข้อมูลว่าง/ซ่อนอยู่', 'nan': 'ข้อมูลว่าง/ซ่อนอยู่', 'None': 'ข้อมูลว่าง/ซ่อนอยู่'
                })

                col_chart1, col_chart2 = st.columns(2)

                # กราฟโดนัท 3D (Pop-out effect)
                with col_chart1:
                    st.markdown("<h4 style='text-align: center; color: #334155;'>สัดส่วนผลลัพธ์โดยรวม</h4>", unsafe_allow_html=True)
                    
                    # หาสัดส่วนและดึงชิ้น "ชนะเต็ม" ให้เด้งออกมา 10%
                    pull_array = [0.1 if label == 'ชนะเต็ม' else 0 for label in df_comp['ผลเปรียบเทียบ'].unique()]
                    
                    pie_fig = px.pie(df_comp, names='ผลเปรียบเทียบ', color='ผลเปรียบเทียบ', 
                                     color_discrete_map={
                                         'ชนะเต็ม': '#10b981', 'ชนะครึ่ง': '#34d399', 
                                         'แพ้เต็ม': '#f43f5e', 'แพ้ครึ่ง': '#fb7185', 
                                         'เจ๊า': '#94a3b8'
                                     }, hole=0.5) # เจาะรูตรงกลางเป็น Donut
                    
                    pie_fig.update_traces(pull=pull_array, textinfo='percent+label', textfont_size=14,
                                          marker=dict(line=dict(color='#ffffff', width=2))) # ใส่ขอบขาวให้ดูมีมิติ
                    pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), autosize=True, showlegend=False)
                    st.plotly_chart(pie_fig, use_container_width=True)

                # กราฟแท่งแนวนอน (Rounded & Bordered)
                with col_chart2:
                    st.markdown("<h4 style='text-align: center; color: #334155;'>ความแม่นยำแยกตามกฎ (Sniper Rules)</h4>", unsafe_allow_html=True)
                    bar_df = df_comp.groupby(['แนะนำลงทุน', 'ผลเปรียบเทียบ'], dropna=False).size().reset_index(name='จำนวน')
                    
                    bar_fig = px.bar(bar_df, x='จำนวน', y='แนะนำลงทุน', color='ผลเปรียบเทียบ', barmode='stack',
                                     orientation='h', 
                                     color_discrete_map={
                                         'ชนะเต็ม': '#10b981', 'ชนะครึ่ง': '#34d399', 
                                         'แพ้เต็ม': '#f43f5e', 'แพ้ครึ่ง': '#fb7185', 'เจ๊า': '#94a3b8'
                                     })
                    
                    bar_fig.update_traces(marker=dict(line=dict(color='#ffffff', width=1.5))) # ใส่ขอบแท่งกราฟให้ดูป๊อปอัพ
                    bar_fig.update_layout(
                        margin=dict(t=20, b=20, l=150, r=20), 
                        xaxis_title="จำนวนครั้ง", yaxis_title="", 
                        yaxis=dict(autorange="reversed"), 
                        plot_bgcolor='rgba(0,0,0,0)' # พื้นหลังโปร่งใส
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                    
            st.markdown("<br><hr>", unsafe_allow_html=True)
            
            # 4. ตารางสถิติ (Color-coded)
            st.markdown("<h3 style='color: #1e293b;'>📋 ตารางสรุปสถิติความเชื่อมั่น (แยกรายกฎ)</h3>", unsafe_allow_html=True)
            ws_stats = workbook.worksheet("สรุปสถิติ")
            data_stats = ws_stats.get_all_records()
            df_stats = pd.DataFrame(data_stats)
            
            if not df_stats.empty:
                df_stats.columns = [str(c).strip() for c in df_stats.columns]
                
                if 'อัตราชนะ' in df_stats.columns:
                    def format_winrate(val):
                        if isinstance(val, str) and '%' in val: return val
                        try: return f"{float(val) * 100:.2f}%"
                        except: return val 
                    
                    df_stats['อัตราชนะ'] = df_stats['อัตราชนะ'].apply(format_winrate)
                    
                    # ฟังก์ชันระบายสีตามเกณฑ์ Win Rate
                    def color_winrate_table(val):
                        try:
                            v = float(str(val).replace('%', '').strip())
                            if v >= 60: return 'background-color: #d1fae5; color: #065f46; font-weight: bold;' # สีเขียว
                            elif v >= 55: return 'background-color: #fef3c7; color: #92400e; font-weight: bold;' # สีส้ม
                            else: return 'background-color: #ffe4e6; color: #9f1239; font-weight: bold;' # สีแดง
                        except:
                            return ''
                    
                    # แสดงตารางพร้อมการระบายสี
                    # ใช้ getattr เพื่อให้รองรับ pandas ทั้งเวอร์ชันเก่า (applymap) และใหม่ (map)
                    try:
                        styled_df = df_stats.style.map(color_winrate_table, subset=['อัตราชนะ'])
                    except AttributeError:
                        styled_df = df_stats.style.applymap(color_winrate_table, subset=['อัตราชนะ'])
                        
                    st.dataframe(styled_df, use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
