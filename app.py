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

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP", page_icon="⚽", layout="wide")

# ตั้งค่า API Key ของ Gemini (รองรับทั้งจาก Streamlit Secrets และ Environment Variables)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("❌ ไม่พบ GEMINI_API_KEY กรุณาตั้งค่าใน Streamlit Secrets")

# เชื่อมต่อ Google Sheets
@st.cache_resource
def init_gspread():
    try:
        if "gspread" in st.secrets:
            creds_dict = dict(st.secrets["gspread"])
            client = gspread.service_account_from_dict(creds_dict)
        else:
            creds_path = 'credentials.json'
            client = gspread.service_account(filename=creds_path)
        
        # เชื่อมต่อ Sheet ชื่อ 'ข้อมูลราคาบอลสกัดจากภาพ'
        sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1
        return sheet
    except Exception as e:
        st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
        return None

sheet = init_gspread()
# ใช้โมเดล gemini-1.5-flash เพื่อความรวดเร็วและความเสถียร
model = genai.GenerativeModel('gemini-3.5-flash-lite')

def process_and_analyze(raw_text):
    """
    ฟังก์ชันทำความสะอาดข้อมูลดิบและวิเคราะห์เพื่อหาคำแนะนำการลงทุน
    """
    raw_data = [item.strip() for item in raw_text.split(',')]
    row_data = []
    
    # 1. ทำความสะอาดข้อมูลที่สกัดจากภาพ (ตัดตัวอักษร O/U ออก แปลงเป็นตัวเลข)
    for i, item in enumerate(raw_data):
        if i < 2:
            row_data.append(item) # ชื่อทีมเหย้า, ชื่อทีมเยือน
        else:
            cleaned = re.sub(r'^[oOuU\s]+', '', item)
            try:
                row_data.append(float(cleaned) if '.' in cleaned else int(cleaned))
            except:
                row_data.append(cleaned)
                
    recommendation = "รอดูสถานการณ์ (รอข้อมูล)"
    
    try:
        # เช็กว่าข้อมูลมีครบ 12 ค่าตามที่ Prompt กำหนดหรือไม่
        if len(row_data) < 12:
            return row_data, "ข้อมูลไม่ครบถ้วน (ข้าม)"
            
        # 2. แมปตัวแปรตามลำดับใน Prompt
        home1x2 = float(row_data[2])
        away1x2 = float(row_data[4])
        hdp_line = float(row_data[5])   # แฮนดิแคป (เหย้า)
        hdp_home = float(row_data[6])   # ค่าน้ำ HDP (เหย้า)
        hdp_away = float(row_data[8])   # ค่าน้ำ HDP (เยือน)
        over_odds = float(row_data[10]) # ค่าน้ำสูง
        under_odds = float(row_data[11])# ค่าน้ำต่ำ
        
        # ดักจับข้อมูลค่าน้ำ 1X2 ที่ผิดปกติ
        if home1x2 <= 0 or away1x2 <= 0:
            return row_data, "ข้อมูลผิดพลาด (ไม่ใช่ตัวเลข/ค่าน้ำเสีย - ตรวจสอบ)"
            
        # 3. คำนวณ Implied Probability Gap เพื่อดูความห่างชั้น
        implied_gap = abs((1 / home1x2) - (1 / away1x2))
        if implied_gap > 0.35:
            recommendation = "ข้าม (บอลห่างชั้นเกินไป)"
            
        # 4. เช็กเงื่อนไข VIP (บอลรองเจ้าบ้าน)
        elif home1x2 > away1x2 and hdp_home > 0:
            if hdp_line >= 0.5:
                if 0.5 <= hdp_home <= 1.2:
                    recommendation = "บอลรองเจ้าบ้านน้ำดำ (VIP)"
                else:
                    recommendation = "ข้าม (VIP ค่าน้ำผิดปกติ)"
            else:
                recommendation = "ข้าม (VIP แต้มต่อน้อยเกินไป)"
                
        # 5. กรณีไม่เข้า VIP ให้เช็กค่าน้ำปกติเพื่อหาตัวเลือกที่ดีที่สุด
        else:
            max_odds = max(hdp_home, hdp_away, over_odds, under_odds)
            if max_odds <= 0:
                recommendation = "รอดูสถานการณ์ (น้ำแดงหมด)"
            elif max_odds < 0.75 or max_odds > 0.95:
                recommendation = "ข้าม (ค่าน้ำเสี่ยงเกินไป)"
            else:
                # เรียงลำดับความสำคัญในกรณีที่ค่าน้ำเท่ากัน
                if max_odds == hdp_home:
                    recommendation = "เชียร์เจ้าบ้าน (น้ำดำ)"
                elif max_odds == hdp_away:
                    recommendation = "เชียร์ทีมเยือน (น้ำดำ)"
                elif max_odds == over_odds:
                    recommendation = "ลุ้นสูง (น้ำดำ)"
                else:
                    recommendation = "ลุ้นต่ำ (น้ำดำ)"

    except ValueError:
        recommendation = "ข้อมูลผิดพลาด (ไม่ใช่ตัวเลข/ค่าน้ำเสีย - ตรวจสอบ)"
    except Exception as e:
        recommendation = f"ตรวจสอบความถูกต้องของข้อมูล (Error: {str(e)})"
        
    return row_data, recommendation


# หัวข้อหลักของแอป
st.title("⚽ ระบบวิเคราะห์ราคาบอล VIP")

# สร้าง Tab สลับหน้า
tab1, tab2 = st.tabs(["📸 อัปโหลดราคาบอล", "📊 สถิติ (Dashboard)"])

with tab1:
    st.subheader("สกัดราคาบอล & วิเคราะห์ VIP")
    st.write("อัปโหลดภาพตารางราคาเพื่อสกัดข้อมูลส่งเข้า Google Sheets อัตโนมัติ")
    
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพตารางราคาบอล", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="ภาพที่อัปโหลด", width=600)
        
        if st.button("🚀 อัปโหลดและวิเคราะห์ข้อมูล", type="primary"):
            if not sheet:
                st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
            else:
                try:
                    # ขั้นที่ 1: เตรียมและบีบอัดรูปภาพ
                    with st.spinner("📸 กำลังประมวลผลรูปภาพ..."):
                        img = Image.open(uploaded_file)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((800, 800))
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=80)
                        img_bytes = img_byte_arr.getvalue()
                    
                    # ขั้นที่ 2: ส่งให้ AI วิเคราะห์
                    with st.spinner("🤖 กำลังให้ AI สกัดและวิเคราะห์ราคาบอล..."):
                        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
                        
                        response = model.generate_content([
                            {"mime_type": "image/jpeg", "data": img_bytes}, 
                            prompt
                        ])
                    
                    gc.collect()

                    if not response or not response.text:
                        st.error("❌ AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                    else:
                        # ขั้นที่ 3: บันทึกลง Google Sheets
                        with st.spinner("📊 กำลังบันทึกข้อมูลลง Google Sheets..."):
                            row_data, rec = process_and_analyze(response.text.strip())
                            
                            # ส่งข้อมูลเข้าชีทแค่ 12 ตัวแรก (คอลัมน์ A ถึง L) เพื่อไม่กวนสูตร Array ท้ายตาราง
                            row_data_to_sheet = row_data[:12]
                            
                            # กรองหาบรรทัดว่างที่แท้จริง
                            col_a_values = sheet.col_values(1)
                            col_b_values = sheet.col_values(2) 
                            
                            last_row = 0
                            for i, val in enumerate(col_a_values):
                                if str(val).strip() != "":
                                    last_row = i + 1
                            
                            # 🛡️ ระบบป้องกันข้อมูลเบิ้ล (Duplicate Check)
                            is_duplicate = False
                            if last_row > 1 and len(row_data_to_sheet) >= 2: 
                                last_home_team = str(col_a_values[last_row - 1]).strip()
                                last_away_team = str(col_b_values[last_row - 1]).strip()
                                
                                if str(row_data_to_sheet[0]).strip() == last_home_team and str(row_data_to_sheet[1]).strip() == last_away_team:
                                    is_duplicate = True

                            if is_duplicate:
                                st.warning(f"⚠️ ข้อมูลคู่นี้ ({row_data_to_sheet[0]} vs {row_data_to_sheet[1]}) ถูกบันทึกลง Sheet ไปแล้ว (ระบบข้ามการบันทึกซ้ำ)")
                            else:
                                next_row = last_row + 1
                                # สั่งเขียนเฉพาะคอลัมน์ A-L เมื่อข้อมูลไม่ซ้ำ
                                sheet.update(range_name=f"A{next_row}", values=[row_data_to_sheet])
                                st.success("✅ บันทึกข้อมูลลง Google Sheets สำเร็จ!")
                        
                        # แสดงผลลัพธ์บนหน้าเว็บ
                        st.markdown(f"""
                        <div style="padding: 15px; background: #e8f8f5; border: 1px solid #1abc9c; border-radius: 8px; color: #16a085;">
                            <strong>คู่แข่งขัน:</strong> {row_data[0]} vs {row_data[1]}<br>
                            <strong>ผลการวิเคราะห์:</strong> <b>{rec}</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                finally:
                    gc.collect()

with tab2:
    st.subheader("📊 สถิติและแดชบอร์ดสรุปผล")
    if not sheet:
        st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
    else:
        try:
            data = sheet.get_all_records()
            df = pd.DataFrame(data)
            
            if df.empty or 'ผลเปรียบเทียบ' not in df.columns:
                st.warning("ไม่พบข้อมูลผลเปรียบเทียบใน Google Sheets")
            else:
                # กรองเฉพาะแถวที่มีการสรุปผลแล้ว
                df_comp = df[df['ผลเปรียบเทียบ'].astype(str).str.contains('ชนะ|แพ้|เจ๊า', na=False)].copy()
                total = len(df_comp)
                
                # นับคะแนนแบบแยก ชนะเต็ม(1) ชนะครึ่ง(0.5)
                wins_full = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะเต็ม'])
                wins_half = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะครึ่ง'])
                total_wins = wins_full + (wins_half * 0.5)
                draws = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'เจ๊า'])
                
                win_rate = round((total_wins / (total - draws)) * 100, 2) if (total - draws) > 0 else 0

                # แสดง Metric ด้านบน
                col1, col2, col3 = st.columns(3)
                col1.metric("แมตช์ที่สรุปผลแล้ว", f"{total} คู่")
                col2.metric("คะแนนชนะ (Win Score)", f"{total_wins}")
                col3.metric("Win Rate รวม", f"{win_rate}%")

                st.markdown("---")

                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].astype(str).str.strip()
                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].replace({
                    '': 'ข้อมูลว่าง/ซ่อนอยู่',
                    'nan': 'ข้อมูลว่าง/ซ่อนอยู่',
                    'None': 'ข้อมูลว่าง/ซ่อนอยู่'
                })

                # สร้างกราฟ 2 ฝั่ง
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("<h5 style='text-align: center;'>สัดส่วนผลลัพธ์โดยรวม</h5>", unsafe_allow_html=True)
                    pie_fig = px.pie(df_comp, names='ผลเปรียบเทียบ', color='ผลเปรียบเทียบ', 
                                     color_discrete_map={
                                         'ชนะเต็ม': '#2ecc71', 'ชนะครึ่ง': '#27ae60', 
                                         'แพ้เต็ม': '#e74c3c', 'แพ้ครึ่ง': '#c0392b', 
                                         'เจ๊า': '#95a5a6', 'ชนะ': '#2ecc71', 'แพ้': '#e74c3c'
                                     }, hole=0.4)
                    pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), autosize=True)
                    st.plotly_chart(pie_fig, use_container_width=True)

                with col_chart2:
                    st.markdown("<h5 style='text-align: center;'>ความแม่นยำแยกตามสูตรการลงทุน</h5>", unsafe_allow_html=True)
                    bar_df = df_comp.groupby(['แนะนำลงทุน', 'ผลเปรียบเทียบ'], dropna=False).size().reset_index(name='จำนวน')
                    
                    bar_fig = px.bar(bar_df, x='จำนวน', y='แนะนำลงทุน', color='ผลเปรียบเทียบ', barmode='stack',
                                     orientation='h', 
                                     color_discrete_map={
                                         'ชนะเต็ม': '#2ecc71', 'ชนะครึ่ง': '#27ae60', 
                                         'แพ้เต็ม': '#e74c3c', 'แพ้ครึ่ง': '#c0392b', 
                                         'เจ๊า': '#95a5a6', 'ชนะ': '#2ecc71', 'แพ้': '#e74c3c'
                                     })
                    
                    bar_fig.update_layout(
                        margin=dict(t=20, b=20, l=140, r=20), 
                        xaxis_title="จำนวน (ครั้ง)", 
                        yaxis_title="", 
                        yaxis=dict(autorange="reversed"), 
                        autosize=True
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
