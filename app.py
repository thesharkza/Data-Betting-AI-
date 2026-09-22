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
st.set_page_config(page_title="ระบบวิเคราะห์ราคาบอล VIP", page_icon="⚽", layout="centered")

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
        
        sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1
        return sheet
    except Exception as e:
        st.error(f"❌ เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e}")
        return None

sheet = init_gspread()
model = genai.GenerativeModel('gemini-3.5-flash-lite')

def process_and_analyze(raw_text):
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
                
    recommendation = "รอดูสถานการณ์ (รอข้อมูล)"
    
    try:
        home_1x2, away_1x2 = float(row_data[2]), float(row_data[4])
        hdp_home, hdp_away = float(row_data[6]), float(row_data[8])
        over_odds, under_odds = float(row_data[10]), float(row_data[11])

        if any(pd.isna(x) for x in [home_1x2, away_1x2, hdp_home, hdp_away, over_odds, under_odds]):
             recommendation = "ข้อมูลไม่ครบถ้วน (ข้าม)"
        elif abs(home_1x2 - away_1x2) > 2.5:
             recommendation = "ข้าม (บอลห่างชั้นเกินไป)"
        elif home_1x2 > away_1x2 and hdp_home > 0:
            recommendation = "บอลรองเจ้าบ้านน้ำดำ (VIP)"
        else:
            odds_dict = {
                "เชียร์เจ้าบ้าน (น้ำดำ)": hdp_home,
                "เชียร์ทีมเยือน (น้ำดำ)": hdp_away,
                "ลุ้นสูง (น้ำดำ)": over_odds,
                "ลุ้นต่ำ (น้ำดำ)": under_odds
            }
            positive_odds = {k: v for k, v in odds_dict.items() if v > 0}
            if positive_odds:
                best_choice = max(positive_odds, key=positive_odds.get)
                max_value = positive_odds[best_choice]
                if max_value < 0.75 or max_value > 0.95:
                    recommendation = "ข้าม (ค่าน้ำเสี่ยงเกินไป)"
                else:
                    recommendation = best_choice
            else:
                recommendation = "รอดูสถานการณ์ (น้ำแดงหมด)"
    except Exception as e: 
        recommendation = "ตรวจสอบความถูกต้องของข้อมูล (Error)"
        
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
        st.image(uploaded_file, caption="ภาพที่อัปโหลด", use_container_width=True)
        
        if st.button("🚀 อัปโหลดและวิเคราะห์ข้อมูล", type="primary"):
            if not sheet:
                st.error("ไม่สามารถเชื่อมต่อ Google Sheets ได้")
            else:
                with st.spinner("⏳ กำลังให้ AI วิเคราะห์ข้อมูลและบันทึกเข้า Google Sheets (รอสักครู่)..."):
                    try:
                        img = Image.open(uploaded_file)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((800, 800))
                        
                        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
                        response = model.generate_content([img, prompt])
                        
                        gc.collect()

                        if not response or not response.text:
                            st.error("AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                        else:
                            row_data, rec = process_and_analyze(response.text.strip())
                            sheet.append_row(row_data)
                            
                            st.success("✅ สำเร็จ!")
                            st.markdown(f"""
                            <div style="padding: 15px; background: #e8f8f5; border: 1px solid #1abc9c; border-radius: 8px; color: #16a085;">
                                <strong>คู่แข่งขัน:</strong> {row_data[0]} vs {row_data[1]}<br>
                                <strong>ผลการวิเคราะห์:</strong> <b>{rec}</b>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {str(e)}")
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
                df_comp = df[df['ผลเปรียบเทียบ'].isin(['ชนะ', 'แพ้', 'เจ๊า'])].copy()
                total = len(df_comp)
                wins = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะ'])
                win_rate = round((wins / total) * 100, 2) if total > 0 else 0

                # แสดง Metric ด้านบน
                col1, col2, col3 = st.columns(3)
                col1.metric("แมตช์ที่สรุปผลแล้ว", f"{total} คู่")
                col2.metric("จำนวนที่ชนะ", f"{wins} คู่")
                col3.metric("Win Rate รวม", f"{win_rate}%")

                st.markdown("---")

                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].astype(str).str.strip()
                df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].replace({
                    '': 'ข้อมูลว่าง/ซ่อนอยู่',
                    'nan': 'ข้อมูลว่าง/ซ่อนอยู่',
                    'None': 'ข้อมูลว่าง/ซ่อนอยู่',
                    'บอลรองเจ้าบ้านน้ำดำ (VIP ⭐️)': 'บอลรองเจ้าบ้านน้ำดำ (VIP)'
                })

                # แสดงกราฟ 2 ฝั่ง
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("<h5 style='text-align: center;'>สัดส่วน ชนะ/แพ้</h5>", unsafe_allow_html=True)
                    pie_fig = px.pie(df_comp, names='ผลเปรียบเทียบ', color='ผลเปรียบเทียบ', 
                                     color_discrete_map={'ชนะ': '#2ecc71', 'แพ้': '#e74c3c', 'เจ๊า': '#95a5a6'}, hole=0.4)
                    pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), autosize=True)
                    st.plotly_chart(pie_fig, use_container_width=True)

                with col_chart2:
                    st.markdown("<h5 style='text-align: center;'>ความแม่นยำแยกตามสูตร</h5>", unsafe_allow_html=True)
                    bar_df = df_comp.groupby(['แนะนำลงทุน', 'ผลเปรียบเทียบ'], dropna=False).size().reset_index(name='จำนวน')
                    
                    bar_fig = px.bar(bar_df, x='จำนวน', y='แนะนำลงทุน', color='ผลเปรียบเทียบ', barmode='group',
                                     orientation='h', 
                                     color_discrete_map={'ชนะ': '#2ecc71', 'แพ้': '#e74c3c', 'เจ๊า': '#95a5a6'})
                    
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
