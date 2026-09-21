import streamlit as st
import gspread
import pandas as pd
import plotly.express as px
import os

# ตั้งค่าหน้าเว็บให้กว้างขึ้น
st.set_page_config(page_title="Football VIP Dashboard", layout="wide")
st.title("📊 Dashboard สรุปสถิติความแม่นยำ AI & VIP")

# 1. ฟังก์ชันดึงข้อมูลจาก Google Sheets (ใช้ Cache เพื่อไม่ให้ดึงข้อมูลซ้ำบ่อยเกินไป)
@st.cache_data(ttl=300) # อัปเดตข้อมูลใหม่ทุก 5 นาที
def load_data():
    creds_path = '/etc/secrets/credentials.json'
    if not os.path.exists(creds_path):
        creds_path = 'credentials.json'
        
    client = gspread.service_account(filename=creds_path)
    sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1
    
    # ดึงข้อมูลทั้งหมดมาแปลงเป็น Pandas DataFrame
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# โหลดข้อมูล
df = load_data()

if df.empty:
    st.warning("ยังไม่มีข้อมูลในระบบ")
else:
    # กรองเฉพาะแถวที่ทราบผลแล้ว (ตัดคำว่า 'รอผล' หรือค่าว่างทิ้ง)
    df_completed = df[df['ผลเปรียบเทียบ'].isin(['ชนะ', 'แพ้', 'เจ๊า'])]

    # --- ส่วนที่ 1: การ์ดสรุปตัวเลข (Metrics) ---
    total_matches = len(df_completed)
    win_matches = len(df_completed[df_completed['ผลเปรียบเทียบ'] == 'ชนะ'])
    win_rate = (win_matches / total_matches) * 100 if total_matches > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("แมตช์ที่สรุปผลแล้ว", f"{total_matches} คู่")
    col2.metric("จำนวนแมตช์ที่ชนะ", f"{win_matches} คู่")
    col3.metric("Win Rate รวม", f"{win_rate:.2f}%")

    st.markdown("---")

    # --- ส่วนที่ 2: กราฟ (Charts) ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("ภาพรวมสัดส่วน ชนะ/แพ้")
        # กราฟโดนัท (Pie Chart)
        fig_pie = px.pie(
            df_completed, 
            names='ผลเปรียบเทียบ', 
            color='ผลเปรียบเทียบ',
            color_discrete_map={'ชนะ': '#2ecc71', 'แพ้': '#e74c3c', 'เจ๊า': '#95a5a6'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("ความแม่นยำแยกตามสูตรวิเคราะห์")
        # กราฟแท่ง (Bar Chart) จัดกลุ่มตามคำแนะนำลงทุน
        strategy_df = df_completed.groupby(['แนะนำลงทุน', 'ผลเปรียบเทียบ']).size().reset_index(name='จำนวน')
        fig_bar = px.bar(
            strategy_df, 
            x='แนะนำลงทุน', 
            y='จำนวน', 
            color='ผลเปรียบเทียบ',
            barmode='group',
            color_discrete_map={'ชนะ': '#2ecc71', 'แพ้': '#e74c3c', 'เจ๊า': '#95a5a6'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- ส่วนที่ 3: ตารางข้อมูลล่าสุด ---
    st.subheader("📝 ข้อมูลล่าสุด (10 แมตช์ล่าสุด)")
    st.dataframe(df.tail(10))
