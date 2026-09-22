from flask import Flask, request, jsonify, render_template_string
import gspread
import google.generativeai as genai
from PIL import Image
import requests
import io
import os
import re
import pandas as pd
import plotly.express as px

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

creds_path = '/etc/secrets/credentials.json'
if not os.path.exists(creds_path):
    creds_path = 'credentials.json'

client = gspread.service_account(filename=creds_path)
sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1

# HTML แบบมี Tab สลับหน้า
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ระบบวิเคราะห์ราคาบอล VIP</title>
    <!-- โหลด Plotly JS -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Tahoma', sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; padding-top: 50px; margin: 0; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); width: 90%; max-width: {% if active_tab == 'dashboard' %}1000px{% else %}550px{% endif %}; margin-bottom: 50px;}
        .tab-menu { display: flex; border-bottom: 2px solid #ecf0f1; margin-bottom: 25px; }
        .tab-menu a { flex: 1; text-align: center; padding: 12px; text-decoration: none; color: #7f8c8d; font-weight: bold; font-size: 16px; transition: 0.3s; }
        .tab-menu a:hover { background-color: #f9f9f9; }
        .tab-menu a.active { border-bottom: 3px solid #27ae60; color: #27ae60; }
        
        /* CSS สำหรับหน้า Upload */
        h2 { color: #2c3e50; margin-bottom: 5px; text-align: center; }
        p.desc { color: #7f8c8d; font-size: 14px; margin-bottom: 25px; text-align: center; }
        input[type="file"] { margin: 10px 0 20px 0; padding: 10px; border: 2px dashed #bdc3c7; border-radius: 8px; width: 90%; background: #fafafa; cursor: pointer; }
        button { background-color: #27ae60; color: white; border: none; padding: 12px 25px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; transition: background 0.3s; }
        button:hover { background-color: #219150; }
        .result { margin-top: 25px; padding: 15px; background: #e8f8f5; border: 1px solid #1abc9c; border-radius: 8px; color: #16a085; font-size: 14px; }
        .error { margin-top: 25px; padding: 15px; background: #fadbd8; border: 1px solid #e74c3c; border-radius: 8px; color: #c0392b; }
        
        /* CSS สำหรับหน้า Dashboard */
        .metric-box { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 30px; }
        .metric { flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }
        .metric h3 { margin: 0; font-size: 32px; color: #2c3e50; }
        .metric p { margin: 5px 0 0 0; color: #7f8c8d; font-size: 14px; }
        .charts-row { display: flex; gap: 20px; flex-wrap: wrap; }
        .chart-col { flex: 1; min-width: 300px; min-height: 400px; background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; overflow: hidden; }
    </style>
</head>
<body>
    <div class="container">
        <!-- เมนู Tab สลับหน้า -->
        <div class="tab-menu">
            <a href="/" class="{% if active_tab == 'upload' %}active{% endif %}">📸 อัปโหลดราคาบอล</a>
            <a href="/dashboard" class="{% if active_tab == 'dashboard' %}active{% endif %}">📊 สถิติ (Dashboard)</a>
        </div>

        {% if active_tab == 'upload' %}
            <!-- หน้าจออัปโหลดภาพ -->
            <h2>⚽ สกัดราคาบอล & วิเคราะห์ VIP</h2>
            <p class="desc">อัปโหลดภาพตารางราคาเพื่อสกัดข้อมูลส่งเข้า Google Sheets</p>
            <form action="/" method="POST" enctype="multipart/form-data" style="text-align: center;">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">🚀 อัปโหลดและวิเคราะห์ข้อมูล</button>
            </form>
            {% if result %}
                <div class="result"><strong>✅ สำเร็จ!</strong><br><br>{{ result|safe }}</div>
            {% elif error %}
                <div class="error"><strong>❌ เกิดข้อผิดพลาด:</strong><br>{{ error }}</div>
            {% endif %}
        
        {% elif active_tab == 'dashboard' %}
            <!-- หน้าจอ Dashboard -->
            <div class="metric-box">
                <div class="metric"><h3>{{ total_matches }}</h3><p>แมตช์ที่สรุปผลแล้ว</p></div>
                <div class="metric"><h3>{{ win_matches }}</h3><p>จำนวนที่ชนะ</p></div>
                <div class="metric"><h3>{{ win_rate }}%</h3><p>Win Rate รวม</p></div>
            </div>
            <div class="charts-row">
                <div class="chart-col">
                    <h4 style="text-align:center; color:#2c3e50;">สัดส่วน ชนะ/แพ้</h4>
                    {{ pie_html|safe }}
                </div>
                <div class="chart-col">
                    <h4 style="text-align:center; color:#2c3e50;">ความแม่นยำแยกตามสูตร</h4>
                    {{ bar_html|safe }}
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def process_and_analyze(raw_text):
    raw_data = [item.strip() for item in raw_text.split(',')]
    row_data = []
    
    # 1. ทำความสะอาดข้อมูล
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

        # กฎข้อ 1: ตรวจสอบความสมบูรณ์ของข้อมูล
        if any(pd.isna(x) for x in [home_1x2, away_1x2, hdp_home, hdp_away, over_odds, under_odds]):
             recommendation = "ข้อมูลไม่ครบถ้วน (ข้าม)"

        # กฎข้อ 2: กรองบอลห่างชั้นเกินไป
        elif abs(home_1x2 - away_1x2) > 2.5:
             recommendation = "ข้าม (บอลห่างชั้นเกินไป)"

        # กฎข้อ 3: VIP Strategy (สถิติ Win Rate สูงสุด)
        elif home_1x2 > away_1x2 and hdp_home > 0:
            recommendation = "บอลรองเจ้าบ้านน้ำดำ (VIP)"
            
        # กฎข้อ 4: Value Betting
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
        print(f"Calculation Error: {e}")
        recommendation = "ตรวจสอบความถูกต้องของข้อมูล (Error)"
        
    return row_data, recommendation


# Route: หน้าอัปโหลด (Tab 1) - ปรับปรุงระบบป้องกัน Server Error
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template_string(HTML_TEMPLATE, active_tab='upload', error="ไม่ได้เลือกไฟล์")
        try:
            img = Image.open(io.BytesIO(file.read()))
            prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
            response = model.generate_content([img, prompt])
            
            if not response or not response.text:
                return render_template_string(HTML_TEMPLATE, active_tab='upload', error="AI ไม่สามารถอ่านข้อมูลจากภาพนี้ได้ กรุณาลองอัปโหลดภาพใหม่อีกครั้ง")
                
            row_data, rec = process_and_analyze(response.text.strip())
            sheet.append_row(row_data)
            return render_template_string(HTML_TEMPLATE, active_tab='upload', result=f"ทีม: {row_data[0]} vs {row_data[1]}<br>วิเคราะห์: <b>{rec}</b>")
        except Exception as e:
            # พิมพ์ Error จริงไว้ดูใน Terminal ของเซิร์ฟเวอร์
            print(f"🔥 UPLOAD ERROR: {str(e)}")
            # แสดงข้อความปลอดภัยบนหน้าเว็บแทนที่จะเป็นหน้าขาว Internal Server Error
            return render_template_string(HTML_TEMPLATE, active_tab='upload', error="เกิดข้อผิดพลาดในการประมวลผลภาพหรือเชื่อมต่อ Google Sheets กรุณาลองใหม่อีกครั้ง")
    return render_template_string(HTML_TEMPLATE, active_tab='upload')


# Route: หน้า Dashboard (Tab 2)
@app.route('/dashboard')
def dashboard_page():
    try:
        df = pd.DataFrame(sheet.get_all_records())
        if df.empty or 'ผลเปรียบเทียบ' not in df.columns:
            return render_template_string(HTML_TEMPLATE, active_tab='dashboard', error="ไม่พบข้อมูลผลเปรียบเทียบในชีท")
        
        df_comp = df[df['ผลเปรียบเทียบ'].isin(['ชนะ', 'แพ้', 'เจ๊า'])].copy()
        total = len(df_comp)
        wins = len(df_comp[df_comp['ผลเปรียบเทียบ'] == 'ชนะ'])
        win_rate = round((wins / total) * 100, 2) if total > 0 else 0

        df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].astype(str).str.strip()
        df_comp['แนะนำลงทุน'] = df_comp['แนะนำลงทุน'].replace({
            '': 'ข้อมูลว่าง/ซ่อนอยู่',
            'nan': 'ข้อมูลว่าง/ซ่อนอยู่',
            'None': 'ข้อมูลว่าง/ซ่อนอยู่',
            'บอลรองเจ้าบ้านน้ำดำ (VIP ⭐️)': 'บอลรองเจ้าบ้านน้ำดำ (VIP)'
        })

        pie_fig = px.pie(df_comp, names='ผลเปรียบเทียบ', color='ผลเปรียบเทียบ', 
                         color_discrete_map={'ชนะ': '#2ecc71', 'แพ้': '#e74c3c', 'เจ๊า': '#95a5a6'}, hole=0.4)
        pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), autosize=True)
        pie_html = pie_fig.to_html(full_html=False, include_plotlyjs=False, 
                                   default_width='100%', default_height='350px', 
                                   config={'responsive': True})

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
        
        bar_html = bar_fig.to_html(full_html=False, include_plotlyjs=False, 
                                   default_width='100%', default_height='400px', 
                                   config={'responsive': True})

        return render_template_string(
            HTML_TEMPLATE, 
            active_tab='dashboard', 
            total_matches=total, 
            win_matches=wins, 
            win_rate=win_rate,
            pie_html=pie_html,
            bar_html=bar_html
        )
    except Exception as e:
        print(f"🔥 DASHBOARD ERROR: {str(e)}")
        return render_template_string(HTML_TEMPLATE, active_tab='dashboard', error="เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ")


# Route: Webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        image_url = request.json.get('image_url')
        img = Image.open(io.BytesIO(requests.get(image_url).content))
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ..."
        response = model.generate_content([img, prompt])
        row_data, rec = process_and_analyze(response.text.strip())
        sheet.append_row(row_data)
        return jsonify({"status": "success", "data": row_data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
