from flask import Flask, request, jsonify, render_template_string
import gspread
import google.generativeai as genai
from PIL import Image
import requests
import io
import os
import re

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

creds_path = '/etc/secrets/credentials.json'
if not os.path.exists(creds_path):
    creds_path = 'credentials.json'

client = gspread.service_account(filename=creds_path)
sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ระบบสกัดข้อมูลราคาบอลด้วย AI (VIP Formula)</title>
    <style>
        body { font-family: 'Tahoma', sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center; max-width: 550px; width: 90%; }
        h2 { color: #2c3e50; margin-bottom: 5px; }
        p { color: #7f8c8d; font-size: 14px; margin-bottom: 25px; }
        input[type="file"] { margin: 10px 0 20px 0; padding: 10px; border: 2px dashed #bdc3c7; border-radius: 8px; width: 90%; background: #fafafa; cursor: pointer; }
        button { background-color: #27ae60; color: white; border: none; padding: 12px 25px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; transition: background 0.3s; }
        button:hover { background-color: #219150; }
        .result { margin-top: 25px; padding: 15px; background: #e8f8f5; border: 1px solid #1abc9c; border-radius: 8px; color: #16a085; text-align: left; font-size: 14px; line-height: 1.6; }
        .error { margin-top: 25px; padding: 15px; background: #fadbd8; border: 1px solid #e74c3c; border-radius: 8px; color: #c0392b; }
        .vip-badge { background-color: #f1c40f; color: #8e44ad; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>⚽ สกัดราคาบอล & วิเคราะห์ VIP</h2>
        <p>AI สกัดข้อมูล 12 คอลัมน์ ส่งเข้า Sheet (วิเคราะห์ผลอัตโนมัติด้วย ARRAYFORMULA)</p>
        
        <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">🚀 อัปโหลดและวิเคราะห์ข้อมูล</button>
        </form>

        {% if result %}
            <div class="result">
                <strong>✅ สำเร็จ! ข้อมูล 12 คอลัมน์บันทึกลงชีทแล้ว:</strong><br><br>
                {{ result|safe }}
            </div>
        {% elif error %}
            <div class="error">
                <strong>❌ เกิดข้อผิดพลาด:</strong><br>
                {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def process_and_analyze(raw_text):
    raw_data = [item.strip() for item in raw_text.split(',')]
    row_data = []
    
    for i, item in enumerate(raw_data):
        if i < 2:
            row_data.append(item)
        else:
            cleaned_item = re.sub(r'^[oOuU\s]+', '', item)
            try:
                if '.' in cleaned_item:
                    row_data.append(float(cleaned_item))
                else:
                    row_data.append(int(cleaned_item))
            except ValueError:
                row_data.append(cleaned_item)
                
    recommendation = "รอข้อมูลวิเคราะห์..."
    try:
        home_1x2 = float(row_data[2])
        away_1x2 = float(row_data[4])
        home_hdp_odds = float(row_data[6])
        away_hdp_odds = float(row_data[8])
        
        if home_1x2 > away_1x2 and home_hdp_odds > 0:
            recommendation = "บอลรองเจ้าบ้านน้ำดำ (VIP ⭐️)"
        elif home_hdp_odds > 0:
            recommendation = "เชียร์เจ้าบ้าน (น้ำดำ)"
        elif away_hdp_odds > 0:
            recommendation = "เชียร์ทีมเยือน (น้ำดำ)"
        else:
            recommendation = "รอดูสถานการณ์ (น้ำแดงทั้งคู่)"
            
    except (IndexError, ValueError, TypeError):
         recommendation = "ข้อมูลไม่ครบถ้วน"
         
    # แยกคืนค่า row_data (สำหรับชีท) และ recommendation (สำหรับโชว์หน้าเว็บ) ออกจากกัน
    # ไม่มีการ .append(recommendation) ลงใน row_data แล้ว
    return row_data, recommendation

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files: return render_template_string(HTML_TEMPLATE, error="ไม่พบไฟล์รูปภาพ")
        file = request.files['file']
        if file.filename == '': return render_template_string(HTML_TEMPLATE, error="ไม่ได้เลือกไฟล์")

        try:
            img = Image.open(io.BytesIO(file.read()))
            prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
            
            response = model.generate_content([img, prompt])
            
            # รับค่า 2 ตัวแปรจากฟังก์ชัน
            row_data, recommendation = process_and_analyze(response.text.strip())
            
            # ส่งแค่ row_data (12 คอลัมน์) ลงชีท
            sheet.append_row(row_data)
            
            # โชว์คำแนะนำบนหน้าเว็บเหมือนเดิม
            display_text = f"ทีม: {row_data[0]} vs {row_data[1]}<br>วิเคราะห์: <span class='vip-badge'>{recommendation}</span>"
            return render_template_string(HTML_TEMPLATE, result=display_text)
        
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))

    return render_template_string(HTML_TEMPLATE)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        image_url = data.get('image_url')
        if not image_url: return jsonify({"status": "error", "message": "ไม่พบลิงก์"}), 400

        img = Image.open(io.BytesIO(requests.get(image_url).content))
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,1X2 เหย้า,1X2 เสมอ,1X2 เยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ (ระบุเฉพาะตัวเลข ห้ามมีตัวอักษร o หรือ u นำหน้า),ค่าน้ำสูง,ค่าน้ำต่ำ โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
        
        response = model.generate_content([img, prompt])
        row_data, recommendation = process_and_analyze(response.text.strip())
        
        sheet.append_row(row_data)
        return jsonify({"status": "success", "data": row_data, "recommendation": recommendation}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
