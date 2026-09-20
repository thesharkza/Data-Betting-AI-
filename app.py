from flask import Flask, request, jsonify, render_template_string
import gspread
import google.generativeai as genai
from PIL import Image
import requests
import io
import os

app = Flask(__name__)

# 1. ตั้งค่า Gemini API Key (ดึงจาก Environment Variable บน Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash-lite')

# 2. อ่านไฟล์ Credentials จาก Secret File ของ Render เพื่อเชื่อมต่อ Google Sheets
creds_path = '/etc/secrets/credentials.json'
if not os.path.exists(creds_path):
    creds_path = 'credentials.json' # สำรองไว้เผื่อรันทดสอบในเครื่องตัวเอง

client = gspread.service_account(filename=creds_path)
# เปิดไฟล์ Google Sheets (แก้ชื่อไฟล์ตรงนี้หากมีการเปลี่ยนชื่อ)
sheet = client.open("ข้อมูลราคาบอลสกัดจากภาพ").sheet1

# 3. โค้ดหน้าเว็บ HTML สำหรับให้ผู้ใช้อัปโหลดรูปภาพผ่านเบราว์เซอร์
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ระบบสกัดข้อมูลราคาบอลด้วย AI</title>
    <style>
        body { font-family: 'Tahoma', sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center; max-width: 500px; width: 90%; }
        h2 { color: #2c3e50; margin-bottom: 5px; }
        p { color: #7f8c8d; font-size: 14px; margin-bottom: 25px; }
        input[type="file"] { margin: 10px 0 20px 0; padding: 10px; border: 2px dashed #bdc3c7; border-radius: 8px; width: 90%; background: #fafafa; cursor: pointer; }
        button { background-color: #27ae60; color: white; border: none; padding: 12px 25px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; transition: background 0.3s; }
        button:hover { background-color: #219150; }
        .result { margin-top: 25px; padding: 15px; background: #e8f8f5; border: 1px solid #1abc9c; border-radius: 8px; color: #16a085; text-align: left; font-size: 14px; }
        .error { margin-top: 25px; padding: 15px; background: #fadbd8; border: 1px solid #e74c3c; border-radius: 8px; color: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <h2>⚽ อัปโหลดตารางราคาบอล</h2>
        <p>ให้ AI สกัดข้อมูล 12 คอลัมน์ และบันทึกลง Google Sheets อัตโนมัติ</p>
        
        <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">🚀 อัปโหลดและสกัดข้อมูลเลย</button>
        </form>

        {% if result %}
            <div class="result">
                <strong>✅ สำเร็จ! ข้อมูลที่บันทึกลงชีท:</strong><br><br>
                {{ result }}
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

# Route หลัก สำหรับเปิดหน้าเว็บและรับไฟล์รูปจากเครื่อง
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # ตรวจสอบว่ามีไฟล์แนบมาหรือไม่
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="ไม่พบไฟล์รูปภาพที่อัปโหลด")
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="คุณยังไม่ได้เลือกไฟล์รูปภาพ")

        try:
            # อ่านไฟล์รูปภาพที่ผู้ใช้อัปโหลดมา
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # คำสั่ง Prompt ให้ Gemini สกัดข้อมูล
            prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ,ค่าน้ำสูง,ค่าน้ำต่ำ,1X2 เหย้า,1X2 เสมอ,1X2 เยือน โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
            
            # ส่งให้ Gemini ประมวลผล
            response = model.generate_content([img, prompt])
            raw_text = response.text.strip()
            
            # ตัดแบ่งข้อมูลด้วยลูกน้ำ (,) เพื่อเตรียมลงตาราง
            row_data = [item.strip() for item in raw_text.split(',')]
            
            # บันทึกลง Google Sheets ทันที
            sheet.append_row(row_data)
            
            # ส่งผลลัพธ์กลับไปโชว์ที่หน้าเว็บ
            return render_template_string(HTML_TEMPLATE, result=" | ".join(row_data))
        
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))

    # ถ้าเข้าเว็บมาครั้งแรก (GET) ให้โชว์หน้าอัปโหลดปกติ
    return render_template_string(HTML_TEMPLATE)

# (เก็บเผื่อไว้) Route สำหรับรับ Webhook แบบเดิม เผื่ออนาคตใช้โปรแกรมอื่นยิงข้อมูลเข้ามา
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"status": "error", "message": "ไม่พบลิงก์รูปภาพ"}), 400

        img_bytes = requests.get(image_url).content
        img = Image.open(io.BytesIO(img_bytes))
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ,ค่าน้ำสูง,ค่าน้ำต่ำ,1X2 เหย้า,1X2 เสมอ,1X2 เยือน โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
        
        response = model.generate_content([img, prompt])
        raw_text = response.text.strip()
        row_data = [item.strip() for item in raw_text.split(',')]
        
        sheet.append_row(row_data)
        return jsonify({"status": "success", "data": row_data}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
