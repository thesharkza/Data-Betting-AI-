from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from PIL import Image
import requests
import io

app = Flask(__name__)

# ตั้งค่า API Key ของ Gemini
genai.configure(api_key="ใส่_GEMINI_API_KEY_ของคุณที่นี่")
model = genai.GenerativeModel('gemini-1.5-flash')

# ตั้งค่าการเชื่อมต่อ Google Sheets โดยอ้างอิงจากไฟล์ credentials.json
scope = ["https://www.spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

# เปิดไฟล์ Google Sheets (เปลี่ยนชื่อไฟล์ให้ตรงกับของคุณ)
sheet = client.open("ชื่อไฟล์_Google_Sheets_ของคุณ").sheet1

@app.route('/webhook', methods=['POST'])
def receive_image():
    try:
        # รับข้อมูล JSON ที่ส่งเข้ามา (ต้องมี key ชื่อ image_url)
        data = request.json
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"status": "error", "message": "ไม่พบลิงก์รูปภาพ"}), 400

        # ดาวน์โหลดรูปภาพจากลิงก์
        img_bytes = requests.get(image_url).content
        img = Image.open(io.BytesIO(img_bytes))
        
        # คำสั่ง Prompt ให้ Gemini สกัดข้อมูล
        prompt = "สกัดข้อมูลจากภาพนี้เรียงตามลำดับ: ชื่อทีมเหย้า,ชื่อทีมเยือน,แฮนดิแคปเหย้า,ค่าน้ำHDPเหย้า,แฮนดิแคปเยือน,ค่าน้ำHDPเยือน,โกลสูงต่ำ,ค่าน้ำสูง,ค่าน้ำต่ำ,1X2 เหย้า,1X2 เสมอ,1X2 เยือน โดยคั่นแต่ละค่าด้วยลูกน้ำ (,) เท่านั้น ห้ามมีข้อความอื่น"
        
        # ให้ Gemini ประมวลผลภาพ
        response = model.generate_content([img, prompt])
        raw_text = response.text.strip()
        
        # ตัดแบ่งข้อมูลด้วยลูกน้ำ (,)
        row_data = [item.strip() for item in raw_text.split(',')]
        
        # บันทึกลง Google Sheets ทันที
        sheet.append_row(row_data)
        
        return jsonify({"status": "success", "data": row_data}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
